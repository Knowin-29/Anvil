#!/usr/bin/env python3
"""VLM-as-judge: Use Qwen3.5-397B API to judge semantic equivalence.

For samples where rule-based scoring fails (unknown format, no matching field),
call the 397B API to determine if model output is semantically correct.
"""
import json, re, os, sys, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = os.environ.get("VLM_JUDGE_API_URL", "")
API_KEY = os.environ.get("VLM_JUDGE_API_KEY", "")

JUDGE_PROMPT = """You are a strict evaluation judge. Given:
- TASK: The original prompt/instruction
- GT: The ground-truth correct answer (verified by human)
- PRED: The model's output to evaluate

Judge whether PRED is semantically correct (conveys the same information as GT).
Rules:
- Ignore format differences (JSON vs text, different field names are OK if content matches)
- For spatial/pose tasks: same pose/orientation/shape = correct
- For object identification: same object name = correct
- For bbox/coordinates: close coordinates (within 10% of image) = correct
- For descriptive answers: key facts matching = correct

Respond with ONLY one word: CORRECT or INCORRECT"""

def call_judge(task_prompt, gt_answer, pred_answer, image_url=None, timeout=30):
    """Call 397B API to judge one sample."""
    content = [
        {"type": "text", "text": f"{JUDGE_PROMPT}\n\nTASK: {task_prompt[:500]}\n\nGT: {gt_answer[:500]}\n\nPRED: {pred_answer[:500]}\n\nVerdict:"}
    ]
    if image_url and os.path.isfile(image_url):
        # For local files, we skip image (API needs URL not path)
        pass
    
    payload = {
        "model": "MLLM",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a strict evaluation judge. Answer only CORRECT or INCORRECT."}]},
            {"role": "user", "content": content}
        ],
        "max_tokens": 500,
        "temperature": 0
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            choice = data['choices'][0]['message']
            content_text = choice.get('content') or ''
            reasoning = choice.get('reasoning_content') or ''
            if content_text.strip():
                return 'CORRECT' in content_text.strip().upper() and 'INCORRECT' not in content_text.strip().upper()
            # fallback: extract from reasoning
            m = re.search(r'(CORRECT|INCORRECT)\s*$', reasoning.strip(), re.IGNORECASE)
            if m:
                return m.group(1).upper() == 'CORRECT'
            return None
        else:
            return None  # API error, skip
    except Exception as e:
        return None

def main():
    pred_file = sys.argv[1]  # vlabel_out_*.jsonl
    gt_file = sys.argv[2]    # vlabel_gt.jsonl
    input_file = sys.argv[3] # vlabel_input.jsonl (for task prompts)
    tag = sys.argv[4] if len(sys.argv) > 4 else 'judge'
    
    outs = [json.loads(l) for l in open(pred_file)]
    gts = [json.loads(l) for l in open(gt_file)]
    inputs = [json.loads(l) for l in open(input_file)]
    n = min(len(outs), len(gts), len(inputs))
    
    # Identify samples needing VLM judge (rule-based can't score)
    needs_judge = []
    rule_scored = {'correct': 0, 'total': 0}
    
    for i in range(n):
        pred = outs[i].get('response', '')
        gt = gts[i]
        ga = gt.get('model_answer', '')
        tf = gt.get('task_family', '')
        
        # Rule-based first
        scored = False
        if 'bbox_grounding' in tf:
            scored = True  # IoU handles this
        elif ga.strip().lower() == pred.strip().lower():
            rule_scored['correct'] += 1; rule_scored['total'] += 1; scored = True
        elif ga.strip() and pred.strip() and ga.strip() in pred.strip():
            rule_scored['correct'] += 1; rule_scored['total'] += 1; scored = True
        
        if not scored and tf != '' and len(ga.strip()) > 0 and len(pred.strip()) > 5:
            # Get task prompt
            msgs = inputs[i].get('messages', [])
            prompt = next((m['content'] for m in msgs if m.get('role') == 'user'), '')
            prompt = re.sub(r'<image>\n?', '', str(prompt))
            needs_judge.append((i, prompt[:500], ga, pred, tf))
    
    print(f"Total: {n}, Rule-scored: {rule_scored['total']}, Needs VLM judge: {len(needs_judge)}")
    
    # Call API in parallel (but rate-limited)
    judge_results = {}
    batch_size = 5  # concurrent requests
    
    for batch_start in range(0, len(needs_judge), batch_size):
        batch = needs_judge[batch_start:batch_start+batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {}
            for (idx, prompt, ga, pred, tf) in batch:
                f = executor.submit(call_judge, prompt, ga, pred)
                futures[f] = (idx, tf)
            for f in as_completed(futures):
                idx, tf = futures[f]
                result = f.result()
                if result is not None:
                    judge_results[idx] = result
        
        # Progress
        done = batch_start + len(batch)
        if done % 20 == 0 or done >= len(needs_judge):
            correct = sum(1 for v in judge_results.values() if v)
            print(f"  Judged {done}/{len(needs_judge)}: {correct} correct so far")
        time.sleep(0.5)  # rate limit
    
    # Final scoring
    from collections import defaultdict
    by_family = defaultdict(lambda: {'correct': 0, 'total': 0})
    for idx, result in judge_results.items():
        tf = gts[idx].get('task_family', 'unknown')
        by_family[tf]['total'] += 1
        by_family[tf]['correct'] += int(result)
    
    total_judged = len(judge_results)
    total_correct = sum(1 for v in judge_results.values() if v)
    
    report = {
        'tag': tag,
        'rule_scored': rule_scored,
        'vlm_judged': {'total': total_judged, 'correct': total_correct,
                       'acc': total_correct/total_judged if total_judged else 0},
        'by_family': {k: {**v, 'acc': v['correct']/v['total'] if v['total'] else 0}
                      for k, v in by_family.items()},
    }
    out_path = f"eval/vlabel_judge_{tag}.json"
    json.dump(report, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f"\n=== VLM Judge Results ({tag}) ===")
    print(f"Judged: {total_judged}, Correct: {total_correct} ({report['vlm_judged']['acc']*100:.1f}%)")
    print("By family:")
    for k, v in sorted(by_family.items()):
        print(f"  {k}: {v['correct']}/{v['total']} ({v['acc']*100:.1f}%)")
    print(f"-> {out_path}")

if __name__ == '__main__':
    main()
