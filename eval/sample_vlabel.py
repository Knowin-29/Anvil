#!/usr/bin/env python3
"""Stratified sampling of correct-balanced test set for checkpoint evaluation.

Produces:
  - eval/vlabel_input.jsonl : prompt-only ms-swift records (with <image> + remapped abs paths)
  - eval/vlabel_gt.jsonl    : ground-truth (model_answer + task_family + capabilities + bbox)
"""
import json, os, re, random

random.seed(42)
TEST = "${DATA_ROOT}/datasets/correct-balanced-train-test-v1/test.jsonl"
OUT_DIR = "${GSA_FORGE_ROOT}/eval"
os.makedirs(OUT_DIR, exist_ok=True)
INPUT = os.path.join(OUT_DIR, "vlabel_input.jsonl")
GT = os.path.join(OUT_DIR, "vlabel_gt.jsonl")

OLD = "${DATA_ROOT}/qwen-proxy-log-mirror"
NEW = "${DATA_ROOT}/datasets/qwen-proxy-log-mirror-fast"

PER_FAMILY = 200  # samples per task_family

def remap(p):
    return p.replace(OLD, NEW)

def parse_bbox(text):
    text = str(text)
    for key in ['bbox_2d', 'bbox']:
        m = re.search(key + r'"?\s*:?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
        if m:
            return [float(m.group(i)) for i in range(1, 5)]
    m = re.search(r'<box>\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]</box>', text)
    if m:
        return [float(m.group(i)) for i in range(1, 5)]
    return None

# Group by task_family
from collections import defaultdict
buckets = defaultdict(list)
with open(TEST) as f:
    for line in f:
        d = json.loads(line)
        qc = (d.get('tags') or {}).get('qc') or {}
        tf = qc.get('task_family', 'untagged')
        buckets[tf].append(d)

print(f"task_family buckets: {[(k, len(v)) for k, v in buckets.items()]}")

n_in = 0
with open(INPUT, 'w') as fi, open(GT, 'w') as fg:
    for tf, items in buckets.items():
        sample = random.sample(items, min(PER_FAMILY, len(items)))
        for d in sample:
            imgs = [remap(p) for p in (d.get('images') or [])]
            if not imgs or not os.path.isfile(imgs[0]):
                continue
            # Build prompt-only messages, ensure <image> present
            msgs = []
            for m in d.get('messages', []):
                if m.get('role') == 'assistant':
                    continue
                msgs.append({"role": m['role'], "content": m['content']})
            if not msgs:
                continue
            # Ensure first user msg has <image>
            for m in msgs:
                if m['role'] == 'user' and '<image>' not in str(m['content']):
                    m['content'] = '<image>\n' + str(m['content'])
                    break
            rec_id = d.get('id', f"{tf}_{n_in}")
            fi.write(json.dumps({"messages": msgs, "images": imgs}, ensure_ascii=False) + '\n')
            qc = (d.get('tags') or {}).get('qc') or {}
            fg.write(json.dumps({
                "id": rec_id,
                "task_family": tf,
                "capabilities": qc.get('capabilities', []),
                "fmt": qc.get('fmt'),
                "model_answer": d.get('model_answer', ''),
                "gt_bbox": parse_bbox(d.get('model_answer', '')),
                "model": d.get('model', ''),
            }, ensure_ascii=False) + '\n')
            n_in += 1

print(f"Sampled {n_in} records -> {INPUT}")
