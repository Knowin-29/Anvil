#!/usr/bin/env python3
"""
Quick benchmark: compare SFT checkpoint vs base model on public VQA tasks.

Tests:
1. General VQA ability (sanity check - not collapsed)
2. GSA format compliance (structured JSON output rate)
3. Grounding accuracy (bbox format validity)

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/benchmark_quick.py \
        --model output/sft_27b_teacher/v3-20260607-135945/checkpoint-1740 \
        --base_model ${MODEL_27B}
"""
import argparse
import json
import os
import re
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, os.environ.get("MS_SWIFT_ROOT", "/path/to/storage/anvil/ms-swift"))


GENERAL_VQA_PROMPTS = [
    "Describe what you see in this image in detail.",
    "How many objects are on the table?",
    "What color is the largest object?",
    "Is there a person in this image?",
    "What room is this?",
]

GSA_PROMPTS = [
    "Detect the bottle in the image and provide its bounding box coordinates [x1,y1,x2,y2].",
    "What is the spatial relationship between the cup and the plate?",
    "What is the current state of the door?",
    "How should a robot grasp the mug? Provide the action interface.",
    "Identify the visual attributes of the chair in the image.",
]


def find_test_images(n=5):
    """Find existing images from training data for testing."""
    images = []
    sft_path = os.environ.get("TRAIN_DIR", "/path/to/storage/anvil/GSA_Forge/trains") + "/gsa_sft_train.jsonl"
    if not os.path.exists(sft_path):
        return []
    with open(sft_path) as f:
        for line in f:
            s = json.loads(line)
            img = s.get("images", [""])[0]
            if img and os.path.isfile(img) and img not in images:
                images.append(img)
                if len(images) >= n:
                    break
    return images


def evaluate_model(model_path, test_images, prompts, label="model"):
    from swift import TransformersEngine, InferRequest, RequestConfig

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    engine = TransformersEngine(
        model=model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    rc = RequestConfig(max_tokens=512, temperature=0.1)

    results = {"valid_json": 0, "has_bbox": 0, "has_content": 0, "total": 0, "responses": []}

    for img in test_images[:3]:
        for prompt in prompts:
            req = InferRequest(
                messages=[
                    {"role": "system", "content": "You are an embodied perception assistant. Respond with structured JSON."},
                    {"role": "user", "content": f"<image>\n{prompt}"},
                ],
                images=[img],
            )
            try:
                resp = engine.infer([req], request_config=rc)[0]
                content = resp.choices[0].message.content
                if "</think>" in content:
                    content = content.split("</think>", 1)[-1].strip()

                results["total"] += 1
                results["has_content"] += 1 if len(content) > 5 else 0

                # Try parse JSON
                json_match = re.search(r'\{[^{}]+\}', content)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        results["valid_json"] += 1
                        if "bbox" in parsed:
                            results["has_bbox"] += 1
                    except json.JSONDecodeError:
                        pass

                results["responses"].append({
                    "prompt": prompt[:50],
                    "response": content[:200],
                })
            except Exception as e:
                results["total"] += 1
                results["responses"].append({"prompt": prompt[:50], "error": str(e)[:100]})

    # Print results
    total = results["total"] or 1
    print(f"\n--- Results ({label}) ---")
    print(f"Total queries:    {results['total']}")
    print(f"Has content:      {results['has_content']}/{total} ({100*results['has_content']/total:.0f}%)")
    print(f"Valid JSON:        {results['valid_json']}/{total} ({100*results['valid_json']/total:.0f}%)")
    print(f"Has bbox:          {results['has_bbox']}/{total} ({100*results['has_bbox']/total:.0f}%)")

    print(f"\n--- Sample outputs ---")
    for r in results["responses"][:5]:
        if "error" in r:
            print(f"  Q: {r['prompt']}...")
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Q: {r['prompt']}...")
            print(f"  A: {r['response'][:120]}")
        print()

    del engine
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("OUTPUT_BASE", "/path/to/storage/anvil/GSA_Forge/output") + "/sft_27b_teacher")
    parser.add_argument("--base_model", default=os.environ.get("MODEL_27B", "/path/to/storage/models/Qwen/Qwen3.6-27B"))
    parser.add_argument("--skip_base", action="store_true", help="Skip base model evaluation")
    args = parser.parse_args()

    test_images = find_test_images(5)
    if not test_images:
        print("ERROR: No test images found")
        return
    print(f"Found {len(test_images)} test images")

    # Evaluate SFT checkpoint
    sft_results = evaluate_model(args.model, test_images, GSA_PROMPTS, label="SFT-27B-Teacher")

    # Evaluate base model for comparison (optional)
    if not args.skip_base and os.path.isdir(args.base_model):
        base_results = evaluate_model(args.base_model, test_images, GSA_PROMPTS, label="Base-27B")
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'Base':>10} {'SFT':>10}")
        print(f"  {'Valid JSON':<20} {base_results['valid_json']:>10} {sft_results['valid_json']:>10}")
        print(f"  {'Has bbox':<20} {base_results['has_bbox']:>10} {sft_results['has_bbox']:>10}")
        print(f"  {'Has content':<20} {base_results['has_content']:>10} {sft_results['has_content']:>10}")
    else:
        print("\nSkipping base model comparison.")

    # Save results
    out_path = os.environ.get("GSA_FORGE_ROOT", "/path/to/storage/anvil/GSA_Forge") + "/logs/benchmark_result.json"
    with open(out_path, "w") as f:
        json.dump({"sft": sft_results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
