"""
Build DPO preference pairs and GRPO training data from flywheel verified/rejected data.

Usage:
    python build_preference_pairs.py
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def load_scored_data(data_dir):
    samples = []
    if not os.path.isdir(data_dir):
        return samples
    for dim_dir in sorted(Path(data_dir).iterdir()):
        if not dim_dir.is_dir():
            continue
        for jf in sorted(dim_dir.glob("*.jsonl")):
            with open(jf) as f:
                for line in f:
                    try:
                        samples.append(json.loads(line.strip()))
                    except:
                        continue
    return samples


def extract_score(sample):
    v = sample.get("_verification", {})
    return v.get("score", 0.0) if isinstance(v, dict) else 0.0


def main():
    base = "/path/to/Anvil/data/output/gsa_flywheel_exp"
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified_dir", default=f"{base}/verified")
    parser.add_argument("--rejected_dir", default=f"{base}/rejected")
    parser.add_argument("--needs_review_dir", default=f"{base}/needs_review")
    parser.add_argument("--output_dir", default="/path/to/datasets/dataflywheel/output/preference_pairs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    verified = load_scored_data(args.verified_dir)
    rejected = load_scored_data(args.rejected_dir)
    needs_review = load_scored_data(args.needs_review_dir)
    print(f"Verified: {len(verified)}, Rejected: {len(rejected)}, Review: {len(needs_review)}")

    # DPO pairs: same (image, dim), high vs low score
    high_score = defaultdict(list)
    low_score = defaultdict(list)
    for s in verified:
        if extract_score(s) >= 0.85:
            high_score[(s.get("image_path"), s.get("dimension"))].append(s)
    for s in rejected + [r for r in needs_review if extract_score(r) < 0.6]:
        low_score[(s.get("image_path"), s.get("dimension"))].append(s)

    pairs = []
    for key in high_score:
        if key in low_score:
            for c in high_score[key][:3]:
                for r in low_score[key][:3]:
                    pairs.append({"chosen": c, "rejected": r, "dimension": key[1]})

    with open(os.path.join(args.output_dir, "dpo_pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"DPO pairs: {len(pairs)}")

    # GRPO data with scores
    all_data = verified + rejected + needs_review
    grpo = [{"sample": s, "reward": extract_score(s)} for s in all_data if extract_score(s) > 0]
    with open(os.path.join(args.output_dir, "grpo_data.jsonl"), "w") as f:
        for g in grpo:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"GRPO samples: {len(grpo)}")


if __name__ == "__main__":
    main()
