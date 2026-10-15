#!/usr/bin/env python3
"""Build v0616 ms-swift data using answer-type buckets.

This builder is deliberately more conservative than the legacy ratio-only
builder.  It first routes records into answer buckets, then samples each bucket
against the v0616 target mix and source-cap policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
sys.path.insert(0, str(EVAL_DIR))

from vlabel_ir_v0616 import build_typed_ir, remap_image_path, stable_hash  # noqa: E402


SYSTEM_PROMPT = (
    "You are an embodied perception assistant specialized in indoor scene understanding. "
    "You can detect objects, understand spatial relationships, identify object attributes and states, "
    "and reason about robot manipulation affordances. Respond precisely with structured output."
)

BUCKET_FOR_ANSWER_TYPE = {
    "bbox_region": "bbox_region",
    "point_affordance": "point_affordance",
    "choice": "attribute_state",
    "attribute_state": "attribute_state",
    "count": "general_vqa",
    "relation": "spatial_relation",
    "action_free_text": "action_planning",
    "unknown": "general_vqa",
}


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def assistant_text(record: Dict[str, Any]) -> str:
    for m in record.get("messages") or []:
        if m.get("role") == "assistant":
            return str(m.get("content", ""))
    return str(record.get("model_answer", ""))


def prompt_text(record: Dict[str, Any]) -> str:
    for m in record.get("messages") or []:
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return str(record.get("prompt", ""))


def validate_msswift(record: Dict[str, Any]) -> bool:
    msgs = record.get("messages")
    images = record.get("images")
    if not isinstance(msgs, list) or not isinstance(images, list) or not images:
        return False
    roles = {m.get("role") for m in msgs}
    return "user" in roles and "assistant" in roles and bool(assistant_text(record).strip())


def make_prompt_only(record: Dict[str, Any], bucket: str) -> Dict[str, Any]:
    msgs = []
    for m in record.get("messages") or []:
        if m.get("role") == "assistant":
            break
        msgs.append(m)
    if not msgs:
        msgs = [{"role": "user", "content": "<image>\nDescribe this image."}]
    return {"messages": msgs, "images": record.get("images", []), "dimension": bucket}


def source_record_key(record: Dict[str, Any]) -> str:
    img = (record.get("images") or [""])[0]
    key = {"image": img, "prompt": prompt_text(record), "answer": assistant_text(record)}
    return hashlib.sha1(json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def route_bucket(record: Dict[str, Any], source_name: str, source_dims: List[str]) -> str:
    text = assistant_text(record)
    prompt = prompt_text(record)
    task_family = record.get("task_family") or record.get("answer_type") or ""
    ir = build_typed_ir(prompt, text, task_family)
    bucket = BUCKET_FOR_ANSWER_TYPE.get(ir["answer_type"], "general_vqa")
    if bucket == "general_vqa":
        dims = set(source_dims or [])
        if "S_spatial" in dims:
            bucket = "spatial_relation"
        elif "A_action_precondition" in dims:
            bucket = "action_planning"
        elif "A_obj_state" in dims or "A_identity_attr" in dims:
            bucket = "attribute_state"
        elif "A_action_interface" in dims:
            bucket = "point_affordance"
        elif "G_grounding" in dims and ("refcoco" in source_name or "bbox" in source_name):
            bucket = "bbox_region"
    return bucket


def convert_correct_balanced_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    images = [remap_image_path(p) for p in (row.get("images") or [])]
    if not images:
        return None
    msgs = []
    for m in row.get("messages") or []:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "user" and "<image>" not in content:
            content = "<image>\n" + content
        if role in {"system", "user", "assistant"}:
            msgs.append({"role": role, "content": content})
    if not any(m.get("role") == "assistant" for m in msgs):
        msgs.append({"role": "assistant", "content": str(row.get("model_answer", ""))})
    if not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    qc = (row.get("tags") or {}).get("qc") or {}
    return {
        "messages": msgs,
        "images": images,
        "source_req_id": row.get("req_id") or row.get("id"),
        "task_family": qc.get("task_family", "untagged"),
        "capabilities": qc.get("capabilities") or [],
    }


def reservoir_sample(path: Path, k: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    n = 0
    for row in iter_jsonl(str(path)):
        n += 1
        if len(sample) < k:
            sample.append(row)
        else:
            j = rng.randint(1, n)
            if j <= k:
                sample[j - 1] = row
    rng.shuffle(sample)
    return sample


def quota_for_bucket(counts: Dict[str, int], target: int, cap_frac: float, cap_overrides: Dict[str, Any], bucket: str) -> Dict[str, int]:
    if target <= 0:
        return {}
    quota = {}
    remaining = {}
    for source, count in counts.items():
        cap = cap_overrides.get(source, {}).get(bucket, cap_frac)
        q = min(count, int(target * cap))
        quota[source] = q
        remaining[source] = count - q
    assigned = sum(quota.values())
    if assigned < target:
        for source, rem in sorted(remaining.items(), key=lambda kv: kv[1], reverse=True):
            if assigned >= target:
                break
            add = min(rem, target - assigned)
            quota[source] += add
            assigned += add
    elif assigned > target:
        scale = target / assigned
        for source in list(quota):
            quota[source] = int(quota[source] * scale)
        assigned = sum(quota.values())
        for source in sorted(quota, key=lambda s: counts[s], reverse=True):
            if assigned >= target:
                break
            if quota[source] < counts[source]:
                quota[source] += 1
                assigned += 1
    return {s: q for s, q in quota.items() if q > 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "data" / "sources_registry_v0616.yaml"))
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_records_per_source", type=int, default=0, help="Smoke-test cap before bucketing.")
    ap.add_argument("--dry_run", action="store_true", help="Build candidate counts/report only; do not write final train files.")
    args = ap.parse_args()

    cfg = read_yaml(args.registry)
    base = read_yaml(cfg["base_registry"])
    output_dir = Path(args.output_dir or cfg["output"]["train_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["targets"].get("seed", 42))
    rng = random.Random(seed)
    ratio_overrides = cfg.get("ratio_overrides", {})
    disabled = set(cfg.get("disabled_sources", []))
    bucket_mix = cfg["targets"]["bucket_mix"]
    target_total = int(cfg["targets"]["total_sft"])
    cap_frac = float(cfg["targets"].get("default_source_cap_per_bucket", 0.35))
    cap_overrides = cfg["targets"].get("source_cap_overrides", {})

    test_req_ids = set()
    special = cfg.get("special_sources", {}).get("correct_balanced_train", {})
    if special.get("enabled") and special.get("test_path"):
        for row in iter_jsonl(special["test_path"]):
            rid = row.get("req_id") or row.get("id")
            if rid:
                test_req_ids.add(rid)

    counts = defaultdict(Counter)
    source_stats: Dict[str, Dict[str, Any]] = {}
    candidate_stack = ExitStack()
    candidate_handles: Dict[Tuple[str, str], TextIO] = {}

    def write_candidate(bucket: str, source: str, rec: Dict[str, Any]) -> None:
        path = candidate_dir / f"{bucket}__{source}.jsonl"
        key = (bucket, source)
        handle = candidate_handles.get(key)
        if handle is None:
            handle = candidate_stack.enter_context(open(path, "a", encoding="utf-8"))
            candidate_handles[key] = handle
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[bucket][source] += 1

    # Clear old candidate shards for deterministic reruns in this output dir.
    for old in candidate_dir.glob("*.jsonl"):
        old.unlink()

    sources = list(base.get("sources", []))
    for src in sources:
        name = src.get("name")
        if name in disabled:
            continue
        ratio = float(ratio_overrides.get(name, src.get("ratio", 0)))
        if ratio <= 0 or src.get("format") != "msswift_ready":
            continue
        path = src.get("path")
        if not path or not os.path.isfile(path):
            source_stats[name] = {"status": "missing", "path": path}
            continue
        seen = set()
        loaded = clean = dedup = 0
        for rec in iter_jsonl(path):
            loaded += 1
            if args.max_records_per_source and loaded > args.max_records_per_source:
                break
            if not validate_msswift(rec):
                continue
            key = source_record_key(rec) if src.get("dedup_key") == "image_response" or name == "desktop_bbox_train" else None
            if key and key in seen:
                dedup += 1
                continue
            if key:
                seen.add(key)
            if ratio < 1.0 and rng.random() > ratio:
                continue
            copies = int(ratio) if ratio >= 1.0 else 1
            extra = ratio - int(ratio) if ratio >= 1.0 else 0.0
            bucket = route_bucket(rec, name, src.get("dimensions") or [])
            out_rec = dict(rec)
            out_rec["v0616_source"] = name
            out_rec["v0616_bucket"] = bucket
            for _ in range(copies):
                write_candidate(bucket, name, out_rec)
                clean += 1
            if extra > 0 and rng.random() < extra:
                write_candidate(bucket, name, out_rec)
                clean += 1
        source_stats[name] = {"loaded": loaded, "candidate_records": clean, "deduped": dedup, "ratio": ratio}
        print(f"[v0616-build] source={name} loaded={loaded} candidates={clean} deduped={dedup}", flush=True)

    if special.get("enabled"):
        name = "correct_balanced_train"
        loaded = excluded = clean = 0
        for row in iter_jsonl(special["path"]):
            loaded += 1
            if args.max_records_per_source and loaded > args.max_records_per_source:
                break
            rid = row.get("req_id") or row.get("id")
            if rid in test_req_ids:
                excluded += 1
                continue
            rec = convert_correct_balanced_row(row)
            if not rec or not validate_msswift(rec):
                continue
            bucket = route_bucket(rec, name, special.get("dimensions") or [])
            rec["v0616_source"] = name
            rec["v0616_bucket"] = bucket
            write_candidate(bucket, name, rec)
            clean += 1
        source_stats[name] = {
            "loaded": loaded,
            "candidate_records": clean,
            "excluded_test_req_ids": excluded,
            "test_req_ids": len(test_req_ids),
        }
        print(
            f"[v0616-build] source={name} loaded={loaded} candidates={clean} excluded_test_req_ids={excluded}",
            flush=True,
        )

    # Close candidate shards before reservoir sampling reads them back.
    for handle in candidate_handles.values():
        handle.flush()
    candidate_stack.close()
    candidate_handles.clear()

    bucket_targets = {b: int(target_total * frac) for b, frac in bucket_mix.items()}
    bucket_quotas = {
        b: quota_for_bucket(dict(counts[b]), bucket_targets[b], cap_frac, cap_overrides, b)
        for b in bucket_targets
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "registry": args.registry,
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "target_total": target_total,
        "bucket_targets": bucket_targets,
        "candidate_counts": {b: dict(c) for b, c in counts.items()},
        "bucket_quotas": bucket_quotas,
        "source_stats": source_stats,
        "finevision": cfg.get("finevision", {}),
    }

    if not args.dry_run:
        sft_path = output_dir / "gsa_sft_train_v0616.jsonl"
        grpo_path = output_dir / "gsa_grpo_train_v0616.jsonl"
        opd_path = output_dir / "gsa_opd_train_v0616.jsonl"
        final_counts = Counter()
        with open(sft_path, "w", encoding="utf-8") as fs, \
                open(grpo_path, "w", encoding="utf-8") as fg, \
                open(opd_path, "w", encoding="utf-8") as fo:
            for bucket, quotas in bucket_quotas.items():
                for source, q in quotas.items():
                    shard = candidate_dir / f"{bucket}__{source}.jsonl"
                    selected = reservoir_sample(shard, q, seed + int(hashlib.sha1(str(shard).encode()).hexdigest()[:8], 16))
                    for rec in selected:
                        fs.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        final_counts[bucket] += 1
                        if rng.random() < float(cfg["targets"].get("grpo_ratio", 0.3)):
                            fg.write(json.dumps(make_prompt_only(rec, bucket), ensure_ascii=False) + "\n")
                        if rng.random() < float(cfg["targets"].get("opd_ratio", 0.2)):
                            fo.write(json.dumps(make_prompt_only(rec, bucket), ensure_ascii=False) + "\n")
        report["outputs"] = {"sft": str(sft_path), "grpo": str(grpo_path), "opd": str(opd_path)}
        report["final_bucket_counts"] = dict(final_counts)
        report["final_total"] = sum(final_counts.values())

    report["report_hash"] = stable_hash({
        "bucket_targets": report["bucket_targets"],
        "candidate_counts": report["candidate_counts"],
        "bucket_quotas": report["bucket_quotas"],
    })
    report_path = output_dir / "data_report_v0616.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:5000])
    print(f"-> {report_path}")


if __name__ == "__main__":
    main()
