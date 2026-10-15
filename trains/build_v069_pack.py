#!/usr/bin/env python3
"""Build v069 GSA training packs.

Outputs:
  trains/gsa_volc_sft_v069.jsonl
  trains/gsa_volc_val_v069.jsonl
  trains/gsa_volc_grpo_v069.jsonl
  trains/gsa_train_v069.jsonl
  trains/gsa_val_v069.jsonl
  trains/gsa_grpo_train_v069.jsonl
  trains/gsa_opd_train_v069.jsonl
  trains/gsa_v069_report.json
  trains/diag/out/gsa_v069_data_inspection.html

The v069 policy is intentionally conservative:
  - train only image-readable records;
  - drop known dirty GSA_volc source=other records;
  - normalize pixel bboxes/points to 0-999 when image_size is present;
  - strip source/provenance metadata from assistant JSON;
  - replace old GSA_volc rows in the merged train set with the cleaned v069 subset;
  - add high-value 20260618 repair sources, desktop prompt variants, and PixMo point replay.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional but available in this env.
    Image = None


ROOT = Path("${GSA_FORGE_ROOT}")
TRAINS = ROOT / "trains"
VOLC_ROOT = Path("${HF_DOWNLOAD_ROOT}/datasets/GSA_volc")

SYSTEM_PROMPT = (
    "You are an embodied perception assistant specialized in indoor scene understanding. "
    "You can detect objects, understand spatial relationships, identify object attributes and states, "
    "and reason about robot manipulation affordances. Respond precisely with structured output."
)

META_DROP_KEYS = {
    "source",
    "dataset",
    "image_size",
    "bucket",
    "dimension",
    "object_id",
    "sample_id",
    "timestamp",
    "scene_type",
    "pipeline",
    "s_context",
    "_verification",
    "_scan_count",
}

BBOX_KEYS = {"bbox", "subject_bbox", "object_bbox", "evidence_region"}
POINT_KEYS = {"point", "grasp_point_2d", "affordance_point"}
POINT_LIST_KEYS = {"points", "jaw_axis_2d", "fallback_points"}
COORD_KEYS = BBOX_KEYS | POINT_KEYS | POINT_LIST_KEYS
NULLISH_STRINGS = {"", "none", "null", "n/a", "na", "unknown"}
TARGET_KEYS = {"object", "object_class", "target", "target_object", "subject", "predicate"}
MEANINGFUL_EMPTY_LIST_KEYS = {"blocked_by", "candidates"}
SEMANTIC_NEGATIVE_PATTERNS = [
    re.compile(r"^there are none\.?$", re.I),
    re.compile(r"this isn'?t in the image", re.I),
    re.compile(r"\bnot in the image\b", re.I),
    re.compile(r"\bno (?:such|visible|matching)\b", re.I),
]


def stable_int(text: str, mod: int = 10_000_000) -> int:
    return int(hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12], 16) % mod


def record_key(rec: Dict[str, Any]) -> bytes:
    img = (rec.get("images") or [""])[0]
    msgs = rec.get("messages") or []
    user = next((str(m.get("content", "")) for m in msgs if m.get("role") == "user"), "")
    asst = next((str(m.get("content", "")) for m in msgs if m.get("role") == "assistant"), "")
    return hashlib.sha1((img + "\0" + user + "\0" + asst).encode("utf-8", errors="ignore")).digest()[:16]


def prompt_key(rec: Dict[str, Any]) -> str:
    img = (rec.get("images") or [""])[0]
    msgs = rec.get("messages") or []
    user = next((str(m.get("content", "")) for m in msgs if m.get("role") == "user"), "")
    return img + "\0" + user


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def assistant_text(rec: Dict[str, Any]) -> str:
    return next((str(m.get("content", "")) for m in rec.get("messages") or [] if m.get("role") == "assistant"), "")


def user_text(rec: Dict[str, Any]) -> str:
    return next((str(m.get("content", "")) for m in rec.get("messages") or [] if m.get("role") == "user"), "")


def parse_jsonish(text: str) -> Optional[Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
        s = re.sub(r"```$", "", s).strip()
    if not s.startswith(("{", "[")):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def normalized_answer_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def semantic_negative_kind(text: str) -> Optional[str]:
    norm = normalized_answer_text(text)
    for pattern in SEMANTIC_NEGATIVE_PATTERNS:
        if pattern.search(norm):
            return pattern.pattern
    return None


def is_semantic_negative(rec: Dict[str, Any]) -> bool:
    return semantic_negative_kind(assistant_text(rec)) is not None


def infer_dim(obj: Any, rec: Dict[str, Any]) -> str:
    if rec.get("dimension"):
        return str(rec["dimension"])
    if isinstance(obj, dict):
        if obj.get("dimension"):
            return str(obj["dimension"])
        if {"subject", "predicate", "object"}.issubset(obj):
            return "S_spatial"
        if "attributes" in obj:
            return "A_identity_attr"
        if "state_type" in obj or "state_value" in obj or "pose_family" in obj:
            return "A_obj_state"
        if "can_do_now" in obj or "blocked_by" in obj:
            return "A_action_precondition"
        if any(k in obj for k in ("grasp_point_2d", "jaw_axis_2d", "affordance_point")):
            return "A_action_interface"
        if "bbox" in obj or "point" in obj:
            return "G_grounding"
    return "unknown"


def clamp999(x: float) -> int:
    return max(0, min(999, int(round(x))))


def norm_xy(pt: List[Any], size: Optional[List[Any]]) -> List[int]:
    if not (isinstance(pt, list) and len(pt) == 2):
        return pt
    try:
        x, y = float(pt[0]), float(pt[1])
    except Exception:
        return pt
    if size and len(size) == 2:
        try:
            w, h = float(size[0]), float(size[1])
        except Exception:
            w = h = 0.0
        if w > 0 and h > 0 and (w != 1000 or h != 1000):
            return [clamp999(x / w * 999), clamp999(y / h * 999)]
    return [clamp999(x), clamp999(y)]


def norm_bbox(box: List[Any], size: Optional[List[Any]]) -> List[int]:
    if not (isinstance(box, list) and len(box) == 4):
        return box
    p1 = norm_xy([box[0], box[1]], size)
    p2 = norm_xy([box[2], box[3]], size)
    if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) == 2 and len(p2) == 2):
        return box
    x1, y1 = p1
    x2, y2 = p2
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def bbox_area_ratio(box: Any) -> Optional[float]:
    if not (isinstance(box, list) and len(box) == 4):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return -1.0
    return (x2 - x1) * (y2 - y1) / (999.0 * 999.0)


def has_numeric_coord(value: Any) -> bool:
    if isinstance(value, list):
        if value and all(isinstance(x, (int, float)) for x in value):
            return True
        return any(has_numeric_coord(x) for x in value)
    return False


def has_meaningful_supervision(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in META_DROP_KEYS:
                continue
            if key in COORD_KEYS and has_numeric_coord(value):
                return True
            if isinstance(value, list) and not value and (
                key == "candidates" or (key == "blocked_by" and "can_do_now" in obj)
            ):
                return True
            if isinstance(value, bool):
                return True
            if isinstance(value, (int, float)):
                return True
            if isinstance(value, str) and value.strip().lower() not in NULLISH_STRINGS:
                return True
            if isinstance(value, (dict, list)) and has_meaningful_supervision(value):
                return True
        return False
    if isinstance(obj, list):
        return any(has_meaningful_supervision(x) for x in obj)
    if isinstance(obj, str):
        return obj.strip().lower() not in NULLISH_STRINGS
    return obj is not None


def prune_nullish_fields(obj: Any, parent_key: str = "") -> Tuple[Any, Counter]:
    stats = Counter()
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            pruned, sub_stats = prune_nullish_fields(value, key)
            stats.update(sub_stats)
            keep_empty_list = isinstance(pruned, list) and not pruned and (
                key == "candidates" or (key == "blocked_by" and "can_do_now" in obj)
            )
            nullish = (
                pruned is None
                or (isinstance(pruned, str) and pruned.strip().lower() in NULLISH_STRINGS)
                or (isinstance(pruned, dict) and not pruned)
                or (isinstance(pruned, list) and not pruned and not keep_empty_list)
            )
            if nullish:
                joined = f"{parent_key}.{key}" if parent_key else key
                stats["nullish_fields_removed"] += 1
                stats[f"nullish_removed:{joined}"] += 1
                continue
            out[key] = pruned
        return out, stats
    if isinstance(obj, list):
        out_list = []
        for value in obj:
            pruned, sub_stats = prune_nullish_fields(value, parent_key)
            stats.update(sub_stats)
            if pruned is None or (isinstance(pruned, str) and pruned.strip().lower() in NULLISH_STRINGS) or (isinstance(pruned, dict) and not pruned):
                stats["nullish_list_items_removed"] += 1
                continue
            out_list.append(pruned)
        return out_list, stats
    return obj, stats


def clean_obj(obj: Any, dim: str) -> Tuple[Any, Counter]:
    stats = Counter()
    if not isinstance(obj, dict):
        return obj, stats
    size = obj.get("image_size")
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if k in META_DROP_KEYS:
            stats["metadata_keys_removed"] += 1
            continue
        if k in BBOX_KEYS:
            nv = norm_bbox(v, size)
            if nv != v:
                stats["coords_normalized"] += 1
            out[k] = nv
        elif k in POINT_KEYS:
            nv = norm_xy(v, size)
            if nv != v:
                stats["coords_normalized"] += 1
            out[k] = nv
        elif k in POINT_LIST_KEYS and isinstance(v, list):
            new_pts = [norm_xy(p, size) if isinstance(p, list) and len(p) == 2 else p for p in v]
            if new_pts != v:
                stats["coords_normalized"] += 1
            out[k] = new_pts
        else:
            out[k] = v
    out, prune_stats = prune_nullish_fields(out)
    stats.update(prune_stats)
    if dim != "unknown":
        stats[f"dim_{dim}"] += 1
    return out, stats


def make_prompt_only(rec: Dict[str, Any], dim: str) -> Dict[str, Any]:
    msgs = []
    for m in rec.get("messages") or []:
        if m.get("role") == "assistant":
            break
        msgs.append(m)
    if not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {"messages": msgs, "images": rec.get("images") or [], "dimension": dim}


def image_ok(rec: Dict[str, Any]) -> bool:
    imgs = rec.get("images") or []
    return bool(imgs) and all(os.path.isfile(p) for p in imgs)


def is_gsa_volc_image(rec: Dict[str, Any]) -> bool:
    imgs = rec.get("images") or []
    return any("GSA_volc" in str(p) for p in imgs)


def build_volc_image_index() -> Dict[str, Any]:
    local_rel = set()
    basename_index: Dict[str, str] = {}
    for base in ("images_for_upload", "images"):
        root = VOLC_ROOT / base
        if not root.exists():
            continue
        for dirpath, _, files in os.walk(root):
            rel_dir = os.path.relpath(dirpath, VOLC_ROOT)
            for fn in files:
                rel = os.path.join(rel_dir, fn)
                local_rel.add(rel)
                basename_index.setdefault(fn, rel)
                if fn.startswith("images_"):
                    basename_index.setdefault(fn[len("images_"):], rel)
    mapping = {}
    mapping_path = VOLC_ROOT / "image_mapping.json"
    if mapping_path.exists():
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception:
            mapping = {}
    mapping_key_base = {os.path.basename(str(k)): str(v) for k, v in mapping.items()}
    mapping_val_base = {os.path.basename(str(v)): str(v) for v in mapping.values()}
    return {
        "local_rel": local_rel,
        "basename_index": basename_index,
        "mapping_key_base": mapping_key_base,
        "mapping_val_base": mapping_val_base,
        "mapping_size": len(mapping),
    }


def resolve_volc_image(path: str, idx: Dict[str, Any]) -> Tuple[Optional[str], str]:
    if not path:
        return None, "empty_path"
    local_rel = idx["local_rel"]
    if path.startswith("/") and os.path.isfile(path):
        return path, "absolute_existing"
    if path.startswith(str(VOLC_ROOT) + "/"):
        rel = os.path.relpath(path, VOLC_ROOT)
        if rel in local_rel:
            return str(VOLC_ROOT / rel), "absolute_under_root"
    if path in local_rel:
        return str(VOLC_ROOT / path), "exact_relative"

    fn = os.path.basename(path)
    checks = [(idx["basename_index"].get(fn), "basename")]
    if fn.startswith("images_"):
        checks.append((idx["basename_index"].get(fn[len("images_"):]), "strip_images_prefix"))
    checks.extend([
        (idx["mapping_key_base"].get(fn), "mapping_key_basename"),
        (idx["mapping_val_base"].get(fn), "mapping_value_basename"),
    ])
    if fn.startswith("images_"):
        checks.extend([
            (idx["mapping_key_base"].get(fn[len("images_"):]), "mapping_key_strip"),
            (idx["mapping_val_base"].get(fn[len("images_"):]), "mapping_value_strip"),
        ])
    for rel, reason in checks:
        if rel and rel in local_rel:
            return str(VOLC_ROOT / rel), reason
    return None, "unresolved"


def iter_latest_volc_records(stats: Counter, drops: Counter, examples: Dict[str, List[str]]) -> Iterable[Dict[str, Any]]:
    idx = build_volc_image_index()
    stats["local_image_files_indexed"] = len(idx["local_rel"])
    stats["image_mapping_entries"] = idx["mapping_size"]
    inputs = [
        ("hf_train_sft", VOLC_ROOT / "train/sft_train.jsonl"),
        ("hf_annotations_sft", VOLC_ROOT / "annotations/sft_train.jsonl"),
        ("hf_bbox_grounding_1m", VOLC_ROOT / "train/bbox_grounding_1m/sft_train.jsonl"),
    ]
    for source_name, path in inputs:
        if not path.exists():
            drops[f"{source_name}:missing_input_file"] += 1
            continue
        for rec in iter_jsonl(path):
            stats[f"{source_name}:read"] += 1
            imgs = rec.get("images") or []
            if not imgs:
                drops[f"{source_name}:no_image_field"] += 1
                continue
            resolved = []
            reasons = []
            ok = True
            for img in imgs:
                r, why = resolve_volc_image(str(img), idx)
                reasons.append(why)
                if not r:
                    ok = False
                    break
                resolved.append(r)
            if not ok:
                drops[f"{source_name}:missing_image_unresolved"] += 1
                if len(examples["missing_image_unresolved"]) < 12:
                    examples["missing_image_unresolved"].append(str(imgs[0]))
                continue
            out = dict(rec)
            out["images"] = resolved
            stats[f"{source_name}:kept_image_resolved"] += 1
            for why in reasons:
                stats[f"image_resolve:{why}"] += 1
            yield out


def build_gsa_volc_v069(args: argparse.Namespace) -> Dict[str, Any]:
    src = VOLC_ROOT
    out_train = TRAINS / "gsa_volc_sft_v069.jsonl"
    out_val = TRAINS / "gsa_volc_val_v069.jsonl"
    out_grpo = TRAINS / "gsa_volc_grpo_v069.jsonl"
    stats = Counter()
    by_dim = Counter()
    by_source = Counter()
    drops = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    seen = set()

    train_f = out_train.open("w", encoding="utf-8")
    val_f = out_val.open("w", encoding="utf-8")
    grpo_f = out_grpo.open("w", encoding="utf-8")
    try:
        for rec in iter_latest_volc_records(stats, drops, examples):
            stats["read"] += 1
            if not image_ok(rec):
                drops["missing_image"] += 1
                continue
            raw_obj = parse_jsonish(assistant_text(rec))
            dim = infer_dim(raw_obj, rec)
            if isinstance(raw_obj, dict):
                src_name = str(raw_obj.get("source") or raw_obj.get("dataset") or "unknown")
            else:
                src_name = "unknown"
            img0 = str((rec.get("images") or [""])[0])
            if src_name == "other" or "/images/other/" in img0 or "/images_for_upload/other/" in img0:
                drops["source_other"] += 1
                if len(examples["source_other"]) < 8:
                    examples["source_other"].append(img0)
                continue
            if not isinstance(raw_obj, dict):
                drops["assistant_not_json"] += 1
                continue
            cleaned_obj, cstats = clean_obj(raw_obj, dim)
            stats.update(cstats)
            if not has_meaningful_supervision(cleaned_obj):
                drops["structural_empty_supervision"] += 1
                if len(examples["structural_empty_supervision"]) < 8:
                    examples["structural_empty_supervision"].append(json.dumps(raw_obj, ensure_ascii=False)[:240])
                continue
            ar = bbox_area_ratio(cleaned_obj.get("bbox")) if isinstance(cleaned_obj, dict) else None
            if ar is not None and ar < 0:
                drops["bad_bbox"] += 1
                continue
            if dim == "G_grounding" and ar is not None and ar >= 0.98:
                drops["grounding_full_image_bbox"] += 1
                continue

            new_rec = {"messages": [], "images": rec.get("images") or []}
            for m in rec.get("messages") or []:
                if m.get("role") == "assistant":
                    new_rec["messages"].append({"role": "assistant", "content": json.dumps(cleaned_obj, ensure_ascii=False, separators=(",", ":"))})
                elif m.get("role") in {"system", "user"}:
                    new_rec["messages"].append({"role": m.get("role"), "content": str(m.get("content", ""))})
            if not any(m.get("role") == "system" for m in new_rec["messages"]):
                new_rec["messages"].insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            k = record_key(new_rec)
            if k in seen:
                drops["duplicate"] += 1
                continue
            seen.add(k)
            by_dim[dim] += 1
            by_source[src_name] += 1
            split_hash = stable_int(prompt_key(new_rec), 10_000)
            if split_hash < args.volc_val_permille * 10:
                val_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                stats["val_written"] += 1
            else:
                train_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                grpo_f.write(json.dumps(make_prompt_only(new_rec, dim), ensure_ascii=False) + "\n")
                stats["train_written"] += 1
                stats["grpo_written"] += 1
    finally:
        train_f.close()
        val_f.close()
        grpo_f.close()

    return {
        "input": str(src),
        "outputs": {"sft": str(out_train), "val": str(out_val), "grpo": str(out_grpo)},
        "stats": dict(stats),
        "drops": dict(drops),
        "by_dim": dict(by_dim.most_common()),
        "by_source": dict(by_source.most_common(20)),
        "drop_examples": examples,
        "policy": {
            "drop_source_other": True,
            "drop_missing_images": True,
            "drop_grounding_full_image_bbox": True,
            "strip_metadata_keys": sorted(META_DROP_KEYS),
            "coordinate_system": "normalized integer [0,999]",
        },
    }


def sampled_pixmo_rows(path: Path, target: int) -> Iterable[Dict[str, Any]]:
    for rec in iter_jsonl(path):
        if stable_int(record_key(rec).hex(), 1_000_000) < target * 1_000_000 // 1_191_851:
            yield rec


def append_records(
    out_f,
    path: Path,
    source_name: str,
    seen: set,
    stats: Dict[str, Any],
    limit: Optional[int] = None,
    skip_gsa_volc: bool = False,
    check_image: bool = True,
    semantic_negative_cap: Optional[int] = None,
    semantic_negative_total: Optional[int] = None,
) -> int:
    kept = read = dup = skipped = 0
    neg_seen = neg_kept = neg_dropped = 0
    neg_threshold = None
    if semantic_negative_cap is not None and semantic_negative_total:
        neg_threshold = min(1_000_000, semantic_negative_cap * 1_000_000 // semantic_negative_total)
    for rec in iter_jsonl(path):
        read += 1
        if skip_gsa_volc and is_gsa_volc_image(rec):
            skipped += 1
            continue
        if check_image and not image_ok(rec):
            skipped += 1
            continue
        is_neg = is_semantic_negative(rec)
        if is_neg:
            neg_seen += 1
            keep_neg = True
            if semantic_negative_cap is not None:
                if neg_threshold is not None:
                    keep_neg = stable_int("neg:" + record_key(rec).hex(), 1_000_000) < neg_threshold
                else:
                    keep_neg = neg_kept < semantic_negative_cap
            if not keep_neg:
                skipped += 1
                neg_dropped += 1
                continue
        k = record_key(rec)
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
        if is_neg:
            neg_kept += 1
        if limit and kept >= limit:
            break
    stats[source_name] = {
        "path": str(path), "read": read, "kept": kept, "duplicates": dup, "skipped": skipped,
        "semantic_negative_seen": neg_seen,
        "semantic_negative_kept": neg_kept,
        "semantic_negative_dropped_by_cap": neg_dropped,
        "semantic_negative_cap": semantic_negative_cap,
    }
    return kept


def append_sampled_pixmo(out_f, seen: set, stats: Dict[str, Any], target: int, semantic_negative_cap: Optional[int]) -> int:
    path = TRAINS / "pixmo_points_tag_norm1000_v3.jsonl"
    total_known = 1_191_851
    negative_total_known = 196_761
    target = min(target, total_known)
    kept = read = dup = skipped = 0
    neg_seen = neg_kept = neg_dropped = 0
    threshold = min(1_000_000, target * 1_000_000 // total_known)
    neg_threshold = None
    if semantic_negative_cap is not None:
        neg_threshold = min(1_000_000, semantic_negative_cap * 1_000_000 // negative_total_known)
    for rec in iter_jsonl(path):
        read += 1
        if stable_int(record_key(rec).hex(), 1_000_000) >= threshold:
            continue
        if not image_ok(rec):
            skipped += 1
            continue
        is_neg = is_semantic_negative(rec)
        if is_neg:
            neg_seen += 1
            keep_neg = True
            if neg_threshold is not None:
                keep_neg = stable_int("pixmo-neg:" + record_key(rec).hex(), 1_000_000) < neg_threshold
            elif semantic_negative_cap is not None:
                keep_neg = neg_kept < semantic_negative_cap
            if not keep_neg:
                skipped += 1
                neg_dropped += 1
                continue
        k = record_key(rec)
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
        if is_neg:
            neg_kept += 1
        if kept >= target:
            break
    stats["pixmo_points_tag_v3_replay"] = {
        "path": str(path), "read": read, "kept": kept, "duplicates": dup,
        "skipped": skipped, "target": target, "sampling": "stable hash cap",
        "semantic_negative_seen": neg_seen,
        "semantic_negative_kept": neg_kept,
        "semantic_negative_dropped_by_cap": neg_dropped,
        "semantic_negative_cap": semantic_negative_cap,
    }
    return kept


def build_merged_v069(args: argparse.Namespace) -> Dict[str, Any]:
    out_train = TRAINS / "gsa_train_v069.jsonl"
    out_val = TRAINS / "gsa_val_v069.jsonl"
    out_grpo = TRAINS / "gsa_grpo_train_v069.jsonl"
    out_opd = TRAINS / "gsa_opd_train_v069.jsonl"
    seen = set()
    per_source: Dict[str, Any] = {}
    total = 0

    with out_train.open("w", encoding="utf-8") as f:
        total += append_records(
            f,
            TRAINS / "gsa_sft_train.jsonl",
            "base_train_without_old_gsa_volc",
            seen,
            per_source,
            skip_gsa_volc=True,
            check_image=False,
            semantic_negative_cap=args.base_negative_cap,
            semantic_negative_total=196_913,
        )
        for name, path in [
            ("gsa_volc_sft_v069", TRAINS / "gsa_volc_sft_v069.jsonl"),
            ("refcoco_grounding_tag_norm1000_v3", TRAINS / "refcoco_grounding_tag_norm1000_v3.jsonl"),
            ("desktop_bbox_fused_train", TRAINS / "desktop_bbox_fused_train.jsonl"),
            ("desktop_bbox_shortname_phrasingdedup_train", TRAINS / "desktop_bbox_shortname_phrasingdedup_train.jsonl"),
            ("gsa_allcorrect_train_native_v1", TRAINS / "gsa_allcorrect_train_native_v1.jsonl"),
            ("aobjstate_verifier_v1", TRAINS / "aobjstate_verifier_v1.jsonl"),
        ]:
            total += append_records(f, path, name, seen, per_source)
        total += append_sampled_pixmo(f, seen, per_source, args.pixmo_replay_cap, args.pixmo_negative_cap)

    val_stats: Dict[str, Any] = {}
    val_seen = set()
    val_total = 0
    with out_val.open("w", encoding="utf-8") as f:
        for name, path in [
            ("desktop_bbox_fused_val", TRAINS / "desktop_bbox_fused_val.jsonl"),
            ("desktop_bbox_shortname_phrasingdedup_val", TRAINS / "desktop_bbox_shortname_phrasingdedup_val.jsonl"),
            ("gsa_volc_val_v069", TRAINS / "gsa_volc_val_v069.jsonl"),
        ]:
            val_total += append_records(f, path, name, val_seen, val_stats)

    grpo_n = opd_n = 0
    with out_grpo.open("w", encoding="utf-8") as fg, out_opd.open("w", encoding="utf-8") as fo:
        for rec in iter_jsonl(out_train):
            dim = infer_dim(parse_jsonish(assistant_text(rec)), rec)
            pk = prompt_key(rec)
            if stable_int("grpo:" + pk, 10_000) < int(args.grpo_ratio * 10_000):
                fg.write(json.dumps(make_prompt_only(rec, dim), ensure_ascii=False) + "\n")
                grpo_n += 1
            if stable_int("opd:" + pk, 10_000) < int(args.opd_ratio * 10_000):
                fo.write(json.dumps(make_prompt_only(rec, dim), ensure_ascii=False) + "\n")
                opd_n += 1

    return {
        "outputs": {
            "train": str(out_train),
            "val": str(out_val),
            "grpo": str(out_grpo),
            "opd": str(out_opd),
        },
        "counts": {"train": total, "val": val_total, "grpo": grpo_n, "opd": opd_n},
        "per_source": per_source,
        "validation_sources": val_stats,
        "policy": {
            "base_train_filter": "drop rows whose image path contains GSA_volc, then re-add cleaned gsa_volc_sft_v069",
            "pixmo_replay_cap": args.pixmo_replay_cap,
            "base_semantic_negative_cap": args.base_negative_cap,
            "pixmo_semantic_negative_cap": args.pixmo_negative_cap,
            "semantic_negative_policy": "stable hash cap for absence-only answers such as 'There are none.'",
            "dedup_key": "sha1(image,user,assistant)",
        },
    }


def audit_file(path: Path, sample_n: int = 20000) -> Dict[str, Any]:
    by_dim = Counter()
    by_shape = Counter()
    by_img_prefix = Counter()
    bad_img = 0
    json_fail = 0
    sampled = 0
    bbox_full = 0
    semantic_negative = 0
    structural_empty = 0
    nullish_keys = Counter()
    for rec in iter_jsonl(path):
        sampled += 1
        if not image_ok(rec):
            bad_img += 1
        if semantic_negative_kind(assistant_text(rec)):
            semantic_negative += 1
        obj = parse_jsonish(assistant_text(rec))
        if obj is None and any(m.get("role") == "assistant" for m in rec.get("messages") or []):
            json_fail += 1
        dim = infer_dim(obj, rec)
        by_dim[dim] += 1
        if isinstance(obj, dict):
            _, prune_stats = prune_nullish_fields(obj)
            for key, value in prune_stats.items():
                if str(key).startswith("nullish_removed:"):
                    nullish_keys[str(key).split(":", 1)[1]] += int(value)
            if not has_meaningful_supervision(obj):
                structural_empty += 1
            if "bbox" in obj:
                by_shape["bbox"] += 1
                ar = bbox_area_ratio(obj.get("bbox"))
                if ar and ar >= 0.98:
                    bbox_full += 1
            if "point" in obj or "affordance_point" in obj or "grasp_point_2d" in obj:
                by_shape["point"] += 1
            if "attributes" in obj:
                by_shape["attributes"] += 1
            if "predicate" in obj:
                by_shape["spatial"] += 1
        else:
            txt = assistant_text(rec)
            if "<bbox>" in txt:
                by_shape["bbox_tag"] += 1
            elif "<point>" in txt:
                by_shape["point_tag"] += 1
            else:
                by_shape["text_or_choice"] += 1
        img = str((rec.get("images") or [""])[0])
        parts = img.split("/")
        prefix = "/".join(parts[:6]) if img.startswith("/") else "/".join(parts[:2])
        by_img_prefix[prefix] += 1
        if sampled >= sample_n:
            break
    return {
        "sampled": sampled,
        "image_missing": bad_img,
        "assistant_json_fail_or_non_json": json_fail,
        "semantic_negative": semantic_negative,
        "structural_empty_json": structural_empty,
        "bbox_full_image": bbox_full,
        "by_dim": dict(by_dim.most_common()),
        "by_answer_shape": dict(by_shape.most_common()),
        "by_image_prefix": dict(by_img_prefix.most_common(20)),
        "top_nullish_keys": dict(nullish_keys.most_common(20)),
    }


def thumb_b64(path: str, max_side: int = 220) -> str:
    if Image is None or not os.path.isfile(path):
        return ""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = max_side / max(w, h)
        if s < 1:
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
        bio = io.BytesIO()
        im.save(bio, "JPEG", quality=72)
        return base64.b64encode(bio.getvalue()).decode("ascii")
    except Exception:
        return ""


def esc(x: Any) -> str:
    return html.escape(str(x))


def build_html(report: Dict[str, Any]) -> None:
    out = ROOT / "trains/diag/out/gsa_v069_data_inspection.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    volc = report["gsa_volc_v069"]
    merged = report["merged_v069"]
    rows = []
    sample_paths = [
        ("gsa_volc_sft_v069", TRAINS / "gsa_volc_sft_v069.jsonl", 15),
        ("desktop_bbox_fused", TRAINS / "desktop_bbox_fused_train.jsonl", 10),
        ("allcorrect", TRAINS / "gsa_allcorrect_train_native_v1.jsonl", 10),
        ("refcoco_tag", TRAINS / "refcoco_grounding_tag_norm1000_v3.jsonl", 10),
        ("merged_train_v069", TRAINS / "gsa_train_v069.jsonl", 20),
    ]
    for source, path, limit in sample_paths:
        n = 0
        for rec in iter_jsonl(path):
            if not image_ok(rec):
                continue
            img = (rec.get("images") or [""])[0]
            b64 = thumb_b64(img)
            if not b64:
                continue
            rows.append(
                f"<tr><td>{esc(source)}</td><td><img src='data:image/jpeg;base64,{b64}'></td>"
                f"<td class='qa'>{esc(user_text(rec)[:220])}</td><td class='qa'>{esc(assistant_text(rec)[:220])}</td></tr>"
            )
            n += 1
            if n >= limit:
                break
    source_rows = ""
    for name, st in merged["per_source"].items():
        source_rows += f"<tr><td>{esc(name)}</td><td class=num>{st.get('read', 0):,}</td><td class=num>{st.get('kept', 0):,}</td><td class=num>{st.get('skipped', 0):,}</td><td class=num>{st.get('duplicates', 0):,}</td></tr>"
    drop_rows = "".join(f"<tr><td>{esc(k)}</td><td class=num>{v:,}</td></tr>" for k, v in volc["drops"].items())
    dim_rows = "".join(f"<tr><td>{esc(k)}</td><td class=num>{v:,}</td></tr>" for k, v in volc["by_dim"].items())
    audit_train = report["audits"]["train_sample"]
    empty_policy = report.get("empty_semantics_policy", {})
    empty_rows = ""
    for name, st in merged["per_source"].items():
        if st.get("semantic_negative_seen") or st.get("semantic_negative_dropped_by_cap"):
            empty_rows += (
                f"<tr><td>{esc(name)}</td><td class=num>{st.get('semantic_negative_seen', 0):,}</td>"
                f"<td class=num>{st.get('semantic_negative_kept', 0):,}</td>"
                f"<td class=num>{st.get('semantic_negative_dropped_by_cap', 0):,}</td>"
                f"<td class=num>{st.get('semantic_negative_cap') or ''}</td></tr>"
            )
    doc = f"""<!doctype html><html lang=zh><head><meta charset=utf-8><title>GSA v069 数据诊断与改进报告</title><style>
body{{font-family:-apple-system,Segoe UI,'PingFang SC','Microsoft YaHei',sans-serif;max-width:1280px;margin:18px auto;padding:0 14px;color:#111827}}
h1{{border-bottom:3px solid #2563eb;padding-bottom:8px}} h2{{color:#1e3a8a;margin-top:24px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}} th,td{{border:1px solid #e5e7eb;padding:6px 8px;vertical-align:top}} th{{background:#f1f5f9}}
.num{{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} .ok{{color:#16a34a;font-weight:700}} .bad{{color:#dc2626;font-weight:700}}
.hl{{background:#eff6ff;border-left:4px solid #2563eb;padding:10px 14px;border-radius:6px;margin:10px 0}} .warn{{background:#fff7ed;border-left:4px solid #f97316;padding:10px 14px;border-radius:6px;margin:10px 0}}
td img{{width:210px;display:block}} .qa{{max-width:360px;font-size:12px;white-space:pre-wrap}}
</style></head><body>
<h1>GSA v069 数据诊断与改进报告</h1>
<div class=hl><b>结论:</b> v069 生成了干净 GSA_volc pack 子集 <code>gsa_volc_sft_v069.jsonl</code>，并合并出 5M+ SFT 总训练集。旧版问题集中在 source=other 图文错配、坐标尺度混杂、source/provenance 字段污染输出、GSA_volc 图片缺失导致的不可训练样本；v069 已做可读图片过滤、source=other 剔除、坐标统一到 [0,999]、metadata 字段剥离、旧 GSA_volc 行替换，并补入 PixMo tag/norm1000 replay 与桌面 bbox 改写提示。</div>
<div class=warn><b>HF 同步状态:</b> 已强制同步 core manifest / train JSONL / image_mapping，并按 manifest 单文件补齐可下载图片；但仓库中 <code>annotations/sft_train.jsonl</code> 与 <code>bbox_grounding_1m</code> 大量引用的 <code>images/visual_genome/*.jpg</code> 仍有 404/未上传问题，<code>train/sft_train.jsonl</code> 的部分扁平 <code>images_for_upload/*</code> 路径也和实际子目录不一致。本地可索引图片 {report['sync']['local_image_files']:,} 张；v069 使用路径解析器修复可定位图片，并严格过滤缺图样本。</div>
<h2>① 输出规模</h2>
<table><tr><th>文件</th><th>样本数</th></tr>
<tr><td>gsa_volc_sft_v069</td><td class=num>{volc['stats'].get('train_written',0):,}</td></tr>
<tr><td>gsa_volc_val_v069</td><td class=num>{volc['stats'].get('val_written',0):,}</td></tr>
<tr><td>gsa_train_v069</td><td class=num>{merged['counts']['train']:,}</td></tr>
<tr><td>gsa_val_v069</td><td class=num>{merged['counts']['val']:,}</td></tr>
<tr><td>gsa_grpo_train_v069</td><td class=num>{merged['counts']['grpo']:,}</td></tr>
<tr><td>gsa_opd_train_v069</td><td class=num>{merged['counts']['opd']:,}</td></tr></table>
<h2>② GSA_volc v069 清洗</h2>
<table><tr><th>drop reason</th><th>count</th></tr>{drop_rows}</table>
<table><tr><th>dimension</th><th>kept count</th></tr>{dim_rows}</table>
<h2>③ 合并源贡献</h2>
<table><tr><th>source</th><th>read</th><th>kept</th><th>skipped</th><th>duplicates</th></tr>{source_rows}</table>
<h2>④ 训练抽样审计</h2>
<pre>{esc(json.dumps(audit_train, ensure_ascii=False, indent=2))}</pre>
<h2>⑤ 空信息治理</h2>
<div class=hl><b>策略:</b> {esc(empty_policy.get('summary', ''))}</div>
<table><tr><th>source</th><th>semantic negative seen</th><th>kept</th><th>dropped by cap</th><th>cap</th></tr>{empty_rows}</table>
<pre>{esc(json.dumps(volc.get('drop_examples', {}).get('structural_empty_supervision', []), ensure_ascii=False, indent=2))}</pre>
<h2>⑥ 相比旧数据的主要改进</h2>
<table><tr><th>旧问题</th><th>v069 处理</th><th>影响</th></tr>
<tr><td>GSA_volc source=other 里图-标注错配/用错帧</td><td>解析 assistant JSON 的 source/dataset，剔除 source=other 和 images/other 路径</td><td>降低属性/状态幻觉和错误 bbox 监督</td></tr>
<tr><td>GSA_volc 输出带 source/dataset/image_size 等训练无关字段</td><td>assistant JSON 中剥离 provenance/metadata 字段</td><td>减少模型输出无关字段，能力边界更清晰</td></tr>
<tr><td>GSA_volc 存在 object=null、robot_state={{}}、distance=&quot;&quot; 这类纯空监督</td><td>递归删除 null/空字符串/空 dict 的可选字段；清完后无有效监督信号的样本直接丢弃</td><td>避免训练模型输出无对象、无状态、无距离的空 JSON</td></tr>
<tr><td>base 与 PixMo replay 重复包含大量 There are none.</td><td>对 absence-only 答案做稳定 hash cap，保留校准拒答能力但降低占比</td><td>减少训练分布被空语义样本主导</td></tr>
<tr><td>bbox/point 同时存在像素坐标和 0-1000 坐标</td><td>有 image_size 时统一归一化到 [0,999]</td><td>统一评测/训练坐标语义</td></tr>
<tr><td>旧总训练集中 GSA_volc 脏行已混入</td><td>合并时先剔除旧 GSA_volc image path，再加入 gsa_volc_sft_v069</td><td>避免旧脏样本残留</td></tr>
<tr><td>能力来源不清楚</td><td>合并报告按源列出贡献：base、GSA_volc、desktop、desktop phrasing、allcorrect、aobjstate、refcoco、pixmo replay</td><td>可定位能力提升/退化来源</td></tr>
</table>
<h2>⑦ 抽样图文检查</h2>
<table><tr><th>source</th><th>image</th><th>prompt</th><th>answer</th></tr>{''.join(rows)}</table>
</body></html>"""
    out.write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixmo_replay_cap", type=int, default=1_191_851)
    ap.add_argument("--base_negative_cap", type=int, default=150_000)
    ap.add_argument("--pixmo_negative_cap", type=int, default=150_000)
    ap.add_argument("--volc_val_permille", type=int, default=20)
    ap.add_argument("--grpo_ratio", type=float, default=0.3)
    ap.add_argument("--opd_ratio", type=float, default=0.2)
    ap.add_argument("--audit_sample", type=int, default=20000)
    args = ap.parse_args()

    local_images = 0
    for sub in ("images", "images_for_upload"):
        base = VOLC_ROOT / sub
        for _, _, files in os.walk(base):
            local_images += len(files)
    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v069",
        "sync": {
            "volc_root": str(VOLC_ROOT),
            "local_image_files": local_images,
            "hf_probe": {
                "path": "images/visual_genome/1159966.jpg",
                "status": "404 via hf download on 2026-06-22 after force-syncing manifests; v069 filters unresolved images",
            },
            "policy": "force-download core files, use image_mapping/local basename resolver, keep only locally readable images",
        },
        "empty_semantics_policy": {
            "summary": (
                "Drop GSA_volc JSON records that become supervision-empty after nullish field pruning; "
                "cap absence-only text answers with stable hashing in base and PixMo replay."
            ),
            "semantic_negative_patterns": [p.pattern for p in SEMANTIC_NEGATIVE_PATTERNS],
            "base_negative_cap": args.base_negative_cap,
            "pixmo_negative_cap": args.pixmo_negative_cap,
            "nullish_strings": sorted(NULLISH_STRINGS),
            "meaningful_empty_list_keys": sorted(MEANINGFUL_EMPTY_LIST_KEYS),
        },
    }
    print("[1] Build cleaned GSA_volc v069", flush=True)
    report["gsa_volc_v069"] = build_gsa_volc_v069(args)
    print("[2] Build merged v069 train/val/GRPO/OPD", flush=True)
    report["merged_v069"] = build_merged_v069(args)
    print("[3] Audits", flush=True)
    report["audits"] = {
        "gsa_volc_sample": audit_file(TRAINS / "gsa_volc_sft_v069.jsonl", args.audit_sample),
        "train_sample": audit_file(TRAINS / "gsa_train_v069.jsonl", args.audit_sample),
        "val_sample": audit_file(TRAINS / "gsa_val_v069.jsonl", args.audit_sample),
    }
    report_path = TRAINS / "gsa_v069_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    build_html(report)
    print(json.dumps({
        "gsa_volc_sft_v069": report["gsa_volc_v069"]["stats"].get("train_written", 0),
        "gsa_train_v069": report["merged_v069"]["counts"]["train"],
        "gsa_val_v069": report["merged_v069"]["counts"]["val"],
        "gsa_grpo_train_v069": report["merged_v069"]["counts"]["grpo"],
        "gsa_opd_train_v069": report["merged_v069"]["counts"]["opd"],
        "report": str(report_path),
        "html": str(ROOT / "trains/diag/out/gsa_v069_data_inspection.html"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
