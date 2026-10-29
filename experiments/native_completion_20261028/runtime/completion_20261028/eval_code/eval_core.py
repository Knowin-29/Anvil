"""Small, deterministic integrity and failure-detection helpers."""
import hashlib
import json
import math

BUDGETS = {"bbox": 2048, "detection": 8192, "mask": 16384, "point": 16384,
           "direction": 512, "metric_comparison": 512, "relation_graph": 8192,
           "depth_order": 2048, "affordance": 512, "dof": 2048, "action_cot": 2048}
FAMILIES = {**dict.fromkeys(("bbox", "detection", "mask", "point"), "G"),
            **dict.fromkeys(("depth_order", "metric_comparison", "direction", "relation_graph"), "S"),
            **dict.fromkeys(("affordance", "dof", "action_cot"), "A")}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def repeat_period(tokens):
    """Flag only exact, sustained cycles; never synthesize or repair an answer."""
    if len(tokens) < 512:
        return None
    for period in range(1, 33):
        repeats = max(16, math.ceil(256 / period))
        n = period * repeats
        if list(tokens[-n:]) == list(tokens[-period:]) * repeats:
            return {"period_tokens": period, "identical_repeats": repeats, "span_tokens": n}
    return None


def select_files(manifest, scope, shard=0, shards=1):
    if scope not in {"G", "S", "A", "joint"} or not 0 <= shard < shards:
        raise ValueError("Invalid scope/shard")
    if (manifest["records"] != 580000 or set(manifest["by_form"]) != set(FAMILIES)
            or manifest["dropped_records"] != 0 or manifest["targets_repaired"] != 0
            or sum(manifest["by_form"].values()) != 580000):
        raise ValueError("Full unchanged 580K/11-form export required")
    result = []
    for name, info in manifest["files"].items():
        form = info["form"]
        if form not in FAMILIES or name != f"eval_{form}.jsonl" or info["records"] != manifest["by_form"][form]:
            raise ValueError("Invalid form inventory")
        if scope == "joint" or FAMILIES[form] == scope:
            result.append((name, {**info, "shard_records": max(0, (info["records"] - shard + shards - 1) // shards)}))
    return sorted(result, key=lambda pair: pair[1]["records"])


def validate_row(row, form, selection_sha):
    meta = row["evaluation_meta"]
    if (row["native_form"] != form or row["family"] != FAMILIES[form]
            or row["training_approved"] is not False or meta["split"] != "benchmark"
            or meta["selection_sha256"] != selection_sha
            or meta["evaluation_content_sha256"] != digest({k: v for k, v in row.items() if k != "evaluation_meta"})):
        raise ValueError("Input identity/split changed")


def check_saved(item, row, ordinal, run_sha):
    if (item["id"] != row["id"] or item["ordinal"] != ordinal
            or item["protocol_sha256"] != run_sha
            or item["evaluation_content_sha256"] != row["evaluation_meta"]["evaluation_content_sha256"]):
        raise ValueError("Saved output does not match the exact input and protocol")
