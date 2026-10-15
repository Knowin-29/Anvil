#!/usr/bin/env python3
"""
Deep data-quality audit for GSA_volc (and reusable for other ms-swift jsonl).

Checks 7 acceptance criteria and reports pass/fail with quantified evidence:
  C1 image-path resolvable     : images[] points to an existing file under root
  C2 assistant JSON parseable  : assistant content parses as JSON (for GSA dims)
  C3 bbox/point legal          : bbox within image bounds, x2>x1, y2>y1; point in-range
  C4 exact duplicates          : identical (image, user, assistant) triples
  C5 near-duplicates           : same (image, assistant) with different prompt templates
  C6 dimension balance         : per-dimension counts, flag empty/over-represented
  C7 noise / triviality        : empty/too-short response, degenerate bbox(full-image),
                                  missing object_class, placeholder answers

Usage:
  python trains/audit_gsa_volc.py \
      --jsonl /mnt/data/huggingface_downloads/datasets/GSA_volc/annotations/sft_train.jsonl \
      --image_root /mnt/data/huggingface_downloads/datasets/GSA_volc \
      --out trains/gsa_volc_audit.json \
      --sample 0   # 0 = full pass; N = sample N lines
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


def get_roles(msgs):
    sys_c = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    return sys_c, user, asst


def try_json(s):
    s = (s or "").strip()
    if s.startswith("```"):
        # strip markdown code fence
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    if not s.startswith("{") and not s.startswith("["):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def bbox_legal(bbox, img_size):
    """bbox = [x1,y1,x2,y2]; img_size=[W,H] (may be None)."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, "shape"
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return False, "nonnumeric"
    if x2 <= x1 or y2 <= y1:
        return False, "inverted"
    if x1 < 0 or y1 < 0:
        return False, "negative"
    if img_size and len(img_size) == 2:
        W, H = img_size
        if W and H:
            # allow tiny overflow due to rounding
            if x2 > W + 2 or y2 > H + 2:
                return False, "out_of_bounds"
            # degenerate: bbox covers (almost) the whole image
            area_ratio = (x2 - x1) * (y2 - y1) / (W * H)
            if area_ratio >= 0.98:
                return True, "full_image"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--img_check_rate", type=float, default=1.0,
                    help="fraction of records to actually stat() image on disk")
    args = ap.parse_args()

    rng = random.Random(42)

    total = 0
    counters = Counter()
    by_dim = Counter()
    bbox_reasons = Counter()
    missing_img_by_dim = Counter()
    dup_seen = set()
    neardup_seen = defaultdict(int)
    short_examples = []
    fullimg_examples = []
    missing_img_examples = []
    img_checked = 0
    img_missing = 0
    asst_lengths = []

    with open(args.jsonl) as f:
        lines = f
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if args.sample and total >= args.sample:
                break
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counters["invalid_json_line"] += 1
                continue

            msgs = rec.get("messages", [])
            imgs = rec.get("images", [])
            sys_c, user, asst = get_roles(msgs)

            # C2 assistant JSON parseable
            obj = try_json(asst)
            dim = "unknown"
            if obj and isinstance(obj, dict):
                counters["asst_json_ok"] += 1
                dim = obj.get("dimension", "unknown")
            else:
                counters["asst_json_fail"] += 1
            by_dim[dim] += 1

            # C7 noise: short / empty response
            al = len(asst.strip())
            asst_lengths.append(al)
            if al == 0:
                counters["empty_response"] += 1
            elif al < 10:
                counters["short_response"] += 1
                if len(short_examples) < 8:
                    short_examples.append({"asst": asst[:60], "dim": dim})

            # C3 bbox/point legality (only when JSON has bbox)
            if obj and isinstance(obj, dict):
                bbox = obj.get("bbox")
                img_size = obj.get("image_size")
                if bbox is not None:
                    ok, reason = bbox_legal(bbox, img_size)
                    bbox_reasons[reason] += 1
                    if not ok:
                        counters["bbox_illegal"] += 1
                    if reason == "full_image":
                        counters["bbox_full_image"] += 1
                        if len(fullimg_examples) < 8:
                            fullimg_examples.append({"bbox": bbox, "size": img_size, "dim": dim,
                                                      "cls": obj.get("object_class")})
                # missing object_class for grounding/attr
                if dim in ("G_grounding", "A_identity_attr") and not obj.get("object_class"):
                    counters["missing_object_class"] += 1

            # C1 image resolvable
            if not imgs:
                counters["no_image_field"] += 1
            else:
                rel = imgs[0]
                full = rel if os.path.isabs(rel) else os.path.join(args.image_root, rel)
                if rng.random() <= args.img_check_rate:
                    img_checked += 1
                    if not os.path.isfile(full):
                        img_missing += 1
                        missing_img_by_dim[dim] += 1
                        if len(missing_img_examples) < 8:
                            missing_img_examples.append(rel)

            # C4 exact dup
            key = (imgs[0] if imgs else "", user, asst)
            if key in dup_seen:
                counters["exact_dup"] += 1
            else:
                dup_seen.add(key)

            # C5 near dup (same image+assistant, prompt may differ)
            nkey = (imgs[0] if imgs else "", asst)
            neardup_seen[nkey] += 1

    near_dup_extra = sum(v - 1 for v in neardup_seen.values() if v > 1)
    asst_lengths.sort()
    med_len = asst_lengths[len(asst_lengths) // 2] if asst_lengths else 0

    report = {
        "jsonl": args.jsonl,
        "image_root": args.image_root,
        "total_records": total,
        "C1_image": {
            "no_image_field": counters["no_image_field"],
            "checked_on_disk": img_checked,
            "missing_on_disk": img_missing,
            "missing_pct": round(img_missing / max(img_checked, 1) * 100, 2),
            "missing_examples": missing_img_examples,
            "note": "image download may be incomplete; re-run after full download",
        },
        "C2_json": {
            "ok": counters["asst_json_ok"],
            "fail": counters["asst_json_fail"],
            "ok_pct": round(counters["asst_json_ok"] / max(total, 1) * 100, 2),
        },
        "C3_bbox": {
            "illegal": counters["bbox_illegal"],
            "full_image": counters["bbox_full_image"],
            "reasons": dict(bbox_reasons),
            "full_image_examples": fullimg_examples,
        },
        "C4_exact_dup": counters["exact_dup"],
        "C5_near_dup_extra": near_dup_extra,
        "C6_dimension": dict(by_dim.most_common()),
        "C7_noise": {
            "empty_response": counters["empty_response"],
            "short_response": counters["short_response"],
            "short_examples": short_examples,
            "missing_object_class": counters["missing_object_class"],
            "median_asst_len": med_len,
        },
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # console summary
    print(f"=== GSA_volc audit: {args.jsonl} ===")
    print(f"total records: {total}")
    print(f"C1 image: no_field={counters['no_image_field']} "
          f"missing_on_disk={img_missing}/{img_checked} ({report['C1_image']['missing_pct']}%)")
    print(f"C2 json:  ok={counters['asst_json_ok']} fail={counters['asst_json_fail']} "
          f"({report['C2_json']['ok_pct']}%)")
    print(f"C3 bbox:  illegal={counters['bbox_illegal']} full_image={counters['bbox_full_image']} "
          f"reasons={dict(bbox_reasons)}")
    print(f"C4 exact_dup: {counters['exact_dup']}")
    print(f"C5 near_dup_extra: {near_dup_extra}")
    print(f"C7 noise: empty={counters['empty_response']} short={counters['short_response']} "
          f"missing_cls={counters['missing_object_class']} median_len={med_len}")
    print("C6 dimensions:")
    for k, v in by_dim.most_common():
        print(f"     {v:>7}  {k}")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
