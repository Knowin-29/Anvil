#!/usr/bin/env python3
"""
bbox 1M 数据紧急补充 - 整合多来源 bbox/point/affordance grounding 数据
目标: 1M+ samples, bbox > 50%

数据源:
1. VG G_grounding converted (787K bbox)
2. VG S_spatial converted (509K -> 拆为 ~1M bbox)
3. 飞轮 verified (144K bbox)
4. 飞轮 labels (187K bbox, 过滤 conf>=0.7)
5. A_action_interface (25K affordance)
6. Point: 从 bbox 中心/grasp_point 生成
"""
import json
import os
import random
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

random.seed(42)

# Paths
GSA_VOLC = Path("/path/to/datasets/GSA_volc")
FLYWHEEL_ROOT = Path("/path/to/Anvil/data/output")
OUTPUT_DIR = GSA_VOLC / "annotations" / "bbox_grounding_1m"
IMAGES_DIR = GSA_VOLC / "images_for_upload"

SYSTEM_PROMPT = (
    "You are an embodied perception assistant. "
    "Detect objects and provide precise bounding boxes in [0,999] coordinates."
)

# Quality thresholds
MIN_BBOX_AREA_PCT = 0.01   # 1% of image
MAX_BBOX_AREA_PCT = 0.70   # 70% of image
MIN_CONFIDENCE = 0.5


def validate_bbox(bbox):
    """Validate bbox [x1,y1,x2,y2] in [0,999]"""
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (ValueError, TypeError):
        return False
    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
        return False
    area = (x2 - x1) * (y2 - y1)
    total = 999 * 999
    if area / total < MIN_BBOX_AREA_PCT or area / total > MAX_BBOX_AREA_PCT:
        return False
    return True


def bbox_to_point(bbox):
    """Convert bbox center to point [x, y]"""
    x1, y1, x2, y2 = bbox
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def make_bbox_sft(obj_name, bbox, img_path, source="vg"):
    """Create bbox SFT sample"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\nLocate the '{obj_name}' in this image. Provide bounding box [x1,y1,x2,y2] in [0,999]."},
            {"role": "assistant", "content": json.dumps({"object": obj_name, "bbox": bbox}, ensure_ascii=False)},
        ],
        "images": [img_path],
        "_type": "bbox",
        "_source": source,
    }


def make_point_sft(obj_name, point, img_path, source="vg"):
    """Create point SFT sample"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\nPoint to the '{obj_name}' in this image. Provide [x,y] in [0,999]."},
            {"role": "assistant", "content": json.dumps({"object": obj_name, "point": point}, ensure_ascii=False)},
        ],
        "images": [img_path],
        "_type": "point",
        "_source": source,
    }


def make_affordance_sft(obj_name, bbox, action_type, grasp_point, img_path, source="flywheel"):
    """Create affordance grounding SFT sample"""
    resp = {"object": obj_name, "bbox": bbox, "action_type": action_type}
    if grasp_point:
        resp["grasp_point"] = grasp_point
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\nHow to {action_type} the '{obj_name}'? Provide bbox and grasp point in [0,999]."},
            {"role": "assistant", "content": json.dumps(resp, ensure_ascii=False)},
        ],
        "images": [img_path],
        "_type": "affordance",
        "_source": source,
    }


def load_vg_grounding():
    """Load VG G_grounding (787K bbox samples)"""
    path = GSA_VOLC / "annotations/converted/G_grounding/converted.jsonl"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []
    samples = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            bbox = r.get("bbox")
            obj = r.get("object_class", "")
            img = r.get("image_path", "")
            if not validate_bbox(bbox) or not obj or not img:
                continue
            samples.append(make_bbox_sft(obj, bbox, img, "vg_grounding"))
    return samples


def load_vg_spatial_split():
    """Load VG S_spatial and split into bbox samples"""
    path = GSA_VOLC / "annotations/converted/S_spatial/converted.jsonl"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []
    samples = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            img = r.get("image_path", "")
            if not img:
                continue
            # Split subject
            subj_bbox = r.get("subject_bbox")
            subj_name = r.get("subject", "")
            if validate_bbox(subj_bbox) and subj_name:
                samples.append(make_bbox_sft(subj_name, subj_bbox, img, "vg_spatial"))
            # Split object
            obj_bbox = r.get("object_bbox")
            obj_name = r.get("object", "")
            if validate_bbox(obj_bbox) and obj_name:
                samples.append(make_bbox_sft(obj_name, obj_bbox, img, "vg_spatial"))
    return samples


def load_flywheel_verified():
    """Load all flywheel verified bbox samples"""
    samples = []
    for jf in sorted(FLYWHEEL_ROOT.rglob("*.jsonl")):
        if "/verified/" not in str(jf):
            continue
        with open(jf) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                bbox = r.get("bbox")
                obj = r.get("object_class", r.get("object_name", ""))
                img = r.get("image_path", "")
                if not img or not obj:
                    continue
                if validate_bbox(bbox):
                    samples.append(make_bbox_sft(obj, bbox, img, "flywheel_verified"))
                # Affordance samples
                grasp = r.get("grasp_point_2d")
                action = r.get("action_type")
                if action and validate_bbox(bbox):
                    samples.append(make_affordance_sft(obj, bbox, action, grasp, img, "flywheel_verified"))
                # Point from grasp_point
                if grasp and len(grasp) == 2:
                    samples.append(make_point_sft(obj, grasp, img, "flywheel_verified"))
    return samples


def load_flywheel_labels():
    """Load flywheel labels (less strict, filter by confidence)"""
    samples = []
    for jf in sorted(FLYWHEEL_ROOT.rglob("*.jsonl")):
        if "/labels/" not in str(jf):
            continue
        with open(jf) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                conf = r.get("confidence", 0)
                if conf < MIN_CONFIDENCE:
                    continue
                bbox = r.get("bbox")
                obj = r.get("object_class", r.get("object_name", ""))
                img = r.get("image_path", "")
                if not img or not obj or not validate_bbox(bbox):
                    continue
                samples.append(make_bbox_sft(obj, bbox, img, "flywheel_labels"))
    return samples


def generate_points_from_bbox(bbox_samples, count=200000):
    """Generate point samples from bbox center points"""
    source = random.sample(bbox_samples, min(count, len(bbox_samples)))
    points = []
    for s in source:
        resp = json.loads(s["messages"][2]["content"])
        bbox = resp.get("bbox")
        obj = resp.get("object", "")
        img = s["images"][0]
        if bbox and obj:
            pt = bbox_to_point(bbox)
            points.append(make_point_sft(obj, pt, img, s.get("_source", "derived")))
    return points


def deduplicate(samples):
    """Remove duplicates by (image_path, bbox/point)"""
    seen = set()
    deduped = []
    for s in samples:
        img = s["images"][0]
        resp = s["messages"][2]["content"]
        key = f"{img}|{resp}"
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f" bbox 1M 数据整合")
    print(f" {datetime.now().isoformat()}")
    print(f"{'='*60}")

    # Load all sources
    print("\n[1/6] VG G_grounding (787K)...")
    vg_bbox = load_vg_grounding()
    print(f"  Loaded: {len(vg_bbox):,}")

    print("\n[2/6] VG S_spatial split (509K -> ~1M)...")
    spatial_bbox = load_vg_spatial_split()
    print(f"  Loaded: {len(spatial_bbox):,}")

    print("\n[3/6] Flywheel verified...")
    fw_verified = load_flywheel_verified()
    print(f"  Loaded: {len(fw_verified):,}")

    print("\n[4/6] Flywheel labels (conf >= 0.5)...")
    fw_labels = load_flywheel_labels()
    print(f"  Loaded: {len(fw_labels):,}")

    # Combine all bbox samples
    all_bbox = vg_bbox + spatial_bbox + fw_verified + fw_labels
    print(f"\n  Total bbox+affordance raw: {len(all_bbox):,}")

    # Deduplicate
    print("\n[5/6] Deduplication...")
    all_bbox = deduplicate(all_bbox)
    print(f"  After dedup: {len(all_bbox):,}")

    # Split by type
    bbox_only = [s for s in all_bbox if s.get("_type") == "bbox"]
    affordance = [s for s in all_bbox if s.get("_type") == "affordance"]
    points_existing = [s for s in all_bbox if s.get("_type") == "point"]

    print(f"  bbox: {len(bbox_only):,}")
    print(f"  affordance: {len(affordance):,}")
    print(f"  point (existing): {len(points_existing):,}")

    # Generate additional points from bbox centers
    print("\n[6/6] Generating point samples from bbox centers...")
    target_points = 200000 - len(points_existing)
    if target_points > 0:
        extra_points = generate_points_from_bbox(bbox_only, target_points)
        points_existing.extend(extra_points)
    print(f"  Total points: {len(points_existing):,}")

    # Sample to target: bbox 600K, point 200K, affordance 200K
    print(f"\n{'='*60}")
    print(f" Sampling to target 1M...")

    target_bbox = 600000
    target_point = 200000
    target_afford = 200000

    # Sample bbox
    if len(bbox_only) > target_bbox:
        final_bbox = random.sample(bbox_only, target_bbox)
    else:
        final_bbox = bbox_only
        print(f"  [INFO] bbox only {len(bbox_only):,}, using all + repeat")
        while len(final_bbox) < target_bbox:
            final_bbox.extend(random.sample(bbox_only, min(len(bbox_only), target_bbox - len(final_bbox))))

    # Sample points
    if len(points_existing) > target_point:
        final_point = random.sample(points_existing, target_point)
    else:
        final_point = points_existing
        while len(final_point) < target_point:
            final_point.extend(random.sample(points_existing, min(len(points_existing), target_point - len(final_point))))

    # Sample affordance
    if len(affordance) > target_afford:
        final_afford = random.sample(affordance, target_afford)
    else:
        final_afford = affordance
        # Supplement from bbox with action_type added
        while len(final_afford) < target_afford:
            supplement = random.sample(bbox_only, min(len(bbox_only), target_afford - len(final_afford)))
            for s in supplement:
                s_copy = json.loads(json.dumps(s))
                s_copy["_type"] = "affordance"
                resp = json.loads(s_copy["messages"][2]["content"])
                resp["action_type"] = "pick"
                s_copy["messages"][2]["content"] = json.dumps(resp, ensure_ascii=False)
                s_copy["messages"][1]["content"] = f"<image>\nHow to pick the '{resp['object']}'? Provide bbox and action in [0,999]."
                final_afford.append(s_copy)

    # Combine and shuffle
    all_final = final_bbox + final_point + final_afford
    random.shuffle(all_final)

    # Remove internal fields before writing
    for s in all_final:
        s.pop("_type", None)
        s.pop("_source", None)

    # Write SFT
    print(f"\n Writing {len(all_final):,} SFT samples...")
    sft_path = OUTPUT_DIR / "sft_train.jsonl"
    with open(sft_path, "w") as f:
        for s in all_final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write GRPO (25% subset)
    grpo_samples = random.sample(all_final, len(all_final) // 4)
    grpo_path = OUTPUT_DIR / "grpo_train.jsonl"
    with open(grpo_path, "w") as f:
        for s in grpo_samples:
            grpo_s = {"messages": s["messages"][:2], "images": s["images"]}
            f.write(json.dumps(grpo_s, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "created_at": datetime.now().isoformat(),
        "total_sft": len(all_final),
        "total_grpo": len(grpo_samples),
        "composition": {
            "bbox": len(final_bbox),
            "point": len(final_point),
            "affordance": len(final_afford),
        },
        "sources": {
            "vg_grounding": len(vg_bbox),
            "vg_spatial_split": len(spatial_bbox),
            "flywheel_verified": len(fw_verified),
            "flywheel_labels": len(fw_labels),
        },
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f" Done!")
    print(f" SFT:  {sft_path} ({len(all_final):,})")
    print(f" GRPO: {grpo_path} ({len(grpo_samples):,})")
    print(f" Composition: bbox={len(final_bbox):,} point={len(final_point):,} afford={len(final_afford):,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
