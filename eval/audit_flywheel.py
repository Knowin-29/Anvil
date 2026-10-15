#!/usr/bin/env python3
"""Geometric diagnostics of flywheel output. Drives prompt/validator iteration.

Measures, per form, concrete defect rates that a human would otherwise have to
eyeball:
  - background-ish boxes (huge area or full-width/full-height strips)
  - label vocabulary quality (generic / non-object / duplicate-label detections)
  - geometric sanity (contact point inside part, grasp inside object, axis diversity)
  - polygon quality (vertex count, self-degenerate to bbox, area vs its own hull bbox)
  - embodied relevance (label against a non-embodied stop-list)
Prints defects per record and orientation diagnostics, not semantic accuracy or
a human audit pass rate. Different prompt/validator versions must be compared explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path

# Labels that are scene/background, not manipulable objects.
# Kept in sync with trains/flywheel/validators.py:BACKGROUND.
BACKGROUND = {
    "wall", "floor", "ceiling", "background", "room", "table surface", "tabletop",
    "counter", "countertop", "surface", "sky", "ground", "wall panel", "cabinet door",
    "shadow", "light", "window", "door frame", "sink basin", "backsplash", "tile",
    "washing machine", "dryer", "machine", "appliance", "cabinet",
    "table", "desk", "shelf", "drawer front", "wood surface", "carpet", "rug",
    "wooden table", "white wall", "kitchen counter",
}
GENERIC = {"object", "object_class", "point_1", "item", "thing", "label", "unknown",
           "main manipulable object", "single most prominent object", "part"}


def area(b) -> float:
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1])) / 1e6


def is_stripe(b) -> bool:
    """Full-width or full-height band => almost certainly background."""
    w, h = b[2] - b[0], b[3] - b[1]
    return (w >= 900 and h >= 250) or (h >= 900 and w >= 250)


def is_diagonal_axis(axis) -> bool:
    """Describe near-45-degree geometry without inferring semantic invalidity."""
    if not isinstance(axis, (list, tuple)) or len(axis) != 2:
        return False
    try:
        norm = math.hypot(float(axis[0]), float(axis[1]))
        if norm <= 1e-6:
            return False
        return all(abs(abs(float(value) / norm) - 2 ** -0.5) < 0.018 for value in axis)
    except (TypeError, ValueError):
        return False


def audit(data_dir: Path) -> dict:
    res = {}
    for jf in sorted((data_dir / "raw").glob("*.jsonl")):
        form = jf.stem
        n = 0
        defects = Counter()
        diagnostics = Counter()
        versions = Counter()
        axes = []
        invalid_json = 0
        records_with_defects = 0
        image_ids = set()
        input_bytes = jf.read_bytes()
        for ln in input_bytes.decode("utf-8", errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except Exception:
                invalid_json += 1
                continue
            n += 1
            before = sum(defects.values())
            image_ids.add(str(d.get("image_sha256") or d.get("image_path")))
            versions[str(d.get("label_version", "unknown"))] += 1
            items = d.get("payload")
            items = items if isinstance(items, list) else [items]
            for it in items:
                if not isinstance(it, dict):
                    defects["not_dict"] += 1
                    continue
                lab = str(it.get("label") or it.get("object_class") or "").strip().lower()
                b = it.get("bbox_2d") or it.get("bbox")
                if lab in GENERIC:
                    defects["generic_label"] += 1
                if lab in BACKGROUND:
                    defects["background_label"] += 1
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    if area(b) > 0.55:
                        defects["huge_box"] += 1
                    if is_stripe(b):
                        defects["stripe_box"] += 1
                poly = it.get("polygon")
                if isinstance(poly, list) and poly and isinstance(poly[0], (list, tuple)):
                    # 6 is the enforced floor in validators.v_mask; below that is a defect.
                    if len(poly) < 6:
                        defects["polygon_lt6_vertices"] += 1
                    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                    hull = (min(xs), min(ys), max(xs), max(ys))
                    # polygon that is basically its own bbox (axis-aligned rectangle)
                    if len(poly) <= 5 and area(hull) > 0.0:
                        defects["polygon_rectangular"] += 1
                ax = it.get("jaw_axis_2d")
                if isinstance(ax, (list, tuple)) and len(ax) == 2:
                    if is_diagonal_axis(ax):
                        diagnostics["axis_near_diagonal"] += 1
                    m = math.hypot(ax[0], ax[1])
                    if not math.isfinite(m):
                        defects["nonfinite_axis"] += 1
                    elif m < 1e-6:
                        defects["zero_axis"] += 1
                    else:
                        if not .85 <= m <= 1.15:
                            defects["axis_not_unit"] += 1
                        if min(abs(value / m) for value in ax) < .05:
                            diagnostics["axis_near_cardinal"] += 1
                        axes.append(round(math.degrees(math.atan2(ax[1], ax[0])) % 180, 1))
                cp = it.get("contact_point_2d")
                if (isinstance(cp, (list, tuple)) and len(cp) == 2
                        and isinstance(b, (list, tuple)) and len(b) == 4):
                    if not (b[0] <= cp[0] <= b[2] and b[1] <= cp[1] <= b[3]):
                        defects["contact_outside_box"] += 1
            # mask must segment the SAME object the cascade grounded, and must not be
            # a rectangle (otherwise it adds nothing over the bbox form).
            if form == "mask":
                ob = (d.get("g_context") or {}).get("bbox")
                it0 = items[0] if items and isinstance(items[0], dict) else {}
                poly = it0.get("polygon")
                if isinstance(poly, list) and poly and isinstance(poly[0], (list, tuple)):
                    if isinstance(ob, (list, tuple)) and len(ob) == 4:
                        out = sum(1 for x, y in poly
                                  if not (ob[0] - 40 <= x <= ob[2] + 40
                                          and ob[1] - 40 <= y <= ob[3] + 40))
                        if out > max(1, len(poly) // 8):
                            defects["mask_unaligned_with_anchor"] += 1
                    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                    hull = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    nn = len(poly)
                    sh = abs(sum(xs[i] * ys[(i + 1) % nn] - xs[(i + 1) % nn] * ys[i]
                                 for i in range(nn))) / 2.0
                    if hull > 0 and sh / hull > 0.97:
                        defects["mask_is_rectangle"] += 1

            # detection-specific: all boxes same label => low information
            if form == "detection":
                labs = [str(x.get("label", "")).lower() for x in items if isinstance(x, dict)]
                if len(labs) >= 2 and len(set(labs)) == 1:
                    defects["detection_monolabel"] += 1
            records_with_defects += int(sum(defects.values()) > before)
        entry = {"n": n, "defects": dict(defects.most_common()),
                 "diagnostics": dict(diagnostics), "label_versions": dict(versions),
                 "semantic_accuracy_measured": False}
        entry.update({"records_with_defects": records_with_defects,
                      "invalid_json": invalid_json,
                      "unique_images": len(image_ids),
                      "input_sha256": hashlib.sha256(input_bytes).hexdigest()})
        if axes:
            uniq = len(set(axes))
            entry["axis_unique_frac"] = round(uniq / len(axes), 3)
        # Multiple defects can occur in one record; this is not a rejection rate.
        entry["defects_per_record"] = round(sum(defects.values()) / max(n, 1), 3)
        res[form] = entry
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/flywheel_mf")
    ap.add_argument("--out", default="eval/flywheel_audit.json")
    ap.add_argument("--with-provenance", action="store_true",
                    help="wrap per-form findings with measured CPU audit runtime")
    a = ap.parse_args()
    started = time.perf_counter()
    r = audit(Path(a.data))
    seconds = time.perf_counter() - started
    result = r
    if a.with_provenance:
        result = {"schema": "anvil-geometric-recheck-efficiency-v1", "forms": r,
                  "data": str(Path(a.data).resolve()), "cpu_audit_seconds": seconds,
                  "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "scope": "Selected retained outputs; listed geometric checks only. Not semantic accuracy or generation yield."}
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"{'form':<12}{'n':>7}{'defects/rec':>13}  top defects")
    for form, e in sorted(r.items()):
        top = ", ".join(f"{k}={v}" for k, v in list(e["defects"].items())[:3])
        extra = f"  axis_uniq={e['axis_unique_frac']}" if "axis_unique_frac" in e else ""
        print(f"{form:<12}{e['n']:>7}{e['defects_per_record']:>13}  {top}{extra}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
