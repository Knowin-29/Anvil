#!/usr/bin/env python3
"""Build a desensitized source registry for the ICLR (open-data-only) rebuild.

Input : Rex-Embodied-anvil_dev/data/sources_registry_local.yaml (48 sources).
Output: GSA_Forge/configs/data/sources_clean.yaml.

Desensitization rules (reproducible, auditable):
  1. DROP any source whose license is genuinely proprietary/internal:
     the desktop_bbox_*, *_pointctl, proxy_log_* families and the old
     flywheel export (gsa_volc_flywheel_sft) -- these are business data.
  2. RELABEL the refspatial_* family: they were mislabeled `internal` but are
     the public RoboRefer / RefSpatial release -> `open (RefSpatial/RoboRefer)`.
  3. KEEP every `open (...)` source.
  4. Verify each kept source's jsonl exists and (sampled) images resolve after
     the /mnt/data -> /path/to/datasets remap.

A source is only written to the clean registry if its jsonl is present on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("GSA_SOURCE_REGISTRY", "/path/to/Anvil/data/sources_registry_local.yaml"))
OUT = REPO_ROOT / "configs/data/sources_clean.yaml"

# Genuinely proprietary source names to drop (business / proxy-log / desktop capture).
DROP_NAMES = {
    "desktop_bbox_train", "desktop_bbox_val",
    "point_spatial_15k_pointctl", "point_vsi_vlm_pointctl", "point_rootdata_pointctl",
    "proxy_log_sqlite", "gsa_volc_flywheel_sft",
}
# refspatial family: mislabeled internal, actually public RoboRefer/RefSpatial.
REFSPATIAL_PREFIX = "refspatial_"
REFSPATIAL_LICENSE = "open (RefSpatial/RoboRefer)"

PATH_REMAP = [("/mnt/data/datasets/", "/path/to/datasets/datasets/"),
              ("/mnt/data/", "/path/to/datasets/"),
              ("/path/to/storage/", "/path/to/workspace/foundation/")]


def remap(p: str) -> str:
    for a, b in PATH_REMAP:
        if p.startswith(a):
            return b + p[len(a):]
    return p


def sample_image_hit_rate(jsonl: Path, n: int = 20) -> tuple[int, int]:
    """Sample up to n records, remap image paths, return (hits, checked)."""
    hits = checked = 0
    try:
        with jsonl.open() as f:
            lines = [next(f) for _ in range(min(200, 10_000)) if True]  # noqa
    except StopIteration:
        lines = []
    except Exception:
        return 0, 0
    if not lines:
        try:
            lines = jsonl.read_text(errors="ignore").splitlines()[:200]
        except Exception:
            return 0, 0
    random.seed(0)
    random.shuffle(lines)
    for ln in lines:
        if checked >= n:
            break
        try:
            d = json.loads(ln)
        except Exception:
            continue
        imgs = d.get("images") or d.get("image") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        if not imgs:
            continue
        checked += 1
        if os.path.exists(remap(str(imgs[0]))):
            hits += 1
    return hits, checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-images", action="store_true", help="sample-check image paths")
    args = ap.parse_args()

    reg = yaml.safe_load(SRC.read_text())
    sources = reg.get("sources", reg) if isinstance(reg, dict) else reg

    kept, dropped, relabeled, missing = [], [], [], []
    for s in sources:
        name = s.get("name", "")
        lic = str(s.get("license", ""))
        if name in DROP_NAMES or lic == "internal" and not name.startswith(REFSPATIAL_PREFIX):
            dropped.append(name)
            continue
        if name.startswith(REFSPATIAL_PREFIX):
            s = dict(s)
            s["license"] = REFSPATIAL_LICENSE
            relabeled.append(name)
        # remap the source jsonl path itself
        s = dict(s)
        s["path"] = remap(str(s.get("path", "")))
        if not os.path.isfile(s["path"]):
            missing.append((name, s["path"]))
            continue
        s["image_path_remap"] = "/mnt/data->/path/to/datasets"
        kept.append(s)

    # ---- additional open sources found on disk but absent from the v0 registry ----
    _D = "/path/to/datasets/datasets/dataset_jsons_qwen3_swift"
    EXTRA = [
        dict(name="visual_genome", path=f"{_D}/visual_genome/vg_grounding_swift.jsonl",
             dimensions=["G_grounding"], est_samples=5408476, ratio=1.0,
             scene_type="indoor_general", license="open (Visual Genome, CC BY 4.0)",
             notes="bbox grounding; image root re-rooted to robopoint-data mirror"),
        dict(name="visual_genome_cleaned", path=f"{_D}/visual_genome_cleaned/vg_2src_full.jsonl",
             dimensions=["G_grounding"], est_samples=1053390, ratio=1.0,
             scene_type="indoor_general", license="open (Visual Genome, CC BY 4.0)"),
        dict(name="gs_spatial_qa", path=f"{_D}/gs_spatial_qa/gs_spatial_qa_manifest.jsonl",
             dimensions=["S_spatial"], est_samples=539700, ratio=1.0,
             scene_type="indoor_general", license="open (GS spatial QA)"),
        dict(name="vsi_590k", path=f"{_D}/VSI_590K/vsi_590k_msswift.jsonl",
             dimensions=["S_spatial"], est_samples=45888, ratio=1.0,
             scene_type="3d_spatial", license="open (VSI-590K)"),
        dict(name="mimic_cgd", path=f"{_D}/mimic_cgd/mimic_cgd_qwen_to_swift.jsonl",
             dimensions=["S_spatial"], est_samples=70939, ratio=0.5,
             scene_type="indoor_general", license="open (MIMIC-CGD)"),
        # pixmo_points ships a second, larger "labelpoint" variant (labelled points)
        dict(name="pixmo_points_label",
             path=f"{_D}/pixmo_points/point_points_both_to_swift_labelpoint.jsonl",
             dimensions=["G_grounding"], est_samples=2540990, ratio=1.0,
             scene_type="mixed", license="open (Pixmo Points)"),
        # larger re-splits of the tabletop manipulation sets (more rows than the
        # variants already registered)
        dict(name="robopoint_ref_40k",
             path=f"{_D}/robopoint_rft_point_ref_train_40000_test_2000_0417/"
                  "data_train_internvl_no_think_to_swift_onlypoint.jsonl",
             dimensions=["A_action_interface", "G_grounding"], est_samples=160000,
             ratio=1.0, scene_type="tabletop_manipulation", license="open (RoboPoint)"),
        dict(name="roborefit_rec_105k",
             path=f"{_D}/roborefit_rft_point_rec_train_35000_test_1000_0502/"
                  "data_train_internvl_no_think_to_swift_onlypoint.jsonl",
             dimensions=["A_action_interface", "G_grounding"], est_samples=105000,
             ratio=1.0, scene_type="tabletop_manipulation", license="open (RoboRefit)"),
        dict(name="fsd_free_point_98k",
             path=f"{_D}/FSD_points_rft_fsd_free_point_train_32790_test_300_0425/"
                  "data_train_internvl_no_think_to_swift_onlypoint.jsonl",
             dimensions=["A_action_interface"], est_samples=98370, ratio=1.0,
             scene_type="tabletop_manipulation", license="open (FSD)"),
    ]
    for e in EXTRA:
        if os.path.isfile(e["path"]) and not any(k["name"] == e["name"] for k in kept):
            e["available_local"] = True
            e["image_path_remap"] = "auto"
            kept.append(e)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_doc = {
        "meta": {
            "built_by": "trains/build_clean_registry.py",
            "source_registry": str(SRC),
            "policy": "open-license public data only; internal/business sources removed",
            "n_kept": len(kept), "n_dropped": len(dropped),
            "n_relabeled": len(relabeled), "n_missing_jsonl": len(missing),
        },
        "image_roots": {k: remap(v) for k, v in (reg.get("image_roots", {}) or {}).items()},
        "sources": kept,
    }
    OUT.write_text(yaml.safe_dump(out_doc, allow_unicode=True, sort_keys=False))

    print(f"kept={len(kept)} dropped={len(dropped)} relabeled={len(relabeled)} missing_jsonl={len(missing)}")
    print(f"DROPPED (internal/business): {sorted(dropped)}")
    if missing:
        print(f"MISSING jsonl (excluded): {missing}")
    if args.verify_images:
        print("\n--- sampled image hit-rate per kept source ---")
        total_ok = total_ck = 0
        for s in kept:
            h, c = sample_image_hit_rate(Path(s["path"]))
            total_ok += h; total_ck += c
            flag = "" if (c == 0 or h == c) else "  <-- CHECK"
            print(f"  {s['name']:<36} {h}/{c}{flag}")
        print(f"overall image hit-rate: {total_ok}/{total_ck}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
