#!/usr/bin/env python3
"""Land one sweep row (or rebuild the whole table) from eval_results/ + output/.

Every number here is read back off disk. Nothing is carried in memory between
configurations, so a dead shell costs at most the configuration that was running.

Missing cells are emitted as null, never 0: a 0 would read as "the model scored
nothing" when it means "not measured yet".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "artifacts" / "sweep_v1" / "sweep_table.json"
BENCHES = ("refspatial_bench", "cv_bench", "gsa_bench_test")

# Frozen references, read from the recorded runs rather than retyped:
#   base            eval_results/qwen3_5_9b_base_full
#   G-only seed42   eval_results/g_family_v3_seed42_ckpt53_full_v7_diagnostic_local4_20260822
BASE_TAG = "qwen3_5_9b_base_full"
GATE_TAG = "g_family_v3_seed42_ckpt53_full_v7_diagnostic_local4_20260822"
# The gate the sweep has to clear, as stated in the run brief.
GATE_DELTA = {"refspatial_location": 3.33, "gsa_bench_test": 2.81}


def read_cell(tag: str, bench: str) -> dict:
    p = ROOT / "eval_results" / tag / f"{bench}.json"
    if not p.is_file():
        return {"present": False}
    d = json.loads(p.read_text())
    det = d.get("detail") or {}
    pg = det.get("per_group") or {}
    row = {
        "present": True,
        "score": d.get("score"),
        "score_if_untrusted": d.get("score_if_untrusted"),
        "trusted": bool(d.get("parsed")),
        "n_evaluated": d.get("n_evaluated"),
        "parse_failure_rate": d.get("parse_failure_rate"),
        "truncation_rate": d.get("truncation_rate"),
        "schema_version": d.get("schema_version"),
        "protocol": d.get("evaluation_protocol"),
        "thinking": d.get("thinking"),
        "subset_limit": d.get("subset_limit"),
        "elapsed_s": d.get("elapsed_s"),
    }
    if bench == "refspatial_bench":
        row["per_split"] = {k: v.get("mean_in_mask") for k, v in pg.items()}
        row["location"] = (pg.get("location") or {}).get("mean_in_mask")
        row["points_per_question"] = det.get("points_per_question")
        row["coord_scale_used"] = (det.get("coord_scale") or {}).get("used")
    return row


def train_meta(cfg: str) -> dict:
    d = ROOT / "output" / "sweep_v1" / cfg
    out = {}
    for name in ("sweep_config.json", "train_status.json"):
        p = d / name
        if p.is_file():
            out[name.replace(".json", "")] = json.loads(p.read_text())
    ck = sorted(d.glob("**/checkpoint-*"))
    out["checkpoints"] = [str(c) for c in ck]
    return out


def collect(tags: list[str]) -> dict:
    rows = {}
    for tag in tags:
        cfg = tag[len("sweep_"):] if tag.startswith("sweep_") else tag
        rows[tag] = {"cfg": cfg,
                     "cells": {b: read_cell(tag, b) for b in BENCHES},
                     "train": train_meta(cfg)}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", default=[],
                    help="add/refresh these tags; default refreshes every known tag")
    a = ap.parse_args()

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(TABLE.read_text()) if TABLE.is_file() else {}
    known = set((prev.get("configs") or {}).keys())
    known.update(a.tag)
    if not a.tag:
        known.update(p.name for p in (ROOT / "eval_results").glob("sweep_*"))

    base = {b: read_cell(BASE_TAG, b) for b in BENCHES}
    gate = {b: read_cell(GATE_TAG, b) for b in BENCHES}
    payload = {
        "references": {
            "base": {"tag": BASE_TAG, "cells": base},
            "g_only_seed42": {"tag": GATE_TAG, "cells": gate},
            "gate_delta_required": GATE_DELTA,
        },
        "benchmarks": list(BENCHES),
        "configs": collect(sorted(known)),
        "missing_value_policy": "null means not measured; never substituted with 0",
    }
    tmp = TABLE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(TABLE)

    bl = base["refspatial_bench"].get("location")
    bt = base["gsa_bench_test"].get("score")
    bc = base["cv_bench"].get("score")
    print(f"base: refspatial_loc={bl} cv_bench={bc} internal_test={bt}")
    hdr = f"{'cfg':28} {'refLoc':>8} {'dLoc':>7} {'cv':>7} {'test':>7} {'dTest':>7} {'pf%':>6} {'tr%':>6}"
    print(hdr)
    for tag, row in payload["configs"].items():
        rs = row["cells"]["refspatial_bench"]
        cv = row["cells"]["cv_bench"]
        gt = row["cells"]["gsa_bench_test"]
        loc = rs.get("location")
        tst = gt.get("score")
        dl = f"{loc - bl:+.2f}" if (loc is not None and bl is not None) else "--"
        dt = f"{tst - bt:+.2f}" if (tst is not None and bt is not None) else "--"
        fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "--"
        pf = max((c.get("parse_failure_rate") or 0) for c in row["cells"].values()) \
            if any(c.get("present") for c in row["cells"].values()) else None
        tr = max((c.get("truncation_rate") or 0) for c in row["cells"].values()) \
            if any(c.get("present") for c in row["cells"].values()) else None
        print(f"{row['cfg']:28} {fmt(loc):>8} {dl:>7} {fmt(cv.get('score')):>7} "
              f"{fmt(tst):>7} {dt:>7} {fmt(pf):>6} {fmt(tr):>6}")
    print(f"wrote {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
