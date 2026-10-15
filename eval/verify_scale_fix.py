#!/usr/bin/env python3
"""Prove the prompt-driven scale rule is a narrowing, not a relaxation.

Audit defects F1/F2 said `score_point_in_mask` admitted a third `pixel` reading, chosen
whenever the coordinates happened not to overflow the image bounds. The fix restricts the
reading to the two conventions the prompt can plausibly elicit. Three checks, all on real
masks, no inference:

1. **Synthetic ground-truth injection.** A prediction placed at the mask's centroid must
   score ~100 whether it is written in [0,1] or in thousandths, and a prediction placed
   outside the mask must score ~0. If the fix had relaxed anything, the deliberately wrong
   answer would also gain.
2. **Pixel-branch reachability.** Under the fix no sample may be read as `pixel` on these
   benchmarks, because the prompt states [0,1].
3. **The specific rows the audit named.** refspatial_bench `location_15` and friends were
   scored 0.00 by `auto` and 1.00 read as thousandth; they must now score 1.00.

    python eval/verify_scale_fix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gsa_eval.registry import get_spec  # noqa: E402
from eval.gsa_eval.scorers import _load_mask, score_point_in_mask  # noqa: E402

SCRATCH = ROOT / ".cache" / "eval_assets"


def centroid_and_outside(mask):
    """A point inside the mask and one provably outside it, both in [0,1]."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None, None
    mh, mw = mask.shape
    # Densest row/col rather than the raw centroid: a C-shaped mask's centroid can fall
    # in the hole, which would make a correct scorer look broken.
    y = int(np.bincount(ys).argmax())
    row = np.nonzero(mask[y])[0]
    x = int(row[len(row) // 2])
    inside = (x / (mw - 1), y / (mh - 1))
    outside = None
    for cy in range(mh):
        free = np.nonzero(~mask[cy])[0]
        if len(free):
            outside = (int(free[len(free) // 2]) / (mw - 1), cy / (mh - 1))
            break
    return inside, outside


def as_text(pt, scale):
    x, y = pt
    if scale == "unit":
        return f"[({x:.4f}, {y:.4f})]"
    return f"[({round(x * 1000)}, {round(y * 1000)})]"


def check_injection(bench: str) -> dict:
    spec = get_spec(bench)
    samples = spec.loader(SCRATCH)
    keep, ins, outs = [], [], []
    for s in samples:
        try:
            m = _load_mask(s.gt)
        except Exception:
            continue
        i, o = centroid_and_outside(m)
        if i is None or o is None:
            continue
        keep.append(s)
        ins.append(i)
        outs.append(o)
    res = {"benchmark": bench, "n": len(keep)}
    for scale in ("unit", "thousandth"):
        sc_in, d_in, f_in = score_point_in_mask(
            keep, [as_text(p, scale) for p in ins], spec=spec)
        sc_out, _, _ = score_point_in_mask(
            keep, [as_text(p, scale) for p in outs], spec=spec)
        res[f"perfect_as_{scale}"] = sc_in
        res[f"wrong_as_{scale}"] = sc_out
        res[f"scale_used_{scale}"] = d_in["coord_scale"]["used"]
        res[f"parse_failures_{scale}"] = f_in
    return res


def check_pixel_unreachable(bench: str, tags=("base", "sft_9b_lora")) -> dict:
    spec = get_spec(bench)
    by_uid = {str(s.uid): s for s in spec.loader(SCRATCH)}
    out = {}
    for tag in tags:
        p = ROOT / "save" / "logs" / f"preds_{tag}_{bench}.jsonl"
        if not p.is_file():
            out[tag] = "no dump"
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if str(r["uid"]) in by_uid]
        samples = [by_uid[str(r["uid"])] for r in rows]
        texts = [r.get("response") or "" for r in rows]
        _, d, _ = score_point_in_mask(samples, texts, spec=spec)
        used = d["coord_scale"]["used"]
        out[tag] = {"used": used, "pixel_rows": used.get("pixel", 0),
                    "auto_would_have_said": d["coord_scale"]["auto_would_have_said"],
                    "n_reinterpreted": d["coord_scale"]["n_reinterpreted_vs_auto"]}
    return out


def check_named_rows() -> dict:
    """The exact refspatial_bench uids the audit reported as 'auto 0.00 / thousandth 1.00'."""
    named = ["location_15", "location_9", "location_22", "location_24", "location_26",
             "location_30", "location_50", "unseen_16", "unseen_35", "unseen_38"]
    spec = get_spec("refspatial_bench")
    by_uid = {str(s.uid): s for s in spec.loader(SCRATCH)}
    out = {}
    for uid in named:
        s = by_uid.get(uid)
        if s is None:
            out[uid] = "uid absent"
            continue
        try:
            m = _load_mask(s.gt)
        except Exception:
            out[uid] = "mask unreadable"
            continue
        i, _ = centroid_and_outside(m)
        if i is None:
            out[uid] = "empty mask"
            continue
        # A thousandth-scale answer at a true in-mask location: exactly the shape the
        # audit said `auto` was scoring 0.00 on these rows.
        sc, d, _ = score_point_in_mask([s], [as_text(i, "thousandth")], spec=spec)
        out[uid] = {"score_thousandth_answer": sc,
                    "scale_used": d["coord_scale"]["used"]}
    return out


def main() -> int:
    report = {"injection": {}, "pixel_unreachable": {}, "named_rows": {}}
    ok = True
    for b in ("where2place", "refspatial_bench"):
        r = check_injection(b)
        report["injection"][b] = r
        print(json.dumps(r, indent=2), flush=True)
        # A perfect answer must score ~100 in BOTH conventions, and a wrong one ~0.
        for scale in ("unit", "thousandth"):
            if r[f"perfect_as_{scale}"] < 99.0:
                print(f"FAIL {b}: perfect {scale} answer scored "
                      f"{r[f'perfect_as_{scale}']}, expected ~100")
                ok = False
            if r[f"wrong_as_{scale}"] > 5.0:
                print(f"FAIL {b}: deliberately wrong {scale} answer scored "
                      f"{r[f'wrong_as_{scale}']}, expected ~0 -- the rule has been "
                      f"relaxed, not narrowed")
                ok = False
        report["pixel_unreachable"][b] = check_pixel_unreachable(b)
        for tag, v in report["pixel_unreachable"][b].items():
            if isinstance(v, dict) and v["pixel_rows"]:
                print(f"FAIL {b}/{tag}: {v['pixel_rows']} rows still read as pixel")
                ok = False
    print(json.dumps(report["pixel_unreachable"], indent=2), flush=True)

    report["named_rows"] = check_named_rows()
    print(json.dumps(report["named_rows"], indent=2), flush=True)
    for uid, v in report["named_rows"].items():
        if isinstance(v, dict) and v["score_thousandth_answer"] < 99.0:
            print(f"FAIL {uid}: thousandth answer at an in-mask point scored "
                  f"{v['score_thousandth_answer']}")
            ok = False

    report["verdict"] = "PASS" if ok else "FAIL"
    out = ROOT / "eval_results" / "verify_scale_fix.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{report['verdict']}  -> wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
