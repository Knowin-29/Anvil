#!/usr/bin/env python3
"""Re-score recorded gsa_bench_test predictions for BOTH models with one scorer.

Exists so a parsing/scoring change is always evaluated symmetrically. A fix that
raises one model and is never checked against the other is indistinguishable from
loosening the scorer in that model's favour, so this replays the stored dumps for
every tag in one run and prints them side by side.

No inference. Reads save/logs/preds_<tag>_gsa_bench_test.jsonl.

    python eval/rescore_gsa_test.py
    python eval/rescore_gsa_test.py --tags base sft_9b_lora --out /tmp/r.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gsa_eval.parsing import parse_boxes, parse_points  # noqa: E402
from eval.gsa_eval.registry import get_spec  # noqa: E402
from eval.gsa_eval.scorers import score_gsa_geometry  # noqa: E402

BENCH = "gsa_bench_test"


def load_dump(tag: str) -> dict[str, str]:
    p = ROOT / "save" / "logs" / f"preds_{tag}_{BENCH}.jsonl"
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["uid"])] = row.get("response") or ""
    return out


def form_compliance(samples, texts) -> dict:
    """Split failures into 'wrong geometric form' vs 'no answer at all'.

    The bbox-group prompts are `Locate "X" in the image.` and name no output form
    (measured 378/378 rows), so a point answer is a prompt-contract gap, not an
    unparseable response. Keeping the two apart stops a prompt defect from being read
    as a model failure.
    """
    out: dict[str, dict[str, int]] = {}
    for s, t in zip(samples, texts):
        want_box = s.meta.get("form") == "bbox"
        got_box, got_pt = bool(parse_boxes(t)), bool(parse_points(t))
        want, other = (got_box, got_pt) if want_box else (got_pt, got_box)
        key = "right_form" if want else ("wrong_form" if other else "no_answer")
        out.setdefault(s.group, {}).setdefault(key, 0)
        out[s.group][key] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["base", "sft_9b_lora"])
    ap.add_argument("--out",
                    default=str(ROOT / "eval_results" / "rescore_gsa_test.json"))
    a = ap.parse_args()

    spec = get_spec(BENCH)
    by_uid = {str(s.uid): s for s in spec.loader(ROOT / ".cache" / "eval_assets")}
    dumps = {t: load_dump(t) for t in a.tags}
    empty = [t for t, d in dumps.items() if not d]
    if empty:
        print(f"no dump for {empty}", file=sys.stderr)
        return 2
    common = sorted(set.intersection(*(set(d) for d in dumps.values())) & set(by_uid))
    print(f"{len(common)} uids common to {a.tags}")

    report: dict = {"benchmark": BENCH, "tags": a.tags, "n_common": len(common),
                    "per_model": {}}
    samples = [by_uid[u] for u in common]
    for t in a.tags:
        texts = [dumps[t][u] for u in common]
        score, detail, fails = score_gsa_geometry(samples, texts)
        report["per_model"][t] = {
            "score": score,
            "parse_failures": fails,
            "parse_failure_rate": round(100.0 * fails / len(common), 2),
            "bbox": detail.get("bbox"),
            "point": detail.get("point"),
            "coord_scale": detail.get("coord_scale"),
            "form_compliance": form_compliance(samples, texts),
        }
    if len(a.tags) == 2:
        x, y = a.tags
        report["gap"] = round(report["per_model"][y]["score"]
                              - report["per_model"][x]["score"], 2)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
