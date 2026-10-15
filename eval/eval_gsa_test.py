#!/usr/bin/env python3
"""GSA-Bench held-out test split -- thin wrapper over the unified harness.

The test labels are HUMAN ground truth from open datasets (never model output), so this
is the one number in the paper that cannot be circular.

This script used to carry its own inference and scoring. That second implementation was
audited (docs/audit_gsa_test_report.md) and had five defects the unified harness does
not, so all scoring logic has been removed and the CLI now forwards to
`eval/gsa_eval/runner.py`. Measured consequences of the old path, kept here because they
explain why the numbers changed:

  * No coordinate normalisation at all. A perfectly located prediction expressed in
    pixels scored mean_iou 0.0826 / IoU@0.5 3.17 instead of 1.0 / 100. This is the
    origin of the reported bbox mean_iou=0.0003, iou@0.5=0.0 ("105 boxes all missed").
  * `res["score"] = bbox["iou@0.5"] or point["hit@0.1"]` -- 0.0 is falsey in Python, so
    when every box missed, the point hit rate was silently reported under the name
    "score". The paper's 32.75 (base) / 22.68 (SFT) are point-only hit@0.1 over 295
    point rows, not combined scores. Reproduced exactly by the audit.
  * `<think>.*?</think>` needs a closing tag, so a truncated generation left the whole
    reasoning span in the text and decoy coordinates inside it were scored as the answer.
  * `pred[0]` vs `gt[0]` positional pairing, though the model is trained to emit
    multi-target lists (14% of training answers) whose order is meaningless.
  * No truncation / parse-failure gate, so a run with a 53% parse failure rate still
    emitted a usable-looking score.

`iou` and `parse_items` are retained below, unused by this entry point, because
eval/audit_gsa_test.py imports them to reproduce those defects.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def iou(a, b) -> float:
    """LEGACY, forensic use only: compares two boxes with no unit normalisation."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_items(text: str):
    """LEGACY, forensic use only: requires a closing </think> to strip reasoning."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    s = text.find("[")
    if s >= 0:
        depth = 0
        for i, ch in enumerate(text[s:], s):
            depth += (ch == "[") - (ch == "]")
            if depth == 0:
                try:
                    v = json.loads(text[s:i + 1])
                    return v if isinstance(v, list) else [v]
                except Exception:
                    break
    try:
        v = json.loads(text)
        return v if isinstance(v, list) else [v]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "trains/gsa_bench/test.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=400,
                    help="seeded STRATIFIED subset (the old --limit took a prefix "
                         "while its help text claimed stratified); 0 = full split")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dump-predictions",
                    help="raw responses as jsonl; the old script had no such option, "
                         "so its coordinates could not be traced back")
    a = ap.parse_args()

    from eval.gsa_eval.runner import evaluate
    scratch = ROOT / ".cache" / "eval_assets"
    scratch.mkdir(parents=True, exist_ok=True)
    res = evaluate("gsa_bench_test", a.model, limit=a.limit,
                   max_tokens=a.max_new_tokens, batch_size=a.batch_size,
                   scratch=scratch, data=a.data, device_map=a.device_map,
                   dump_predictions=(Path(a.dump_predictions)
                                     if a.dump_predictions else None))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "detail"}, indent=2,
                     ensure_ascii=False))
    print(f"wrote {out}")
    return 0 if res.get("parsed") or res.get("skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
