#!/usr/bin/env python3
"""Render admitted General Retention v3 scores into a small LaTeX macro file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import general_retention_formal_v3 as formal


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verified_values(paired_path: Path) -> dict[str, float]:
    paired_path = paired_path.expanduser().resolve()
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    identity = paired.get("identity_sha256")
    unsigned = {key: value for key, value in paired.items() if key != "identity_sha256"}
    if identity != _canonical_hash(unsigned):
        raise formal.FormalProtocolError("paired result identity drift")
    if paired.get("protocol") != formal.PROTOCOL:
        raise formal.FormalProtocolError("paired result protocol drift")
    if paired.get("status") != "PAIRED_GENERAL_RETENTION_ADMITTED":
        raise formal.FormalProtocolError("paired result is not admitted")
    if paired.get("paired_health_admitted") is not True:
        raise formal.FormalProtocolError("paired health admission is false")

    base_path = formal.verify_file_binding(paired.get("base_result"), deep=True)
    e1_path = formal.verify_file_binding(paired.get("e1_result"), deep=True)
    base = formal._verify_result(base_path)
    e1 = formal._verify_result(e1_path)
    if base.get("status") != "FORMAL_SCORE_ADMITTED" or e1.get("status") != "FORMAL_SCORE_ADMITTED":
        raise formal.FormalProtocolError("base or E1 result is not admitted")
    if base.get("model_role") != "base" or e1.get("model_role") != "e1":
        raise formal.FormalProtocolError("paired model roles drift")

    expected = {
        "base": float(base["general_avg_percent"]),
        "e1": float(e1["general_avg_percent"]),
        "delta": float(e1["general_avg_percent"]) - float(base["general_avg_percent"]),
    }
    observed = {
        "base": float(paired["base_general_avg_percent"]),
        "e1": float(paired["e1_general_avg_percent"]),
        "delta": float(paired["raw_general_avg_delta_points"]),
    }
    if any(abs(expected[key] - observed[key]) > 1e-9 for key in expected):
        raise formal.FormalProtocolError("paired General Avg values do not match bound results")
    return observed


def _signed(value: float) -> str:
    return f"{value:+.2f}"


def render(paired_path: Path, out: Path) -> dict[str, Any]:
    values = _verified_values(paired_path)
    delta_macro = "posdelta" if values["delta"] >= 0 else "negdelta"
    text = "\n".join(
        [
            "% Generated from an admitted general-retention-formal-v3 paired result.",
            "% Do not edit these numeric macros by hand.",
            rf"\newcommand{{\BaseGeneralAvg}}{{{values['base']:.2f}}}",
            rf"\newcommand{{\EOneGeneralAvg}}{{{values['e1']:.2f}}}",
            rf"\newcommand{{\EOneGeneralDelta}}{{\{delta_macro}{{({_signed(values['delta'])})}}}}",
            "",
        ]
    )
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, out)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return {"status": "GENERAL_RETENTION_PAPER_VALUES_RENDERED", "out": str(out), **values}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.paired_result, args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (formal.FormalProtocolError, OSError, ValueError, KeyError) as exc:
        print(f"FATAL: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
