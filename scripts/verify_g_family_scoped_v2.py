#!/usr/bin/env python3
"""Replay and verify the v2 G-only train/dev/qualification partition."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trains.materialize_g_family_scoped_v2 import (  # noqa: E402
    PROTOCOL,
    SOURCE_PROTOCOL,
    file_hash,
    load_json,
    materialize,
)


def verify(*, manifest_path: Path, source: Path, scoped_report: Path,
           full_report: Path) -> dict:
    manifest_path = manifest_path.resolve()
    source = source.resolve()
    scoped_report = scoped_report.resolve()
    full_report = full_report.resolve()
    corpus = manifest_path.parent
    observed = load_json(manifest_path)
    if observed.get("protocol") != PROTOCOL:
        raise ValueError("v2 manifest protocol drift")
    if observed.get("source_release_protocol") != SOURCE_PROTOCOL:
        raise ValueError("v2 source protocol drift")
    if observed.get("counts") != {"train": 84, "dev": 32, "qualification": 32}:
        raise ValueError("v2 row counts drift")
    if observed.get("unique_images") != {"train": 24, "dev": 8, "qualification": 8}:
        raise ValueError("v2 image counts drift")
    if observed.get("overlap") != {
        "train_dev": 0, "train_qualification": 0, "dev_qualification": 0
    }:
        raise ValueError("v2 cross-split overlap is nonzero")
    if (observed.get("exposure_policy") or {}).get("qualification_seen_by_training_launcher") is not False:
        raise ValueError("qualification exposure policy drift")

    audit = observed.get("audit_bindings") or {}
    for name, path in (("scoped_report", scoped_report),
                       ("full_compiler_report", full_report)):
        spec = audit.get(name) or {}
        if Path(str(spec.get("path", ""))).resolve(strict=False) != path:
            raise ValueError(f"v2 audit path drift: {name}")
        if spec.get("sha256") != file_hash(path):
            raise ValueError(f"v2 audit hash drift: {name}")
    scoped = load_json(scoped_report)
    full = load_json(full_report)
    if scoped.get("protocol") != SOURCE_PROTOCOL or scoped.get("verdict") != "G_ONLY_PASS":
        raise ValueError("source scoped audit is not G_ONLY_PASS")
    if scoped.get("global_audit_verdict") != "BLOCKED" or full.get("verdict") != "BLOCKED":
        raise ValueError("global schema-v3 BLOCKED state was not preserved")

    with tempfile.TemporaryDirectory(prefix="g_family_v2_replay_") as temp:
        replay_root = Path(temp) / "corpus"
        expected = materialize(source, replay_root, seed=20260820)
        if expected != observed:
            raise ValueError("v2 manifest does not replay from the frozen v1 corpus")
        for split in ("train", "dev", "qualification"):
            name = f"{split}_G_only.jsonl"
            if (replay_root / name).read_bytes() != (corpus / name).read_bytes():
                raise ValueError(f"v2 split does not replay byte-for-byte: {name}")
    return {
        "status": "G_V2_CORPUS_READY",
        "protocol": PROTOCOL,
        "counts": observed["counts"],
        "unique_images": observed["unique_images"],
        "qualification_exposed_to_trainer": False,
        "global_audit_verdict": "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scoped-report", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(manifest_path=args.manifest, source=args.source,
                        scoped_report=args.scoped_report, full_report=args.full_report)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
