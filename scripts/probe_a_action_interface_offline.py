#!/usr/bin/env python3
"""Run the read-only A_action_interface offline probe.

The script reads candidate datasets and writes derived reports only inside this
worktree. It does not download data, call VLM services, or mutate NAS paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apps.data_flywheel.gsa.data_adapters.a_action_interface_offline_probe import (
    WORKTREE_ROOT,
    run_offline_probe,
    write_probe_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKTREE_ROOT / "data/output/a_action_interface_probe",
    )
    parser.add_argument(
        "--fail-on-verification-errors",
        action="store_true",
        help="Exit non-zero if any candidate has schema/provenance/geometry errors.",
    )
    args = parser.parse_args()

    report = run_offline_probe(max_samples=args.max_samples)
    paths = write_probe_outputs(report, args.output_dir)
    print("A_action_interface offline probe complete")
    print(f"schema_version: {report['schema_version']}")
    print(f"candidate_count: {report['candidate_count']}")
    print(f"stable_training_label_count: {report['stable_training_label_count']}")
    print(f"promotion_eligible_count: {report['promotion_eligible_count']}")
    print(f"verification_error_count: {report['verification_error_count']}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    if args.fail_on_verification_errors and report["verification_error_count"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
