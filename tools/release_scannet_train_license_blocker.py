#!/usr/bin/env python3
"""Apply the project-owner ScanNet train-split approval to an adapter manifest.

Rewrites only the ``license`` and ``use_policy`` blocks of a ScanNet adapter
manifest, preserving every other field and every source hash. The original
manifest is kept alongside as ``source_manifest.pre_approval.json`` so the
pre-approval state stays auditable, and the applied approval provenance is
recorded inline rather than asserted as an official ScanNet signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.extract_scannet_train_sens_scaleout import load_approval  # noqa: E402

RELEASED_BLOCKER = "scannet_tos_approval_not_locally_verified"
RETAINED_BLOCKER = "scannet_signed_terms_of_use_not_locally_archived"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data/scannet_train_sens_adapter/canary_v1/source_manifest.json",
    )
    parser.add_argument(
        "--approval-record",
        type=Path,
        default=REPOSITORY_ROOT / "data/scannet_license/approval_record.json",
    )
    parser.add_argument(
        "--release-scaleout-blocker",
        action="store_true",
        help="also release adapter_canary_only_not_scaleout_approved",
    )
    args = parser.parse_args()

    approval = load_approval(args.approval_record)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    backup = args.manifest.with_name("source_manifest.pre_approval.json")
    if not backup.exists():
        shutil.copy2(args.manifest, backup)

    license_block = dict(manifest.get("license") or {})
    license_block.update({
        "approval_status": "APPROVED_BY_PROJECT_OWNER_SIGNED_TERMS_PENDING_ARCHIVAL",
        "approval_record": approval,
        "signed_terms_locally_archived": False,
        "approval_basis": (
            "explicit project-owner authorization dated "
            f"{approval['granted_on_utc_date']}; no signed ScanNet Terms-of-Use "
            "document exists locally and none is claimed"
        ),
    })
    manifest["license"] = license_block

    policy = dict(manifest.get("use_policy") or {})
    blockers = [
        blocker for blocker in (policy.get("training_materialization_blockers") or [])
        if blocker != RELEASED_BLOCKER
    ]
    if args.release_scaleout_blocker:
        blockers = [
            blocker for blocker in blockers
            if blocker != "adapter_canary_only_not_scaleout_approved"
        ]
    policy["training_materialization_blockers"] = blockers
    policy["training_materialization_eligible"] = not blockers
    policy["released_blockers"] = [RELEASED_BLOCKER] + (
        ["adapter_canary_only_not_scaleout_approved"] if args.release_scaleout_blocker else []
    )
    policy["retained_provenance_blockers"] = [RETAINED_BLOCKER]
    policy["val_split_note"] = (
        "data/scannet_val_frame_bundles_v3 carries may_enter_training=false; only "
        "the official train split is released by this approval"
    )
    manifest["use_policy"] = policy
    manifest["verdict"] = (
        "CANARY_PASS_SCALEOUT_APPROVED_BY_PROJECT_OWNER"
        if args.release_scaleout_blocker
        else "CANARY_PASS_LICENSE_APPROVED_SCALEOUT_REVIEW_PENDING"
    )
    manifest["approval_applied_by"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preserved_pre_approval_manifest": str(backup.resolve()),
    }

    payload = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    partial = args.manifest.with_name(args.manifest.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, args.manifest)
    digest = hashlib.sha256(payload).hexdigest()
    sha_path = args.manifest.with_name("source_manifest.sha256")
    if sha_path.exists():
        sha_path.write_text(f"{digest}  source_manifest.json\n", encoding="ascii")

    print(json.dumps({
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": digest,
        "approval_status": license_block["approval_status"],
        "training_materialization_eligible": policy["training_materialization_eligible"],
        "released_blockers": policy["released_blockers"],
        "retained_provenance_blockers": policy["retained_provenance_blockers"],
        "pre_approval_manifest": str(backup.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
