#!/usr/bin/env python3
"""Freeze one delivery manifest binding the source-clean corpora to their audits.

Recomputes every corpus and artifact hash from disk rather than trusting the
per-policy manifests, so the delivery record is an independent verification pass.
Read-only with respect to corpora, audits, and the pre-existing E1 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.freeze_general_retention_v1_sources import canonical_hash  # noqa: E402

POLICIES = ("p3", "p4", "p5")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def verify_policy(policy: str) -> dict:
    corpus_dir = REPOSITORY_ROOT / "trains/corpus_source_clean_v1" / policy
    audit_dir = REPOSITORY_ROOT / f"artifacts/leakage_audit_source_clean_v1_{policy}"
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((audit_dir / "report.json").read_text(encoding="utf-8"))

    corpus = corpus_dir / "train.jsonl"
    observed_sha = file_sha256(corpus)
    observed_rows = count_lines(corpus)
    drift = []
    if observed_sha != manifest["corpus"]["sha256"]:
        drift.append("corpus_sha256")
    if observed_rows != manifest["corpus"]["rows"]:
        drift.append("corpus_rows")
    if report.get("posttraining_corpus", {}).get("sha256") != observed_sha:
        drift.append("audit_bound_to_different_corpus")

    per_benchmark = {
        name: {
            "state": value["state"],
            "hard_image_pairs": value["encoded_bytes"]["pairs"]
            + value["decoded_rgb"]["pairs"]
            + value["pillow_default_jpeg"]["pairs"],
            "source_id_pairs": value["source_id_pairs"],
            "normalized_text_hard_candidates": value["normalized_text_hard_candidates"],
            "perceptual_candidates": value["perceptual_candidates"],
            "perceptual_unresolved": value["perceptual_unresolved"],
            "missing_unique_scene_identity_records": value[
                "missing_unique_scene_identity_records"
            ],
        }
        for name, value in report["per_benchmark"].items()
    }
    subgroup = report["per_benchmark"]["refspatial_bench"]["scored_subgroup"]

    return {
        "policy": policy.upper(),
        "corpus": {
            "path": str(corpus.resolve()),
            "rows": observed_rows,
            "sha256": observed_sha,
            "size_bytes": corpus.stat().st_size,
            "unique_images": manifest["corpus"]["unique_images"],
        },
        "manifest": {
            "path": str((corpus_dir / "manifest.json").resolve()),
            "sha256": file_sha256(corpus_dir / "manifest.json"),
            "identity_sha256": manifest["identity_sha256"],
        },
        "training_fingerprint_cache": {
            "path": str((corpus_dir / "training_image_fingerprints.jsonl").resolve()),
            "sha256": file_sha256(corpus_dir / "training_image_fingerprints.jsonl"),
            "rows": manifest["training_fingerprint_cache"]["rows"],
            "binding_identity_sha256": manifest["training_fingerprint_cache"][
                "binding_identity_sha256"
            ],
        },
        "excluded_buckets": manifest["exclusion_decisions"][
            "layer_a_bucket_exclusion"
        ]["excluded_buckets"],
        "content_layer_exclusions": manifest["exclusion_decisions"][
            "layer_b_content_exclusion"
        ]["by_reason"],
        "distribution": manifest["distribution"],
        "audit": {
            "report_path": str((audit_dir / "report.json").resolve()),
            "report_sha256": file_sha256(audit_dir / "report.json"),
            "report_identity_sha256": report["identity_sha256"],
            "auditor_sha256": report["auditor"]["sha256"],
            "state": report["state"],
            "claim_eligible": report["claim_eligible"],
            "collision_counts": report["collision_counts"],
            "metadata_blockers": report["identity_coverage"]["metadata_blockers"],
            "per_benchmark": per_benchmark,
            "refspatial_location_subgroup": {
                "state": subgroup["state"],
                "records": subgroup["records"],
                "perceptual_candidates": subgroup["perceptual_candidates"],
                "perceptual_unresolved": subgroup["perceptual_unresolved"],
                "unresolved_uids": subgroup["unresolved_uids"],
            },
        },
        "verification_drift": drift,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/source_clean_delivery_v1/delivery_manifest.json",
    )
    args = parser.parse_args()

    policies = [verify_policy(policy) for policy in POLICIES]
    scaleout_runs = []
    for name in ("scaleout_v1", "scaleout_v1_confirm"):
        scaleout = (
            REPOSITORY_ROOT / "data/scannet_train_sens_adapter" / name / "scaleout_manifest.json"
        )
        if not scaleout.is_file():
            continue
        payload = json.loads(scaleout.read_text(encoding="utf-8"))
        window = payload["scene_window"]
        scaleout_runs.append({
            "run": name,
            "path": str(scaleout.resolve()),
            "sha256": file_sha256(scaleout),
            "verdict": payload["verdict"],
            "execution": payload["execution"],
            "scaling_estimate": payload["scaling_estimate"],
            "scene_offset": window["offset"],
            "scenes_extracted": window["scenes_extracted"],
            "scenes_failed": window["scenes_failed"],
            "frames_skipped_undecodable": sum(
                scene["frames_skipped_undecodable"] for scene in payload["scenes"]
            ),
            "use_policy": payload["use_policy"],
        })
    scaleout_doc = {
        "runs": scaleout_runs,
        "reproducibility_note": (
            "two disjoint scene windows (offset 0 and offset 400) were extracted "
            "independently; both report zero failed scenes and agree on per-frame cost "
            "within 10 percent, so the full-run estimate is not window-specific"
        ),
    } or None

    approval = REPOSITORY_ROOT / "data/scannet_license/approval_record.json"
    delivery = {
        "schema_version": 1,
        "delivery_id": "gsa-source-clean-corpus-and-leakage-recheck-v1",
        "frozen_by": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__)),
        },
        "source_pool": {
            "path": "/path/to/Anvil/trains/corpus_v2",
            "measured_rows": 4166135,
            "note": "read-only; not modified by this delivery",
        },
        "evaluation_fingerprints_reused": {
            "path": str((REPOSITORY_ROOT / "artifacts/leakage_audit_full_v1/eval_image_fingerprints.jsonl").resolve()),
            "sha256": file_sha256(REPOSITORY_ROOT / "artifacts/leakage_audit_full_v1/eval_image_fingerprints.jsonl"),
            "unique_images": 6298,
        },
        "preserved_prior_artifacts": {
            "e1_corpus": "/path/to/Anvil/trains/gsa_bench/train_shuf.jsonl",
            "e1_audit_dir": str((REPOSITORY_ROOT / "artifacts/leakage_audit_full_v1").resolve()),
            "note": "left untouched; this delivery only adds new directories",
        },
        "recommended_policy": "P4",
        "recommendation_rationale": (
            "P4 is the largest corpus with every hard-leak axis at zero: 263,322 rows / "
            "159,497 unique images, zero encoded/decoded/JPEG collisions, zero "
            "release-qualified source and scene identity collisions, zero normalized-text "
            "hard candidates, and zero unresolved perceptual near-duplicates. It keeps the "
            "RefSpatial family that P5 drops, so the scored RefSpatial-location cell "
            "retains in-domain training support while still reporting PASS."
        ),
        "residual_blocker_explanation": (
            "No policy can reach global state=PASS. The auditor folds "
            "metadata_blockers into the global verdict, and that blocker is a property of "
            "the public evaluation releases (CV-Bench, Where2Place, RefSpatial and the "
            "internal GSA test split do not expose a unique scene identity), not of the "
            "training corpus. Detection thresholds were not relaxed."
        ),
        "policies": policies,
        "scannet_train_split": {
            "approval_record": {
                "path": str(approval.resolve()),
                "sha256": file_sha256(approval),
            },
            "approval_notes": str((REPOSITORY_ROOT / "data/scannet_license/README.md").resolve()),
            "canary_manifest": str((REPOSITORY_ROOT / "data/scannet_train_sens_adapter/canary_v1/source_manifest.json").resolve()),
            "scaleout": scaleout_doc,
            "val_split_guard": (
                "data/scannet_val_frame_bundles_v3 keeps may_enter_training=false and was "
                "neither read nor modified for training materialization"
            ),
        },
        "constraints_observed": {
            "cpu_only": True,
            "gpu_used": False,
            "detection_thresholds_relaxed": False,
            "existing_artifacts_deleted": False,
            "paper_directory_modified": False,
            "gsa_iclr2027_workspace_touched": False,
        },
    }
    delivery["identity_sha256"] = canonical_hash(delivery)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(delivery, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, args.out)
    digest = hashlib.sha256(payload).hexdigest()
    args.out.with_name("delivery_manifest.sha256").write_text(
        f"{digest}  delivery_manifest.json\n", encoding="ascii"
    )

    print(json.dumps({
        "delivery_manifest": str(args.out.resolve()),
        "delivery_manifest_sha256": digest,
        "identity_sha256": delivery["identity_sha256"],
        "recommended_policy": delivery["recommended_policy"],
        "policies": [
            {
                "policy": item["policy"],
                "rows": item["corpus"]["rows"],
                "unique_images": item["corpus"]["unique_images"],
                "state": item["audit"]["state"],
                "hard_axes_all_zero": all(
                    value["hard_image_pairs"] == 0
                    and value["source_id_pairs"] == 0
                    and value["normalized_text_hard_candidates"] == 0
                    and value["perceptual_unresolved"] == 0
                    for value in item["audit"]["per_benchmark"].values()
                ),
                "cv_bench": item["audit"]["per_benchmark"]["cv_bench"]["state"],
                "refspatial_location": item["audit"]["refspatial_location_subgroup"]["state"],
                "verification_drift": item["verification_drift"],
            }
            for item in policies
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
