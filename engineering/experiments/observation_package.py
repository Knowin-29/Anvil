"""Fail-closed validation for one Anvil evaluation observation package.

This module validates row identity and output-health semantics only. It does not run a
model, compute benchmark metrics, or promote a package into the result registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ObservationPackageError(ValueError):
    """Raised when an observation package violates its frozen row contract."""


MANIFEST_REQUIRED = {
    "uid",
    "scene_id",
    "source_image_sha256",
    "split_revision",
    "form",
}

PREDICTION_REQUIRED = {
    "uid",
    "scene_id",
    "source_image_sha256",
    "run_id",
    "canonical_method_id",
    "protocol_id",
    "split_revision",
    "form",
    "seed",
    "raw_prediction",
    "parse_state",
    "primary_score",
    "malformed",
    "truncated",
    "missing",
}

PREDICTION_STRING_IDENTITIES = MANIFEST_REQUIRED | {
    "run_id",
    "canonical_method_id",
    "protocol_id",
}

VALID_FORMS = {
    "bbox",
    "detection",
    "mask",
    "point",
    "relative_depth",
    "affordance",
    "gripper_dof",
    "action_plan",
}


@dataclass(frozen=True)
class PackageIdentity:
    run_id: str
    canonical_method_id: str
    protocol_id: str
    split_revision: str
    seed: int


@dataclass(frozen=True)
class PackageSummary:
    manifest_rows: int
    prediction_rows: int
    malformed_rows: int
    truncated_rows: int
    missing_rows: int
    manifest_sha256: str
    prediction_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_rows": self.manifest_rows,
            "prediction_rows": self.prediction_rows,
            "malformed_rows": self.malformed_rows,
            "truncated_rows": self.truncated_rows,
            "missing_rows": self.missing_rows,
            "manifest_sha256": self.manifest_sha256,
            "prediction_sha256": self.prediction_sha256,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ObservationPackageError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ObservationPackageError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ObservationPackageError(
                    f"{path}:{line_number}: row must be a JSON object"
                )
            rows.append(row)
    if not rows:
        raise ObservationPackageError(f"{path}: no rows")
    return rows


def _require_fields(row: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(key for key in required if key not in row or row[key] is None)
    if missing:
        raise ObservationPackageError(f"{label}: missing fields {', '.join(missing)}")


def _require_nonblank_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ObservationPackageError(f"{label} must be a non-empty string")


def _index_unique(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        uid = row["uid"]
        _require_nonblank_string(uid, f"{label} row {row_number}: uid")
        if uid in indexed:
            raise ObservationPackageError(f"{label}: duplicate uid {uid}")
        indexed[uid] = row
    return indexed


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def validate_package(
    manifest_path: Path,
    prediction_path: Path,
    identity: PackageIdentity,
) -> PackageSummary:
    for label, value in {
        "run_id": identity.run_id,
        "canonical_method_id": identity.canonical_method_id,
        "protocol_id": identity.protocol_id,
        "split_revision": identity.split_revision,
    }.items():
        _require_nonblank_string(value, f"package identity {label}")
    if type(identity.seed) is not int:
        raise ObservationPackageError("package identity seed must be an integer")

    manifest_rows = load_jsonl(manifest_path)
    prediction_rows = load_jsonl(prediction_path)

    for row_number, row in enumerate(manifest_rows, start=1):
        _require_fields(row, MANIFEST_REQUIRED, f"manifest row {row_number}")
        for field in sorted(MANIFEST_REQUIRED - {"uid"}):
            _require_nonblank_string(row[field], f"manifest row {row_number}: {field}")
        if row["form"] not in VALID_FORMS:
            raise ObservationPackageError(
                f"manifest row {row_number}: unknown form {row['form']!r}"
            )
        image_hash = row["source_image_sha256"]
        if not isinstance(image_hash, str) or len(image_hash) != 64 or any(
            character not in "0123456789abcdef" for character in image_hash
        ):
            raise ObservationPackageError(
                f"manifest row {row_number}: source_image_sha256 must be lowercase hex"
            )
    for row_number, row in enumerate(prediction_rows, start=1):
        _require_fields(row, PREDICTION_REQUIRED, f"prediction row {row_number}")
        for field in sorted(PREDICTION_STRING_IDENTITIES - {"uid"}):
            _require_nonblank_string(row[field], f"prediction row {row_number}: {field}")
        if type(row["seed"]) is not int:
            raise ObservationPackageError(f"prediction row {row_number}: seed must be an integer")

    manifest = _index_unique(manifest_rows, "manifest")
    predictions = _index_unique(prediction_rows, "predictions")
    expected = set(manifest)
    observed = set(predictions)
    missing_uids = sorted(expected - observed)
    extra_uids = sorted(observed - expected)
    if missing_uids or extra_uids:
        raise ObservationPackageError(
            "UID coverage mismatch: "
            f"missing={missing_uids[:10]} extra={extra_uids[:10]}"
        )

    malformed_count = 0
    truncated_count = 0
    missing_count = 0
    for uid in sorted(expected):
        target = manifest[uid]
        row = predictions[uid]
        comparisons = {
            "scene_id": target["scene_id"],
            "source_image_sha256": target["source_image_sha256"],
            "split_revision": identity.split_revision,
            "form": target["form"],
            "run_id": identity.run_id,
            "canonical_method_id": identity.canonical_method_id,
            "protocol_id": identity.protocol_id,
            "seed": identity.seed,
        }
        for field, expected_value in comparisons.items():
            if row[field] != expected_value:
                raise ObservationPackageError(
                    f"prediction {uid}: {field}={row[field]!r}, expected {expected_value!r}"
                )
        if target["split_revision"] != identity.split_revision:
            raise ObservationPackageError(
                f"manifest {uid}: split_revision={target['split_revision']!r}, "
                f"expected {identity.split_revision!r}"
            )
        score = row["primary_score"]
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ObservationPackageError(
                f"prediction {uid}: primary_score must be finite and within [0, 1]"
            )
        for flag in ("malformed", "truncated", "missing"):
            if not _is_bool(row[flag]):
                raise ObservationPackageError(f"prediction {uid}: {flag} must be boolean")
        malformed_count += int(row["malformed"])
        truncated_count += int(row["truncated"])
        missing_count += int(row["missing"])
        if any(row[flag] for flag in ("malformed", "truncated", "missing")):
            if float(score) != 0.0:
                raise ObservationPackageError(
                    f"prediction {uid}: invalid/missing output must receive zero primary_score"
                )

    return PackageSummary(
        manifest_rows=len(manifest_rows),
        prediction_rows=len(prediction_rows),
        malformed_rows=malformed_count,
        truncated_rows=truncated_count,
        missing_rows=missing_count,
        manifest_sha256=sha256_file(manifest_path),
        prediction_sha256=sha256_file(prediction_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split-revision", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    identity = PackageIdentity(
        run_id=args.run_id,
        canonical_method_id=args.method,
        protocol_id=args.protocol,
        split_revision=args.split_revision,
        seed=args.seed,
    )
    try:
        summary = validate_package(args.manifest, args.predictions, identity)
    except ObservationPackageError as error:
        print(json.dumps({"valid": False, "error": str(error)}, indent=2))
        return 1
    print(json.dumps({"valid": True, **summary.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
