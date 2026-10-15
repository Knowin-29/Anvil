"""GSA Manifest — Dataset manifest and coverage matrix.

Builds dataset_manifest_v0.jsonl from real logs, anchor set, and public sources.
Tracks source readiness and coverage by source x scene_family x bucket x subtype.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .taxonomy import Bucket, BUCKET_SUBTYPES, SceneFamily, VerifierStatus

logger = logging.getLogger("gsa.manifest")


class SourceStatus:
    READY = "ready"
    ADAPTER_NEEDED = "adapter_needed"
    MANUAL_DOWNLOAD = "manual_download"
    LICENSE_REVIEW = "license_review"
    MANIFEST_ONLY = "manifest_only"
    REJECT_FOR_V0 = "reject_for_v0"


@dataclass
class ManifestRow:
    sample_id: str
    source_id: str
    source_type: str  # real_log | public_dataset | sim | anchor | vlm
    source_provenance: str
    scene_family: str
    scenario_id: str
    bucket: str
    subtype: str
    split: str = "train"
    label_path: Optional[str] = None
    prediction_path: Optional[str] = None
    verifier_status: str = VerifierStatus.REVIEW_REQUIRED.value
    adapter_status: str = SourceStatus.READY
    license_status: str = "internal"
    review_queue_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class CoverageCell:
    count: int = 0
    auto_verifiable: int = 0
    review_required: int = 0


class GSAManifest:
    """Manages the dataset manifest and coverage tracking."""

    def __init__(self):
        self.rows: List[ManifestRow] = []
        self._coverage: Dict[str, CoverageCell] = defaultdict(CoverageCell)

    def add_row(self, row: ManifestRow):
        self.rows.append(row)
        key = f"{row.source_type}|{row.scene_family}|{row.bucket}|{row.subtype}"
        cell = self._coverage[key]
        cell.count += 1
        if row.verifier_status == VerifierStatus.AUTO_VERIFIABLE.value:
            cell.auto_verifiable += 1
        elif row.verifier_status == VerifierStatus.REVIEW_REQUIRED.value:
            cell.review_required += 1

    def add_from_labels(self, labels: List[Dict[str, Any]], source_id: str = "vlm_gen",
                        source_type: str = "vlm"):
        """Bulk add manifest rows from generated labels."""
        for label in labels:
            row = ManifestRow(
                sample_id=label.get("sample_id", ""),
                source_id=source_id,
                source_type=source_type,
                source_provenance=label.get("source_provenance", "vlm"),
                scene_family=label.get("scene_family", ""),
                scenario_id=label.get("scenario_id", ""),
                bucket=label.get("bucket", ""),
                subtype=label.get("subtype", ""),
                verifier_status=label.get("verifier_status", VerifierStatus.REVIEW_REQUIRED.value),
            )
            self.add_row(row)

    def get_coverage_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get coverage matrix: {bucket/subtype: {scene_family: count}}."""
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in self.rows:
            key = f"{row.bucket}/{row.subtype}"
            matrix[key][row.scene_family] += 1
        return {k: dict(v) for k, v in matrix.items()}

    def get_source_readiness(self) -> Dict[str, Dict[str, Any]]:
        """Get source readiness summary."""
        sources: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "status": ""})
        for row in self.rows:
            sources[row.source_id]["count"] += 1
            sources[row.source_id]["status"] = row.adapter_status
            sources[row.source_id]["source_type"] = row.source_type
        return dict(sources)

    def save(self, path: str):
        """Save manifest to JSONL file."""
        with open(path, "w") as f:
            for row in self.rows:
                f.write(row.to_jsonl() + "\n")
        logger.info(f"Saved manifest with {len(self.rows)} rows to {path}")

    def load(self, path: str):
        """Load manifest from JSONL file."""
        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    row = ManifestRow(**{k: v for k, v in data.items()
                                        if k in ManifestRow.__dataclass_fields__})
                    self.add_row(row)

    def summary(self) -> Dict[str, Any]:
        """Get manifest summary statistics."""
        buckets = defaultdict(int)
        scenes = defaultdict(int)
        sources = defaultdict(int)
        for row in self.rows:
            buckets[row.bucket] += 1
            scenes[row.scene_family] += 1
            sources[row.source_type] += 1

        return {
            "total_rows": len(self.rows),
            "by_bucket": dict(buckets),
            "by_scene_family": dict(scenes),
            "by_source_type": dict(sources),
        }
