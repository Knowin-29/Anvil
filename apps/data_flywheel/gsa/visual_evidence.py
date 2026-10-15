"""Visual Evidence — GsaEvidenceBundle, GsaObjectRef, and deterministic object IDs.

Provides the shared input structure for A_state and A_action pipelines.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GsaObjectRef:
    """Stable reference to a detected object within a sample."""

    object_id: str
    object_name: str
    raw_object_name: str
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] normalized or pixel
    mask_path: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)
    blocked_by: List[str] = field(default_factory=list)
    directly_reachable: Optional[bool] = None
    source_provenance: str = "vlm"
    confidence: Optional[float] = None

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        if self.bbox and len(self.bbox) == 4:
            return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
        return None


@dataclass
class GsaSpatialFact:
    """A spatial relation fact between two objects."""

    fact_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: Optional[float] = None
    evidence_type: str = "rule"  # rule | vlm | depth


@dataclass
class GsaPlacementRegion:
    """A placement region derived from scene understanding."""

    region_id: str
    region_type: str  # free-space | inside-container
    polygon: List[List[float]] = field(default_factory=list)
    parent_object_id: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class GsaEvidenceBundle:
    """The unified input to A_state and A_action pipelines."""

    sample_id: str
    image_path: Optional[str] = None
    scene_family: str = ""
    scenario_id: str = ""
    objects: List[GsaObjectRef] = field(default_factory=list)
    spatial_facts: List[GsaSpatialFact] = field(default_factory=list)
    placement_regions: List[GsaPlacementRegion] = field(default_factory=list)
    business_context: Dict[str, Any] = field(default_factory=dict)


# --- Deterministic Object ID Generation ---

_NORMALIZE_RE = re.compile(r"[^a-z0-9_]")


def normalize_class_name(raw_name: str) -> str:
    """Normalize an object class name to a canonical form."""
    name = str(raw_name or "unknown").lower().strip()
    name = name.replace(" ", "_").replace("-", "_")
    name = _NORMALIZE_RE.sub("", name)
    return name or "unknown"


def generate_object_id(
    normalized_name: str,
    bbox: Optional[List[float]],
    index: int,
) -> str:
    """Generate a deterministic object_id: {class}_{index}.

    Index is assigned by sorting objects top-to-bottom, left-to-right (y then x).
    """
    return f"{normalized_name}_{index}"


def assign_object_ids(objects: List[Dict[str, Any]]) -> List[GsaObjectRef]:
    """Assign deterministic IDs to a list of detected objects.

    Objects are sorted by (y_center, x_center) to ensure stable ordering.
    Same-class objects get incrementing indices.
    """
    # Sort by y then x center
    def sort_key(obj: Dict[str, Any]) -> Tuple[float, float]:
        bbox = obj.get("bbox")
        if bbox and len(bbox) == 4:
            cy = (bbox[1] + bbox[3]) / 2
            cx = (bbox[0] + bbox[2]) / 2
            return (cy, cx)
        return (float("inf"), float("inf"))

    sorted_objs = sorted(objects, key=sort_key)

    # Assign IDs per class
    class_counters: Dict[str, int] = {}
    results: List[GsaObjectRef] = []

    for obj in sorted_objs:
        raw_name = obj.get("object_name", obj.get("name", "unknown"))
        norm_name = normalize_class_name(raw_name)
        idx = class_counters.get(norm_name, 0)
        class_counters[norm_name] = idx + 1

        obj_id = generate_object_id(norm_name, obj.get("bbox"), idx)
        ref = GsaObjectRef(
            object_id=obj_id,
            object_name=norm_name,
            raw_object_name=raw_name,
            bbox=obj.get("bbox"),
            mask_path=obj.get("mask_path"),
            state=obj.get("state", {}),
            blocked_by=obj.get("blocked_by", []),
            directly_reachable=obj.get("directly_reachable"),
            source_provenance=obj.get("source_provenance", "vlm"),
            confidence=obj.get("confidence"),
        )
        results.append(ref)

    return results
