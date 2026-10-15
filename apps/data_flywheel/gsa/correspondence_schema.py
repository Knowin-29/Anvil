"""Correspondence pipeline schema — sample format for pixel/point correspondence.

Follows the design in 02_pipeline2_correspondence.md §7:
- Dual-image input
- QA pairs with src/dst pixel coordinates (normalized to [0,1000))
- Covisibility, label_source metadata
- Supports both visible (positive) and occluded (negative) samples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class LabelSource(str, Enum):
    NATIVE_FLOW = "native_flow"
    NATIVE_DEPTH_POSE = "native_depth_pose"
    GEOMETRIC_REPROJECTION = "geometric_reprojection"


class BaselineCategory(str, Enum):
    SMALL = "small"
    MID = "mid"
    LARGE = "large"


@dataclass
class CorrespondenceQA:
    """A single correspondence query-answer pair."""
    src: List[int]  # [x, y] in [0, 1000)
    dst: Optional[List[int]]  # [x, y] in [0, 1000) or None if occluded
    visible: bool = True
    invisible_reason: Optional[str] = None  # "occluded" | None
    label_source: str = "geometric_reprojection"

    @property
    def question(self) -> str:
        sx, sy = self.src
        return (
            f"Given these two images, what pixel in the second image "
            f"corresponds to pixel ({sx}, {sy}) in the first image? "
            f"Both x and y are normalized to [0,1000). Report as (x2, y2)."
        )

    @property
    def answer(self) -> str:
        if not self.visible or self.dst is None:
            return "There is no corresponding visible pixel (occluded)."
        dx, dy = self.dst
        return f"The corresponding pixel is ({dx}, {dy})."


@dataclass
class ImageMeta:
    """Metadata for one image in the pair."""
    orig_wh: List[int]  # [width, height]
    intrinsics_source: str = "gt"
    focal_unified: bool = True
    focal_unified_px: int = 1000


@dataclass
class CorrespondenceSample:
    """A complete correspondence sample (one image pair, ~10 QA)."""
    sample_id: str
    pipeline: str = "correspondence"
    images: List[str] = field(default_factory=list)  # 2 paths
    image_meta: List[ImageMeta] = field(default_factory=list)
    coord_norm: int = 1000
    covisibility: float = 0.0
    qas: List[CorrespondenceQA] = field(default_factory=list)
    source: Dict = field(default_factory=dict)
    flywheel_tags: Dict = field(default_factory=lambda: {
        "primary": "pipeline_native",
        "sub": "pixel_correspondence",
        "g_context": True,
    })

    def to_dict(self) -> Dict:
        """Serialize to dict matching 02_pipeline2_correspondence.md §7."""
        return {
            "sample_id": self.sample_id,
            "pipeline": self.pipeline,
            "images": self.images,
            "image_meta": [
                {"orig_wh": m.orig_wh, "intrinsics_source": m.intrinsics_source,
                 "focal_unified": m.focal_unified, "focal_unified_px": m.focal_unified_px}
                for m in self.image_meta
            ],
            "coord_norm": self.coord_norm,
            "covisibility": self.covisibility,
            "qas": [
                {
                    "q": qa.question,
                    "a": qa.answer,
                    "type": "corr_pt",
                    "gt": {
                        "src": qa.src,
                        "dst": qa.dst,
                        "visible": qa.visible,
                        "invisible_reason": qa.invisible_reason,
                        "label_source": qa.label_source,
                    },
                }
                for qa in self.qas
            ],
            "source": self.source,
            "flywheel_tags": self.flywheel_tags,
        }


def validate_correspondence_sample(sample: CorrespondenceSample) -> Tuple[bool, str]:
    """Validate a correspondence sample for export readiness."""
    if len(sample.images) != 2:
        return False, "must_have_2_images"
    if len(sample.qas) < 1:
        return False, "no_qa_pairs"
    if sample.covisibility < 0.25:
        return False, f"covisibility_too_low:{sample.covisibility:.3f}"
    for i, qa in enumerate(sample.qas):
        if qa.src is None or len(qa.src) != 2:
            return False, f"qa[{i}]_invalid_src"
        if any(v < 0 or v >= sample.coord_norm for v in qa.src):
            return False, f"qa[{i}]_src_out_of_range"
        if qa.visible:
            if qa.dst is None or len(qa.dst) != 2:
                return False, f"qa[{i}]_visible_but_no_dst"
            if any(v < 0 or v >= sample.coord_norm for v in qa.dst):
                return False, f"qa[{i}]_dst_out_of_range"
    return True, "ok"
