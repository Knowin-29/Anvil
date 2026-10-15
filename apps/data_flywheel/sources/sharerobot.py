"""ShareRobot dataset source for sidecar flywheel pipelines."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence

from PIL import Image

from apps.data_flywheel.sources.base import DataSource, Sample
from apps.data_flywheel.sources.registry import register_source

logger = logging.getLogger("flywheel.sources.sharerobot")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_DEFAULT_SUBSETS = ["trajectory", "affordance"]


def _normalize_subsets(subsets: Optional[Sequence[str] | str]) -> List[str]:
    if subsets is None:
        return list(_DEFAULT_SUBSETS)
    if isinstance(subsets, str):
        parts = [part.strip() for part in subsets.split(",")]
    else:
        parts = [str(part).strip() for part in subsets]
    return [part for part in parts if part]


def _safe_id(value: Any) -> str:
    text = str(value) if value is not None else "sample"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:80] or "sample"


@register_source("sharerobot")
class ShareRobotSource(DataSource):
    """Load ShareRobot affordance/trajectory single-frame records.

    The planning split is intentionally not included by default: it is a large
    multi-frame VQA corpus and is not the right first input for A_robot_state.
    """

    source_type = "sharerobot"

    def __init__(
        self,
        paths: List[str],
        subsets: Optional[Sequence[str] | str] = None,
        max_images: Optional[int] = None,
        recursive: bool = True,
    ):
        self.paths = paths
        self.subsets = _normalize_subsets(subsets)
        self.max_images = int(max_images) if max_images else None
        self.recursive = recursive
        self._records: Optional[List[Dict[str, Any]]] = None

    def _subset_json_path(self, root: str, subset: str) -> str:
        if subset == "trajectory":
            return os.path.join(root, "trajectory", "trajectory.json")
        if subset == "affordance":
            return os.path.join(root, "affordance", "affordance.json")
        return os.path.join(root, subset, f"{subset}.json")

    def _subset_image_root(self, root: str, subset: str) -> str:
        return os.path.join(root, subset, "images")

    def _resolve_image_path(self, root: str, subset: str, rel_path: str) -> str:
        rel_path = rel_path.lstrip("/")
        candidates = [
            os.path.join(self._subset_image_root(root, subset), rel_path),
            os.path.join(root, rel_path),
        ]
        if rel_path.startswith("rt_frames_success/"):
            candidates.append(os.path.join(root, "planning", "images", rel_path[len("rt_frames_success/"):]))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return candidates[0]

    def _load_subset_records(self, root: str, subset: str) -> List[Dict[str, Any]]:
        json_path = self._subset_json_path(root, subset)
        if not os.path.isfile(json_path):
            logger.warning("ShareRobot %s annotation not found: %s", subset, json_path)
            return []

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            logger.warning("ShareRobot %s annotation is not a list: %s", subset, json_path)
            return []

        records: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            rel_path = str(item.get("image_path") or "")
            if not rel_path:
                continue
            image_path = self._resolve_image_path(root, subset, rel_path)
            if os.path.splitext(image_path)[1].lower() not in _IMAGE_EXTS:
                continue
            if not os.path.isfile(image_path):
                continue
            records.append({
                "root": root,
                "subset": subset,
                "image_path": image_path,
                "relative_image_path": rel_path,
                "annotation_path": json_path,
                "annotation": item,
            })
        return records

    def _scan_records(self) -> List[Dict[str, Any]]:
        if self._records is not None:
            return self._records

        records: List[Dict[str, Any]] = []
        for root in self.paths:
            if not os.path.isdir(root):
                logger.warning("ShareRobot root not found, skipping: %s", root)
                continue
            for subset in self.subsets:
                records.extend(self._load_subset_records(root, subset))
                if self.max_images and len(records) >= self.max_images:
                    self._records = records[: self.max_images]
                    logger.info("ShareRobotSource: found %s records", len(self._records))
                    return self._records

        self._records = records[: self.max_images] if self.max_images else records
        logger.info("ShareRobotSource: found %s records", len(self._records))
        return self._records

    def iter_samples(self) -> Iterator[Sample]:
        for record in self._scan_records():
            image_path = record["image_path"]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                logger.warning("Failed to open ShareRobot image %s: %s", image_path, exc)
                continue

            annotation = record["annotation"]
            sample_id = f"sharerobot_{record['subset']}_{_safe_id(annotation.get('id'))}"
            meta_data = annotation.get("meta_data") if isinstance(annotation.get("meta_data"), dict) else {}
            yield Sample(
                sample_id=sample_id,
                source_type="sharerobot",
                frames=[image],
                metadata={
                    "image_path": image_path,
                    "filename": os.path.basename(image_path),
                    "dataset": "sharerobot",
                    "share_robot_subset": record["subset"],
                    "relative_image_path": record["relative_image_path"],
                    "annotation_path": record["annotation_path"],
                    "annotation": annotation,
                    "instruction": annotation.get("instruction"),
                    "original_dataset": meta_data.get("original_dataset"),
                },
            )

    def __len__(self) -> int:
        return len(self._scan_records())

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "sharerobot",
            "paths": self.paths,
            "subsets": self.subsets,
            "total_images": len(self),
        }
