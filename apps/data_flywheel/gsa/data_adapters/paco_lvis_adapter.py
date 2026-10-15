"""PACO-LVIS Adapter — Convert PACO-LVIS annotations to A_identity_attr labels.

PACO-LVIS provides 641K part masks over 260K object boxes with rich
attribute annotations (color, material, pattern, transparency).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZipFile

from ..a_pipeline_schemas import AIdentityAttrLabel, validate_identity_attr
from ..taxonomy import (
    AIdentityAttrSubtype,
    Bucket,
    SourceProvenance,
    VerifierStatus,
)

logger = logging.getLogger("gsa.data_adapters.paco_lvis")

PACO_ATTR_MAP = {
    "color": "color",
    "pattern_marking": None,
    "material": "material",
    "transparency": None,
}


def load_paco_annotations(dataset_path: Path, split: str = "val") -> Optional[Dict]:
    """Load PACO-LVIS annotations from raw zip or extracted JSON.

    The official archive stores files as paco_lvis_v1_{train,val,test}.json.
    Older code looked for "annotations" in the member name, which does not
    match the real archive and caused PACO to appear unreadable.
    """
    extracted_json = dataset_path / "extracted" / f"paco_lvis_v1_{split}.json"
    if extracted_json.exists():
        try:
            with open(extracted_json, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load extracted PACO annotations: %s", e)
            return None

    ann_zip = dataset_path / "raw" / "paco_lvis_v1.zip"
    if ann_zip.exists():
        try:
            with ZipFile(ann_zip) as zf:
                names = zf.namelist()
                preferred = f"paco_lvis_v1_{split}.json"
                ann_file = next((n for n in names if n.endswith(preferred)), None)
                if not ann_file:
                    json_members = [n for n in names if n.endswith(".json")]
                    split_order = ["val", "test", "train"]
                    for fallback_split in split_order:
                        fallback = f"paco_lvis_v1_{fallback_split}.json"
                        ann_file = next(
                            (n for n in json_members if n.endswith(fallback)),
                            None,
                        )
                        if ann_file:
                            break
                if ann_file:
                    with zf.open(ann_file) as f:
                        return json.load(f)
        except Exception as e:
            logger.error("Failed to load PACO annotations: %s", e)
    return None


def convert_to_identity_attr_labels(
    annotations: Dict,
    max_samples: int = 1000,
) -> List[AIdentityAttrLabel]:
    """Convert PACO-LVIS annotations to AIdentityAttrLabel instances."""
    labels = []
    objects = annotations.get("annotations", [])
    categories = {c["id"]: c for c in annotations.get("categories", [])}

    for i, obj in enumerate(objects[:max_samples]):
        cat_id = obj.get("category_id")
        cat = categories.get(cat_id, {})
        obj_class = cat.get("name", "unknown")

        bbox = obj.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            bbox = [x, y, x + w, y + h]

        attrs = {"color": None, "material": None, "shape": None, "size": None}
        paco_attrs = obj.get("attributes", {})
        if isinstance(paco_attrs, dict):
            if "color" in paco_attrs:
                attrs["color"] = paco_attrs["color"]
            if "material" in paco_attrs:
                attrs["material"] = paco_attrs["material"]

        label = AIdentityAttrLabel(
            sample_id=f"paco_lvis_{obj.get('id', i)}",
            object_id=f"paco_{obj.get('id', i)}",
            object_class=obj_class,
            bbox=bbox,
            attributes=attrs,
            subtype=AIdentityAttrSubtype.MULTI_ATTR.value,
            source_provenance=SourceProvenance.PUBLIC_DATASET.value,
            verifier_status=VerifierStatus.AUTO_VERIFIABLE.value,
            g_context=True,
        )

        result = validate_identity_attr(label)
        if result.valid:
            labels.append(label)

    logger.info(f"Converted {len(labels)} PACO-LVIS labels to A_identity_attr")
    return labels
