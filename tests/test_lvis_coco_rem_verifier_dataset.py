from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_lvis_coco_rem_verifier_dataset",
        SCRIPTS_DIR / "build_lvis_coco_rem_verifier_dataset.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load build_lvis_coco_rem_verifier_dataset")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_lvis_coco_rem_verifier_dataset"] = module
    spec.loader.exec_module(module)
    return module


def test_xywh_to_xyxy_converts_lvis_coco_bbox_format():
    generator = _load_generator()

    assert generator.xywh_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]
    assert generator.xywh_to_xyxy([1.5, 2.25, 3.5, 4.75]) == [1.5, 2.25, 5.0, 7.0]


def test_make_verifier_record_matches_required_jsonl_schema(tmp_path: Path):
    generator = _load_generator()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpg")

    record = generator.make_verifier_record(
        sample_id="case_0001",
        from_dataset="Lvis v1.0",
        annotation_id=123,
        image_path=image,
        object_class="coffee mug",
        bbox_xywh=[10, 20, 30, 40],
        image_size=(640, 480),
        confidence=0.95,
    )

    assert record == {
        "sample_id": "case_0001",
        "pipeline": "G_grounding",
        "image_path": str(image.resolve()),
        "object_class": "coffee mug",
        "bbox": [10, 20, 40, 60],
        "image_size": [640, 480],
        "confidence": 0.95,
        "from_dataset": "Lvis v1.0",
        "annotation_id": 123,
    }
