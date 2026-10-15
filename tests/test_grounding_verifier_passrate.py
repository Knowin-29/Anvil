"""Tests for direct G_grounding verifier pass-rate evaluation."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_script_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / f"{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


verifier_service = _load_script_module("verifier_service")
evaluate_grounding_verifier = _load_script_module("evaluate_grounding_verifier")
app = verifier_service.app
evaluate_grounding_batch = verifier_service.evaluate_grounding_batch
load_grounding_verifier_config = verifier_service.load_grounding_verifier_config
DEFAULT_GROUNDING_CONFIG_PATH = verifier_service.DEFAULT_GROUNDING_VERIFIER_CONFIG_PATH
load_labels = evaluate_grounding_verifier.load_labels
search_grounding_config_for_target = evaluate_grounding_verifier.search_grounding_config_for_target


def _sample_labels():
    return [
        {
            "sample_id": "good",
            "pipeline": "G_grounding",
            "image_path": "/tmp/good.jpg",
            "object_class": "red_cup",
            "bbox": [100, 100, 300, 300],
            "image_size": [1000, 1000],
            "confidence": 0.95,
        },
        {
            "sample_id": "generic",
            "pipeline": "G_grounding",
            "image_path": "/tmp/generic.jpg",
            "object_class": "object",
            "bbox": [100, 100, 300, 300],
            "image_size": [1000, 1000],
            "confidence": 0.9,
        },
        {
            "sample_id": "bad_bbox",
            "pipeline": "G_grounding",
            "image_path": "/tmp/bad.jpg",
            "object_class": "cup",
            "bbox": [100, 100, 50, 200],
            "image_size": [1000, 1000],
            "confidence": 0.55,
        },
    ]


def test_evaluate_grounding_batch_reports_independent_and_cumulative_passrates():
    report = evaluate_grounding_batch(_sample_labels(), specialist_mode="off")

    assert report["total"] == 3
    assert report["overall"]["verified_auto"]["count"] == 1
    assert report["overall"]["needs_human_review"]["count"] == 1
    assert report["overall"]["rejected"]["count"] == 1

    assert report["rule_passrate"]["schema_valid"]["passed"] == 2
    assert report["rule_passrate"]["content_quality"]["passed"] == 2
    assert report["rule_passrate"]["geometry_valid"]["passed"] == 2
    assert report["rule_passrate"]["confidence_check"]["passed"] == 1

    assert report["cumulative_passrate"]["schema_valid"]["passed"] == 2
    assert report["cumulative_passrate"]["content_quality"]["passed"] == 1
    assert report["cumulative_passrate"]["confidence_check"]["passed"] == 1

    item_by_id = {item["sample_id"]: item for item in report["items"]}
    assert item_by_id["good"]["tag"] == "verified_auto"
    assert item_by_id["generic"]["reason"] == "content_quality=0.00"
    assert item_by_id["bad_bbox"]["checks"]["schema_valid"] == 0.0


def test_verify_grounding_batch_endpoint_accepts_items_payload():
    client = app.test_client()

    response = client.post(
        "/verify_grounding_batch",
        json={"items": _sample_labels(), "specialist": "off"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["report"]["total"] == 3
    assert math.isclose(payload["report"]["overall"]["verified_auto"]["rate"], 1 / 3, abs_tol=1e-6)


def test_load_labels_reads_jsonl(tmp_path: Path):
    input_path = tmp_path / "labels.jsonl"
    labels = _sample_labels()[:2]
    input_path.write_text(
        "\n".join(json.dumps(label, ensure_ascii=False) for label in labels) + "\n",
        encoding="utf-8",
    )

    assert load_labels(input_path) == labels


def test_default_grounding_config_preserves_existing_passrate():
    config = load_grounding_verifier_config(DEFAULT_GROUNDING_CONFIG_PATH)

    report = evaluate_grounding_batch(_sample_labels(), specialist_mode="off", grounding_config=config)

    assert report["overall"]["verified_auto"]["count"] == 1
    assert report["overall"]["needs_human_review"]["count"] == 1
    assert report["overall"]["rejected"]["count"] == 1
    assert report["rule_passrate"]["content_quality"]["passed"] == 2


def test_rule_config_change_can_shift_passrate():
    config = load_grounding_verifier_config(DEFAULT_GROUNDING_CONFIG_PATH)
    relaxed_config = copy.deepcopy(config)
    relaxed_config["content_quality"]["generic_name_score"] = 1.0

    report = evaluate_grounding_batch(_sample_labels(), specialist_mode="off", grounding_config=relaxed_config)

    assert report["overall"]["verified_auto"]["count"] == 2
    assert report["items"][1]["checks"]["content_quality"] == 1.0


def test_auto_search_recommends_config_near_target_passrate():
    config = load_grounding_verifier_config(DEFAULT_GROUNDING_CONFIG_PATH)

    result = search_grounding_config_for_target(
        _sample_labels(),
        base_config=config,
        target_passrate=2 / 3,
        specialist_mode="off",
    )

    assert math.isclose(result["best"]["actual_passrate"], 2 / 3, abs_tol=1e-6)
    assert result["best"]["diff"]["content_quality.generic_name_score"] == 1.0
    assert result["best"]["report"]["overall"]["verified_auto"]["count"] == 2
