"""Offline tests for the independent G_grounding generator validation module.

All VLM network calls are monkeypatched so these run without a backend.
"""

import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "grounding_verifier_rules.yaml"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_script_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / f"{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_script_module("evaluate_grounding_generator")


# ----------------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------------

def test_shared_config_loads_grounding_generator_section():
    config = gen.load_grounding_generator_config(CONFIG_PATH)

    # endpoint/model are environment-specific; only assert they are configured.
    assert isinstance(config["model"]["endpoint"], str) and config["model"]["endpoint"]
    assert isinstance(config["model"]["model"], str) and config["model"]["model"]
    assert config["detection"]["temperature"] == 0.7
    assert config["detection"]["max_tokens"] == 8192
    assert "visual grounding system" in config["detection"]["prompt"]
    # enabled is a user-tunable switch (live parity wants it false), so only
    # assert the type, not a specific value.
    assert isinstance(config["cross_check"]["enabled"], bool)
    assert config["cross_check"]["bbox_coord_space"] == "normalized"
    assert config["coordinate_normalized_range"] == 1000
    assert config["bbox_filters"]["max_area_ratio"] == 0.95


def test_config_expands_env_placeholders(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_URL", "http://localhost:10300/v1")
    monkeypatch.setenv("VLM_MODEL", "Qwen3.6-27B")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "grounding_generator:\n"
        "  model:\n"
        "    endpoint: '${VLM_URL}'\n"
        "    model: '${VLM_MODEL}'\n"
        "  detection:\n"
        "    prompt: 'detect {object}'\n"
        "  cross_check:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )

    config = gen.load_grounding_generator_config(cfg)

    assert config["model"]["endpoint"] == "http://localhost:10300/v1"
    assert config["model"]["model"] == "Qwen3.6-27B"


def test_config_requires_endpoint(monkeypatch, tmp_path):
    monkeypatch.delenv("VLM_URL", raising=False)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "grounding_generator:\n"
        "  model:\n"
        "    endpoint: ''\n"
        "  detection:\n"
        "    prompt: 'detect {object}'\n"
        "  cross_check:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    try:
        gen.load_grounding_generator_config(bad)
    except ValueError as exc:
        assert "endpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing endpoint")


# ----------------------------------------------------------------------------
# Input loading
# ----------------------------------------------------------------------------

def test_load_inputs_txt_tab_separated(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("/tmp/a.jpg\tsrc\tkitchen\n/tmp/b.jpg\n", encoding="utf-8")

    items = gen.load_inputs(path)

    assert items == [
        {"image_path": "/tmp/a.jpg", "scene_type": "kitchen"},
        {"image_path": "/tmp/b.jpg", "scene_type": "unknown"},
    ]


def test_load_inputs_jsonl(tmp_path):
    path = tmp_path / "list.jsonl"
    path.write_text(json.dumps({"image_path": "/tmp/a.jpg", "sample_id": "s1"}) + "\n", encoding="utf-8")

    assert gen.load_inputs(path) == [{"image_path": "/tmp/a.jpg", "sample_id": "s1"}]


# ----------------------------------------------------------------------------
# JSON parsing + bbox denormalization (mirror generator_service)
# ----------------------------------------------------------------------------

def test_parse_json_strips_think_and_fences():
    text = "<think>reasoning</think>\n```json\n[{\"object_name\": \"cup\"}]\n```"
    assert gen.parse_json(text) == [{"object_name": "cup"}]


def test_denormalize_bbox_scales_from_1000_to_pixels():
    box = gen.denormalize_bbox([100, 100, 500, 500], 2000, 2000, norm_range=1000)
    assert box == [200, 200, 1000, 1000]


def test_denormalize_bbox_drops_full_image_box():
    # area ratio > max_area_ratio should be rejected
    assert gen.denormalize_bbox([0, 0, 1000, 1000], 1000, 1000, norm_range=1000) is None


def test_denormalize_bbox_drops_degenerate_box():
    assert gen.denormalize_bbox([500, 500, 100, 100], 1000, 1000) is None


def _base_config(cross_check=True):
    return {
        "worker_count": 1,
        "max_images": 0,
        "coordinate_normalized_range": 1000,
        "default_confidence": 0.5,
        "model": {"endpoint": "mock://x", "api_key_env": "MISSING", "model": "m", "timeout_seconds": 1, "chat_template_kwargs": {}},
        "image": {"max_size": 960, "jpeg_quality": 85},
        "bbox_filters": {"max_area_ratio": 0.95, "min_area_ratio": 0.001},
        "detection": {"temperature": 0.7, "max_tokens": 8192, "retries": 1, "system_prompt": "", "prompt": "detect"},
        "cross_check": {
            "enabled": cross_check,
            "temperature": 0.7,
            "max_tokens": 8192,
            "retries": 1,
            "keep_on_unparsed": True,
            "prompt": 'claim "{object}" at {bbox}',
            "bbox_coord_space": "pixel",
        },
    }


# ----------------------------------------------------------------------------
# Label building
# ----------------------------------------------------------------------------

def test_build_label_matches_production_schema():
    config = _base_config()
    item = {"object_name": "red_cup", "bbox": [100, 100, 300, 300], "confidence": 0.9}
    label = gen.build_label(item, "/data/fixtures/x.jpg", 1000, 1000, "sid", 0, config, scene_type="kitchen")

    assert label["dimension"] == "G_grounding"
    assert label["bucket"] == "G"
    assert label["object_class"] == "red_cup"
    assert label["object_id"] == "sid_obj00"
    assert label["bbox"] == [100, 100, 300, 300]
    assert label["confidence"] == 0.9
    assert label["source"] == "test_fixtures"
    assert label["image_size"] == [1000, 1000]


# ----------------------------------------------------------------------------
# Cross-check decision logic
# ----------------------------------------------------------------------------

def test_cross_check_drops_mismatched_label(monkeypatch):
    config = _base_config()
    label = {"object_class": "red_cup", "bbox": [100, 100, 300, 300], "image_size": [1000, 1000]}

    def fake_chat(*args, **kwargs):
        return {"content": json.dumps({"match": False, "actual_object": "plate", "confidence": 0.8}),
                "raw_response": "", "request_seconds": 0.0, "http_status": 200, "error": None}

    monkeypatch.setattr(gen, "chat_completion_detail", fake_chat)
    result = gen.cross_check_label("/tmp/x.jpg", label, config)
    assert result["match"] is False
    assert result["verified_class"] == "plate"


def test_cross_check_keeps_on_unparsed(monkeypatch):
    config = _base_config()
    label = {"object_class": "red_cup", "bbox": [100, 100, 300, 300], "image_size": [1000, 1000]}

    def fake_chat(*args, **kwargs):
        return {"content": "not json", "raw_response": "not json", "request_seconds": 0.0, "http_status": 200, "error": None}

    monkeypatch.setattr(gen, "chat_completion_detail", fake_chat)
    result = gen.cross_check_label("/tmp/x.jpg", label, config)
    assert result["match"] is True
    assert result["parsed_ok"] is False


def test_cross_check_normalized_space_rescales_bbox(monkeypatch):
    config = _base_config()
    config["cross_check"]["bbox_coord_space"] = "normalized"
    label = {"object_class": "cup", "bbox": [100, 100, 500, 500], "image_size": [2000, 2000]}
    captured = {}

    def fake_chat(model_cfg, image_path, prompt, **kwargs):
        captured["prompt"] = prompt
        return {"content": json.dumps({"match": True}), "raw_response": "", "request_seconds": 0.0, "http_status": 200, "error": None}

    monkeypatch.setattr(gen, "chat_completion_detail", fake_chat)
    gen.cross_check_label("/tmp/x.jpg", label, config)
    # 100/2000*1000 = 50, 500/2000*1000 = 250
    assert "[50,50,250,250]" in captured["prompt"]


# ----------------------------------------------------------------------------
# End-to-end orchestration (mocked VLM)
# ----------------------------------------------------------------------------

def test_generate_one_image_emits_and_drops(monkeypatch, tmp_path):
    config = _base_config()
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"fake")

    monkeypatch.setattr(gen, "get_image_size", lambda p: (1000, 1000))

    def fake_detection(image_path, cfg):
        return {
            "parsed_ok": True,
            "http_status": 200,
            "error": None,
            "request_seconds": 0.01,
            "attempts": 1,
            "raw_response": "[...]",
            "objects": [
                {"object_name": "red_cup", "bbox": [100, 100, 300, 300], "confidence": 0.9},
                {"object_name": "ghost", "bbox": [400, 400, 600, 600], "confidence": 0.6},
            ],
        }

    def fake_cross(image_path, label, cfg):
        match = label["object_class"] != "ghost"
        return {"match": match, "verified_class": "", "confidence": 0.9, "parsed_ok": True,
                "http_status": 200, "error": None, "raw_response": "", "request_seconds": 0.01}

    monkeypatch.setattr(gen, "run_detection", fake_detection)
    monkeypatch.setattr(gen, "cross_check_label", fake_cross)

    result = gen.generate_one_image({"image_path": str(image), "scene_type": "kitchen"}, 0, config)

    assert result["objects_detected"] == 2
    assert result["labels_built"] == 2
    assert result["cross_check_failed"] == 1
    assert result["labels_emitted"] == 1
    assert result["emitted_labels"][0]["object_class"] == "red_cup"


def test_emitted_bbox_stays_pixel_while_cross_check_uses_normalized(monkeypatch, tmp_path):
    config = _base_config()
    config["cross_check"]["bbox_coord_space"] = "normalized"
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"fake")
    captured = {}

    monkeypatch.setattr(gen, "get_image_size", lambda p: (2000, 2000))
    monkeypatch.setattr(gen, "run_detection", lambda image_path, cfg: {
        "parsed_ok": True, "http_status": 200, "error": None, "request_seconds": 0.0,
        "attempts": 1, "raw_response": "[]",
        # 0-1000 detection box -> denormalized to pixels at 2000x2000
        "objects": [{"object_name": "cup", "bbox": [100, 100, 300, 300], "confidence": 0.9}],
    })

    def fake_chat(model_cfg, image_path, prompt, **kwargs):
        captured["prompt"] = prompt
        return {"content": json.dumps({"match": True}), "raw_response": "", "request_seconds": 0.0, "http_status": 200, "error": None}

    monkeypatch.setattr(gen, "chat_completion_detail", fake_chat)

    result = gen.generate_one_image({"image_path": str(image)}, 0, config)

    # Emitted bbox must remain in pixel space (200..600 at 2000px) for the verifier.
    assert result["emitted_labels"][0]["bbox"] == [200, 200, 600, 600]
    # The cross-check prompt must have used 0-1000 normalized coords.
    assert "[100,100,300,300]" in captured["prompt"]


def test_evaluate_grounding_generator_aggregates(monkeypatch, tmp_path):
    config = _base_config()
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"fake")

    monkeypatch.setattr(gen, "get_image_size", lambda p: (1000, 1000))
    monkeypatch.setattr(gen, "run_detection", lambda image_path, cfg: {
        "parsed_ok": True, "http_status": 200, "error": None, "request_seconds": 0.01,
        "attempts": 1, "raw_response": "[]",
        "objects": [{"object_name": "cup", "bbox": [100, 100, 300, 300], "confidence": 0.9}],
    })
    monkeypatch.setattr(gen, "cross_check_label", lambda image_path, label, cfg: {
        "match": True, "verified_class": "", "confidence": 0.9, "parsed_ok": True,
        "http_status": 200, "error": None, "raw_response": "", "request_seconds": 0.01,
    })

    report = gen.evaluate_grounding_generator([{"image_path": str(image)}], config)

    assert report["total_images"] == 1
    assert report["counts"]["objects_detected"] == 1
    assert report["counts"]["labels_emitted"] == 1
    assert report["stage_passrate"]["image_detection_success"]["rate"] == 1.0
    assert report["stage_passrate"]["object_emit"]["rate"] == 1.0
    labels = gen.collect_emitted_labels(report)
    assert len(labels) == 1
    assert labels[0]["dimension"] == "G_grounding"


def test_missing_image_reports_error():
    config = _base_config()
    result = gen.generate_one_image({"image_path": "/no/such/file.jpg"}, 0, config)
    assert result["error"] == "missing_image"
    assert result["labels_emitted"] == 0 if "labels_emitted" in result else result["emitted_labels"] == []
