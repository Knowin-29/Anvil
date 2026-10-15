"""Tests for the independent multi-model specialist voting verifier."""

import importlib.util
import io
import json
import sys
import threading
import time
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


specialist_voting = _load_script_module("evaluate_specialist_voting")


def _label():
    return {
        "sample_id": "case_1",
        "image_path": "/tmp/image.jpg",
        "object_class": "red_cup",
        "bbox": [100, 100, 300, 300],
        "image_size": [1000, 1000],
    }




def _detail(values, raw="raw", seconds=0.01):
    return {
        "values": values,
        "raw_response": raw,
        "request_seconds": seconds,
        "encode_seconds": 0.0,
        "total_seconds": seconds,
        "http_status": 200,
        "error": None,
    }


def _config(model_count=1, vote_policy="majority"):
    return {
        "vote_policy": vote_policy,
        "max_models": 3,
        "iou_threshold": 0.4,
        "enable_point_fallback": True,
        "point_margin_1000": 50,
        "timeout_seconds": 1,
        "worker_count": 1,
        "bbox_prompt_index": 0,
        "pointing_prompt_index": 0,
        "bbox_prompts": [
            "bbox prompt zero {object} {bbox}",
            "bbox prompt one {object} {bbox}",
        ],
        "pointing_prompts": [
            "point prompt zero {object} {bbox}",
            "point prompt one {object} {bbox}",
        ],
        "models": [
            {
                "name": f"model_{idx}",
                "endpoint": "mock://local",
                "api_key_env": "MISSING_KEY",
                "model": f"mock-{idx}",
                "enabled": True,
            }
            for idx in range(model_count)
        ],
    }


def test_load_labels_reads_jsonl(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(_label()) + "\n", encoding="utf-8")

    assert specialist_voting.load_labels(path) == [_label()]


def test_shared_config_loads_qwen_and_volc_models():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "grounding_verifier_rules.yaml"

    config = specialist_voting.load_specialist_voting_config(config_path)

    names = [model["name"] for model in config["models"]]
    assert "qwen3_6_27b" in names
    assert "volc_llm" in names
    qwen = next(model for model in config["models"] if model["name"] == "qwen3_6_27b")
    volc = next(model for model in config["models"] if model["name"] == "volc_llm")
    assert qwen["chat_template_kwargs"] == {"enable_thinking": False}
    assert volc["api_key_env"] == "VOLC_API_KEY"
    assert config["vote_policy"] == "any"
    assert config["worker_count"] >= 2
    assert len(config["bbox_prompts"]) == 30
    assert len(config["pointing_prompts"]) == 30


def test_render_prompt_uses_object_and_normalized_bbox():
    prompt = specialist_voting.render_prompt(
        "Locate {object} in {bbox}",
        "red cup",
        [100.0, 120.5, 300.25, 400.0],
    )

    assert prompt == "Locate red cup in [100,121,300,400]"


def test_prompt_indices_select_bbox_and_point_prompts(monkeypatch):
    prompts = []

    def fake_chat(model_cfg, image_path, prompt, timeout):
        prompts.append(prompt)
        return "[100,100,300,300]" if len(prompts) == 1 else "(200,200)"

    config = _config(model_count=1)
    config["bbox_prompt_index"] = 1
    config["pointing_prompt_index"] = 1
    monkeypatch.setattr(specialist_voting, "_chat_completion", fake_chat)

    specialist_voting.query_model_for_bbox(
        config["models"][0],
        "/tmp/image.jpg",
        "red_cup",
        [100, 100, 300, 300],
        1,
        config,
    )
    specialist_voting.query_model_for_point(
        config["models"][0],
        "/tmp/image.jpg",
        "red_cup",
        [100, 100, 300, 300],
        1,
        config,
    )

    assert prompts[0].startswith("bbox prompt one")
    assert prompts[1].startswith("point prompt one")


def test_prompt_sweep_runs_all_bbox_prompt_indices(monkeypatch):
    seen_indices = []

    def fake_evaluate(labels, config):
        seen_indices.append(config["bbox_prompt_index"])
        return {
            "overall": {"verified_auto": {"rate": 1.0}},
            "stage_passrate": {},
            "items": [{"sample_id": "case_1", "tag": "verified_auto"}],
        }

    monkeypatch.setattr(specialist_voting, "evaluate_specialist_voting", fake_evaluate)

    result = specialist_voting.run_prompt_sweep([_label()], _config(model_count=1), "bbox")

    assert seen_indices == [0, 1]
    assert len(result["results"]) == 2
    assert result["results"][0]["prompt_id"] == "bbox_0"
    assert result["results"][0]["prompt_type"] == "bbox"
    assert result["results"][0]["tested_prompt_text"] == "bbox prompt zero {object} {bbox}"
    assert result["results"][0]["report"]["items"][0]["sample_id"] == "case_1"


def test_chat_completion_uses_per_model_payload_options(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "[100,100,300,300]"}}]}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(specialist_voting, "encode_image_data_url", lambda image_path: "data:image/jpeg;base64,abc")
    monkeypatch.setattr(specialist_voting.requests, "post", fake_post)
    monkeypatch.setenv("VOLC_API_KEY", "secret-token")

    text = specialist_voting._chat_completion(
        {
            "endpoint": "https://example.com/v1",
            "api_key_env": "VOLC_API_KEY",
            "model": "LLM",
            "system_prompt": "You are a helpful AI assistant",
            "temperature": 0.6,
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "/tmp/image.jpg",
        "detect",
        12,
    )

    assert text == "[100,100,300,300]"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["temperature"] == 0.6
    assert captured["json"]["max_tokens"] == 8192
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_single_model_iou_passes(monkeypatch):
    monkeypatch.setattr(
        specialist_voting,
        "query_model_for_bbox_detail",
        lambda *args, **kwargs: _detail([[100, 100, 300, 300]], "[100,100,300,300]"),
    )

    report = specialist_voting.evaluate_specialist_voting([_label()], _config(model_count=1))

    assert report["overall"]["verified_auto"]["count"] == 1
    assert report["stage_passrate"]["bbox_iou_agree"]["passed"] == 1
    assert report["items"][0]["agree_votes"] == 1
    assert report["prompt_pair"] == {
        "bbox_prompt_index": 0,
        "pointing_prompt_index": 0,
        "bbox_prompt_number": 1,
        "pointing_prompt_number": 1,
        "label": "1-1",
        "python_index_label": "0-0",
    }
    result = report["items"][0]["model_results"][0]
    assert result["bbox_used"]
    assert result["bbox_result"] == [[100, 100, 300, 300]]
    assert not result["pointing_used"]
    assert result["pointing_result"] == []
    assert result["pointing_agree"] is None


def test_three_model_majority_passes_with_two_votes(monkeypatch):
    responses = {
        "model_0": [[100, 100, 300, 300]],
        "model_1": [[110, 110, 290, 290]],
        "model_2": [[600, 600, 800, 800]],
    }

    monkeypatch.setattr(
        specialist_voting,
        "query_model_for_bbox_detail",
        lambda model_cfg, *args, **kwargs: _detail(responses[model_cfg["name"]], str(responses[model_cfg["name"]])),
    )

    report = specialist_voting.evaluate_specialist_voting([_label()], _config(model_count=3))

    assert report["items"][0]["votes_required"] == 2
    assert report["items"][0]["agree_votes"] == 2
    assert report["stage_passrate"]["vote_pass"]["passed"] == 1


def test_point_fallback_can_pass_when_bbox_missing(monkeypatch):
    monkeypatch.setattr(specialist_voting, "query_model_for_bbox_detail", lambda *args, **kwargs: _detail([], "None"))
    monkeypatch.setattr(specialist_voting, "query_model_for_point_detail", lambda *args, **kwargs: _detail([[200, 200]], "(200,200)"))

    report = specialist_voting.evaluate_specialist_voting([_label()], _config(model_count=1))

    result = report["items"][0]["model_results"][0]
    assert result["agree"]
    assert result["method"] == "point_in_box"
    assert not result["bbox_used"]
    assert result["bbox_result"] == []
    assert result["bbox_agree"] is None
    assert result["pointing_used"]
    assert result["pointing_result"] == [[200, 200]]
    assert result["pointing_agree"]
    assert report["stage_passrate"]["point_fallback_agree"]["passed"] == 1


def test_invalid_bbox_fails_before_model_call(monkeypatch):
    calls = []
    bad_label = {**_label(), "bbox": [300, 100, 100, 300]}
    monkeypatch.setattr(specialist_voting, "query_model_for_bbox_detail", lambda *args, **kwargs: calls.append(True))

    report = specialist_voting.evaluate_specialist_voting([bad_label], _config(model_count=1))

    assert calls == []
    assert report["stage_passrate"]["input_valid"]["passed"] == 0
    assert report["overall"]["rejected"]["count"] == 1


def test_query_detail_records_raw_response_and_request_timing(monkeypatch):
    captured = {}

    def fake_chat_detail(model_cfg, image_path, prompt, timeout):
        captured["prompt"] = prompt
        return {
            "content": "[100,100,300,300]",
            "raw_response": "[100,100,300,300]",
            "request_seconds": 0.25,
            "http_status": 200,
            "error": None,
        }

    config = _config(model_count=1)
    monkeypatch.setattr(specialist_voting, "_chat_completion_detail", fake_chat_detail)

    detail = specialist_voting.query_model_for_bbox_detail(
        config["models"][0],
        "/tmp/image.jpg",
        "red_cup",
        [100, 100, 300, 300],
        1,
        config,
    )

    assert captured["prompt"].startswith("bbox prompt zero")
    assert detail["values"] == [[100, 100, 300, 300]]
    assert detail["raw_response"] == "[100,100,300,300]"
    assert detail["request_seconds"] == 0.25
    assert detail["http_status"] == 200
    assert detail["error"] is None


def test_report_includes_raw_model_outputs_and_timing(monkeypatch):
    def fake_bbox_detail(model_cfg, *args, **kwargs):
        return {
            "values": [[100, 100, 300, 300]],
            "raw_response": f"raw bbox from {model_cfg['name']}",
            "request_seconds": 0.4,
            "http_status": 200,
            "error": None,
        }

    monkeypatch.setattr(specialist_voting, "query_model_for_bbox_detail", fake_bbox_detail)

    report = specialist_voting.evaluate_specialist_voting([_label()], _config(model_count=1))

    result = report["items"][0]["model_results"][0]
    assert result["raw_response"] == "raw bbox from model_0"
    assert result["bbox_raw_response"] == "raw bbox from model_0"
    assert result["bbox_request_seconds"] == 0.4
    assert result["request_seconds"] == 0.4
    assert result["http_status"] == 200
    assert report["timing"]["total_seconds"] >= 0
    assert report["timing"]["model_request_count"] == 1
    assert report["timing"]["model_request_total_seconds"] == 0.4
    assert report["timing"]["by_model"]["model_0"]["total_seconds"] == 0.4


def test_worker_count_processes_labels_concurrently(monkeypatch):
    running = 0
    max_running = 0
    lock = threading.Lock()

    def fake_evaluate_one_label(label, config):
        nonlocal running, max_running
        with lock:
            running += 1
            max_running = max(max_running, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return {
            "sample_id": label["sample_id"],
            "image_path": label["image_path"],
            "object_class": label["object_class"],
            "bbox": label["bbox"],
            "input_valid": True,
            "model_results": [],
            "votes_required": 1,
            "agree_votes": 0,
            "tag": "needs_human_review",
            "reason": "mock",
            "item_seconds": 0.05,
        }

    labels = [{**_label(), "sample_id": f"case_{idx}"} for idx in range(4)]
    config = _config(model_count=1)
    config["worker_count"] = 4
    monkeypatch.setattr(specialist_voting, "evaluate_one_label", fake_evaluate_one_label)

    report = specialist_voting.evaluate_specialist_voting(labels, config)

    assert max_running > 1
    assert report["worker_count"] == 4
    assert [item["sample_id"] for item in report["items"]] == ["case_0", "case_1", "case_2", "case_3"]


def test_progress_output_reports_completion(monkeypatch):
    labels = [{**_label(), "sample_id": f"case_{idx}"} for idx in range(3)]

    def fake_evaluate_one_label(label, config):
        return {
            "sample_id": label["sample_id"],
            "image_path": label["image_path"],
            "object_class": label["object_class"],
            "bbox": label["bbox"],
            "input_valid": True,
            "model_results": [],
            "votes_required": 1,
            "agree_votes": 0,
            "tag": "needs_human_review",
            "reason": "mock",
            "item_seconds": 0.0,
        }

    config = _config(model_count=1)
    stream = io.StringIO()
    monkeypatch.setattr(specialist_voting, "evaluate_one_label", fake_evaluate_one_label)

    specialist_voting.evaluate_specialist_voting(labels, config, show_progress=True, progress_stream=stream)

    assert "Specialist voting" in stream.getvalue()
    assert "3/3" in stream.getvalue()


def test_original_prompts_can_be_selected_and_reported(monkeypatch):
    prompts = []

    def fake_chat_detail(model_cfg, image_path, prompt, timeout):
        prompts.append(prompt)
        content = "[100,100,300,300]" if prompt.startswith("Detect") else "(200,200)"
        return {
            "content": content,
            "raw_response": content,
            "request_seconds": 0.01,
            "encode_seconds": 0.0,
            "total_seconds": 0.01,
            "http_status": 200,
            "error": None,
        }

    config = _config(model_count=1, vote_policy="any")
    config["bbox_prompt_index"] = "orig"
    config["pointing_prompt_index"] = "orig"
    monkeypatch.setattr(specialist_voting, "_chat_completion_detail", fake_chat_detail)

    bbox_detail = specialist_voting.query_model_for_bbox_detail(
        config["models"][0],
        "/tmp/image.jpg",
        "red_cup",
        [100, 100, 300, 300],
        1,
        config,
    )
    point_detail = specialist_voting.query_model_for_point_detail(
        config["models"][0],
        "/tmp/image.jpg",
        "red_cup",
        [100, 100, 300, 300],
        1,
        config,
    )
    report = specialist_voting.evaluate_specialist_voting([_label()], config)

    assert prompts[0] == specialist_voting.ORIGINAL_BBOX_PROMPT.format(object="red_cup")
    assert prompts[1] == specialist_voting.ORIGINAL_POINTING_PROMPT.format(object="red_cup")
    assert bbox_detail["values"] == [[100, 100, 300, 300]]
    assert point_detail["values"] == [[200, 200]]
    assert report["bbox_prompt_index"] == "orig"
    assert report["pointing_prompt_index"] == "orig"
    assert report["prompt_pair"] == {
        "bbox_prompt_index": "orig",
        "pointing_prompt_index": "orig",
        "bbox_prompt_number": "orig",
        "pointing_prompt_number": "orig",
        "label": "orig-orig",
        "python_index_label": "orig-orig",
    }
