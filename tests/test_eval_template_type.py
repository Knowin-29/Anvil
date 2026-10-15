from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from eval import run_benchmark as formal_cli
from eval import run_benchmark_template_diagnostic as diagnostic_cli
from eval.gsa_eval import runner as formal
from eval.gsa_eval import template_diagnostic


# Frozen September 14 public-benchmark expansion (five-module fingerprint).
FORMAL_FINGERPRINT = "38af902acaa80c47e370bb73f81424729e428689a15fa51de9be13a0bbd06ead"


@pytest.fixture(autouse=True)
def restore_process_local_adapters():
    original_cli_functions = (
        formal_cli.generate_shard_multi,
        formal_cli.evaluate,
        formal_cli.score_shards,
    )
    yield
    formal._engine = template_diagnostic._FORMAL_ENGINE
    formal._ENGINE_CACHE.clear()
    template_diagnostic._template_type = None
    template_diagnostic._model_type = None
    (
        formal_cli.generate_shard_multi,
        formal_cli.evaluate,
        formal_cli.score_shards,
    ) = original_cli_functions


def test_formal_evaluator_source_identity_is_unchanged() -> None:
    assert formal.evaluation_code_sha256() == FORMAL_FINGERPRINT
    formal_entrypoint = Path(formal_cli.__file__)
    assert "--template-type" not in formal_entrypoint.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("template_type", "expected_in_kwargs"),
    [("qwen2_5_vl", True), (None, False)],
)
def test_isolated_engine_forwards_only_explicit_template_type(
    monkeypatch: pytest.MonkeyPatch,
    template_type: str | None,
    expected_in_kwargs: bool,
) -> None:
    calls: list[tuple[str, int, dict[str, object]]] = []

    class FakeEngine:
        def __init__(self, model: str, max_batch_size: int, **kwargs: object):
            calls.append((model, max_batch_size, kwargs))

    swift = types.ModuleType("swift")
    swift.RequestConfig = object
    infer_engine = types.ModuleType("swift.infer_engine")
    infer_engine.TransformersEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.infer_engine", infer_engine)

    template_diagnostic.configure_template_type(template_type)
    formal._engine("/models/test", 16, device_map="cuda:0", thinking=None)

    assert calls[0][0:2] == ("/models/test", 16)
    kwargs = calls[0][2]
    assert kwargs["device_map"] == "cuda:0"
    assert ("template_type" in kwargs) is expected_in_kwargs
    if template_type is not None:
        assert kwargs["template_type"] == template_type


def test_isolated_result_stamp_binds_both_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(formal, "score_shards", lambda *args, **kwargs: {"score": 1.0})
    template_diagnostic.configure_template_type("qwen2_5_vl")
    record = template_diagnostic.score_shards("bench", "model", [])
    assert record["evaluation_protocol"] == template_diagnostic.DIAGNOSTIC_PROTOCOL
    assert record["evaluation_code_sha256"] == template_diagnostic.diagnostic_code_sha256()
    assert record["formal_evaluation_code_sha256"] == FORMAL_FINGERPRINT
    assert record["template_type"] == "qwen2_5_vl"
    assert record["diagnostic_isolated"] is True


@pytest.mark.parametrize("template_type", ["qwen2_5_vl", None])
def test_isolated_cli_passes_optional_template_without_changing_formal_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    template_type: str | None,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate(benchmarks, model, **kwargs):
        captured.update({"benchmarks": benchmarks, "model": model, **kwargs})
        return {"n_failed": 0, "template_type": template_type}

    monkeypatch.setattr(template_diagnostic, "generate_shard_multi", fake_generate)
    out = tmp_path / "worker.json"
    argv = [
        "run_benchmark_template_diagnostic.py",
        "--model",
        "/models/test",
        "--benchmarks",
        "cv_bench",
        "--out",
        str(out),
        "--shard-dir",
        str(tmp_path / "shards"),
        "--no-thinking",
    ]
    if template_type is not None:
        argv.extend(["--template-type", template_type])
    monkeypatch.setattr(sys, "argv", argv)

    assert diagnostic_cli.main() == 0
    assert json.loads(out.read_text(encoding="utf-8"))["template_type"] == template_type
    assert template_diagnostic._template_type == template_type


def test_isolated_cli_rejects_unsafe_template_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmark_template_diagnostic.py", "--template-type", "bad;value"],
    )
    with pytest.raises(SystemExit, match="unsafe template_type"):
        diagnostic_cli.main()


def test_explicit_model_type_is_forwarded_and_stamped(monkeypatch):
    captured = {}
    class FakeEngine:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)
    swift = types.ModuleType("swift")
    swift.RequestConfig = object
    infer_engine = types.ModuleType("swift.infer_engine")
    infer_engine.TransformersEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.infer_engine", infer_engine)
    template_diagnostic.configure_template_type("qwen3_vl", "qwen3_vl")
    formal._engine("/models/specialist", 4)
    assert captured["model_type"] == "qwen3_vl"
    assert template_diagnostic._stamp({})["model_type_override"] == "qwen3_vl"


def test_model_option_preserves_template_option():
    value, remaining = diagnostic_cli._extract_template_type(
        ["cli", "--model-type=qwen3_vl", "--template-type", "qwen3_vl"], "--model-type")
    assert value == "qwen3_vl"
    assert remaining == ["cli", "--template-type", "qwen3_vl"]
    with pytest.raises(ValueError, match="unsafe model_type"):
        template_diagnostic.configure_template_type(None, "bad;value")
