from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_registered_baseline_canary_l20.sh"


def _formal_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("datasets.py", "parsing.py", "registry.py", "runner.py", "scorers.py"):
        path = root / "eval" / "gsa_eval" / name
        digest.update(name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _fake_model(root: Path) -> Path:
    model = root / "checkpoints" / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"model_type": "fake_vlm", "architectures": ["FakeVLM"]}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"fake-weights")
    (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return model


def _fake_swift_repo(root: Path) -> Path:
    repo = root / "ms-swift"
    repo.mkdir()
    (repo / "README.md").write_text("test checkout\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "contact@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return repo


def _write_registry(root: Path, entries: list[dict[str, object]]) -> Path:
    path = root / "configs" / "models" / "baselines.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is a strict YAML subset and keeps the fixture independent of PyYAML.
    path.write_text(json.dumps({"schema_version": 1, "models": entries}), encoding="utf-8")
    return path


def _entry(model: Path, **updates: object) -> dict[str, object]:
    item: dict[str, object] = {
        "id": "local_ok",
        "display_name": "Local test model",
        "source": "local",
        "path": str(model),
        "protocol": "image_text_to_text",
        "harness": "unified_gsa_eval",
        "table_eligible": True,
    }
    item.update(updates)
    return item


def _fake_nvidia_smi(root: Path) -> Path:
    executable = root / "fake_nvidia_smi.py"
    executable.write_text(
        """#!/usr/bin/env python3
import datetime
import json
import os
import sys

args = sys.argv[1:]
selector = args[args.index("-i") + 1]
profiles = json.loads(os.environ["FAKE_GPU_PROFILES"])
if selector not in profiles:
    raise SystemExit(4)
profile = profiles[selector]
query = next(arg for arg in args if arg.startswith("--query-"))
if query == "--query-gpu=index,uuid,name,memory.total,driver_version":
    print(f"{profile.get('index', selector)}, {profile.get('uuid', 'GPU-test')}, "
          f"{profile['name']}, {profile['memory_mib']}, {profile.get('driver', '535.1')}")
elif query == "--query-compute-apps=pid":
    if profile.get("pid") is not None:
        print(profile["pid"])
elif query == "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu":
    now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    print(f"{now}, {profile.get('index', selector)}, {profile['name']}, "
          f"{profile.get('used_mib', 1234)}, {profile['memory_mib']}, {profile.get('util', 17)}")
else:
    raise SystemExit(5)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".env").write_text("# test environment\n", encoding="utf-8")
    eval_dir = root / "eval" / "gsa_eval"
    eval_dir.mkdir(parents=True)
    for name in ("datasets.py", "parsing.py", "registry.py", "runner.py", "scorers.py"):
        (eval_dir / name).write_text(f"# {name}\n", encoding="utf-8")
    (eval_dir / "template_diagnostic.py").write_text(
        "# isolated template diagnostic\n", encoding="utf-8"
    )
    formal_runner = root / "eval" / "run_benchmark.py"
    formal_runner.write_text("# frozen formal runner\n", encoding="utf-8")
    diagnostic_runner = root / "eval" / "run_benchmark_template_diagnostic.py"
    diagnostic_runner.write_text(
        "raise AssertionError('runner must not execute during preflight')\n",
        encoding="utf-8",
    )
    model = _fake_model(root)
    registry = _write_registry(root, [_entry(model)])
    return root, model, registry


def _base_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    root, model, registry = _workspace(tmp_path)
    swift_repo = _fake_swift_repo(root)
    profiles = {
        "3": {
            "index": "3",
            "uuid": "GPU-single-l20",
            "name": "NVIDIA L20",
            "memory_mib": 46068,
            "used_mib": 2048,
        }
    }
    env = {
        **os.environ,
        "GSA_ROOT": str(root),
        "PYTHON_BIN": sys.executable,
        "NVIDIA_SMI_BIN": str(_fake_nvidia_smi(root)),
        "FAKE_GPU_PROFILES": json.dumps(profiles),
        "CUDA_VISIBLE_DEVICES": "3",
        "MS_SWIFT_ROOT": str(swift_repo),
        "MODEL_ID": "local_ok",
        "REGISTRY": str(registry),
        "REQUIRED_FORMAL_EVAL_FINGERPRINT": _formal_fingerprint(root),
        "REQUIRED_FORMAL_ENTRYPOINT_SHA256": hashlib.sha256(
            (root / "eval" / "run_benchmark.py").read_bytes()
        ).hexdigest(),
        "RUNTIME_CACHE_ROOT": str(tmp_path / "runtime-cache"),
        "TAG": "focused_cpu_test",
    }
    return env, root, model, registry


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_launcher_syntax_and_static_contract() -> None:
    syntax = subprocess.run(["bash", "-n", str(LAUNCHER)], text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'MODEL_ID="${MODEL_ID:-}"' in text
    assert "registered-baseline-single-l20-diagnostic-v1" in text
    assert "configs/models/baselines.yaml" in text
    assert "artifacts/diagnostic_runs" in text
    assert "artifacts/results" in text
    assert "evaluation_code_sha256" in text
    assert "formal_evaluation_code_sha256" in text
    assert "run_benchmark_template_diagnostic.py" in text
    assert "formal evaluator drift" in text
    assert "formal entrypoint drift" in text
    assert "git_diff_sha256" in text
    assert "--template-type" in text
    assert "--device-map cuda:0 --no-thinking" in text
    assert "--benchmarks" in text
    assert "gpu_memory_raw.csv" in text
    assert "gpu_memory_summary.json" in text
    assert "run_eval_dp.sh" not in text
    assert "download_baselines.py" in text  # audit producer is documented, never invoked


def test_preflight_resolves_eligible_local_checkpoint_and_pins_runtime(tmp_path: Path) -> None:
    env, root, model, _ = _base_env(tmp_path)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert "REGISTERED_BASELINE_CANARY_PREFLIGHT_READY: no CUDA process was started" in result.stdout
    out = root / "artifacts" / "diagnostic_runs" / "baseline_canaries" / "focused_cpu_test"
    model_manifest = json.loads((out / "model_manifest.json").read_text(encoding="utf-8"))
    runtime = json.loads((out / "runtime_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((out / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert model_manifest["model_id"] == "local_ok"
    assert model_manifest["path"] == str(model.resolve())
    assert model_manifest["weight_file_count"] == 1
    assert model_manifest["download_audit"] is None
    assert model_manifest["template_type"] is None
    assert runtime["evaluator"]["code_sha256"] == protocol["evaluation_code_sha256"]
    assert (
        runtime["evaluator"]["formal_code_sha256"]
        == protocol["formal_evaluation_code_sha256"]
    )
    assert (
        runtime["evaluator"]["formal_entrypoint_sha256"]
        == protocol["formal_entrypoint_sha256"]
    )
    assert runtime["evaluator"]["protocol"] == "gsa-eval-v7-template-diagnostic-v1"
    assert protocol["evaluation_protocol"] == "gsa-eval-v7-template-diagnostic-v1"
    assert runtime["evaluator"]["code_sha256"] != runtime["evaluator"]["formal_code_sha256"]
    assert runtime["evaluator"]["template_type"] is None
    assert runtime["evaluator"]["template_type_source"] == "automatic"
    assert len(runtime["ms_swift"]["git_head"]) == 40
    assert len(runtime["ms_swift"]["git_diff_sha256"]) == 64
    assert runtime["gpu"]["requested_selector"] == "3"
    assert protocol["formal_result_registry_eligible"] is False
    assert protocol["thinking"] is False
    assert protocol["template_type"] is None
    assert protocol["template_type_source"] == "automatic"
    assert protocol["device_map"] == "cuda:0"
    assert not (out / "run.started").exists()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"table_eligible": False}, "not table_eligible"),
        ({"harness": "action_policy_eval"}, "not assigned to unified_gsa_eval"),
        ({"protocol": "image_to_action"}, "incompatible protocol"),
    ],
)
def test_preflight_rejects_registry_ineligible_entries(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    env, _, model, registry = _base_env(tmp_path)
    _write_registry(registry.parents[2], [_entry(model, **updates)])
    result = _run(env)
    assert result.returncode != 0
    assert message in result.stderr


def test_huggingface_entry_requires_full_revision_and_matching_audit(tmp_path: Path) -> None:
    env, root, model, registry = _base_env(tmp_path)
    hf = _entry(
        model,
        id="hf_model",
        source="huggingface",
        repo_id="Org/Model",
        revision="main",
    )
    _write_registry(root, [hf])
    env["MODEL_ID"] = "hf_model"
    result = _run(env)
    assert result.returncode != 0
    assert "full 40-hex pinned revision" in result.stderr

    revision = "a" * 40
    hf["revision"] = revision
    _write_registry(root, [hf])
    result = _run(env)
    assert result.returncode != 0
    assert "pinned download audit is missing" in result.stderr

    audit_dir = root / "artifacts" / "downloads"
    audit_dir.mkdir(parents=True)
    (audit_dir / "hf_model.json").write_text(
        json.dumps(
            {
                "model_id": "hf_model",
                "repo_id": "Org/Model",
                "requested_revision": revision,
                "destination": str(model),
                "downloaded_at_utc": "2026-08-22T00:00:00+00:00",
                "local_bytes": 12,
            }
        ),
        encoding="utf-8",
    )
    result = _run(env)
    assert result.returncode == 0, result.stderr
    out = root / "artifacts" / "diagnostic_runs" / "baseline_canaries" / "focused_cpu_test"
    manifest = json.loads((out / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == revision
    assert len(manifest["download_audit"]["sha256"]) == 64


@pytest.mark.parametrize(
    ("visible", "profile", "message"),
    [
        ("3,1", {"3": {"name": "NVIDIA L20", "memory_mib": 46068}}, "exactly one visible"),
        ("3", {"3": {"name": "NVIDIA A40", "memory_mib": 46068}}, "is not an L20"),
        ("3", {"3": {"name": "NVIDIA L20", "memory_mib": 46068, "pid": 991}}, "already has a compute process"),
    ],
)
def test_preflight_requires_exactly_one_empty_l20(
    tmp_path: Path, visible: str, profile: dict[str, dict[str, object]], message: str
) -> None:
    env, _, _, _ = _base_env(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["FAKE_GPU_PROFILES"] = json.dumps(profile)
    result = _run(env)
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize("relative", ["eval_results/x", "runs/x", "artifacts/results/x"])
def test_output_isolation_rejects_formal_trees(tmp_path: Path, relative: str) -> None:
    env, root, _, _ = _base_env(tmp_path)
    env["OUT"] = str(root / relative)
    result = _run(env)
    assert result.returncode != 0
    assert "OUT must be below" in result.stderr or "may not enter" in result.stderr


def _fake_runner(path: Path) -> Path:
    path.write_text(
        """import json
import os
import sys
import time
from pathlib import Path

args = sys.argv[1:]
log = Path(os.environ["FAKE_CALL_LOG"])
with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"args": args, "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")

def value(flag):
    return args[args.index(flag) + 1]

template_type = value("--template-type") if "--template-type" in args else None

if "--benchmarks" in args:
    start = args.index("--benchmarks") + 1
    benchmarks = []
    for item in args[start:]:
        if item.startswith("--"):
            break
        benchmarks.append(item)
    limit = int(value("--limit"))
    shard_root = Path(value("--shard-dir"))
    status = {}
    for benchmark in benchmarks:
        shard = shard_root / benchmark / "shard_0.jsonl"
        shard.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.dumps({"uid": f"{benchmark}-{i}", "response": "A", "hit_token_cap": False}) for i in range(limit)]
        shard.write_text("\\n".join(rows) + "\\n", encoding="utf-8")
        status[benchmark] = {"status": "ok", "n": limit}
    Path(value("--out")).write_text(json.dumps({"benchmarks": status, "n_failed": 0, "template_type": template_type}) + "\\n", encoding="utf-8")
    time.sleep(0.15)
    raise SystemExit(0)

benchmark = value("--benchmark")
failed = benchmark == os.environ.get("TRUST_FAIL_BENCH")
record = {
    "benchmark": benchmark,
    "parsed": not failed,
    "score": None if failed else 1.0,
    "evaluation_code_sha256": os.environ["PINNED_EVAL_FINGERPRINT"],
    "formal_evaluation_code_sha256": os.environ["PINNED_FORMAL_EVAL_FINGERPRINT"],
    "evaluation_protocol": os.environ["PINNED_EVAL_PROTOCOL"],
    "subset_limit": int(value("--limit")),
    "generation_batch_size": int(value("--batch-size")),
    "subset_seed": int(value("--seed")),
    "thinking": False,
    "template_type": template_type,
}
result = Path(value("--out"))
result.parent.mkdir(parents=True, exist_ok=True)
result.write_text(json.dumps(record) + "\\n", encoding="utf-8")
predictions = Path(value("--dump-predictions"))
predictions.parent.mkdir(parents=True, exist_ok=True)
predictions.write_text("{}\\n", encoding="utf-8")
raise SystemExit(1 if failed else 0)
""",
        encoding="utf-8",
    )
    return path


def test_cpu_fake_run_records_gpu_trace_and_scores_every_shard(tmp_path: Path) -> None:
    env, root, _, _ = _base_env(tmp_path)
    call_log = tmp_path / "calls.jsonl"
    runner = _fake_runner(root / "eval" / "run_benchmark_template_diagnostic.py")
    env.update(
        {
            "PREFLIGHT_ONLY": "0",
            "RUN_BENCHMARK": str(runner),
            "FAKE_CALL_LOG": str(call_log),
            "BENCHMARKS": "cv_bench gsa_bench_test",
            "LIMIT": "2",
            "BATCH": "2",
            "THINKING": "off",
        }
    )
    result = _run(env)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    generation = [call for call in calls if "--benchmarks" in call["args"]]
    scoring = [call for call in calls if "--score-shards" in call["args"]]
    assert len(generation) == 1
    assert len(scoring) == 2
    assert all(call["visible"] == "3" for call in calls)
    args = generation[0]["args"]
    assert args[args.index("--device-map") + 1] == "cuda:0"
    assert args[args.index("--limit") + 1] == "2"
    assert args[args.index("--batch-size") + 1] == "2"
    assert "--no-thinking" in args
    out = root / "artifacts" / "diagnostic_runs" / "baseline_canaries" / "focused_cpu_test"
    summary = json.loads((out / "gpu_memory_summary.json").read_text(encoding="utf-8"))
    assert summary["samples"] >= 1
    assert summary["peak_used_mib"] == 2048
    assert (out / "gpu_memory_raw.csv").is_file()
    assert (out / "run.completed").is_file()
    assert not (out / "run.failed").exists()
    assert len(list((out / "results").glob("*.json"))) == 2


def test_registry_template_is_manifested_and_forwarded_to_every_cli_call(
    tmp_path: Path,
) -> None:
    env, root, model, registry = _base_env(tmp_path)
    _write_registry(
        registry.parents[2],
        [_entry(model, template_type="qwen2_5_vl")],
    )
    call_log = tmp_path / "template_calls.jsonl"
    runner = _fake_runner(root / "eval" / "run_benchmark_template_diagnostic.py")
    env.update(
        {
            "PREFLIGHT_ONLY": "0",
            "RUN_BENCHMARK": str(runner),
            "FAKE_CALL_LOG": str(call_log),
            "BENCHMARKS": "cv_bench gsa_bench_test",
            "LIMIT": "2",
            "BATCH": "2",
        }
    )
    result = _run(env)
    assert result.returncode == 0, result.stderr

    calls = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 3
    for call in calls:
        args = call["args"]
        assert args[args.index("--template-type") + 1] == "qwen2_5_vl"

    out = root / "artifacts" / "diagnostic_runs" / "baseline_canaries" / "focused_cpu_test"
    model_manifest = json.loads((out / "model_manifest.json").read_text(encoding="utf-8"))
    runtime = json.loads((out / "runtime_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((out / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert model_manifest["template_type"] == "qwen2_5_vl"
    assert runtime["evaluator"]["template_type"] == "qwen2_5_vl"
    assert runtime["evaluator"]["template_type_source"] == "model_registry"
    assert protocol["template_type"] == "qwen2_5_vl"
    assert protocol["template_type_source"] == "model_registry"


def test_preflight_rejects_unsafe_registry_template_type(tmp_path: Path) -> None:
    env, _, model, registry = _base_env(tmp_path)
    _write_registry(
        registry.parents[2],
        [_entry(model, template_type="qwen2_5_vl; unsafe")],
    )
    result = _run(env)
    assert result.returncode != 0
    assert "invalid template_type" in result.stderr


def test_trust_gate_nonzero_is_preserved_and_run_is_not_completed(tmp_path: Path) -> None:
    env, root, _, _ = _base_env(tmp_path)
    runner = _fake_runner(root / "eval" / "run_benchmark_template_diagnostic.py")
    env.update(
        {
            "PREFLIGHT_ONLY": "0",
            "RUN_BENCHMARK": str(runner),
            "FAKE_CALL_LOG": str(tmp_path / "calls.jsonl"),
            "BENCHMARKS": "cv_bench gsa_bench_test",
            "LIMIT": "2",
            "BATCH": "1",
            "TRUST_FAIL_BENCH": "gsa_bench_test",
        }
    )
    result = _run(env)
    assert result.returncode != 0
    assert "trust gate withheld" in result.stderr
    out = root / "artifacts" / "diagnostic_runs" / "baseline_canaries" / "focused_cpu_test"
    assert (out / "run.finished").is_file()
    assert (out / "run.failed").is_file()
    assert not (out / "run.completed").exists()
    assert len(list((out / "results").glob("*.json"))) == 2
