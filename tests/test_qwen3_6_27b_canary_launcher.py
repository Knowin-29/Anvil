from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_qwen3_6_27b_dual_l20_canary.sh"


def _fake_model(tmp_path: Path) -> Path:
    model = tmp_path / "qwen3.6-27b"
    model.mkdir()
    shards = [f"model-{index:05d}-of-00015.safetensors" for index in range(1, 16)]
    for index, name in enumerate(shards):
        (model / name).write_bytes(f"fake-shard-{index}".encode())
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"weight.{index}": name for index, name in enumerate(shards)}}),
        encoding="utf-8",
    )
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    (model / "LICENSE").write_text("test license\n", encoding="utf-8")
    (model / "README.md").write_text("# test model\n", encoding="utf-8")
    return model


def _fake_nvidia_smi(tmp_path: Path) -> Path:
    executable = tmp_path / "fake_nvidia_smi.py"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
selector = args[args.index("-i") + 1]
profiles = json.loads(os.environ["FAKE_GPU_PROFILES"])
if selector not in profiles:
    raise SystemExit(4)
query = next(arg for arg in args if arg.startswith("--query-"))
if query == "--query-gpu=name,memory.total":
    profile = profiles[selector]
    print(f"{profile['name']}, {profile['memory_mib']}")
elif query == "--query-compute-apps=pid":
    pid = profiles[selector].get("pid")
    if pid is not None:
        print(pid)
else:
    raise SystemExit(5)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _base_env(
    tmp_path: Path,
    model: Path,
    gpu_profiles: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, str], Path]:
    profiles = gpu_profiles or {
        "3": {"name": "NVIDIA L20", "memory_mib": 46068},
        "1": {"name": "NVIDIA L20", "memory_mib": 46068},
    }
    out = tmp_path / "diagnostic-output"
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "NVIDIA_SMI_BIN": str(_fake_nvidia_smi(tmp_path)),
        "FAKE_GPU_PROFILES": json.dumps(profiles),
        "CUDA_VISIBLE_DEVICES": ",".join(profiles),
        "MODEL": str(model),
        "OUT": str(out),
        "TAG": "focused_cpu_test",
    }
    return env, out


def test_launcher_is_syntax_valid_and_encodes_isolated_contract() -> None:
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-1}"' in text
    assert '[[ -r "$W/.env" ]]' in text
    assert 'qwen3.6-27b-dual-l20-diagnostic-canary-v3' in text
    assert "/path/to/environment/hf_models/Qwen/Qwen3.6-27B" in text
    assert 'LIMIT="${LIMIT:-8}"' in text
    assert 'BATCH="${BATCH:-1}"' in text
    assert "expected exactly 15 canonical weight shards" in text
    assert 'CUDA_VISIBLE_DEVICES="$UPSTREAM_CUDA_VISIBLE_DEVICES" timeout' in text
    assert "--device-map auto" in text
    assert "run_eval_dp.sh" not in text
    command_lines = [
        line for line in text.splitlines()
        if "--no-thinking" in line and not line.lstrip().startswith("#")
    ]
    assert len(command_lines) == 2
    assert "--thinking" not in text
    assert "experiments/result_registry.yaml" in text


def test_default_cpu_preflight_checks_model_and_two_l20s_without_runner(tmp_path: Path) -> None:
    model = _fake_model(tmp_path)
    env, out = _base_env(tmp_path, model)
    env["RUN_BENCHMARK"] = str(tmp_path / "must_not_run.py")
    Path(env["RUN_BENCHMARK"]).write_text(
        "raise AssertionError('preflight launched the benchmark runner')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "QWEN3_6_27B_CANARY_PREFLIGHT_READY: no CUDA process was started" in result.stdout
    inventory = json.loads((out / "model_inventory.json").read_text(encoding="utf-8"))
    protocol = json.loads((out / "diagnostic_protocol.json").read_text(encoding="utf-8"))
    assert inventory["weight_shard_count"] == 15
    assert set(inventory["required_metadata"]) == {
        "model.safetensors.index.json",
        "config.json",
        "LICENSE",
        "README.md",
    }
    assert protocol["tier"] == "diagnostic"
    assert protocol["formal_result_registry_eligible"] is False
    assert protocol["subset_limit"] == 8
    assert protocol["generation_batch_size"] == 1
    assert protocol["thinking_control"] == "enable_thinking_false_with_empty_think_prefix"
    assert protocol["process_count"] == 1
    assert protocol["device_map"] == "auto"
    assert protocol["cuda_visible_devices"] == "3,1"
    assert protocol["runtime_cache_root"].startswith("/tmp/")
    assert not (out / "run.started").exists()
    assert not (out / "shards").exists()


def test_batch16_cpu_preflight_records_dynamic_diagnostic_contract(tmp_path: Path) -> None:
    model = _fake_model(tmp_path)
    env, out = _base_env(tmp_path, model)
    env.update(
        {
            "LIMIT": "16",
            "BATCH": "16",
            "BENCHMARKS": "cv_bench gsa_bench_test",
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    protocol = json.loads((out / "diagnostic_protocol.json").read_text(encoding="utf-8"))
    assert protocol["protocol"] == "qwen3.6-27b-dual-l20-diagnostic-canary-v3"
    assert protocol["benchmarks"] == ["cv_bench", "gsa_bench_test"]
    assert protocol["subset_limit"] == 16
    assert protocol["generation_batch_size"] == 16
    reasons = protocol["exclusion_reasons"]
    assert "seeded limit-16 diagnostic subset rather than a full split" in reasons
    assert all("batch size 1" not in reason for reason in reasons)
    assert all("differs from frozen formal batch size" not in reason for reason in reasons)
    assert not (out / "run.started").exists()


@pytest.mark.parametrize(
    ("profiles", "expected_error"),
    [
        ({"0": {"name": "NVIDIA L20", "memory_mib": 46068}}, "exactly two visible GPU selectors"),
        (
            {
                "0": {"name": "NVIDIA A40", "memory_mib": 46068},
                "1": {"name": "NVIDIA L20", "memory_mib": 46068},
            },
            "is not an L20",
        ),
        (
            {
                "0": {"name": "NVIDIA L20", "memory_mib": 45055},
                "1": {"name": "NVIDIA L20", "memory_mib": 46068},
            },
            "at least 45056 MiB is required",
        ),
    ],
)
def test_cpu_preflight_rejects_invalid_visible_hardware(
    tmp_path: Path,
    profiles: dict[str, dict[str, object]],
    expected_error: str,
) -> None:
    model = _fake_model(tmp_path)
    env, _ = _base_env(tmp_path, model, profiles)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert expected_error in result.stderr


def test_cpu_preflight_rejects_incomplete_fifteen_shard_inventory(tmp_path: Path) -> None:
    model = _fake_model(tmp_path)
    (model / "model-00015-of-00015.safetensors").unlink()
    env, _ = _base_env(tmp_path, model)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "expected exactly 15 canonical weight shards" in result.stderr


def test_cpu_preflight_rejects_wrong_model_identity(tmp_path: Path) -> None:
    model = _fake_model(tmp_path)
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_type"] = "other_model"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env, _ = _base_env(tmp_path, model)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected model_type" in result.stderr


def test_cpu_fake_run_uses_one_process_auto_placement_and_preserves_slice(tmp_path: Path) -> None:
    model = _fake_model(tmp_path)
    env, out = _base_env(tmp_path, model)
    call_log = tmp_path / "runner_calls.jsonl"
    runner = tmp_path / "fake_run_benchmark.py"
    runner.write_text(
        """import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_CALL_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"args": args, "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")

def value(flag):
    return args[args.index(flag) + 1]

if "--benchmarks" in args:
    start = args.index("--benchmarks") + 1
    benchmarks = []
    for item in args[start:]:
        if item.startswith("--"):
            break
        benchmarks.append(item)
    shard_dir = Path(value("--shard-dir"))
    for benchmark in benchmarks:
        path = shard_dir / benchmark / "shard_0.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"uid":"fake","response":"A"}\\n', encoding="utf-8")
    out = Path(value("--out"))
    out.write_text(json.dumps({"n_failed": 0}) + "\\n", encoding="utf-8")
else:
    benchmark = value("--benchmark")
    out = Path(value("--out"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmark": benchmark, "parsed": True, "score": 1.0}) + "\\n", encoding="utf-8")
    predictions = Path(value("--dump-predictions"))
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text('{"uid":"fake"}\\n', encoding="utf-8")
""",
        encoding="utf-8",
    )
    env.update(
        {
            "PREFLIGHT_ONLY": "0",
            "RUN_BENCHMARK": str(runner),
            "FAKE_CALL_LOG": str(call_log),
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    generation = [call for call in calls if "--benchmarks" in call["args"]]
    scoring = [call for call in calls if "--score-shards" in call["args"]]
    assert len(generation) == 1
    assert len(scoring) == 5
    args = generation[0]["args"]
    assert generation[0]["visible"] == "3,1"
    assert args[args.index("--device-map") + 1] == "auto"
    assert args[args.index("--limit") + 1] == "8"
    assert args[args.index("--batch-size") + 1] == "1"
    assert "--thinking" not in args
    assert "--no-thinking" in args
    assert all("--no-thinking" in call["args"] for call in scoring)
    assert (out / "run.completed").is_file()
    assert len(list((out / "results").glob("*.json"))) == 5
