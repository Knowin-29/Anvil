import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sft_explicit_missing_inputs_never_use_historical_fallback(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    fallback = tmp_path / "trains/gsa_bench"
    fallback.mkdir(parents=True)
    for name in ("train.jsonl", "val.jsonl"):
        (fallback / name).write_text("{}\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text("#!/bin/sh\nprintf '1\\n'\n")
    python.chmod(0o755)
    environment = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                   "GSA_ROOT": str(tmp_path), "MODEL": str(model), "PREFLIGHT_ONLY": "1",
                   "TRAIN": str(tmp_path / "missing.jsonl"), "VAL": str(fallback / "val.jsonl")}
    result = subprocess.run(["bash", str(ROOT / "scripts/run_sft_9b_lora_l20_dlc.sh")],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "explicit TRAIN is missing or empty" in result.stderr
    environment.update(TRAIN=str(fallback / "train.jsonl"), VAL=str(tmp_path / "missing-val.jsonl"))
    result = subprocess.run(["bash", str(ROOT / "scripts/run_sft_9b_lora_l20_dlc.sh")],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "explicit VAL is missing or empty" in result.stderr


def test_sft_initial_adapter_cannot_silently_resume_another_run(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    run = tmp_path / "run"
    (run / "checkpoint-3").mkdir(parents=True)
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text("#!/bin/sh\nprintf '1\\n'\n")
    python.chmod(0o755)
    environment = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                   "GSA_ROOT": str(tmp_path), "MODEL": str(model), "PREFLIGHT_ONLY": "1",
                   "TRAIN": str(data), "VAL": str(data), "OUT": str(run), "INITIAL_ADAPTER": str(adapter)}
    result = subprocess.run(["bash", str(ROOT / "scripts/run_sft_9b_lora_l20_dlc.sh")],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "INITIAL_ADAPTER cannot be combined" in result.stderr


def test_l20_sft_cadence_is_parameterized_and_compatible_with_best_model_loading():
    text = (ROOT / "scripts/run_sft_9b_lora_l20_dlc.sh").read_text()
    assert 'EVAL_STEPS="${EVAL_STEPS:-750}"' in text
    assert 'SAVE_STEPS="${SAVE_STEPS:-750}"' in text
    assert '--eval_steps "$EVAL_STEPS"' in text
    assert '--save_steps "$SAVE_STEPS"' in text
    assert "SAVE_STEPS % EVAL_STEPS == 0" in text


def test_teacher_cadence_is_parameterized():
    text = (ROOT / "scripts/run_teacher_dim_full_dlc.sh").read_text()
    assert 'EVAL_STEPS="${EVAL_STEPS:-500}"' in text
    assert 'SAVE_STEPS="${SAVE_STEPS:-500}"' in text
    assert '--eval_steps "$EVAL_STEPS"' in text
    assert '--save_steps "$SAVE_STEPS"' in text


def test_pilot_forces_final_step_eval_and_save_for_joint_and_specialists():
    text = (ROOT / "scripts/run_pilot_qualification_l20.sh").read_text()
    assert text.count('EVAL_STEPS="${EVAL_STEPS:-$STEPS}"') == 2
    assert text.count('SAVE_STEPS="${SAVE_STEPS:-$STEPS}"') == 2
    assert text.count('EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS"') == 2


def test_pilot_budget_uses_actual_effective_batch_and_multiple_epochs():
    pilot = (ROOT / "scripts/run_pilot_qualification_l20.sh").read_text()
    sft = (ROOT / "scripts/run_sft_9b_lora_l20_dlc.sh").read_text()
    assert 'BS="${BS:-1}"' in pilot
    assert 'GA="${GA:-1}"' in pilot
    assert "GLOBAL_BATCH=$(( NG * BS * GA ))" in pilot
    formula = "(ROWS * EPOCHS + GLOBAL_BATCH - 1) / GLOBAL_BATCH"
    assert pilot.count(formula) == 2
    assert 'EPOCHS="${EPOCHS:-3}"' in pilot
    assert "S|A) DEFAULT_EPOCHS=5" in pilot
    assert 'GA="${GA:-$(( 32 / (NG * BS) ))}"' in sft


def test_qualified_mopd_wrapper_has_no_historical_fallback_and_requires_eligibility():
    text = (ROOT / "scripts/run_mopd_qualified_dlc.sh").read_text()
    executable = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "_find_sft_adapter" not in executable
    assert "merged_e1" not in executable
    assert 'AUDIT="${SEMANTIC_AUDIT:-$W/artifacts/audits/pilot_semantic_audit_composite.json}"' in text
    assert 'BENCH_DIR="${BENCH_DIR:-$W/trains/gsa_bench_v4_composite_pilot_clean}"' in text
    assert "verify_composite_qualification.py" in text
    assert 'OUT="$W/output/mopd_9b_lora"' in text
    assert "adapter_model.safetensors" in text
    assert "TEACHER_ELIGIBILITY_JSON" in text
    assert 'ALLOW_UNQUALIFIED_TEACHERS:-0' in text


def test_dashboard_mopd_readiness_is_bound_to_eligibility_paths():
    text = (ROOT / "run_all_exp.sh").read_text()
    assert "declared_joint == pathlib.Path(joint).resolve()" in text
    assert '(declared_teachers.get(family) or {}).get("eligible") is True' in text
    assert "== pathlib.Path(teachers[family]).resolve()" in text
    assert "qualification_ready and eligibility_ok" in text


def _fake_qualified_root(tmp_path):
    (tmp_path / "artifacts/audits").mkdir(parents=True)
    (tmp_path / "artifacts/audits/pilot_semantic_audit_composite.json").write_text(
        json.dumps({
            "protocol": "semantic-audit-composite-v3-nondepth-v4-depth",
            "verdict": "PASS",
        }))
    verifier = tmp_path / "scripts/verify_composite_qualification.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("raise SystemExit(0)\n")
    train = tmp_path / "trains/gsa_bench_v4_composite_pilot_clean/train.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text("{}\n")
    selected = {}
    for name in ("pilot_clean_joint", "teacher_lora_G", "teacher_lora_S", "teacher_lora_A"):
        for step in (2, 10):
            checkpoint = tmp_path / "output" / name / "v0-test" / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"test")
        selected[name] = str((tmp_path / "output" / name / "v0-test/checkpoint-10").resolve())
    return selected


def test_qualified_wrapper_blocks_missing_eligibility_manifest(tmp_path):
    _fake_qualified_root(tmp_path)
    env = {**os.environ, "GSA_ROOT": str(tmp_path), "PREFLIGHT_ONLY": "1"}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_mopd_qualified_dlc.sh")],
        cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 3
    assert "teacher eligibility manifest missing" in result.stderr
    assert "Do not treat completed teacher training as qualification" in result.stderr


def test_qualified_wrapper_accepts_only_path_bound_latest_adapters(tmp_path):
    selected = _fake_qualified_root(tmp_path)
    eligibility = {
        "verdict": "PASS",
        "joint_adapter": selected["pilot_clean_joint"],
        "teachers": {
            family: {"eligible": True, "adapter": selected[f"teacher_lora_{family}"]}
            for family in "GSA"
        },
    }
    eligibility_path = tmp_path / "artifacts/audits/teacher_eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility))
    env = {**os.environ, "GSA_ROOT": str(tmp_path), "PREFLIGHT_ONLY": "1"}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_mopd_qualified_dlc.sh")],
        cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "teacher eligibility gate PASS" in result.stdout
    assert "checkpoint-10" in result.stdout
    assert "PREFLIGHT_ONLY=1: not launching training" in result.stdout
