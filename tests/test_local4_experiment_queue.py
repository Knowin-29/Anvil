from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_queue_contains_only_evidence_authorized_training() -> None:
    text = (ROOT / "scripts/run_local4_experiment_queue.sh").read_text()
    assert "run_g_family_qualification_4gpu_queue.sh" in text
    assert "run_pilot_qualification_l20.sh" not in text
    assert "run_mopd" not in text
    assert "run_teacher_dim" not in text
    assert "run_grpo" not in text


def test_queue_preregisters_primary_and_conditional_replication() -> None:
    text = (ROOT / "scripts/run_local4_experiment_queue.sh").read_text()
    assert "g_seed42_train_qualification" in text
    assert 'decision=$(g_decision 42)' in text
    assert "G_PILOT_DIRECTIONAL_CONTINUE" in text
    assert "for seed in 43 44" in text
    assert "g_replication SKIPPED" in text


def test_queue_covers_missing_full_split_baselines() -> None:
    text = (ROOT / "scripts/run_local4_experiment_queue.sh").read_text()
    assert "gsa_joint_sft_e1_full_local4_v8" in text
    assert "qwen2_5_vl_7b_full_local4_v8" in text
    assert "qwen2_5_vl_3b_full_local4_v8" in text
    assert "run_fresh_full_eval" in text
    assert text.count("NUM_GPUS=4") >= 2
    assert "qwen3_5_9b_base_full" not in text


def test_queue_waits_for_idle_four_l20s_and_uses_project_storage() -> None:
    text = (ROOT / "scripts/run_local4_experiment_queue.sh").read_text()
    assert "wait_for_four_idle_l20s" in text
    assert "torch.cuda.device_count() == 4" in text
    assert "--query-compute-apps=pid" in text
    assert 'TMPDIR="$W/.cache/tmp"' in text
    assert "flock -n" in text
    assert "209715200" in text
    assert 'check_storage || {' in text


def test_queue_is_run_scoped_and_does_not_reuse_model_outputs() -> None:
    text = (ROOT / "scripts/run_local4_experiment_queue.sh").read_text()
    assert 'RUN_TOKEN=' in text
    assert 'TRAIN_OUT="$seed42_train"' in text
    assert 'EVAL_OUT="$seed42_eval"' in text
    assert 'MERGED_DIR="$merged"' in text
    assert "refusing to reuse G full-eval output" in text
    assert "refusing to reuse baseline output" in text


def test_queue_propagates_infrastructure_failures_and_cleans_process_groups() -> None:
    text = (ROOT / "scripts/run_local4_experiment_queue.sh").read_text()
    assert 'INFRA_FAILURES=0' in text
    assert "COMPLETE_WITH_FAILURES" in text
    assert 'setsid "$@"' in text
    assert 'kill -TERM -- "-${CURRENT_PGID:-$CURRENT_PID}"' in text
    assert 'queue.pid' in text
    assert 'CURRENT_JOB_FILE' in text


def test_queue_validates_the_scientific_gate_binding() -> None:
    text = (ROOT / "scripts/run_local4_experiment_queue.sh").read_text()
    assert "qualification report identity drift" in text
    assert "qualification receipt topology/seed drift" in text
    assert "qualification report is bound to a different training run" in text
    assert "INVALID_REPORT" in text
