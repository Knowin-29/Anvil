from pathlib import Path
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_gsa_4xl20_dlc.sh"
SHORT_SCRIPT = ROOT / "scripts/run_general_retention_v3_dlc.sh"


def test_dlc_orchestrator_shell_and_contract() -> None:
    result = subprocess.run(["bash", "-n", str(SCRIPT)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "PREFLIGHT_ONLY",
        "exactly_4_unique_physical_L20_at_least_44GiB",
        "torch.cuda.device_count()",
        "RUN_STAGE=paired",
        "BATCH=16",
        "IMAGE_MAX_TOKEN_NUM=4096",
        "WANDB_MODE=offline",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "setsid",
        "ACTIVE_PGID",
        "gpu_trace.csv",
        "EXTERNAL_GPU_PROCESS",
        "GPU_MONITOR_QUERY_FAILED",
        "COMPLETE_WITH_BLOCKED",
        "record firstfault BLOCKED",
        "record mopd BLOCKED",
        "BLOCKED_ANNOTATION_GATE",
        "SKIPPED_PENDING_ANNOTATION",
        "FORMAL_PASS preflight failed",
        "RESUME_RUN",
        "v52_formal_compile.json",
        "eval/scannet_rgb_depth_v52_eval.py",
        "eval/scannet_rgb_depth_v52_dlc.py",
        "eval/scannet_rgb_depth_v51_dlc.py",
        "eval/scannet_rgb_depth_v52_interface.py",
        "eval/gsa_eval/runner.py",
        "experiments/scannet_rgb_depth_v52_gs_intervention_contract.json",
        "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh",
    ):
        assert required in text
    general_invocation = next(
        line for line in text.splitlines() if 'scripts/run_general_retention_dp.sh" || exit 6' in line
    )
    assert "|| true" not in general_invocation
    assert re.search(r"RUN_ROOT=.*artifacts/dlc_runs", text)


def test_dlc_orchestrator_does_not_enable_legacy_pipeline() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "run_all_dlc.sh" not in text
    assert "ALLOW_LEGACY_PIPELINE" not in text
    assert "RUN_GRPO" not in text
    assert "RUN_FULL_SFT" not in text
    assert "paper/compile_check.sh" not in text
    assert "run_scannet_rgb_depth_v51_eval_l20_4gpu.sh" not in text


def test_formal_pass_v52_preflight_failure_is_top_level_fatal(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "eval").mkdir(parents=True)
    (project / "data/semantic_audit_v52_scannet_unique_referent_gs_formal").mkdir(
        parents=True
    )
    (project / ".env").write_text("\n", encoding="utf-8")
    orchestrator = project / "scripts/run_gsa_4xl20_dlc.sh"
    orchestrator.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    orchestrator.chmod(0o755)

    evaluator = project / "eval/scannet_rgb_depth_v52_eval.py"
    evaluator.write_text(
        f"""#!{sys.executable}
import json
from pathlib import Path
import sys

args = sys.argv[1:]
if args[0] != "gate":
    raise SystemExit(91)
report = Path(args[args.index("--report") + 1])
report.write_text(json.dumps({{"status": "FORMAL_PASS"}}) + "\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    evaluator.chmod(0o755)

    preflight_calls = project / "launcher.calls"
    gpu_marker = project / "gpu-started"
    launcher = project / "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh"
    launcher.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$PREFLIGHT_ONLY" >> {preflight_calls}
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  exit 23
fi
touch {gpu_marker}
exit 31
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    fake_nvidia = project / "nvidia-smi"
    fake_nvidia.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu*)
    for index in 0 1 2 3; do
      printf '%s, GPU-fixture-%s, NVIDIA L20, 46068, 0, 0\n' "$index" "$index"
    done ;;
  *--query-compute-apps=*) ;;
  *) exit 17 ;;
esac
""",
        encoding="utf-8",
    )
    fake_nvidia.chmod(0o755)

    run_root = project / "run"
    result = subprocess.run(
        ["bash", str(orchestrator)],
        cwd=project,
        env={
            **os.environ,
            "GSA_ROOT": str(project),
            "PYTHON_BIN": sys.executable,
            "EXPECTED_GSA_ENV": sys.prefix,
            "NVIDIA_SMI_BIN": str(fake_nvidia),
            "RUN_ID": "formal_pass_preflight_failure",
            "RUN_ROOT": str(run_root),
            "DLC_RUN_LOCK": str(project / "run.lock"),
            "MIN_DISK_GB": "1",
            "PREFLIGHT_ONLY": "1",
            "RUN_GENERAL_RETENTION": "0",
            "RUN_V52_PAIRED": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 5, (result.stdout, result.stderr)
    status = (run_root / "status.tsv").read_text(encoding="utf-8")
    assert "\tv52_formal\tPASS\t0\t" in status
    assert "\tv52_paired\tFAILED\t23\t" in status
    assert "\tpipeline\tFAILED\t5\t" in status
    assert preflight_calls.read_text(encoding="utf-8").splitlines() == ["1"]
    assert not gpu_marker.exists()
    assert not list(project.rglob("READY.json"))


def test_short_v3_entrypoint_is_pinned_and_scoped() -> None:
    result = subprocess.run(["bash", "-n", str(SHORT_SCRIPT)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    text = SHORT_SCRIPT.read_text(encoding="utf-8")
    for required in (
        "GENERAL_RETENTION_V3_CODE_READY",
        "eval/general_retention_formal_v3.py",
        "configs/eval/general_retention_formal_v3.yaml",
        "scripts/run_general_retention_dp.sh",
        "tools/check_scannet_rgb_depth_v52_unique_referent_formal.py",
        "eval/scannet_rgb_depth_v52_eval.py",
        "eval/scannet_rgb_depth_v52_dlc.py",
        "eval/scannet_rgb_depth_v51_dlc.py",
        "eval/scannet_rgb_depth_v52_interface.py",
        "eval/gsa_eval/runner.py",
        "experiments/scannet_rgb_depth_v52_gs_intervention_contract.json",
        "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh",
        "scripts/run_gsa_4xl20_dlc.sh",
        "eval/compile_iclr_claim_evidence.py",
        "eval/render_iclr_claim_artifacts.py",
        "experiments/claim_evidence_visual_contract.yaml",
        "RUN_GENERAL_RETENTION=1",
        "RUN_V52_PAIRED=auto",
        "TextVQA=1",
        "IMAGE_MAX_TOKEN_NUM=4096",
        "RESUME_RUN",
        "PIN_CHECK_ONLY",
        "ICLR_CLAIM_HANDOFF_READY",
        "EXPERIMENT_SUCCEEDED_EVIDENCE_COMPILATION_FAILED",
        "EXPERIMENT_AND_EVIDENCE_SUCCEEDED_CLAIM_RENDER_FAILED",
        "CANONICAL_RUN_ROOT",
    ):
        assert required in text
    assert "sha256sum" in text
    assert "general_retention_formal_v2.py" not in text
    assert "render_general_retention_paper_values.py" not in text
    assert "paper/" not in text
    assert "iclr_claims_4xl20_$(date -u +%Y%m%d_%H%M%S)" in text

    dp_text = (ROOT / "scripts/run_general_retention_dp.sh").read_text(encoding="utf-8")
    assert "render_general_retention_paper_values.py" not in dp_text
    assert "paper/tables" not in dp_text
    assert "REUSED_VERIFIED" in dp_text
    assert "verify-comparison" in dp_text
    assert "setsid timeout" in dp_text

    pin_check = subprocess.run(
        ["bash", str(SHORT_SCRIPT)],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "PIN_CHECK_ONLY": "1"},
        text=True,
        capture_output=True,
    )
    assert pin_check.returncode == 0, pin_check.stderr
    assert "GENERAL_RETENTION_V3_PIN_CHECK_COMPLETE" in pin_check.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_short_wrapper_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "short-project"
    pinned = (
        "eval/general_retention_formal_v3.py",
        "configs/eval/general_retention_formal_v3.yaml",
        "scripts/run_general_retention_dp.sh",
        "tools/check_scannet_rgb_depth_v52_unique_referent_formal.py",
        "eval/scannet_rgb_depth_v52_eval.py",
        "eval/scannet_rgb_depth_v52_dlc.py",
        "eval/scannet_rgb_depth_v51_dlc.py",
        "eval/scannet_rgb_depth_v52_interface.py",
        "eval/gsa_eval/runner.py",
        "experiments/scannet_rgb_depth_v52_gs_intervention_contract.json",
        "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh",
        "scripts/run_gsa_4xl20_dlc.sh",
        "eval/compile_iclr_claim_evidence.py",
        "eval/render_iclr_claim_artifacts.py",
        "experiments/claim_evidence_visual_contract.yaml",
    )
    for relative in pinned:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    orchestrator = project / "scripts/run_gsa_4xl20_dlc.sh"
    orchestrator.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s %s %s\n' "$RUN_ID" "$PREFLIGHT_ONLY" "$RESUME_RUN" >> "$FAKE_ORCHESTRATOR_CALLS"
mkdir -p "$RUN_ROOT"
printf '{"run_id":"%s","resume_count":%s}\n' "$RUN_ID" "${FAKE_RESUME_COUNT:-0}" > "$RUN_ROOT/run_manifest.json"
if [[ "${FAKE_MUTATE_COMPILER:-0}" == 1 ]]; then
  printf '%s\n' '# post-run mutation' >> "$GSA_ROOT/eval/compile_iclr_claim_evidence.py"
fi
if [[ -n "${FAKE_ORCHESTRATOR_PID_MARKER:-}" ]]; then
  sleep 300 &
  child=$!
  printf '%s %s\n' "$$" "$child" > "$FAKE_ORCHESTRATOR_PID_MARKER"
  trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 143' TERM INT
  wait "$child"
fi
exit "${FAKE_ORCHESTRATOR_RC:-0}"
""",
        encoding="utf-8",
    )
    orchestrator.chmod(0o755)

    compiler = project / "eval/compile_iclr_claim_evidence.py"
    compiler.write_text(
        """#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--markdown", required=True)
args = parser.parse_args()
calls = Path(os.environ["FAKE_COMPILER_CALLS"])
calls.parent.mkdir(parents=True, exist_ok=True)
with calls.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(vars(args), sort_keys=True) + "\\n")
rc = int(os.environ.get("FAKE_COMPILER_RC", "0"))
if rc:
    print(f"fixture compiler rejected rc={rc}")
    raise SystemExit(rc)
evidence = {
    "status": "CLAIM_EVIDENCE_VERIFIED",
    "scannet_v52_paired_gs": {"availability": "unavailable"},
}
canonical = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
evidence["identity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
if os.environ.get("FAKE_BAD_EVIDENCE_IDENTITY") == "1":
    evidence["identity_sha256"] = "0" * 64
Path(args.out).write_text(json.dumps(evidence, sort_keys=True) + "\\n", encoding="utf-8")
if os.environ.get("FAKE_OMIT_MARKDOWN") != "1":
    Path(args.markdown).write_text("# verified fixture\\n", encoding="utf-8")
if os.environ.get("FAKE_MUTATE_RENDERER") == "1":
    renderer = Path(os.environ["GSA_ROOT"]) / "eval/render_iclr_claim_artifacts.py"
    with renderer.open("a", encoding="utf-8") as handle:
        handle.write("# post-compile mutation\\n")
print(json.dumps({"status": "CLAIM_EVIDENCE_VERIFIED", "out": args.out}))
""",
        encoding="utf-8",
    )
    compiler.chmod(0o755)

    renderer = project / "eval/render_iclr_claim_artifacts.py"
    renderer.write_text(
        """#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if "--help" in sys.argv and os.environ.get("FAKE_RENDERER_HELP_RC"):
    raise SystemExit(int(os.environ["FAKE_RENDERER_HELP_RC"]))
parser = argparse.ArgumentParser()
parser.add_argument("--evidence", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
calls = Path(os.environ["FAKE_RENDERER_CALLS"])
calls.parent.mkdir(parents=True, exist_ok=True)
with calls.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(vars(args), sort_keys=True) + "\\n")
rc = int(os.environ.get("FAKE_RENDERER_RC", "0"))
if rc:
    print(f"fixture renderer rejected rc={rc}")
    raise SystemExit(rc)

def binding(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

def relative_binding(path):
    value = binding(path)
    value["path"] = Path(path).name
    return value

output = Path(args.output_dir)
output.mkdir()
names = {
    "general_retention_table": "tab_general_retention.tex",
    "claim_figure_svg": "fig_claim_evidence.svg",
    "claim_figure_pdf": "fig_claim_evidence.pdf",
    "claim_figure_png": "fig_claim_evidence.png",
}
for name in names.values():
    (output / name).write_text(f"fixture {name}\\n", encoding="utf-8")
evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
root = Path(os.environ["GSA_ROOT"])
manifest = {
    "schema_version": 1,
    "protocol": "iclr-claim-artifacts-v3",
    "status": "CLAIM_ARTIFACTS_RENDERED",
    "evidence": {
        **binding(args.evidence),
        "identity_sha256": evidence["identity_sha256"],
    },
    "compiler": binding(root / "eval/compile_iclr_claim_evidence.py"),
    "renderer": binding(Path(__file__)),
    "visual_contract": binding(root / "experiments/claim_evidence_visual_contract.yaml"),
    "v52_availability": evidence["scannet_v52_paired_gs"]["availability"],
    "outputs": {key: relative_binding(output / name) for key, name in names.items()},
    "publication_policy": {
        "paper_was_not_modified": True,
        "missing_values_filled": False,
        "unavailable_v52_table_created": False,
        "editable_vector_output_created": True,
    },
}
canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
manifest["identity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
if os.environ.get("FAKE_BAD_RENDER_MANIFEST") == "1":
    manifest["status"] = "TAMPERED"
if os.environ.get("FAKE_OMIT_RENDER_MANIFEST") != "1":
    (output / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
print(json.dumps({"status": "CLAIM_ARTIFACTS_RENDERED"}))
""",
        encoding="utf-8",
    )
    renderer.chmod(0o755)

    wrapper_text = SHORT_SCRIPT.read_text(encoding="utf-8")
    for relative in pinned:
        production_hash = _sha256(ROOT / relative)
        fixture_hash = _sha256(project / relative)
        assert production_hash in wrapper_text
        wrapper_text = wrapper_text.replace(production_hash, fixture_hash)
    wrapper = project / "scripts/run_general_retention_v3_dlc.sh"
    wrapper.write_text(wrapper_text, encoding="utf-8")
    wrapper.chmod(0o755)
    return project, wrapper


def _fake_short_env(project: Path) -> dict[str, str]:
    return {
        **os.environ,
        "GSA_ROOT": str(project),
        "PYTHON_BIN": sys.executable,
        "FAKE_ORCHESTRATOR_CALLS": str(project / "orchestrator.calls"),
        "FAKE_COMPILER_CALLS": str(project / "compiler.calls"),
        "FAKE_RENDERER_CALLS": str(project / "renderer.calls"),
    }


def test_short_wrapper_auto_compiles_evidence_and_preflight_skips(tmp_path: Path) -> None:
    project, wrapper = _fake_short_wrapper_project(tmp_path)
    env = _fake_short_env(project)
    preflight = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={**env, "RUN_ID": "fixture_preflight", "PREFLIGHT_ONLY": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert "CLAIM_EVIDENCE_SKIPPED" in preflight.stdout
    assert not (project / "compiler.calls").exists()
    assert not (project / "renderer.calls").exists()
    assert not (project / "artifacts/claim_evidence/fixture_preflight").exists()

    completed = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={**env, "RUN_ID": "fixture_complete", "PREFLIGHT_ONLY": "0"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DLC_EXPERIMENT_SUCCEEDED" in completed.stdout
    assert completed.stdout.count("ICLR_CLAIM_HANDOFF_READY:") == 7
    evidence_dirs = list(
        (project / "artifacts/claim_evidence/fixture_complete/resume_0").glob(
            "*_pid_*"
        )
    )
    assert len(evidence_dirs) == 1
    evidence = evidence_dirs[0]
    assert (evidence / "claim_evidence.json").is_file()
    assert (evidence / "claim_evidence.md").is_file()
    assert (evidence / "claim_artifacts/render_manifest.json").is_file()
    assert (evidence / "claim_artifacts/tab_general_retention.tex").is_file()
    assert (evidence / "claim_artifacts/fig_claim_evidence.svg").is_file()
    assert (evidence / "claim_artifacts/fig_claim_evidence.pdf").is_file()
    assert (evidence / "claim_artifacts/fig_claim_evidence.png").is_file()
    assert not (evidence / "claim_artifacts/tab_scannet_v52_paired_gs.tex").exists()
    ready = json.loads((evidence / "READY.json").read_text(encoding="utf-8"))
    assert ready["protocol"] == "iclr-claim-handoff-v2"
    assert ready["status"] == "ICLR_CLAIM_HANDOFF_READY"
    assert ready["run_id"] == "fixture_complete"
    assert ready["resume_count"] == 0
    assert set(ready["outputs"]) == {
        "evidence_json",
        "evidence_markdown",
        "render_manifest",
    }
    assert ready["outputs"]["evidence_json"]["sha256"] == _sha256(
        evidence / "claim_evidence.json"
    )
    assert ready["outputs"]["evidence_markdown"]["sha256"] == _sha256(
        evidence / "claim_evidence.md"
    )
    assert ready["outputs"]["render_manifest"]["sha256"] == _sha256(
        evidence / "claim_artifacts/render_manifest.json"
    )
    assert set(ready["claim_artifacts"]) == {
        "general_retention_table",
        "claim_figure_svg",
        "claim_figure_pdf",
        "claim_figure_png",
    }
    assert "CLAIM_EVIDENCE_VERIFIED" in (evidence / "compile.log").read_text(
        encoding="utf-8"
    )
    calls = (project / "compiler.calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert json.loads(calls[0])["run_root"].endswith(
        "artifacts/dlc_runs/fixture_complete"
    )
    render_calls = (project / "renderer.calls").read_text(encoding="utf-8").splitlines()
    assert len(render_calls) == 1
    assert json.loads(render_calls[0])["evidence"].endswith("claim_evidence.json")

    resumed = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_complete",
            "RESUME_RUN": "1",
            "FAKE_RESUME_COUNT": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert evidence.is_dir()
    resumed_dirs = list(
        (project / "artifacts/claim_evidence/fixture_complete/resume_1").glob(
            "*_pid_*"
        )
    )
    assert len(resumed_dirs) == 1
    assert (resumed_dirs[0] / "READY.json").is_file()
    assert len((project / "compiler.calls").read_text(encoding="utf-8").splitlines()) == 2
    assert len((project / "renderer.calls").read_text(encoding="utf-8").splitlines()) == 2


def test_short_wrapper_propagates_orchestrator_and_compiler_failures(tmp_path: Path) -> None:
    project, wrapper = _fake_short_wrapper_project(tmp_path)
    env = _fake_short_env(project)
    noncanonical = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_noncanonical",
            "RUN_ROOT": str(project / "somewhere-else"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert noncanonical.returncode == 2
    assert "RUN_ROOT must be canonical" in noncanonical.stderr
    assert not (project / "orchestrator.calls").exists()

    renderer_preflight_failed = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_renderer_preflight_failure",
            "FAKE_RENDERER_HELP_RC": "13",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert renderer_preflight_failed.returncode == 2
    assert "renderer dependency/CLI preflight failed with rc=13" in renderer_preflight_failed.stderr
    assert not (project / "orchestrator.calls").exists()
    assert not (project / "compiler.calls").exists()
    assert not (project / "renderer.calls").exists()

    experiment_failed = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_orchestrator_failure",
            "FAKE_ORCHESTRATOR_RC": "6",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert experiment_failed.returncode == 6
    assert "DLC_EXPERIMENT_FAILED" in experiment_failed.stderr
    assert not (project / "compiler.calls").exists()
    assert not (project / "renderer.calls").exists()

    evidence_failed = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_compiler_failure",
            "FAKE_COMPILER_RC": "9",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert evidence_failed.returncode == 9
    assert "DLC_EXPERIMENT_SUCCEEDED" in evidence_failed.stdout
    assert "EXPERIMENT_SUCCEEDED_EVIDENCE_COMPILATION_FAILED" in evidence_failed.stderr
    evidence_dirs = list(
        (project / "artifacts/claim_evidence/fixture_compiler_failure/resume_0").glob(
            "*_pid_*"
        )
    )
    assert len(evidence_dirs) == 1
    assert "fixture compiler rejected rc=9" in (
        evidence_dirs[0] / "compile.log"
    ).read_text(encoding="utf-8")
    assert not (evidence_dirs[0] / "claim_evidence.json").exists()
    assert not (evidence_dirs[0] / "claim_evidence.md").exists()
    assert not (evidence_dirs[0] / "READY.json").exists()
    assert not (project / "renderer.calls").exists()

    pair_failed = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_pair_failure",
            "FAKE_OMIT_MARKDOWN": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert pair_failed.returncode == 7
    assert "compiler omitted JSON or Markdown" in pair_failed.stderr
    partial_dirs = list(
        (project / "artifacts/claim_evidence/fixture_pair_failure/resume_0").glob(
            "*_pid_*"
        )
    )
    assert len(partial_dirs) == 1
    assert (partial_dirs[0] / "claim_evidence.json").is_file()
    assert not (partial_dirs[0] / "claim_evidence.md").exists()
    assert not (partial_dirs[0] / "READY.json").exists()
    assert not (project / "renderer.calls").exists()

    calls_before_mutation = len(
        (project / "compiler.calls").read_text(encoding="utf-8").splitlines()
    )
    compiler_mutated = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_compiler_mutation",
            "FAKE_MUTATE_COMPILER": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert compiler_mutated.returncode == 7
    assert "compiler SHA changed during the run" in compiler_mutated.stderr
    assert len((project / "compiler.calls").read_text(encoding="utf-8").splitlines()) == calls_before_mutation
    assert not (project / "renderer.calls").exists()


def test_short_wrapper_fails_closed_on_render_errors_and_mutation(tmp_path: Path) -> None:
    project, wrapper = _fake_short_wrapper_project(tmp_path)
    env = _fake_short_env(project)

    render_failed = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_renderer_failure",
            "FAKE_RENDERER_RC": "11",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert render_failed.returncode == 11
    assert (
        "EXPERIMENT_AND_EVIDENCE_SUCCEEDED_CLAIM_RENDER_FAILED"
        in render_failed.stderr
    )
    failed_attempts = list(
        (project / "artifacts/claim_evidence/fixture_renderer_failure/resume_0").glob(
            "*_pid_*"
        )
    )
    assert len(failed_attempts) == 1
    assert (failed_attempts[0] / "claim_evidence.json").is_file()
    assert "fixture renderer rejected rc=11" in (
        failed_attempts[0] / "render.log"
    ).read_text(encoding="utf-8")
    assert not (failed_attempts[0] / "READY.json").exists()

    bad_manifest = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_bad_render_manifest",
            "FAKE_BAD_RENDER_MANIFEST": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad_manifest.returncode == 7
    assert "render verification or READY commit failed" in bad_manifest.stderr
    bad_attempts = list(
        (project / "artifacts/claim_evidence/fixture_bad_render_manifest/resume_0").glob(
            "*_pid_*"
        )
    )
    assert len(bad_attempts) == 1
    assert (bad_attempts[0] / "claim_artifacts/render_manifest.json").is_file()
    assert not (bad_attempts[0] / "READY.json").exists()

    bad_evidence_identity = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_bad_evidence_identity",
            "FAKE_BAD_EVIDENCE_IDENTITY": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad_evidence_identity.returncode == 7
    assert "claim evidence identity mismatch" in bad_evidence_identity.stderr
    identity_attempts = list(
        (project / "artifacts/claim_evidence/fixture_bad_evidence_identity/resume_0").glob(
            "*_pid_*"
        )
    )
    assert len(identity_attempts) == 1
    assert not (identity_attempts[0] / "READY.json").exists()

    renderer_calls_before_mutation = len(
        (project / "renderer.calls").read_text(encoding="utf-8").splitlines()
    )
    renderer_mutated = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **env,
            "RUN_ID": "fixture_renderer_mutation",
            "FAKE_MUTATE_RENDERER": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert renderer_mutated.returncode == 7
    assert "renderer SHA changed before render" in renderer_mutated.stderr
    assert len((project / "renderer.calls").read_text(encoding="utf-8").splitlines()) == renderer_calls_before_mutation
    mutated_attempts = list(
        (project / "artifacts/claim_evidence/fixture_renderer_mutation/resume_0").glob(
            "*_pid_*"
        )
    )
    assert len(mutated_attempts) == 1
    assert (mutated_attempts[0] / "claim_evidence.json").is_file()
    assert not (mutated_attempts[0] / "READY.json").exists()


def test_short_wrapper_term_reaches_orchestrator_and_cleans_group(tmp_path: Path) -> None:
    project, wrapper = _fake_short_wrapper_project(tmp_path)
    marker = project / "orchestrator-pids.txt"
    process = subprocess.Popen(
        ["bash", str(wrapper)],
        cwd=project,
        env={
            **_fake_short_env(project),
            "RUN_ID": "fixture_wrapper_signal",
            "FAKE_ORCHESTRATOR_PID_MARKER": str(marker),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 15
    while not marker.is_file() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.is_file(), process.communicate(timeout=5)
    pids = [int(value) for value in marker.read_text(encoding="utf-8").split()]
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 143, (stdout, stderr)
    for pid in pids:
        proc_stat = Path(f"/proc/{pid}/stat")
        assert not proc_stat.exists() or proc_stat.read_text().split()[2] == "Z"
    assert not (project / "compiler.calls").exists()


def test_orchestrator_signal_cleans_owned_general_process_group(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    for directory in (
        scripts,
        project / "eval",
        project / "configs/eval",
        project / "tools",
        project / "base-model",
        project / "e1-model",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (project / ".env").write_text("\n", encoding="utf-8")
    (scripts / "run_gsa_4xl20_dlc.sh").write_text(
        SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for relative in (
        "scripts/run_general_retention_v3_dlc.sh",
        "eval/general_retention_formal_v3.py",
        "configs/eval/general_retention_formal_v3.yaml",
        "tools/check_scannet_rgb_depth_v52_unique_referent_formal.py",
        "eval/scannet_rgb_depth_v52_eval.py",
        "eval/scannet_rgb_depth_v52_dlc.py",
        "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    marker = project / "worker-pids.txt"
    fake_general = scripts / "run_general_retention_dp.sh"
    fake_general.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
sleep 300 &
child=$!
printf '%s %s\n' "$$" "$child" > "$WORKER_PID_MARKER"
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 143' TERM INT
wait "$child"
""",
        encoding="utf-8",
    )
    fake_general.chmod(0o755)

    fake_python = project / "python-shim"
    fake_python.write_text(
        f"""#!{sys.executable}
import subprocess
import sys

args = sys.argv[1:]
payload = sys.stdin.read() if args and args[0] == "-" else None
if payload is not None and "torch.cuda.device_count" in payload:
    print("fixture torch hardware gate: exactly 4 physical L20 GPUs")
    raise SystemExit(0)
completed = subprocess.run([sys.executable, *args], input=payload, text=True)
raise SystemExit(completed.returncode)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_nvidia = project / "nvidia-smi"
    fake_nvidia.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu*)
    for index in 0 1 2 3; do
      printf '%s, GPU-fixture-%s, NVIDIA L20, 46068, 0, 0\n' "$index" "$index"
    done ;;
  *--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu*)
    for index in 0 1 2 3; do
      printf '%s, GPU-fixture-%s, 1024, 45044, 10\n' "$index" "$index"
    done ;;
  *--query-compute-apps=*) ;;
  *) exit 17 ;;
esac
""",
        encoding="utf-8",
    )
    fake_nvidia.chmod(0o755)

    run_root = project / "run"
    process = subprocess.Popen(
        ["bash", str(scripts / "run_gsa_4xl20_dlc.sh")],
        cwd=project,
        env={
            **os.environ,
            "GSA_ROOT": str(project),
            "PYTHON_BIN": str(fake_python),
            "EXPECTED_GSA_ENV": sys.prefix,
            "NVIDIA_SMI_BIN": str(fake_nvidia),
            "BASE_MODEL": str(project / "base-model"),
            "E1_MODEL": str(project / "e1-model"),
            "RUN_ID": "signal_cleanup_fixture",
            "RUN_ROOT": str(run_root),
            "DLC_RUN_LOCK": str(project / "run.lock"),
            "MIN_DISK_GB": "1",
            "PREFLIGHT_ONLY": "0",
            "RUN_GENERAL_RETENTION": "1",
            "RUN_V52_PAIRED": "0",
            "WORKER_PID_MARKER": str(marker),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 15
    while not marker.is_file() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.is_file(), process.communicate(timeout=5)
    worker_pids = [int(value) for value in marker.read_text(encoding="utf-8").split()]
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode != 0, (stdout, stderr)

    for pid in worker_pids:
        proc_stat = Path(f"/proc/{pid}/stat")
        assert not proc_stat.exists() or proc_stat.read_text().split()[2] == "Z"
    status = (run_root / "status.tsv").read_text(encoding="utf-8")
    assert "\tpipeline\tFAILED\t" in status
