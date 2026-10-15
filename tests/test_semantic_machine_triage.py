import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/semantic_machine_triage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("semantic_machine_triage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _human_annotation_snapshot():
    snapshot = {}
    pack = ROOT / "data/semantic_audit_v2"
    for annotator in ("annotator_a", "annotator_b"):
        directory = pack / annotator / "annotations"
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path)] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
    return snapshot


def test_prompt_only_renders_40_image_requests_without_touching_human_files(tmp_path):
    before = _human_annotation_snapshot()
    prompt_output = tmp_path / "prompts.jsonl"
    result_output = tmp_path / "results.jsonl"
    env = {
        **os.environ,
        "PROMPT_ONLY": "1",
        "SEMANTIC_TRIAGE_PROMPT_OUT": str(prompt_output),
        "SEMANTIC_TRIAGE_OUT": str(result_output),
    }
    result = subprocess.run(
        ["python", str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert '"api_calls": 0' in result.stdout
    assert not result_output.exists()
    rows = [json.loads(line) for line in prompt_output.read_text().splitlines()]
    assert len(rows) == 40
    assert all(len(row["sample_ids"]) == 8 for row in rows)
    assert all("Return exactly eight judgments" in row["prompt"] for row in rows)
    assert all("annotator_a/labels" in row["source_label_root"] for row in rows)
    assert _human_annotation_snapshot() == before


def test_dry_run_writes_no_artifacts(tmp_path):
    prompt_output = tmp_path / "prompts.jsonl"
    result_output = tmp_path / "results.jsonl"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "SEMANTIC_TRIAGE_PROMPT_OUT": str(prompt_output),
        "SEMANTIC_TRIAGE_OUT": str(result_output),
    }
    result = subprocess.run(
        ["python", str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert not prompt_output.exists()
    assert not result_output.exists()


def test_machine_outputs_are_rejected_inside_human_roots(tmp_path):
    module = _load_module()
    pack = ROOT / "data/semantic_audit_v2"
    forbidden = pack / "annotator_b/annotations/machine.jsonl"
    try:
        module.assert_machine_only_path(forbidden, pack)
    except ValueError as exc:
        assert "must not be inside human root" in str(exc)
    else:
        raise AssertionError("human annotation path was accepted for machine output")
    module.assert_machine_only_path(tmp_path / "machine.jsonl", pack)


def test_live_response_normalizes_to_exact_non_human_schema():
    module = _load_module()
    pack = module.SemanticTriagePack(
        ROOT / "data/semantic_audit_v2",
        ROOT / "configs/prompts/semantic_machine_triage_v1.txt",
    )
    case = pack.cases[0]
    raw = {
        "judgments": [
            {
                "sample_id": case.sample_ids[form],
                "form": form,
                "suggested_correct": True,
                "confidence": 0.75,
                "reason": "visible evidence supports the candidate",
            }
            for form in pack.forms
        ]
    }
    rows = module.normalize_response(raw, case, "test-model")
    assert len(rows) == 8
    assert all(tuple(row) == module.RESULT_FIELDS for row in rows)
    assert all("label_snapshot" not in row and "correct" not in row for row in rows)
