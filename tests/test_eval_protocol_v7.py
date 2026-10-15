import hashlib
import json

from eval.build_result_index import audit_result
from eval.gsa_eval.registry import get_spec
from eval.gsa_eval.runner import (
    EVALUATION_PROTOCOL,
    RESULT_SCHEMA,
    _resolve_max_tokens,
)


def valid_record(code_hash: str) -> dict:
    return {
        "schema_version": RESULT_SCHEMA,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "evaluation_code_sha256": code_hash,
        "model": "/models/test-model",
        "thinking": False,
        "subset_seed": 0,
        "subset_limit": 600,
        "generation_batch_size": 16,
        "max_tokens": 128,
        "n_evaluated": 600,
        "parsed": True,
        "score": 50.0,
    }


def test_registry_budget_is_default_and_explicit_override_is_diagnostic():
    spec = get_spec("cv_bench")
    assert _resolve_max_tokens(spec, 0) == 128
    assert _resolve_max_tokens(spec, 64) == 64


def test_wrong_budget_cannot_enter_result_table():
    code_hash = "current-code"
    run = {"tier": "formal", "model_path": "/models/test-model"}
    benchmark = {
        "required_n": 600,
        "required_max_tokens": 128,
        "value_path": "score",
    }
    protocol = {
        "required_schema": 7,
        "name": "gsa-eval-v7",
        "thinking": False,
        "subset_seed": 0,
        "subset_limit": 600,
        "generation_batch_size": 16,
    }

    ok, reasons, value = audit_result(
        valid_record(code_hash), run, benchmark, protocol, code_hash)
    assert ok and not reasons and value == 50.0

    wrong = valid_record(code_hash)
    wrong["max_tokens"] = 2048
    ok, reasons, _ = audit_result(wrong, run, benchmark, protocol, code_hash)
    assert not ok
    assert "max_tokens=2048 (required=128)" in reasons


def test_formal_result_must_match_registered_model_path(tmp_path):
    code_hash = "current-code"
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    run = {"tier": "formal", "model_path": str(model_dir)}
    benchmark = {
        "required_n": 600,
        "required_max_tokens": 128,
        "value_path": "score",
    }
    protocol = {
        "required_schema": 7,
        "name": "gsa-eval-v7",
        "thinking": False,
        "subset_seed": 0,
        "subset_limit": 600,
        "generation_batch_size": 16,
    }

    matching = valid_record(code_hash)
    matching["model"] = str(model_dir / ".." / model_dir.name)
    ok, reasons, value = audit_result(
        matching, run, benchmark, protocol, code_hash)
    assert ok and not reasons and value == 50.0

    mismatching = valid_record(code_hash)
    mismatching["model"] = str(tmp_path / "other-model")
    ok, reasons, _ = audit_result(
        mismatching, run, benchmark, protocol, code_hash)
    assert not ok
    assert "model_path_mismatch" in reasons


def test_formal_result_validates_optional_pinned_download_audit(tmp_path):
    code_hash = "current-code"
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    audit_path = tmp_path / "download.json"
    audit_payload = {
        "schema_version": 2,
        "repo_id": "org/test-model",
        "requested_revision": "a" * 40,
        "resolved_revision": "a" * 40,
        "revision_is_pinned_commit": True,
        "resolved_path": str(model_dir),
        "local_inventory": [
            {"path": "model.safetensors", "size_bytes": 7, "sha256": "b" * 64}
        ],
        "weight_shards": ["model.safetensors"],
    }
    audit_bytes = json.dumps(audit_payload).encode("utf-8")
    audit_path.write_bytes(audit_bytes)
    run = {
        "tier": "formal",
        "model_path": str(model_dir),
        "repo_id": "org/test-model",
        "model_revision": "a" * 40,
        "download_audit": str(audit_path),
        "download_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
    }
    benchmark = {
        "required_n": 600,
        "required_max_tokens": 128,
        "value_path": "score",
    }
    protocol = {
        "required_schema": 7,
        "name": "gsa-eval-v7",
        "thinking": False,
        "subset_seed": 0,
        "subset_limit": 600,
        "generation_batch_size": 16,
    }
    record = valid_record(code_hash)
    record["model"] = str(model_dir)
    ok, reasons, value = audit_result(
        record, run, benchmark, protocol, code_hash)
    assert ok and not reasons and value == 50.0

    run["model_revision"] = "b" * 40
    ok, reasons, _ = audit_result(
        record, run, benchmark, protocol, code_hash)
    assert not ok
    assert "model_requested_revision_mismatch" in reasons
    assert "model_resolved_revision_mismatch" in reasons

    run["model_revision"] = "a" * 40
    run["download_audit_sha256"] = "0" * 64
    ok, reasons, _ = audit_result(
        record, run, benchmark, protocol, code_hash)
    assert not ok
    assert "model_audit_sha256_mismatch" in reasons
