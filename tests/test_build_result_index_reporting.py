from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.build_result_index import audit_claim_evidence, canonical_hash, write_latex


def test_withheld_cell_keeps_zero_credit_diagnostic_visible(tmp_path: Path) -> None:
    index = {
        "benchmarks": [{"id": "bench", "required_n": 10}],
        "runs": [
            {
                "id": "anchor",
                "display_name": "Anchor",
                "include_in_main_table": True,
                "results": {
                    "bench": {
                        "exists": True,
                        "eligible": True,
                        "value": 50.0,
                        "score_if_untrusted": None,
                        "reasons": [],
                    }
                },
            },
            {
                "id": "base",
                "display_name": "Base",
                "include_in_main_table": True,
                "results": {
                    "bench": {
                        "exists": True,
                        "eligible": False,
                        "value": None,
                        "score_if_untrusted": 12.5,
                        "reasons": ["trust_gate_withheld"],
                    }
                },
            }
        ],
    }
    path = tmp_path / "table.tex"

    # Use a real benchmark id because the renderer also checks its family style.
    index["benchmarks"][0]["id"] = "cv_bench"
    for run in index["runs"]:
        run["results"]["cv_bench"] = run["results"].pop("bench")
    write_latex(index, path)

    rendered = path.read_text(encoding="utf-8")
    assert r"\textit{12.50}$^{\dagger}$" in rendered
    assert "withheld" not in rendered


def test_leakage_quarantined_cell_keeps_raw_value_but_is_not_ranked(tmp_path: Path) -> None:
    index = {
        "benchmarks": [{"id": "cv_bench", "required_n": 10}],
        "runs": [
            {
                "display_name": "Anchor",
                "include_in_main_table": True,
                "results": {"cv_bench": {"eligible": True, "value": 50.0}},
            },
            {
                "display_name": "E1",
                "table_group": "posttrained",
                "include_in_main_table": True,
                "results": {
                    "cv_bench": {
                        "exists": True,
                        "eligible": False,
                        "value": 42.5,
                        "reasons": ["verified_training_evaluation_overlap"],
                    }
                },
            },
        ],
    }
    path = tmp_path / "table.tex"
    write_latex(index, path)
    assert r"\textit{42.50}$^{\mathsection}$" in path.read_text(encoding="utf-8")


def test_no_admitted_pretrained_cell_still_renders_without_ranking(tmp_path: Path) -> None:
    index = {"benchmarks": [{"id": "cv_bench", "required_n": 10}], "runs": [{
        "display_name": "Base", "include_in_main_table": True,
        "results": {"cv_bench": {"exists": True, "eligible": False, "value": 50.0,
                                 "reasons": ["evaluation_code_hash_mismatch"]}},
    }]}
    path = tmp_path / "table.tex"
    write_latex(index, path)
    assert "Base & --" in path.read_text()
    assert "\\bestscore{" not in path.read_text()


@pytest.mark.parametrize("contamination,metadata,expected_reason", [
    (None, None, "unresolved_perceptual_training_evaluation_overlap"),
    ("PASS", "INCOMPLETE_EVALUATION_RELEASE_METADATA", "incomplete_evaluation_release_metadata"),
    ("BLOCKED", "INCOMPLETE_EVALUATION_RELEASE_METADATA", "unresolved_perceptual_training_evaluation_overlap"),
])
def test_claim_audit_is_hash_bound_and_fail_closed(tmp_path: Path, contamination, metadata, expected_reason) -> None:
    report = {
        "posttraining_corpus": {"sha256": "c" * 64},
        "per_benchmark": {"cv_bench": {"state": "BLOCKED", "contamination_verdict": contamination,
                                      "metadata_completeness_verdict": metadata}},
    }
    report["identity_sha256"] = canonical_hash(report)
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    run = {
        "claim_evidence_audit": {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "identity_sha256": report["identity_sha256"],
            "corpus_sha256": "c" * 64,
            "cells": {
                "cv_bench": {
                    "report_path": "per_benchmark.cv_bench",
                    "required_state": "PASS",
                }
            },
        }
    }
    reasons, binding = audit_claim_evidence(run, "cv_bench")
    assert reasons == [expected_reason]
    assert binding and binding["state"] == "BLOCKED"
    assert binding["contamination_verdict"] == contamination
    assert binding["metadata_completeness_verdict"] == metadata

    run["claim_evidence_audit"]["sha256"] = "0" * 64
    reasons, _ = audit_claim_evidence(run, "cv_bench")
    assert "claim_audit_sha256_mismatch" in reasons
