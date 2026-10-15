from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from eval import compile_iclr_claim_evidence as compiler
from eval import render_iclr_claim_artifacts as renderer
from eval import scannet_rgb_depth_v52_interface as v52_interface


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": renderer._file_hash(path)}


def _payload(*, v52_available: bool) -> dict[str, object]:
    datasets = {}
    values = {
        "ai2d_test": (80.0, 78.0, [-3.0, -1.0]),
        "scienceqa_test": (70.0, 72.0, [1.0, 3.0]),
        "textvqa_val": (60.0, 54.0, [-7.0, -5.0]),
    }
    for dataset, n in renderer.EXPECTED_DATASET_COUNTS.items():
        base, e1, ci = values[dataset]
        datasets[dataset] = {
            "n": n,
            "metric": "accuracy",
            "base_score_percent": base,
            "e1_score_percent": e1,
            "e1_minus_base_points": e1 - base,
            "e1_minus_base_points_ci95": ci,
            "retention_ratio_percent": 100.0 * e1 / base,
            "paired_transitions": {"improved": n // 4, "regressed": n // 4, "unchanged": n - 2 * (n // 4)},
            "output_validity": {
                "base_percent": 100.0,
                "e1_percent": 100.0 * (n - 1) / n,
                "base_malformed_count": 0,
                "e1_malformed_count": 1,
                "base_truncation_count": 0,
                "e1_truncation_count": 1,
                "base_health_admitted": True,
                "e1_health_admitted": True,
            },
        }
    if v52_available:
        metrics = {
            "g_valid_pair_coverage": 0.9,
            "g_both_box_iou_at_0_5": 0.75,
            "g_mean_min_pair_iou": 0.66,
            "s_oracle_accuracy": 0.9,
            "s_oracle_parse_coverage": 1.0,
            "s_absent_accuracy": 0.55,
            "s_absent_parse_coverage": 1.0,
            "s_corrupted_accuracy": 0.4,
            "s_corrupted_parse_coverage": 1.0,
            "s_pred_accuracy_given_emitted_request": 0.75,
            "s_pred_parse_coverage_all_scenes": 0.9,
            "s_pred_correct_all_scenes_invalid_g_zero": 0.675,
            "s_pred_strict_e2e_accuracy": 0.6,
            "paired_dependency_gap_oracle_minus_predicted": 0.225,
            "paired_context_gain_oracle_minus_absent": 0.35,
            "paired_context_gain_oracle_minus_corrupted": 0.5,
            "positive_oracle_headroom": 0.225,
            "bounded_oracle_recovery": 0.75,
        }
        v52 = {
            "availability": "available",
            "admission": "FORMAL_SCORE_COMPLETE",
            "protocol": "fixture",
            "records": 40,
            "metrics": metrics,
            "transition_counts": {
                "oracle_correct_to_predicted_correct": 27,
                "oracle_correct_to_predicted_wrong": 9,
                "oracle_wrong_to_predicted_wrong": 4,
            },
            "strict_e2e_transition_counts": {
                "oracle_correct_to_e2e_correct": 24,
                "oracle_correct_to_e2e_wrong": 12,
                "oracle_wrong_to_e2e_wrong": 4,
            },
            "scene_bootstrap": {
                "metric": "paired_dependency_gap_oracle_minus_predicted",
                "unit": "scene",
                "paired_scenes": 40,
                "samples": 2000,
                "seed": 20260824,
                "confidence": 0.95,
                "ci": [0.1, 0.35],
            },
            "context_control_bootstrap": {
                "oracle_minus_absent": {
                    "metric": "paired_context_gain_oracle_minus_absent",
                    "unit": "scene",
                    "paired_scenes": 40,
                    "samples": 2000,
                    "seed": 20260824,
                    "confidence": 0.95,
                    "ci": [0.2, 0.5],
                },
                "oracle_minus_corrupted": {
                    "metric": "paired_context_gain_oracle_minus_corrupted",
                    "unit": "scene",
                    "paired_scenes": 40,
                    "samples": 2000,
                    "seed": 20260824,
                    "confidence": 0.95,
                    "ci": [0.35, 0.65],
                },
            },
            "parse_status": {
                "G": {"parsed_exact": 36, "invalid": 4},
                "S_oracle": {"parsed_exact": 40},
                "S_absent": {"parsed_exact": 40},
                "S_corrupted": {"parsed_exact": 40},
                "S_pred": {"parsed_exact": 36, "no_request_due_to_invalid_g": 4},
            },
            "token_cap_counts": {
                "G": 0,
                "S_oracle": 0,
                "S_absent": 0,
                "S_corrupted": 0,
                "S_pred": 0,
            },
            "attribution": {
                "states": ["Attributed(G)", "Local(S)", "Inconclusive"],
                "counts": {"Attributed(G)": 5, "Local(S)": 3, "Inconclusive": 32},
                "reason_counts": {
                    "failed_G_and_oracle_repairs_S": 5,
                    "correct_G_but_S_fails_under_oracle_context": 3,
                    "no_downstream_failure": 32,
                },
                "conclusive_failures": 8,
                "policy": _binding(
                    renderer.ROOT
                    / "experiments/scannet_rgb_depth_v52_gs_intervention_contract.json"
                ),
            },
        }
    else:
        v52 = {
            "availability": "unavailable",
            "status_state": "SKIPPED_PENDING_ANNOTATION",
            "reason": "compile_status=PENDING_HUMAN_REVIEW",
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": compiler.EVIDENCE_PROTOCOL,
        "status": "CLAIM_EVIDENCE_VERIFIED",
        "generated_utc": "2026-08-27T00:00:00Z",
        "run_root": "/fixture/run",
        "compiler": _binding(Path(compiler.__file__)),
        "run_envelope": {"run_id": "fixture"},
        "general_retention": {
            "availability": "available",
            "admission": "PAIRED_GENERAL_RETENTION_ADMITTED",
            "protocol": "fixture",
            "records": sum(renderer.EXPECTED_DATASET_COUNTS.values()),
            "dataset_counts": dict(renderer.EXPECTED_DATASET_COUNTS),
            "datasets": datasets,
            "macro": {
                "base_general_avg_percent": 70.0,
                "e1_general_avg_percent": 68.0,
                "e1_minus_base_points": -2.0,
                "e1_minus_base_points_ci95": [-3.0, -1.0],
            },
        },
        "scannet_v52_paired_gs": v52,
        "publication_policy": {
            "paper_was_not_modified": True,
            "console_logs_used_as_evidence": False,
            "missing_values_filled": False,
            "only_deeply_recomputed_values_exported": True,
        },
    }
    payload["identity_sha256"] = compiler.content_hash(payload)
    return payload


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rehash(payload: dict[str, object]) -> None:
    payload.pop("identity_sha256", None)
    payload["identity_sha256"] = compiler.content_hash(payload)


@pytest.mark.parametrize("v52_available", [False, True])
def test_render_bundle_creates_only_evidence_backed_outputs(tmp_path: Path, v52_available: bool) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "rendered"
    _write(evidence, _payload(v52_available=v52_available))
    manifest = renderer.render_bundle(evidence, output)
    assert manifest["status"] == "CLAIM_ARTIFACTS_RENDERED"
    assert (output / "tab_general_retention.tex").is_file()
    assert (output / "fig_claim_evidence.svg").is_file()
    assert (output / "fig_claim_evidence.pdf").is_file()
    assert (output / "fig_claim_evidence.png").is_file()
    assert (output / "render_manifest.json").is_file()
    paired = output / "tab_scannet_v52_paired_gs.tex"
    assert paired.exists() is v52_available
    assert manifest["v52_availability"] == ("available" if v52_available else "unavailable")
    assert "pending" not in (output / "tab_general_retention.tex").read_text(encoding="utf-8").lower()
    assert "(pp)" in (output / "tab_general_retention.tex").read_text(encoding="utf-8")
    svg = (output / "fig_claim_evidence.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert ("Paired G-to-S" in svg) is v52_available


def test_tampered_identity_fails_before_output(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "rendered"
    payload = _payload(v52_available=False)
    payload["general_retention"]["macro"]["e1_minus_base_points"] = 99.0  # type: ignore[index]
    _write(evidence, payload)
    with pytest.raises(renderer.RenderError, match="identity mismatch"):
        renderer.render_bundle(evidence, output)
    assert not output.exists()


def test_missing_value_fails_closed(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "rendered"
    payload = _payload(v52_available=True)
    payload["scannet_v52_paired_gs"]["metrics"]["s_pred_strict_e2e_accuracy"] = None  # type: ignore[index]
    payload["identity_sha256"] = compiler.content_hash({k: v for k, v in payload.items() if k != "identity_sha256"})
    _write(evidence, payload)
    with pytest.raises(renderer.RenderError, match="must be numeric"):
        renderer.render_bundle(evidence, output)
    assert not output.exists()


def test_general_validity_must_match_bounded_counts(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    payload = _payload(v52_available=False)
    validity = payload["general_retention"]["datasets"]["ai2d_test"]["output_validity"]  # type: ignore[index]
    validity["base_percent"] = 12.3
    validity["base_truncation_count"] = 999_999
    _rehash(payload)
    _write(evidence, payload)
    with pytest.raises(renderer.RenderError, match="count exceeds n"):
        renderer.verify_evidence(evidence)


def test_mathematically_undefined_boundary_metrics_render_as_na(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "rendered"
    payload = _payload(v52_available=True)
    general = payload["general_retention"]  # type: ignore[assignment]
    ai2d = general["datasets"]["ai2d_test"]  # type: ignore[index]
    ai2d.update(
        {
            "base_score_percent": 0.0,
            "e1_score_percent": 10.0,
            "e1_minus_base_points": 10.0,
            "e1_minus_base_points_ci95": [9.0, 11.0],
            "retention_ratio_percent": None,
        }
    )
    general["macro"].update(  # type: ignore[index]
        {
            "base_general_avg_percent": (0.0 + 70.0 + 60.0) / 3,
            "e1_general_avg_percent": (10.0 + 72.0 + 54.0) / 3,
            "e1_minus_base_points": 2.0,
            "e1_minus_base_points_ci95": [1.0, 3.0],
        }
    )
    v52 = payload["scannet_v52_paired_gs"]  # type: ignore[assignment]
    v52["metrics"].update(  # type: ignore[index]
        {
            "g_valid_pair_coverage": 0.0,
            "g_both_box_iou_at_0_5": 0.0,
            "g_mean_min_pair_iou": 0.0,
            "s_oracle_accuracy": 0.0,
            "s_oracle_parse_coverage": 1.0,
            "paired_context_gain_oracle_minus_absent": -0.55,
            "paired_context_gain_oracle_minus_corrupted": -0.4,
            "s_pred_accuracy_given_emitted_request": None,
            "s_pred_parse_coverage_all_scenes": 0.0,
            "s_pred_correct_all_scenes_invalid_g_zero": 0.0,
            "s_pred_strict_e2e_accuracy": 0.0,
            "paired_dependency_gap_oracle_minus_predicted": 0.0,
            "positive_oracle_headroom": 0.0,
            "bounded_oracle_recovery": None,
        }
    )
    v52["transition_counts"] = {"oracle_wrong_to_predicted_wrong": 40}  # type: ignore[index]
    v52["strict_e2e_transition_counts"] = {  # type: ignore[index]
        "oracle_wrong_to_e2e_wrong": 40
    }
    v52["scene_bootstrap"]["ci"] = [0.0, 0.0]  # type: ignore[index]
    v52["parse_status"] = {  # type: ignore[index]
        "G": {"invalid": 40},
        "S_oracle": {"parsed_exact": 40},
        "S_absent": {"parsed_exact": 40},
        "S_corrupted": {"parsed_exact": 40},
        "S_pred": {"no_request_due_to_invalid_g": 40},
    }
    v52["attribution"] = {  # type: ignore[index]
        **v52["attribution"],
        "counts": {"Attributed(G)": 0, "Local(S)": 0, "Inconclusive": 40},
        "reason_counts": {"invalid_or_capped_or_unparsed_output": 40},
        "conclusive_failures": 0,
    }
    _rehash(payload)
    _write(evidence, payload)
    renderer.render_bundle(evidence, output)
    table = (output / "tab_scannet_v52_paired_gs.tex").read_text(encoding="utf-8")
    assert table.count(r"\textit{n/a}") == 1


def test_transition_metric_contradiction_fails_closed(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    payload = _payload(v52_available=True)
    payload["scannet_v52_paired_gs"]["transition_counts"] = {  # type: ignore[index]
        "oracle_wrong_to_predicted_wrong": 40
    }
    _rehash(payload)
    _write(evidence, payload)
    with pytest.raises(renderer.RenderError, match="transitions disagree"):
        renderer.verify_evidence(evidence)


def test_local_s_cannot_exceed_oracle_context_failures(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    payload = _payload(v52_available=True)
    attribution = payload["scannet_v52_paired_gs"]["attribution"]  # type: ignore[index]
    attribution["counts"] = {  # type: ignore[index]
        "Attributed(G)": 5,
        "Local(S)": 5,
        "Inconclusive": 30,
    }
    attribution["reason_counts"] = {  # type: ignore[index]
        "failed_G_and_oracle_repairs_S": 5,
        "correct_G_but_S_fails_under_oracle_context": 5,
        "no_downstream_failure": 30,
    }
    attribution["conclusive_failures"] = 10  # type: ignore[index]
    _rehash(payload)
    _write(evidence, payload)
    with pytest.raises(renderer.RenderError, match=r"Local\(S\) exceeds oracle-context"):
        renderer.verify_evidence(evidence)


def test_v52_success_status_matches_the_frozen_interface() -> None:
    grounding = v52_interface.parse_grounding_response(
        '{"A": [10, 20, 110, 220], "B": [300, 320, 500, 620]}'
    )
    spatial = v52_interface.parse_spatial_response("A")
    assert grounding.valid and grounding.status == "parsed_exact"
    assert spatial.valid and spatial.status == "parsed_exact"
    renderer._verify_v52(_payload(v52_available=True)["scannet_v52_paired_gs"])


def test_percentile_ci_need_not_contain_point_estimate(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "rendered"
    payload = _payload(v52_available=True)
    payload["general_retention"]["datasets"]["ai2d_test"][  # type: ignore[index]
        "e1_minus_base_points_ci95"
    ] = [-1.0, 0.0]
    payload["scannet_v52_paired_gs"]["scene_bootstrap"]["ci"] = [0.31, 0.45]  # type: ignore[index]
    _rehash(payload)
    _write(evidence, payload)
    renderer.render_bundle(evidence, output)
    assert (output / "fig_claim_evidence.pdf").is_file()


def test_v52_axis_reserves_space_for_100_percent_labels() -> None:
    payload = _payload(v52_available=True)
    v52 = payload["scannet_v52_paired_gs"]  # type: ignore[assignment]
    v52["metrics"].update(  # type: ignore[index]
        {
            "g_valid_pair_coverage": 1.0,
            "g_both_box_iou_at_0_5": 1.0,
            "g_mean_min_pair_iou": 1.0,
            "s_oracle_accuracy": 1.0,
            "s_oracle_parse_coverage": 1.0,
            "s_pred_accuracy_given_emitted_request": 1.0,
            "s_pred_parse_coverage_all_scenes": 1.0,
            "s_pred_correct_all_scenes_invalid_g_zero": 1.0,
            "s_pred_strict_e2e_accuracy": 1.0,
            "paired_dependency_gap_oracle_minus_predicted": 0.0,
            "paired_context_gain_oracle_minus_absent": 0.45,
            "paired_context_gain_oracle_minus_corrupted": 0.6,
            "positive_oracle_headroom": 0.0,
            "bounded_oracle_recovery": 1.0,
        }
    )
    v52["transition_counts"] = {"oracle_correct_to_predicted_correct": 40}  # type: ignore[index]
    v52["strict_e2e_transition_counts"] = {  # type: ignore[index]
        "oracle_correct_to_e2e_correct": 40
    }
    v52["scene_bootstrap"]["ci"] = [0.0, 0.0]  # type: ignore[index]
    v52["parse_status"] = {  # type: ignore[index]
        "G": {"parsed_exact": 40},
        "S_oracle": {"parsed_exact": 40},
        "S_absent": {"parsed_exact": 40},
        "S_corrupted": {"parsed_exact": 40},
        "S_pred": {"parsed_exact": 40},
    }
    v52["attribution"] = {  # type: ignore[index]
        **v52["attribution"],
        "counts": {"Attributed(G)": 0, "Local(S)": 0, "Inconclusive": 40},
        "reason_counts": {"no_downstream_failure": 40},
        "conclusive_failures": 0,
    }
    _rehash(payload)
    renderer._verify_v52(v52)
    figure, axes = renderer.plt.subplots(
        1,
        2,
        figsize=(5.5, 2.85),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
    )
    renderer._draw_general(axes[0], payload)
    axis = axes[1]
    renderer._draw_v52(axis, payload)
    figure.subplots_adjust(left=0.15, right=0.985, bottom=0.24, top=0.88, wspace=0.69)
    figure.canvas.draw()
    assert axis.get_xlim()[1] >= 112.0
    assert sum(text.get_text() == "100.0" for text in axis.texts) == 3
    canvas = figure.bbox
    for artist in [axis.title, *axis.texts]:
        bounds = artist.get_window_extent(figure.canvas.get_renderer())
        assert bounds.x0 >= canvas.x0 and bounds.x1 <= canvas.x1
        assert bounds.y0 >= canvas.y0 and bounds.y1 <= canvas.y1
    renderer.plt.close(figure)


def test_evidence_swap_during_render_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "rendered"
    _write(evidence, _payload(v52_available=True))
    original = renderer.render_figure

    def render_then_swap(payload: dict[str, object], prefix: Path) -> list[Path]:
        outputs = original(payload, prefix)
        replacement = _payload(v52_available=False)
        replacement["generated_utc"] = "2026-08-27T00:00:01Z"
        _rehash(replacement)
        _write(evidence, replacement)
        return outputs

    monkeypatch.setattr(renderer, "render_figure", render_then_swap)
    with pytest.raises(renderer.RenderError, match="changed during rendering"):
        renderer.render_bundle(evidence, output)
    assert not output.exists()


def test_refuses_overwrite_and_paper_path(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    _write(evidence, _payload(v52_available=False))
    output = tmp_path / "rendered"
    renderer.render_bundle(evidence, output)
    with pytest.raises(renderer.RenderError, match="overwrite"):
        renderer.render_bundle(evidence, output)
    paper_output = renderer.ROOT / "paper" / "forbidden_claim_render_test"
    with pytest.raises(renderer.RenderError, match="paper"):
        renderer.render_bundle(evidence, paper_output)
    assert not paper_output.exists()


def test_same_evidence_renders_byte_identically(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    _write(evidence, _payload(v52_available=True))
    first = tmp_path / "first"
    second = tmp_path / "second"
    renderer.render_bundle(evidence, first)
    renderer.render_bundle(evidence, second)
    first_files = sorted(path.name for path in first.iterdir())
    second_files = sorted(path.name for path in second.iterdir())
    assert first_files == second_files
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable")
def test_generated_tables_fit_the_iclr_text_width(tmp_path: Path) -> None:
    payload = _payload(v52_available=True)
    source = "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage{booktabs}",
            r"\setlength{\textwidth}{5.5in}",
            r"\begin{document}",
            renderer.render_general_table(payload),
            r"\bigskip",
            renderer.render_v52_table(payload),
            r"\end{document}",
        ]
    )
    tex = tmp_path / "tables.tex"
    tex.write_text(source, encoding="utf-8")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (tmp_path / "tables.log").read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Overfull \\hbox" not in log


def test_cli_reports_rejection_without_partial_directory(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "rendered"
    _write(evidence, {"status": "not-evidence"})
    assert renderer.main(["--evidence", str(evidence), "--output-dir", str(output)]) == 2
    assert not output.exists()
