#!/usr/bin/env python3
"""Compile claim-ready evidence from one completed 4xL20 DLC run.

This is a fail-closed, read-only consumer of formal run artifacts.  It never
parses console logs and never edits paper sources.  General-retention scores are
recomputed from all eight bound shards, while a completed v5.2 result is
recomputed from its verified G, S-oracle, S-absent, S-corrupted, and S-pred
bundles.  Missing evidence
is reported as unavailable only when the orchestrator status explicitly says
that v5.2 was skipped or pending.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import general_retention_formal_v3 as general  # noqa: E402
from eval import scannet_rgb_depth_v52_dlc as v52_dlc  # noqa: E402
from eval import scannet_rgb_depth_v52_eval as v52_eval  # noqa: E402


EVIDENCE_PROTOCOL = "iclr-claim-evidence-v3"
EXPECTED_TRACKED_CODE = (
    "configs/eval/general_retention_formal_v3.yaml",
    "eval/general_retention_formal_v3.py",
    "eval/gsa_eval/runner.py",
    "eval/scannet_rgb_depth_v51_dlc.py",
    "eval/scannet_rgb_depth_v52_dlc.py",
    "eval/scannet_rgb_depth_v52_eval.py",
    "eval/scannet_rgb_depth_v52_interface.py",
    "experiments/scannet_rgb_depth_v52_gs_intervention_contract.json",
    "scripts/run_general_retention_dp.sh",
    "scripts/run_general_retention_v3_dlc.sh",
    "scripts/run_gsa_4xl20_dlc.sh",
    "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh",
    "tools/check_scannet_rgb_depth_v52_unique_referent_formal.py",
)
EXPECTED_DATASET_COUNTS = {
    "ai2d_test": 3088,
    "scienceqa_test": 2017,
    "textvqa_val": 5000,
}
STATUS_HEADER = ("timestamp_utc", "stage", "state", "rc", "detail")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
CI_SEED = 20260827
CI_SAMPLES = 2000


class EvidenceError(ValueError):
    """Raised when a run cannot support publication-facing claims."""


@dataclass(frozen=True)
class StatusRow:
    timestamp_utc: str
    stage: str
    state: str
    rc: str
    detail: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise EvidenceError(f"missing evidence file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise EvidenceError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid {label}: {path} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object: {path}")
    return value


def _require_within(path: Path, parent: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    parent = parent.expanduser().resolve()
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise EvidenceError(f"{label} escapes its run directory: {path}") from exc
    return path


def _parse_status(path: Path) -> list[StatusRow]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise EvidenceError(f"missing orchestrator status: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise EvidenceError("orchestrator status is empty") from exc
        if header != STATUS_HEADER:
            raise EvidenceError("orchestrator status schema drift")
        rows: list[StatusRow] = []
        for line_number, values in enumerate(reader, 2):
            if len(values) != len(STATUS_HEADER):
                raise EvidenceError(f"status.tsv:{line_number}: expected five fields")
            row = StatusRow(*values)
            if not TIMESTAMP_RE.fullmatch(row.timestamp_utc):
                raise EvidenceError(f"status.tsv:{line_number}: timestamp drift")
            if not row.stage or not row.state or "\n" in row.detail or "\r" in row.detail:
                raise EvidenceError(f"status.tsv:{line_number}: invalid status field")
            if row.rc != "-":
                try:
                    int(row.rc)
                except ValueError as exc:
                    raise EvidenceError(f"status.tsv:{line_number}: invalid return code") from exc
            rows.append(row)
    if not rows:
        raise EvidenceError("orchestrator status has no events")
    return rows


def _latest(rows: Iterable[StatusRow], stage: str) -> StatusRow | None:
    selected = [row for row in rows if row.stage == stage]
    return selected[-1] if selected else None


def _verify_run_envelope(
    run_root: Path, project_root: Path
) -> tuple[dict[str, Any], list[StatusRow], dict[str, Any]]:
    project_root = project_root.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    expected_parent = (project_root / "artifacts/dlc_runs").resolve()
    _require_within(run_root, expected_parent, "run root")
    if run_root.parent != expected_parent or not RUN_ID_RE.fullmatch(run_root.name):
        raise EvidenceError("run root must be artifacts/dlc_runs/<RUN_ID>")

    manifest_path = run_root / "run_manifest.json"
    status_path = run_root / "status.tsv"
    manifest = _load_json(manifest_path, "run manifest")
    if set(manifest) != {
        "schema_version",
        "run_id",
        "root",
        "git_commit",
        "tracked_code",
        "environment",
        "resume_count",
    }:
        raise EvidenceError("run manifest schema drift")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("run_id") != run_root.name
        or Path(str(manifest.get("root", ""))).expanduser().resolve() != project_root
        or not GIT_RE.fullmatch(str(manifest.get("git_commit", "")))
        or not isinstance(manifest.get("resume_count"), int)
        or int(manifest["resume_count"]) < 0
    ):
        raise EvidenceError("run manifest identity or root drift")
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "CUDA_VISIBLE_DEVICES",
        "PYTHON_BIN",
        "BASE_MODEL",
        "E1_MODEL",
        "HF_HUB_OFFLINE",
        "WANDB_MODE",
    }:
        raise EvidenceError("run manifest environment schema drift")
    selectors = str(environment.get("CUDA_VISIBLE_DEVICES", "")).split(",")
    if len(selectors) != 4 or len(set(selectors)) != 4 or any(not item for item in selectors):
        raise EvidenceError("run manifest does not bind four unique GPU selectors")
    tracked = manifest.get("tracked_code")
    if not isinstance(tracked, dict) or set(tracked) != set(EXPECTED_TRACKED_CODE):
        raise EvidenceError("run manifest tracked-code coverage drift")
    for relative in EXPECTED_TRACKED_CODE:
        recorded = tracked.get(relative)
        path = project_root / relative
        if not isinstance(recorded, str) or not SHA256_RE.fullmatch(recorded):
            raise EvidenceError(f"invalid run-manifest code hash: {relative}")
        if not path.is_file() or file_hash(path) != recorded:
            raise EvidenceError(f"run code is stale or tampered: {relative}")

    rows = _parse_status(status_path)
    terminal = _latest(rows, "pipeline")
    if terminal is None or terminal.state != "COMPLETE_WITH_BLOCKED" or terminal.rc != "0":
        raise EvidenceError("pipeline lacks the exact successful terminal status")
    envelope = {
        "run_manifest": file_binding(manifest_path),
        "status": file_binding(status_path),
        "run_id": run_root.name,
        "git_commit": manifest["git_commit"],
        "tracked_code": dict(sorted(tracked.items())),
        "terminal": terminal.__dict__,
    }
    return manifest, rows, envelope


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _paired_bootstrap(
    deltas_by_dataset: dict[str, list[float]], *, seed: int = CI_SEED, samples: int = CI_SAMPLES
) -> dict[str, Any]:
    if set(deltas_by_dataset) != set(general.DATASET_ORDER):
        raise EvidenceError("paired CI dataset coverage drift")
    rng = random.Random(seed)
    replicates = {dataset: [] for dataset in general.DATASET_ORDER}
    macro_replicates: list[float] = []
    for _ in range(samples):
        current: list[float] = []
        for dataset in general.DATASET_ORDER:
            values = deltas_by_dataset[dataset]
            if not values:
                raise EvidenceError(f"paired CI has no samples for {dataset}")
            estimate = 100.0 * sum(rng.choices(values, k=len(values))) / len(values)
            replicates[dataset].append(estimate)
            current.append(estimate)
        macro_replicates.append(sum(current) / len(current))

    def interval(values: list[float]) -> list[float]:
        return [round(_percentile(values, 0.025), 6), round(_percentile(values, 0.975), 6)]

    return {
        "method": "stratified_paired_nonparametric_bootstrap",
        "unit": "evaluation_item_within_dataset",
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
        "dataset_delta_points_ci": {
            dataset: interval(replicates[dataset]) for dataset in general.DATASET_ORDER
        },
        "macro_delta_points_ci": interval(macro_replicates),
    }


def _verify_general(
    run_root: Path,
    *,
    allow_test_fixture: bool = False,
    expected_dataset_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    general_root = (run_root / "general_retention_base_e1").resolve()
    result_paths = {
        "base": general_root / "results/base.json",
        "e1": general_root / "results/e1.json",
        "comparison": general_root / "results/base_vs_e1.json",
    }
    try:
        base = general._verify_result(
            result_paths["base"], allow_test_fixture=allow_test_fixture
        )
        e1 = general._verify_result(
            result_paths["e1"], allow_test_fixture=allow_test_fixture
        )
        comparison = general.verify_comparison(
            result_paths["comparison"],
            require_admitted=True,
            allow_test_fixture=allow_test_fixture,
        )
    except (general.FormalProtocolError, OSError, KeyError, TypeError) as exc:
        raise EvidenceError(f"General Retention deep verification failed: {exc}") from exc
    if (
        base.get("status") != "FORMAL_SCORE_ADMITTED"
        or e1.get("status") != "FORMAL_SCORE_ADMITTED"
        or comparison.get("status") != "PAIRED_GENERAL_RETENTION_ADMITTED"
    ):
        raise EvidenceError("General Retention results are not fully admitted")
    if base.get("model_role") != "base" or e1.get("model_role") != "e1":
        raise EvidenceError("General Retention model-role drift")

    for role, result in (("base", base), ("e1", e1)):
        binding_path = Path(str((result.get("model_binding") or {}).get("path", "")))
        try:
            binding = general.verify_model_binding(
                binding_path, deep=True, allow_test_fixture=allow_test_fixture
            )
        except (general.FormalProtocolError, OSError, KeyError, TypeError) as exc:
            raise EvidenceError(f"{role} model deep verification failed: {exc}") from exc
        if binding.get("identity_sha256") != result.get("model_binding_identity_sha256"):
            raise EvidenceError(f"{role} result/model identity drift")
        shards = result.get("prediction_shards")
        expected_paths = [
            (general_root / f"predictions/{role}/shard_{index}.jsonl").resolve()
            for index in range(general.NUM_SHARDS)
        ]
        observed_paths = [Path(str(item.get("path", ""))).resolve() for item in shards]
        if observed_paths != expected_paths:
            raise EvidenceError(f"{role} result does not bind the expected four run shards")

    receipt_path = Path(str((base.get("input_receipt") or {}).get("path", ""))).resolve()
    if receipt_path != (general_root / "inputs/input_receipt.json").resolve():
        raise EvidenceError("General Retention input receipt is outside the fixed run layout")
    try:
        receipt = general.verify_input_receipt(
            receipt_path,
            deep_sources=True,
            allow_test_fixture=allow_test_fixture,
        )
        runtime_rows = general._runtime_rows(receipt)
    except (general.FormalProtocolError, OSError, KeyError, TypeError) as exc:
        raise EvidenceError(f"General Retention receipt verification failed: {exc}") from exc
    counts = {
        dataset: sum(row.get("dataset") == dataset for row in runtime_rows)
        for dataset in general.DATASET_ORDER
    }
    expected_counts = expected_dataset_counts or EXPECTED_DATASET_COUNTS
    if counts != expected_counts or sum(counts.values()) != len(runtime_rows):
        raise EvidenceError(f"General Retention dataset-count drift: {counts}")
    expected_uids = [row.get("global_uid") for row in runtime_rows]
    if len(set(expected_uids)) != len(expected_uids) or any(not uid for uid in expected_uids):
        raise EvidenceError("General Retention runtime UIDs are missing or duplicated")
    for role, result in (("base", base), ("e1", e1)):
        items = result.get("per_sample")
        if not isinstance(items, list) or [row.get("global_uid") for row in items] != expected_uids:
            raise EvidenceError(f"{role} per-sample UID order/coverage drift")

    deltas_by_dataset: dict[str, list[float]] = {
        dataset: [] for dataset in general.DATASET_ORDER
    }
    for left, right in zip(base["per_sample"], e1["per_sample"]):
        if left.get("dataset") != right.get("dataset"):
            raise EvidenceError("General Retention paired dataset order drift")
        deltas_by_dataset[left["dataset"]].append(
            float(right["credit"]) - float(left["credit"])
        )
    ci = _paired_bootstrap(deltas_by_dataset)

    datasets: dict[str, Any] = {}
    for dataset in general.DATASET_ORDER:
        base_cell = base["benchmarks"][dataset]
        e1_cell = e1["benchmarks"][dataset]
        paired_cell = comparison["benchmarks"][dataset]
        recomputed_delta = (
            100.0 * sum(deltas_by_dataset[dataset]) / len(deltas_by_dataset[dataset])
        )
        if not math.isclose(
            recomputed_delta,
            float(paired_cell["e1_minus_base_points"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise EvidenceError(f"{dataset}: paired delta does not match per-sample records")
        datasets[dataset] = {
            "n": counts[dataset],
            "metric": base_cell["metric"],
            "base_score_percent": base_cell["raw_score_percent"],
            "e1_score_percent": e1_cell["raw_score_percent"],
            "e1_minus_base_points": paired_cell["e1_minus_base_points"],
            "e1_minus_base_points_ci95": ci["dataset_delta_points_ci"][dataset],
            "retention_ratio_percent": paired_cell["retention_ratio_percent"],
            "paired_transitions": {
                "improved": paired_cell["paired_improved"],
                "regressed": paired_cell["paired_regressed"],
                "unchanged": paired_cell["paired_unchanged"],
            },
            "output_validity": {
                "base_percent": base_cell["output_validity_percent"],
                "e1_percent": e1_cell["output_validity_percent"],
                "base_malformed_count": base_cell["malformed_count"],
                "e1_malformed_count": e1_cell["malformed_count"],
                "base_truncation_count": base_cell["truncation_count"],
                "e1_truncation_count": e1_cell["truncation_count"],
                "base_health_admitted": base_cell["health_admitted"],
                "e1_health_admitted": e1_cell["health_admitted"],
            },
        }

    macro_delta = float(e1["raw_general_avg_percent"]) - float(
        base["raw_general_avg_percent"]
    )
    if not math.isclose(
        macro_delta,
        float(comparison["raw_general_avg_delta_points"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise EvidenceError("General Retention macro delta drift")
    return {
        "availability": "available",
        "admission": comparison["status"],
        "protocol": comparison["protocol"],
        "records": len(runtime_rows),
        "dataset_counts": counts,
        "datasets": datasets,
        "macro": {
            "base_general_avg_percent": base["general_avg_percent"],
            "e1_general_avg_percent": e1["general_avg_percent"],
            "e1_minus_base_points": comparison["raw_general_avg_delta_points"],
            "e1_minus_base_points_ci95": ci["macro_delta_points_ci"],
        },
        "paired_ci": {
            key: value
            for key, value in ci.items()
            if key not in {"dataset_delta_points_ci", "macro_delta_points_ci"}
        },
        "models": {
            "base_checkpoint_identity_sha256": base[
                "model_checkpoint_content_identity_sha256"
            ],
            "e1_checkpoint_identity_sha256": e1[
                "model_checkpoint_content_identity_sha256"
            ],
            "base_binding_identity_sha256": base["model_binding_identity_sha256"],
            "e1_binding_identity_sha256": e1["model_binding_identity_sha256"],
        },
        "artifacts": {key: file_binding(path) for key, path in result_paths.items()},
    }


def _formal_v52_unavailable(run_root: Path, rows: list[StatusRow]) -> dict[str, Any]:
    stage = _latest(rows, "v52_paired")
    if stage is None:
        raise EvidenceError("pipeline status omits v5.2 disposition")
    allowed = {
        "SKIPPED",
        "SKIPPED_PENDING_ANNOTATION",
        "BLOCKED_ANNOTATION_GATE",
    }
    if stage.state not in allowed or stage.rc != "0":
        raise EvidenceError(f"v5.2 did not complete or cleanly skip: {stage.state}/{stage.rc}")
    result_path = run_root / "scannet_rgb_depth_v52_paired/result.json"
    if result_path.exists():
        raise EvidenceError("v5.2 status says unavailable but a stale result is present")
    output: dict[str, Any] = {
        "availability": "unavailable",
        "status_state": stage.state,
        "reason": stage.detail,
        "status_event": stage.__dict__,
    }
    if stage.state == "SKIPPED_PENDING_ANNOTATION":
        match = re.fullmatch(r"compile_status=([A-Z0-9_]+)", stage.detail)
        formal_stage = _latest(rows, "v52_formal")
        if match is None or formal_stage is None or formal_stage.state != "BLOCKED":
            raise EvidenceError("pending v5.2 status lacks a matching formal-gate event")
        gate_match = re.fullmatch(r"status=([A-Z0-9_]+) report=(/.+)", formal_stage.detail)
        if gate_match is None or gate_match.group(1) != match.group(1):
            raise EvidenceError("v5.2 pending reason/formal-gate status mismatch")
        report_path = Path(gate_match.group(2)).expanduser().resolve()
        if report_path != (run_root / "v52_formal_compile.json").resolve():
            raise EvidenceError("v5.2 formal-gate report path drift")
        report = _load_json(report_path, "v5.2 formal-gate report")
        try:
            v52_eval._validate_report_identity(report)
        except (v52_eval.ContractError, KeyError, TypeError) as exc:
            raise EvidenceError(f"v5.2 pending report identity failed: {exc}") from exc
        if report.get("status") != match.group(1):
            raise EvidenceError("v5.2 pending report status drift")
        output["formal_gate_report"] = file_binding(report_path)
        output["formal_gate_status"] = report["status"]
    return output


def _verify_v52_complete(run_root: Path, rows: list[StatusRow]) -> dict[str, Any]:
    stage = _latest(rows, "v52_paired")
    if stage is None or stage.state != "SUCCEEDED" or stage.rc != "0":
        return _formal_v52_unavailable(run_root, rows)
    root = (run_root / "scannet_rgb_depth_v52_paired").resolve()
    paths = {
        "result": root / "result.json",
        "release": root / "release/release.manifest.json",
        "model_binding": root / "model_binding/model.json",
        "g_predictions": root / "g/predictions.jsonl",
        "g_manifest": root / "g/predictions.manifest.json",
        "s_oracle_predictions": root / "s_oracle/predictions.jsonl",
        "s_oracle_manifest": root / "s_oracle/predictions.manifest.json",
        "s_absent_predictions": root / "s_absent/predictions.jsonl",
        "s_absent_manifest": root / "s_absent/predictions.manifest.json",
        "s_corrupted_predictions": root / "s_corrupted/predictions.jsonl",
        "s_corrupted_manifest": root / "s_corrupted/predictions.manifest.json",
        "s_pred_requests": root / "s_pred/requests.manifest.json",
        "s_pred_predictions": root / "s_pred/predictions.jsonl",
        "s_pred_manifest": root / "s_pred/predictions.manifest.json",
    }
    try:
        release = v52_eval.validate_request_release(paths["release"], verify_images=True)
        binding = v52_dlc.verify_model_binding(paths["model_binding"], deep=True)
        g_bundle = v52_dlc.verify_prediction_bundle(
            manifest_path=paths["g_manifest"],
            binding_path=paths["model_binding"],
            predictions_path=paths["g_predictions"],
            expected_track=v52_eval.TRACK_G,
            deep_model=False,
        )
        oracle_bundle = v52_dlc.verify_prediction_bundle(
            manifest_path=paths["s_oracle_manifest"],
            binding_path=paths["model_binding"],
            predictions_path=paths["s_oracle_predictions"],
            expected_track=v52_eval.TRACK_S_ORACLE,
            deep_model=False,
        )
        absent_bundle = v52_dlc.verify_prediction_bundle(
            manifest_path=paths["s_absent_manifest"],
            binding_path=paths["model_binding"],
            predictions_path=paths["s_absent_predictions"],
            expected_track=v52_eval.TRACK_S_ABSENT,
            deep_model=False,
        )
        corrupted_bundle = v52_dlc.verify_prediction_bundle(
            manifest_path=paths["s_corrupted_manifest"],
            binding_path=paths["model_binding"],
            predictions_path=paths["s_corrupted_predictions"],
            expected_track=v52_eval.TRACK_S_CORRUPTED,
            deep_model=False,
        )
        predicted_request = v52_dlc.verify_predicted_request_release(
            manifest_path=paths["s_pred_requests"],
            binding_path=paths["model_binding"],
            g_manifest_path=paths["g_manifest"],
            g_predictions_path=paths["g_predictions"],
            verify_images=True,
        )
        predicted_bundle = v52_dlc.verify_prediction_bundle(
            manifest_path=paths["s_pred_manifest"],
            binding_path=paths["model_binding"],
            predictions_path=paths["s_pred_predictions"],
            expected_track=v52_eval.TRACK_S_PRED,
            dynamic_request_manifest=paths["s_pred_requests"],
            deep_model=False,
        )
    except (v52_eval.ContractError, v52_dlc.RunnerError, OSError, KeyError, TypeError) as exc:
        raise EvidenceError(f"v5.2 deep verification failed: {exc}") from exc
    if len(release.source.pairs) != v52_eval.EXPECTED_RECORDS:
        raise EvidenceError("v5.2 formal source is not exact40")
    bundles = (g_bundle, oracle_bundle, absent_bundle, corrupted_bundle, predicted_bundle)
    identities = {
        bundle["provenance"]["model_binding_identity_sha256"] for bundle in bundles
    }
    receipts = {bundle["receipt"]["identity_sha256"] for bundle in bundles}
    if identities != {binding["identity_sha256"]} or len(receipts) != 1:
        raise EvidenceError("v5.2 stages do not share one model/code/runtime receipt")
    if predicted_request["manifest"]["g_prediction_manifest"] != v52_eval.file_binding(
        paths["g_manifest"]
    ):
        raise EvidenceError("v5.2 S-pred requests are not bound to scored G outputs")
    provenance: dict[str, Any] = {
        "verified": True,
        "request_release": v52_eval.file_binding(paths["release"]),
        "request_release_identity_sha256": release.manifest["identity_sha256"],
        "formal_source": release.source.binding,
        "input_receipt": g_bundle["receipt"],
        "model_binding": v52_eval.file_binding(paths["model_binding"]),
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_inventory": binding["model"],
        "stages": {
            v52_eval.TRACK_G: g_bundle["provenance"],
            v52_eval.TRACK_S_ORACLE: oracle_bundle["provenance"],
            v52_eval.TRACK_S_ABSENT: absent_bundle["provenance"],
            v52_eval.TRACK_S_CORRUPTED: corrupted_bundle["provenance"],
            v52_eval.TRACK_S_PRED: predicted_bundle["provenance"],
        },
        "predicted_request_release": v52_eval.file_binding(paths["s_pred_requests"]),
        "same_model_binding_all_stages": True,
        "training_materialization_eligible": False,
    }
    provenance["identity_sha256"] = v52_eval.canonical_hash(provenance)
    rebuilt = v52_eval.score_paired_predictions(
        release.source.pairs,
        g_bundle["rows"],
        oracle_bundle["rows"],
        absent_bundle["rows"],
        corrupted_bundle["rows"],
        predicted_bundle["rows"],
        provenance=provenance,
    )
    result = _load_json(paths["result"], "v5.2 result")
    if rebuilt != result:
        raise EvidenceError("v5.2 result does not recompute from verified stage bundles")
    if (
        result.get("status") != "FORMAL_SCORE_COMPLETE"
        or (result.get("metrics") or {}).get("n_scenes") != v52_eval.EXPECTED_RECORDS
        or (result.get("trust") or {}).get("score_available") is not True
        or (result.get("trust") or {}).get("all_static_scenes_complete") is not True
        or (result.get("trust") or {}).get("s_pred_exact_valid_g_coverage") is not True
    ):
        raise EvidenceError("v5.2 result lacks exact40 formal admission")
    metrics = result["metrics"]
    return {
        "availability": "available",
        "admission": result["status"],
        "protocol": result["protocol"],
        "records": metrics["n_scenes"],
        "metrics": {
            key: metrics[key]
            for key in (
                "g_valid_pair_coverage",
                "g_both_box_iou_at_0_5",
                "g_mean_min_pair_iou",
                "s_oracle_accuracy",
                "s_oracle_parse_coverage",
                "s_absent_accuracy",
                "s_absent_parse_coverage",
                "s_corrupted_accuracy",
                "s_corrupted_parse_coverage",
                "s_pred_accuracy_given_emitted_request",
                "s_pred_parse_coverage_all_scenes",
                "s_pred_correct_all_scenes_invalid_g_zero",
                "s_pred_strict_e2e_accuracy",
                "paired_dependency_gap_oracle_minus_predicted",
                "paired_context_gain_oracle_minus_absent",
                "paired_context_gain_oracle_minus_corrupted",
                "positive_oracle_headroom",
                "bounded_oracle_recovery",
            )
        },
        "transition_counts": result["transition_counts"],
        "strict_e2e_transition_counts": result["strict_e2e_transition_counts"],
        "scene_bootstrap": result["scene_bootstrap"],
        "context_control_bootstrap": result["context_control_bootstrap"],
        "attribution": result["attribution"],
        "parse_status": result["parse_status"],
        "token_cap_counts": result["token_cap_counts"],
        "model": {
            "binding_identity_sha256": binding["identity_sha256"],
            "checkpoint_identity_sha256": binding["model"][
                "checkpoint_content_identity_sha256"
            ],
        },
        "provenance_identity_sha256": provenance["identity_sha256"],
        "artifacts": {key: file_binding(path) for key, path in paths.items()},
    }


def compile_evidence(
    run_root: Path,
    *,
    project_root: Path = ROOT,
    allow_test_fixture: bool = False,
    expected_dataset_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    _, status_rows, envelope = _verify_run_envelope(run_root, project_root)
    general_stage = _latest(status_rows, "general_retention")
    if general_stage is None or general_stage.state != "SUCCEEDED" or general_stage.rc != "0":
        raise EvidenceError("General Retention lacks a successful orchestrator stage")
    general_evidence = _verify_general(
        run_root,
        allow_test_fixture=allow_test_fixture,
        expected_dataset_counts=expected_dataset_counts,
    )
    v52_evidence = _verify_v52_complete(run_root, status_rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": EVIDENCE_PROTOCOL,
        "status": "CLAIM_EVIDENCE_VERIFIED",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "run_root": str(run_root),
        "compiler": file_binding(Path(__file__)),
        "run_envelope": envelope,
        "general_retention": general_evidence,
        "scannet_v52_paired_gs": v52_evidence,
        "publication_policy": {
            "paper_was_not_modified": True,
            "console_logs_used_as_evidence": False,
            "missing_values_filled": False,
            "only_deeply_recomputed_values_exported": True,
        },
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def _format_number(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    general_evidence = payload["general_retention"]
    lines = [
        "# Verified ICLR claim evidence",
        "",
        f"- Run: `{payload['run_envelope']['run_id']}`",
        f"- Evidence identity: `{payload['identity_sha256']}`",
        f"- General admission: `{general_evidence['admission']}`",
        "",
        "## General Retention",
        "",
        "| Dataset | n | Base | E1 | E1-Base | 95% paired CI | Output validity (Base/E1) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in general.DATASET_ORDER:
        cell = general_evidence["datasets"][dataset]
        ci = cell["e1_minus_base_points_ci95"]
        validity = cell["output_validity"]
        lines.append(
            "| {dataset} | {n} | {base} | {e1} | {delta} | [{lo}, {hi}] | {bv}/{ev} |".format(
                dataset=dataset,
                n=cell["n"],
                base=_format_number(cell["base_score_percent"]),
                e1=_format_number(cell["e1_score_percent"]),
                delta=_format_number(cell["e1_minus_base_points"]),
                lo=_format_number(ci[0]),
                hi=_format_number(ci[1]),
                bv=_format_number(validity["base_percent"]),
                ev=_format_number(validity["e1_percent"]),
            )
        )
    macro = general_evidence["macro"]
    lines.extend(
        [
            "",
            "Macro Base/E1/delta: {}/{}/{} points; paired CI [{}, {}].".format(
                _format_number(macro["base_general_avg_percent"]),
                _format_number(macro["e1_general_avg_percent"]),
                _format_number(macro["e1_minus_base_points"]),
                _format_number(macro["e1_minus_base_points_ci95"][0]),
                _format_number(macro["e1_minus_base_points_ci95"][1]),
            ),
            "",
            "## ScanNet v5.2 paired G to S",
            "",
        ]
    )
    v52 = payload["scannet_v52_paired_gs"]
    if v52["availability"] == "unavailable":
        lines.append(f"Unavailable: `{v52['status_state']}` ({v52['reason']}).")
    else:
        metrics = v52["metrics"]
        for key, value in metrics.items():
            lines.append(f"- `{key}`: {_format_number(value)}")
        lines.append(f"- scene bootstrap: `{canonical_json(v52['scene_bootstrap'])}`")
        lines.append(
            f"- context-control bootstrap: `{canonical_json(v52['context_control_bootstrap'])}`"
        )
        lines.append(f"- attribution: `{canonical_json(v52['attribution'])}`")
        lines.append(f"- predicted-context transitions: `{canonical_json(v52['transition_counts'])}`")
        lines.append(f"- strict-E2E transitions: `{canonical_json(v52['strict_e2e_transition_counts'])}`")
        lines.append(f"- parse/cap: `{canonical_json({'parse': v52['parse_status'], 'cap': v52['token_cap_counts']})}`")
    lines.extend(
        [
            "",
            "This summary is derived only from deeply verified formal artifacts; console logs and missing-value substitution are excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_fresh(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvidenceError(f"refusing to overwrite existing output: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise EvidenceError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_root = args.run_root.expanduser().resolve()
        requested_outputs = [args.out] + ([args.markdown] if args.markdown is not None else [])
        for requested in requested_outputs:
            try:
                requested.expanduser().resolve().relative_to(run_root)
            except ValueError:
                continue
            raise EvidenceError("claim-evidence outputs must be outside the immutable run root")
        if args.out.exists() or (args.markdown is not None and args.markdown.exists()):
            raise EvidenceError("all requested outputs must be fresh paths")
        payload = compile_evidence(run_root)
        _write_fresh(
            args.out,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if args.markdown is not None:
            _write_fresh(args.markdown, render_markdown(payload))
    except (EvidenceError, OSError) as exc:
        print(f"EVIDENCE_REJECTED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "identity_sha256": payload["identity_sha256"],
                "out": str(args.out.expanduser().resolve()),
                "markdown": str(args.markdown.expanduser().resolve()) if args.markdown else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
