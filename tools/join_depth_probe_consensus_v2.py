#!/usr/bin/env python3
"""Join two hash-bound depth probes for future ScanNet anomaly checks only.

Both formal probe reports, their pair JSONL files, the current local weights,
and the frozen canary identity are independently validated. The joined output
does not create depth labels: it only identifies model-consensus directions
that may be compared with upcoming ScanNet sensor-depth labels to flag cases
for inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_depth_v43_pairwise import (  # noqa: E402
    FROZEN_CANARY_SET_SHA256,
    FROZEN_CANARY_SHA256S,
)
from tools.probe_depth_da2_v1 import (  # noqa: E402
    PROTOCOL as DA2_PROTOCOL,
    _require_artifact_output,
    validate_frozen_v43_trace,
    validate_model_artifact as validate_da2_model,
)
from tools.probe_depth_dpt_v1 import (  # noqa: E402
    INTENDED_USE,
    PROTOCOL as DPT_PROTOCOL,
    validate_model_artifact as validate_dpt_model,
)
from trains.flywheel.build_depth_semantic_audit_pack import (  # noqa: E402
    content_hash,
    ensure_empty_output,
    file_hash,
)

PROTOCOL = "depth-dual-probe-consensus-v2"
ARTIFACT_KIND = "scannet_sensor_depth_anomaly_detector_calibration_only"


def _load_json(path: Path, kind: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {kind}: {path}: {exc}") from exc


def _load_jsonl(path: Path, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read {kind}: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"invalid {kind} at {path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise SystemExit(f"non-object {kind} row at {path}:{line_number}")
        rows.append(row)
    return rows


def _local_pair_file(report_path: Path, value: Any) -> Path:
    base = report_path.parent.resolve()
    path = (base / str(value or "")).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise SystemExit("formal probe pair file escapes its artifact directory") from exc
    if not path.is_file():
        raise SystemExit(f"missing formal probe pair file: {path}")
    return path


def normalize_formal_probe(
    report_path: Path,
    *,
    model_id: str,
    expected_protocol: str,
    current_model_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify a formal probe and normalize only its pair-decision contract."""
    report = _load_json(report_path, f"{model_id} formal probe report")
    safety = report.get("safety")
    if (
        not isinstance(report, dict)
        or report.get("protocol") != expected_protocol
        or report.get("completed") is not True
        or report.get("candidate_images") != 8
        or report.get("candidate_set_sha256") != FROZEN_CANARY_SET_SHA256
        or report.get("planned_pairs") != 40
        or report.get("external_uploads") != 0
        or report.get("training_eligible") is not False
        or report.get("semantic_audit_eligible") is not False
        or report.get("automatic_pass", False) is not False
        or report.get("model") != current_model_manifest
        or not isinstance(safety, dict)
        or safety.get("images_uploaded") != 0
        or safety.get("network_enabled") is not False
        or safety.get("writes_training_records") is not False
        or safety.get("writes_semantic_audit_records") is not False
    ):
        raise SystemExit(f"{model_id} formal probe contract is not eligible for join")
    if model_id == "dpt_hybrid_midas" and (
        report.get("intended_use") != INTENDED_USE
        or report.get("ground_truth_eligible") is not False
        or safety.get("writes_scannet_records") is not False
    ):
        raise SystemExit("DPT formal probe is not restricted to ScanNet anomaly checks")
    trace = report.get("v43_trace")
    if (
        not isinstance(trace, dict)
        or not all(
            isinstance(trace.get(key), str) and len(trace[key]) >= 24
            for key in (
                "run_id",
                "report_sha256",
                "request_log_slice_sha256",
                "response_audit_sha256",
                "cache_manifest_sha256",
            )
        )
    ):
        raise SystemExit(f"{model_id} formal probe lacks frozen trace binding")
    trace_report_path = Path(trace.get("report_path", ""))
    _, _, current_run_binding = validate_frozen_v43_trace(trace_report_path)
    expected_trace = {
        "report_path": str(trace_report_path.resolve()),
        "report_sha256": file_hash(trace_report_path),
        **current_run_binding,
    }
    if trace != expected_trace:
        raise SystemExit(f"{model_id} formal probe frozen trace binding changed")
    pair_meta = report.get("pairs")
    if not isinstance(pair_meta, dict) or pair_meta.get("rows") != 40:
        raise SystemExit(f"{model_id} formal probe pair metadata is invalid")
    pair_path = _local_pair_file(report_path, pair_meta.get("path"))
    rows = _load_jsonl(pair_path, f"{model_id} pair probe")
    if (
        len(rows) != 40
        or file_hash(pair_path) != pair_meta.get("sha256")
        or content_hash(rows) != pair_meta.get("content_sha256")
    ):
        raise SystemExit(f"{model_id} pair probe hash or row count mismatch")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, str]]] = set()
    seen_images: set[str] = set()
    for row in rows:
        image_sha = row.get("image_sha256")
        entities = row.get("entities")
        decision = row.get("decision")
        raw_nearer = row.get("raw_nearer")
        raw_farther = row.get("raw_farther")
        if (
            image_sha not in FROZEN_CANARY_SHA256S
            or not isinstance(entities, list)
            or len(entities) != 2
            or not all(isinstance(item, str) and item.strip() for item in entities)
            or len({item.casefold() for item in entities}) != 2
            or decision not in {"directional_probe", "indeterminate"}
            or raw_nearer not in entities
            or raw_farther not in entities
            or raw_nearer == raw_farther
        ):
            raise SystemExit(f"malformed {model_id} formal pair row")
        if decision == "directional_probe":
            if row.get("nearer") != raw_nearer or row.get("farther") != raw_farther:
                raise SystemExit(f"inconsistent directional decision in {model_id}")
        elif row.get("nearer") is not None or row.get("farther") is not None:
            raise SystemExit(f"indeterminate {model_id} pair exposes a direction label")
        pair_key = tuple(sorted(entities))
        key = (image_sha, pair_key)
        if key in seen:
            raise SystemExit(f"duplicate {model_id} formal pair: {key}")
        seen.add(key)
        seen_images.add(image_sha)
        normalized.append({
            "image_sha256": image_sha,
            "pair_key": list(pair_key),
            "raw_nearer": raw_nearer,
            "raw_farther": raw_farther,
            "within_model_reliable": decision == "directional_probe",
            "source_decision": decision,
            "source_pair": row,
        })
    if seen_images != set(FROZEN_CANARY_SHA256S):
        raise SystemExit(f"{model_id} formal probe does not cover all frozen images")
    return {
        "model_id": model_id,
        "protocol": "formal-depth-probe-view-v2",
        "source": {
            "report_path": str(report_path.resolve()),
            "report_sha256": file_hash(report_path),
            "pair_path": str(pair_path),
            "pair_sha256": file_hash(pair_path),
        },
        "model": current_model_manifest,
        "v43_trace": trace,
        "candidate_set_sha256": FROZEN_CANARY_SET_SHA256,
        "pairs": normalized,
        "automatic_pass": False,
        "ground_truth_eligible": False,
    }


def _pair_map(view: dict[str, Any]) -> dict[tuple[str, tuple[str, str]], dict[str, Any]]:
    rows = view.get("pairs")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("formal model view must contain 40 pairs")
    mapped: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    for row in rows:
        key = (row["image_sha256"], tuple(row["pair_key"]))
        if key in mapped:
            raise ValueError(f"duplicate formal pair key: {key}")
        mapped[key] = row
    return mapped


def join_formal_views(
    first_view: dict[str, Any], second_view: dict[str, Any]
) -> list[dict[str, Any]]:
    """Join reliable agreement without promoting consensus to a depth label."""
    first = _pair_map(first_view)
    second = _pair_map(second_view)
    if set(first) != set(second):
        raise ValueError("formal probes do not cover identical frozen entity pairs")
    rows: list[dict[str, Any]] = []
    for key in sorted(first):
        left, right = first[key], second[key]
        agree = (
            left["raw_nearer"] == right["raw_nearer"]
            and left["raw_farther"] == right["raw_farther"]
        )
        reliable = bool(
            left["within_model_reliable"] and right["within_model_reliable"]
        )
        candidate = agree and reliable
        rows.append({
            "image_sha256": key[0],
            "pair_key": list(key[1]),
            "models_agree": agree,
            "both_within_model_reliable": reliable,
            "sensor_depth_anomaly_check_candidate": candidate,
            "expected_nearer_for_comparison": (
                left["raw_nearer"] if candidate else None
            ),
            "expected_farther_for_comparison": (
                left["raw_farther"] if candidate else None
            ),
            "status": (
                "compare_only_after_scannet_sensor_depth_label_exists"
                if candidate
                else "not_selected_for_sensor_depth_anomaly_check"
            ),
            "automatic_label": False,
            "automatic_pass": False,
            "ground_truth_eligible": False,
            "model_views": {
                first_view["model_id"]: left,
                second_view["model_id"]: right,
            },
        })
    return rows


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--da2-report",
        type=Path,
        default=ROOT / "artifacts/depth_da2_v1/canary_v1/probe_report.json",
    )
    parser.add_argument(
        "--dpt-report",
        type=Path,
        default=ROOT / "artifacts/depth_dpt_v1/canary_v1/probe_report.json",
    )
    parser.add_argument(
        "--da2-model-dir",
        type=Path,
        default=Path(
            "/path/to/environment/models/depth-anything/"
            "Depth-Anything-V2-Metric-Indoor-Small-hf"
        ),
    )
    parser.add_argument(
        "--dpt-model-dir",
        type=Path,
        default=Path(
            "/path/to/environment/models/depth-estimation/Intel-dpt-hybrid-midas"
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/depth_dual_probe_v2/canary_v1"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = _require_artifact_output(args.out)
    ensure_empty_output(out)
    da2_manifest = validate_da2_model(args.da2_model_dir)
    dpt_manifest = validate_dpt_model(args.dpt_model_dir)
    da2_view = normalize_formal_probe(
        args.da2_report,
        model_id="da2_indoor_metric_small",
        expected_protocol=DA2_PROTOCOL,
        current_model_manifest=da2_manifest,
    )
    dpt_view = normalize_formal_probe(
        args.dpt_report,
        model_id="dpt_hybrid_midas",
        expected_protocol=DPT_PROTOCOL,
        current_model_manifest=dpt_manifest,
    )
    if da2_view["v43_trace"] != dpt_view["v43_trace"]:
        raise SystemExit("formal probes are not bound to the same frozen v4.3 trace")
    consensus = _round_floats(join_formal_views(da2_view, dpt_view))
    out.mkdir(parents=True, exist_ok=True)
    view_files: dict[str, dict[str, Any]] = {}
    for view in (da2_view, dpt_view):
        model_id = view["model_id"]
        path = out / f"{model_id}_formal_view.json"
        path.write_text(
            json.dumps(_round_floats(view), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        view_files[model_id] = {"path": path.name, "sha256": file_hash(path)}
    consensus_path = out / "sensor_depth_anomaly_candidates.jsonl"
    consensus_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in consensus),
        encoding="utf-8",
    )
    candidates = sum(
        row["sensor_depth_anomaly_check_candidate"] for row in consensus
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_kind": ARTIFACT_KIND,
        "intended_use": INTENDED_USE,
        "candidate_set_sha256": FROZEN_CANARY_SET_SHA256,
        "formal_model_views": view_files,
        "pairs": {
            "rows": len(consensus),
            "raw_direction_agreements": sum(row["models_agree"] for row in consensus),
            "both_model_reliable_agreements": candidates,
            "sensor_depth_anomaly_check_candidates": candidates,
            "path": consensus_path.name,
            "sha256": file_hash(consensus_path),
            "content_sha256": content_hash(consensus),
        },
        "scannet_sensor_depth_labels_present": False,
        "scannet_builder_modified": False,
        "comparison_may_run_only_after_sensor_labels_exist": True,
        "automatic_label": False,
        "automatic_pass": False,
        "semantic_audit_pass": False,
        "ground_truth_eligible": False,
        "training_eligible": False,
        "writes_scannet_records": False,
        "known_limitation": (
            "Two monocular depth models can share a wrong direction. Consensus is "
            "only an anomaly detector input after independent sensor-depth labels "
            "exist; it is never a substitute for those labels."
        ),
    }
    report_path = out / "consensus_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "protocol": PROTOCOL,
        "report": str(report_path),
        "raw_direction_agreements": report["pairs"]["raw_direction_agreements"],
        "sensor_depth_anomaly_check_candidates": candidates,
        "intended_use": INTENDED_USE,
        "automatic_label": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
