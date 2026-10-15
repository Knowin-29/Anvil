#!/usr/bin/env python3
"""Normalize two frozen dense-depth probe logs and join audit-only consensus.

The current DA2 and DPT logs predate the versioned probe artifact schema, so
this tool makes their limitations explicit: it validates the present local
weights and binds each normalized view to its source-log hash, but cannot prove
which weights produced an old log. Consensus is therefore only a candidate for
human semantic audit and can never produce an automatic PASS or training row.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
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
    _require_artifact_output,
    validate_model_artifact,
)
from trains.flywheel.build_depth_semantic_audit_pack import (  # noqa: E402
    content_hash,
    ensure_empty_output,
    file_hash,
)

PROTOCOL = "depth-dual-probe-consensus-v1"
NORMALIZATION_PROTOCOL = "depth-dense-log-normalization-v1"
MIN_RELATIVE_MARGIN = 0.03
MIN_UNCERTAINTY_RATIO = 1.0
EXPECTED_ENTITY_COUNTS = (4, 2, 4, 3, 4, 4, 4, 4)
DPT_MODEL_REPO = "Intel/dpt-hybrid-midas"
DPT_MODEL_REVISION = "11eaf7a1cf4bd70740697dbc216f98980c0aeb03"
DPT_MODEL_FILES = {
    "config.json": "d469783f4a0bbfdbdafb9af7b193480ce9d87c5687984056ccc3738e5367ca66",
    "preprocessor_config.json": (
        "4da22e615634b6316e2564136943b8550ebaa77d4e300004e8a7e99c92b68c6a"
    ),
    "pytorch_model.bin": (
        "b6c4d44f9d96ca3fa76dd3bbb153989a60b4ad5526559f3c598562a368d687ec"
    ),
}
MODEL_SPECS = {
    "da2_indoor_metric_small": {
        "runtime_class": "DepthAnythingForDepthEstimation",
        "depth_semantics": "metric_depth_m",
        "nearer_rule": "lower",
        "median_field": "median_m",
        "mad_field": "mad_m",
        "q25_field": "q25_m",
        "q75_field": "q75_m",
    },
    "dpt_hybrid_midas": {
        "runtime_class": "DPTForDepthEstimation",
        "depth_semantics": "relative_inverse_depth",
        "nearer_rule": "higher",
        "median_field": "median_inverse_depth",
        "mad_field": "mad",
        "q25_field": "q25",
        "q75_field": "q75",
    },
}


def validate_dpt_model_artifact(model_dir: Path) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise SystemExit(f"missing local DPT model directory: {model_dir}")
    files: dict[str, dict[str, Any]] = {}
    for name, expected_sha in DPT_MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"missing DPT model file: {path}")
        actual_sha = file_hash(path)
        if actual_sha != expected_sha:
            raise SystemExit(
                f"DPT model hash mismatch for {name}: {actual_sha} != {expected_sha}"
            )
        files[name] = {"sha256": actual_sha, "bytes": path.stat().st_size}
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        processor = json.loads(
            (model_dir / "preprocessor_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid local DPT configuration: {exc}") from exc
    if (
        config.get("model_type") != "dpt"
        or config.get("architectures") != ["DPTForDepthEstimation"]
        or processor.get("image_processor_type") != "DPTImageProcessor"
        or processor.get("size") != {"height": 384, "width": 384}
    ):
        raise SystemExit("DPT configuration does not match Intel/dpt-hybrid-midas")
    return {
        "repo_id": DPT_MODEL_REPO,
        "revision": DPT_MODEL_REVISION,
        "local_path": str(model_dir.resolve()),
        "files": files,
        "depth_semantics": "relative_inverse_depth",
    }


def _json_events(log_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"cannot read dense-depth probe log {log_path}: {exc}") from exc
    if "probe_rc=0" not in {line.strip() for line in text.splitlines()}:
        raise SystemExit(f"probe log has no exact successful exit marker: {log_path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"invalid JSON event in {log_path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise SystemExit(f"invalid probe event in {log_path}:{line_number}")
        events.append(event)
    return events, file_hash(log_path)


def _one_event(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    selected = [event for event in events if event.get("event") == name]
    if len(selected) != 1:
        raise SystemExit(f"expected exactly one {name!r} event, found {len(selected)}")
    return selected[0]


def _number(item: dict[str, Any], field: str, context: str) -> float:
    value = item.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise SystemExit(f"invalid positive {field} for {context}")
    return float(value)


def model_pairwise_probes(
    image_sha256: str,
    entities: list[dict[str, Any]],
    nearer_rule: str,
) -> list[dict[str, Any]]:
    if nearer_rule not in {"lower", "higher"}:
        raise ValueError("nearer_rule must be lower or higher")
    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(entities, 2):
        first_value = float(first["median_value"])
        second_value = float(second["median_value"])
        if first_value == second_value:
            nearer = None
            farther = None
        else:
            first_is_nearer = (
                first_value < second_value
                if nearer_rule == "lower"
                else first_value > second_value
            )
            nearer = first["label"] if first_is_nearer else second["label"]
            farther = second["label"] if first_is_nearer else first["label"]
        margin = abs(first_value - second_value)
        relative_margin = margin / max(abs(first_value), abs(second_value), 1e-12)
        combined_uncertainty = math.hypot(
            float(first["uncertainty"]), float(second["uncertainty"])
        )
        uncertainty_ratio = (
            margin / combined_uncertainty if combined_uncertainty > 0 else None
        )
        failed: list[str] = []
        if nearer is None:
            failed.append("exact_tie")
        if relative_margin < MIN_RELATIVE_MARGIN:
            failed.append("relative_margin")
        if uncertainty_ratio is not None and uncertainty_ratio < MIN_UNCERTAINTY_RATIO:
            failed.append("uncertainty_ratio")
        rows.append({
            "image_sha256": image_sha256,
            "pair_key": sorted([first["label"], second["label"]]),
            "raw_nearer": nearer,
            "raw_farther": farther,
            "within_model_reliable": not failed,
            "relative_margin": relative_margin,
            "combined_uncertainty": combined_uncertainty,
            "uncertainty_ratio": uncertainty_ratio,
            "failed_thresholds": failed,
        })
    return rows


def normalize_probe_log(
    log_path: Path,
    model_id: str,
    current_model_manifest: dict[str, Any],
) -> dict[str, Any]:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"unknown model_id: {model_id}")
    spec = MODEL_SPECS[model_id]
    events, log_sha = _json_events(log_path)
    start = _one_event(events, "start")
    imports = _one_event(events, "imports")
    loaded = _one_event(events, "model_loaded")
    complete = _one_event(events, "complete")
    if (
        start.get("cuda_visible_devices") != ""
        or imports.get("cuda_available") is not False
        or loaded.get("device") != "cpu"
        or loaded.get("class") != spec["runtime_class"]
    ):
        raise SystemExit(f"{model_id} log does not prove a CPU-only expected runtime")
    image_events = sorted(
        (event for event in events if event.get("event") == "image"),
        key=lambda event: event.get("index", -1),
    )
    if (
        len(image_events) != 8
        or [event.get("index") for event in image_events] != list(range(8))
        or tuple(event.get("sha256") for event in image_events)
        != FROZEN_CANARY_SHA256S
    ):
        raise SystemExit(f"{model_id} log is not the frozen 8-image canary")

    normalized_images: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for image_event, expected_count in zip(image_events, EXPECTED_ENTITY_COUNTS):
        raw_entities = image_event.get("entities")
        if not isinstance(raw_entities, list) or len(raw_entities) != expected_count:
            raise SystemExit(f"unexpected entity count in {model_id} log")
        entities: list[dict[str, Any]] = []
        for raw in raw_entities:
            if (
                not isinstance(raw, dict)
                or not isinstance(raw.get("label"), str)
                or not raw["label"].strip()
                or not isinstance(raw.get("bbox"), list)
                or len(raw["bbox"]) != 4
            ):
                raise SystemExit(f"malformed entity in {model_id} log")
            median = _number(raw, spec["median_field"], raw["label"])
            mad = _number(raw, spec["mad_field"], raw["label"])
            q25 = _number(raw, spec["q25_field"], raw["label"])
            q75 = _number(raw, spec["q75_field"], raw["label"])
            if q75 < q25:
                raise SystemExit(f"reversed quartiles for {raw['label']}")
            uncertainty = max(1.4826 * mad, 0.5 * (q75 - q25))
            entities.append({
                "label": raw["label"].strip(),
                "bbox_2d": list(raw["bbox"]),
                "median_value": median,
                "mad": mad,
                "q25": q25,
                "q75": q75,
                "uncertainty": uncertainty,
            })
        if len({item["label"].casefold() for item in entities}) != len(entities):
            raise SystemExit(f"duplicate entity labels in {model_id} log")
        sha = image_event["sha256"]
        normalized_images.append({
            "index": image_event["index"],
            "name": image_event.get("name"),
            "image_sha256": sha,
            "entities": entities,
        })
        pair_rows.extend(model_pairwise_probes(sha, entities, spec["nearer_rule"]))
    if len(pair_rows) != 40:
        raise SystemExit(f"{model_id} normalization expected 40 pairs")
    return {
        "schema_version": 1,
        "protocol": NORMALIZATION_PROTOCOL,
        "artifact_kind": "normalized_legacy_dense_depth_probe",
        "model_id": model_id,
        "depth_semantics": spec["depth_semantics"],
        "nearer_rule": spec["nearer_rule"],
        "source_log": {
            "path": str(log_path.resolve()),
            "sha256": log_sha,
            "completed_seconds": complete.get("seconds"),
        },
        "current_local_model": current_model_manifest,
        "runtime_log_binds_weight_hash": False,
        "provenance_limitation": (
            "The legacy runtime log names the model class but does not embed the "
            "weight hash; current local weights are verified independently."
        ),
        "candidate_set_sha256": FROZEN_CANARY_SET_SHA256,
        "images": normalized_images,
        "pairs": pair_rows,
        "training_eligible": False,
        "semantic_audit_eligible": False,
        "automatic_pass": False,
    }


def _pair_map(view: dict[str, Any]) -> dict[tuple[str, tuple[str, ...]], dict[str, Any]]:
    rows = view.get("pairs")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("normalized model view must contain 40 pairs")
    mapped: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in rows:
        key = (row["image_sha256"], tuple(row["pair_key"]))
        if key in mapped:
            raise ValueError(f"duplicate normalized pair key: {key}")
        mapped[key] = row
    return mapped


def join_consensus(
    first_view: dict[str, Any], second_view: dict[str, Any]
) -> list[dict[str, Any]]:
    first = _pair_map(first_view)
    second = _pair_map(second_view)
    if set(first) != set(second):
        raise ValueError("model views do not cover identical frozen entity pairs")
    rows: list[dict[str, Any]] = []
    for key in sorted(first):
        left, right = first[key], second[key]
        agree = (
            left.get("raw_nearer") is not None
            and left.get("raw_nearer") == right.get("raw_nearer")
            and left.get("raw_farther") == right.get("raw_farther")
        )
        both_reliable = bool(
            left.get("within_model_reliable")
            and right.get("within_model_reliable")
        )
        audit_candidate = agree and both_reliable
        rows.append({
            "image_sha256": key[0],
            "pair_key": list(key[1]),
            "models_agree": agree,
            "both_within_model_reliable": both_reliable,
            "audit_candidate": audit_candidate,
            "nearer": left["raw_nearer"] if audit_candidate else None,
            "farther": left["raw_farther"] if audit_candidate else None,
            "status": (
                "human_audit_candidate_only"
                if audit_candidate
                else "reject_or_manual_adjudication"
            ),
            "automatic_pass": False,
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
        "--da2-log",
        type=Path,
        default=ROOT / "logs/da2_indoor_small_cpu_canary_probe.log",
    )
    parser.add_argument(
        "--dpt-log",
        type=Path,
        default=ROOT / "logs/dpt_hybrid_midas_cpu_canary_probe_torch26.log",
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
        "--out", type=Path, default=ROOT / "artifacts/depth_dual_probe_v1/canary_v1"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = _require_artifact_output(args.out)
    ensure_empty_output(out)
    da2_manifest = validate_model_artifact(args.da2_model_dir)
    dpt_manifest = validate_dpt_model_artifact(args.dpt_model_dir)
    da2_view = normalize_probe_log(
        args.da2_log, "da2_indoor_metric_small", da2_manifest
    )
    dpt_view = normalize_probe_log(
        args.dpt_log, "dpt_hybrid_midas", dpt_manifest
    )
    for da2_image, dpt_image in zip(da2_view["images"], dpt_view["images"]):
        da2_entities = [
            (item["label"], item["bbox_2d"]) for item in da2_image["entities"]
        ]
        dpt_entities = [
            (item["label"], item["bbox_2d"]) for item in dpt_image["entities"]
        ]
        if (
            da2_image["image_sha256"] != dpt_image["image_sha256"]
            or da2_entities != dpt_entities
        ):
            raise SystemExit("DA2 and DPT logs do not share exact frozen entities")
    consensus = join_consensus(da2_view, dpt_view)
    out.mkdir(parents=True, exist_ok=True)
    normalized = {
        "da2_indoor_metric_small": _round_floats(da2_view),
        "dpt_hybrid_midas": _round_floats(dpt_view),
    }
    view_files: dict[str, dict[str, Any]] = {}
    for model_id, view in normalized.items():
        path = out / f"{model_id}.json"
        path.write_text(
            json.dumps(view, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        view_files[model_id] = {"path": path.name, "sha256": file_hash(path)}
    consensus = _round_floats(consensus)
    consensus_path = out / "dual_consensus.jsonl"
    consensus_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in consensus),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_kind": "human_audit_candidate_ranking_only",
        "candidate_set_sha256": FROZEN_CANARY_SET_SHA256,
        "models": view_files,
        "thresholds": {
            "min_relative_margin": MIN_RELATIVE_MARGIN,
            "min_uncertainty_ratio": MIN_UNCERTAINTY_RATIO,
        },
        "pairs": {
            "rows": len(consensus),
            "raw_direction_agreements": sum(row["models_agree"] for row in consensus),
            "audit_candidates": sum(row["audit_candidate"] for row in consensus),
            "path": consensus_path.name,
            "sha256": file_hash(consensus_path),
            "content_sha256": content_hash(consensus),
        },
        "automatic_pass": False,
        "semantic_audit_pass": False,
        "training_eligible": False,
        "requires_human_review_of_every_selected_pair": True,
        "known_limitation": (
            "Shared model error remains possible even when directions agree; the "
            "frozen canary contains such examples, so consensus never serves as GT."
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
        "audit_candidates": report["pairs"]["audit_candidates"],
        "automatic_pass": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
