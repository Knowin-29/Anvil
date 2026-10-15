#!/usr/bin/env python3
"""Run a hash-bound, CPU-only DA2 metric-depth probe on the frozen canary.

This is a diagnostic artifact generator, not a label generator. It emits robust
per-entity bbox statistics and fail-closed pairwise decisions under ``artifacts/``.
No image or result is sent to an external service, and the output is explicitly
ineligible for training or semantic-audit materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_depth_v4 import load_candidates  # noqa: E402
from tools.generate_depth_v43_pairwise import (  # noqa: E402
    FROZEN_ALL_CANDIDATE_SET_SHA256,
    FROZEN_CANARY_SET_SHA256,
    FROZEN_CANARY_SHA256S,
    FROZEN_REMAINDER_SET_SHA256,
    FROZEN_SEED,
    PROTOCOL as V43_PROTOCOL,
    _candidate_set_sha256,
    _frozen_source_matches,
    _validate_run_audit,
)
from trains.flywheel import prompts, prompts_ext  # noqa: E402
from trains.flywheel.build_depth_semantic_audit_pack import (  # noqa: E402
    content_hash,
    ensure_empty_output,
    file_hash,
)
from trains.flywheel.contracts import grounding  # noqa: E402
from trains.flywheel.validators import validate  # noqa: E402

PROTOCOL = "depth-da2-v1-local-probe"
ARTIFACT_KIND = "local_depth_probe_only"
MODEL_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
MODEL_REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"
EXPECTED_MODEL_FILES = {
    "config.json": "0b1d9fc591693d16b864249a9cfcf5264c8c00be999c1643fa1c64dc105d55f8",
    "preprocessor_config.json": (
        "533b16a60445d7cab5086d39b45f92be45624b977972c62d5984d93e98366063"
    ),
    "model.safetensors": (
        "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3"
    ),
}
CANARY_IMAGES = 8
MAX_ENTITIES = 4
GROUNDING_MIN_CONFIDENCE = 0.85
EROSION_FRACTION = 0.10
MIN_ENTITY_PIXELS = 64
MIN_USED_FRACTION = 0.20
MIN_ABSOLUTE_MARGIN_M = 0.08
MIN_RELATIVE_MARGIN = 0.04
MIN_UNCERTAINTY_RATIO = 1.50


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
                f"invalid {kind} JSON at {path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise SystemExit(f"non-object {kind} row at {path}:{line_number}")
        rows.append(row)
    return rows


def validate_model_artifact(model_dir: Path) -> dict[str, Any]:
    """Validate exact downloaded model files and metric-depth configuration."""
    if not model_dir.is_dir():
        raise SystemExit(f"missing local DA2 model directory: {model_dir}")
    files: dict[str, dict[str, Any]] = {}
    for name, expected_sha in EXPECTED_MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"missing DA2 model file: {path}")
        actual_sha = file_hash(path)
        if actual_sha != expected_sha:
            raise SystemExit(
                f"DA2 model hash mismatch for {name}: {actual_sha} != {expected_sha}"
            )
        files[name] = {"sha256": actual_sha, "bytes": path.stat().st_size}

    config = _load_json(model_dir / "config.json", "DA2 config")
    processor = _load_json(
        model_dir / "preprocessor_config.json", "DA2 processor config"
    )
    if (
        not isinstance(config, dict)
        or config.get("model_type") != "depth_anything"
        or config.get("depth_estimation_type") != "metric"
        or config.get("max_depth") != 20
        or config.get("architectures") != ["DepthAnythingForDepthEstimation"]
    ):
        raise SystemExit("DA2 config is not the frozen indoor metric-depth model")
    if (
        not isinstance(processor, dict)
        or processor.get("image_processor_type") != "DPTImageProcessor"
        or processor.get("size") != {"height": 518, "width": 518}
    ):
        raise SystemExit("DA2 processor config does not match the frozen model")
    return {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "local_path": str(model_dir.resolve()),
        "files": files,
        "depth_estimation_type": "metric",
        "max_depth_m": 20,
    }


def validate_frozen_v43_trace(
    report_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate the full v4.3 call audit before reusing its frozen groundings."""
    report = _load_json(report_path, "v4.3 canary report")
    if (
        not isinstance(report, dict)
        or report.get("protocol") != V43_PROTOCOL
        or report.get("stage") != "canary"
        or report.get("candidate_images") != CANARY_IMAGES
        or tuple(report.get("candidate_image_sha256s", ())) != FROZEN_CANARY_SHA256S
        or report.get("candidate_set_sha256") != FROZEN_CANARY_SET_SHA256
    ):
        raise SystemExit("v4.3 report is not the frozen 8-image canary")
    run_binding = _validate_run_audit(report, report_path)
    response_meta = report["run_audit"]["response_audit"]
    response_path = (report_path.parent / response_meta["path"]).resolve()
    rows = _load_jsonl(response_path, "v4.3 response audit")
    if file_hash(response_path) != response_meta.get("sha256"):
        raise SystemExit("v4.3 response audit changed after validation")
    return report, rows, run_binding


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and 0 <= value[0] < value[2] <= 1000
        and 0 <= value[1] < value[3] <= 1000
    )


def recover_trace_groundings(
    candidates: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recover only v4.3's exactly audited detection/cross-check selections."""
    recovered: list[dict[str, Any]] = []
    for source_candidate in candidates:
        candidate = dict(source_candidate)
        if candidate.get("entities") is not None:
            candidate["grounding_origin"] = "frozen_v3_detection"
            recovered.append(candidate)
            continue

        image_path = str(Path(candidate["image_path"]).resolve())
        image_sha = candidate["image_sha256"]
        common = [
            row
            for row in response_rows
            if row.get("image_sha256") == image_sha
            and row.get("image_path_local_only") == image_path
            and row.get("error_type") == ""
        ]
        detection_rows = [
            row
            for row in common
            if row.get("prompt") == prompts_ext.DETECTION
            and isinstance(row.get("response"), list)
        ]
        if len(detection_rows) != 1:
            raise SystemExit(
                f"expected one frozen detection call for {image_sha}, "
                f"found {len(detection_rows)}"
            )
        raw = detection_rows[0]["response"]
        valid, reason = validate("detection", raw, None)
        if not valid or not 2 <= len(raw) <= 6:
            raise SystemExit(f"invalid frozen detection for {image_sha}: {reason}")
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not item["label"].strip()
            or not _valid_bbox(item.get("bbox_2d") or item.get("bbox"))
            for item in raw
        ):
            raise SystemExit(f"malformed frozen grounding for {image_sha}")

        proposed = [
            grounding(item["label"], item.get("bbox_2d") or item.get("bbox"))
            for item in raw
        ]
        expected_prompt = prompts.cross_check_set(proposed)
        check_rows = [
            row
            for row in common
            if row.get("prompt") == expected_prompt
            and isinstance(row.get("response"), list)
        ]
        if len(check_rows) != 1:
            raise SystemExit(
                f"expected one frozen cross-check call for {image_sha}, "
                f"found {len(check_rows)}"
            )
        checks = check_rows[0]["response"]
        indices = [item.get("index") for item in checks if isinstance(item, dict)]
        if len(indices) != len(raw) or sorted(indices) != list(range(len(raw))):
            raise SystemExit(f"incomplete frozen cross-check for {image_sha}")
        by_index = {item["index"]: item for item in checks}
        keep = [
            index
            for index in range(len(raw))
            if by_index[index].get("match") is True
            and isinstance(by_index[index].get("confidence"), (int, float))
            and not isinstance(by_index[index].get("confidence"), bool)
            and GROUNDING_MIN_CONFIDENCE
            <= float(by_index[index]["confidence"])
            <= 1.0
        ][:MAX_ENTITIES]
        if not 2 <= len(keep) <= MAX_ENTITIES:
            raise SystemExit(f"too few verified frozen entities for {image_sha}")
        entities = [
            {
                "label": raw[index]["label"].strip(),
                "bbox_2d": list(raw[index].get("bbox_2d") or raw[index]["bbox"]),
            }
            for index in keep
        ]
        if len({entity["label"].casefold() for entity in entities}) != len(entities):
            raise SystemExit(f"duplicate frozen entity labels for {image_sha}")
        candidate["entities"] = entities
        candidate["grounding_origin"] = "frozen_v43_run_audit"
        recovered.append(candidate)
    return recovered


def _pixel_box(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    if not _valid_bbox(box) or width <= 0 or height <= 0:
        raise ValueError("invalid normalized bbox or image dimensions")
    x1 = max(0, min(width - 1, math.floor(box[0] * width / 1000)))
    y1 = max(0, min(height - 1, math.floor(box[1] * height / 1000)))
    x2 = max(x1 + 1, min(width, math.ceil(box[2] * width / 1000)))
    y2 = max(y1 + 1, min(height, math.ceil(box[3] * height / 1000)))
    return x1, y1, x2, y2


def robust_entity_depth(
    depth_m: np.ndarray,
    entity: dict[str, Any],
    other_entities: list[dict[str, Any]],
    *,
    erosion_fraction: float = EROSION_FRACTION,
    min_pixels: int = MIN_ENTITY_PIXELS,
    min_used_fraction: float = MIN_USED_FRACTION,
) -> dict[str, Any]:
    """Summarize an eroded bbox while removing smaller overlapping bboxes."""
    if not isinstance(depth_m, np.ndarray) or depth_m.ndim != 2:
        raise ValueError("depth map must be a 2-D numpy array")
    if not np.all(np.isfinite(depth_m)) or np.any(depth_m <= 0):
        raise ValueError("depth map must contain only finite positive metric depths")
    if not 0 <= erosion_fraction < 0.5:
        raise ValueError("erosion_fraction must be in [0, 0.5)")
    if min_pixels < 1 or not 0 < min_used_fraction <= 1:
        raise ValueError("invalid robust-stat sampling thresholds")
    label = entity.get("label")
    box = entity.get("bbox_2d")
    if not isinstance(label, str) or not label.strip() or not _valid_bbox(box):
        raise ValueError("entity must have a non-empty label and valid bbox_2d")

    height, width = depth_m.shape
    x1, y1, x2, y2 = _pixel_box(box, width, height)
    inset_x = max(1, int(math.floor((x2 - x1) * erosion_fraction)))
    inset_y = max(1, int(math.floor((y2 - y1) * erosion_fraction)))
    cx1, cy1, cx2, cy2 = x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(f"eroded bbox is empty for {label}")
    core = depth_m[cy1:cy2, cx1:cx2]
    mask = np.ones(core.shape, dtype=bool)
    own_area = (box[2] - box[0]) * (box[3] - box[1])
    excluded_boxes = 0
    for other in other_entities:
        other_box = other.get("bbox_2d")
        if not _valid_bbox(other_box):
            raise ValueError("other entity has an invalid bbox_2d")
        other_area = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
        if other_area > own_area:
            continue
        ox1, oy1, ox2, oy2 = _pixel_box(other_box, width, height)
        ix1, iy1 = max(cx1, ox1), max(cy1, oy1)
        ix2, iy2 = min(cx2, ox2), min(cy2, oy2)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1 - cy1 : iy2 - cy1, ix1 - cx1 : ix2 - cx1] = False
            excluded_boxes += 1
    core_pixels = int(mask.size)
    used_pixels = int(mask.sum())
    used_fraction = used_pixels / core_pixels
    if used_pixels < min_pixels or used_fraction < min_used_fraction:
        raise ValueError(
            f"insufficient uncontaminated bbox support for {label}: "
            f"pixels={used_pixels}, fraction={used_fraction:.4f}"
        )
    values = core[mask].astype(np.float64, copy=False)
    median = float(np.median(values))
    q10, q25, q75, q90 = (float(value) for value in np.quantile(values, [0.1, 0.25, 0.75, 0.9]))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    iqr = q75 - q25

    block_medians: list[float] = []
    y_edges = np.linspace(0, core.shape[0], 5, dtype=int)
    x_edges = np.linspace(0, core.shape[1], 5, dtype=int)
    for row in range(4):
        for column in range(4):
            block_mask = mask[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            if int(block_mask.sum()) < 8:
                continue
            block = core[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            block_medians.append(float(np.median(block[block_mask])))
    if len(block_medians) < 4:
        raise ValueError(f"insufficient spatial support blocks for {label}")
    block_center = float(np.median(block_medians))
    spatial_sigma = 1.4826 * float(
        np.median(np.abs(np.asarray(block_medians) - block_center))
    )
    uncertainty = max(robust_sigma, 0.5 * iqr, spatial_sigma)
    return {
        "label": label.strip(),
        "bbox_2d": list(box),
        "median_depth_m": median,
        "q10_depth_m": q10,
        "q25_depth_m": q25,
        "q75_depth_m": q75,
        "q90_depth_m": q90,
        "mad_depth_m": mad,
        "robust_sigma_m": robust_sigma,
        "spatial_sigma_m": spatial_sigma,
        "uncertainty_m": uncertainty,
        "core_pixels": core_pixels,
        "used_pixels": used_pixels,
        "used_fraction": used_fraction,
        "excluded_smaller_or_equal_boxes": excluded_boxes,
        "spatial_blocks": len(block_medians),
    }


def pairwise_depth_probes(entity_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build pairwise directions; weak margins are explicitly indeterminate."""
    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(entity_stats, 2):
        first_depth = float(first["median_depth_m"])
        second_depth = float(second["median_depth_m"])
        if first_depth == second_depth:
            raw_nearer = None
            raw_farther = None
        elif first_depth < second_depth:
            raw_nearer, raw_farther = first["label"], second["label"]
        else:
            raw_nearer, raw_farther = second["label"], first["label"]
        margin = abs(first_depth - second_depth)
        relative_margin = margin / max(min(first_depth, second_depth), 1e-12)
        combined_uncertainty = math.hypot(
            float(first["uncertainty_m"]), float(second["uncertainty_m"])
        )
        uncertainty_ratio = (
            margin / combined_uncertainty if combined_uncertainty > 0 else None
        )
        failed: list[str] = []
        if raw_nearer is None:
            failed.append("exact_tie")
        if margin < MIN_ABSOLUTE_MARGIN_M:
            failed.append("absolute_margin")
        if relative_margin < MIN_RELATIVE_MARGIN:
            failed.append("relative_margin")
        if uncertainty_ratio is not None and uncertainty_ratio < MIN_UNCERTAINTY_RATIO:
            failed.append("uncertainty_ratio")
        decisive = not failed
        rows.append({
            "entities": [first["label"], second["label"]],
            "raw_nearer": raw_nearer,
            "raw_farther": raw_farther,
            "decision": "directional_probe" if decisive else "indeterminate",
            "nearer": raw_nearer if decisive else None,
            "farther": raw_farther if decisive else None,
            "absolute_margin_m": margin,
            "relative_margin": relative_margin,
            "combined_uncertainty_m": combined_uncertainty,
            "uncertainty_ratio": uncertainty_ratio,
            "failed_thresholds": failed,
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


def _require_artifact_output(path: Path) -> Path:
    resolved = path.resolve()
    artifacts = (ROOT / "artifacts").resolve()
    try:
        resolved.relative_to(artifacts)
    except ValueError as exc:
        raise SystemExit("DA2 probe output must stay under the repository artifacts/") from exc
    if resolved == artifacts:
        raise SystemExit("DA2 probe output cannot be the artifacts/ root")
    return resolved


def _validate_frozen_candidates(
    candidates: list[dict[str, Any]], provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    canary = candidates[:CANARY_IMAGES]
    if (
        tuple(row["image_sha256"] for row in canary) != FROZEN_CANARY_SHA256S
        or _candidate_set_sha256(canary) != FROZEN_CANARY_SET_SHA256
        or _candidate_set_sha256(candidates) != FROZEN_ALL_CANDIDATE_SET_SHA256
        or _candidate_set_sha256(candidates[CANARY_IMAGES:])
        != FROZEN_REMAINDER_SET_SHA256
        or not _frozen_source_matches(provenance)
    ):
        raise SystemExit("candidate source no longer matches the frozen v4 set")
    for candidate in canary:
        image = Path(candidate["image_path"])
        if not image.is_file() or file_hash(image) != candidate["image_sha256"]:
            raise SystemExit(f"frozen canary image hash mismatch: {image}")
    return canary


def _load_runtime(model_dir: Path, threads: int) -> tuple[Any, Any, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        raise SystemExit("CPU-only invariant failed: CUDA is still visible")
    processor = AutoImageProcessor.from_pretrained(
        model_dir, local_files_only=True
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        model_dir, local_files_only=True
    ).eval().to("cpu")
    if next(model.parameters()).device.type != "cpu":
        raise SystemExit("DA2 model was not loaded on CPU")
    if (
        getattr(model.config, "depth_estimation_type", None) != "metric"
        or getattr(model.config, "max_depth", None) != 20
    ):
        raise SystemExit("loaded DA2 runtime is not indoor metric depth")
    return torch, processor, model


def _predict_depth_m(torch: Any, processor: Any, model: Any, image: Any) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    if any(getattr(value, "device", None).type != "cpu" for value in inputs.values()):
        raise SystemExit("DA2 processor unexpectedly produced a non-CPU tensor")
    with torch.inference_mode():
        predicted = model(**inputs).predicted_depth
    resized = torch.nn.functional.interpolate(
        predicted.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth = resized.detach().cpu().float().numpy()
    if (
        depth.shape != (image.height, image.width)
        or not np.all(np.isfinite(depth))
        or np.any(depth <= 0)
        or float(depth.max()) > 20.2
    ):
        raise SystemExit("DA2 produced an invalid indoor metric-depth map")
    return depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=ROOT / "artifacts/manifests/v3_generation.index"
    )
    parser.add_argument(
        "--detection",
        type=Path,
        default=ROOT / "data/viewer_v3_audit_ready_v34/labels/G_grounding/detection.jsonl",
    )
    parser.add_argument(
        "--bbox",
        type=Path,
        default=ROOT / "data/viewer_v3_audit_ready_v34/labels/G_grounding/bbox.jsonl",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=None,
        help="Defaults to the frozen v2 and v3 semantic-audit manifests.",
    )
    parser.add_argument(
        "--v43-report",
        type=Path,
        default=ROOT / "data/depth_v43_pairwise/canary_v1/generation_report.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/path/to/environment/models/depth-anything/"
            "Depth-Anything-V2-Metric-Indoor-Small-hf"
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/depth_da2_v1/canary_v1"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.threads <= 16:
        raise SystemExit("--threads must be in [1,16]")
    exclusions = args.exclude_manifest or [
        ROOT / "data/semantic_audit_v2/manifest.json",
        ROOT / "data/semantic_audit_v3/manifest.json",
    ]
    model_manifest = validate_model_artifact(args.model_dir)
    candidates, provenance = load_candidates(
        args.detection, args.bbox, args.index, exclusions, FROZEN_SEED
    )
    canary = _validate_frozen_candidates(candidates, provenance)
    v43_report, response_rows, run_binding = validate_frozen_v43_trace(
        args.v43_report
    )
    canary = recover_trace_groundings(canary, response_rows)
    entity_counts = [len(candidate["entities"]) for candidate in canary]
    if entity_counts != [4, 2, 4, 3, 4, 4, 4, 4]:
        raise SystemExit(f"unexpected frozen canary entity counts: {entity_counts}")
    planned_pairs = sum(
        len(list(itertools.combinations(candidate["entities"], 2)))
        for candidate in canary
    )
    if planned_pairs != 40:
        raise SystemExit(f"expected exactly 40 frozen canary pairs, got {planned_pairs}")

    plan = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_kind": ARTIFACT_KIND,
        "training_eligible": False,
        "semantic_audit_eligible": False,
        "external_uploads": 0,
        "runtime": "cpu_only_offline",
        "candidate_images": len(canary),
        "candidate_set_sha256": _candidate_set_sha256(canary),
        "entity_counts": entity_counts,
        "planned_pairs": planned_pairs,
        "model": model_manifest,
        "v43_trace": {
            "report_path": str(args.v43_report.resolve()),
            "report_sha256": file_hash(args.v43_report),
            **run_binding,
        },
    }
    if args.check_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    out = _require_artifact_output(args.out)
    ensure_empty_output(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch, processor, model = _load_runtime(args.model_dir, args.threads)
    from PIL import Image

    image_results: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for image_index, candidate in enumerate(canary):
        with Image.open(candidate["image_path"]) as source_image:
            image = source_image.convert("RGB")
        depth = _predict_depth_m(torch, processor, model, image)
        entity_stats: list[dict[str, Any]] = []
        for entity_index, entity in enumerate(candidate["entities"]):
            others = [
                other
                for other_index, other in enumerate(candidate["entities"])
                if other_index != entity_index
            ]
            entity_stats.append(robust_entity_depth(depth, entity, others))
        pairs = pairwise_depth_probes(entity_stats)
        for pair_index, pair in enumerate(pairs):
            pair_results.append({
                "image_index": image_index,
                "pair_index": pair_index,
                "image_sha256": candidate["image_sha256"],
                "image_name": Path(candidate["image_path"]).name,
                **pair,
            })
        image_results.append({
            "image_index": image_index,
            "image_path_local_only": str(Path(candidate["image_path"]).resolve()),
            "image_sha256": candidate["image_sha256"],
            "width": image.width,
            "height": image.height,
            "grounding_origin": candidate["grounding_origin"],
            "depth_range_m": [float(depth.min()), float(depth.max())],
            "entities": entity_stats,
        })
    if len(pair_results) != planned_pairs:
        raise SystemExit("DA2 runtime did not emit every planned pair")

    image_results = _round_floats(image_results)
    pair_results = _round_floats(pair_results)
    pair_path = out / "pairwise_probe.jsonl"
    pair_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pair_results),
        encoding="utf-8",
    )
    report = {
        **plan,
        "completed": True,
        "elapsed_seconds": round(time.time() - started, 3),
        "thresholds": {
            "erosion_fraction": EROSION_FRACTION,
            "min_entity_pixels": MIN_ENTITY_PIXELS,
            "min_used_fraction": MIN_USED_FRACTION,
            "min_absolute_margin_m": MIN_ABSOLUTE_MARGIN_M,
            "min_relative_margin": MIN_RELATIVE_MARGIN,
            "min_uncertainty_ratio": MIN_UNCERTAINTY_RATIO,
        },
        "uncertainty_semantics": (
            "max(1.4826*MAD, 0.5*IQR, 1.4826*MAD of 4x4 block medians); "
            "within-bbox dispersion only, not calibrated model uncertainty"
        ),
        "requires_independent_second_signal": True,
        "images": image_results,
        "pairs": {
            "path": pair_path.name,
            "rows": len(pair_results),
            "directional_probe_rows": sum(
                row["decision"] == "directional_probe" for row in pair_results
            ),
            "indeterminate_rows": sum(
                row["decision"] == "indeterminate" for row in pair_results
            ),
            "sha256": file_hash(pair_path),
            "content_sha256": content_hash(pair_results),
        },
        "safety": {
            "images_uploaded": 0,
            "network_enabled": False,
            "writes_training_records": False,
            "writes_semantic_audit_records": False,
        },
        "v43_canary_machine_gate_passed": v43_report["canary_gate"]["passed"],
    }
    report_path = out / "probe_report.json"
    report_path.write_text(
        json.dumps(_round_floats(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "protocol": PROTOCOL,
        "report": str(report_path),
        "pairs": len(pair_results),
        "directional": report["pairs"]["directional_probe_rows"],
        "indeterminate": report["pairs"]["indeterminate_rows"],
        "training_eligible": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
