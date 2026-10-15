#!/usr/bin/env python3
"""Run a hash-bound, CPU-only DPT inverse-depth probe on the frozen canary.

The output is a diagnostic model view for anomaly detection against upcoming
ScanNet sensor-depth labels. It is not a ground-truth generator, cannot PASS a
semantic audit, and never writes training or ScanNet records.
"""
from __future__ import annotations

import argparse
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
    FROZEN_SEED,
    _candidate_set_sha256,
)
from tools.probe_depth_da2_v1 import (  # noqa: E402
    EROSION_FRACTION,
    MIN_ENTITY_PIXELS,
    MIN_USED_FRACTION,
    _require_artifact_output,
    _round_floats,
    _validate_frozen_candidates,
    recover_trace_groundings,
    robust_entity_depth,
    validate_frozen_v43_trace,
)
from trains.flywheel.build_depth_semantic_audit_pack import (  # noqa: E402
    content_hash,
    ensure_empty_output,
    file_hash,
)

PROTOCOL = "depth-dpt-v1-local-probe"
ARTIFACT_KIND = "local_inverse_depth_probe_only"
INTENDED_USE = "upcoming_scannet_sensor_depth_label_anomaly_detection_only"
MODEL_REPO = "Intel/dpt-hybrid-midas"
MODEL_REVISION = "11eaf7a1cf4bd70740697dbc216f98980c0aeb03"
REQUIRED_INTERPRETER = Path("/path/to/workspace/lvmem/envs/lvmem_cf/bin/python")
EXPECTED_MODEL_FILES = {
    "config.json": "d469783f4a0bbfdbdafb9af7b193480ce9d87c5687984056ccc3738e5367ca66",
    "preprocessor_config.json": (
        "4da22e615634b6316e2564136943b8550ebaa77d4e300004e8a7e99c92b68c6a"
    ),
    "pytorch_model.bin": (
        "b6c4d44f9d96ca3fa76dd3bbb153989a60b4ad5526559f3c598562a368d687ec"
    ),
}
EXPECTED_ENTITY_COUNTS = [4, 2, 4, 3, 4, 4, 4, 4]
MIN_RELATIVE_MARGIN = 0.04
MIN_UNCERTAINTY_RATIO = 1.50


def _load_json(path: Path, kind: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {kind}: {path}: {exc}") from exc


def validate_model_artifact(model_dir: Path) -> dict[str, Any]:
    """Validate the exact DPT files used by the CPU runtime."""
    if not model_dir.is_dir():
        raise SystemExit(f"missing local DPT model directory: {model_dir}")
    files: dict[str, dict[str, Any]] = {}
    for name, expected_sha in EXPECTED_MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"missing DPT model file: {path}")
        actual_sha = file_hash(path)
        if actual_sha != expected_sha:
            raise SystemExit(
                f"DPT model hash mismatch for {name}: {actual_sha} != {expected_sha}"
            )
        files[name] = {"sha256": actual_sha, "bytes": path.stat().st_size}
    config = _load_json(model_dir / "config.json", "DPT config")
    processor = _load_json(
        model_dir / "preprocessor_config.json", "DPT processor config"
    )
    if (
        not isinstance(config, dict)
        or config.get("model_type") != "dpt"
        or config.get("architectures") != ["DPTForDepthEstimation"]
    ):
        raise SystemExit("DPT config does not match Intel/dpt-hybrid-midas")
    if (
        not isinstance(processor, dict)
        or processor.get("image_processor_type") != "DPTImageProcessor"
        or processor.get("size") != {"height": 384, "width": 384}
    ):
        raise SystemExit("DPT processor config does not match the frozen model")
    return {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "local_path": str(model_dir.resolve()),
        "files": files,
        "depth_semantics": "relative_inverse_depth",
        "nearer_rule": "higher_value_is_nearer",
    }


def validate_required_interpreter(executable: str | Path = sys.executable) -> str:
    actual = Path(executable).resolve()
    expected = REQUIRED_INTERPRETER.resolve()
    if actual != expected:
        raise SystemExit(
            "DPT pickle weights require the pinned torch-2.6 interpreter: "
            f"{REQUIRED_INTERPRETER} (got {executable})"
        )
    return str(actual)


def inverse_depth_entity_stats(
    inverse_depth: np.ndarray,
    entity: dict[str, Any],
    other_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the shared robust bbox sampler and make inverse units explicit."""
    metric_named = robust_entity_depth(inverse_depth, entity, other_entities)
    return {
        "label": metric_named["label"],
        "bbox_2d": metric_named["bbox_2d"],
        "median_value": metric_named["median_depth_m"],
        "q10_value": metric_named["q10_depth_m"],
        "q25_value": metric_named["q25_depth_m"],
        "q75_value": metric_named["q75_depth_m"],
        "q90_value": metric_named["q90_depth_m"],
        "mad_value": metric_named["mad_depth_m"],
        "robust_sigma": metric_named["robust_sigma_m"],
        "spatial_sigma": metric_named["spatial_sigma_m"],
        "uncertainty": metric_named["uncertainty_m"],
        "value_semantics": "relative_inverse_depth",
        "core_pixels": metric_named["core_pixels"],
        "used_pixels": metric_named["used_pixels"],
        "used_fraction": metric_named["used_fraction"],
        "excluded_smaller_or_equal_boxes": metric_named[
            "excluded_smaller_or_equal_boxes"
        ],
        "spatial_blocks": metric_named["spatial_blocks"],
    }


def pairwise_inverse_depth_probes(
    entity_stats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit higher-is-nearer directions only after scale-free robust gates."""
    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(entity_stats, 2):
        first_value = float(first["median_value"])
        second_value = float(second["median_value"])
        if first_value == second_value:
            raw_nearer = None
            raw_farther = None
        elif first_value > second_value:
            raw_nearer, raw_farther = first["label"], second["label"]
        else:
            raw_nearer, raw_farther = second["label"], first["label"]
        margin = abs(first_value - second_value)
        relative_margin = margin / max(abs(first_value), abs(second_value), 1e-12)
        combined_uncertainty = math.hypot(
            float(first["uncertainty"]), float(second["uncertainty"])
        )
        uncertainty_ratio = (
            margin / combined_uncertainty if combined_uncertainty > 0 else None
        )
        failed: list[str] = []
        if raw_nearer is None:
            failed.append("exact_tie")
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
            "absolute_margin_value": margin,
            "relative_margin": relative_margin,
            "combined_uncertainty": combined_uncertainty,
            "uncertainty_ratio": uncertainty_ratio,
            "failed_thresholds": failed,
            "value_semantics": "relative_inverse_depth",
        })
    return rows


def _load_runtime(model_dir: Path, threads: int) -> tuple[Any, Any, Any, dict[str, Any]]:
    interpreter = validate_required_interpreter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if not torch.__version__.startswith("2.6."):
        raise SystemExit(f"DPT runtime requires torch 2.6.x, got {torch.__version__}")
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
        raise SystemExit("DPT model was not loaded on CPU")
    if type(model).__name__ != "DPTForDepthEstimation":
        raise SystemExit(f"unexpected DPT runtime class: {type(model).__name__}")
    runtime = {
        "interpreter": interpreter,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cpu",
        "cuda_available": False,
        "hf_offline": True,
        "threads": threads,
    }
    return torch, processor, model, runtime


def _predict_inverse_depth(
    torch: Any, processor: Any, model: Any, image: Any
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    for value in inputs.values():
        device = getattr(value, "device", None)
        if device is not None and device.type != "cpu":
            raise SystemExit("DPT processor unexpectedly produced a non-CPU tensor")
    with torch.inference_mode():
        predicted = model(**inputs).predicted_depth
    resized = torch.nn.functional.interpolate(
        predicted.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    values = resized.detach().cpu().float().numpy()
    if (
        values.shape != (image.height, image.width)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise SystemExit("DPT produced an invalid relative inverse-depth map")
    return values


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
        "--exclude-manifest", type=Path, action="append", default=None
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
            "/path/to/environment/models/depth-estimation/Intel-dpt-hybrid-midas"
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/depth_dpt_v1/canary_v1"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_required_interpreter()
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
    if entity_counts != EXPECTED_ENTITY_COUNTS:
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
        "intended_use": INTENDED_USE,
        "training_eligible": False,
        "semantic_audit_eligible": False,
        "ground_truth_eligible": False,
        "automatic_pass": False,
        "external_uploads": 0,
        "runtime": "pinned_torch26_cpu_only_offline",
        "required_interpreter": str(REQUIRED_INTERPRETER),
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
    torch, processor, model, runtime_details = _load_runtime(
        args.model_dir, args.threads
    )
    from PIL import Image

    image_results: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for image_index, candidate in enumerate(canary):
        with Image.open(candidate["image_path"]) as source_image:
            image = source_image.convert("RGB")
        inverse_depth = _predict_inverse_depth(torch, processor, model, image)
        entity_stats: list[dict[str, Any]] = []
        for entity_index, entity in enumerate(candidate["entities"]):
            others = [
                other
                for other_index, other in enumerate(candidate["entities"])
                if other_index != entity_index
            ]
            entity_stats.append(
                inverse_depth_entity_stats(inverse_depth, entity, others)
            )
        pairs = pairwise_inverse_depth_probes(entity_stats)
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
            "inverse_depth_range": [
                float(inverse_depth.min()),
                float(inverse_depth.max()),
            ],
            "entities": entity_stats,
        })
    if len(pair_results) != planned_pairs:
        raise SystemExit("DPT runtime did not emit every planned pair")
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
        "runtime_details": runtime_details,
        "thresholds": {
            "erosion_fraction": EROSION_FRACTION,
            "min_entity_pixels": MIN_ENTITY_PIXELS,
            "min_used_fraction": MIN_USED_FRACTION,
            "min_relative_margin": MIN_RELATIVE_MARGIN,
            "min_uncertainty_ratio": MIN_UNCERTAINTY_RATIO,
        },
        "uncertainty_semantics": (
            "max(1.4826*MAD, 0.5*IQR, 1.4826*MAD of 4x4 block medians); "
            "within-bbox dispersion in relative inverse-depth units"
        ),
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
            "writes_scannet_records": False,
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
        "intended_use": INTENDED_USE,
        "automatic_pass": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
