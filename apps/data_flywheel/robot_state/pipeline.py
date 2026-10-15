"""Independent A_robot_state pipeline.

This sidecar pipeline intentionally does not register a new CapabilityTag and does not
write into the existing data/output/flywheel annotations tree.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from apps.data_flywheel.robot_state.worker import RobotStateWorker
from apps.data_flywheel.sources.multi_path import MultiPathSource, SourcesConfig
from apps.data_flywheel.sources.registry import get_source
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.robot_state.pipeline")


def _write_jsonl(handle, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def validate_robot_state_source_config(source_config: str) -> Dict[str, Any]:
    """Validate robot-state sources and count scannable samples without running VLM."""
    config = SourcesConfig.from_yaml(source_config)
    sources = []
    total_images = 0

    for index, sc in enumerate(config.sources, start=1):
        record: Dict[str, Any] = {
            "index": index,
            "path": sc.path,
            "source_type": sc.source_type,
            "exists": os.path.isdir(sc.path),
            "num_images": 0,
            "error": None,
        }
        try:
            kwargs: Dict[str, Any] = {"paths": [sc.path]}
            kwargs.update(sc.filters)
            kwargs.update(sc.extra)
            source = get_source(sc.source_type, **kwargs)
            record["num_images"] = len(source)
        except Exception as exc:
            record["error"] = str(exc)

        total_images += int(record["num_images"])
        sources.append(record)

    return {
        "source_config": source_config,
        "num_sources": len(sources),
        "available_sources": sum(1 for item in sources if item["exists"]),
        "total_images": total_images,
        "ok": bool(sources) and total_images > 0,
        "sources": sources,
    }


def run_robot_state_pipeline(
    source_config: str,
    output_root: str = "data/output/robot_state",
    run_id: Optional[str] = None,
    vlm_url: str = "http://localhost:10301",
    vlm_model: str = "qwen2.5-vl-32b-instruct",
    vlm_client: Optional[VLMClient] = None,
) -> Dict[str, Any]:
    run_id = run_id or f"robot_state_{int(time.time())}"
    output_dir = os.path.join(output_root, run_id)
    annotations_dir = os.path.join(output_dir, "annotations")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    source = MultiPathSource(SourcesConfig.from_yaml(source_config))
    client = vlm_client or VLMClient(base_url=vlm_url, model=vlm_model)
    worker = RobotStateWorker(vlm_client=client)

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    annotations_path = os.path.join(annotations_dir, "A_robot_state.jsonl")
    events_path = os.path.join(logs_dir, "robot_state_events.jsonl")

    total_samples = 0
    total_annotations = 0
    started = time.time()

    with open(manifest_path, "w", encoding="utf-8") as manifest_f, \
            open(annotations_path, "w", encoding="utf-8") as ann_f, \
            open(events_path, "w", encoding="utf-8") as events_f:
        for sample in source:
            total_samples += 1
            annotations = worker.process(sample)
            total_annotations += len(annotations)

            _write_jsonl(manifest_f, {
                "sample_id": sample.sample_id,
                "source_type": sample.source_type,
                "image_path": sample.image_path,
                "pipeline_type": "A_robot_state",
                "num_annotations": len(annotations),
            })
            _write_jsonl(events_f, {
                "event": "sample_processed",
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "num_annotations": len(annotations),
            })
            for ann in annotations:
                _write_jsonl(ann_f, ann)

    elapsed = time.time() - started
    stats = {
        "run_id": run_id,
        "pipeline_type": "A_robot_state",
        "total_samples": total_samples,
        "total_annotations": total_annotations,
        "elapsed_seconds": elapsed,
        "samples_per_second": total_samples / elapsed if elapsed > 0 else 0.0,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "robot_state_stats.json"), "w", encoding="utf-8") as stats_f:
        json.dump(stats, stats_f, ensure_ascii=False, indent=2)

    logger.info(
        "Robot-state run complete: %s samples, %s annotations in %.1fs",
        total_samples, total_annotations, elapsed,
    )
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="A_robot_state sidecar pipeline")
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-root", default="data/output/robot_state")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--vlm-url", default="http://localhost:10301")
    parser.add_argument("--vlm-model", default="qwen2.5-vl-32b-instruct")
    parser.add_argument("--dry", action="store_true", help="Validate source config and exit without VLM calls")
    args = parser.parse_args()

    if args.dry:
        result = validate_robot_state_source_config(args.source_config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["ok"] else 1)

    result = run_robot_state_pipeline(
        source_config=args.source_config,
        output_root=args.output_root,
        run_id=args.run_id,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
