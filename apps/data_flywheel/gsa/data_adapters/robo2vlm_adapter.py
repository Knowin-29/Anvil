"""Robo2VLM adapter — parquet VQA -> A_robot_state labels."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq

from ..a_pipeline_schemas import ARobotStateLabel
from ..taxonomy import ARobotStateSubtype, CameraView, EndEffector, VerifierStatus
from .robot_state_ingest import RobotStateDatasetAdapter, finalize_robot_state_label
from .robot_state_text_rules import infer_state_from_text

logger = logging.getLogger("gsa.data_adapters.robo2vlm")


def _resolve_parquet_path(dataset_path: Path, config: Dict[str, Any]) -> Optional[Path]:
    parquet_glob = config.get("parquet_glob", "data/*.parquet")
    matches = sorted(dataset_path.glob(parquet_glob))
    if matches:
        return matches[0]
    fallback = dataset_path / "data" / "train-00000-of-00001.parquet"
    return fallback if fallback.exists() else None


def _export_image(row_idx: int, image_struct: dict, export_dir: Path) -> Optional[str]:
    if not image_struct:
        return None
    out_path = export_dir / f"{row_idx}.jpg"
    if out_path.exists():
        return str(out_path)
    raw = image_struct.get("bytes")
    if not raw:
        return None
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return str(out_path)


class Robo2VLMAdapter(RobotStateDatasetAdapter):
    name = "robo2vlm"

    def discover(self, dataset_path: Path, max_samples: int, config: Dict[str, Any]) -> List[dict]:
        parquet_path = _resolve_parquet_path(dataset_path, config)
        if parquet_path is None:
            logger.warning("Robo2VLM parquet not found under %s", dataset_path)
            return []

        table = pq.read_table(
            parquet_path,
            columns=["question", "choices", "correct_answer", "category", "image"],
        )
        samples: List[dict] = []
        limit = min(max_samples, table.num_rows)
        for idx in range(limit):
            samples.append(
                {
                    "_row_idx": idx,
                    "question": table["question"][idx].as_py(),
                    "choices": table["choices"][idx].as_py(),
                    "correct_answer": table["correct_answer"][idx].as_py(),
                    "category": table["category"][idx].as_py(),
                    "image": table["image"][idx].as_py(),
                }
            )
        return samples

    def convert(self, samples: List[dict], dataset_path: Path, config: Dict[str, Any]) -> List[ARobotStateLabel]:
        labels: List[ARobotStateLabel] = []
        export_root = Path(
            config.get("extracted_images_dir") or config.get("images_root") or "extracted_images"
        )
        if not export_root.is_absolute():
            export_root = dataset_path / export_root

        for sample in samples:
            question = str(sample.get("question") or "")
            mapped = infer_state_from_text(question)
            if mapped is None:
                continue

            state_type, state_value = mapped
            row_idx = int(sample.get("_row_idx", 0))
            image_path = _export_image(row_idx, sample.get("image") or {}, export_root)
            image_exists = bool(image_path and Path(image_path).exists())

            label = ARobotStateLabel(
                sample_id=f"robo2vlm_{row_idx}",
                object_id=f"gripper_r2v_{row_idx}",
                camera_view=CameraView.EXTERNAL.value,
                end_effector=EndEffector.PARALLEL_JAW.value,
                state_type=state_type,
                state_value=state_value,
                confidence=0.75 if image_exists else 0.5,
            )
            finalized = finalize_robot_state_label(
                label,
                image_path=image_path,
                source_dataset=self.name,
                image_exists=image_exists,
            )
            labels.append(finalized)

        logger.info("Converted %d Robo2VLM labels", len(labels))
        return labels
