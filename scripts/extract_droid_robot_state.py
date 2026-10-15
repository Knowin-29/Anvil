#!/usr/bin/env python3
"""Extract DROID RLDS wrist frames for A_robot_state ingest.

This script reads a local TFDS/RLDS DROID dataset, saves sampled wrist
camera frames, and writes a JSONL manifest consumed by DroidRobotStateAdapter.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("extract_droid_robot_state")


def _import_tfds():
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "tensorflow and tensorflow_datasets are required. Install them in "
            "rex-vllm-local, e.g. pip install tensorflow-cpu tensorflow-datasets"
        ) from exc
    return tf, tfds


def _to_py(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _sample_indices(length: int, frames_per_episode: int) -> List[int]:
    if length <= 0:
        return []
    if frames_per_episode <= 0 or length <= frames_per_episode:
        return list(range(length))
    if frames_per_episode == 1:
        return [length // 2]
    return sorted(
        {
            round(i * (length - 1) / (frames_per_episode - 1))
            for i in range(frames_per_episode)
        }
    )


def _scalar(value: Any) -> Any:
    value = _to_py(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _write_image(array: Any, output_path: Path) -> None:
    image = Image.fromarray(array.numpy() if hasattr(array, "numpy") else array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def _iter_episodes(data_dir: Path, dataset_name: str) -> Iterable[Dict[str, Any]]:
    _tf, tfds = _import_tfds()
    dataset_dir = data_dir / dataset_name / "1.0.0"
    if dataset_dir.exists():
        builder = tfds.builder_from_directory(str(dataset_dir))
    else:
        builder = tfds.builder(dataset_name, data_dir=str(data_dir))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    for episode in tfds.as_numpy(ds):
        yield episode


def _sequence_len(seq: Any) -> int:
    if isinstance(seq, dict):
        for value in seq.values():
            return _sequence_len(value)
        return 0
    return len(seq)


def _index_nested(value: Any, index: int) -> Any:
    if isinstance(value, dict):
        return {k: _index_nested(v, index) for k, v in value.items()}
    return value[index]


def _materialize_steps(steps: Any) -> List[Dict[str, Any]]:
    if isinstance(steps, dict):
        return [_index_nested(steps, i) for i in range(_sequence_len(steps))]
    return list(steps)


def extract(args: argparse.Namespace) -> Dict[str, Any]:
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    image_dir = output_root / "wrist_images"
    manifest_path = output_root / "droid_100_robot_state_manifest.jsonl"

    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    episode_count = 0
    total_steps = 0

    for episode_idx, episode in enumerate(_iter_episodes(data_dir, args.dataset_name)):
        if args.max_episodes is not None and episode_idx >= args.max_episodes:
            break

        steps = _materialize_steps(episode["steps"])
        step_count = len(steps)
        total_steps += step_count
        episode_count += 1
        indices = _sample_indices(step_count, args.frames_per_episode)

        metadata = episode.get("episode_metadata", {})
        file_path = _to_py(metadata.get("file_path", ""))
        recording_folderpath = _to_py(metadata.get("recording_folderpath", ""))

        for step_index in indices:
            step = steps[step_index]
            obs = step["observation"]
            image_name = f"ep{episode_idx:04d}_step{step_index:05d}_wrist.jpg"
            image_path = image_dir / image_name
            _write_image(obs["wrist_image_left"], image_path)

            action_dict = step.get("action_dict", {})
            row = {
                "source_dataset": "droid_100",
                "episode_id": f"ep{episode_idx:04d}",
                "step_index": step_index,
                "episode_step_count": step_count,
                "image_path": str(image_path),
                "camera_view": "wrist",
                "language_instruction": _to_py(step.get("language_instruction", "")),
                "language_instruction_2": _to_py(step.get("language_instruction_2", "")),
                "language_instruction_3": _to_py(step.get("language_instruction_3", "")),
                "gripper_position": _scalar(obs.get("gripper_position")),
                "action_gripper_position": _scalar(action_dict.get("gripper_position")),
                "action_gripper_velocity": _scalar(action_dict.get("gripper_velocity")),
                "reward": _scalar(step.get("reward")),
                "is_first": bool(_scalar(step.get("is_first"))),
                "is_last": bool(_scalar(step.get("is_last"))),
                "is_terminal": bool(_scalar(step.get("is_terminal"))),
                "file_path": file_path,
                "recording_folderpath": recording_folderpath,
            }
            rows.append(row)

        logger.info(
            "episode=%s steps=%d sampled=%d total_rows=%d",
            episode_idx,
            step_count,
            len(indices),
            len(rows),
        )

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset_name": args.dataset_name,
        "data_dir": str(data_dir),
        "episodes": episode_count,
        "total_steps": total_steps,
        "sampled_wrist_images": len(rows),
        "frames_per_episode": args.frames_per_episode,
        "image_dir": str(image_dir),
        "manifest_path": str(manifest_path),
    }
    summary_path = output_root / "droid_100_robot_state_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Summary: %s", summary_path)
    logger.info("Manifest: %s", manifest_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract DROID wrist frames for A_robot_state")
    parser.add_argument(
        "--data-dir",
        default="/path/to/datasets/downloads/droid",
        help="TFDS data_dir containing droid_100/",
    )
    parser.add_argument("--dataset-name", default="droid_100")
    parser.add_argument(
        "--output-root",
        default="/path/to/datasets/downloads/droid/droid_100_extracted",
    )
    parser.add_argument("--frames-per-episode", type=int, default=5)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()

    summary = extract(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
