#!/usr/bin/env python3
"""Extract FMB wrist frames for A_robot_state ingest.

Reads official HuggingFace FMB .npy trajectories, saves sampled wrist camera
frames, and writes a JSONL manifest consumed by FmbRobotStateAdapter.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("extract_fmb_robot_state")


def _to_py(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sample_indices(length: int, frames_per_trajectory: int) -> List[int]:
    if length <= 0:
        return []
    if frames_per_trajectory <= 0 or length <= frames_per_trajectory:
        return list(range(length))
    if frames_per_trajectory == 1:
        return [length // 2]
    return sorted(
        {
            round(i * (length - 1) / (frames_per_trajectory - 1))
            for i in range(frames_per_trajectory)
        }
    )


def _load_trajectory(path: Path) -> Dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        loaded = loaded.item()
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected dict trajectory in {path}, got {type(loaded)!r}")
    return loaded


def _array_at(traj: Dict[str, Any], key: str, index: int) -> Any:
    value = traj.get(key)
    if value is None:
        return None
    try:
        return value[index]
    except Exception:
        return None


def _norm(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
        return float(np.linalg.norm(arr))
    except Exception:
        return None


def _scalar_at(traj: Dict[str, Any], key: str, index: int) -> Any:
    return _to_py(_array_at(traj, key, index))


def _image_length(traj: Dict[str, Any], image_key: str) -> int:
    image_seq = traj.get(image_key)
    if image_seq is None:
        return 0
    return int(len(image_seq))


def _write_bgr_image(array: Any, output_path: Path) -> None:
    image = np.asarray(array)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB/BGR image, got shape={image.shape}")
    image = image[..., ::-1]  # FMB HF README states RGB arrays are saved in BGR format.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(output_path, quality=95)


def _iter_trajectory_files(input_root: Path, glob_pattern: str, max_trajectories: Optional[int]) -> Iterable[Path]:
    count = 0
    for path in sorted(input_root.glob(glob_pattern)):
        if not path.is_file() or path.suffix.lower() != ".npy":
            continue
        yield path
        count += 1
        if max_trajectories is not None and count >= max_trajectories:
            break


def _task_family(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if "multi_object_manipulation_dataset" in parts:
        return "multi_object"
    if "single_object_manipulation_dataset" in parts:
        return "single_object"
    return "unknown"


def _object_metadata(traj: Dict[str, Any], path: Path, index: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "task_family": _task_family(path),
        "trajectory_file": str(path),
    }
    object_info = traj.get("object_info")
    if isinstance(object_info, dict):
        info["object_info"] = {str(k): _to_py(v) for k, v in object_info.items()}
    object_id = _scalar_at(traj, "object_id", index)
    if object_id is not None:
        info["object_id"] = object_id
    return info


def extract(args: argparse.Namespace) -> Dict[str, Any]:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    image_dir = output_root / "wrist_images"
    manifest_path = output_root / "fmb_robot_state_manifest.jsonl"

    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    trajectory_count = 0
    total_steps = 0
    skipped: List[str] = []

    for traj_idx, traj_path in enumerate(_iter_trajectory_files(input_root, args.glob, args.max_trajectories)):
        try:
            traj = _load_trajectory(traj_path)
        except Exception as exc:
            logger.warning("Skip unreadable FMB trajectory %s: %s", traj_path, exc)
            skipped.append(str(traj_path))
            continue

        step_count = _image_length(traj, args.image_key)
        if step_count <= 0:
            logger.warning("Skip FMB trajectory without %s: %s", args.image_key, traj_path)
            skipped.append(str(traj_path))
            continue

        trajectory_count += 1
        total_steps += step_count
        indices = _sample_indices(step_count, args.frames_per_trajectory)
        rel_stem = traj_path.relative_to(input_root).with_suffix("").as_posix().replace("/", "__")

        for step_index in indices:
            image_name = f"traj{traj_idx:04d}_{rel_stem}_step{step_index:05d}_wrist1.jpg"
            image_path = image_dir / image_name
            _write_bgr_image(_array_at(traj, args.image_key, step_index), image_path)

            tcp_force = _array_at(traj, "obs/tcp_force", step_index)
            tcp_torque = _array_at(traj, "obs/tcp_torque", step_index)
            action = _array_at(traj, "action", step_index)
            if action is None:
                action = _array_at(traj, "actions", step_index)
            action_gripper = None
            if action is not None:
                try:
                    action_gripper = _to_py(np.asarray(action)[-1])
                except Exception:
                    action_gripper = None

            row = {
                "source_dataset": "fmb",
                "trajectory_id": rel_stem,
                "trajectory_index": traj_idx,
                "step_index": step_index,
                "trajectory_step_count": step_count,
                "image_path": str(image_path),
                "camera_view": "wrist",
                "image_key": args.image_key,
                "has_wrist_2": "obs/wrist_2" in traj,
                "primitive": _scalar_at(traj, "primitive", step_index),
                "gripper_pose": _scalar_at(traj, "obs/gripper_pose", step_index),
                "action_gripper": action_gripper,
                "tcp_force": _to_py(tcp_force) if tcp_force is not None else None,
                "tcp_torque": _to_py(tcp_torque) if tcp_torque is not None else None,
                "tcp_force_norm": _norm(tcp_force),
                "tcp_torque_norm": _norm(tcp_torque),
                "metadata": _object_metadata(traj, traj_path, step_index),
            }
            rows.append(row)

        logger.info(
            "trajectory=%s steps=%d sampled=%d total_rows=%d",
            traj_path,
            step_count,
            len(indices),
            len(rows),
        )

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset_name": "fmb",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "trajectories": trajectory_count,
        "total_steps": total_steps,
        "sampled_wrist_images": len(rows),
        "frames_per_trajectory": args.frames_per_trajectory,
        "image_key": args.image_key,
        "image_dir": str(image_dir),
        "manifest_path": str(manifest_path),
        "skipped": skipped,
    }
    summary_path = output_root / "fmb_robot_state_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Summary: %s", summary_path)
    logger.info("Manifest: %s", manifest_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract FMB wrist frames for A_robot_state")
    parser.add_argument(
        "--input-root",
        default="/path/to/datasets/downloads/fmb/raw_hf",
        help="Root containing official FMB HuggingFace .npy trajectories.",
    )
    parser.add_argument(
        "--output-root",
        default="/path/to/datasets/downloads/fmb/fmb_extracted",
    )
    parser.add_argument("--glob", default="**/*.npy")
    parser.add_argument("--image-key", default="obs/wrist_1")
    parser.add_argument("--frames-per-trajectory", type=int, default=5)
    parser.add_argument("--max-trajectories", type=int)
    args = parser.parse_args()

    summary = extract(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
