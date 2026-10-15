"""Correspondence pipeline — Method G geometric reprojection for posed RGB-D.

Minimal implementation (volc_jun12 framework):
1. Load CA-1M scene (RGB + GT depth + intrinsics + pose)
2. Sample frame pairs with covisibility > 25%
3. For each pair, sample query pixels and reproject to target
4. Validate via depth consistency check
5. Output CorrespondenceSample JSONL

Based on 02_pipeline2_correspondence.md §4 (Method G).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from apps.data_flywheel.gsa.correspondence_schema import (
    BaselineCategory,
    CorrespondenceQA,
    CorrespondenceSample,
    ImageMeta,
    LabelSource,
    validate_correspondence_sample,
)

logger = logging.getLogger("correspondence_pipeline")

COORD_NORM = 1000
DEPTH_CONSISTENCY_TAU = 0.05  # 5cm threshold
COVISIBILITY_THRESHOLD = 0.25
NEGATIVE_SAMPLE_RATIO = 0.15
QAS_PER_PAIR = 10


def load_depth_png(path: str) -> Optional[np.ndarray]:
    """Load 16-bit depth PNG. Returns depth in meters (assuming mm encoding)."""
    try:
        from PIL import Image
        img = Image.open(path)
        depth = np.array(img, dtype=np.float32)
        # CA-1M: 16-bit PNG in mm (to be confirmed per §4.4 unit verification)
        depth = depth / 1000.0
        return depth
    except Exception as e:
        logger.warning(f"Failed to load depth: {path}: {e}")
        return None


def load_intrinsics(k_json_path: str) -> Optional[np.ndarray]:
    """Load 3x3 intrinsics matrix from K.json."""
    try:
        with open(k_json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            K = np.array(data, dtype=np.float64).reshape(3, 3)
        elif isinstance(data, dict):
            fx = data.get("fx", data.get("focal_x", 500))
            fy = data.get("fy", data.get("focal_y", 500))
            cx = data.get("cx", data.get("principal_x", 320))
            cy = data.get("cy", data.get("principal_y", 240))
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        else:
            return None
        return K
    except Exception as e:
        logger.warning(f"Failed to load K: {k_json_path}: {e}")
        return None


def load_pose(rt_json_path: str) -> Optional[np.ndarray]:
    """Load 4x4 camera-to-world (or world-to-camera) pose from RT.json."""
    try:
        with open(rt_json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            RT = np.array(data, dtype=np.float64).reshape(4, 4)
        else:
            return None
        return RT
    except Exception as e:
        logger.warning(f"Failed to load RT: {rt_json_path}: {e}")
        return None


def reproject_pixel(
    p1: Tuple[int, int],
    D1: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    T_1to2: np.ndarray,
    D2: np.ndarray,
    tau: float = DEPTH_CONSISTENCY_TAU,
) -> Tuple[Optional[Tuple[int, int]], bool, str]:
    """Reproject pixel from image1 to image2 using Method G.

    Returns: (p2_or_None, is_visible, reason)
    """
    u1, v1 = p1
    h1, w1 = D1.shape
    h2, w2 = D2.shape

    # Get depth at p1 (nearest neighbor in depth resolution)
    du1 = int(round(v1 * h1 / max(h1, 1)))
    dv1 = int(round(u1 * w1 / max(w1, 1)))
    du1 = min(max(du1, 0), h1 - 1)
    dv1 = min(max(dv1, 0), w1 - 1)
    d1 = D1[du1, dv1]
    if d1 <= 0 or d1 > 20.0:
        return None, False, "invalid_depth_src"

    # Back-project to 3D in camera1 frame
    K1_inv = np.linalg.inv(K1)
    p1_h = np.array([u1, v1, 1.0])
    X1 = d1 * (K1_inv @ p1_h)

    # Transform to camera2 frame
    X1_h = np.array([X1[0], X1[1], X1[2], 1.0])
    X2_h = T_1to2 @ X1_h
    X2 = X2_h[:3]

    # Check in front of camera2
    if X2[2] <= 0:
        return None, False, "behind_camera"

    # Project to image2
    p2_h = K2 @ X2
    u2 = p2_h[0] / p2_h[2]
    v2 = p2_h[1] / p2_h[2]

    # Check within image bounds
    # Use depth image dimensions for consistency check
    if u2 < 0 or u2 >= w2 or v2 < 0 or v2 >= h2:
        return None, False, "out_of_frame"

    # Depth consistency check (nearest neighbor, no bilinear)
    du2 = int(round(v2))
    dv2 = int(round(u2))
    du2 = min(max(du2, 0), h2 - 1)
    dv2 = min(max(dv2, 0), w2 - 1)
    d2_actual = D2[du2, dv2]
    z2_projected = X2[2]

    if d2_actual <= 0:
        return None, False, "invalid_depth_tgt"

    if abs(z2_projected - d2_actual) > tau:
        # Occluded — depth inconsistency
        return (int(round(u2)), int(round(v2))), False, "occluded"

    return (int(round(u2)), int(round(v2))), True, "visible"


def normalize_coord(x: int, y: int, w: int, h: int, norm: int = COORD_NORM) -> List[int]:
    """Normalize pixel coordinate to [0, norm) range."""
    nx = int(round(x * norm / w))
    ny = int(round(y * norm / h))
    return [min(max(nx, 0), norm - 1), min(max(ny, 0), norm - 1)]


def compute_covisibility(
    D1: np.ndarray, K1: np.ndarray, K2: np.ndarray,
    T_1to2: np.ndarray, D2: np.ndarray,
    n_samples: int = 200,
) -> float:
    """Estimate covisibility by sampling random pixels from image1."""
    h1, w1 = D1.shape
    rng = np.random.default_rng(42)
    valid = 0
    visible = 0

    us = rng.integers(0, w1, size=n_samples)
    vs = rng.integers(0, h1, size=n_samples)

    for u, v in zip(us, vs):
        d = D1[v, u]
        if d <= 0 or d > 20.0:
            continue
        valid += 1
        _, is_vis, _ = reproject_pixel((u, v), D1, K1, K2, T_1to2, D2)
        if is_vis:
            visible += 1

    return visible / max(valid, 1)


def generate_correspondence_for_pair(
    img1_path: str,
    img2_path: str,
    D1: np.ndarray,
    D2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    T_1to2: np.ndarray,
    covisibility: float,
    scene_id: str,
    frame1_id: str,
    frame2_id: str,
    n_queries: int = QAS_PER_PAIR,
) -> Optional[CorrespondenceSample]:
    """Generate a correspondence sample for one frame pair."""
    h1, w1 = D1.shape
    h2, w2 = D2.shape
    rng = np.random.default_rng(hash(f"{scene_id}_{frame1_id}_{frame2_id}") % (2**31))

    qas: List[CorrespondenceQA] = []
    attempts = 0
    max_attempts = n_queries * 10
    n_negative_target = max(1, int(n_queries * NEGATIVE_SAMPLE_RATIO))
    n_positive_target = n_queries - n_negative_target

    positives = []
    negatives = []

    while attempts < max_attempts and (len(positives) < n_positive_target or len(negatives) < n_negative_target):
        attempts += 1
        u1 = rng.integers(0, w1)
        v1 = rng.integers(0, h1)

        p2, is_visible, reason = reproject_pixel(
            (u1, v1), D1, K1, K2, T_1to2, D2
        )

        if reason in ("invalid_depth_src", "invalid_depth_tgt"):
            continue

        src_norm = normalize_coord(u1, v1, w1, h1)

        if is_visible and p2 is not None and len(positives) < n_positive_target:
            dst_norm = normalize_coord(p2[0], p2[1], w2, h2)
            positives.append(CorrespondenceQA(
                src=src_norm, dst=dst_norm, visible=True,
                label_source=LabelSource.GEOMETRIC_REPROJECTION.value,
            ))
        elif not is_visible and reason == "occluded" and len(negatives) < n_negative_target:
            negatives.append(CorrespondenceQA(
                src=src_norm, dst=None, visible=False,
                invisible_reason="occluded",
                label_source=LabelSource.GEOMETRIC_REPROJECTION.value,
            ))

    qas = positives + negatives
    if len(qas) < 3:
        return None

    sample = CorrespondenceSample(
        sample_id=f"p2_ca1m_{scene_id}_{frame1_id}_{frame2_id}",
        images=[img1_path, img2_path],
        image_meta=[
            ImageMeta(orig_wh=[w1, h1]),
            ImageMeta(orig_wh=[w2, h2]),
        ],
        covisibility=covisibility,
        qas=qas,
        source={
            "dataset": "CA-1M",
            "scene": scene_id,
            "frames": [frame1_id, frame2_id],
            "baseline": "mid",
        },
    )

    valid, reason = validate_correspondence_sample(sample)
    if not valid:
        logger.debug(f"Sample validation failed: {reason}")
        return None

    return sample


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run correspondence pipeline (CA-1M demo)")
    parser.add_argument("--ca1m-dir", type=str,
                       default="/path/to/datasets/downloads/CA-1M",
                       help="CA-1M root directory")
    parser.add_argument("--scene", type=str, default=None,
                       help="Specific scene ID (default: first available)")
    parser.add_argument("--max-pairs", type=int, default=10,
                       help="Max frame pairs to process")
    parser.add_argument("--output", type=str, default="data/output/correspondence_demo.jsonl",
                       help="Output JSONL path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ca1m_root = Path(args.ca1m_dir)
    if not ca1m_root.exists():
        logger.error(f"CA-1M not found: {ca1m_root}")
        logger.info("Correspondence pipeline framework is ready but CA-1M data not accessible.")
        logger.info("To run: ensure CA-1M is mounted and specify --ca1m-dir")
        exit(0)

    # Find scenes
    scenes = sorted([d.name for d in ca1m_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if args.scene:
        scenes = [args.scene] if args.scene in scenes else []
    if not scenes:
        logger.error("No scenes found")
        exit(1)

    scene_id = scenes[0]
    scene_dir = ca1m_root / scene_id
    logger.info(f"Processing scene: {scene_id}")

    # Find frame timestamps
    frames = sorted([d.name.replace(".wide", "") for d in scene_dir.iterdir()
                    if d.is_dir() and d.name.endswith(".wide")])
    logger.info(f"  Found {len(frames)} frames")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_written = 0

    with open(output_path, "w") as fout:
        for i in range(min(args.max_pairs, len(frames) - 1)):
            f1 = frames[i]
            f2 = frames[min(i + 5, len(frames) - 1)]  # skip frames for baseline
            if f1 == f2:
                continue

            # Load data for frame 1
            d1_path = scene_dir / f"{f1}.gt" / "depth.png"
            k1_path = scene_dir / f"{f1}.wide" / "image" / "K.json"
            rt1_path = scene_dir / f"{f1}.gt" / "RT.json"
            img1_path = str(scene_dir / f"{f1}.wide" / "image.png")

            # Load data for frame 2
            d2_path = scene_dir / f"{f2}.gt" / "depth.png"
            k2_path = scene_dir / f"{f2}.wide" / "image" / "K.json"
            rt2_path = scene_dir / f"{f2}.gt" / "RT.json"
            img2_path = str(scene_dir / f"{f2}.wide" / "image.png")

            # Check files exist
            required = [d1_path, k1_path, rt1_path, d2_path, k2_path, rt2_path]
            if not all(Path(p).exists() for p in required):
                continue

            D1 = load_depth_png(str(d1_path))
            D2 = load_depth_png(str(d2_path))
            K1 = load_intrinsics(str(k1_path))
            K2 = load_intrinsics(str(k2_path))
            RT1 = load_pose(str(rt1_path))
            RT2 = load_pose(str(rt2_path))

            if any(x is None for x in [D1, D2, K1, K2, RT1, RT2]):
                continue

            # Compute relative transform cam1 -> cam2
            T_1to2 = np.linalg.inv(RT2) @ RT1

            # Compute covisibility
            covis = compute_covisibility(D1, K1, K2, T_1to2, D2)
            if covis < COVISIBILITY_THRESHOLD:
                continue

            sample = generate_correspondence_for_pair(
                img1_path, img2_path, D1, D2, K1, K2, T_1to2,
                covis, scene_id, f1, f2,
            )
            if sample:
                fout.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                samples_written += 1
                logger.info(f"  Pair {f1}<->{f2}: covis={covis:.2f}, {len(sample.qas)} QAs")

    logger.info(f"\nDone: {samples_written} samples -> {output_path}")
