"""Reproducible inference primitives for Anvil's crossed seed-by-scene design.

This module creates shared resampling indices. Metric values are intentionally not
computed here: the frozen benchmark scorer must recompute non-additive metrics such as
detection mAP inside every replicate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class InferencePlanError(ValueError):
    """Raised when an inference plan is incomplete or inconsistent."""


T_CRITICAL_975 = {
    1: 12.706204736,
    2: 4.30265273,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    11: 2.20098516,
    12: 2.17881283,
    13: 2.160368656,
    14: 2.144786688,
    15: 2.131449546,
    16: 2.119905299,
    17: 2.109815578,
    18: 2.10092204,
    19: 2.093024054,
    20: 2.085963447,
    21: 2.079613845,
    22: 2.073873068,
    23: 2.06865761,
    24: 2.063898562,
    25: 2.059538553,
    26: 2.055529439,
    27: 2.051830516,
    28: 2.048407142,
    29: 2.045229642,
    30: 2.042272456,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_crossed_plan(
    seed_ids: Sequence[int],
    scene_count: int,
    replicates: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(seed_ids) < 2:
        raise InferencePlanError("at least two training seeds are required")
    if len(set(seed_ids)) != len(seed_ids):
        raise InferencePlanError("training seed IDs must be unique")
    if scene_count < 2:
        raise InferencePlanError("at least two scene clusters are required")
    if replicates < 1:
        raise InferencePlanError("replicates must be positive")
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
        raise InferencePlanError("rng_seed must be an integer")

    rng = np.random.default_rng(rng_seed)
    seed_draws = rng.integers(
        0, len(seed_ids), size=(replicates, len(seed_ids)), dtype=np.int32
    )
    scene_draws = rng.integers(
        0, scene_count, size=(replicates, scene_count), dtype=np.int32
    )
    return seed_draws, scene_draws


def exact_one_sided_sign_flip_pvalue(seed_differences: Sequence[float]) -> float:
    values = np.asarray(seed_differences, dtype=float)
    if values.ndim != 1 or not 2 <= values.size <= 20:
        raise InferencePlanError("sign-flip inference requires 2 to 20 seed differences")
    if not np.isfinite(values).all():
        raise InferencePlanError("seed differences must be finite")
    observed = float(values.mean())
    exceedances = 0
    total = 2 ** values.size
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        randomized = float(np.mean(values * np.asarray(signs)))
        if randomized >= observed - 1e-15:
            exceedances += 1
    return exceedances / total


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    values = [float(value) for value in pvalues]
    if not values:
        raise InferencePlanError("Holm adjustment requires at least one p-value")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise InferencePlanError("p-values must be finite and within [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    family_size = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def seed_t_interval_95(seed_differences: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(seed_differences, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise InferencePlanError("seed interval requires at least two differences")
    if not np.isfinite(values).all():
        raise InferencePlanError("seed differences must be finite")
    degrees = values.size - 1
    if degrees not in T_CRITICAL_975:
        raise InferencePlanError("seed interval supports 2 to 31 seeds")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    half_width = T_CRITICAL_975[degrees] * standard_error
    return mean - half_width, mean + half_width


def read_scene_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise InferencePlanError(f"missing scene manifest: {path}")
    scene_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise InferencePlanError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            scene_id = row.get("scene_id") if isinstance(row, dict) else None
            if not isinstance(scene_id, str) or not scene_id:
                raise InferencePlanError(f"{path}:{line_number}: missing scene_id")
            scene_ids.append(scene_id)
    if len(scene_ids) < 2:
        raise InferencePlanError("scene manifest must contain at least two rows")
    if len(set(scene_ids)) != len(scene_ids):
        raise InferencePlanError("scene manifest contains duplicate scene_id values")
    return scene_ids


def write_plan(
    output_path: Path,
    metadata_path: Path,
    scene_manifest_path: Path,
    seed_ids: Sequence[int],
    replicates: int,
    rng_seed: int,
) -> dict:
    if output_path.suffix != ".npz":
        raise InferencePlanError("resampling plan output must use .npz")
    if metadata_path.suffix != ".json":
        raise InferencePlanError("resampling metadata must use .json")
    scene_ids = read_scene_ids(scene_manifest_path)
    seed_draws, scene_draws = create_crossed_plan(
        seed_ids, len(scene_ids), replicates, rng_seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, seed_draws=seed_draws, scene_draws=scene_draws)
    metadata = {
        "schema_version": "anvil-crossed-resampling-plan-v1",
        "seed_ids": list(seed_ids),
        "scene_count": len(scene_ids),
        "replicates": replicates,
        "rng_seed": rng_seed,
        "scene_manifest_sha256": sha256_file(scene_manifest_path),
        "plan_sha256": sha256_file(output_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "numpy_version": np.__version__,
        "metric_policy": "frozen scorer recomputes every endpoint inside each replicate",
        "detection_policy": "COCO mAP must be recomputed; never average item-level proxy scores",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def validate_saved_plan(
    output_path: Path,
    metadata_path: Path,
    scene_manifest_path: Path | None = None,
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "anvil-crossed-resampling-plan-v1":
        raise InferencePlanError("wrong saved-plan schema")
    if sha256_file(output_path) != metadata.get("plan_sha256"):
        raise InferencePlanError("saved plan hash mismatch")
    if scene_manifest_path is not None:
        if not scene_manifest_path.is_file():
            raise InferencePlanError("saved plan scene manifest is missing")
        if sha256_file(scene_manifest_path) != metadata.get("scene_manifest_sha256"):
            raise InferencePlanError("saved plan scene-manifest hash mismatch")
    with np.load(output_path) as plan:
        seed_draws = plan["seed_draws"]
        scene_draws = plan["scene_draws"]
    expected_seed_shape = (metadata["replicates"], len(metadata["seed_ids"]))
    expected_scene_shape = (metadata["replicates"], metadata["scene_count"])
    if seed_draws.shape != expected_seed_shape:
        raise InferencePlanError("saved seed-draw shape mismatch")
    if scene_draws.shape != expected_scene_shape:
        raise InferencePlanError("saved scene-draw shape mismatch")
    if seed_draws.min() < 0 or seed_draws.max() >= len(metadata["seed_ids"]):
        raise InferencePlanError("saved seed draw is out of range")
    if scene_draws.min() < 0 or scene_draws.max() >= metadata["scene_count"]:
        raise InferencePlanError("saved scene draw is out of range")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=2_026_090_6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = write_plan(
            args.output,
            args.metadata,
            args.scene_manifest,
            args.seeds,
            args.replicates,
            args.rng_seed,
        )
        validate_saved_plan(args.output, args.metadata, args.scene_manifest)
    except (InferencePlanError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, indent=2))
        return 1
    print(json.dumps({"valid": True, **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
