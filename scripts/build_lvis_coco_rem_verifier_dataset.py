"""Build a temporary G_grounding verifier dataset from LVIS v1.0 and COCO-ReM.

The generated JSONL is intentionally simple and matches the direct verifier
schema consumed by scripts/evaluate_grounding_verifier.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from PIL import Image
except Exception:  # pragma: no cover - validated in the runtime environment
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_sample_viewer.adapters.annotation_zip import iter_array_from_zip
from dataset_sample_viewer.adapters.coco_paths import COCOPathMap
from dataset_sample_viewer.adapters.instance_overlay import category_label

REX_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REX_ROOT / "data" / "input" / "grounding_verifier_lvis_coco_rem_1k"
DEFAULT_COCO_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_LVIS_ROOT = Path("/path/to/datasets/026_done_P1_lvis")
DEFAULT_COCO_REM_ROOT = Path("/path/to/datasets/020_done_P0_coco-rem")
DEFAULT_SEED = 20260617


@dataclass(frozen=True)
class SourceSpec:
    dataset_id: str
    display_name: str
    root: Path
    annotations: dict[str, str]


LVIS_ANNOTATIONS = {
    "train": "raw/lvis/lvis_v1_train.json.zip",
    "val": "raw/lvis/lvis_v1_val.json.zip",
}
COCO_REM_ANNOTATIONS = {
    "train": "raw/hf/kdexd__coco-rem/instances_trainrem.json.zip",
    "val": "raw/hf/kdexd__coco-rem/instances_valrem.json.zip",
}


def xywh_to_xyxy(bbox_xywh: list[int | float]) -> list[int | float]:
    if len(bbox_xywh) != 4:
        raise ValueError(f"bbox must have four values: {bbox_xywh}")
    x_min, y_min, width, height = [float(value) for value in bbox_xywh]
    return [_clean_number(x_min), _clean_number(y_min), _clean_number(x_min + width), _clean_number(y_min + height)]


def make_verifier_record(
    *,
    sample_id: str,
    from_dataset: str,
    annotation_id: Any,
    image_path: str | Path,
    object_class: str,
    bbox_xywh: list[int | float],
    image_size: tuple[int, int] | list[int],
    confidence: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "pipeline": "G_grounding",
        "image_path": str(Path(image_path).expanduser().resolve()),
        "object_class": object_class,
        "bbox": xywh_to_xyxy(bbox_xywh),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "confidence": float(confidence),
        "from_dataset": from_dataset,
        "annotation_id": annotation_id,
    }


def source_candidates(source: SourceSpec) -> Iterator[dict[str, Any]]:
    for split, relative_zip in source.annotations.items():
        zip_path = source.root / relative_zip
        if not zip_path.exists():
            print(f"[warn] missing annotation archive: {zip_path}", file=sys.stderr)
            continue
        for annotation_index, ann in iter_array_from_zip(zip_path, "annotations", offset=0, limit=10**12):
            bbox = ann.get("bbox")
            if not _valid_xywh(bbox):
                continue
            image_id = ann.get("image_id")
            if image_id is None:
                continue
            yield {
                "source": source,
                "split": split,
                "annotation_index": annotation_index,
                "annotation_id": ann.get("id"),
                "image_id": image_id,
                "category_id": ann.get("category_id"),
                "object_class": category_label(source.dataset_id, ann.get("category_id")),
                "bbox_xywh": bbox,
            }


def reservoir_sample(candidates: Iterable[dict[str, Any]], count: int, rng: random.Random) -> tuple[list[dict[str, Any]], int]:
    sample: list[dict[str, Any]] = []
    seen = 0
    for candidate in candidates:
        seen += 1
        if len(sample) < count:
            sample.append(candidate)
            continue
        replacement_index = rng.randrange(seen)
        if replacement_index < count:
            sample[replacement_index] = candidate
    if len(sample) < count:
        raise RuntimeError(f"Only found {len(sample)} eligible annotations; requested {count}")
    return sample, seen


def resolve_existing_coco_image(coco_paths: COCOPathMap, image_id: Any, preferred_split: str) -> Path:
    splits = [preferred_split] + [split for split in ("train", "val") if split != preferred_split]
    tried: list[Path] = []
    for split in splits:
        resolved = coco_paths.resolve(image_id=image_id, split=split, year="2017")
        if resolved is None:
            continue
        tried.append(resolved.path)
        if resolved.path.exists():
            return resolved.path
    raise FileNotFoundError(
        "Resolved image path is missing; tried " + ", ".join(str(path) for path in tried)
    )


def read_image_size(image_path: Path) -> tuple[int, int]:
    if Image is None:
        raise RuntimeError("Pillow is required to read image dimensions")
    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for {image_path}: {(width, height)}")
    return int(width), int(height)


def build_records(
    *,
    per_dataset: int,
    seed: int,
    confidence: float,
    coco_root: Path,
    lvis_root: Path,
    coco_rem_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    coco_paths = COCOPathMap(coco_root)
    sources = [
        SourceSpec("lvis_v1", "Lvis v1.0", lvis_root, LVIS_ANNOTATIONS),
        SourceSpec("coco_rem", "COCO-Rem", coco_rem_root, COCO_REM_ANNOTATIONS),
    ]
    records_without_ids: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}

    for source in sources:
        print(f"[sample] {source.display_name}: reservoir sampling {per_dataset} annotations")
        sampled, eligible_seen = reservoir_sample(source_candidates(source), per_dataset, rng)
        valid = []
        for candidate in sampled:
            image_path = resolve_existing_coco_image(coco_paths, candidate["image_id"], candidate["split"])
            image_size = read_image_size(image_path)
            valid.append(
                {
                    "from_dataset": source.display_name,
                    "annotation_id": candidate["annotation_id"],
                    "image_path": image_path,
                    "object_class": candidate["object_class"],
                    "bbox_xywh": candidate["bbox_xywh"],
                    "image_size": image_size,
                    "confidence": confidence,
                }
            )
        records_without_ids.extend(valid)
        source_stats[source.display_name] = {
            "requested": per_dataset,
            "sampled": len(valid),
            "eligible_annotations_seen": eligible_seen,
            "annotation_root": str(source.root),
        }

    rng.shuffle(records_without_ids)
    records = [
        make_verifier_record(sample_id=f"case_{index:04d}", **record)
        for index, record in enumerate(records_without_ids, start=1)
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "confidence": confidence,
        "per_dataset": per_dataset,
        "total_records": len(records),
        "coco_root": str(coco_root),
        "sources": source_stats,
    }
    return records, manifest


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LVIS/COCO-ReM G_grounding verifier JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-dataset", type=int, default=500, help="Samples per source dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--lvis-root", type=Path, default=DEFAULT_LVIS_ROOT)
    parser.add_argument("--coco-rem-root", type=Path, default=DEFAULT_COCO_REM_ROOT)
    parser.add_argument("--filename", default="lvis_coco_rem_grounding_1k.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_dataset <= 0:
        raise ValueError("--per-dataset must be positive")
    records, manifest = build_records(
        per_dataset=args.per_dataset,
        seed=args.seed,
        confidence=args.confidence,
        coco_root=args.coco_root.expanduser().resolve(),
        lvis_root=args.lvis_root.expanduser().resolve(),
        coco_rem_root=args.coco_rem_root.expanduser().resolve(),
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / args.filename
    write_jsonl(records, output_path)
    manifest["jsonl_path"] = str(output_path)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {len(records)} records to {output_path}")
    print(f"[done] wrote manifest to {manifest_path}")
    return 0


def _valid_xywh(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x_min, y_min, width, height = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return width > 0 and height > 0 and x_min >= 0 and y_min >= 0


def _clean_number(value: float) -> int | float:
    rounded = round(float(value), 4)
    if rounded.is_integer():
        return int(rounded)
    return rounded


if __name__ == "__main__":
    raise SystemExit(main())
