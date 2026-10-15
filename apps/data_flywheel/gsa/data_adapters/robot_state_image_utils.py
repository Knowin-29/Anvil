"""Image path resolution helpers for A_robot_state adapters."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("gsa.data_adapters.robot_state_image_utils")

DEFAULT_BRIDGE_IMAGES_ROOT = (
    "/path/to/datasets/downloads/bridge_data_v2/images"
)


def resolve_path_under_root(rel_path: str, dataset_root: Path) -> Path:
    """Resolve a dataset-relative path like ./data/images/foo/bar.png."""
    cleaned = rel_path.strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return (dataset_root / cleaned).resolve()


def resolve_er_image(
    rel_path: str,
    dataset_root: Path,
    cache_dir: Optional[Path] = None,
) -> Tuple[Optional[str], bool]:
    """Resolve Embodied-Reasoner image; extract from zip if needed."""
    if not rel_path:
        return None, False

    candidate = resolve_path_under_root(rel_path, dataset_root)
    if candidate.exists():
        return str(candidate), True

    # rel: data/images/{bundle}/subdir/file.png -> zip data/images/{bundle}.zip
    cleaned = rel_path.strip().lstrip("./")
    parts = Path(cleaned).parts
    if len(parts) < 3 or parts[0] != "data" or parts[1] != "images":
        return str(candidate), False

    bundle = parts[2]
    inner_member = "/".join(parts[3:])
    zip_path = dataset_root / "data" / "images" / f"{bundle}.zip"
    if not zip_path.exists():
        return str(candidate), False

    cache_root = cache_dir or (dataset_root / ".image_cache")
    out_path = cache_root / bundle / inner_member
    if out_path.exists():
        return str(out_path), True

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            member = inner_member
            if member not in names:
                alt = f"{bundle}/{inner_member}"
                if alt in names:
                    member = alt
                else:
                    matches = [n for n in names if n.endswith(Path(inner_member).name)]
                    if not matches:
                        return str(candidate), False
                    member = matches[0]
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        return str(out_path), True
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        logger.debug("Zip extract failed for %s: %s", rel_path, exc)
        return str(candidate), False
