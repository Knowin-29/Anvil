from __future__ import annotations

from typing import List, Tuple


BINS = 1000


def quantize_coord(x: float, dim_size: int, bins: int = BINS) -> int:
    rel = x / max(dim_size, 1)
    quantized = round(rel * (bins - 1))
    return max(0, min(bins - 1, int(quantized)))


def quantize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[int]:
    width, height = image_size
    return [
        quantize_coord(bbox[0], width),
        quantize_coord(bbox[1], height),
        quantize_coord(bbox[2], width),
        quantize_coord(bbox[3], height),
    ]


def quantize_point(point: List[float], image_size: Tuple[int, int]) -> List[int]:
    width, height = image_size
    return [quantize_coord(point[0], width), quantize_coord(point[1], height)]
