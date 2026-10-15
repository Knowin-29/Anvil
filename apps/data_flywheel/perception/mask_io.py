"""Mask persistence helpers for the perception layer."""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

from apps.data_flywheel.perception.protocol import MaskPersistenceError


def save_mask_png(mask_b64: str, rel_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(rel_path) or ".", exist_ok=True)
        img_bytes = base64.b64decode(mask_b64)
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != "L":
            image = image.convert("L")
        image.save(rel_path, format="PNG")
    except Exception as exc:
        raise MaskPersistenceError(rel_path, exc) from exc


def load_mask_png(rel_path: str) -> np.ndarray:
    image = Image.open(rel_path).convert("L")
    return np.array(image, dtype=np.uint8)
