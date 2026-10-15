"""Bridge Adapter — Convert BridgeDataV2 images to A_action_interface / A_robot_state.

BridgeDataV2 contains 25K+ robot tabletop manipulation images from wrist cameras,
ideal for both action interface (grasp points) and robot state recognition.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from .robot_state_image_utils import DEFAULT_BRIDGE_IMAGES_ROOT

logger = logging.getLogger("gsa.data_adapters.bridge")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def resolve_bridge_root(config_root: Optional[str] = None) -> Path:
    if config_root:
        return Path(config_root)
    env_root = os.environ.get("BRIDGE_IMAGES_ROOT")
    if env_root:
        return Path(env_root)
    return Path(DEFAULT_BRIDGE_IMAGES_ROOT)


def discover_bridge_images(max_images: int = 200, root: Optional[str] = None) -> List[str]:
    """Discover images from BridgeDataV2 for A pipeline processing."""
    bridge_root = resolve_bridge_root(root)
    if not bridge_root.exists():
        logger.warning("BridgeDataV2 root not found: %s", bridge_root)
        return []
    images = []
    for f in sorted(bridge_root.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(str(f))
            if len(images) >= max_images:
                break
    logger.info("Discovered %d BridgeDataV2 images from %s", len(images), bridge_root)
    return images


def is_wrist_camera_image(image_path: str) -> bool:
    """Heuristic: BridgeDataV2 images are all from robot wrist cameras."""
    path_lower = image_path.lower()
    return any(k in path_lower for k in ("bridgedatav2", "bridge_data_v2", "episode_"))
