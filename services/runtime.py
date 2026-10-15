import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class RexRuntime:
    def __init__(self) -> None:
        self._models: Dict[Tuple[str, str], Any] = {}

    def _load_model(self, model_path: str, backend: str):
        from rex_omni import RexOmniWrapper

        key = (model_path, backend)
        if key not in self._models:
            self._models[key] = RexOmniWrapper(
                model_path=model_path,
                backend=backend,
                max_tokens=2048,
                temperature=0.0,
                top_p=0.05,
                top_k=1,
                repetition_penalty=1.05,
            )
        return self._models[key]

    def infer(
        self,
        image_path: str,
        task: str,
        categories: List[str],
        model_path: str,
        backend: str,
    ) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        model = self._load_model(model_path=model_path, backend=backend)
        return model.inference(images=image, task=task, categories=categories)[0]

    def segment_with_sam(
        self,
        image_path: str,
        predictions: Dict[str, Any],
        sam_checkpoint: Optional[str],
        sam_model_type: str,
    ) -> Dict[str, Any]:
        try:
            from applications._1_rexomni_sam.rexomni_sam_demo import (
                extract_boxes_from_predictions,
                generate_masks_with_sam,
                setup_sam_model,
            )
        except ImportError as e:
            raise RuntimeError(
                "SAM 依赖缺失。请 pip install git+https://github.com/facebookresearch/segment-anything.git "
                "并下载 SAM 权重文件。"
            ) from e

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        boxes = extract_boxes_from_predictions(predictions)
        if not boxes:
            return {"boxes": [], "masks": [], "scores": []}

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = setup_sam_model(
            sam_checkpoint=sam_checkpoint,
            model_type=sam_model_type,
            device=device,
        )
        masks, scores = generate_masks_with_sam(predictor, image_np, boxes)
        return {
            "boxes": [b.tolist() for b in boxes],
            "scores": [float(s) for s in scores],
        }


@lru_cache(maxsize=1)
def get_runtime() -> RexRuntime:
    return RexRuntime()


def resolve_rex_omni_repo() -> str:
    return os.getenv("REX_OMNI_REPO", "/path/to/Rex-Omni")
