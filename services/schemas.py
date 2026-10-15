from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    image_path: str = Field(..., description="Absolute/relative image path")
    categories: List[str] = Field(..., min_length=1)
    backend: str = Field(default="transformers")
    model_path: str = Field(default="IDEA-Research/Rex-Omni")


class PointingRequest(BaseModel):
    image_path: str
    categories: List[str] = Field(..., min_length=1)
    backend: str = Field(default="transformers")
    model_path: str = Field(default="IDEA-Research/Rex-Omni")


class SegmentRequest(BaseModel):
    image_path: str
    categories: List[str] = Field(..., min_length=1)
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"
    backend: str = Field(default="transformers")
    model_path: str = Field(default="IDEA-Research/Rex-Omni")


class GroundingItem(BaseModel):
    image_path: str
    caption: str


class GroundingRequest(BaseModel):
    items: List[GroundingItem] = Field(..., min_length=1)
    output_jsonl: str = "data/output/grounding_annotations.jsonl"
    output_dir: str = "data/output/grounding_visualizations"
    backend: str = Field(default="transformers")
    model_path: str = Field(default="IDEA-Research/Rex-Omni")
    visualize: bool = True


class GraspDecisionRequest(BaseModel):
    image_path: str
    target_object: str
    candidate_objects: List[str] = Field(default_factory=list)
    backend: str = Field(default="transformers")
    model_path: str = Field(default="IDEA-Research/Rex-Omni")
    robot_exec_mode: str = Field(default="mock")


class FlywheelRunRequest(BaseModel):
    image_dir: Optional[str] = Field(default=None, description="图片目录，为空则用默认")
    rounds: int = Field(default=1, ge=1, le=10)
    max_images: int = Field(default=50, ge=1)
    augment_per_image: int = Field(default=2, ge=0)
    filter_adjectives: bool = True
    backend: str = Field(default="transformers")
    model_path: str = Field(default="IDEA-Research/Rex-Omni")
    sam_checkpoint: Optional[str] = None


class FlywheelEvaluateRequest(BaseModel):
    grounding_jsonl: str = Field(..., description="待评估的 grounding JSONL 路径")
    iou_thresh: float = Field(default=0.5)
    backend: str = Field(default="transformers")
    model_path: str = Field(default="IDEA-Research/Rex-Omni")


class ApiResponse(BaseModel):
    ok: bool = True
    message: str = "success"
    data: Dict[str, Any] = Field(default_factory=dict)

