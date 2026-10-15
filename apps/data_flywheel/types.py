from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Tuple


PERCEPTION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "ReferenceResolution",
        "Affordance",
        "Position",
        "GraspPoint",
        "Distance",
        "Orientation",
        "Containment",
        "Stacking",
        "Occlusion",
        "BlockingDependency",
        "ContainerLookup",
        "PlanOrder",
    }
)

REQUIRED_THINKING_TYPES: Final[frozenset[str]] = frozenset(
    {
        "ReferenceResolution",
        "Affordance",
        "PlanOrder",
    }
)

THINKING_LINE_REGEX_PATTERN: Final[str] = r"^\[([A-Z][a-zA-Z]+)\]\s*([\w_]*):\s*(.*)$"
THINKING_LINE_REGEX = re.compile(THINKING_LINE_REGEX_PATTERN)


@dataclass
class Detection:
    class_name: str
    bbox: List[float]
    confidence: float
    source: str


@dataclass
class Segmentation:
    detection: Detection
    mask_polygon: List[List[float]]
    mask_file: str
    grasp_point: List[float]
    mask_source: str = "mask_polygon"


@dataclass
class InstanceAttributes:
    color_primary: Optional[str] = None
    color_primary_hsv: Optional[List[float]] = None
    color_secondary: Optional[str] = None
    size: Optional[str] = None
    shape: Optional[str] = None


@dataclass
class SceneNode:
    id: str
    class_name: str
    bbox: List[float]
    mask_polygon: Optional[List[List[float]]] = None
    mask_file: Optional[str] = None
    mask_source: Optional[str] = None
    has_true_mask: bool = False
    grasp_point: Optional[List[float]] = None
    center_3d: Optional[List[float]] = None
    size_3d: Optional[List[float]] = None
    mean_depth: Optional[float] = None
    valid_3d: bool = False
    operation: Optional[str] = None
    is_container: bool = False
    container_placeability: Optional[str] = None
    placement_policy: Optional[str] = None
    extra_tags: List[str] = field(default_factory=list)
    canonical_name: Optional[str] = None
    object_class_id: Optional[str] = None
    ontology_status: str = "unknown"
    triggered_constraints: List[str] = field(default_factory=list)
    hierarchical_name: Optional[str] = None
    referring_expressions: List[str] = field(default_factory=list)
    attributes: InstanceAttributes = field(default_factory=InstanceAttributes)
    role_ids: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(default_factory=list)
    preferred_operation: Optional[str] = None
    descriptor_axes: List[str] = field(default_factory=list)
    support_parent: Optional[str] = None
    support_type: Optional[str] = None
    support_excluded: bool = False
    directly_reachable: Optional[bool] = None
    blocked_by: List[str] = field(default_factory=list)
    blocking: List[str] = field(default_factory=list)
    excluded_from_action_graph: bool = False
    point_cloud_sample: List[List[float]] = field(default_factory=list)


@dataclass
class SceneEdge:
    subject: str
    object: str
    predicate: str
    source: str
    confidence: float
    frame: str = "untyped"


@dataclass
class TernaryRelation:
    subject: str
    reference_a: str
    reference_b: str
    predicate: str
    confidence: float


@dataclass
class SupportRelation:
    subject: str
    object: str
    relation: str
    confidence: float


@dataclass
class PlacementRegion:
    region_id: str
    region_type: str
    support_id: str
    anchor_ids: List[str] = field(default_factory=list)
    polygon_2d: List[List[float]] = field(default_factory=list)
    point_2d: Optional[List[float]] = None
    reason: str = ""


@dataclass
class SceneGraph:
    image_id: str
    image_path: str
    image_size: Tuple[int, int]
    nodes: List[SceneNode]
    edges: List[SceneEdge]
    ternary_relations: List[TernaryRelation] = field(default_factory=list)
    blocking_dag: List[Tuple[str, str]] = field(default_factory=list)
    support_relations: List[SupportRelation] = field(default_factory=list)
    placement_regions: List[PlacementRegion] = field(default_factory=list)
    orientation_facts: List[SceneEdge] = field(default_factory=list)


@dataclass
class ThinkingStep:
    perception_type: str
    target: str
    content: str
    output: Any = None

    def to_line(self) -> str:
        target_part = self.target if self.target else ""
        return f"[{self.perception_type}] {target_part}: {self.content}"


@dataclass
class ThinkingTrace:
    steps: List[ThinkingStep]

    def to_text(self) -> str:
        return "\n".join(step.to_line() for step in self.steps)


@dataclass
class Action:
    type: str
    target: str
    grasp_point_pixel: Optional[List[float]] = None
    grasp_point_quantized: Optional[List[int]] = None
    destination: Optional[str] = None
    place_point_pixel: Optional[List[float]] = None
    place_point_quantized: Optional[List[int]] = None
    reason: Optional[str] = None
    destination_type: Optional[str] = None
    destination_anchor_ids: List[str] = field(default_factory=list)
    placement_reason: Optional[str] = None


@dataclass
class SceneObjectSample:
    id: str
    class_name: str
    bbox_pixel: List[float]
    bbox_quantized: List[int]
    grasp_point_pixel: Optional[List[float]] = None
    grasp_point_quantized: Optional[List[int]] = None
    mask_file: Optional[str] = None
    operation: Optional[str] = None
    is_container: bool = False
    placement_policy: Optional[str] = None
    extra_tags: List[str] = field(default_factory=list)
    hierarchical_name: Optional[str] = None
    color_primary: Optional[str] = None


@dataclass
class TrainingSample:
    sample_id: str
    image_id: str
    image_path: str
    image_size: Tuple[int, int]
    data_source: str
    instruction: str
    instruction_template_id: str
    instruction_language: str
    scene_objects: List[SceneObjectSample]
    blocking_dag: List[Tuple[str, str]]
    thinking: ThinkingTrace
    actions: List[Action]
    rex_omni_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
