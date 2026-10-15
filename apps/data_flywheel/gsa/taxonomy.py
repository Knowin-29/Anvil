"""GSA Taxonomy — Bucket, subtype, scene family, and scenario definitions.

Encodes the GSA buckets (G, S, and 5 A pipelines), their subtypes,
scene families, and scenario crosswalk. Provides validation helpers
and pipeline routing logic aligned with the fusion classification doc.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set


class Bucket(str, Enum):
    G = "G"
    S = "S"
    # Legacy (kept for backward compat)
    A_STATE = "A_state"
    A_ACTION = "A_action"
    # 6 fine-grained A pipelines
    A_IDENTITY_ATTR = "A_identity_attr"
    A_OBJ_STATE = "A_obj_state"
    A_ROBOT_STATE = "A_robot_state"
    A_ACTION_PRECONDITION = "A_action_precondition"
    A_AFFORDANCE = "A_affordance"
    A_ACTION_INTERFACE = "A_action_interface"
    # Correspondence (geometric label pipeline, volc_jun12)
    CORRESPONDENCE = "correspondence"


class GSubtype(str, Enum):
    SIMPLE_GROUNDING = "simple_grounding"
    DESCRIPTIVE_GROUNDING = "descriptive_grounding"
    RELATIONAL_GROUNDING = "relational_grounding"
    COUNTING = "counting"


class SSubtype(str, Enum):
    RELATION_2D = "2d_relation"
    DEPTH_RELATION = "depth_relation"
    TOPOLOGY_RELATION = "topology_relation"
    OCCLUSION_RELATION = "occlusion_relation"
    SUPPORT_RELATION = "support_relation"
    CONTAINMENT_RELATION = "containment_relation"
    STACKING_RELATION = "stacking_relation"


class AStateSubtype(str, Enum):
    """Legacy A_state subtypes (kept for backward compat)."""
    OBJECT_RECOGNITION = "object_recognition"
    STATE_RECOGNITION = "state_recognition"


class AActionSubtype(str, Enum):
    """Legacy A_action subtypes (kept for backward compat)."""
    AFFORDANCE_REGION = "affordance_region"
    GRASP_POINT_2D = "grasp_point_2d"
    JAW_AXIS_2D = "jaw_axis_2d"
    PUSH_CONTACT_POINT_2D = "push_contact_point_2d"
    PUSH_DIRECTION_2D = "push_direction_2d"
    PLACE_REGION_2D = "place_region_2d"
    CAN_PICK_NOW = "can_pick_now"
    CAN_PUSH_NOW = "can_push_now"
    CAN_PLACE_NOW = "can_place_now"
    BLOCKED_BY = "blocked_by"
    NEED_PRE_PUSH = "need_pre_push"
    FALLBACK_ACTION = "fallback_action"


# ============================================================
# Fine-grained A pipeline subtypes (fusion classification)
# ============================================================

class AIdentityAttrSubtype(str, Enum):
    OBJECT_CLASS = "object_class"
    COLOR = "color"
    MATERIAL = "material"
    SHAPE = "shape"
    SIZE = "size"
    MULTI_ATTR = "multi_attr"


class AObjStateSubtype(str, Enum):
    POSE = "pose"
    OPEN_CLOSE = "open_close"
    OCCUPANCY = "occupancy"
    CLEANLINESS = "cleanliness"
    DAMAGE = "damage"
    LIQUID_LEVEL = "liquid_level"
    ON_OFF = "on_off"
    OTHER = "other"


class ARobotStateSubtype(str, Enum):
    GRIPPER_EMPTY = "gripper_empty"
    HOLDING_OBJECT = "holding_object"
    OBJECT_BETWEEN_FINGERS = "object_between_fingers"
    CONTACT_STATE = "contact_state"
    TOOL_READY = "tool_ready"


class AActionPreconditionSubtype(str, Enum):
    CAN_PICK = "can_pick"
    CAN_PUSH = "can_push"
    CAN_PLACE = "can_place"
    BLOCKED = "blocked"
    NEED_PRE_ACTION = "need_pre_action"
    UNREACHABLE = "unreachable"


class AActionInterfaceSubtype(str, Enum):
    GRASP_POINT = "grasp_point"
    JAW_AXIS = "jaw_axis"
    PUSH_CONTACT = "push_contact"
    PUSH_DIRECTION = "push_direction"
    PLACE_REGION = "place_region"
    REORIENT = "reorient"


class AAffordanceSubtype(str, Enum):
    HANDLE_POINT = "handle_point"
    BUTTON_POINT = "button_point"
    KNOB_POINT = "knob_point"
    EDGE_POINT = "edge_point"
    RIM_POINT = "rim_point"
    BODY_POINT = "body_point"
    SIDE_POINT = "side_point"
    INTERACTION_HOTSPOT = "interaction_hotspot"


class CorrespondenceSubtype(str, Enum):
    """Pixel/point correspondence between two views (geometric labels)."""
    PIXEL_CORRESPONDENCE = "pixel_correspondence"
    OCCLUDED_POINT = "occluded_point"


class SceneFamily(str, Enum):
    DELIVERY = "delivery"
    TABLETOP_TIDY = "tabletop_tidy"
    STORAGE = "storage"


class Scenario(str, Enum):
    SC1 = "SC.1"
    SC2 = "SC.2"
    SC3 = "SC.3"


SCENARIO_TO_SCENE_FAMILY: Dict[Scenario, SceneFamily] = {
    Scenario.SC1: SceneFamily.DELIVERY,
    Scenario.SC2: SceneFamily.TABLETOP_TIDY,
    Scenario.SC3: SceneFamily.STORAGE,
}

SCENE_FAMILY_TO_SCENARIO: Dict[SceneFamily, Scenario] = {
    v: k for k, v in SCENARIO_TO_SCENE_FAMILY.items()
}

BUCKET_SUBTYPES: Dict[Bucket, List[str]] = {
    Bucket.G: [s.value for s in GSubtype],
    Bucket.S: [s.value for s in SSubtype],
    # Legacy
    Bucket.A_STATE: [s.value for s in AStateSubtype],
    Bucket.A_ACTION: [s.value for s in AActionSubtype],
    # Fine-grained A pipelines
    Bucket.A_IDENTITY_ATTR: [s.value for s in AIdentityAttrSubtype],
    Bucket.A_OBJ_STATE: [s.value for s in AObjStateSubtype],
    Bucket.A_ROBOT_STATE: [s.value for s in ARobotStateSubtype],
    Bucket.A_ACTION_PRECONDITION: [s.value for s in AActionPreconditionSubtype],
    Bucket.A_AFFORDANCE: [s.value for s in AAffordanceSubtype],
    Bucket.A_ACTION_INTERFACE: [s.value for s in AActionInterfaceSubtype],
    Bucket.CORRESPONDENCE: [s.value for s in CorrespondenceSubtype],
}


# ============================================================
# Dynamic Scene Tag Registry (volc_jun12)
# ============================================================

# Core scene tags — always available as seed. Additional tags are auto-registered
# at runtime when new scene_type values appear in dimension_sources or VLM output.
_SCENE_TAG_REGISTRY: Dict[str, dict] = {
    "tabletop_manipulation": {"priority": "P0", "task": "G+S+A"},
    "kitchen": {"priority": "P0", "task": "G+S+A"},
    "object_state_change": {"priority": "P0", "task": "A_obj_state"},
    "household": {"priority": "P1", "task": "G+S+A"},
    "navigation": {"priority": "P1", "task": "S"},
    "living_room": {"priority": "P1", "task": "G+S"},
    "indoor_general": {"priority": "P1", "task": "G+S"},
    "object_closeup": {"priority": "P2", "task": "A_identity_attr"},
    "bathroom": {"priority": "P2", "task": "G+S+A"},
    "bedroom": {"priority": "P2", "task": "G+S"},
    "office": {"priority": "P2", "task": "G+S"},
    "warehouse": {"priority": "P2", "task": "G+S+A"},
    "outdoor": {"priority": "P2", "task": "G+S"},
    "hallway": {"priority": "P2", "task": "S"},
    "staircase": {"priority": "P2", "task": "S"},
    "toy_closeup": {"priority": "P2", "task": "A_identity_attr"},
    "mixed": {"priority": "P2", "task": "G+S+A"},
}


def register_scene_tag(tag: str, priority: str = "P2", task: str = "G+S+A") -> None:
    """Dynamically register a new scene tag at runtime."""
    if tag not in _SCENE_TAG_REGISTRY:
        _SCENE_TAG_REGISTRY[tag] = {"priority": priority, "task": task}


def get_all_scene_tags() -> Dict[str, dict]:
    """Return current scene tag registry (read-only copy)."""
    return dict(_SCENE_TAG_REGISTRY)


def is_valid_scene_tag(tag: str) -> bool:
    """Check if tag is registered; auto-register unknown tags as P2."""
    if tag in _SCENE_TAG_REGISTRY:
        return True
    if tag and tag != "unknown":
        register_scene_tag(tag)
        return True
    return False


class ActionType(str, Enum):
    PICK = "pick"
    PUSH = "push"
    PLACE = "place"
    REORIENT = "reorient"


class LabelType(str, Enum):
    POINT = "point"
    VECTOR = "vector"
    REGION = "region"
    YES_NO = "yes_no"
    TEXT = "text"


class VerifierStatus(str, Enum):
    AUTO_VERIFIABLE = "auto_verifiable"
    REVIEW_REQUIRED = "review_required"
    HEURISTIC_ONLY = "heuristic_only"


class SourceProvenance(str, Enum):
    REAL_LOG = "real_log"
    PUBLIC_DATASET = "public_dataset"
    SIM = "sim"
    ADAPTER = "adapter"
    HUMAN_ANCHOR = "human_anchor"
    VLM = "vlm"


class FallbackAction(str, Enum):
    PUSH_FIRST = "push_first"
    CHOOSE_OTHER_REGION = "choose_other_region"
    SKIP = "skip"


STATE_KEYS: Set[str] = {
    "open_closed",
    "empty_filled",
    "upright_fallen",
    "clean_dirty",
    "whole_broken",
    "visible_occluded",
    "inside_outside",
    "on_off",
}

# Extended state type mapping for A_obj_state pipeline
STATE_TYPE_VALUES: Dict[str, List[str]] = {
    "pose": ["upright", "lying", "tilted", "inverted", "unknown"],
    "open_close": ["open", "closed", "half_open", "unknown"],
    "occupancy": ["empty", "full", "partial", "unknown"],
    "cleanliness": ["clean", "dirty", "stained", "unknown"],
    "damage": ["intact", "broken", "cracked", "dented", "unknown"],
    "liquid_level": ["empty", "low", "half", "full", "overflowing", "unknown"],
    "on_off": ["on", "off", "standby", "unknown"],
    "other": [],
}

# Camera views for A_robot_state
class CameraView(str, Enum):
    WRIST = "wrist"
    HEAD = "head"
    EXTERNAL = "external"

# End effector types
class EndEffector(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    SUCTION = "suction"
    DEXTEROUS = "dexterous"
    UNKNOWN = "unknown"

# Failure reasons for A_action_precondition
class FailureReason(str, Enum):
    OCCLUDED = "occluded"
    UNSTABLE = "unstable"
    UNREACHABLE = "unreachable"
    NO_CLEARANCE = "no_clearance"
    OCCUPIED = "occupied"
    TOO_HEAVY = "too_heavy"
    TOO_LARGE = "too_large"
    FIXED = "fixed"
    UNKNOWN = "unknown"

# Pre-action types
class PreAction(str, Enum):
    NONE = "none"
    PUSH_FIRST = "push_first"
    REORIENT_FIRST = "reorient_first"
    CHOOSE_OTHER_REGION = "choose_other_region"
    SKIP = "skip"

# Known attribute values for A_identity_attr validation
KNOWN_COLORS: Set[str] = {
    "red", "blue", "green", "yellow", "white", "black", "brown",
    "orange", "pink", "purple", "gray", "silver", "gold", "beige",
    "transparent", "multicolor", "unknown",
}

KNOWN_MATERIALS: Set[str] = {
    "plastic", "metal", "wood", "glass", "ceramic", "fabric",
    "paper", "cardboard", "rubber", "foam", "leather", "stone",
    "concrete", "composite", "unknown",
}

KNOWN_SHAPES: Set[str] = {
    "rectangular", "cylindrical", "spherical", "conical", "flat",
    "irregular", "L-shaped", "ring", "disc", "unknown",
}


def validate_bucket_subtype(bucket: str, subtype: str) -> bool:
    """Check that subtype belongs to the given bucket."""
    try:
        b = Bucket(bucket)
    except ValueError:
        return False
    return subtype in BUCKET_SUBTYPES.get(b, [])


def validate_scene_family(scene_family: str) -> bool:
    """Check that scene_family is a valid enum value."""
    return scene_family in {sf.value for sf in SceneFamily}


def validate_scenario(scenario: str) -> bool:
    """Check that scenario is a valid enum value."""
    return scenario in {sc.value for sc in Scenario}


def get_bucket_for_subtype(subtype: str) -> Optional[Bucket]:
    """Return the bucket that owns a given subtype, or None."""
    for bucket, subtypes in BUCKET_SUBTYPES.items():
        if subtype in subtypes:
            return bucket
    return None


def route_to_pipeline(
    html_tags: Optional[List[str]] = None,
    gsa_bucket: Optional[str] = None,
    has_gripper_view: bool = False,
    has_action_output: bool = False,
    has_state_output: bool = False,
    has_attribute_output: bool = False,
) -> Bucket:
    """Route a sample to the correct A pipeline based on available signals.

    Priority order (per fusion classification doc Section 8):
    1. Robot state (gripper/wrist camera) -> A_robot_state
    2. Action feasibility question -> A_action_precondition
    3. Action geometric interface -> A_action_interface
    4. Object state -> A_obj_state
    5. Object identity/attribute -> A_identity_attr
    """
    if has_gripper_view:
        return Bucket.A_ROBOT_STATE

    if gsa_bucket == "A_state;A_action":
        if has_gripper_view:
            return Bucket.A_ROBOT_STATE
        return Bucket.A_ACTION_PRECONDITION

    if gsa_bucket in ("A_action", "G;A_action", "S;A_action",
                      "G;S;A_state;A_action"):
        if has_action_output:
            return Bucket.A_ACTION_INTERFACE
        return Bucket.A_ACTION_PRECONDITION

    if gsa_bucket in ("A_state", "G;A_state", "S;A_state",
                      "G;S;A_state"):
        if has_state_output:
            return Bucket.A_OBJ_STATE
        return Bucket.A_IDENTITY_ATTR

    if html_tags:
        tag_set = set(html_tags)
        if tag_set & {"A.1", "A.2"} and has_gripper_view:
            return Bucket.A_ROBOT_STATE
        if "A.3" in tag_set:
            return Bucket.A_OBJ_STATE
        if "A.2" in tag_set and has_attribute_output:
            return Bucket.A_IDENTITY_ATTR
        if "A.1" in tag_set:
            return Bucket.A_IDENTITY_ATTR

    if has_action_output:
        return Bucket.A_ACTION_INTERFACE
    if has_state_output:
        return Bucket.A_OBJ_STATE
    return Bucket.A_IDENTITY_ATTR


def needs_g_context(label: Dict) -> bool:
    """Check if a label needs G_context (grounding info)."""
    if label.get("bbox") or label.get("mask") or label.get("point"):
        return True
    prompt = label.get("prompt", "")
    constraint_keywords = ["黑色", "白色", "桌面", "中央", "左边", "右边"]
    return any(kw in prompt for kw in constraint_keywords)


def needs_s_context(label: Dict) -> bool:
    """Check if a label needs S_context (spatial info)."""
    indicators = [
        label.get("blocked_by"),
        label.get("s_evidence_ids"),
        label.get("s_context"),
    ]
    if any(indicators):
        return True
    prompt = label.get("prompt", "")
    spatial_keywords = ["遮挡", "支撑", "堆叠", "inside", "stacked",
                        "occluded", "blocked", "reachable"]
    return any(kw in prompt for kw in spatial_keywords)


# ============================================================
# Viewer System Alignment (YOUR_VIEWER_URL/viewer/)
# 能力轴定义 — 对齐 Qwen Proxy Viewer 的 8 维能力体系
# ============================================================

VIEWER_CAPABILITY_AXES = {
    "G": "视觉定位 / grounding — 点定位、bbox 定位、关系定位和计数",
    "S": "空间理解 — 平面关系、深度/3D 关系、接触/堆叠/拓扑关系",
    "A": "属性与状态 — 物体识别、属性识别、状态或 affordance 判断",
    "R": "场景与常识推理 — 场景理解、物理可行性和安全风险判断",
    "D": "变化与执行验证 — 双图变化描述和 before/after 任务是否完成",
    "IF": "输出格式遵循 — 点坐标、选项、yes/no 等格式是否严格合规",
    "H": "幻觉控制 — 是否编造不存在物体、属性或空间关系",
    "Refuse": "目标不存在时拒绝定位 — absent grounding",
}

# Subtype → Viewer 能力标签映射 (格式: G.1, S.2, A.3 等)
SUBTYPE_TO_VIEWER_TAG: Dict[str, str] = {
    # G: 视觉定位
    "simple_grounding": "G.1",       # bbox 定位
    "descriptive_grounding": "G.2",  # 属性描述定位
    "relational_grounding": "G.3",   # 关系定位
    "counting": "G.4",              # 计数
    # S: 空间理解
    "2d_relation": "S.1",           # 平面关系
    "depth_relation": "S.2",        # 深度/3D 关系
    "topology_relation": "S.3",     # 接触/拓扑
    "occlusion_relation": "S.4",    # 遮挡
    "support_relation": "S.5",      # 支撑
    "containment_relation": "S.6",  # 包含
    "stacking_relation": "S.7",     # 堆叠
    # A_identity_attr
    "object_class": "A.1",
    "color": "A.2",
    "material": "A.2",
    "shape": "A.2",
    "size": "A.2",
    "multi_attr": "A.2",
    # A_obj_state
    "pose": "A.3",
    "open_close": "A.3",
    "occupancy": "A.3",
    "cleanliness": "A.3",
    "damage": "A.3",
    "liquid_level": "A.3",
    "on_off": "A.3",
    # A_robot_state
    "gripper_empty": "A.7",
    "holding_object": "A.7",
    "object_between_fingers": "A.7",
    "contact_state": "A.7",
    "tool_ready": "A.7",
    # A_action_precondition
    "can_pick": "A.8",
    "can_push": "A.8",
    "can_place": "A.8",
    "blocked": "A.8",
    "need_pre_action": "A.8",
    "unreachable": "A.8",
    # A_action_interface
    "grasp_point": "A.4",
    "jaw_axis": "A.4",
    "push_contact": "A.5",
    "push_direction": "A.5",
    "place_region": "A.6",
    "reorient": "A.6",
    # A_affordance
    "handle_point": "A.9",
    "button_point": "A.9",
    "knob_point": "A.9",
    "edge_point": "A.9",
    "rim_point": "A.9",
    "body_point": "A.9",
    "side_point": "A.9",
    "interaction_hotspot": "A.9",
    # Legacy subtypes
    "object_recognition": "A.1",
    "state_recognition": "A.3",
    "affordance_region": "A.3",
    "grasp_point_2d": "A.4",
    "push_contact_point_2d": "A.5",
    "place_region_2d": "A.6",
}

# Bucket → Viewer 主标签
BUCKET_TO_VIEWER_TAG: Dict[str, str] = {
    "G": "G",
    "S": "S",
    "A_state": "A",
    "A_action": "A",
    "A_affordance": "A",
    "pick_place": "A",
    "A_identity_attr": "A",
    "A_obj_state": "A",
    "A_robot_state": "A",
    "A_action_precondition": "A",
    "A_action_interface": "A",
}


def get_viewer_tag(bucket: str, subtype: str) -> str:
    """Get the Viewer system capability tag for a label."""
    tag = SUBTYPE_TO_VIEWER_TAG.get(subtype)
    if tag:
        return tag
    return BUCKET_TO_VIEWER_TAG.get(bucket, bucket)


# ============================================================
# Application-aligned enums (Pick-and-Place system)
# ============================================================

class ObjectType(str, Enum):
    BOX = "box"
    CYLINDER = "cylinder"
    FRUSTUM = "frustum"
    BOWL = "bowl"
    PLATE = "plate"
    SLENDER = "slender"
    THIN = "thin"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


class PoseHint(str, Enum):
    UPRIGHT = "upright"
    HORIZONTAL = "horizontal"
    UNKNOWN = "unknown"


class GraspHint(str, Enum):
    RIM = "rim"
    SIDE = "side"
    MIDDLE = "middle"
    EDGE = "edge"
    HANDLE = "handle"
    BODY = "body"
    UNKNOWN = "unknown"


class ContainerOpening(str, Enum):
    OPEN_VISIBLE = "open_visible"
    CLOSED_OR_LIDDED = "closed_or_lidded"
    UNKNOWN = "unknown"


class ApproachDirection(str, Enum):
    HORIZONTAL = "horizontal"
    TOP_DOWN = "top_down"
    DIAGONAL_DOWN = "diagonal_down"
    UNKNOWN = "unknown"
