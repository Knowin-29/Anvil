"""离线词表 schema。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional


def normalize_name(value: str) -> str:
    """用于别名匹配的统一规范化。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", (value or "").lower())
    text = re.sub(r"[\s_]+", " ", text).strip()
    return text


def slugify_identifier(value: str) -> str:
    """将名称转换成稳定 ID。保留中文，英文转小写。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", (value or "").lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


@dataclass
class GovernanceMetadata:
    status: str = "draft"
    source: str = "manual"
    confidence: float = 1.0
    version: int = 1
    reviewer: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GovernanceMetadata":
        return cls(**(data or {}))


@dataclass
class BaseAffordance:
    can_pick: Optional[bool] = None
    can_push: Optional[bool] = None
    can_stack: Optional[bool] = None
    is_container: Optional[bool] = None
    is_surface: Optional[bool] = None
    extra_tags: List[str] = field(default_factory=list)

    def merged_with(self, override: Optional["BaseAffordance"]) -> "BaseAffordance":
        if override is None:
            return BaseAffordance(**self.to_dict())
        merged = self.to_dict()
        for key, value in override.to_dict().items():
            if key == "extra_tags":
                merged[key] = list(dict.fromkeys([*merged[key], *value]))
            elif value is not None:
                merged[key] = value
        return BaseAffordance(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BaseAffordance":
        return cls(**(data or {}))


@dataclass
class PlacementProfile:
    default_container_types: List[str] = field(default_factory=list)
    placement_modes: List[str] = field(default_factory=list)
    preferred_supports: List[str] = field(default_factory=list)
    notes: str = ""

    def merged_with(self, override: Optional["PlacementProfile"]) -> "PlacementProfile":
        if override is None:
            return PlacementProfile(**self.to_dict())
        merged = self.to_dict()
        for key, value in override.to_dict().items():
            if isinstance(value, list):
                if value:
                    merged[key] = list(value)
            elif value:
                merged[key] = value
        return PlacementProfile(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PlacementProfile":
        return cls(**(data or {}))


@dataclass
class PerceptionRequired:
    needs_size_estimate: bool = False
    needs_pose_estimate: bool = False
    needs_container_state: bool = False
    needs_depth_validation: bool = False
    extra_checks: List[str] = field(default_factory=list)

    def merged_with(self, override: Optional["PerceptionRequired"]) -> "PerceptionRequired":
        if override is None:
            return PerceptionRequired(**self.to_dict())
        merged = self.to_dict()
        merged["needs_size_estimate"] = merged["needs_size_estimate"] or override.needs_size_estimate
        merged["needs_pose_estimate"] = merged["needs_pose_estimate"] or override.needs_pose_estimate
        merged["needs_container_state"] = merged["needs_container_state"] or override.needs_container_state
        merged["needs_depth_validation"] = merged["needs_depth_validation"] or override.needs_depth_validation
        merged["extra_checks"] = list(dict.fromkeys([*merged["extra_checks"], *override.extra_checks]))
        return PerceptionRequired(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PerceptionRequired":
        return cls(**(data or {}))


@dataclass
class ConstraintRule:
    rule_id: str
    source_field: str
    operator: str
    value: Any
    target_field: str
    effect_value: Any
    reason: str = ""
    severity: str = "block"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintRule":
        return cls(**data)


@dataclass
class ObjectClass:
    class_id: str
    name: str
    description: str = ""
    affordance: BaseAffordance = field(default_factory=BaseAffordance)
    placement: PlacementProfile = field(default_factory=PlacementProfile)
    perception: PerceptionRequired = field(default_factory=PerceptionRequired)
    constraints: List[ConstraintRule] = field(default_factory=list)
    metadata: GovernanceMetadata = field(default_factory=GovernanceMetadata)
    alias_auto_approve: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["affordance"] = self.affordance.to_dict()
        data["placement"] = self.placement.to_dict()
        data["perception"] = self.perception.to_dict()
        data["constraints"] = [item.to_dict() for item in self.constraints]
        data["metadata"] = self.metadata.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectClass":
        payload = dict(data)
        payload["affordance"] = BaseAffordance.from_dict(payload.get("affordance"))
        payload["placement"] = PlacementProfile.from_dict(payload.get("placement"))
        payload["perception"] = PerceptionRequired.from_dict(payload.get("perception"))
        payload["constraints"] = [ConstraintRule.from_dict(item) for item in payload.get("constraints", [])]
        payload["metadata"] = GovernanceMetadata.from_dict(payload.get("metadata"))
        return cls(**payload)


@dataclass
class ObjectSubtype:
    subtype_id: str
    class_id: str
    name: str
    description: str = ""
    affordance_override: BaseAffordance = field(default_factory=BaseAffordance)
    placement_override: PlacementProfile = field(default_factory=PlacementProfile)
    perception_override: PerceptionRequired = field(default_factory=PerceptionRequired)
    constraints: List[ConstraintRule] = field(default_factory=list)
    metadata: GovernanceMetadata = field(default_factory=GovernanceMetadata)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["affordance_override"] = self.affordance_override.to_dict()
        data["placement_override"] = self.placement_override.to_dict()
        data["perception_override"] = self.perception_override.to_dict()
        data["constraints"] = [item.to_dict() for item in self.constraints]
        data["metadata"] = self.metadata.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectSubtype":
        payload = dict(data)
        payload["affordance_override"] = BaseAffordance.from_dict(payload.get("affordance_override"))
        payload["placement_override"] = PlacementProfile.from_dict(payload.get("placement_override"))
        payload["perception_override"] = PerceptionRequired.from_dict(payload.get("perception_override"))
        payload["constraints"] = [ConstraintRule.from_dict(item) for item in payload.get("constraints", [])]
        payload["metadata"] = GovernanceMetadata.from_dict(payload.get("metadata"))
        return cls(**payload)


@dataclass
class AliasEntry:
    alias: str
    class_id: str
    subtype_id: Optional[str] = None
    locale: str = "und"
    status: str = "approved"
    source: str = "manual"
    confidence: float = 1.0
    auto_generated: bool = False

    @property
    def normalized_alias(self) -> str:
        return normalize_name(self.alias)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AliasEntry":
        return cls(**data)
