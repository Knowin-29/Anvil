import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Final, List, Optional


KNOWN_EXTRA_TAGS: Final[frozenset[str]] = frozenset(
    {
        "planar",
        "push_only",
        "stackable_2d",
        "fragile",
    }
)

PLACEMENT_POLICIES: Final[frozenset[str]] = frozenset(
    {
        "must_place",
        "leave_in_place",
        "push_only",
    }
)


def normalize_name(value: str) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", (value or "").lower())
    text = re.sub(r"[\s_]+", " ", text).strip()
    return text


def slugify_identifier(value: str) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", (value or "").lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


@dataclass
class BaseAffordance:
    operation: Optional[str] = None
    is_container: Optional[bool] = None
    container_placeability: Optional[str] = None
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
        if not data:
            return cls()
        known = {"operation", "is_container", "container_placeability", "extra_tags"}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class IntrinsicActionProfile:
    allowed_operations: List[str] = field(default_factory=list)
    preferred_operation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "IntrinsicActionProfile":
        if not data:
            return cls()
        known = {"allowed_operations", "preferred_operation"}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class DescriptorSchema:
    priority: List[str] = field(default_factory=list)
    alias_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DescriptorSchema":
        if not data:
            return cls()
        known = {"priority", "alias_hints"}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ContainerProfile:
    container_id: int = 0
    container_category: str = ""
    enabled: bool = True
    capacity_hint: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ContainerProfile":
        return cls(**(data or {}))


@dataclass
class PlacementPreference:
    object_class_id: str = ""
    container_id: int = 0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlacementPreference":
        return cls(**data)


@dataclass
class PerceptionProfile:
    needs_size_estimate: bool = False
    needs_pose_estimate: bool = False
    needs_container_state: bool = False
    needs_depth_validation: bool = False
    extra_checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PerceptionProfile":
        return cls(**(data or {}))


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
class AliasEntry:
    alias: str = ""
    class_id: str = ""
    subtype_id: Optional[str] = None
    locale: str = ""
    status: str = "draft"
    source: str = "manual"
    confidence: float = 1.0
    auto_generated: bool = False

    @property
    def normalized_alias(self) -> str:
        return normalize_name(self.alias)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AliasEntry":
        return cls(**(data or {}))


@dataclass
class ConstraintRule:
    rule_id: str = ""
    source_field: str = ""
    operator: str = ""
    value: Any = None
    target_field: str = ""
    effect_value: Any = None
    reason: str = ""
    severity: str = "block"
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConstraintRule":
        return cls(**(data or {}))


@dataclass
class ObjectClass:
    class_id: str = ""
    name: str = ""
    description: str = ""
    affordance: BaseAffordance = field(default_factory=BaseAffordance)
    role_ids: List[str] = field(default_factory=list)
    intrinsic_action_profile: IntrinsicActionProfile = field(
        default_factory=IntrinsicActionProfile
    )
    descriptor_schema: DescriptorSchema = field(default_factory=DescriptorSchema)
    container_profile: Optional[ContainerProfile] = None
    placement_policy: str = "must_place"
    perception: PerceptionProfile = field(default_factory=PerceptionProfile)
    constraints: List[ConstraintRule] = field(default_factory=list)
    metadata: GovernanceMetadata = field(default_factory=GovernanceMetadata)
    alias_auto_approve: bool = False

    def __post_init__(self) -> None:
        self.role_ids = _dedupe_preserve_order(self.role_ids)
        self.intrinsic_action_profile.allowed_operations = _dedupe_preserve_order(
            self.intrinsic_action_profile.allowed_operations
        )

        role_has_container = "container" in self.role_ids
        affordance_is_container = self.affordance.is_container
        if role_has_container:
            if affordance_is_container is False:
                raise ValueError(
                    "Conflicting container state: role_ids includes 'container' "
                    "but affordance.is_container is False"
                )
            if affordance_is_container is None:
                self.affordance.is_container = True
        elif affordance_is_container is True:
            if self.role_ids:
                raise ValueError(
                    "Conflicting container state: role_ids omit 'container' "
                    "but affordance.is_container is True"
                )
            self.role_ids = ["container"]

        preferred_operation = self.intrinsic_action_profile.preferred_operation
        affordance_operation = self.affordance.operation
        if preferred_operation is None and affordance_operation is not None:
            self.intrinsic_action_profile.preferred_operation = affordance_operation
            preferred_operation = affordance_operation
        elif (
            preferred_operation is not None
            and affordance_operation is not None
            and preferred_operation != affordance_operation
        ):
            raise ValueError(
                "Conflicting operation state: "
                "intrinsic_action_profile.preferred_operation does not match "
                "affordance.operation"
            )

        if not self.intrinsic_action_profile.allowed_operations:
            if preferred_operation is not None:
                self.intrinsic_action_profile.allowed_operations = [preferred_operation]
        elif (
            preferred_operation is not None
            and preferred_operation not in self.intrinsic_action_profile.allowed_operations
        ):
            raise ValueError(
                "Conflicting operation state: preferred_operation is not listed in "
                "allowed_operations"
            )

        if (
            self.intrinsic_action_profile.preferred_operation is None
            and self.intrinsic_action_profile.allowed_operations
        ):
            self.intrinsic_action_profile.preferred_operation = (
                self.intrinsic_action_profile.allowed_operations[0]
            )
            preferred_operation = self.intrinsic_action_profile.preferred_operation

        if self.affordance.operation is None and preferred_operation is not None:
            self.affordance.operation = preferred_operation
        if (
            self.placement_policy
            and self.placement_policy not in PLACEMENT_POLICIES
        ):
            raise ValueError(
                f"Invalid placement_policy '{self.placement_policy}'. "
                f"Must be one of {sorted(PLACEMENT_POLICIES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "description": self.description,
            "affordance": self.affordance.to_dict(),
            "role_ids": list(self.role_ids),
            "intrinsic_action_profile": self.intrinsic_action_profile.to_dict(),
            "descriptor_schema": self.descriptor_schema.to_dict(),
            "container_profile": (
                self.container_profile.to_dict() if self.container_profile else None
            ),
            "placement_policy": self.placement_policy,
            "perception": self.perception.to_dict(),
            "constraints": [rule.to_dict() for rule in self.constraints],
            "metadata": self.metadata.to_dict(),
            "alias_auto_approve": self.alias_auto_approve,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectClass":
        return cls(
            class_id=data.get("class_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            affordance=BaseAffordance.from_dict(data.get("affordance")),
            role_ids=list(data.get("role_ids") or []),
            intrinsic_action_profile=IntrinsicActionProfile.from_dict(
                data.get("intrinsic_action_profile")
            ),
            descriptor_schema=DescriptorSchema.from_dict(data.get("descriptor_schema")),
            container_profile=ContainerProfile.from_dict(data.get("container_profile"))
            if data.get("container_profile")
            else None,
            placement_policy=data.get("placement_policy", "must_place"),
            perception=PerceptionProfile.from_dict(data.get("perception")),
            constraints=[
                ConstraintRule.from_dict(item)
                for item in (data.get("constraints") or [])
            ],
            metadata=GovernanceMetadata.from_dict(data.get("metadata")),
            alias_auto_approve=data.get("alias_auto_approve", False),
        )


@dataclass
class ObjectSubtype:
    subtype_id: str = ""
    class_id: str = ""
    name: str = ""
    description: str = ""
    affordance_override: Optional[BaseAffordance] = None
    container_profile_override: Optional[ContainerProfile] = None
    perception_override: Optional[PerceptionProfile] = None
    constraints: List[ConstraintRule] = field(default_factory=list)
    metadata: GovernanceMetadata = field(default_factory=GovernanceMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtype_id": self.subtype_id,
            "class_id": self.class_id,
            "name": self.name,
            "description": self.description,
            "affordance_override": (
                self.affordance_override.to_dict()
                if self.affordance_override
                else None
            ),
            "container_profile_override": (
                self.container_profile_override.to_dict()
                if self.container_profile_override
                else None
            ),
            "perception_override": (
                self.perception_override.to_dict()
                if self.perception_override
                else None
            ),
            "constraints": [rule.to_dict() for rule in self.constraints],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectSubtype":
        return cls(
            subtype_id=data.get("subtype_id", ""),
            class_id=data.get("class_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            affordance_override=BaseAffordance.from_dict(data.get("affordance_override"))
            if data.get("affordance_override")
            else None,
            container_profile_override=ContainerProfile.from_dict(
                data.get("container_profile_override")
            )
            if data.get("container_profile_override")
            else None,
            perception_override=PerceptionProfile.from_dict(
                data.get("perception_override")
            )
            if data.get("perception_override")
            else None,
            constraints=[
                ConstraintRule.from_dict(item)
                for item in (data.get("constraints") or [])
            ],
            metadata=GovernanceMetadata.from_dict(data.get("metadata")),
        )
