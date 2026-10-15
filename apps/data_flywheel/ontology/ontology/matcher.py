"""离线词表运行时匹配。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from apps.data_flywheel.ontology.schema import normalize_name
from apps.data_flywheel.ontology.store import OntologyStore


@dataclass
class RuntimeInstanceContext:
    bbox: List[float]
    image_width: int
    image_height: int
    estimated_max_dimension_cm: Optional[float] = None
    estimated_thickness_mm: Optional[float] = None
    in_container: Optional[bool] = None
    mean_depth: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extras.get(key)


@dataclass
class ResolvedRuntimeLabel:
    observation_name: str
    normalized_name: str
    ontology_status: str
    canonical_name: str
    object_class_id: Optional[str]
    object_subtype_id: Optional[str]
    matched_alias: Optional[str]
    affordance: Dict[str, Any]
    placement: Dict[str, Any]
    perception: Dict[str, Any]
    triggered_constraints: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_name": self.observation_name,
            "normalized_name": self.normalized_name,
            "ontology_status": self.ontology_status,
            "canonical_name": self.canonical_name,
            "object_class_id": self.object_class_id,
            "object_subtype_id": self.object_subtype_id,
            "matched_alias": self.matched_alias,
            "affordance": self.affordance,
            "placement": self.placement,
            "perception": self.perception,
            "triggered_constraints": self.triggered_constraints,
        }


@dataclass
class NameResolution:
    status: str
    class_id: Optional[str] = None
    subtype_id: Optional[str] = None
    matched_alias: Optional[str] = None


def _pick_alias_match(store: OntologyStore, normalized_name: str) -> Optional[NameResolution]:
    approved = [
        item for item in store.aliases
        if item.status in {"approved", "auto", "auto_approved"} and item.normalized_alias == normalized_name
    ]
    if not approved:
        return None
    approved.sort(key=lambda item: (item.confidence, not item.auto_generated), reverse=True)
    alias = approved[0]
    return NameResolution(
        status="subtype" if alias.subtype_id else "class",
        class_id=alias.class_id,
        subtype_id=alias.subtype_id,
        matched_alias=alias.alias,
    )


def resolve_observation_name(observation_name: str, store: OntologyStore) -> NameResolution:
    normalized = normalize_name(observation_name)
    alias_match = _pick_alias_match(store, normalized)
    if alias_match:
        return alias_match

    for subtype in store.subtypes.values():
        if normalize_name(subtype.name) == normalized:
            return NameResolution(status="subtype", class_id=subtype.class_id, subtype_id=subtype.subtype_id)

    for object_class in store.classes.values():
        if normalize_name(object_class.name) == normalized:
            return NameResolution(status="class", class_id=object_class.class_id)

    return NameResolution(status="unknown")


def _matches(operator: str, actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "eq":
        return actual == expected
    if operator == "neq":
        return actual != expected
    if operator == "in":
        return actual in expected
    if operator == "contains":
        return expected in actual
    raise ValueError(f"Unsupported operator: {operator}")


def build_runtime_label(
    observation_name: str,
    store: OntologyStore,
    context: Optional[RuntimeInstanceContext] = None,
) -> ResolvedRuntimeLabel:
    normalized = normalize_name(observation_name)
    resolution = resolve_observation_name(observation_name, store)

    if resolution.status == "unknown" or not resolution.class_id:
        return ResolvedRuntimeLabel(
            observation_name=observation_name,
            normalized_name=normalized,
            ontology_status="unknown",
            canonical_name=observation_name,
            object_class_id=None,
            object_subtype_id=None,
            matched_alias=None,
            affordance={},
            placement={},
            perception={},
            triggered_constraints=[],
        )

    object_class = store.classes[resolution.class_id]
    subtype = store.get_subtype(resolution.subtype_id)
    affordance = object_class.affordance.merged_with(subtype.affordance_override if subtype else None).to_dict()
    placement = object_class.placement.merged_with(subtype.placement_override if subtype else None).to_dict()
    perception = object_class.perception.merged_with(subtype.perception_override if subtype else None).to_dict()
    constraints = [*object_class.constraints, *(subtype.constraints if subtype else [])]

    triggered: List[str] = []
    if context is not None:
        for rule in constraints:
            actual_value = context.get(rule.source_field)
            if _matches(rule.operator, actual_value, rule.value):
                affordance[rule.target_field] = rule.effect_value
                if rule.reason:
                    triggered.append(rule.reason)

    return ResolvedRuntimeLabel(
        observation_name=observation_name,
        normalized_name=normalized,
        ontology_status=resolution.status,
        canonical_name=subtype.name if subtype else object_class.name,
        object_class_id=object_class.class_id,
        object_subtype_id=subtype.subtype_id if subtype else None,
        matched_alias=resolution.matched_alias,
        affordance=affordance,
        placement=placement,
        perception=perception,
        triggered_constraints=triggered,
    )
