from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    BaseAffordance,
    ConstraintRule,
    IntrinsicActionProfile,
    ObjectClass,
    ObjectSubtype,
    normalize_name,
)
from apps.data_flywheel.ontology.store import OntologyStore


@dataclass
class RuntimeInstanceContext:
    bbox: Optional[List[float]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    estimated_max_dimension_cm: Optional[float] = None
    mean_depth: Optional[float] = None
    in_container: Optional[bool] = None
    mask_polygon: Optional[List[List[float]]] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedRuntimeLabel:
    observation_name: str
    normalized_name: str
    ontology_status: str
    canonical_name: str
    object_class_id: Optional[str]
    object_subtype_id: Optional[str]
    matched_alias: Optional[str]
    operation: Optional[str]
    is_container: Optional[bool]
    container_placeability: Optional[str]
    placement_policy: Optional[str]
    extra_tags: List[str] = field(default_factory=list)
    triggered_constraints: List[str] = field(default_factory=list)
    perception_notes: List[str] = field(default_factory=list)
    role_ids: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(default_factory=list)
    preferred_operation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_LEADING_ARTICLES = {"a", "an", "the"}


def _singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalized_candidates(name: str) -> List[str]:
    normalized = normalize_name(name)
    if not normalized:
        return []

    candidates: List[str] = []

    def _add(tokens: List[str]) -> None:
        if not tokens:
            return
        phrase = " ".join(tokens).strip()
        if phrase and phrase not in candidates:
            candidates.append(phrase)
        singular_last = _singularize_token(tokens[-1])
        if singular_last != tokens[-1]:
            singular_phrase = " ".join([*tokens[:-1], singular_last]).strip()
            if singular_phrase and singular_phrase not in candidates:
                candidates.append(singular_phrase)

    tokens = normalized.split()
    _add(tokens)
    if tokens and tokens[0] in _LEADING_ARTICLES:
        _add(tokens[1:])
    return candidates


def _pick_alias_match(name: str, aliases: List[AliasEntry]) -> Optional[AliasEntry]:
    candidates = _normalized_candidates(name)
    for candidate in candidates:
        approved = [
            alias
            for alias in aliases
            if alias.normalized_alias == candidate and alias.status == "approved"
        ]
        if approved:
            return approved[0]

        auto_approved = [
            alias
            for alias in aliases
            if alias.normalized_alias == candidate and alias.status == "auto_approved"
        ]
        if auto_approved:
            return auto_approved[0]
    return None


def _resolve_entry(
    name: str, store: OntologyStore
) -> Tuple[Optional[ObjectClass], Optional[ObjectSubtype], Optional[str]]:
    candidates = _normalized_candidates(name)
    alias = _pick_alias_match(name, store.aliases)
    if alias:
        obj_class = store.classes.get(alias.class_id)
        if obj_class is None:
            return None, None, None
        obj_subtype = store.subtypes.get(alias.subtype_id) if alias.subtype_id else None
        return obj_class, obj_subtype, alias.alias

    for candidate in candidates:
        for subtype in store.subtypes.values():
            if normalize_name(subtype.name) == candidate:
                obj_class = store.classes.get(subtype.class_id)
                if obj_class is None:
                    return None, None, None
                return obj_class, subtype, None

        for obj_class in store.classes.values():
            if normalize_name(obj_class.name) == candidate:
                return obj_class, None, None

    return None, None, None


def _get_context_value(context: RuntimeInstanceContext, source_field: str) -> Any:
    if hasattr(context, source_field):
        return getattr(context, source_field)
    return context.extras.get(source_field)


def _rule_matches(rule: ConstraintRule, context_value: Any) -> bool:
    if context_value is None:
        return False
    if rule.operator == "gt":
        return context_value > rule.value
    if rule.operator == "gte":
        return context_value >= rule.value
    if rule.operator == "lt":
        return context_value < rule.value
    if rule.operator == "lte":
        return context_value <= rule.value
    if rule.operator == "eq":
        return context_value == rule.value
    if rule.operator == "neq":
        return context_value != rule.value
    if rule.operator == "in":
        try:
            return context_value in rule.value
        except TypeError:
            return False
    if rule.operator == "contains":
        try:
            return rule.value in context_value
        except TypeError:
            return False
    return False


def _build_unknown_label(name: str) -> ResolvedRuntimeLabel:
    return ResolvedRuntimeLabel(
        observation_name=name,
        normalized_name=name,
        ontology_status="unknown",
        canonical_name=name,
        object_class_id=None,
        object_subtype_id=None,
        matched_alias=None,
        operation=None,
        is_container=None,
        container_placeability=None,
        placement_policy=None,
        extra_tags=[],
        triggered_constraints=[],
        perception_notes=[],
        role_ids=[],
        allowed_operations=[],
        preferred_operation=None,
    )


def _resolve_container_state(
    role_ids: List[str], merged_affordance: BaseAffordance
) -> Tuple[List[str], Optional[bool]]:
    normalized_role_ids = list(dict.fromkeys(role_ids))
    has_container_role = "container" in normalized_role_ids
    if has_container_role:
        if merged_affordance.is_container is False:
            raise ValueError(
                "Conflicting container state: role_ids includes 'container' "
                "but affordance.is_container is False"
            )
        return normalized_role_ids, True

    if merged_affordance.is_container is True:
        if normalized_role_ids:
            raise ValueError(
                "Conflicting container state: role_ids omit 'container' "
                "but affordance.is_container is True"
            )
        return ["container"], True

    return normalized_role_ids, merged_affordance.is_container


def _resolve_operation_state(
    intrinsic_profile: IntrinsicActionProfile, merged_affordance: BaseAffordance
) -> Tuple[List[str], Optional[str]]:
    allowed_operations = list(dict.fromkeys(intrinsic_profile.allowed_operations))
    preferred_operation = intrinsic_profile.preferred_operation
    affordance_operation = merged_affordance.operation

    if preferred_operation is None and affordance_operation is not None:
        preferred_operation = affordance_operation
    elif (
        preferred_operation is not None
        and affordance_operation is not None
        and preferred_operation != affordance_operation
    ):
        raise ValueError(
            "Conflicting operation state: intrinsic_action_profile.preferred_operation "
            "does not match affordance.operation"
        )

    if not allowed_operations:
        if preferred_operation is not None:
            allowed_operations = [preferred_operation]
    if preferred_operation is None and allowed_operations:
        preferred_operation = allowed_operations[0]
    elif (
        allowed_operations
        and preferred_operation is not None
        and preferred_operation not in allowed_operations
    ):
        raise ValueError(
            "Conflicting operation state: preferred_operation is not listed in "
            "allowed_operations"
        )

    return allowed_operations, preferred_operation


def build_runtime_label(
    name: str,
    store: OntologyStore,
    context: Optional[RuntimeInstanceContext] = None,
) -> ResolvedRuntimeLabel:
    normalized_name = name.strip() if isinstance(name, str) else ""
    if not normalized_name:
        return _build_unknown_label(normalized_name)

    obj_class, obj_subtype, matched_alias = _resolve_entry(normalized_name, store)
    if obj_class is None:
        return _build_unknown_label(normalized_name)

    merged_affordance: BaseAffordance = obj_class.affordance.merged_with(
        obj_subtype.affordance_override if obj_subtype else None
    )
    allowed_operations, preferred_operation = _resolve_operation_state(
        obj_class.intrinsic_action_profile, merged_affordance
    )
    role_ids, resolved_is_container = _resolve_container_state(
        obj_class.role_ids, merged_affordance
    )

    perception_notes = list(obj_class.perception.extra_checks)
    if obj_subtype and obj_subtype.perception_override:
        perception_notes = list(
            dict.fromkeys(perception_notes + obj_subtype.perception_override.extra_checks)
        )

    label = ResolvedRuntimeLabel(
        observation_name=name,
        normalized_name=normalized_name,
        ontology_status="resolved",
        canonical_name=(obj_subtype.name if obj_subtype else obj_class.name),
        object_class_id=obj_class.class_id,
        object_subtype_id=(obj_subtype.subtype_id if obj_subtype else None),
        matched_alias=matched_alias,
        operation=preferred_operation,
        role_ids=role_ids,
        allowed_operations=allowed_operations,
        preferred_operation=preferred_operation,
        is_container=resolved_is_container,
        container_placeability=merged_affordance.container_placeability,
        placement_policy=obj_class.placement_policy,
        extra_tags=list(merged_affordance.extra_tags),
        triggered_constraints=[],
        perception_notes=perception_notes,
    )

    if context is None:
        return label

    constraints = list(obj_class.constraints)
    if obj_subtype:
        constraints.extend(obj_subtype.constraints)

    for rule in sorted(constraints, key=lambda item: item.priority):
        context_value = _get_context_value(context, rule.source_field)
        if not _rule_matches(rule, context_value):
            continue
        if hasattr(label, rule.target_field):
            setattr(label, rule.target_field, rule.effect_value)
        if rule.reason:
            label.triggered_constraints.append(rule.reason)

    return label


def resolve_observation_name(
    name: str,
    store: OntologyStore,
    context: Optional[RuntimeInstanceContext] = None,
) -> ResolvedRuntimeLabel:
    return build_runtime_label(name, store, context=context)
