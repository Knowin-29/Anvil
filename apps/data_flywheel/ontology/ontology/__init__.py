"""离线词表（ontology）核心数据结构与匹配能力。"""

from apps.data_flywheel.ontology.matcher import (
    ResolvedRuntimeLabel,
    RuntimeInstanceContext,
    build_runtime_label,
    resolve_observation_name,
)
from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    BaseAffordance,
    ConstraintRule,
    GovernanceMetadata,
    ObjectClass,
    ObjectSubtype,
    PerceptionRequired,
    PlacementProfile,
)
from apps.data_flywheel.ontology.store import OntologyStore

__all__ = [
    "AliasEntry",
    "BaseAffordance",
    "ConstraintRule",
    "GovernanceMetadata",
    "ObjectClass",
    "ObjectSubtype",
    "OntologyStore",
    "PerceptionRequired",
    "PlacementProfile",
    "ResolvedRuntimeLabel",
    "RuntimeInstanceContext",
    "build_runtime_label",
    "resolve_observation_name",
]
