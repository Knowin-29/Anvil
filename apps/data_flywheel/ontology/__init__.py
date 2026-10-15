"""Public ontology package surface for downstream modules."""

from apps.data_flywheel.ontology.matcher import (
    ResolvedRuntimeLabel,
    RuntimeInstanceContext,
    build_runtime_label,
    resolve_observation_name,
)
from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    BaseAffordance,
    DescriptorSchema,
    ConstraintRule,
    ContainerProfile,
    GovernanceMetadata,
    IntrinsicActionProfile,
    KNOWN_EXTRA_TAGS,
    ObjectClass,
    ObjectSubtype,
    PLACEMENT_POLICIES,
    PerceptionProfile,
    PlacementPreference,
    normalize_name,
    slugify_identifier,
)
from apps.data_flywheel.ontology.store import OntologyStore, QueryEntry

__all__ = [
    "AliasEntry",
    "BaseAffordance",
    "DescriptorSchema",
    "ConstraintRule",
    "ContainerProfile",
    "GovernanceMetadata",
    "IntrinsicActionProfile",
    "KNOWN_EXTRA_TAGS",
    "ObjectClass",
    "ObjectSubtype",
    "OntologyStore",
    "PLACEMENT_POLICIES",
    "PerceptionProfile",
    "PlacementPreference",
    "QueryEntry",
    "ResolvedRuntimeLabel",
    "RuntimeInstanceContext",
    "build_runtime_label",
    "normalize_name",
    "resolve_observation_name",
    "slugify_identifier",
]
