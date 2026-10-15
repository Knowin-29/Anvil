"""候选簇审核动作。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Optional

from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    BaseAffordance,
    GovernanceMetadata,
    ObjectClass,
    ObjectSubtype,
    slugify_identifier,
)
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.review.observation import CandidateCluster, load_candidate_clusters, save_candidate_clusters


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PromotionAction:
    cluster_id: str
    action_type: str
    target_class_id: str = ""
    target_subtype_id: str = ""
    new_name: str = ""
    reviewer: str = ""
    notes: str = ""
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PromotionAction":
        return cls(**data)


def load_review_actions(path: str) -> List[PromotionAction]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [PromotionAction.from_dict(json.loads(line)) for line in f if line.strip()]


def _append_action(path: str, action: PromotionAction) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(action.to_dict(), ensure_ascii=False) + "\n")


def _find_cluster(clusters: List[CandidateCluster], cluster_id: str) -> CandidateCluster:
    for cluster in clusters:
        if cluster.cluster_id == cluster_id:
            return cluster
    raise KeyError(f"Unknown cluster_id: {cluster_id}")


def _cluster_aliases(cluster: CandidateCluster, class_id: str, subtype_id: Optional[str]) -> List[AliasEntry]:
    aliases = []
    for alias in cluster.labels.keys():
        aliases.append(
            AliasEntry(
                alias=alias,
                class_id=class_id,
                subtype_id=subtype_id,
                status="approved",
                source="review",
                confidence=1.0,
                auto_generated=True,
            )
        )
    return aliases


def _add_cluster_aliases(store: OntologyStore, cluster: CandidateCluster, class_id: str, subtype_id: Optional[str]) -> List[str]:
    added_aliases: List[str] = []
    for alias in _cluster_aliases(cluster, class_id, subtype_id):
        if store.add_alias(alias):
            added_aliases.append(alias.alias)
    return added_aliases


def _rollback_cluster_mapping(store: OntologyStore, cluster: CandidateCluster) -> None:
    promoted_aliases = set(cluster.promoted_aliases)
    promoted_class_id = cluster.promoted_class_id
    promoted_subtype_id = cluster.promoted_subtype_id
    if promoted_aliases and promoted_class_id:
        store.aliases = [
            item for item in store.aliases
            if not (
                item.alias in promoted_aliases
                and item.class_id == promoted_class_id
                and item.subtype_id == (promoted_subtype_id or None)
            )
        ]
    created_target_kind = cluster.created_target_kind
    created_target_id = cluster.created_target_id
    if created_target_kind == "subtype" and created_target_id:
        still_referenced = any(item.subtype_id == created_target_id for item in store.aliases)
        if not still_referenced:
            store.subtypes.pop(created_target_id, None)
    elif created_target_kind == "class" and created_target_id:
        has_alias = any(item.class_id == created_target_id for item in store.aliases)
        has_subtype = any(item.class_id == created_target_id for item in store.subtypes.values())
        if not has_alias and not has_subtype:
            store.classes.pop(created_target_id, None)


def apply_review_action(
    store_path: str,
    clusters_path: str,
    audit_path: str,
    action: PromotionAction,
) -> Dict[str, object]:
    store = OntologyStore.load(store_path)
    clusters = load_candidate_clusters(clusters_path)
    cluster = _find_cluster(clusters, action.cluster_id)

    if action.action_type == "create_subtype":
        if not action.target_class_id or not action.new_name:
            raise ValueError("create_subtype requires target_class_id and new_name")
        subtype_id = slugify_identifier(action.new_name)
        subtype = ObjectSubtype(
            subtype_id=subtype_id,
            class_id=action.target_class_id,
            name=action.new_name,
            metadata=GovernanceMetadata(status="approved", source="review", reviewer=action.reviewer, notes=action.notes),
        )
        store.add_subtype(subtype)
        added_aliases = _add_cluster_aliases(store, cluster, action.target_class_id, subtype_id)
        cluster.status = "promoted"
        cluster.suggested_class_id = action.target_class_id
        cluster.suggested_subtype_id = subtype_id
        cluster.promoted_aliases = added_aliases
        cluster.promoted_class_id = action.target_class_id
        cluster.promoted_subtype_id = subtype_id
        cluster.created_target_kind = "subtype"
        cluster.created_target_id = subtype_id
    elif action.action_type == "link_subtype":
        if not action.target_subtype_id:
            raise ValueError("link_subtype requires target_subtype_id")
        subtype = store.get_subtype(action.target_subtype_id)
        if subtype is None:
            raise KeyError(f"Unknown subtype_id: {action.target_subtype_id}")
        added_aliases = _add_cluster_aliases(store, cluster, subtype.class_id, subtype.subtype_id)
        cluster.status = "promoted"
        cluster.suggested_class_id = subtype.class_id
        cluster.suggested_subtype_id = subtype.subtype_id
        cluster.promoted_aliases = added_aliases
        cluster.promoted_class_id = subtype.class_id
        cluster.promoted_subtype_id = subtype.subtype_id
        cluster.created_target_kind = ""
        cluster.created_target_id = ""
    elif action.action_type == "create_class":
        class_name = action.new_name or cluster.representative_label
        class_id = slugify_identifier(class_name)
        object_class = ObjectClass(
            class_id=class_id,
            name=class_name,
            affordance=BaseAffordance(),
            metadata=GovernanceMetadata(
                status="draft",
                source="review",
                reviewer=action.reviewer,
                notes=action.notes,
            ),
        )
        store.add_class(object_class)
        added_aliases = _add_cluster_aliases(store, cluster, class_id, None)
        cluster.status = "promoted"
        cluster.suggested_class_id = class_id
        cluster.suggested_subtype_id = None
        cluster.promoted_aliases = added_aliases
        cluster.promoted_class_id = class_id
        cluster.promoted_subtype_id = ""
        cluster.created_target_kind = "class"
        cluster.created_target_id = class_id
    elif action.action_type == "reject":
        if cluster.promoted_aliases or cluster.created_target_kind or cluster.promoted_class_id or cluster.promoted_subtype_id:
            _rollback_cluster_mapping(store, cluster)
        cluster.status = "rejected"
        cluster.suggested_class_id = None
        cluster.suggested_subtype_id = None
        cluster.promoted_aliases = []
        cluster.promoted_class_id = ""
        cluster.promoted_subtype_id = ""
        cluster.created_target_kind = ""
        cluster.created_target_id = ""
    elif action.action_type == "mark_mismerge":
        _rollback_cluster_mapping(store, cluster)
        cluster.status = "needs_review"
        cluster.suggested_class_id = None
        cluster.suggested_subtype_id = None
        cluster.promoted_aliases = []
        cluster.promoted_class_id = ""
        cluster.promoted_subtype_id = ""
        cluster.created_target_kind = ""
        cluster.created_target_id = ""
    else:
        raise ValueError(f"Unsupported action_type: {action.action_type}")

    store.save(store_path)
    save_candidate_clusters(clusters_path, clusters)
    _append_action(audit_path, action)
    return {
        "store_path": store_path,
        "clusters_path": clusters_path,
        "audit_path": audit_path,
        "action": action.to_dict(),
        "cluster": cluster.to_dict(),
    }


__all__ = [
    "PromotionAction",
    "apply_review_action",
    "load_candidate_clusters",
    "load_review_actions",
]
