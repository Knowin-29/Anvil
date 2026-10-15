"""词表治理指标。"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

from apps.data_flywheel.review.observation import CandidateCluster, Observation
from apps.data_flywheel.review.promotion import PromotionAction


def compute_governance_metrics(
    observations: Iterable[Observation],
    clusters: Iterable[CandidateCluster],
    actions: Iterable[PromotionAction],
) -> Dict[str, float]:
    observations = list(observations)
    clusters = list(clusters)
    actions = list(actions)
    total_obs = len(observations)
    matched_obs = sum(1 for item in observations if item.matched_class_id or item.matched_subtype_id)
    alias_matched_obs = sum(1 for item in observations if item.match_source == "alias")
    total_clusters = len(clusters)
    promoted_clusters = sum(1 for item in clusters if item.status == "promoted")
    rejected_clusters = sum(1 for item in clusters if item.status == "rejected")
    promotion_actions = sum(1 for item in actions if item.action_type in {"create_class", "create_subtype", "link_subtype"})
    mismerge_actions = sum(1 for item in actions if item.action_type == "mark_mismerge")

    return {
        "total_observations": total_obs,
        "alias_hit_rate": alias_matched_obs / total_obs if total_obs else 0.0,
        "ontology_match_rate": matched_obs / total_obs if total_obs else 0.0,
        "unknown_candidate_rate": (total_obs - matched_obs) / total_obs if total_obs else 0.0,
        "total_candidate_clusters": total_clusters,
        "candidate_cluster_promotion_rate": promoted_clusters / total_clusters if total_clusters else 0.0,
        "candidate_cluster_rejection_rate": rejected_clusters / total_clusters if total_clusters else 0.0,
        "promotion_actions": promotion_actions,
        "mismerge_rate": mismerge_actions / promotion_actions if promotion_actions else 0.0,
    }


def save_governance_metrics(path: str, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
