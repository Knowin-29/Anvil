"""将 grounding 结果转换为词表审核素材。"""

from __future__ import annotations

import os
from typing import Dict, Optional

from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.review.metrics import compute_governance_metrics, save_governance_metrics
from apps.data_flywheel.review.observation import (
    cluster_observations,
    collect_observations_from_grounding,
)
from apps.data_flywheel.review.promotion import load_review_actions


def prepare_review_artifacts(
    grounding_jsonl: str,
    round_dir: str,
    ontology_store_path: Optional[str] = None,
) -> Dict[str, object]:
    os.makedirs(round_dir, exist_ok=True)
    observations_path = os.path.join(round_dir, "ontology_observations.jsonl")
    clusters_path = os.path.join(round_dir, "candidate_clusters.jsonl")
    metrics_path = os.path.join(round_dir, "ontology_metrics.json")
    crop_dir = os.path.join(round_dir, "ontology_crops")
    audit_path = os.path.join(round_dir, "review_actions.jsonl")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("")

    store = OntologyStore.load(ontology_store_path or "")
    observations = collect_observations_from_grounding(
        grounding_jsonl=grounding_jsonl,
        ontology_store=store,
        output_jsonl=observations_path,
        crop_dir=crop_dir,
    )
    clusters = cluster_observations(observations, output_jsonl=clusters_path)
    actions = load_review_actions(audit_path)
    metrics = compute_governance_metrics(observations, clusters, actions)
    save_governance_metrics(metrics_path, metrics)

    return {
        "observations": {
            "count": len(observations),
            "output_jsonl": observations_path,
        },
        "clusters": {
            "count": len(clusters),
            "output_jsonl": clusters_path,
        },
        "metrics": metrics,
        "metrics_path": metrics_path,
        "audit_path": audit_path,
        "crop_dir": crop_dir,
        "ontology_store_path": ontology_store_path,
    }
