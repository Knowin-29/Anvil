"""审核工作台辅助函数。"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.review.metrics import compute_governance_metrics, save_governance_metrics
from apps.data_flywheel.review.observation import load_candidate_clusters, load_observations
from apps.data_flywheel.review.promotion import PromotionAction, apply_review_action, load_review_actions


def render_clusters_markdown(round_dir: str, max_clusters: int = 20) -> str:
    clusters = load_candidate_clusters(os.path.join(round_dir, "candidate_clusters.jsonl"))
    if not clusters:
        return "无候选簇数据。"
    parts: List[str] = ["### 候选簇", "| cluster_id | 代表标签 | 计数 | 建议类目 | 建议子类 | 状态 |", "|---|---|---:|---|---|---|"]
    for cluster in clusters[:max_clusters]:
        suggestion_class = cluster.suggested_class_id or "-"
        suggestion_subtype = cluster.suggested_subtype_id or "-"
        parts.append(
            f"| {cluster.cluster_id} | {cluster.representative_label} | {cluster.count} | "
            f"{suggestion_class} | {suggestion_subtype} | {cluster.status} |"
        )
    return "\n".join(parts)


def render_store_markdown(store_path: str) -> str:
    store = OntologyStore.load(store_path)
    if not store.classes:
        return "无正式词表，等待首次审核入库。"
    parts: List[str] = [
        "### 正式词表",
        "| 大类ID | 名称 | 子类数 | 自动别名 | placement_policy |",
        "|---|---|---:|---|---|",
    ]
    for class_id, object_class in sorted(store.classes.items()):
        subtype_count = len(store.list_subtypes(class_id))
        parts.append(
            f"| {class_id} | {object_class.name} | {subtype_count} | "
            f"{'是' if object_class.alias_auto_approve else '否'} | "
            f"{object_class.placement_policy or '-'} |"
        )
    if store.subtypes:
        parts.append("\n#### 子类")
        parts.append("| 子类ID | 所属大类 | 名称 |")
        parts.append("|---|---|---|")
        for subtype_id, subtype in sorted(store.subtypes.items()):
            parts.append(f"| {subtype_id} | {subtype.class_id} | {subtype.name} |")
    return "\n".join(parts)


def _refresh_metrics(round_dir: str) -> Dict[str, object]:
    observations = load_observations(os.path.join(round_dir, "ontology_observations.jsonl"))
    clusters = load_candidate_clusters(os.path.join(round_dir, "candidate_clusters.jsonl"))
    actions = load_review_actions(os.path.join(round_dir, "review_actions.jsonl"))
    metrics = compute_governance_metrics(observations, clusters, actions)
    save_governance_metrics(os.path.join(round_dir, "ontology_metrics.json"), metrics)
    return metrics


def render_metrics_markdown(round_dir: str) -> str:
    metrics_path = os.path.join(round_dir, "ontology_metrics.json")
    if not (
        os.path.exists(metrics_path)
        or os.path.exists(os.path.join(round_dir, "ontology_observations.jsonl"))
        or os.path.exists(os.path.join(round_dir, "candidate_clusters.jsonl"))
    ):
        return "无治理指标。"
    metrics = _refresh_metrics(round_dir)
    parts = ["### 治理指标", "| 指标 | 数值 |", "|---|---:|"]
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"| {key} | {value:.4f} |")
        else:
            parts.append(f"| {key} | {value} |")
    return "\n".join(parts)


def build_review_dashboard(round_dir: str, store_path: str, max_clusters: int = 20) -> Dict[str, object]:
    store = OntologyStore.load(store_path)
    clusters = load_candidate_clusters(os.path.join(round_dir, "candidate_clusters.jsonl"))
    return {
        "cluster_choices": [item.cluster_id for item in clusters[:max_clusters]],
        "class_choices": sorted(store.classes.keys()),
        "subtype_choices": sorted(store.subtypes.keys()),
        "clusters_markdown": render_clusters_markdown(round_dir, max_clusters=max_clusters),
        "store_markdown": render_store_markdown(store_path),
        "metrics_markdown": render_metrics_markdown(round_dir),
    }


def apply_review_action_from_round(
    round_dir: str,
    store_path: str,
    action: PromotionAction,
) -> Dict[str, object]:
    clusters_path = os.path.join(round_dir, "candidate_clusters.jsonl")
    audit_path = os.path.join(round_dir, "review_actions.jsonl")
    result = apply_review_action(store_path=store_path, clusters_path=clusters_path, audit_path=audit_path, action=action)
    result["metrics"] = _refresh_metrics(round_dir)
    return result
