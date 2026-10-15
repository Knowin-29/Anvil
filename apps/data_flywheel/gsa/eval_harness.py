"""GSA Eval Harness — Offline evaluation without VLM calls.

Reads manifest + labels + predictions, outputs scores and review queue.
Each bucket gets either an automatic score or a verifiable_rate.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .action_schema import ACTION_REQUIRED_FIELDS, AActionLabel, validate_a_action_label
from .metrics import (
    bbox_iou,
    blocked_by_consistency,
    direction_cosine,
    exact_match,
    point_in_bbox,
    point_in_polygon,
    required_fields_score,
    verifiable_rate,
)
from .taxonomy import Bucket, BUCKET_SUBTYPES

logger = logging.getLogger("gsa.eval_harness")


@dataclass
class BucketScore:
    bucket: str
    total: int = 0
    auto_verifiable: int = 0
    review_required: int = 0
    heuristic_only: int = 0
    avg_score: float = 0.0
    verifiable_rate: float = 0.0
    subtype_scores: Dict[str, float] = field(default_factory=dict)
    subtype_counts: Dict[str, int] = field(default_factory=dict)
    state_value_legal_rate: float = 0.0


@dataclass
class EvalResults:
    bucket_scores: Dict[str, BucketScore] = field(default_factory=dict)
    review_queue: List[Dict[str, Any]] = field(default_factory=list)
    total_labels: int = 0
    total_auto_verifiable: int = 0
    overall_verifiable_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_labels": self.total_labels,
            "total_auto_verifiable": self.total_auto_verifiable,
            "overall_verifiable_rate": self.overall_verifiable_rate,
            "bucket_scores": {
                k: {
                    "bucket": v.bucket,
                    "total": v.total,
                    "auto_verifiable": v.auto_verifiable,
                    "review_required": v.review_required,
                    "heuristic_only": v.heuristic_only,
                    "avg_score": round(v.avg_score, 4),
                    "verifiable_rate": round(v.verifiable_rate, 4),
                    "subtype_scores": v.subtype_scores,
                    "subtype_counts": v.subtype_counts,
                    "state_value_legal_rate": round(v.state_value_legal_rate, 4),
                }
                for k, v in self.bucket_scores.items()
            },
            "review_queue_size": len(self.review_queue),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class GSAEvalHarness:
    """Offline evaluator for GSA labels. No VLM calls required."""

    def evaluate(self, labels: List[Dict[str, Any]],
                 spatial_facts: Optional[List[Dict[str, Any]]] = None) -> EvalResults:
        """Evaluate a set of GSA labels."""
        results = EvalResults()
        spatial_facts = spatial_facts or []

        # Group by bucket
        by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for label in labels:
            bucket = label.get("bucket", "")
            by_bucket[bucket].append(label)

        results.total_labels = len(labels)

        for bucket_name, bucket_labels in by_bucket.items():
            score = self._evaluate_bucket(bucket_name, bucket_labels, spatial_facts)
            results.bucket_scores[bucket_name] = score
            results.total_auto_verifiable += score.auto_verifiable

            # Add review items
            for label in bucket_labels:
                if label.get("verifier_status") == "review_required":
                    results.review_queue.append({
                        "sample_id": label.get("sample_id", ""),
                        "bucket": bucket_name,
                        "object_id": label.get("object_id", ""),
                        "subtype": label.get("subtype", ""),
                        "reason": "review_required_status",
                    })

        if results.total_labels > 0:
            results.overall_verifiable_rate = results.total_auto_verifiable / results.total_labels

        return results

    def _evaluate_bucket(self, bucket_name: str, labels: List[Dict[str, Any]],
                         spatial_facts: List[Dict[str, Any]]) -> BucketScore:
        score = BucketScore(bucket=bucket_name, total=len(labels))

        subtype_scores: Dict[str, List[float]] = defaultdict(list)
        subtype_counts: Dict[str, int] = defaultdict(int)
        legal_state_values = 0

        for label in labels:
            status = label.get("verifier_status", "")
            if status == "auto_verifiable":
                score.auto_verifiable += 1
            elif status == "review_required":
                score.review_required += 1
            elif status == "heuristic_only":
                score.heuristic_only += 1

            subtype = label.get("subtype", "unknown")
            subtype_counts[subtype] += 1

            if bucket_name == Bucket.A_ROBOT_STATE.value:
                if label.get("state_value") in {"yes", "no", "unknown"}:
                    legal_state_values += 1

            # Compute per-label score
            label_score = self._score_label(bucket_name, label, spatial_facts)
            if label_score is not None:
                subtype_scores[subtype].append(label_score)

        # Aggregate
        all_scores = []
        for st, scores in subtype_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                score.subtype_scores[st] = round(avg, 4)
                all_scores.extend(scores)

        score.subtype_counts = dict(subtype_counts)
        score.avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        score.verifiable_rate = score.auto_verifiable / score.total if score.total > 0 else 0.0
        if bucket_name == Bucket.A_ROBOT_STATE.value and score.total > 0:
            score.state_value_legal_rate = legal_state_values / score.total

        return score

    def _score_label(self, bucket: str, label: Dict[str, Any],
                     spatial_facts: List[Dict[str, Any]]) -> Optional[float]:
        """Score a single label. Returns None if not scoreable."""
        if bucket == Bucket.G.value:
            return self._score_g_label(label)
        elif bucket == Bucket.S.value:
            return self._score_s_label(label)
        elif bucket == Bucket.A_STATE.value:
            return self._score_a_state_label(label)
        elif bucket == Bucket.A_ACTION.value:
            return self._score_a_action_label(label, spatial_facts)
        elif bucket == Bucket.A_ROBOT_STATE.value:
            return self._score_a_robot_state_label(label)
        return None

    def _score_g_label(self, label: Dict[str, Any]) -> float:
        """Score a G label: check bbox exists and answer is complete."""
        answer = label.get("answer", {})
        if not isinstance(answer, dict):
            return 0.0
        has_bbox = answer.get("bbox") is not None and len(answer.get("bbox", [])) == 4
        has_name = bool(answer.get("object_name") or answer.get("count") is not None)
        return 1.0 if (has_bbox or has_name) else 0.0

    def _score_s_label(self, label: Dict[str, Any]) -> float:
        """Score an S label: check predicate and subject/object exist."""
        has_subject = bool(label.get("subject_id"))
        has_object = bool(label.get("object_id"))
        has_predicate = bool(label.get("predicate"))
        score = (int(has_subject) + int(has_object) + int(has_predicate)) / 3
        return score

    def _score_a_state_label(self, label: Dict[str, Any]) -> float:
        """Score an A_state label."""
        has_obj_id = bool(label.get("object_id"))
        has_name = bool(label.get("object_name"))
        subtype = label.get("subtype", "")

        if subtype == "state_recognition":
            has_key = bool(label.get("state_key"))
            has_value = bool(label.get("state_value"))
            return (int(has_obj_id) + int(has_name) + int(has_key) + int(has_value)) / 4
        else:
            return (int(has_obj_id) + int(has_name)) / 2

    def _score_a_robot_state_label(self, label: Dict[str, Any]) -> float:
        """Score an A_robot_state label."""
        has_camera = bool(label.get("camera_view"))
        has_type = bool(label.get("state_type"))
        has_value = label.get("state_value") in {"yes", "no", "unknown"}
        has_image = bool(label.get("image_path"))
        return (int(has_camera) + int(has_type) + int(has_value) + int(has_image)) / 4

    def _score_a_action_label(self, label: Dict[str, Any],
                              spatial_facts: List[Dict[str, Any]]) -> float:
        """Score an A_action label: required fields + consistency."""
        action_type = label.get("action_type", "")
        required = ACTION_REQUIRED_FIELDS.get(action_type, set())

        fields_score = required_fields_score(label, required)

        # Check blocked_by consistency if applicable
        consistency = 1.0
        if action_type == "pick" and label.get("blocked_by") is not None:
            consistency = blocked_by_consistency(
                label.get("blocked_by", []),
                spatial_facts,
                label.get("object_id", ""),
            )

        return (fields_score + consistency) / 2
