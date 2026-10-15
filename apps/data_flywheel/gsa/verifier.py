"""GSA Verifier — Quality verification aligned with Viewer system.

Implements multi-level verification:
1. Schema validation (required fields, types)
2. Geometric consistency (bbox in image, point in bbox, direction unit vector)
3. Cross-label consistency (S facts match G objects, A_action references valid objects)
4. Hallucination detection (H axis from Viewer)
5. Format compliance (IF axis from Viewer)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .taxonomy import (
    Bucket, BUCKET_SUBTYPES, get_viewer_tag,
    SUBTYPE_TO_VIEWER_TAG, VIEWER_CAPABILITY_AXES,
)

logger = logging.getLogger("gsa.verifier")


@dataclass
class VerifyResult:
    """Result of verifying a single label."""
    label_id: str
    bucket: str
    subtype: str
    viewer_tag: str
    passed: bool
    score: float  # 0.0 - 1.0
    checks: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_id": self.label_id,
            "bucket": self.bucket,
            "subtype": self.subtype,
            "viewer_tag": self.viewer_tag,
            "passed": self.passed,
            "score": round(self.score, 4),
            "checks": self.checks,
            "issues": self.issues,
        }


@dataclass
class VerifyReport:
    """Aggregated verification report."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    by_bucket: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_viewer_tag: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    issues_summary: Dict[str, int] = field(default_factory=dict)
    results: List[VerifyResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.passed / self.total, 4) if self.total else 0,
            "avg_score": round(self.avg_score, 4),
            "by_bucket": self.by_bucket,
            "by_viewer_tag": self.by_viewer_tag,
            "issues_summary": self.issues_summary,
        }


class GSAVerifier:
    """Multi-level quality verifier aligned with Viewer system."""

    def __init__(self, image_size: Tuple[int, int] = (1280, 960)):
        self.image_w, self.image_h = image_size

    def verify_batch(self, labels: List[Dict[str, Any]]) -> VerifyReport:
        """Verify a batch of labels and produce a report."""
        report = VerifyReport()
        all_object_ids: Set[str] = set()
        all_bboxes: Dict[str, List[float]] = {}

        # First pass: collect all objects
        for label in labels:
            oid = label.get("object_id", "")
            if oid:
                all_object_ids.add(oid)
            answer = label.get("answer", {})
            if isinstance(answer, dict) and answer.get("bbox"):
                all_bboxes[oid] = answer["bbox"]

        # Second pass: verify each label
        scores = []
        for label in labels:
            result = self._verify_single(label, all_object_ids, all_bboxes)
            report.results.append(result)
            scores.append(result.score)

            if result.passed:
                report.passed += 1
            else:
                report.failed += 1

            # Aggregate by bucket
            b = result.bucket
            if b not in report.by_bucket:
                report.by_bucket[b] = {"total": 0, "passed": 0, "avg_score": 0, "scores": []}
            report.by_bucket[b]["total"] += 1
            report.by_bucket[b]["scores"].append(result.score)
            if result.passed:
                report.by_bucket[b]["passed"] += 1

            # Aggregate by viewer tag
            vt = result.viewer_tag
            if vt not in report.by_viewer_tag:
                report.by_viewer_tag[vt] = {"total": 0, "passed": 0, "scores": []}
            report.by_viewer_tag[vt]["total"] += 1
            report.by_viewer_tag[vt]["scores"].append(result.score)
            if result.passed:
                report.by_viewer_tag[vt]["passed"] += 1

            # Issues
            for issue in result.issues:
                report.issues_summary[issue] = report.issues_summary.get(issue, 0) + 1

        report.total = len(labels)
        report.avg_score = sum(scores) / len(scores) if scores else 0

        # Compute averages
        for b_data in report.by_bucket.values():
            s = b_data.pop("scores")
            b_data["avg_score"] = round(sum(s) / len(s), 4) if s else 0
        for vt_data in report.by_viewer_tag.values():
            s = vt_data.pop("scores")
            vt_data["avg_score"] = round(sum(s) / len(s), 4) if s else 0

        return report

    def _verify_single(self, label: Dict, all_objects: Set[str],
                       all_bboxes: Dict[str, List[float]]) -> VerifyResult:
        """Verify a single label through all checks."""
        bucket = label.get("bucket", "")
        subtype = label.get("subtype", "")
        viewer_tag = get_viewer_tag(bucket, subtype)
        label_id = f"{label.get('sample_id', '')}:{bucket}:{subtype}"

        checks = []
        issues = []
        score_parts = []

        # Check 1: Schema validation
        schema_ok = self._check_schema(label, bucket, subtype)
        checks.append({"name": "schema", "passed": schema_ok})
        score_parts.append(1.0 if schema_ok else 0.0)
        if not schema_ok:
            issues.append("schema_invalid")

        # Check 2: Geometric consistency
        geo_score = self._check_geometry(label, bucket)
        checks.append({"name": "geometry", "score": geo_score})
        score_parts.append(geo_score)
        if geo_score < 0.5:
            issues.append("geometry_invalid")

        # Check 3: Cross-reference consistency
        xref_score = self._check_cross_reference(label, bucket, all_objects, all_bboxes)
        checks.append({"name": "cross_reference", "score": xref_score})
        score_parts.append(xref_score)
        if xref_score < 0.5:
            issues.append("cross_ref_broken")

        # Check 4: Hallucination (H axis)
        halluc_score = self._check_hallucination(label, bucket, all_objects)
        checks.append({"name": "hallucination", "score": halluc_score})
        score_parts.append(halluc_score)
        if halluc_score < 0.5:
            issues.append("hallucination_detected")

        # Check 5: Format compliance (IF axis)
        format_score = self._check_format(label, bucket, subtype)
        checks.append({"name": "format_compliance", "score": format_score})
        score_parts.append(format_score)
        if format_score < 0.5:
            issues.append("format_noncompliant")

        # Check 6: Semantic consistency (A-dimension specific)
        semantic_score = self._check_semantic(label, bucket, subtype)
        checks.append({"name": "semantic_consistency", "score": semantic_score})
        score_parts.append(semantic_score)
        if semantic_score < 0.5:
            issues.append("semantic_inconsistent")

        total_score = sum(score_parts) / len(score_parts) if score_parts else 0
        passed = total_score >= 0.7 and not any(s < 0.3 for s in score_parts)

        return VerifyResult(
            label_id=label_id,
            bucket=bucket,
            subtype=subtype,
            viewer_tag=viewer_tag,
            passed=passed,
            score=total_score,
            checks=checks,
            issues=issues,
        )

    def _check_schema(self, label: Dict, bucket: str, subtype: str) -> bool:
        """Validate required fields are present and non-null."""
        required = {"sample_id", "bucket", "verifier_status"}

        if bucket == "G":
            required.update({"question", "answer"})
        elif bucket == "S":
            required.update({"question"})
        elif bucket == "A_state":
            required.update({"object_id"})
        elif bucket == "A_action":
            required.update({"object_id", "action_type", "grasp_point_2d"})
        elif bucket == "A_identity_attr":
            required.update({"object_id", "object_class"})
        elif bucket == "A_obj_state":
            required.update({"object_id", "state_type", "state_value"})
        elif bucket == "A_robot_state":
            required.update({"camera_view", "state_type", "state_value"})
        elif bucket == "A_action_precondition":
            required.update({"object_id", "action_type"})
        elif bucket == "A_action_interface":
            required.update({"object_id", "action_type"})

        for field in required:
            val = label.get(field)
            if val is None or val == "":
                return False
        return True

    def _check_geometry(self, label: Dict, bucket: str) -> float:
        """Check geometric validity of coordinates."""
        score = 1.0

        # Check bbox bounds (shared across multiple pipelines)
        bbox = label.get("bbox")
        if not bbox:
            answer = label.get("answer", {})
            if isinstance(answer, dict):
                bbox = answer.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            if x1 >= x2 or y1 >= y2:
                score -= 0.5
            if x1 < 0 or y1 < 0:
                score -= 0.2
            if x2 > self.image_w * 1.5 or y2 > self.image_h * 1.5:
                score -= 0.2

        # Check grasp point in reasonable range
        grasp = label.get("grasp_point_2d")
        if grasp and len(grasp) == 2:
            if grasp[0] < 0 or grasp[1] < 0:
                score -= 0.3
            if grasp[0] > self.image_w * 1.5 or grasp[1] > self.image_h * 1.5:
                score -= 0.3

        # Check jaw_axis is unit-ish vector
        jaw = label.get("jaw_axis_2d")
        if jaw and len(jaw) == 2:
            mag = math.sqrt(jaw[0]**2 + jaw[1]**2)
            if mag < 0.01:
                score -= 0.5
            elif abs(mag - 1.0) > 0.5:
                score -= 0.2

        # Check push direction
        push_dir = label.get("push_direction_2d")
        if push_dir and len(push_dir) == 2:
            mag = math.sqrt(push_dir[0]**2 + push_dir[1]**2)
            if mag < 0.01:
                score -= 0.5

        # Check push contact point
        push_pt = label.get("push_contact_point_2d")
        if push_pt and len(push_pt) == 2:
            if push_pt[0] < 0 or push_pt[1] < 0:
                score -= 0.3
            if push_pt[0] > self.image_w * 1.5 or push_pt[1] > self.image_h * 1.5:
                score -= 0.3

        return max(0.0, score)

    def _check_cross_reference(self, label: Dict, bucket: str,
                               all_objects: Set[str],
                               all_bboxes: Dict[str, List[float]]) -> float:
        """Check that references to other objects are valid."""
        score = 1.0

        # A_action and A_action_interface should reference known objects
        if bucket in ("A_action", "A_action_interface", "A_action_precondition",
                      "A_identity_attr", "A_obj_state"):
            obj_id = label.get("object_id", "")
            if obj_id and obj_id not in all_objects:
                score -= 0.3

        # blocked_by references
        blocked = label.get("blocked_by", [])
        if blocked:
            for b in blocked:
                if b not in all_objects:
                    score -= 0.2

        # S labels should reference known objects
        if bucket == "S":
            subj = label.get("subject_id", "")
            obj = label.get("object_id", "")
            if subj and subj not in all_objects:
                score -= 0.3
            if obj and obj not in all_objects:
                score -= 0.3

        # A_action_precondition: can_do_now=false needs failure_reason
        if bucket == "A_action_precondition":
            if not label.get("can_do_now", True):
                if not label.get("failure_reason"):
                    score -= 0.4

        return max(0.0, score)

    def _check_hallucination(self, label: Dict, bucket: str,
                             all_objects: Set[str]) -> float:
        """Check for potential hallucinations (H axis)."""
        score = 1.0

        if bucket == "G":
            answer = label.get("answer", {})
            if isinstance(answer, dict):
                obj_name = answer.get("object_name", "")
                # Flag generic/placeholder names
                if obj_name in ("unknown", "object", "thing", "item", ""):
                    score -= 0.3

        # Check for impossible counts
        if bucket == "G" and label.get("subtype") == "counting":
            answer = label.get("answer", {})
            if isinstance(answer, dict):
                count = answer.get("count", 0)
                if count and count > 50:
                    score -= 0.5  # Likely hallucinated

        return max(0.0, score)

    def _check_format(self, label: Dict, bucket: str, subtype: str) -> float:
        """Check output format compliance (IF axis)."""
        score = 1.0

        # bbox should be [x1, y1, x2, y2] integers
        answer = label.get("answer", {})
        if isinstance(answer, dict):
            bbox = answer.get("bbox")
            if bbox:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    score -= 0.5
                elif not all(isinstance(v, (int, float)) for v in bbox):
                    score -= 0.3

        # grasp_point should be [x, y] numbers
        grasp = label.get("grasp_point_2d")
        if grasp:
            if not isinstance(grasp, list) or len(grasp) != 2:
                score -= 0.5
            elif not all(isinstance(v, (int, float)) for v in grasp):
                score -= 0.3

        # action_type should be valid
        if bucket in ("A_action", "A_action_interface", "A_action_precondition"):
            action = label.get("action_type", "")
            if action not in ("pick", "push", "place", "reorient"):
                score -= 0.5

        # A_identity_attr: attributes dict format
        if bucket == "A_identity_attr":
            attrs = label.get("attributes")
            if attrs is not None and not isinstance(attrs, dict):
                score -= 0.5

        # A_obj_state: state_type + state_value format
        if bucket == "A_obj_state":
            if not isinstance(label.get("state_type", ""), str):
                score -= 0.3
            if not isinstance(label.get("state_value", ""), str):
                score -= 0.3

        # A_robot_state: camera_view format
        if bucket == "A_robot_state":
            cv = label.get("camera_view", "")
            if cv not in ("wrist", "head", "external"):
                score -= 0.4

        return max(0.0, score)

    def _check_semantic(self, label: Dict, bucket: str, subtype: str) -> float:
        """Check semantic consistency: object-state plausibility, action logic."""
        score = 1.0

        # A_obj_state: state must be plausible for the object type
        if bucket == "A_obj_state":
            obj_class = label.get("object_class", "").lower()
            state_type = label.get("state_type", "")
            state_value = label.get("state_value", "")

            # Containers can have occupancy; non-containers shouldn't
            non_container_kw = {"pen", "key", "phone", "coin", "card", "knife"}
            if state_type == "occupancy" and any(k in obj_class for k in non_container_kw):
                score = 0.0  # Hard fail

            # Only openable objects should have open_close state
            non_openable_kw = {"ball", "pen", "spoon", "fork", "plate", "coin"}
            if state_type == "open_close" and any(k in obj_class for k in non_openable_kw):
                score = 0.0

            # on_off only for electronic/mechanical objects
            non_electronic_kw = {"cup", "plate", "spoon", "fork", "ball", "book"}
            if state_type == "on_off" and any(k in obj_class for k in non_electronic_kw):
                score = 0.0

            # Confidence too low
            conf = label.get("confidence", 1.0)
            if conf is not None and conf < 0.5:
                score = min(score, 0.3)

        # A_action_precondition: logical consistency
        if bucket == "A_action_precondition":
            can_do = label.get("can_do_now", True)
            failure = label.get("failure_reason")
            blocked = label.get("blocked_by", [])

            # can_do=True but has blockers -> hard fail
            if can_do and blocked:
                score = 0.0

            # can_do=False but no reason -> incomplete
            if not can_do and not failure:
                score = min(score, 0.2)

        # A_action_interface: grasp point must be inside bbox
        if bucket in ("A_action_interface", "A_action"):
            bbox = label.get("bbox")
            grasp = label.get("grasp_point_2d")
            if bbox and grasp and len(bbox) == 4 and len(grasp) == 2:
                x1, y1, x2, y2 = bbox
                gx, gy = grasp
                margin = max((x2 - x1), (y2 - y1)) * 0.3
                if gx < x1 - margin or gx > x2 + margin:
                    score = 0.0
                if gy < y1 - margin or gy > y2 + margin:
                    score = 0.0

        # A_identity_attr: object_class should not be generic placeholder
        if bucket == "A_identity_attr":
            obj_class = label.get("object_class", "")
            generic_names = {"object", "thing", "item", "unknown", "stuff", ""}
            if obj_class.lower() in generic_names:
                score = 0.0

        return max(0.0, score)
