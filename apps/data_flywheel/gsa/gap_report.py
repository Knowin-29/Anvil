"""GSA Gap Report — Aggregate coverage and identify gaps.

Generates JSON and HTML reports showing:
- Coverage by bucket / subtype / scene family / source
- Score and verifiable rate aggregation
- Gap owner assignment
- Top missing subtypes
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .eval_harness import EvalResults
from .taxonomy import Bucket, BUCKET_SUBTYPES, SceneFamily

logger = logging.getLogger("gsa.gap_report")


class GapOwner:
    REAL_LOG = "real_log"
    PUBLIC_ADAPTER = "public_adapter"
    SIM = "sim"
    MANUAL_REVIEW = "manual_review"
    MODEL_VERIFIER = "model_verifier"
    SCHEMA = "schema"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class GapEntry:
    bucket: str
    subtype: str
    scene_family: str
    gap_type: str  # missing_data | low_quality | no_verifier
    gap_owner: str
    severity: str  # critical | high | medium | low
    current_count: int = 0
    target_count: int = 100
    score: float = 0.0
    recommendation: str = ""


@dataclass
class GapReport:
    gaps: List[GapEntry] = field(default_factory=list)
    coverage_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "coverage_matrix": self.coverage_matrix,
            "gaps": [
                {
                    "bucket": g.bucket, "subtype": g.subtype,
                    "scene_family": g.scene_family, "gap_type": g.gap_type,
                    "gap_owner": g.gap_owner, "severity": g.severity,
                    "current_count": g.current_count, "target_count": g.target_count,
                    "score": round(g.score, 4), "recommendation": g.recommendation,
                }
                for g in self.gaps
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_html(self) -> str:
        """Generate a human-readable HTML report."""
        html = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>GSA Gap Report v0</title>",
            "<style>",
            "body{font-family:system-ui;margin:2em;background:#f8f9fa}",
            "table{border-collapse:collapse;width:100%;margin:1em 0}",
            "th,td{border:1px solid #dee2e6;padding:8px;text-align:left}",
            "th{background:#343a40;color:white}",
            ".critical{background:#f8d7da}.high{background:#fff3cd}",
            ".medium{background:#d1ecf1}.low{background:#d4edda}",
            "h1{color:#212529}h2{color:#495057;border-bottom:2px solid #dee2e6}",
            "</style></head><body>",
            "<h1>GSA Data Flywheel v0 — Gap Report</h1>",
        ]

        # Summary
        html.append("<h2>Summary</h2><table><tr><th>Metric</th><th>Value</th></tr>")
        for k, v in self.summary.items():
            if isinstance(v, (dict, list)):
                continue
            html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        html.append("</table>")

        robot_summary = self.summary.get("A_robot_state")
        if isinstance(robot_summary, dict):
            html.extend(_robot_state_summary_html(robot_summary))

        # Coverage matrix
        html.append("<h2>Coverage Matrix</h2><table><tr><th>Bucket/Subtype</th>")
        scenes = sorted({scene for counts in self.coverage_matrix.values() for scene in counts})
        for sf in scenes:
            html.append(f"<th>{sf}</th>")
        html.append("<th>Total</th></tr>")

        for key, counts in self.coverage_matrix.items():
            total = sum(counts.values())
            html.append(f"<tr><td>{key}</td>")
            for sf in scenes:
                html.append(f"<td>{counts.get(sf, 0)}</td>")
            html.append(f"<td><b>{total}</b></td></tr>")
        html.append("</table>")

        # Gaps
        html.append("<h2>Identified Gaps</h2><table>")
        html.append("<tr><th>Bucket</th><th>Subtype</th><th>Scene</th><th>Type</th>"
                    "<th>Owner</th><th>Severity</th><th>Count</th><th>Recommendation</th></tr>")
        for g in sorted(self.gaps, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.severity, 4)):
            html.append(f"<tr class='{g.severity}'><td>{g.bucket}</td><td>{g.subtype}</td>"
                        f"<td>{g.scene_family}</td><td>{g.gap_type}</td><td>{g.gap_owner}</td>"
                        f"<td>{g.severity}</td><td>{g.current_count}/{g.target_count}</td>"
                        f"<td>{g.recommendation}</td></tr>")
        html.append("</table></body></html>")

        return "\n".join(html)


def _robot_state_summary(labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    robot_labels = [l for l in labels if l.get("bucket") == Bucket.A_ROBOT_STATE.value]
    verifier_status = Counter(l.get("verifier_status", "?") for l in robot_labels)
    auto = verifier_status.get("auto_verifiable", 0)
    total = len(robot_labels)
    return {
        "candidate_count": total,
        "stable_training_label_count": sum(1 for l in robot_labels if l.get("stable_training_label")),
        "auto_verifiable": auto,
        "review_required": verifier_status.get("review_required", 0),
        "verifiable_rate": round(auto / total, 4) if total else 0.0,
        "subtype_counts": dict(Counter(l.get("state_type") or l.get("subtype") or "?" for l in robot_labels)),
        "source_counts": dict(Counter(l.get("source_dataset", "?") for l in robot_labels)),
    }


def _robot_state_summary_html(summary: Dict[str, Any]) -> List[str]:
    subtype_text = ", ".join(
        f"{k}: {v}" for k, v in sorted((summary.get("subtype_counts") or {}).items())
    )
    source_text = ", ".join(
        f"{k}: {v}" for k, v in sorted((summary.get("source_counts") or {}).items())
    )
    return [
        "<h2>A_robot_state</h2><table>",
        "<tr><th>Metric</th><th>Value</th></tr>",
        f"<tr><td>candidate_count</td><td>{summary.get('candidate_count', 0)}</td></tr>",
        f"<tr><td>stable_training_label_count</td><td>{summary.get('stable_training_label_count', 0)}</td></tr>",
        f"<tr><td>auto_verifiable</td><td>{summary.get('auto_verifiable', 0)}</td></tr>",
        f"<tr><td>review_required</td><td>{summary.get('review_required', 0)}</td></tr>",
        f"<tr><td>verifiable_rate</td><td>{summary.get('verifiable_rate', 0)}</td></tr>",
        f"<tr><td>subtypes</td><td>{subtype_text}</td></tr>",
        f"<tr><td>sources</td><td>{source_text}</td></tr>",
        "</table>",
    ]


class GSAGapReporter:
    """Generates gap reports from eval results and labels."""

    def __init__(self, target_per_subtype: int = 100):
        self.target = target_per_subtype

    def generate(self, labels: List[Dict[str, Any]],
                 eval_results: Optional[EvalResults] = None) -> GapReport:
        """Generate a gap report from labels and optional eval results."""
        report = GapReport()

        # Build coverage matrix
        coverage = defaultdict(lambda: defaultdict(int))
        for label in labels:
            bucket = label.get("bucket", "unknown")
            subtype = label.get("subtype", "unknown")
            scene = label.get("scene_family", "unknown")
            key = f"{bucket}/{subtype}"
            coverage[key][scene] += 1

        report.coverage_matrix = {k: dict(v) for k, v in coverage.items()}

        # Identify gaps only for buckets/scenes present in this run.
        present_buckets = {l.get("bucket") for l in labels}
        present_scenes = {l.get("scene_family") for l in labels if l.get("scene_family")}
        if not present_scenes:
            present_scenes = {"unknown"}
        for bucket in Bucket:
            if bucket.value not in present_buckets:
                continue
            subtypes = BUCKET_SUBTYPES.get(bucket, [])
            for subtype in subtypes:
                key = f"{bucket.value}/{subtype}"
                counts = coverage.get(key, {})
                for scene in sorted(present_scenes):
                    scene_count = counts.get(scene, 0)
                    if scene_count < self.target:
                        gap = GapEntry(
                            bucket=bucket.value,
                            subtype=subtype,
                            scene_family=scene,
                            gap_type="missing_data" if scene_count == 0 else "low_coverage",
                            gap_owner=self._assign_owner(bucket.value, subtype, scene_count),
                            severity=self._severity(scene_count, self.target),
                            current_count=scene_count,
                            target_count=self.target,
                            recommendation=self._recommend(bucket.value, subtype, scene_count),
                        )
                        report.gaps.append(gap)

        # Summary
        report.summary = {
            "total_labels": len(labels),
            "total_buckets": len(set(l.get("bucket") for l in labels)),
            "total_subtypes": len(coverage),
            "total_gaps": len(report.gaps),
            "critical_gaps": sum(1 for g in report.gaps if g.severity == "critical"),
            "high_gaps": sum(1 for g in report.gaps if g.severity == "high"),
            "A_robot_state": _robot_state_summary(labels),
        }

        if eval_results:
            report.summary["overall_verifiable_rate"] = round(eval_results.overall_verifiable_rate, 4)
            report.summary["review_queue_size"] = len(eval_results.review_queue)

        return report

    def _assign_owner(self, bucket: str, subtype: str, count: int) -> str:
        if count == 0:
            if bucket in (Bucket.G.value, Bucket.S.value):
                return GapOwner.REAL_LOG
            return GapOwner.PUBLIC_ADAPTER
        if count < 10:
            return GapOwner.MANUAL_REVIEW
        return GapOwner.MODEL_VERIFIER

    def _severity(self, current: int, target: int) -> str:
        ratio = current / target if target > 0 else 0
        if ratio == 0:
            return "critical"
        elif ratio < 0.25:
            return "high"
        elif ratio < 0.5:
            return "medium"
        return "low"

    def _recommend(self, bucket: str, subtype: str, count: int) -> str:
        if count == 0:
            return f"Generate {subtype} data from business logs or VLM"
        elif count < 50:
            return f"Increase {subtype} coverage with targeted generation"
        return f"Verify quality of existing {subtype} labels"
