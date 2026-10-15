from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from apps.data_flywheel.types import TrainingSample

from .base import RejectedSampleEntry, ValidationReport, ValidationResult, Validator


def validate_samples(
    samples: Iterable[TrainingSample],
    validators: Optional[List[Validator]] = None,
) -> Tuple[List[TrainingSample], List[RejectedSampleEntry]]:
    if validators is None:
        validators = default_validators()

    passed: List[TrainingSample] = []
    rejected: List[RejectedSampleEntry] = []

    for sample in samples:
        context: Dict[str, Any] = {}
        results: List[ValidationResult] = []

        for validator in validators:
            try:
                result = validator.check(sample, context)
            except Exception as exc:  # pragma: no cover - defensive guard
                result = ValidationResult(
                    validator_name=validator.name,
                    passed=False,
                    severity=validator.severity,
                    reason=f"validator crashed: {exc}",
                    details={"exception_type": exc.__class__.__name__},
                )
            results.append(result)

        report = ValidationReport(
            sample_id=sample.sample_id,
            overall_passed=not any(
                result.severity == "critical" and not result.passed for result in results
            ),
            results=results,
        )

        if report.overall_passed:
            passed.append(sample)
        else:
            rejected.append(_build_rejected_entry(sample, report))

    return passed, rejected


def default_validators() -> List[Validator]:
    from .validators import (
        ActionConsistencyValidator,
        CoordinateRangeValidator,
        ImageExistsValidator,
        InstructionLanguageValidator,
        OntologyConsistencyValidator,
        PointInMaskValidator,
        ReferencingUniquenessValidator,
        ThinkingFormatValidator,
    )

    return [
        PointInMaskValidator(),
        CoordinateRangeValidator(),
        ThinkingFormatValidator(),
        ActionConsistencyValidator(),
        OntologyConsistencyValidator(),
        ReferencingUniquenessValidator(),
        ImageExistsValidator(),
        InstructionLanguageValidator(),
    ]


def _thinking_preview(sample: TrainingSample, max_lines: int = 3) -> str:
    lines = [line for line in sample.thinking.to_text().splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


def _action_summary(sample: TrainingSample) -> List[Dict[str, Any]]:
    return [
        {
            "type": action.type,
            "target": action.target,
            "reason": action.reason or "",
        }
        for action in sample.actions
    ]


def _build_rejected_entry(sample: TrainingSample, report: ValidationReport) -> RejectedSampleEntry:
    failed_results = [result for result in report.results if not result.passed]
    critical_fails = [result.validator_name for result in failed_results if result.severity == "critical"]
    warning_fails = [result.validator_name for result in failed_results if result.severity == "warning"]
    reasons = [result.reason or result.validator_name for result in failed_results]

    return RejectedSampleEntry(
        sample_id=sample.sample_id,
        image_id=sample.image_id,
        image_path=sample.image_path,
        reasons=reasons,
        critical_fails=critical_fails,
        warning_fails=warning_fails,
        report=report,
        instruction=sample.instruction,
        actions_summary=_action_summary(sample),
        thinking_preview=_thinking_preview(sample),
    )


def recycle_rejected_to_review(
    rejected: List[RejectedSampleEntry],
    review_pool_path: str = "data/output/review/rejected_samples.jsonl",
) -> int:
    os.makedirs(os.path.dirname(review_pool_path) or ".", exist_ok=True)
    rejected_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    count = 0
    with open(review_pool_path, "a", encoding="utf-8") as handle:
        for entry in rejected:
            row = {
                "sample_id": entry.sample_id,
                "image_id": entry.image_id,
                "image_path": entry.image_path,
                "instruction": entry.instruction,
                "reasons": entry.reasons,
                "critical_fails": entry.critical_fails,
                "warning_fails": entry.warning_fails,
                "actions_summary": entry.actions_summary,
                "thinking_preview": entry.thinking_preview,
                "rejected_at": rejected_at,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
