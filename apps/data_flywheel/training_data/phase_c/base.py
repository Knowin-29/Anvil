from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from apps.data_flywheel.types import TrainingSample


@runtime_checkable
class Validator(Protocol):
    name: str
    severity: str

    def check(self, sample: TrainingSample, context: Dict[str, Any]) -> "ValidationResult":
        ...


@dataclass
class ValidationResult:
    validator_name: str
    passed: bool
    severity: str
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    sample_id: str
    overall_passed: bool
    results: List[ValidationResult]

    def has_critical_failure(self) -> bool:
        return any(result.severity == "critical" and not result.passed for result in self.results)

    def failed_validators(self) -> List[str]:
        return [result.validator_name for result in self.results if not result.passed]


@dataclass
class RejectedSampleEntry:
    sample_id: str
    image_id: str
    image_path: str
    reasons: List[str]
    critical_fails: List[str]
    warning_fails: List[str]
    report: ValidationReport
    instruction: str
    actions_summary: List[Dict[str, Any]]
    thinking_preview: str
