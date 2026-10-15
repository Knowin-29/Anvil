from __future__ import annotations

from typing import Any, Dict

from apps.data_flywheel.types import PERCEPTION_TYPES, REQUIRED_THINKING_TYPES, THINKING_LINE_REGEX
from ..base import ValidationResult


class ThinkingFormatValidator:
    name = "thinking_format"
    severity = "critical"

    def check(self, sample, context: Dict[str, Any]) -> ValidationResult:
        trace_text = sample.thinking.to_text()
        lines = [line for line in trace_text.split("\n") if line.strip()]

        errors = []
        found_types = set()

        for lineno, line in enumerate(lines, 1):
            match = THINKING_LINE_REGEX.match(line)
            if not match:
                errors.append(f"line {lineno} does not match regex: {line}")
                continue

            perception_type = match.group(1)
            if perception_type not in PERCEPTION_TYPES:
                errors.append(f"line {lineno} unknown perception type: {perception_type}")
                continue

            found_types.add(perception_type)

        missing_required = REQUIRED_THINKING_TYPES - found_types
        if missing_required:
            errors.append(f"missing required perception types: {sorted(missing_required)}")

        return ValidationResult(
            validator_name=self.name,
            passed=not errors,
            severity=self.severity,
            reason="; ".join(errors) if errors else None,
        )
