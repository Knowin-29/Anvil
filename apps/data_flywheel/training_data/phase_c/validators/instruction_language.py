from __future__ import annotations

from typing import Any, Dict

from ..base import ValidationResult


class InstructionLanguageValidator:
    name = "instruction_language"
    severity = "warning"

    def check(self, sample, context: Dict[str, Any]) -> ValidationResult:
        has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in sample.instruction)
        expected = sample.instruction_language
        actual = "zh" if has_chinese else "en"

        if expected != actual:
            if expected == "zh" and has_chinese:
                return ValidationResult(
                    validator_name=self.name,
                    passed=True,
                    severity=self.severity,
                )
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=self.severity,
                reason=f"declared={expected}, detected={actual}",
            )

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            severity=self.severity,
        )
