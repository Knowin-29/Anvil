from __future__ import annotations

import os
from typing import Any, Dict

from ..base import ValidationResult


class ImageExistsValidator:
    name = "image_exists"
    severity = "warning"

    def check(self, sample, context: Dict[str, Any]) -> ValidationResult:
        if not os.path.exists(sample.image_path):
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=self.severity,
                reason=f"image_path does not exist: {sample.image_path}",
            )
        return ValidationResult(
            validator_name=self.name,
            passed=True,
            severity=self.severity,
        )
