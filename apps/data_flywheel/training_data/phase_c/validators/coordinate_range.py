from __future__ import annotations

from typing import Any, Dict

from ..base import ValidationResult


class CoordinateRangeValidator:
    name = "coordinate_range"
    severity = "critical"

    def check(self, sample, context: Dict[str, Any]) -> ValidationResult:
        errors = []

        for obj in sample.scene_objects:
            if obj.bbox_quantized is None:
                errors.append(f"{obj.id}.bbox_quantized missing")
                continue

            for coord in obj.bbox_quantized:
                if not (0 <= coord <= 999):
                    errors.append(f"{obj.id}.bbox_quantized has out-of-range: {coord}")
            if obj.bbox_quantized[0] >= obj.bbox_quantized[2]:
                errors.append(f"{obj.id}.bbox_quantized x1 >= x2")
            if obj.bbox_quantized[1] >= obj.bbox_quantized[3]:
                errors.append(f"{obj.id}.bbox_quantized y1 >= y2")
            if obj.grasp_point_quantized is not None:
                for coord in obj.grasp_point_quantized:
                    if not (0 <= coord <= 999):
                        errors.append(f"{obj.id}.grasp_point_quantized out-of-range: {coord}")

        for i, action in enumerate(sample.actions):
            if action.grasp_point_quantized is None:
                pass
            else:
                for coord in action.grasp_point_quantized:
                    if not (0 <= coord <= 999):
                        errors.append(f"action[{i}].grasp_point_quantized out-of-range: {coord}")
            if action.place_point_quantized is None:
                continue
            for coord in action.place_point_quantized:
                if not (0 <= coord <= 999):
                    errors.append(f"action[{i}].place_point_quantized out-of-range: {coord}")

        return ValidationResult(
            validator_name=self.name,
            passed=not errors,
            severity=self.severity,
            reason="; ".join(errors) if errors else None,
        )
