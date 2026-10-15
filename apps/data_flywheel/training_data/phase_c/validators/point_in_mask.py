from __future__ import annotations

from typing import Any, Dict, List

from apps.data_flywheel.perception.mask_io import load_mask_png
from ..base import ValidationResult


class PointInMaskValidator:
    name = "point_in_mask"
    severity = "critical"

    def check(self, sample, context: Dict[str, Any]) -> ValidationResult:
        failures: List[str] = []

        for obj in sample.scene_objects:
            if obj.mask_file is None or obj.grasp_point_quantized is None:
                continue

            w, h = sample.image_size
            qx, qy = obj.grasp_point_quantized
            pixel_x = int(qx * w / 999)
            pixel_y = int(qy * h / 999)

            try:
                mask = load_mask_png(obj.mask_file)
            except Exception as exc:  # pragma: no cover - exercised by focused test
                failures.append(f"{obj.id}: mask file error: {exc}")
                continue

            if mask.size == 0:
                failures.append(f"{obj.id}: mask file error: empty mask")
                continue

            if mask.ndim != 2:
                failures.append(f"{obj.id}: mask file error: expected grayscale mask")
                continue

            pixel_x = max(0, min(mask.shape[1] - 1, pixel_x))
            pixel_y = max(0, min(mask.shape[0] - 1, pixel_y))

            if mask[pixel_y, pixel_x] > 128:
                continue

            failures.append(
                f"{obj.id}: grasp_point ({qx},{qy}) -> pixel ({pixel_x},{pixel_y}) outside mask"
            )

        if failures:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=self.severity,
                reason="; ".join(failures),
                details={"failures": failures},
            )

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            severity=self.severity,
        )
