from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from ..base import ValidationResult


class ReferencingUniquenessValidator:
    name = "referencing_uniqueness"
    severity = "warning"

    def check(self, sample, context: Dict[str, Any]) -> ValidationResult:
        errors = []

        ids = [obj.id for obj in sample.scene_objects]
        if len(ids) != len(set(ids)):
            errors.append("duplicate scene_object ids")

        names = [obj.hierarchical_name for obj in sample.scene_objects if obj.hierarchical_name]
        name_counts = Counter(names)
        duplicates = sorted(name for name, count in name_counts.items() if count > 1)
        if duplicates:
            errors.append(f"duplicate hierarchical_name: {duplicates}")

        node_ids = set(ids)
        for i, action in enumerate(sample.actions):
            if not action.destination_anchor_ids:
                continue
            if len(action.destination_anchor_ids) != len(set(action.destination_anchor_ids)):
                errors.append(f"action[{i}] duplicate destination_anchor_ids")
            missing = sorted(anchor_id for anchor_id in action.destination_anchor_ids if anchor_id not in node_ids)
            if missing:
                errors.append(f"action[{i}] destination_anchor_ids missing from scene_objects: {missing}")

        return ValidationResult(
            validator_name=self.name,
            passed=not errors,
            severity=self.severity,
            reason="; ".join(errors) if errors else None,
        )
