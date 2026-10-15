from __future__ import annotations

from typing import Any, Dict

from ..base import ValidationResult


class OntologyConsistencyValidator:
    name = "ontology_consistency"
    severity = "critical"

    def check(self, sample, context: Dict[str, Any]) -> ValidationResult:
        errors = []
        nodes_by_id = {obj.id: obj for obj in sample.scene_objects}

        for i, action in enumerate(sample.actions):
            node = nodes_by_id.get(action.target)
            if node is None:
                continue

            if "push_only" in node.extra_tags and action.type != "push":
                errors.append(
                    f"action[{i}] {action.type} on push_only node '{node.id}' (extra_tags={node.extra_tags})"
                )

            if action.type == "pick" and node.operation not in ("pick", None):
                if node.operation == "push":
                    errors.append(f"action[{i}] pick on push-operation node '{node.id}'")
                if node.operation == "non_operable":
                    errors.append(f"action[{i}] pick on non_operable node '{node.id}'")

            if action.type == "place" and node.operation not in ("pick", None):
                if node.operation == "push":
                    errors.append(f"action[{i}] place on push-operation node '{node.id}'")
                if node.operation == "non_operable":
                    errors.append(f"action[{i}] place on non_operable node '{node.id}'")

            if action.type == "push" and node.operation == "non_operable":
                errors.append(f"action[{i}] push on non_operable node '{node.id}'")

        return ValidationResult(
            validator_name=self.name,
            passed=not errors,
            severity=self.severity,
            reason="; ".join(errors) if errors else None,
        )
