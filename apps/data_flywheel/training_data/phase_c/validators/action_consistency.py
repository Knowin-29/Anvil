from __future__ import annotations

from typing import Any, Dict

from ..base import ValidationResult


class ActionConsistencyValidator:
    name = "action_consistency"
    severity = "critical"

    def check(self, sample, context: Dict[str, Any]) -> ValidationResult:
        errors = []
        node_ids = {obj.id for obj in sample.scene_objects}

        for i, action in enumerate(sample.actions):
            if action.target not in node_ids:
                errors.append(f"action[{i}].target '{action.target}' not in scene_objects")

            if action.type not in ("pick", "push", "place"):
                errors.append(f"action[{i}].type '{action.type}' invalid")
                continue

            if action.type in ("pick", "push"):
                if action.grasp_point_pixel is None:
                    errors.append(f"action[{i}] {action.type} missing grasp_point_pixel")
                if action.grasp_point_quantized is None:
                    errors.append(f"action[{i}] {action.type} missing grasp_point_quantized")
                if action.destination is not None:
                    errors.append(f"action[{i}] {action.type} should not have destination")
                if action.place_point_pixel is not None:
                    errors.append(f"action[{i}] {action.type} should not have place_point_pixel")
                if action.place_point_quantized is not None:
                    errors.append(f"action[{i}] {action.type} should not have place_point_quantized")
                if action.destination_type is not None:
                    errors.append(f"action[{i}] {action.type} should not have destination_type")
                if action.destination_anchor_ids:
                    errors.append(f"action[{i}] {action.type} should not have destination_anchor_ids")
                if action.placement_reason is not None:
                    errors.append(f"action[{i}] {action.type} should not have placement_reason")
            elif action.type == "place":
                if action.destination is None:
                    errors.append(f"action[{i}] place missing destination")
                elif action.destination not in node_ids:
                    errors.append(f"action[{i}] place destination '{action.destination}' not in scene_objects")
                if action.place_point_pixel is None:
                    errors.append(f"action[{i}] place missing place_point_pixel")
                if action.place_point_quantized is None:
                    errors.append(f"action[{i}] place missing place_point_quantized")
                if action.destination_type is None:
                    errors.append(f"action[{i}] place missing destination_type")
                if not action.destination_anchor_ids:
                    errors.append(f"action[{i}] place missing destination_anchor_ids")
                if action.placement_reason is None:
                    errors.append(f"action[{i}] place missing placement_reason")
                if action.grasp_point_pixel is not None:
                    errors.append(f"action[{i}] place should not have grasp_point_pixel")
                if action.grasp_point_quantized is not None:
                    errors.append(f"action[{i}] place should not have grasp_point_quantized")

        plan_order_steps = [step for step in sample.thinking.steps if step.perception_type == "PlanOrder"]
        if plan_order_steps:
            expected_sequence = plan_order_steps[0].output
            actual_sequence = [f"{action.type}:{action.target}" for action in sample.actions]
            if isinstance(expected_sequence, list):
                if actual_sequence != expected_sequence:
                    errors.append(
                        f"PlanOrder mismatch: thinking={expected_sequence}, actions={actual_sequence}"
                    )
            else:
                errors.append("PlanOrder output must be a list")

        return ValidationResult(
            validator_name=self.name,
            passed=not errors,
            severity=self.severity,
            reason="; ".join(errors) if errors else None,
        )
