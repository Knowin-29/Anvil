from __future__ import annotations

from typing import List

from apps.data_flywheel.types import Action, ThinkingTrace


def format_rex_omni_text(
    instruction: str,
    thinking: ThinkingTrace,
    actions: List[Action],
) -> str:
    thinking_text = thinking.to_text()
    action_lines = "\n".join(
        _format_action_line(action, index + 1) for index, action in enumerate(actions)
    )
    return (
        f"<|user|>{instruction}<|end|>\n"
        f"<|thinking_start|>\n{thinking_text}\n<|thinking_end|>\n"
        f"<|action_start|>\n{action_lines}\n<|action_end|>\n"
        f"<|end|>"
    )


def _format_action_line(action: Action, index: int) -> str:
    if action.type in {"pick", "push"}:
        if action.grasp_point_quantized is None:
            raise ValueError(f"Missing grasp_point_quantized for action type: {action.type}")
        qx, qy = action.grasp_point_quantized
        return (
            f"Action {index}: {action.type} "
            f"<|object_ref_start|>{action.target}<|object_ref_end|>"
            f"<|box_start|><{qx}><{qy}><|box_end|>"
        )
    if action.type == "place":
        if action.destination is None or action.place_point_quantized is None:
            raise ValueError("Missing destination or place_point_quantized for place action")
        qx, qy = action.place_point_quantized
        return (
            f"Action {index}: place "
            f"<|object_ref_start|>{action.target}<|object_ref_end|> into "
            f"<|object_ref_start|>{action.destination}<|object_ref_end|>"
            f"<|box_start|><{qx}><{qy}><|box_end|>"
        )
    raise ValueError(f"Unknown action type: {action.type}")
