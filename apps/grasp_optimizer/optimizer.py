from typing import Any, Dict, List, Optional

from .robot_bridge import RobotBridge


def _collect_first_boxes(predictions: Dict[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for category, detections in predictions.items():
        for det in detections:
            if det.get("type") == "box":
                out[category] = det["coords"]
                break
    return out


def build_grasp_decision(
    target_object: str,
    predictions: Dict[str, Any],
    candidate_objects: List[str],
    robot_exec_mode: str = "mock",
) -> Dict[str, Any]:
    boxes = _collect_first_boxes(predictions)
    target_box: Optional[List[float]] = boxes.get(target_object)
    blockers = [
        name for name in candidate_objects
        if name in boxes and name != target_object
    ]

    if target_box is None:
        strategy = "target_not_found"
    elif blockers:
        strategy = "clear_then_pick"
    else:
        strategy = "pick_directly"

    decision: Dict[str, Any] = {
        "target_object": target_object,
        "target_box": target_box,
        "blockers": blockers[:3],
        "grasp_candidates": [target_object] if target_box else [],
        "push_candidates": blockers[:3],
        "strategy": strategy,
    }

    robot_resp = RobotBridge(mode=robot_exec_mode).execute(decision)
    return {"decision": decision, "robot": robot_resp}
