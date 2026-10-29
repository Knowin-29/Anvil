#!/usr/bin/env python3
"""Cascaded generation prompts. Each downstream prompt is conditioned on the
verified upstream context (G -> S -> A), which is the mechanism that removes
cross-layer hallucination vs flat generation. Thousandth-scale coords, JSON-only.
"""
from __future__ import annotations

_JSON_ONLY = ("Output ONLY valid JSON, no prose and no markdown fence. "
              "All coordinates are thousandth-scale integers in [0,1000), origin top-left.")

# ---- G: grounding ----
G_GROUNDING = (
    "Detect ONE clearly visible, manipulable object and name it with a specific "
    "concrete noun (e.g. 'mug', 'drawer handle', not 'object'). "
    'Reply as a JSON list with one element: [{"label": "<noun>", '
    '"bbox_2d": [x1,y1,x2,y2], "confidence": <0-1>}]. ' + _JSON_ONLY)

# ---- L1 cross-check (reverse verification of a proposed bbox) ----
def cross_check(object_class: str, bbox) -> str:
    return (f'I claim region {list(bbox)} (thousandth scale) contains a "{object_class}". '
            'Verify by looking. Keys: "match" (true/false), "actual_object" (noun), '
            '"confidence" (0-1). ' + _JSON_ONLY)


def cross_check_set(objects: list) -> str:
    claims = [
        {"index": i, "object_class": obj["object_class"], "bbox": obj["bbox"]}
        for i, obj in enumerate(objects)
    ]
    return (
        f"Verify each proposed grounded object against the image: {claims}. "
        "Return one entry per proposal and preserve its index. A valid proposal must "
        "be one distinct physical entity in the same 3-D scene. Mark match=false if "
        "the box or label is wrong, if it is only a face or sub-part of another "
        "proposal, if it is a background fixture, or if it appears in a different "
        "panel of a collage/comparison figure. Format: "
        '[{"index":0,"match":true,"actual_object":"noun","confidence":<your 0-1 confidence, not 0>}]. '
        + _JSON_ONLY
    )


# ---- S: spatial (conditioned on grounded objects) ----
def s_spatial(objects: list) -> str:
    obj_str = ", ".join(f'{o["object_class"]}@{o["bbox"]}' for o in objects[:6])
    return (f"Grounded objects: {obj_str}. State ONE spatial relation between two of them. "
            'Keys: "subject" (noun), "predicate" (one of: left-of,right-of,above,below,'
            'in-front-of,behind,on-top-of,under,inside,next-to), "object" (noun). '
            + _JSON_ONLY)

# ---- A: identity + attributes (conditioned on a grounded object) ----
def a_identity_attr(obj: dict) -> str:
    return (f'For the {obj["object_class"]} at {obj["bbox"]}, describe attributes. '
            'Keys: "object_class", "bbox", "attributes" (dict with color, material, '
            'shape, size). ' + _JSON_ONLY)

# ---- A: object state ----
def a_obj_state(obj: dict) -> str:
    return (f'For the {obj["object_class"]} at {obj["bbox"]}, report its physical state. '
            'Keys: "object_class", "bbox", "state_type" (pose/open_close/on_off/'
            'full_empty/occupancy), "state_value". ' + _JSON_ONLY)

# ---- A: action precondition ----
def a_action_precondition(obj: dict) -> str:
    return (f'For the {obj["object_class"]} at {obj["bbox"]}, assess a pick action now. '
            'Keys: "object_class", "bbox", "action_type" ("pick"), "can_do_now" '
            '(true/false), "blocked_by" (list of nouns), "failure_reason" (short). '
            + _JSON_ONLY)

# ---- A: action interface (grasp point / jaw axis) ----
def a_action_interface(obj: dict) -> str:
    return (f'For the {obj["object_class"]} at {obj["bbox"]}, give a 2D manipulation '
            'interface. Keys: "object_class", "bbox", "action_type" (open/pick/push), '
            '"grasp_point_2d" ([x,y]), "jaw_axis_2d" ([dx,dy]). ' + _JSON_ONLY)

PROMPT_FN = {
    "S_spatial": s_spatial,
    "A_identity_attr": a_identity_attr,
    "A_obj_state": a_obj_state,
    "A_action_precondition": a_action_precondition,
    "A_action_interface": a_action_interface,
}
