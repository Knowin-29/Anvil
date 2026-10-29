#!/usr/bin/env python3
"""Extended task-form prompts (RynnBrain-style) for forms the open sources lack:
detection (multi-object bbox), mask (polygon), depth ordering, affordance region,
DoF interface, and action-planning CoT.

All coordinates thousandth-scale ints in [0,1000), origin top-left. JSON-only
replies; we use Qwen-VL's native `bbox_2d` / `point_2d` keys so the model stays in
its trained output mode (fighting it produced generic "label" values).
"""
from __future__ import annotations

import json

PROMPT_VERSION = "anvil-flywheel-v3.5"
LABEL_VERSION = "anvil-label-v3.5"

_J = ("Reply with ONLY a JSON array, no prose, no markdown fence. "
      "Image point and box coordinates are integers in [0,1000), origin top-left.")

_JO = "Reply with exactly one JSON object, no prose and no markdown fence."

SCENE_ELIGIBILITY = (
    "Decide whether this image is one coherent physical 3-D scene suitable for "
    "robot manipulation annotation. Reject multi-panel comparisons, grids, "
    "collages, diagrams, masks/edge maps, monochrome derived visualizations, and "
    "images with pre-drawn boxes, arrows, points, text labels, or magnified insets, "
    "Accept a coherent textured RGB camera render from an embodied simulator, but "
    "reject standalone game/voxel asset renders with no plausible manipulation "
    "scene. Require at least two visibly distinct physical entities. Return "
    '{"eligible":true,"reason_code":"single_scene","confidence":<your 0-1 confidence, not 0>}. '
    "When rejecting, reason_code must be one of multi_panel, derived_visualization, "
    "preannotated_overlay, non_photographic, insufficient_entities, or "
    "ambiguous_scene. " + _JO)

DETECTION = (
    "This image has already been identified as one coherent 3-D scene. Detect the "
    "distinct MANIPULABLE entities in it (2 to 6). Rules: "
    "(1) each entity must be a separate physical thing a parallel-jaw robot could "
    "grasp or operate; never split one object's visible faces or sub-parts into "
    "separate entities; (2) give each a DIFFERENT specific noun label -- if similar "
    "items exist, distinguish them (e.g. 'red cloth', 'blue towel'); "
    "(3) do NOT include the robot itself, background surfaces, or large fixtures "
    "(wall, floor, table, counter, shelf, appliance body); "
    "(4) all entities must belong to this same physical scene, never separate panels; "
    "(5) no box may span nearly the whole image or form a full-width/full-height band. "
    'Format: [{"label":"<noun>","bbox_2d":[x1,y1,x2,y2]}, ...]. ' + _J)

def visible_point(obj: dict) -> str:
    return (f'Locate one clearly visible material point on the {obj["object_class"]} '
            f'within its bounding box {obj["bbox"]}. Inspect the actual object surface; '
            "a box center can fall in a hole, between chair legs, or on background. "
            "Do not use the box center unless it visibly lies on the named object. "
            "Avoid boundaries, empty openings, occluders, and the supporting floor. "
            f'Keep the exact label "{obj["object_class"]}". Return [] if no unambiguous '
            'visible point exists. Format: [{"label":"<exact label>","point_2d":[x,y]}]. ' + _J)


def verify_visible_point(obj: dict, candidate: list) -> str:
    return (f'Check whether the proposed point is on visible material of the '
            f'{obj["object_class"]} at {obj["bbox"]}: {json.dumps(candidate)}. '
            "A point inside its box is insufficient: reject background, floor, holes, "
            "empty space between chair legs, another object, or ambiguous surface ownership. "
            'Return {"valid":true,"confidence":<0-1>,"reason_code":"supported"} '
            'only when visibly supported. Otherwise valid=false and reason_code is '
            'background, opening, wrong_entity, or ambiguous_surface. ' + _JO)


def mask_polygon(obj: dict) -> str:
    """Segment the SAME object the cascade already grounded, so mask is aligned with
    bbox / dof / affordance / action_cot for that image (single-version consistency)."""
    return (f'Segment the {obj["object_class"]} located at bounding box {obj["bbox"]} '
            "(thousandth scale). Trace ONLY that object's silhouette. Requirements: "
            "(1) 8 to 16 vertices, each a nested [x,y] pair, NOT a flat list; "
            "(2) follow the actual curved/angled outline -- do NOT return the 4 corners "
            "of the bounding box; (3) every vertex must lie inside or on that bounding "
            "box. Cover the complete visible silhouette, including thin limbs; "
            "do not fill large openings with background or connect disconnected parts "
            "through empty space. Return [] if one polygon cannot represent this object "
            "without substantial background inclusion or omission. "
            f'Use the same label "{obj["object_class"]}". Format: '
            '[{"label":"<noun>","polygon":[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],'
            '[x6,y6],[x7,y7],[x8,y8]]}]. ' + _J)


# Back-compat alias: the un-anchored variant is deprecated (produced masks that did
# not correspond to the object used by the other forms).
MASK_POLYGON = mask_polygon({"object_class": "object", "bbox": [0, 0, 1000, 1000]})


def depth_order(objects: list) -> str:
    entities = [
        {"label": o["object_class"], "bbox_2d": o["bbox"]}
        for o in objects[:6]
    ]
    return (
        "The following JSON objects identify distinct entities in one physical scene: "
        f"{json.dumps(entities, ensure_ascii=False)}. Order them by physical camera "
        "z-depth, nearest to farthest. Use direct occlusion/contact layering or a clear "
        "support-plane perspective cue. Image height, apparent size, and left-to-right "
        "position alone are not depth evidence. If any adjacent pair is coplanar, "
        "ambiguous, split across panels, or otherwise lacks a visible ordering cue, "
        "return []. Otherwise return every entity exactly once with ranks 1..N and "
        "strictly increasing relative_depth values in [0,1] (0=nearest). The output "
        "label must exactly equal the input `label` string: never append bbox values or "
        "other text. Format: "
        '[{"label":"<exact input label>","rank":1,"relative_depth":0.0}, ...]. '
        + _J
    )


def verify_depth_order(objects: list, candidate: list) -> str:
    claims = [
        {"label": item["object_class"], "bbox": item["bbox"]}
        for item in objects
    ]
    return (
        "Independently verify this proposed nearest-to-farthest camera-depth order "
        f"against the image. Entities: {json.dumps(claims, ensure_ascii=False)}. "
        f"Proposed order: {json.dumps(candidate, ensure_ascii=False)}. Reject if any "
        "entity is a duplicate face/part, belongs to another panel, is not physically "
        "orderable, if any adjacent pair lacks a direct visible depth cue, if labels "
        "contain coordinates or differ from the entity labels, if relative depths are "
        "not strictly increasing, or if one pair is visibly reversed. Return "
        '{"valid":true,"confidence":<your 0-1 confidence, not 0>,"reason_code":"supported"}. '
        "For false, reason_code must be duplicate_entity, multi_panel, ambiguous_depth, "
        "reversed_pair, or unsupported_entity. " + _JO)


def _spatial_hint(spatial: dict | None) -> str:
    if not spatial:
        return ""
    return f" Verified spatial context: {spatial}."


def affordance_region(obj: dict, spatial: dict | None = None) -> str:
    return (f'The {obj["object_class"]} occupies {obj["bbox"]}. Mark one visible, '
            "physically contactable SUB-PART a parallel-jaw robot should contact to "
            "manipulate it (e.g. handle, rim, edge, side wall, graspable surface). "
            "The returned bbox_2d must be strictly SMALLER than the object bbox and lie "
            "inside it -- do not repeat the object bbox. The contact point must lie on "
            "visible material, never in the empty interior of a bowl, pot, cup, ring, "
            "or handle. The affordance value must be an action verb such as grasp, "
            "pinch, pull, turn, lift, or open, not a part noun. Return [] if no reliable "
            "contact part is visible."
            + _spatial_hint(spatial) + " Format: "
            '[{"label":"<part noun>","affordance":"<verb>","bbox_2d":[x1,y1,x2,y2],'
            '"contact_point_2d":[x,y]}]. ' + _J)


def dof_interface(obj: dict, spatial: dict | None = None,
                  affordance: list | None = None) -> str:
    interface = (
        " The previously verified contact proposal is "
        + json.dumps(affordance, ensure_ascii=False) + "."
        if affordance else ""
    )
    return (f'For the {obj["object_class"]} at {obj["bbox"]}, specify a parallel-jaw '
            "manipulation grasp that uses the visible contact sub-part. First look at "
            "how the object and contact region are oriented, then set "
            "jaw_axis_2d PERPENDICULAR to the object's visible long axis at the grasp "
            "point (that is where parallel jaws close). Give it as a unit vector with "
            "two decimals. Horizontal, vertical and diagonal directions are valid when "
            "supported by the visible contact geometry; do not tilt an axis to avoid a "
            "particular angle. Set wrist_deg consistent with that axis "
            "(wrist_deg = atan2(dy,dx) in degrees, represented in [-90,90]). The "
            "grasp point must lie on visible material inside the verified contact "
            "region; for open containers use a visible rim or handle, never the empty "
            "center. Return [] if a secure parallel-jaw contact is not visible. "
            f'The output label must be exactly "{obj["object_class"]}", not the contact-part label.'
            + interface
            + _spatial_hint(spatial) + " Format: "
            '[{"label":"<noun>","grasp_point_2d":[x,y],"jaw_axis_2d":[dx,dy],'
            '"approach":"top|side|front","wrist_deg":<-90..90>}]. ' + _J)


def action_cot(obj: dict, spatial: dict | None = None,
               affordance: list | None = None, dof: list | None = None) -> str:
    dependencies = {
        "contact": affordance or [],
        "grasp": dof or [],
    }
    return (f'A robot arm must manipulate the {obj["object_class"]} at {obj["bbox"]}. '
            "Choose the physically appropriate goal from the verified contact action: "
            "a loose object may be picked up, while an attached handle/door/seat must "
            "be pulled, opened, or rotated without claiming it is detached. Write a "
            "short executable plan grounded only in visible geometry. Preconditions "
            "and failure modes must refer to visible contact/support geometry (for "
            "example an unobstructed rim or a slipping grasp), never robot calibration, "
            "robot readiness, hidden state, nearby obstacles, or collision. Preconditions "
            "should mention only a visible contact part or visible support; failure modes "
            "should mention only contact misalignment, occlusion, or grasp slippage. Use the verified "
            "grasp point exactly as target_point_2d. Do not invent nearby obstacles, "
            "hidden handles, object weights, robot state, or metric lift distances. "
            f"Verified manipulation dependencies: {json.dumps(dependencies, ensure_ascii=False)}."
            + _spatial_hint(spatial) + " Format: "
            '[{"goal":"<short>","steps":["step 1 ...","step 2 ...","step 3 ..."],'
            '"preconditions":["..."],"failure_modes":["..."],'
            '"target_point_2d":[x,y]}]. ' + _J)


def verify_manipulation(obj: dict, affordance: list, dof: list,
                        action: list) -> str:
    proposal = {
        "object": obj,
        "affordance": affordance,
        "dof": dof,
        "action": action,
    }
    return (
        "Independently verify this manipulation proposal against the image: "
        f"{json.dumps(proposal, ensure_ascii=False)}. It is valid only if the named "
        "contact part is visible, the grasp point lies on material (not an opening), "
        "the jaw orientation is plausible, the goal respects whether the object is "
        "loose or attached, and every obstacle/precondition is visually supported. "
        "Reject generic/template 45-degree jaw axes when they are not supported by "
        "the visible contact geometry, and reject invented robot state or obstacles. "
        "Return "
        '{"valid":true,"confidence":<your 0-1 confidence, not 0>,"reason_code":"supported"}. '
        "For false, reason_code must be invalid_contact, invalid_grasp, attached_object, "
        "unsupported_assumption, inconsistent_target, or ambiguous_action. " + _JO)


ACTION_PROMPT_FN = {
    "affordance": affordance_region,
    "dof": dof_interface,
    "action_cot": action_cot,
}
