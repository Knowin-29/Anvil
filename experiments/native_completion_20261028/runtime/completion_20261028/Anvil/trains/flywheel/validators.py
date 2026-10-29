#!/usr/bin/env python3
"""Per-task-form validators. Each returns (ok: bool, reason: str).

These are the hard geometric/structural constraints applied BEFORE the 6-signal
rule scorer. They exist because the generator, while strong, has systematic
failure modes we measured: flat polygons, affordance boxes echoing the object box,
degenerate jaw axes, out-of-range coords, and single-item "detections".
"""
from __future__ import annotations

import hashlib
import math
import re

from trains.flywheel.contracts import groundings, primary_grounding

GENERIC = {"object", "object_class", "point_1", "item", "thing", "label",
           "single most prominent object", "main manipulable object", "unknown"}

AFFORDANCE_VERBS = {
    "grasp", "pinch", "pull", "push", "press", "turn", "rotate", "lift",
    "open", "close", "hold", "slide", "pick", "pick up",
}
FIXED_OBJECT_TERMS = {
    "handle", "knob", "switch", "button", "door", "drawer", "faucet",
    "toilet seat", "lever",
}
UNSUPPORTED_ACTION_TERMS = {
    "calibrated", "calibration", "operational", "ready to grasp",
    "robot state", "hidden", "unseen", "nearby obstacle", "nearby obstacles",
    "obstacle", "obstacles", "collision", "collide", "nearby object",
    "nearby objects", "attached fixture", "attached fixtures", "adhered",
    "entangled", "none visible", "weight", "kilogram",
    "distance", "centimeter", "metre", "meter", "force", "temperature",
}
DEPTH_V4_PROMPT_VERSION = "gsa-depth-v4.1"
DEPTH_V4_LABEL_VERSION = "gsa-label-v4-depth"
DEPTH_V4_PAIR_PROMPT_VERSION = "gsa-depth-v4.2-pairwise"
DEPTH_V4_PAIR_LABEL_VERSION = "gsa-label-v4-depth-pairwise"
DEPTH_V43_PAIR_PROMPT_VERSION = "gsa-depth-v4.3-pairwise"
DEPTH_V43_PAIR_LABEL_VERSION = "gsa-label-v4.3-depth-pairwise"
DEPTH_V4_MIN_CONFIDENCE = 0.85
DEPTH_V4_CUES = {"occlusion", "contact_layering", "support_plane_perspective"}
DEPTH_V4_BANNED_EVIDENCE = {
    "left of", "right of", "higher in the image", "lower in the image",
    "larger", "smaller", "image height", "looks bigger", "looks smaller",
}
DEPTH_V43_CUE_SEMANTICS = {
    "occlusion": ("occludes", "positive_2d_intersection"),
    "contact_layering": ("contacts_or_layers_over", "overlap_or_near_touch"),
    "support_plane_perspective": ("support_plane_nearer", "support_plane_depth"),
}
DEPTH_V43_MAX_NORMALIZED_EDGE_DISTANCE = 0.01
DEPTH_V43_MAX_AXIS_EDGE_DISTANCE = 10.0
DEPTH_V43_MIN_INTERSECTION_OVER_MIN_BOX_AREA = 0.01
DEPTH_V43_FRAME_HEURISTICS = {
    "top edge", "bottom edge", "foreground", "background",
    "bottom edge of the image", "top edge of the image",
    "bottom edge of the frame", "top edge of the frame",
    "bottom of the image", "top of the image",
    "bottom of the frame", "top of the frame",
    "lower in the frame", "higher in the frame",
    "lower in frame", "higher in frame",
    "image bottom", "image top", "frame bottom", "frame top",
}


def _int_pair(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, (int, float)) for x in v))


def _bbox_ok(b) -> bool:
    return (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) for v in b)
            and 0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000)


def _area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1]) / 1e6


def _inside(inner, outer, tol: int = 20) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _first_dict(value):
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return value if isinstance(value, dict) else {}


def _spatial(ctx) -> dict:
    value = ctx.get("spatial") if isinstance(ctx, dict) else None
    return value if isinstance(value, dict) else {}


def v_bbox(items, ctx=None):
    it = items[0] if isinstance(items, list) and items else items
    if not isinstance(it, dict):
        return False, "not_dict"
    b = it.get("bbox_2d") or it.get("bbox")
    if not _bbox_ok(b):
        return False, "bad_bbox"
    if _area(b) > 0.7:
        return False, "bbox_too_large"
    if _stripe(b):
        return False, "stripe_box"
    lab = str(it.get("label", it.get("object_class", ""))).strip().lower()
    if lab in GENERIC:
        return False, "generic_label"
    if lab in BACKGROUND:
        return False, "background_label"
    return True, ""


# Kept in sync with eval/audit_flywheel.py:BACKGROUND so the validator rejects
# everything the offline auditor would flag (no silent audit/production drift).
BACKGROUND = {
    "wall", "floor", "ceiling", "background", "room", "table surface", "tabletop",
    "counter", "countertop", "surface", "sky", "ground", "wall panel", "window",
    "shadow", "tile", "backsplash", "door frame", "washing machine", "dryer",
    "machine", "appliance", "cabinet", "cabinet door", "sink basin", "light",
    "table", "desk", "shelf", "drawer front", "wood surface", "carpet", "rug",
    "wooden table", "white wall", "kitchen counter",
}


def _stripe(b) -> bool:
    """Full-width or full-height band => background, not an object."""
    w, h = b[2] - b[0], b[3] - b[1]
    return (w >= 900 and h >= 250) or (h >= 900 and w >= 250)


def v_detection(items, ctx=None):
    if not isinstance(items, list) or len(items) < 2:
        return False, "fewer_than_2_objects"
    if len(items) > 6:
        return False, "more_than_6_objects"
    seen = set()
    labs = []
    for it in items:
        if not isinstance(it, dict):
            return False, "not_dict"
        b = it.get("bbox_2d") or it.get("bbox")
        if not _bbox_ok(b):
            return False, "bad_bbox"
        if _area(b) > 0.55:
            return False, "bbox_too_large"
        if _stripe(b):
            return False, "stripe_box"
        lab = str(it.get("label", "")).strip().lower()
        if lab in GENERIC:
            return False, "generic_label"
        if lab in BACKGROUND:
            return False, "background_label"
        labs.append(lab)
        k = (lab, tuple(b))
        if k in seen:
            return False, "duplicate_box"
        seen.add(k)
    # The contract requires entity-level labels. Reusing a label makes downstream
    # depth identity ambiguous and often means one object was split into its faces.
    if len(labs) != len(set(labs)):
        return False, "duplicate_entity_label"
    return True, ""


def v_mask(items, ctx=None):
    it = items[0] if isinstance(items, list) and items else items
    if not isinstance(it, dict):
        return False, "not_dict"
    poly = it.get("polygon")
    if not isinstance(poly, list) or not poly:
        return False, "no_polygon"
    # The generator often emits a FLAT [x1,y1,x2,y2,...] list. That carries the same
    # information, so coerce it in place rather than discarding the sample.
    if all(isinstance(v, (int, float)) for v in poly):
        if len(poly) % 2 != 0:
            return False, "odd_flat_polygon"
        poly = [[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)]
        it["polygon"] = poly
    # Degenerate case: the model returns a bbox dressed as a polygon, e.g.
    # [[x1,y1,x2,y2]] (one nested 4-tuple). That is not a segmentation -> reject.
    if len(poly) == 1 and isinstance(poly[0], (list, tuple)) and len(poly[0]) == 4:
        return False, "polygon_is_bbox"
    # A nested list whose rows are longer than 2 is a flattened multi-vertex row.
    if all(isinstance(p, (list, tuple)) and len(p) > 2 and len(p) % 2 == 0 for p in poly):
        flat = [v for row in poly for v in row]
        if len(flat) >= 12:
            poly = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]
            it["polygon"] = poly
    if not all(_int_pair(p) for p in poly):
        return False, "mixed_polygon"
    if len(poly) < 6:
        return False, "too_few_vertices"
    for x, y in poly:
        if not (0 <= x < 1000 and 0 <= y < 1000):
            return False, "vertex_out_of_range"
    lab = str(it.get("label", "")).strip().lower()
    if lab in GENERIC:
        return False, "generic_label"
    if lab in BACKGROUND:
        return False, "background_label"

    # ---- alignment with the cascade anchor (single-version consistency) ----
    # The mask must segment the SAME object that bbox/dof/affordance describe, so the
    # polygon has to sit inside the anchor box (with a small tolerance).
    anchor = primary_grounding(ctx)
    ob = anchor.get("bbox")
    if _bbox_ok(ob):
        tol = 40
        outside = sum(1 for x, y in poly
                      if not (ob[0] - tol <= x <= ob[2] + tol and ob[1] - tol <= y <= ob[3] + tol))
        if outside > max(1, len(poly) // 8):
            return False, "polygon_outside_anchor"

    # ---- shape quality: a polygon that is effectively the bounding rectangle adds
    # nothing over the bbox form, so require a real (non-rectangular) outline.
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    hull = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if hull <= 0:
        return False, "degenerate_polygon"
    n = len(poly)
    shoelace = abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                       for i in range(n))) / 2.0
    if shoelace / hull > 0.97:
        return False, "polygon_is_rectangle"
    if shoelace / hull < 0.05:
        return False, "polygon_self_degenerate"
    anchor_label = str(anchor.get("object_class", "")).strip().lower()
    if anchor_label and lab != anchor_label:
        return False, "label_mismatch_anchor"
    return True, ""


def v_point(items, ctx=None):
    it = items[0] if isinstance(items, list) and items else items
    if not isinstance(it, dict):
        return False, "not_dict"
    p = it.get("point_2d") or it.get("contact_point_2d")
    if not _int_pair(p):
        return False, "bad_point"
    if not (0 <= p[0] < 1000 and 0 <= p[1] < 1000):
        return False, "point_out_of_range"
    ob = primary_grounding(ctx).get("bbox")
    if _bbox_ok(ob) and not (ob[0] <= p[0] <= ob[2] and ob[1] <= p[1] <= ob[3]):
        return False, "point_outside_anchor"
    return True, ""


def v_depth(items, ctx=None):
    if not isinstance(items, list) or len(items) < 2:
        return False, "fewer_than_2_objects"
    ranks, depths, labels = [], [], []
    for it in items:
        if not isinstance(it, dict):
            return False, "not_dict"
        r = it.get("rank"); d = it.get("relative_depth")
        if not isinstance(r, int):
            return False, "bad_rank"
        if not (isinstance(d, (int, float)) and 0 <= d <= 1):
            return False, "bad_depth"
        ranks.append(r)
        depths.append(float(d))
        labels.append(str(it.get("label", "")).strip().lower())
    if len(set(ranks)) != len(ranks):
        return False, "duplicate_rank"
    if sorted(ranks) != list(range(1, len(items) + 1)):
        return False, "non_contiguous_rank"
    ordered_depths = [d for _, d in sorted(zip(ranks, depths))]
    if any(a >= b for a, b in zip(ordered_depths, ordered_depths[1:])):
        return False, "depth_not_monotonic"
    expected = {str(g.get("object_class", "")).strip().lower() for g in groundings(ctx)}
    if expected and set(labels) != expected:
        return False, "labels_mismatch_groundings"
    return True, ""


def _v4_confidence(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and DEPTH_V4_MIN_CONFIDENCE <= float(value) <= 1.0
    )


def validate_depth_v4_record(record):
    """Validate the complete generator -> evidence -> falsifier chain."""
    if (
        isinstance(record, dict)
        and (
            record.get("prompt_version") == DEPTH_V43_PAIR_PROMPT_VERSION
            or record.get("label_version") == DEPTH_V43_PAIR_LABEL_VERSION
        )
    ):
        return validate_depth_v43_pair_record(record)
    if (
        isinstance(record, dict)
        and record.get("prompt_version") == DEPTH_V4_PAIR_PROMPT_VERSION
    ):
        return validate_depth_v4_pair_record(record)
    if not isinstance(record, dict):
        return False, "record_not_object"
    if record.get("_audit_only") is True:
        return False, "audit_only"
    if record.get("task_form") != "depth":
        return False, "wrong_form"
    if record.get("depth_scope") not in (None, "total_order"):
        return False, "v4_wrong_scope"
    if record.get("label_version") != DEPTH_V4_LABEL_VERSION:
        return False, "wrong_label_version"
    if record.get("prompt_version") != DEPTH_V4_PROMPT_VERSION:
        return False, "wrong_prompt_version"
    payload = record.get("payload")
    context = record.get("gsa_context")
    valid, reason = v_depth(payload, context)
    if not valid:
        return False, reason
    entities = groundings(context)
    if not 2 <= len(entities) <= 4:
        return False, "v4_entity_cardinality"
    entity_labels = [item.get("object_class") for item in entities]
    order_labels = [item.get("label") for item in payload]
    if (
        not all(isinstance(label, str) and label for label in entity_labels)
        or len(set(entity_labels)) != len(entity_labels)
        or set(order_labels) != set(entity_labels)
    ):
        return False, "v4_entity_identity"
    depths = [float(item["relative_depth"]) for item in payload]
    if depths[0] != 0.0 or depths[-1] != 1.0:
        return False, "v4_depth_endpoints"

    spatial = _spatial(context)
    evidence = spatial.get("adjacent_depth_evidence")
    if spatial.get("depth_order") != payload:
        return False, "v4_spatial_order_drift"
    if not isinstance(evidence, list) or len(evidence) != len(payload) - 1:
        return False, "v4_evidence_cardinality"
    expected_pairs = list(zip(order_labels, order_labels[1:]))
    for item, (nearer, farther) in zip(evidence, expected_pairs):
        if not isinstance(item, dict):
            return False, "v4_evidence_item"
        text = item.get("evidence")
        if (
            item.get("nearer") != nearer
            or item.get("farther") != farther
            or item.get("cue_type") not in DEPTH_V4_CUES
            or not isinstance(text, str)
            or len(text.strip()) < 20
        ):
            return False, "v4_evidence_schema"
        folded = text.casefold()
        if nearer.casefold() not in folded or farther.casefold() not in folded:
            return False, "v4_evidence_identity"
        if any(term in folded for term in DEPTH_V4_BANNED_EVIDENCE):
            return False, "v4_unsupported_heuristic"

    gate = record.get("_semantic_gate")
    depth_gate = gate.get("depth_v4") if isinstance(gate, dict) else None
    if not isinstance(depth_gate, dict):
        return False, "v4_missing_semantic_gate"
    generator = depth_gate.get("generator")
    falsifier = depth_gate.get("falsifier")
    if not isinstance(generator, dict) or not isinstance(falsifier, dict):
        return False, "v4_incomplete_semantic_gate"
    if (
        generator.get("orderable") is not True
        or generator.get("reason_code") != "supported"
        or not _v4_confidence(generator.get("confidence"))
        or generator.get("order") != payload
        or generator.get("adjacent_evidence") != evidence
    ):
        return False, "v4_generator_gate"
    checks = falsifier.get("pair_checks")
    if (
        falsifier.get("valid") is not True
        or falsifier.get("reason_code") != "no_counterexample"
        or not _v4_confidence(falsifier.get("confidence"))
        or not isinstance(checks, list)
        or len(checks) != len(expected_pairs)
    ):
        return False, "v4_falsifier_gate"
    for check, (nearer, farther) in zip(checks, expected_pairs):
        if (
            not isinstance(check, dict)
            or check.get("nearer") != nearer
            or check.get("farther") != farther
            or check.get("supported") is not True
            or not _v4_confidence(check.get("confidence"))
            or check.get("counterevidence") not in ("", None)
        ):
            return False, "v4_falsifier_pair"
    verification = record.get("_verification")
    if not isinstance(verification, dict) or verification.get("verdict") != "verified":
        return False, "v4_verification"
    return True, ""


def validate_depth_v4_pair_record(record):
    """Validate one explicitly scoped, visibly supported pairwise depth relation."""
    if not isinstance(record, dict):
        return False, "record_not_object"
    if record.get("_audit_only") is True:
        return False, "audit_only"
    if record.get("task_form") != "depth":
        return False, "wrong_form"
    if record.get("depth_scope") != "pairwise":
        return False, "v4_pair_wrong_scope"
    if record.get("label_version") != DEPTH_V4_PAIR_LABEL_VERSION:
        return False, "wrong_label_version"
    if record.get("prompt_version") != DEPTH_V4_PAIR_PROMPT_VERSION:
        return False, "wrong_prompt_version"

    payload = record.get("payload")
    context = record.get("gsa_context")
    if not isinstance(payload, list) or len(payload) != 2:
        return False, "v4_pair_payload_cardinality"
    if not all(
        isinstance(item, dict)
        and set(item) == {"label", "rank", "relative_depth"}
        for item in payload
    ):
        return False, "v4_pair_payload_schema"
    raw_entities = context.get("groundings") if isinstance(context, dict) else None
    if (
        not isinstance(raw_entities, list)
        or len(raw_entities) != 2
        or not all(isinstance(item, dict) for item in raw_entities)
    ):
        return False, "v4_pair_entity_cardinality"
    valid, reason = v_depth(payload, context)
    if not valid:
        return False, reason
    entities = groundings(context)
    if len(entities) != 2:
        return False, "v4_pair_entity_cardinality"
    if any(not _bbox_ok(item.get("bbox")) for item in entities):
        return False, "v4_pair_grounding_bbox"
    if context.get("primary") != entities[0]:
        return False, "v4_pair_primary_drift"
    entity_labels = [item.get("object_class") for item in entities]
    order_labels = [item.get("label") for item in payload]
    if (
        not all(isinstance(label, str) and label for label in entity_labels)
        or len(set(entity_labels)) != 2
        or order_labels != entity_labels
    ):
        return False, "v4_pair_entity_identity"
    if [item.get("rank") for item in payload] != [1, 2]:
        return False, "v4_pair_rank"
    if [float(item["relative_depth"]) for item in payload] != [0.0, 1.0]:
        return False, "v4_pair_depth_endpoints"

    nearer, farther = order_labels
    spatial = _spatial(context)
    if set(spatial) != {
        "relation_mode", "depth_order", "adjacent_depth_evidence"
    }:
        return False, "v4_pair_spatial_schema"
    if spatial.get("relation_mode") != "pairwise":
        return False, "v4_pair_spatial_scope"
    if spatial.get("depth_order") != payload:
        return False, "v4_pair_spatial_order_drift"
    evidence_items = spatial.get("adjacent_depth_evidence")
    if not isinstance(evidence_items, list) or len(evidence_items) != 1:
        return False, "v4_pair_evidence_cardinality"
    evidence = evidence_items[0]
    if not isinstance(evidence, dict):
        return False, "v4_pair_evidence_item"
    if set(evidence) != {"nearer", "farther", "cue_type", "evidence"}:
        return False, "v4_pair_evidence_schema"
    text = evidence.get("evidence")
    if (
        evidence.get("nearer") != nearer
        or evidence.get("farther") != farther
        or evidence.get("cue_type") not in DEPTH_V4_CUES
        or not isinstance(text, str)
        or len(text.strip()) < 20
    ):
        return False, "v4_pair_evidence_schema"
    folded = text.casefold()
    if nearer.casefold() not in folded or farther.casefold() not in folded:
        return False, "v4_pair_evidence_identity"
    if any(term in folded for term in DEPTH_V4_BANNED_EVIDENCE):
        return False, "v4_pair_unsupported_heuristic"

    gate = record.get("_semantic_gate")
    depth_gate = gate.get("depth_v4_pairwise") if isinstance(gate, dict) else None
    if not isinstance(depth_gate, dict):
        return False, "v4_pair_missing_semantic_gate"
    if depth_gate.get("role_order_counterbalanced") is not True:
        return False, "v4_pair_role_order_not_counterbalanced"
    selection = depth_gate.get("selection")
    if not isinstance(selection, dict):
        return False, "v4_pair_missing_selection_gate"
    stable_hash = selection.get("stable_hash")
    pairs_evaluated = selection.get("pairs_evaluated")
    pair_index = selection.get("pair_index")
    sources = record.get("_v4_sources")
    image_sha = sources.get("image_sha256") if isinstance(sources, dict) else None
    expected_selection_hash = None
    if (
        isinstance(image_sha, str)
        and len(image_sha) == 64
        and not any(char not in "0123456789abcdef" for char in image_sha)
    ):
        sorted_labels = sorted(order_labels)
        selection_value = (
            f"{record['prompt_version']}|{image_sha}|{sorted_labels[0]}|"
            f"{sorted_labels[1]}|{nearer}>{farther}"
        )
        expected_selection_hash = hashlib.sha256(
            selection_value.encode("utf-8")
        ).hexdigest()
    if (
        selection.get("rule") != "stable_hash_after_threshold"
        or not isinstance(stable_hash, str)
        or len(stable_hash) != 64
        or any(char not in "0123456789abcdef" for char in stable_hash)
        or stable_hash != expected_selection_hash
        or not _v4_confidence(selection.get("gate_confidence"))
        or not isinstance(pairs_evaluated, int)
        or isinstance(pairs_evaluated, bool)
        or pairs_evaluated < 1
        or not isinstance(pair_index, int)
        or isinstance(pair_index, bool)
        or not 0 <= pair_index < pairs_evaluated
    ):
        return False, "v4_pair_selection_gate"
    generator = depth_gate.get("generator")
    falsifier = depth_gate.get("falsifier")
    if not isinstance(generator, dict) or not isinstance(falsifier, dict):
        return False, "v4_pair_incomplete_semantic_gate"
    selected = generator.get("selected_pair")
    selected_evidence = (
        {
            key: selected.get(key)
            for key in ("nearer", "farther", "cue_type", "evidence")
        }
        if isinstance(selected, dict)
        else None
    )
    if (
        generator.get("orderable") is not True
        or generator.get("reason_code") != "supported_pair"
        or not _v4_confidence(generator.get("confidence"))
        or not isinstance(selected, dict)
        or not _v4_confidence(selected.get("confidence", generator.get("confidence")))
        or selected_evidence != evidence
    ):
        return False, "v4_pair_generator_gate"
    check = falsifier.get("pair_check")
    if (
        falsifier.get("valid") is not True
        or falsifier.get("reason_code") != "independent_support"
        or not _v4_confidence(falsifier.get("confidence"))
        or not isinstance(check, dict)
        or set(check) != {
            "nearer", "farther", "cue_type", "evidence", "supported",
            "confidence", "counterevidence"
        }
        or check.get("nearer") != nearer
        or check.get("farther") != farther
        or check.get("supported") is not True
        or not _v4_confidence(check.get("confidence"))
        or check.get("counterevidence") not in ("", None)
    ):
        return False, "v4_pair_verifier_gate"
    verifier_text = check.get("evidence")
    if (
        check.get("cue_type") not in DEPTH_V4_CUES
        or not isinstance(verifier_text, str)
        or len(verifier_text.strip()) < 20
    ):
        return False, "v4_pair_verifier_evidence_schema"
    verifier_folded = verifier_text.casefold()
    if nearer.casefold() not in verifier_folded or farther.casefold() not in verifier_folded:
        return False, "v4_pair_verifier_evidence_identity"
    if any(term in verifier_folded for term in DEPTH_V4_BANNED_EVIDENCE):
        return False, "v4_pair_verifier_unsupported_heuristic"
    verification = record.get("_verification")
    if not isinstance(verification, dict) or verification.get("verdict") != "verified_pair":
        return False, "v4_pair_verification"
    return True, ""


def _v43_has_reverse_direction(text: str, nearer: str, farther: str) -> bool:
    folded = " ".join(text.casefold().split())
    near = nearer.casefold()
    far = farther.casefold()
    direct = (
        f"{far} is nearer to the camera than {near}",
        f"{far} is closer to the camera than {near}",
        f"{near} is farther from the camera than {far}",
        f"{near} is further from the camera than {far}",
        f"{far} is in front of {near}",
        f"{near} is behind {far}",
    )
    if any(statement in folded for statement in direct):
        return True
    near_pattern = re.escape(near)
    far_pattern = re.escape(far)
    reverse_patterns = (
        rf"{near_pattern}(?:(?!{far_pattern}).){{0,120}}"
        rf"(?:farther|further)\s+(?:back|from\s+(?:the\s+)?camera)",
        rf"{far_pattern}(?:(?!{near_pattern}).){{0,120}}"
        rf"(?:nearer|closer)\s+to\s+(?:the\s+)?camera",
    )
    return any(re.search(pattern, folded) for pattern in reverse_patterns)


def _validate_v43_claim(
    claim: dict,
    nearer: str,
    farther: str,
    cue_type: str,
    prefix: str,
) -> tuple[bool, str]:
    evidence = claim.get("evidence")
    if (
        claim.get("nearer") != nearer
        or claim.get("farther") != farther
        or claim.get("cue_type") != cue_type
        or not isinstance(evidence, str)
        or len(evidence.strip()) < 30
    ):
        return False, f"{prefix}_claim_schema"
    folded = evidence.casefold()
    if nearer.casefold() not in folded or farther.casefold() not in folded:
        return False, f"{prefix}_evidence_identity"
    banned = DEPTH_V4_BANNED_EVIDENCE | DEPTH_V43_FRAME_HEURISTICS
    if any(term in folded for term in banned):
        return False, f"{prefix}_unsupported_heuristic"
    cue_supported = (
        "occlud" in folded
        if cue_type == "occlusion"
        else (
            ("contact" in folded or "layer" in folded)
            and "occlud" not in folded
        )
        if cue_type == "contact_layering"
        else any(
            term in folded for term in ("support plane", "support-plane", "perspective")
        )
        and "occlud" not in folded
    )
    if not cue_supported:
        return False, f"{prefix}_evidence_cue_mismatch"
    if _v43_has_reverse_direction(evidence, nearer, farther):
        return False, f"{prefix}_reverse_direction"

    direction_statement = f"{nearer} is nearer to the camera than {farther}."
    if direction_statement.casefold() not in folded:
        return False, f"{prefix}_missing_direction_statement"

    observation = claim.get("observation")
    if not isinstance(observation, dict) or set(observation) != {
        "nearer", "farther", "relation", "geometry_claim", "direction_statement"
    }:
        return False, f"{prefix}_observation_schema"
    relation, geometry_claim = DEPTH_V43_CUE_SEMANTICS[cue_type]
    if observation != {
        "nearer": nearer,
        "farther": farther,
        "relation": relation,
        "geometry_claim": geometry_claim,
        "direction_statement": direction_statement,
    }:
        return False, f"{prefix}_observation_entailment"
    return True, ""


def _v43_geometry_metrics(entities: list[dict]) -> dict[str, float]:
    first = entities[0]["bbox"]
    second = entities[1]["bbox"]
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    intersection_area = intersection_width * intersection_height
    return {
        "intersection_width": intersection_width,
        "intersection_height": intersection_height,
        "intersection_area": intersection_area,
        "intersection_over_min_box_area": (
            intersection_area / min(first_area, second_area)
        ),
        "normalized_edge_distance": math.hypot(dx, dy) / (1000.0 * math.sqrt(2.0)),
        "max_axis_edge_distance": max(dx, dy),
    }


def _v43_metric_matches(value, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12)
    )


def validate_depth_v43_pair_record(record):
    """Validate the v4.3 cue-consistent, geometry-bound pairwise protocol."""
    if not isinstance(record, dict):
        return False, "record_not_object"
    if record.get("_audit_only") is True:
        return False, "audit_only"
    if record.get("task_form") != "depth":
        return False, "wrong_form"
    if record.get("depth_scope") != "pairwise":
        return False, "v43_pair_wrong_scope"
    if record.get("label_version") != DEPTH_V43_PAIR_LABEL_VERSION:
        return False, "wrong_label_version"
    if record.get("prompt_version") != DEPTH_V43_PAIR_PROMPT_VERSION:
        return False, "wrong_prompt_version"
    sample_id = record.get("sample_id")
    source_sample_id = record.get("source_sample_id")
    if not (
        isinstance(sample_id, str)
        and (
            sample_id.startswith("v43_depth_")
            or (
                sample_id.startswith("v4_depth_")
                and isinstance(source_sample_id, str)
                and source_sample_id.startswith("v43_depth_")
            )
        )
    ):
        return False, "v43_pair_sample_identity"

    payload = record.get("payload")
    context = record.get("gsa_context")
    if not isinstance(payload, list) or len(payload) != 2:
        return False, "v43_pair_payload_cardinality"
    if not all(
        isinstance(item, dict)
        and set(item) == {"label", "rank", "relative_depth"}
        for item in payload
    ):
        return False, "v43_pair_payload_schema"
    if not isinstance(context, dict) or context.get("contract_version") != "gsa-contract-v2":
        return False, "v43_pair_context_contract"
    raw_entities = context.get("groundings")
    if (
        not isinstance(raw_entities, list)
        or len(raw_entities) != 2
        or not all(isinstance(item, dict) for item in raw_entities)
    ):
        return False, "v43_pair_entity_cardinality"
    valid, reason = v_depth(payload, context)
    if not valid:
        return False, reason
    entities = groundings(context)
    if len(entities) != 2 or any(not _bbox_ok(item.get("bbox")) for item in entities):
        return False, "v43_pair_grounding_bbox"
    if context.get("primary") != entities[0]:
        return False, "v43_pair_primary_drift"
    entity_labels = [item.get("object_class") for item in entities]
    order_labels = [item.get("label") for item in payload]
    if (
        not all(isinstance(label, str) and label for label in entity_labels)
        or len(set(entity_labels)) != 2
        or order_labels != entity_labels
    ):
        return False, "v43_pair_entity_identity"
    if [item.get("rank") for item in payload] != [1, 2]:
        return False, "v43_pair_rank"
    if [float(item["relative_depth"]) for item in payload] != [0.0, 1.0]:
        return False, "v43_pair_depth_endpoints"
    nearer, farther = order_labels
    if (
        record.get("object_class") != nearer
        or record.get("bbox") != entities[0].get("bbox")
    ):
        return False, "v43_pair_primary_projection_drift"

    spatial = _spatial(context)
    if set(spatial) != {
        "relation_mode", "depth_order", "adjacent_depth_evidence"
    }:
        return False, "v43_pair_spatial_schema"
    if spatial.get("relation_mode") != "pairwise" or spatial.get("depth_order") != payload:
        return False, "v43_pair_spatial_order_drift"
    evidence_items = spatial.get("adjacent_depth_evidence")
    if not isinstance(evidence_items, list) or len(evidence_items) != 1:
        return False, "v43_pair_evidence_cardinality"
    evidence = evidence_items[0]
    if not isinstance(evidence, dict) or set(evidence) != {
        "nearer", "farther", "cue_type", "evidence", "observation"
    }:
        return False, "v43_pair_evidence_schema"
    cue_type = evidence.get("cue_type")
    if cue_type not in DEPTH_V43_CUE_SEMANTICS:
        return False, "v43_pair_cue"
    if evidence.get("nearer") != nearer or evidence.get("farther") != farther:
        return False, "v43_pair_evidence_direction"
    valid, reason = _validate_v43_claim(
        evidence, nearer, farther, cue_type, "v43_pair_spatial"
    )
    if not valid:
        return False, reason

    gate = record.get("_semantic_gate")
    depth_gate = gate.get("depth_v43_pairwise") if isinstance(gate, dict) else None
    if not isinstance(depth_gate, dict) or set(depth_gate) != {
        "generator", "verifier", "role_order_counterbalanced", "cue_agreement",
        "geometry", "selection"
    }:
        return False, "v43_pair_semantic_gate_schema"
    if depth_gate.get("role_order_counterbalanced") is not True:
        return False, "v43_pair_role_order_not_counterbalanced"
    if depth_gate.get("cue_agreement") is not True:
        return False, "v43_pair_cue_agreement"

    generator = depth_gate.get("generator")
    verifier = depth_gate.get("verifier")
    if not isinstance(generator, dict) or set(generator) != {
        "orderable", "reason_code", "confidence", "selected_pair"
    }:
        return False, "v43_pair_generator_schema"
    selected = generator.get("selected_pair")
    if (
        generator.get("orderable") is not True
        or generator.get("reason_code") != "supported_pair"
        or not _v4_confidence(generator.get("confidence"))
        or not isinstance(selected, dict)
        or set(selected) != {
            "nearer", "farther", "cue_type", "evidence", "observation", "confidence"
        }
        or not _v4_confidence(selected.get("confidence"))
    ):
        return False, "v43_pair_generator_gate"
    selected_evidence = {
        key: selected.get(key)
        for key in ("nearer", "farther", "cue_type", "evidence", "observation")
    }
    if selected_evidence != evidence:
        return False, "v43_pair_generator_spatial_drift"
    valid, reason = _validate_v43_claim(
        selected, nearer, farther, cue_type, "v43_pair_generator"
    )
    if not valid:
        return False, reason

    if not isinstance(verifier, dict) or set(verifier) != {
        "valid", "reason_code", "confidence", "pair_check"
    }:
        return False, "v43_pair_verifier_schema"
    check = verifier.get("pair_check")
    if (
        verifier.get("valid") is not True
        or verifier.get("reason_code") != "independent_support"
        or not _v4_confidence(verifier.get("confidence"))
        or not isinstance(check, dict)
        or set(check) != {
            "nearer", "farther", "cue_type", "evidence", "observation", "supported",
            "confidence", "counterevidence"
        }
        or check.get("supported") is not True
        or not _v4_confidence(check.get("confidence"))
        or check.get("counterevidence") not in ("", None)
    ):
        return False, "v43_pair_verifier_gate"
    valid, reason = _validate_v43_claim(
        check, nearer, farther, cue_type, "v43_pair_verifier"
    )
    if not valid:
        return False, reason

    geometry = depth_gate.get("geometry")
    if not isinstance(geometry, dict) or set(geometry) != {
        "cue_type", "passed", "intersection_width", "intersection_height",
        "intersection_area", "intersection_over_min_box_area",
        "min_intersection_over_min_box_area", "normalized_edge_distance",
        "max_normalized_edge_distance", "max_axis_edge_distance",
        "max_axis_edge_distance_limit"
    }:
        return False, "v43_pair_geometry_schema"
    expected_geometry = _v43_geometry_metrics(entities)
    if (
        geometry.get("cue_type") != cue_type
        or geometry.get("passed") is not True
        or not all(
            _v43_metric_matches(geometry.get(key), expected)
            for key, expected in expected_geometry.items()
        )
        or not _v43_metric_matches(
            geometry.get("max_normalized_edge_distance"),
            DEPTH_V43_MAX_NORMALIZED_EDGE_DISTANCE,
        )
        or not _v43_metric_matches(
            geometry.get("max_axis_edge_distance_limit"),
            DEPTH_V43_MAX_AXIS_EDGE_DISTANCE,
        )
        or not _v43_metric_matches(
            geometry.get("min_intersection_over_min_box_area"),
            DEPTH_V43_MIN_INTERSECTION_OVER_MIN_BOX_AREA,
        )
    ):
        return False, "v43_pair_geometry_provenance"
    positive_intersection = (
        expected_geometry["intersection_width"] > 0
        and expected_geometry["intersection_height"] > 0
    )
    geometry_supported = (
        (
            positive_intersection
            and expected_geometry["intersection_over_min_box_area"]
            >= DEPTH_V43_MIN_INTERSECTION_OVER_MIN_BOX_AREA
        )
        if cue_type == "occlusion"
        else (
            positive_intersection
            or (
                expected_geometry["normalized_edge_distance"]
                <= DEPTH_V43_MAX_NORMALIZED_EDGE_DISTANCE
                and expected_geometry["max_axis_edge_distance"]
                <= DEPTH_V43_MAX_AXIS_EDGE_DISTANCE
            )
        )
        if cue_type == "contact_layering"
        else True
    )
    if not geometry_supported:
        return False, "v43_pair_geometry_not_supported"

    selection = depth_gate.get("selection")
    if not isinstance(selection, dict) or set(selection) != {
        "rule", "pair_index", "pairs_evaluated", "stable_hash", "gate_confidence"
    }:
        return False, "v43_pair_selection_schema"
    stable_hash = selection.get("stable_hash")
    pairs_evaluated = selection.get("pairs_evaluated")
    pair_index = selection.get("pair_index")
    sources = record.get("_v4_sources")
    image_sha = sources.get("image_sha256") if isinstance(sources, dict) else None
    image_sha_valid = (
        isinstance(image_sha, str)
        and len(image_sha) == 64
        and not any(char not in "0123456789abcdef" for char in image_sha)
    )
    expected_selection_hash = None
    if image_sha_valid:
        sorted_labels = sorted(order_labels)
        selection_value = (
            f"{record['prompt_version']}|{image_sha}|{sorted_labels[0]}|"
            f"{sorted_labels[1]}|{nearer}>{farther}"
        )
        expected_selection_hash = hashlib.sha256(
            selection_value.encode("utf-8")
        ).hexdigest()
    gate_confidence = min(
        float(generator["confidence"]),
        float(selected["confidence"]),
        float(verifier["confidence"]),
        float(check["confidence"]),
    )
    if (
        selection.get("rule") != "stable_hash_after_threshold"
        or not isinstance(stable_hash, str)
        or len(stable_hash) != 64
        or any(char not in "0123456789abcdef" for char in stable_hash)
        or stable_hash != expected_selection_hash
        or not _v4_confidence(selection.get("gate_confidence"))
        or not _v43_metric_matches(selection.get("gate_confidence"), gate_confidence)
        or not isinstance(pairs_evaluated, int)
        or isinstance(pairs_evaluated, bool)
        or pairs_evaluated < 1
        or not isinstance(pair_index, int)
        or isinstance(pair_index, bool)
        or not 0 <= pair_index < pairs_evaluated
    ):
        return False, "v43_pair_selection_gate"
    if (
        not _v4_confidence(record.get("confidence"))
        or not _v43_metric_matches(record.get("confidence"), gate_confidence)
    ):
        return False, "v43_pair_record_confidence"

    verification = record.get("_verification")
    signals = verification.get("signals") if isinstance(verification, dict) else None
    if (
        not isinstance(verification, dict)
        or verification.get("verdict") != "verified_pair"
        or not isinstance(signals, dict)
        or signals.get("exact_cue_agreement") != 1.0
        or signals.get("frozen_bbox_geometry") != 1.0
    ):
        return False, "v43_pair_verification"
    return True, ""


def v_affordance(items, ctx=None):
    it = items[0] if isinstance(items, list) and items else items
    if not isinstance(it, dict):
        return False, "not_dict"
    b = it.get("bbox_2d") or it.get("bbox")
    if not _bbox_ok(b):
        return False, "bad_bbox"
    cp = it.get("contact_point_2d")
    if not _int_pair(cp):
        return False, "bad_contact_point"
    if not (b[0] - 20 <= cp[0] <= b[2] + 20 and b[1] - 20 <= cp[1] <= b[3] + 20):
        return False, "contact_outside_part"
    label = str(it.get("label", "")).strip().lower()
    if label in GENERIC:
        return False, "generic_label"
    if label in BACKGROUND:
        return False, "background_label"
    action = str(it.get("affordance", "")).strip().lower()
    if not action:
        return False, "no_affordance_verb"
    if action not in AFFORDANCE_VERBS:
        return False, "invalid_affordance_verb"
    # the part must be a strict sub-region of the object (this was a real failure mode)
    ob = primary_grounding(ctx).get("bbox")
    object_label = str(primary_grounding(ctx).get("object_class", "")).strip().lower()
    if object_label and label == object_label:
        return False, "part_label_equals_object"
    if _bbox_ok(ob):
        if not _inside(b, ob):
            return False, "part_outside_object"
        if _area(b) > 0.9 * _area(ob):
            return False, "part_equals_object"
    return True, ""


def v_dof(items, ctx=None):
    it = items[0] if isinstance(items, list) and items else items
    if not isinstance(it, dict):
        return False, "not_dict"
    gp = it.get("grasp_point_2d"); ax = it.get("jaw_axis_2d")
    if not _int_pair(gp):
        return False, "bad_grasp_point"
    if not _int_pair(ax):
        return False, "bad_jaw_axis"
    if abs(ax[0]) < 1e-6 and abs(ax[1]) < 1e-6:
        return False, "zero_jaw_axis"
    norm = math.hypot(float(ax[0]), float(ax[1]))
    if not 0.85 <= norm <= 1.15:
        return False, "jaw_axis_not_unit"
    # Cardinal and diagonal axes are legal geometry. A frequent angle alone does
    # not prove a copied template; visible contact support is checked separately
    # by the manipulation verifier and the independent annotation workflow.
    if it.get("approach") not in ("top", "side", "front"):
        return False, "bad_approach"
    w = it.get("wrist_deg")
    if not (isinstance(w, (int, float)) and -90 <= w <= 90):
        return False, "bad_wrist_deg"
    angle = math.degrees(math.atan2(float(ax[1]), float(ax[0])))
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    # Parallel jaws are symmetric under a 180-degree rotation.
    if abs((float(w) - angle + 90) % 180 - 90) > 15:
        return False, "wrist_axis_inconsistent"
    ob = primary_grounding(ctx).get("bbox")
    object_label = str(primary_grounding(ctx).get("object_class", "")).strip().lower()
    label = str(it.get("label", "")).strip().lower()
    if object_label and label != object_label:
        return False, "label_mismatch_anchor"
    if _bbox_ok(ob) and not (ob[0] - 30 <= gp[0] <= ob[2] + 30 and ob[1] - 30 <= gp[1] <= ob[3] + 30):
        return False, "grasp_outside_object"
    affordance = _first_dict(_spatial(ctx).get("affordance"))
    part_box = affordance.get("bbox_2d") or affordance.get("bbox")
    if _bbox_ok(part_box):
        if not (part_box[0] - 20 <= gp[0] <= part_box[2] + 20
                and part_box[1] - 20 <= gp[1] <= part_box[3] + 20):
            return False, "grasp_outside_affordance"
    return True, ""


def v_action_cot(items, ctx=None):
    it = items[0] if isinstance(items, list) and items else items
    if not isinstance(it, dict):
        return False, "not_dict"
    steps = it.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return False, "too_few_steps"
    if not all(isinstance(s, str) and len(s.strip()) > 8 for s in steps):
        return False, "empty_step"
    if not it.get("goal"):
        return False, "no_goal"
    tp = it.get("target_point_2d")
    if not _int_pair(tp):
        return False, "bad_target_point"
    if not (0 <= tp[0] < 1000 and 0 <= tp[1] < 1000):
        return False, "target_point_out_of_range"
    ob = primary_grounding(ctx).get("bbox")
    if _bbox_ok(ob):
        if not (ob[0] - 30 <= tp[0] <= ob[2] + 30 and ob[1] - 30 <= tp[1] <= ob[3] + 30):
            return False, "target_outside_anchor"
    if not isinstance(it.get("preconditions"), list) or not it["preconditions"]:
        return False, "no_preconditions"
    if not isinstance(it.get("failure_modes"), list) or not it["failure_modes"]:
        return False, "no_failure_modes"

    dependencies = _spatial(ctx)
    dof = _first_dict(dependencies.get("dof"))
    grasp = dof.get("grasp_point_2d")
    if not _int_pair(grasp):
        return False, "missing_dof_dependency"
    if math.hypot(float(tp[0]) - float(grasp[0]),
                  float(tp[1]) - float(grasp[1])) > 5:
        return False, "target_dof_mismatch"

    affordance = _first_dict(dependencies.get("affordance"))
    action = str(affordance.get("affordance", "")).strip().lower()
    text = " ".join(
        [str(it.get("goal", "")), *[str(step) for step in steps]]
        + [str(value) for value in it.get("preconditions", [])]
        + [str(value) for value in it.get("failure_modes", [])]
    ).lower()
    if any(term in text for term in UNSUPPORTED_ACTION_TERMS):
        return False, "unsupported_assumption"
    aliases = {
        "grasp": ("grasp", "pick", "lift", "hold"),
        "pinch": ("pinch", "grasp", "pick", "lift"),
        "pull": ("pull", "open"),
        "push": ("push", "close"),
        "press": ("press", "push"),
        "turn": ("turn", "rotate"),
        "rotate": ("rotate", "turn"),
        "lift": ("lift", "pick"),
        "open": ("open", "pull"),
        "close": ("close", "push"),
        "hold": ("hold", "grasp"),
        "slide": ("slide", "pull", "push"),
        "pick": ("pick", "grasp", "lift"),
        "pick up": ("pick", "grasp", "lift"),
    }
    if action not in aliases:
        return False, "missing_affordance_dependency"
    if not any(token in text for token in aliases[action]):
        return False, "action_affordance_mismatch"

    object_label = str(primary_grounding(ctx).get("object_class", "")).lower()
    if any(term in object_label for term in FIXED_OBJECT_TERMS):
        if "pick up" in text or "lift off" in text or "detach" in text:
            return False, "attached_object_detachment"
    return True, ""


VALIDATORS = {
    "bbox": v_bbox, "detection": v_detection, "mask": v_mask, "point": v_point,
    "depth": v_depth, "affordance": v_affordance, "dof": v_dof,
    "action_cot": v_action_cot,
}


def validate_structural(form: str, items, ctx=None):
    """Check only parseable shape for human-audit candidates.

    This deliberately omits semantic agreement checks. It is never used by the
    training path; it lets the independent A/B audit see plausible-but-uncertain
    candidates instead of silently selecting only easy examples.
    """
    if not isinstance(items, list) or not items or not all(
        isinstance(item, dict) for item in items
    ):
        return False, "not_nonempty_object_list"
    if form in ("bbox", "detection"):
        if form == "bbox" and len(items) != 1:
            return False, "bbox_cardinality"
        if form == "detection" and not 2 <= len(items) <= 6:
            return False, "detection_cardinality"
        for item in items:
            box = item.get("bbox_2d") or item.get("bbox")
            label = str(item.get("label", "")).strip()
            if not _bbox_ok(box) or not label:
                return False, "bad_bbox_record"
        return True, ""
    if form == "point":
        point = items[0].get("point_2d") or items[0].get("contact_point_2d")
        return (_int_pair(point) and 0 <= point[0] < 1000
                and 0 <= point[1] < 1000, "bad_point_record")
    if form == "mask":
        polygon = items[0].get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            return False, "bad_polygon_record"
        if all(isinstance(value, (int, float)) for value in polygon):
            if len(polygon) % 2:
                return False, "bad_flat_polygon"
            polygon = [[polygon[i], polygon[i + 1]]
                       for i in range(0, len(polygon), 2)]
        if not all(_int_pair(point) for point in polygon):
            return False, "bad_polygon_record"
        if not all(0 <= point[0] < 1000 and 0 <= point[1] < 1000
                   for point in polygon):
            return False, "polygon_out_of_range"
        return True, ""
    if form == "depth":
        if len(items) < 2:
            return False, "depth_cardinality"
        for item in items:
            if not isinstance(item.get("label"), str) or not item["label"].strip():
                return False, "depth_missing_label"
            if not isinstance(item.get("rank"), int):
                return False, "depth_bad_rank"
            value = item.get("relative_depth")
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                return False, "depth_bad_value"
        return True, ""
    if form == "affordance":
        item = items[0]
        box = item.get("bbox_2d") or item.get("bbox")
        point = item.get("contact_point_2d")
        if (not _bbox_ok(box) or not _int_pair(point)
                or not str(item.get("label", "")).strip()
                or not str(item.get("affordance", "")).strip()):
            return False, "bad_affordance_record"
        return True, ""
    if form == "dof":
        item = items[0]
        point = item.get("grasp_point_2d")
        axis = item.get("jaw_axis_2d")
        if (not _int_pair(point) or not _int_pair(axis)
                or not str(item.get("label", "")).strip()
                or item.get("approach") not in ("top", "side", "front")
                or not isinstance(item.get("wrist_deg"), (int, float))):
            return False, "bad_dof_record"
        if not (0 <= point[0] < 1000 and 0 <= point[1] < 1000):
            return False, "dof_point_out_of_range"
        return True, ""
    if form == "action_cot":
        item = items[0]
        target = item.get("target_point_2d")
        if (not str(item.get("goal", "")).strip()
                or not isinstance(item.get("steps"), list)
                or len(item["steps"]) < 2
                or not all(isinstance(step, str) and step.strip()
                           for step in item["steps"])
                or not isinstance(item.get("preconditions"), list)
                or not isinstance(item.get("failure_modes"), list)
                or not _int_pair(target)
                or not (0 <= target[0] < 1000 and 0 <= target[1] < 1000)):
            return False, "bad_action_record"
        return True, ""
    return False, "unknown_form"


def validate(form: str, payload, ctx=None):
    fn = VALIDATORS.get(form)
    if fn is None:
        return True, ""
    try:
        return fn(payload, ctx)
    except Exception as e:  # defensive: a malformed reply must never crash the run
        return False, f"validator_error:{type(e).__name__}"
