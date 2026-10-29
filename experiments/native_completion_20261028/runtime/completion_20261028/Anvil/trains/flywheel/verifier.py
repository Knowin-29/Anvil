#!/usr/bin/env python3
"""Unified rule verifier (converges the two prior conflicting implementations).

Six deterministic signals in [0,1]; equal-weight mean is the sample score.
Decision (single documented threshold set, replacing the two prior schemes):
  mean >= 0.70 and no signal < 0.30  -> verified
  mean <  0.50 or  any signal < 0.30 -> rejected
  otherwise                          -> needs_review

Signals: schema, geometry, cross_reference (uses L1 result if present),
hallucination (bbox not near-full-image), format, semantic (per-dim sanity).
"""
from __future__ import annotations

from typing import Any

from trains.flywheel.contracts import CONTRACT_VERSION, FORM_SPECS, groundings, primary_grounding
from trains.flywheel.validators import validate

SIGNALS = ["schema", "geometry", "cross_reference", "hallucination", "format", "semantic"]


def _bbox_ok(b) -> bool:
    return (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) for v in b)
            and 0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000)


def _area_frac(b) -> float:
    return ((b[2] - b[0]) * (b[3] - b[1])) / 1_000_000.0


def score(rec: dict[str, Any]) -> dict[str, Any]:
    dim = rec.get("dimension", rec.get("pipeline", ""))
    s = {k: 1.0 for k in SIGNALS}

    # schema: required keys present per dim
    req = {"G_grounding": ["object_class", "bbox"],
           "S_spatial": ["subject", "predicate", "object"],
           "A_identity_attr": ["object_class", "bbox", "attributes"],
           "A_obj_state": ["object_class", "state_type", "state_value"],
           "A_action_precondition": ["object_class", "action_type", "can_do_now"],
           "A_robot_state": ["state_type", "state_value"],
           "A_action_interface": ["object_class", "bbox", "grasp_point_2d"]}.get(dim, [])
    if req and not all(k in rec and rec[k] not in (None, "", []) for k in req):
        s["schema"] = 0.0

    # geometry: bbox / points valid where present
    b = rec.get("bbox")
    if b is not None:
        s["geometry"] = 1.0 if _bbox_ok(b) else 0.0
        # hallucination: near-full-image bbox is suspect
        if _bbox_ok(b) and _area_frac(b) > 0.85:
            s["hallucination"] = 0.2
    gp = rec.get("grasp_point_2d")
    if gp is not None and _bbox_ok(b):
        inside = b[0] - 30 <= gp[0] <= b[2] + 30 and b[1] - 30 <= gp[1] <= b[3] + 30
        s["geometry"] = min(s["geometry"], 1.0 if inside else 0.0)

    # cross_reference: use L1 result if attached
    cc = rec.get("_cross_check")
    if isinstance(cc, dict):
        s["cross_reference"] = 1.0 if cc.get("match", True) else 0.0

    # format: confidence in range, no obviously templated junk
    conf = rec.get("confidence", 1.0)
    if not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
        s["format"] = 0.5

    # semantic: per-dim sanity
    if dim == "A_action_interface" and rec.get("jaw_axis_2d") in ([1, 0], [0, 0]):
        s["semantic"] = 0.4  # templated / degenerate axis
    if dim == "S_spatial" and rec.get("subject") == rec.get("object"):
        s["semantic"] = 0.0
    if dim == "A_obj_state" and str(rec.get("state_value", "")).lower() == "unknown":
        s["semantic"] = 0.3

    mean = sum(s.values()) / len(s)
    if mean >= 0.70 and min(s.values()) >= 0.30:
        verdict = "verified"
    elif mean < 0.50 or min(s.values()) < 0.30:
        verdict = "rejected"
    else:
        verdict = "needs_review"
    return {"signals": s, "mean": round(mean, 4), "verdict": verdict}


def score_form(form: str, payload, context: dict, cross_check=None) -> dict[str, Any]:
    """Audit a v2 multi-form record without pretending every form is grounding."""
    ok, reason = validate(form, payload, context)
    spec = FORM_SPECS.get(form)
    context_ok = spec is not None
    if spec:
        need = spec["context"]
        if need == "primary_grounding":
            context_ok = bool(primary_grounding(context))
        elif need == "multiple_groundings":
            context_ok = len(groundings(context)) >= 2
        elif need == "grounding_and_spatial":
            context_ok = bool(primary_grounding(context) and context.get("spatial"))

    checks = cross_check if isinstance(cross_check, list) else []
    cross_ok = bool(checks) and all(
        isinstance(item, dict) and item.get("match") is True for item in checks
    )
    signals = {
        "contract": 1.0 if ok else 0.0,
        "cascade_context": 1.0 if context_ok else 0.0,
        "grounding_cross_check": 1.0 if cross_ok else 0.0,
    }
    verified = all(value == 1.0 for value in signals.values())
    return {
        "contract_version": CONTRACT_VERSION,
        "form": form,
        "family": spec["family"] if spec else None,
        "signals": signals,
        "contract_reason": reason,
        "verdict": "verified" if verified else "rejected",
    }
