#!/usr/bin/env python3
"""Single source of truth for GSA task identity and cascade context."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "gsa-contract-v2"
TRAINING_INSTRUCTION_VERSION = "anvil-referent-context-v1"

FORM_SPECS = {
    "bbox": {"family": "G", "context": "none"},
    "detection": {"family": "G", "context": "none"},
    "mask": {"family": "G", "context": "primary_grounding"},
    "point": {"family": "G", "context": "primary_grounding"},
    "depth": {"family": "S", "context": "multiple_groundings"},
    "affordance": {"family": "A", "context": "grounding_and_spatial"},
    "dof": {"family": "A", "context": "grounding_and_spatial"},
    "action_cot": {"family": "A", "context": "grounding_and_spatial"},
}


def image_sha256(path: str) -> str:
    """Hash image bytes once so split identity does not depend on mount paths."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grounding(label: str, bbox: list[int]) -> dict[str, Any]:
    return {"object_class": str(label).strip(), "bbox": [int(v) for v in bbox]}


def primary_grounding(ctx: Any) -> dict[str, Any]:
    """Accept v2 cascade context and legacy single-grounding context."""
    if not isinstance(ctx, dict):
        return {}
    primary = ctx.get("primary")
    if isinstance(primary, dict):
        return primary
    if "bbox" in ctx:
        return ctx
    items = ctx.get("groundings")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0]
    return {}


def groundings(ctx: Any) -> list[dict[str, Any]]:
    if not isinstance(ctx, dict):
        return []
    items = ctx.get("groundings")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    primary = primary_grounding(ctx)
    return [primary] if primary else []


def context_for(form: str, g_items: list[dict[str, Any]], spatial: dict | None = None) -> dict:
    if form not in FORM_SPECS:
        raise KeyError(f"unknown GSA form: {form}")
    return {
        "contract_version": CONTRACT_VERSION,
        "primary": g_items[0] if g_items else None,
        "groundings": g_items,
        "spatial": spatial or {},
    }


def training_instruction(form: str, context: dict) -> str:
    """Render the actual task conditioning, excluding its current-stage target."""
    if form not in FORM_SPECS:
        raise ValueError(f"unknown GSA form: {form}")
    suffix = " Return only a JSON array; image coordinates use the 0–1000 scale."
    if form == "detection":
        return "Detect two to six distinct manipulable entities and output their labelled bounding boxes." + suffix
    primary = primary_grounding(context)
    label = str(primary.get("object_class") or "").strip()
    if not label:
        raise ValueError(f"{form} requires a named referent")
    referent = json.dumps(label, ensure_ascii=False)
    if form in {"bbox", "point", "mask"}:
        request = {
            "bbox": "Locate the object named {name} and output its labelled bounding box.",
            "point": "Output a labelled point on the visible object named {name}.",
            "mask": "Trace the visible silhouette of the object named {name} as a labelled polygon.",
        }[form]
        # For grounding targets, do not give away their reference boxes/points.
        return request.format(name=referent) + suffix
    entities = [
        {key: item[key] for key in ("object_class", "bbox") if key in item}
        for item in groundings(context)
    ]
    if form == "depth":
        if len(entities) < 2:
            raise ValueError("depth requires multiple grounded entities")
        # spatial.depth_order is the answer here and must never enter the prompt.
        return ("Order these grounded entities from nearest to farthest in camera depth: "
                + json.dumps(entities, ensure_ascii=False)
                + ". Output labels, ranks and ordinal relative_depth values." + suffix)
    spatial = context.get("spatial") or {}
    if not spatial.get("depth_order"):
        raise ValueError(f"{form} requires upstream spatial context")
    upstream = {"groundings": entities, "depth_order": spatial["depth_order"]}
    if form in {"dof", "action_cot"}:
        if not spatial.get("affordance"):
            raise ValueError(f"{form} requires upstream affordance context")
        upstream["affordance"] = spatial["affordance"]
    if form == "action_cot":
        if not spatial.get("dof"):
            raise ValueError("action_cot requires upstream grasp context")
        upstream["dof"] = spatial["dof"]
    request = {
        "affordance": "Identify a usable contact sub-part, its action, region and contact point",
        "dof": "Specify a parallel-jaw grasp point, jaw axis, approach and wrist angle",
        "action_cot": "Plan a physically appropriate manipulation with steps, preconditions, failure modes and target point",
    }[form]
    return (request + " for the object named " + referent + ". Upstream context: "
            + json.dumps(upstream, ensure_ascii=False) + "." + suffix)


def task_columns(form: str, context: dict, *, source: str, scene: str,
                 image_id: str) -> dict[str, Any]:
    """Top-level columns are preserved by ms-swift and passed to reward ORMs."""
    spec = FORM_SPECS[form]
    return {
        "gsa_form": form,
        "gsa_family": spec["family"],
        "gsa_contract_version": CONTRACT_VERSION,
        "gsa_context": context,
        "gsa_meta": {
            "form": form,
            "family": spec["family"],
            "source": source,
            "scene": scene,
            "image_sha256": image_id,
            "contract_version": CONTRACT_VERSION,
        },
    }


def validate_task_columns(record: dict[str, Any]) -> tuple[bool, str]:
    form = record.get("gsa_form")
    if form not in FORM_SPECS:
        return False, "missing_or_unknown_gsa_form"
    if record.get("gsa_contract_version") != CONTRACT_VERSION:
        return False, "contract_version_mismatch"
    ctx = record.get("gsa_context")
    need = FORM_SPECS[form]["context"]
    if need == "primary_grounding" and not primary_grounding(ctx):
        return False, "missing_primary_grounding"
    if need == "multiple_groundings" and len(groundings(ctx)) < 2:
        return False, "missing_multiple_groundings"
    if need == "grounding_and_spatial":
        if not primary_grounding(ctx):
            return False, "missing_primary_grounding"
        if not isinstance(ctx, dict) or not ctx.get("spatial"):
            return False, "missing_spatial_context"
    return True, ""
