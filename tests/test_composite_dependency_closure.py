from __future__ import annotations

from copy import deepcopy

from trains import materialize_semantic_audit_composite as materializer


def _row(form, sample_id, payload, context):
    return {
        "sample_id": sample_id,
        "task_form": form,
        "payload": payload,
        "gsa_context": context,
    }


def _cascade_state():
    bbox_payload = [{"label": "cup", "bbox_2d": [100, 100, 500, 600]}]
    detection_payload = [
        {"label": "cup", "bbox_2d": [100, 100, 500, 600]},
        {"label": "spoon", "bbox_2d": [600, 200, 800, 500]},
    ]
    depth_payload = [
        {"label": "cup", "rank": 1, "relative_depth": 0.0},
        {"label": "spoon", "rank": 2, "relative_depth": 1.0},
    ]
    affordance_payload = [{
        "label": "handle",
        "affordance": "grasp",
        "bbox_2d": [350, 250, 490, 450],
        "contact_point_2d": [420, 350],
    }]
    dof_payload = [{
        "label": "cup",
        "grasp_point_2d": [420, 350],
        "jaw_axis_2d": [0.866, 0.5],
        "approach": "top",
        "wrist_deg": 30,
    }]
    groundings = [
        {"object_class": "cup", "bbox": [100, 100, 500, 600]},
        {"object_class": "spoon", "bbox": [600, 200, 800, 500]},
    ]

    def context(**spatial):
        return {
            "primary": deepcopy(groundings[0]),
            "groundings": deepcopy(groundings),
            "spatial": deepcopy(spatial),
        }

    rows = {
        "bbox": _row("bbox", "sid-bbox", bbox_payload, context()),
        "detection": _row("detection", "sid-detection", detection_payload, context()),
        "depth": _row("depth", "sid-depth", depth_payload, context(depth_order=depth_payload)),
        "affordance": _row(
            "affordance", "sid-affordance", affordance_payload,
            context(depth_order=depth_payload),
        ),
        "dof": _row(
            "dof", "sid-dof", dof_payload,
            context(depth_order=depth_payload, affordance=affordance_payload),
        ),
        "action_cot": _row(
            "action_cot", "sid-action",
            [{
                "goal": "Lift the cup",
                "steps": ["Approach the handle carefully", "Grasp and lift the cup"],
                "target_point_2d": [420, 350],
                "preconditions": ["The handle is visible"],
                "failure_modes": ["The grasp may slip"],
            }],
            context(
                depth_order=depth_payload,
                affordance=affordance_payload,
                dof=dof_payload,
            ),
        ),
    }
    sample_ids = {form: row["sample_id"] for form, row in rows.items()}
    entries = {
        row["sample_id"]: {
            "decision": True,
            "form": form,
            "item": None,
            "label_sha256": f"label-hash-{form}",
            "row": row,
        }
        for form, row in rows.items()
    }
    state = {
        "manifest": {
            "images": 1,
            "forms": list(rows),
            "selected": [{"sample_ids": sample_ids}],
        },
        "entries": entries,
    }
    candidates = [
        rows["bbox"], rows["detection"], rows["affordance"],
        rows["dof"], rows["action_cot"],
    ]
    return rows, state, candidates


def test_dependency_closure_releases_a_without_releasing_v3_depth():
    rows, state, candidates = _cascade_state()
    released, bindings, stats = materializer._dependency_closed_v3_rows(
        candidates, state
    )
    assert {row["task_form"] for row in released} == {
        "bbox", "detection", "affordance", "dof", "action_cot"
    }
    assert rows["depth"] not in released
    assert set(bindings["sid-affordance"]) == {"bbox", "detection", "depth"}
    assert set(bindings["sid-dof"]) == {
        "bbox", "detection", "depth", "affordance"
    }
    assert set(bindings["sid-action"]) == {
        "bbox", "detection", "depth", "affordance", "dof"
    }
    assert bindings["sid-action"]["depth"]["sample_id"] == "sid-depth"
    assert bindings["sid-action"]["depth"]["payload_sha256"] == (
        materializer._content_hash(rows["depth"]["payload"])
    )
    assert stats["released_by_form"] == {
        "affordance": 1, "dof": 1, "action_cot": 1
    }


def test_dependency_context_mismatch_blocks_downstream_cascade():
    _, state, candidates = _cascade_state()
    candidates = deepcopy(candidates)
    affordance = next(row for row in candidates if row["task_form"] == "affordance")
    affordance["gsa_context"]["spatial"]["depth_order"][0]["rank"] = 2
    released, bindings, stats = materializer._dependency_closed_v3_rows(
        candidates, state
    )
    assert {row["task_form"] for row in released} == {"bbox", "detection"}
    assert bindings == {}
    assert stats["excluded_by_reason"] == {
        "affordance:context_mismatch:depth": 1,
        "dof:dependency_not_closed:affordance": 1,
        "action_cot:dependency_not_closed:affordance": 1,
    }


def test_nonfinal_v3_depth_blocks_a_but_never_blocks_g():
    _, state, candidates = _cascade_state()
    state["entries"]["sid-depth"]["decision"] = None
    released, bindings, stats = materializer._dependency_closed_v3_rows(
        candidates, state
    )
    assert {row["task_form"] for row in released} == {"bbox", "detection"}
    assert bindings == {}
    assert stats["excluded_by_reason"] == {
        "affordance:dependency_not_human_true:depth": 1,
        "dof:dependency_not_human_true:depth": 1,
        "action_cot:dependency_not_human_true:depth": 1,
    }


def test_strict_invalid_v3_depth_blocks_a_but_never_blocks_g():
    rows, state, candidates = _cascade_state()
    rows["depth"]["payload"][1]["rank"] = 1
    released, bindings, stats = materializer._dependency_closed_v3_rows(
        candidates, state
    )
    assert {row["task_form"] for row in released} == {"bbox", "detection"}
    assert bindings == {}
    assert stats["excluded_by_reason"] == {
        "affordance:dependency_not_strict:depth:duplicate_rank": 1,
        "dof:dependency_not_strict:depth:duplicate_rank": 1,
        "action_cot:dependency_not_strict:depth:duplicate_rank": 1,
    }


def test_accepted_rows_use_strict_validator_and_reject_audit_only():
    context = {
        "primary": {"object_class": "cup", "bbox": [100, 100, 500, 600]},
        "groundings": [{"object_class": "cup", "bbox": [100, 100, 500, 600]}],
        "spatial": {},
    }
    invalid_affordance = _row(
        "affordance", "sid-affordance",
        [{
            "label": "cup",
            "affordance": "grasp",
            "bbox_2d": [100, 100, 500, 600],
            "contact_point_2d": [300, 300],
        }],
        context,
    )
    audit_only_point = _row(
        "point", "sid-point", [{"label": "cup", "point_2d": [300, 300]}], context
    )
    audit_only_point["_audit_only"] = True
    state = {
        "manifest": {
            "images": 1,
            "forms": ["affordance", "point"],
            "selected": [{
                "sample_ids": {
                    "affordance": "sid-affordance",
                    "point": "sid-point",
                }
            }],
        },
        "entries": {
            "sid-affordance": {"decision": True, "row": invalid_affordance},
            "sid-point": {"decision": True, "row": audit_only_point},
        },
    }
    accepted, stats = materializer._accepted_rows(
        state, {"affordance", "point"}
    )
    assert accepted == []
    assert stats["strict_invalid:affordance:part_label_equals_object"] == 1
    assert stats["audit_only"] == 1
