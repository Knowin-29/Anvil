from apps.data_flywheel.gsa.a_action_interface_triage import (
    object_triage_class,
    triage_candidates,
)


def _candidate(label_id, object_name, source="vlm", geometry_valid=True):
    verifier_status = "review_required" if source == "vlm" else "heuristic_only"
    label_status = "teacher_generated" if source == "vlm" else "candidate"
    point = [5.0, 5.0] if geometry_valid else [50.0, 50.0]
    return {
        "label_id": label_id,
        "schema_version": "a_action_interface_candidate_v1",
        "sample_id": "sample_001",
        "image_path": "tests/fixtures/desk_sample.jpg",
        "image_width": 160,
        "image_height": 120,
        "object_id": f"{object_name}_0",
        "object_name": object_name,
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "bucket": "A_action_interface",
        "subtype": "grasp_point",
        "action_type": "pick",
        "label_type": "point",
        "source_provenance": source,
        "verifier_status": verifier_status,
        "label_status": label_status,
        "stable_training_label": False,
        "grasp_point_2d": point,
        "jaw_axis_2d": [1.0, 0.0],
        "risk_flags": [],
    }


def test_object_triage_class_prefers_manipulable_and_filters_static_objects():
    assert object_triage_class("tomato sauce can") == "preferred_manipulable"
    assert object_triage_class("metal bowl") == "preferred_manipulable"
    assert object_triage_class("orange object") == "unknown"
    assert object_triage_class("faucet") == "filtered_static_or_background"
    assert object_triage_class("stove burner") == "filtered_static_or_background"


def test_triage_candidates_selects_vlm_geometry_valid_manipulable_candidates():
    rows = [
        _candidate("aif_can", "can"),
        _candidate("aif_bowl", "bowl"),
        _candidate("aif_adapter", "spoon", source="adapter"),
        _candidate("aif_faucet", "faucet"),
        _candidate("aif_invalid", "fork", geometry_valid=False),
    ]

    selected, summary = triage_candidates(rows, max_candidates=10)

    assert [row["label_id"] for row in selected] == ["aif_bowl", "aif_can"]
    assert summary["selected_count"] == 2
    assert summary["filtered_reasons"]["source_not_vlm"] == 1
    assert summary["filtered_reasons"]["filtered_static_or_background_object"] == 1
    assert summary["filtered_reasons"]["invalid_geometry"] == 1
