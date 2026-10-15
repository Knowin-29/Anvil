from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import audit_aligned_a_panel_capacity as audit


def test_depth_referent_binding_distinguishes_duplicate_categories():
    row = {
        "payload": [
            {"label": "cup@[1, 2, 3, 4]", "rank": 1, "relative_depth": 0.0},
            {"label": "cup@[5, 6, 7, 8]", "rank": 2, "relative_depth": 1.0},
        ],
        "gsa_context": {"groundings": [
            {"object_class": "cup", "bbox": [1, 2, 3, 4]},
            {"object_class": "cup", "bbox": [5, 6, 7, 8]},
        ]},
    }
    assert audit.depth_referents_are_declared_one_to_one(row)
    row["payload"][0]["label"] = "cup"
    assert not audit.depth_referents_are_declared_one_to_one(row)


def test_source_episode_key_uses_episode_not_individual_frame():
    first = "/data/FloorPlan9_pickup_0_a/0_init.png"
    second = "/data/FloorPlan9_pickup_0_a/12_observe.png"
    assert audit.source_episode_key(first) == audit.source_episode_key(second)
    assert audit.source_episode_key(first) == "FloorPlan9_pickup_0_a"


def test_repository_capacity_inventory_is_fail_closed():
    args = audit.parse_args([])
    report = audit.audit_capacity(args)
    assert report["integrity_verdict"] == "PASS"
    assert report["identity_checks"] == {
        "A_base_context_expected": 360,
        "A_base_context_passed": 360,
        "A_chain_context_expected": 120,
        "A_chain_context_passed": 120,
        "annotation_snapshot_expected": 640,
        "annotation_snapshot_passed": 640,
        "base_self_context_expected": 120,
        "base_self_context_passed": 120,
        "image_hash_expected": 40,
        "image_hash_passed": 40,
        "label_content_hash_expected": 320,
        "label_content_hash_passed": 320,
        "source_manifest_binding_expected": 40,
        "source_manifest_binding_passed": 40,
    }
    assert report["verdict"] == "FORMAL_PANEL_BLOCKED_NO_HUMAN_GT"
    assert report["v3_inventory"]["images"] == 40
    assert report["v3_inventory"]["A_units"] == 120
    assert report["v3_inventory"]["A_disagreements_adjudicated"] == 11
    assert report["capacity"]["legacy_base_GS_candidate_images"] == 22
    assert report["capacity"]["legacy_base_GS_candidates_by_form"] == {
        "affordance": 15,
        "DoF": 13,
        "action_plan": 21,
    }
    assert report["capacity"]["strict_sequential_release_images"] == 12
    assert report["capacity"]["strict_sequential_release_by_form"] == {
        "affordance": 12,
        "DoF": 6,
        "action_plan": 6,
    }
    assert report["capacity"]["strict_complete_A_triples"] == 6
    assert report["capacity"]["formal_human_GT_scenes"] == 0
    assert report["capacity"]["paired_oracle_predicted_A_scenes"] == 0
    assert report["capacity"]["paired_oracle_predicted_corrupted_A_scenes"] == 0
    assert report["capacity"]["formal_scene_shortfall"] == 40
    assert report["leakage"]["legacy_22_exact_image_exposure"] == {
        "pilot_train": 20,
        "pilot_val": 2,
        "g_train": 13,
        "g_dev": 6,
        "g_qualification": 3,
    }
    source_clean = report["source_clean_candidate_inventory"]
    assert source_clean["images"] == 141
    assert source_clean["image_disjoint_from_pilot_train_val"] == 7
    assert source_clean["source_episode_disjoint_from_pilot_train_val"] == 6
    assert source_clean["source_episode_disjoint_candidate_shortfall"] == 34
    assert (
        source_clean[
            "source_episode_disjoint_from_all_registered_pilot_G_exposure"
        ]
        == 5
    )
    assert source_clean["fully_unexposed_candidate_shortfall"] == 35
    assert source_clean["fully_unexposed_with_all_three_generated_A_forms"] == 1
    assert (
        source_clean[
            "fully_unexposed_with_all_three_non_audit_generated_A_forms"
        ]
        == 0
    )
    assert report["human_packet"]["incremental_A_scored_units"] == 120
    assert report["human_packet"]["incremental_A_or_B_records"] == 240
    assert report["human_packet"]["existing_vote_reuse_as_human_GT"] is False


def test_required_formal_scene_gate_returns_two(monkeypatch):
    fake = {
        "integrity_verdict": "PASS",
        "capacity": {"formal_human_GT_scenes": 0},
    }
    monkeypatch.setattr(audit, "audit_capacity", lambda args: fake)
    assert audit.main(["--require-formal-scenes", "40"]) == 2
