from __future__ import annotations

from tools.audit_multibench_leakage import (
    adjudicate_near_candidates,
    benchmark_breakdown,
    determine_state,
    extract_referents,
    normalize_text,
    refspatial_location_breakdown,
    source_and_scene_collisions,
)


def test_text_normalization_and_referent_extraction() -> None:
    assert normalize_text("<image> Pick-UP: Mug #2!") == "pick up mug 2"
    assert extract_referents('[{"label":"Blue Mug"},{"object":"Drawer"}]') == [
        "blue mug", "drawer"
    ]


def test_extra_benchmark_preserves_inputs_and_does_not_invent_scene_ids(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from eval.gsa_eval import registry
    from eval.gsa_eval.datasets import Sample
    from tools.audit_multibench_leakage import load_extra_eval_records
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "red").save(path)
    sample = Sample("released-row", "Which object?", [str(path)], "A", group="depth")
    monkeypatch.setattr(registry, "get_spec", lambda _: SimpleNamespace(
        available=True, n_expected=1, loader=lambda _: [sample]))
    rows, assets, counts = load_extra_eval_records(["tiny"], tmp_path)
    assert counts == {"tiny": 1} and assets == [path]
    assert rows[0]["uid"] == sample.uid
    assert rows[0]["source_ids"] == rows[0]["scene_ids"] == []
    assert rows[0]["scene_identity_status"] == "NOT_PROVIDED_BY_RELEASE"


def test_extra_benchmark_rejects_missing_rows(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from eval.gsa_eval import registry
    from tools.audit_multibench_leakage import FullLeakageError, load_extra_eval_records
    monkeypatch.setattr(registry, "get_spec", lambda _: SimpleNamespace(
        available=True, n_expected=2, loader=lambda _: []))
    with pytest.raises(FullLeakageError, match="coverage/UID"):
        load_extra_eval_records(["incomplete"], tmp_path)


def test_state_is_fail_closed() -> None:
    assert determine_state(hard_leak=True, unresolved_near=0, metadata_blockers=0) == "LEAKAGE_DETECTED"
    assert determine_state(hard_leak=False, unresolved_near=1, metadata_blockers=0) == "BLOCKED"
    assert determine_state(hard_leak=False, unresolved_near=0, metadata_blockers=1) == "BLOCKED"
    assert determine_state(hard_leak=False, unresolved_near=0, metadata_blockers=0) == "PASS"


def test_where2place_source_id_is_dataset_qualified_hard_collision() -> None:
    evaluation = [{
        "dataset": "where2place",
        "uid": "0",
        "source_ids": ["where2place 00 jpg"],
        "scene_ids": [],
        "source_metadata": {"image": "00.jpg"},
    }]
    training = [{
        "raw_path": "/dataset/where2place/images/00.jpg",
        "path": "/dataset/where2place/images/00.jpg",
    }]
    source, scene = source_and_scene_collisions(
        evaluation, training, {training[0]["raw_path"]: {"where2place"}}
    )
    assert len(source) == 1
    assert scene == []


def test_near_duplicate_adjudication_requires_complete_keyed_decisions() -> None:
    candidates = [{"pair_id": "a"}, {"pair_id": "b"}]
    unresolved, leaks, cleared = adjudicate_near_candidates(
        candidates,
        {"a": {"pair_id": "a", "decision": "NOT_LEAK", "reviewer": "r", "comment": "different"}},
    )
    assert [row["pair_id"] for row in unresolved] == ["b"]
    assert leaks == []
    assert [row["pair_id"] for row in cleared] == ["a"]


def test_benchmark_breakdown_deduplicates_hash_axes_but_keeps_state_strict() -> None:
    evaluation = [{
        "dataset": "bench",
        "uid": "u",
        "sha256": "eval",
        "source_identity_status": "PROVIDED_BY_RELEASE",
        "scene_identity_status": "PROVIDED_BY_RELEASE",
    }]
    collision = {
        "eval_dataset": "bench",
        "eval_datasets": ["bench"],
        "eval_sha256": "eval",
        "training_sha256": "train",
    }
    observed = benchmark_breakdown(
        dataset="bench",
        eval_records=evaluation,
        collisions={
            "encoded_bytes": [collision],
            "decoded_rgb": [collision],
            "pillow_default_jpeg": [collision],
        },
        source_hits=[],
        scene_hits=[],
        text_hard=[],
        near=[],
        unresolved_near=[],
        near_leaks=[],
    )
    assert observed["state"] == "LEAKAGE_DETECTED"
    assert observed["encoded_bytes"] == {
        "pairs": 1, "eval_images": 1, "eval_records": 1
    }


def test_refspatial_location_projection_ignores_other_subgroups() -> None:
    records = [
        {"dataset": "refspatial_bench", "uid": "location_0", "sha256": "location"},
        {"dataset": "refspatial_bench", "uid": "placement_0", "sha256": "placement"},
    ]
    exact_other = {
        "eval_dataset": "refspatial_bench",
        "eval_datasets": ["refspatial_bench"],
        "eval_uids": ["placement_0"],
        "eval_sha256": "placement",
        "training_sha256": "train-a",
    }
    near_location = {
        "eval_dataset": "refspatial_bench",
        "eval_datasets": ["refspatial_bench"],
        "eval_uids": ["location_0"],
        "eval_sha256": "location",
        "training_sha256": "train-b",
    }
    observed = refspatial_location_breakdown(
        eval_records=records,
        collisions={
            "encoded_bytes": [exact_other],
            "decoded_rgb": [exact_other],
            "pillow_default_jpeg": [exact_other],
        },
        near=[near_location],
        unresolved_near=[near_location],
        near_leaks=[],
    )
    assert observed["encoded_bytes"]["pairs"] == 0
    assert observed["perceptual_unresolved"] == 1
    assert observed["state"] == "BLOCKED"
