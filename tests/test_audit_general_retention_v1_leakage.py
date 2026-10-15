from __future__ import annotations

from tools.audit_general_retention_v1_leakage import (
    BKNode,
    normalized_question,
    resolve_training_path,
    source_id_collisions,
)


def test_bk_tree_returns_only_values_within_hamming_radius() -> None:
    root = BKNode(0b0000, [0])
    root.add(0b0011, 1)
    root.add(0b1111, 2)
    observed = sorted(root.query(0b0001, 1))
    assert observed == [(1, 0), (1, 1)]


def test_training_path_remap_is_prefix_scoped() -> None:
    assert str(resolve_training_path("/mnt/data/a.jpg")) == "/path/to/datasets/a.jpg"
    assert str(resolve_training_path("/other/mnt/data/a.jpg")) == "/other/mnt/data/a.jpg"


def test_question_normalization_is_case_and_punctuation_stable() -> None:
    assert normalized_question("What IS this?") == "what is this"


def test_dataset_qualified_source_id_is_a_hard_collision() -> None:
    evaluations = [{"dataset": "ai2d_test", "image_id": "2669"}]
    training = [{
        "raw_path": "/mnt/data/AI2D/images/2669.png",
        "path": "/path/to/datasets/AI2D/images/2669.png",
    }]
    observed = source_id_collisions(evaluations, training, {})
    assert len(observed) == 1
    assert observed[0]["eval_image_id"] == "2669"


def test_unqualified_numeric_basename_is_not_a_source_id_collision() -> None:
    evaluations = [{"dataset": "ai2d_test", "image_id": "2669"}]
    training = [{
        "raw_path": "/mnt/data/unrelated/2669.png",
        "path": "/path/to/datasets/unrelated/2669.png",
    }]
    assert source_id_collisions(evaluations, training, {}) == []
