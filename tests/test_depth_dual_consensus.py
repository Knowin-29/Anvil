from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import join_depth_probe_consensus_v1 as dual


def _entity(label: str, value: float, uncertainty: float) -> dict:
    return {"label": label, "median_value": value, "uncertainty": uncertainty}


def test_model_pairwise_probe_respects_depth_semantics():
    metric = dual.model_pairwise_probes(
        "a" * 64,
        [_entity("near", 1.0, 0.01), _entity("far", 2.0, 0.01)],
        "lower",
    )[0]
    inverse = dual.model_pairwise_probes(
        "a" * 64,
        [_entity("near", 2.0, 0.01), _entity("far", 1.0, 0.01)],
        "higher",
    )[0]
    assert metric["raw_nearer"] == inverse["raw_nearer"] == "near"
    assert metric["within_model_reliable"] is True
    assert inverse["within_model_reliable"] is True


def test_legacy_log_binding_hashes_exact_bytes(tmp_path: Path):
    log = tmp_path / "probe.log"
    payload = b'{"event":"start"}\r\nprobe_rc=0\r\n'
    log.write_bytes(payload)
    events, digest = dual._json_events(log)
    assert events == [{"event": "start"}]
    assert digest == hashlib.sha256(payload).hexdigest()


def _view(model_id: str, nearer: str, reliable: bool = True) -> dict:
    farther = "b" if nearer == "a" else "a"
    row = {
        "image_sha256": "f" * 64,
        "pair_key": ["a", "b"],
        "raw_nearer": nearer,
        "raw_farther": farther,
        "within_model_reliable": reliable,
    }
    return {"model_id": model_id, "pairs": [dict(row) for _ in range(40)]}


def test_join_consensus_never_sets_automatic_pass():
    first = _view("first", "a")
    second = _view("second", "a")
    for index, row in enumerate(first["pairs"]):
        row["image_sha256"] = f"{index:064x}"
    for index, row in enumerate(second["pairs"]):
        row["image_sha256"] = f"{index:064x}"
    joined = dual.join_consensus(first, second)
    assert len(joined) == 40
    assert all(row["audit_candidate"] for row in joined)
    assert all(row["automatic_pass"] is False for row in joined)
    assert all(row["status"] == "human_audit_candidate_only" for row in joined)


def test_join_consensus_rejects_disagreement_or_weak_model():
    first = _view("first", "a")
    second = _view("second", "b")
    for index, row in enumerate(first["pairs"]):
        row["image_sha256"] = f"{index:064x}"
    for index, row in enumerate(second["pairs"]):
        row["image_sha256"] = f"{index:064x}"
    joined = dual.join_consensus(first, second)
    assert not any(row["audit_candidate"] for row in joined)

    second = _view("second", "a", reliable=False)
    for index, row in enumerate(second["pairs"]):
        row["image_sha256"] = f"{index:064x}"
    joined = dual.join_consensus(first, second)
    assert not any(row["audit_candidate"] for row in joined)


def test_model_pairwise_probe_rejects_bad_direction_rule():
    with pytest.raises(ValueError, match="nearer_rule"):
        dual.model_pairwise_probes(
            "a" * 64,
            [_entity("a", 1.0, 0.1), _entity("b", 2.0, 0.1)],
            "sideways",
        )
