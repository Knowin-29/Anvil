from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import join_depth_probe_consensus_v2 as dual
from trains.flywheel.build_depth_semantic_audit_pack import content_hash, file_hash


def _view(model_id: str, nearer: str, reliable: bool = True) -> dict:
    farther = "b" if nearer == "a" else "a"
    pairs = []
    for index in range(40):
        pairs.append({
            "image_sha256": f"{index:064x}",
            "pair_key": ["a", "b"],
            "raw_nearer": nearer,
            "raw_farther": farther,
            "within_model_reliable": reliable,
        })
    return {"model_id": model_id, "pairs": pairs}


def test_formal_join_is_only_sensor_depth_anomaly_input():
    joined = dual.join_formal_views(_view("da2", "a"), _view("dpt", "a"))
    assert len(joined) == 40
    assert all(row["sensor_depth_anomaly_check_candidate"] for row in joined)
    assert all(row["automatic_label"] is False for row in joined)
    assert all(row["automatic_pass"] is False for row in joined)
    assert all(row["ground_truth_eligible"] is False for row in joined)
    assert all(
        row["status"] == "compare_only_after_scannet_sensor_depth_label_exists"
        for row in joined
    )


def test_formal_join_rejects_disagreement_and_weak_view():
    disagree = dual.join_formal_views(_view("da2", "a"), _view("dpt", "b"))
    assert not any(row["sensor_depth_anomaly_check_candidate"] for row in disagree)
    weak = dual.join_formal_views(
        _view("da2", "a"), _view("dpt", "a", reliable=False)
    )
    assert not any(row["sensor_depth_anomaly_check_candidate"] for row in weak)


def _write_formal_fixture(
    root: Path,
    *,
    protocol: str,
    model_manifest: dict,
    dpt: bool,
) -> Path:
    root.mkdir()
    rows = []
    image_hashes = list(dual.FROZEN_CANARY_SHA256S)
    for index in range(40):
        image_sha = image_hashes[index % len(image_hashes)]
        rows.append({
            "image_sha256": image_sha,
            "entities": [f"a-{index}", f"b-{index}"],
            "raw_nearer": f"a-{index}",
            "raw_farther": f"b-{index}",
            "decision": "directional_probe",
            "nearer": f"a-{index}",
            "farther": f"b-{index}",
        })
    pair_path = root / "pairwise_probe.jsonl"
    pair_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    safety = {
        "images_uploaded": 0,
        "network_enabled": False,
        "writes_training_records": False,
        "writes_semantic_audit_records": False,
    }
    if dpt:
        safety["writes_scannet_records"] = False
    trace_path = ROOT / "data/depth_v43_pairwise/canary_v1/generation_report.json"
    _, _, run_binding = dual.validate_frozen_v43_trace(trace_path)
    report = {
        "protocol": protocol,
        "completed": True,
        "candidate_images": 8,
        "candidate_set_sha256": dual.FROZEN_CANARY_SET_SHA256,
        "planned_pairs": 40,
        "external_uploads": 0,
        "training_eligible": False,
        "semantic_audit_eligible": False,
        "automatic_pass": False,
        "model": model_manifest,
        "safety": safety,
        "v43_trace": {
            "report_path": str(trace_path.resolve()),
            "report_sha256": file_hash(trace_path),
            **run_binding,
        },
        "pairs": {
            "path": pair_path.name,
            "rows": 40,
            "sha256": file_hash(pair_path),
            "content_sha256": content_hash(rows),
        },
    }
    if dpt:
        report.update({
            "intended_use": dual.INTENDED_USE,
            "ground_truth_eligible": False,
        })
    report_path = root / "probe_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_formal_probe_normalization_binds_pair_file(tmp_path: Path):
    manifest = {"repo_id": "fixture"}
    report_path = _write_formal_fixture(
        tmp_path / "dpt",
        protocol=dual.DPT_PROTOCOL,
        model_manifest=manifest,
        dpt=True,
    )
    view = dual.normalize_formal_probe(
        report_path,
        model_id="dpt_hybrid_midas",
        expected_protocol=dual.DPT_PROTOCOL,
        current_model_manifest=manifest,
    )
    assert len(view["pairs"]) == 40
    assert view["ground_truth_eligible"] is False

    pair_path = report_path.parent / "pairwise_probe.jsonl"
    pair_path.write_text(pair_path.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="hash or row count mismatch"):
        dual.normalize_formal_probe(
            report_path,
            model_id="dpt_hybrid_midas",
            expected_protocol=dual.DPT_PROTOCOL,
            current_model_manifest=manifest,
        )
