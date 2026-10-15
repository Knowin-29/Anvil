from pathlib import Path

import pytest

from apps.data_flywheel.gsa.data_adapters.a_action_interface_offline_probe import (
    BRIDGE_ROOT,
    NAS_ROOT,
    WORKTREE_ROOT,
    OfflineCandidate,
    probe_bridge_data,
    probe_fsd_dataset,
    run_offline_probe,
    write_probe_outputs,
)
from apps.data_flywheel.gsa.data_adapters.nas_adapter import discover_available_datasets
from apps.data_flywheel.gsa.data_adapters.paco_lvis_adapter import load_paco_annotations


def test_paco_lvis_zip_val_json_is_readable():
    ds_path = NAS_ROOT / "001_done_P1_A0_paco-lvis"
    if not ds_path.exists():
        pytest.skip("PACO-LVIS dataset is not present in this environment")

    data = load_paco_annotations(ds_path, split="val")

    assert data is not None
    assert "images" in data
    assert "annotations" in data
    assert "categories" in data
    assert len(data["annotations"]) > 0


def test_nas_ready_means_direct_images_not_raw_zip_payload():
    ds_path = NAS_ROOT / "001_done_P1_A0_paco-lvis"
    if not ds_path.exists():
        pytest.skip("PACO-LVIS dataset is not present in this environment")

    datasets = {d["name"]: d for d in discover_available_datasets()}
    paco = datasets["paco-lvis"]

    assert paco["adapter_ready"]
    assert not paco["has_direct_images"]
    assert not paco["ready"]


def test_bridge_probe_creates_review_candidate_not_stable_label():
    if not (BRIDGE_ROOT / "data/images.txt").exists():
        pytest.skip("BridgeDataV2 is not present in this environment")

    result = probe_bridge_data(max_samples=1)

    assert result.records
    record = result.records[0]
    assert record.source_dataset == "BridgeDataV2"
    assert record.prompt
    assert record.stable_training_label is False
    assert record.label_status == "candidate"
    assert record.verifier_status == "review_required"
    assert "requires_teacher_or_human_label" in record.risk_flags


def test_fsd_probe_creates_frame_prompt_candidates_when_available():
    ds_root = NAS_ROOT / "004_done_P1_A0_fsd-dataset"
    if not ds_root.exists():
        pytest.skip("FSD dataset is not present in this environment")

    result = probe_fsd_dataset(max_samples=1)

    assert result.records
    record = result.records[0]
    assert record.prompt
    assert record.frame_refs
    assert record.stable_training_label is False
    assert "no_native_2d_action_geometry" in record.risk_flags


def test_run_probe_report_keeps_all_candidates_non_stable():
    if not NAS_ROOT.exists() and not BRIDGE_ROOT.exists():
        pytest.skip("Optional source datasets are not installed")
    report = run_offline_probe(max_samples=1)
    records = [
        record
        for dataset in report["datasets"]
        for record in dataset.get("records", [])
    ]

    assert records
    assert report["stable_training_label_count"] == 0
    assert report["promotion_eligible_count"] == 0
    assert report["label_status_counts"]["candidate"] == len(records)
    assert all(record["stable_training_label"] is False for record in records)
    assert all(record["schema_version"] == "a_action_interface_candidate_v1" for record in records)
    assert all(record["promotion_blockers"] for record in records)


def test_offline_candidate_serialization_omits_empty_values():
    candidate = OfflineCandidate(
        source_dataset="unit",
        sample_id="unit_001",
        candidate_kind="point_coordinate_support",
        action_type="pick",
        label_type="point",
        source_provenance="public_dataset",
        verifier_status="review_required",
        stable_training_label=False,
        source_fields=["points"],
        risk_flags=["not_native_grasp_label"],
    )

    data = candidate.to_dict()

    assert data["sample_id"] == "unit_001"
    assert data["label_status"] == "candidate"
    assert data["schema_version"] == "a_action_interface_candidate_v1"
    assert "image_path" not in data


def test_write_probe_outputs_includes_schema_and_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "apps.data_flywheel.gsa.data_adapters.a_action_interface_offline_probe.WORKTREE_ROOT",
        tmp_path,
    )
    report = run_offline_probe(max_samples=1)
    output_dir = tmp_path / "probe"

    paths = write_probe_outputs(report, output_dir)

    assert paths["summary"].exists()
    assert paths["manifest"].exists()
    assert paths["report"].exists()
    assert paths["schema"].exists()
    assert paths["preview"].exists()
    assert "a_action_interface_candidate_v1" in paths["schema"].read_text(encoding="utf-8")
    assert "A_action_interface Candidate Preview" in paths["preview"].read_text(encoding="utf-8")
