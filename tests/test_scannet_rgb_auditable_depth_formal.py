from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import build_depth_v50_scannet_sensor as sensor
from tools import build_scannet_rgb_auditable_depth_formal as formal
from tools import build_scannet_rgb_auditable_depth_pack as canary
from tools import check_scannet_rgb_auditable_depth_pack as checker
from tools import extract_scannet_val_frame_bundles as val_extractor


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _artifact(root: Path, relative: str, payload: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": relative,
        "sha256": sensor.file_sha256(path),
        "size_bytes": len(payload),
    }


def _write_frame_root_manifest(root: Path, manifest: dict) -> None:
    _write_json(root / "manifest.json", manifest)
    digest = sensor.file_sha256(root / "manifest.json")
    (root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )


def _additional_root_fixture(
    tmp_path: Path,
    scene_ids: tuple[str, ...] = ("scene0011_00",),
) -> tuple[Path, dict, Path]:
    root = tmp_path / "frames"
    scenes = []
    first_rgb: Path | None = None
    total_frames = 0
    for ordinal, scene_id in enumerate(scene_ids):
        directory = f"{ordinal:06d}-scannet-{scene_id}"
        calibration = {
            name: _artifact(
                root,
                f"{directory}/{name}",
                f"{scene_id}-{name}".encode("ascii"),
            )
            for name in val_extractor.CALIBRATION_NAMES
        }
        frame_index = ordinal + 1
        outputs = {
            "rgb": _artifact(
                root,
                f"{directory}/{frame_index:06d}-rgb.png",
                f"{scene_id}-rgb".encode("ascii"),
            ),
            "depth": _artifact(
                root,
                f"{directory}/{frame_index:06d}-depth.png",
                f"{scene_id}-depth".encode("ascii"),
            ),
            "pose": _artifact(
                root,
                f"{directory}/{frame_index:06d}.txt",
                f"{scene_id}-pose".encode("ascii"),
            ),
        }
        if first_rgb is None:
            first_rgb = root / outputs["rgb"]["path"]
        scenes.append(
            {
                "scene_id": scene_id,
                "official_split": "val",
                "ordinal": ordinal,
                "output_directory": directory,
                "requested_max_frames": 1,
                "selected_frame_indices": [frame_index],
                "selected_frame_indices_sha256": val_extractor.canonical_hash(
                    [frame_index]
                ),
                "calibration_outputs": calibration,
                "frames": [{"frame_index": frame_index, "outputs": outputs}],
            }
        )
        total_frames += 1
    stage = "one_scene_canary" if len(scenes) == 1 else "batch"
    manifest = {
        "schema": val_extractor.SCHEMA,
        "protocol": val_extractor.PROTOCOL,
        "stage": stage,
        "verdict": (
            "CANARY_PASS_VAL_AUDIT_ONLY"
            if stage == "one_scene_canary"
            else "BATCH_PASS_VAL_AUDIT_ONLY"
        ),
        "execution": {
            "cpu_only": True,
            "gpu_required": False,
            "network_calls": 0,
            "scene_count": len(scenes),
            "frame_count": total_frames,
        },
        "training_materialization_eligible": False,
        "use_policy": {
            "held_out_benchmark_preparation_only": True,
            "may_enter_training": False,
            "official_split": "val",
            "training_materialization_eligible": False,
        },
        "extractor": {
            "extractor_path": str(Path(val_extractor.__file__).resolve()),
            "extractor_sha256": sensor.file_sha256(Path(val_extractor.__file__)),
            "reader_path": str(Path(val_extractor.sens.__file__).resolve()),
            "reader_sha256": sensor.file_sha256(Path(val_extractor.sens.__file__)),
        },
        "selection": {
            "seed": val_extractor.SELECTION_SEED,
            "maximum_frames_per_scene": val_extractor.MAX_FRAMES_PER_SCENE,
            "requested_max_frames_per_scene": 1,
            "selected_only_from_instance_semantic_member_intersection": True,
            "scene_unique": True,
        },
        "scenes": scenes,
    }
    manifest["integrity"] = {
        "scene_ids_sha256": val_extractor.canonical_hash(list(scene_ids)),
        "manifest_payload_sha256": val_extractor.canonical_hash(manifest),
    }
    _write_frame_root_manifest(root, manifest)
    assert first_rgb is not None
    return root, manifest, first_rgb


def _binding_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    pack = tmp_path / "canary"
    selected = [
        {
            "scene_id": f"scene9{i:03d}_00",
            "image_sha256": f"image-{i:02d}",
            "source_sample_id": f"source-{i:02d}",
        }
        for i in range(canary.CANARY_TARGET)
    ]
    manifest = {
        "schema_version": canary.SCHEMA_VERSION,
        "protocol_version": canary.PACK_PROTOCOL,
        "audit_round": canary.AUDIT_ROUND,
        "audit_contract": canary.AUDIT_CONTRACT,
        "stage": "canary",
        "forms": ["depth"],
        "images": canary.CANARY_TARGET,
        "expected_decisions_per_annotator": canary.CANARY_TARGET,
        "selected": selected,
        "selection": {
            "rule": "strict_filter_best_frame_per_scene_then_stable_hash",
            "seed": canary.SELECTION_SEED,
            "target": canary.CANARY_TARGET,
            "unique_scene_required": True,
            "unique_image_required": True,
            "selected_unique_scenes": canary.CANARY_TARGET,
            "selected_unique_images": canary.CANARY_TARGET,
            "thresholds": canary.STRICT_THRESHOLDS,
        },
        "integrity": {"manifest_payload_sha256": "payload-hash"},
    }
    _write_json(pack / "manifest.json", manifest)
    for side, key in (("a", "reviewer-a"), ("b", "reviewer-b")):
        path = pack / f"annotator_{side}/annotations/annotations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps({"annotator_key": key, "correct": True}) + "\n"
                for _ in range(canary.CANARY_TARGET)
            ),
            encoding="utf-8",
        )
    report = {
        "status": formal.PASS_STATUS,
        "pack": str(pack.resolve()),
        "manifest_sha256": sensor.file_sha256(pack / "manifest.json"),
        "manifest_payload_sha256": "payload-hash",
        "stage": "canary",
        "records": canary.CANARY_TARGET,
        "unique_scenes": canary.CANARY_TARGET,
        "unique_images": canary.CANARY_TARGET,
        "source_artifacts_rehashed": True,
        "source_artifact_errors": 0,
        "formal_qualification_eligible": False,
        "review": {
            "expected_per_annotator": canary.CANARY_TARGET,
            "a_complete": canary.CANARY_TARGET,
            "b_complete": canary.CANARY_TARGET,
            "agreement_count": canary.CANARY_TARGET,
            "disagreement_count": 0,
            "raw_agreement": 1.0,
            "resolved_correct_count": canary.CANARY_TARGET,
            "resolved_correctness": 1.0,
            "calibration_gate_passed": True,
            "formal_40_gate_passed": False,
        },
    }
    report_path = tmp_path / "report.json"
    _write_json(report_path, report)
    monkeypatch.setattr(checker, "validate_pack", lambda *args, **kwargs: dict(report))
    monkeypatch.setattr(
        checker, "compile_reviews", lambda *args, **kwargs: dict(report)
    )
    return pack, report_path


def _formal_pack_fixture(
    tmp_path: Path,
    *,
    additional_bindings: list[dict] | None = None,
) -> tuple[Path, dict]:
    from tests.test_scannet_rgb_auditable_depth_pack import _inventory, _record

    records = [_record(tmp_path, index) for index in range(canary.FORMAL_TARGET)]
    reviewed = {
        "pack_path": str(tmp_path / "missing-reviewed-canary"),
        "report_path": str(tmp_path / "missing-reviewed-report.json"),
        "scene_ids": [f"scene8{index:03d}_00" for index in range(canary.CANARY_TARGET)],
    }
    inventory = _inventory()
    inventory.update(
        {
            "strict_eligible_records": canary.FORMAL_TARGET,
            "strict_eligible_scenes": canary.FORMAL_TARGET,
            "reviewed_rgb_canary_binding": reviewed,
            "additional_frame_root_bindings": additional_bindings or [],
            "excluded_scene_count": len(reviewed["scene_ids"]),
            "excluded_scene_ids": sorted(reviewed["scene_ids"]),
        }
    )
    output = tmp_path / "formal-pack"
    formal._freeze_formal(output, records, inventory)
    return output, reviewed


def test_reviewed_canary_binding_accepts_exact_pass_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack, report = _binding_fixture(tmp_path, monkeypatch)
    binding = formal.load_reviewed_canary_binding(pack, report)
    assert binding["status"] == formal.PASS_STATUS
    assert binding["scene_count"] == canary.CANARY_TARGET
    assert binding["annotation_streams"]["a"]["annotator_key"] == "reviewer-a"
    assert binding["annotation_streams"]["b"]["annotator_key"] == "reviewer-b"


def test_additional_frame_root_is_manifest_bound(tmp_path: Path) -> None:
    root, manifest, _ = _additional_root_fixture(tmp_path)
    binding = formal.load_additional_frame_root(root)
    assert binding["scene_ids"] == ["scene0011_00"]
    assert binding["frame_count"] == 1

    manifest["training_materialization_eligible"] = True
    manifest["integrity"]["manifest_payload_sha256"] = val_extractor.canonical_hash(
        {key: value for key, value in manifest.items() if key != "integrity"}
    )
    _write_frame_root_manifest(root, manifest)
    with pytest.raises(canary.PackError, match="frame-root contract mismatch"):
        formal.load_additional_frame_root(root)


def test_additional_frame_root_accepts_non_sorted_manifest_scene_order(
    tmp_path: Path,
) -> None:
    scene_ids = ("scene0697_03", "scene0278_00")
    root, manifest, _ = _additional_root_fixture(tmp_path, scene_ids)
    assert scene_ids != tuple(sorted(scene_ids))
    assert manifest["integrity"]["scene_ids_sha256"] == val_extractor.canonical_hash(
        list(scene_ids)
    )
    binding = formal.load_additional_frame_root(root)
    assert binding["scene_ids"] == sorted(scene_ids)
    assert binding["scene_count"] == 2


def test_additional_frame_root_rejects_output_and_directory_drift(
    tmp_path: Path,
) -> None:
    root, _, rgb_path = _additional_root_fixture(tmp_path)
    rgb_path.write_bytes(b"tampered-rgb")
    with pytest.raises(canary.PackError, match="artifact drift"):
        formal.load_additional_frame_root(root)

    root, _, _ = _additional_root_fixture(tmp_path / "extra")
    scene_root = next(child for child in root.iterdir() if child.is_dir())
    (scene_root / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(canary.PackError, match="directory drift"):
        formal.load_additional_frame_root(root)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "BLOCKED"),
        ("source_artifacts_rehashed", False),
        ("source_artifact_errors", 1),
    ),
)
def test_reviewed_canary_binding_rejects_failed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    pack, report_path = _binding_fixture(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    _write_json(report_path, report)
    monkeypatch.setattr(checker, "validate_pack", lambda *args, **kwargs: dict(report))
    monkeypatch.setattr(checker, "compile_reviews", lambda *args, **kwargs: dict(report))
    with pytest.raises(
        canary.PackError, match="report does not match|contract mismatch"
    ):
        formal.load_reviewed_canary_binding(pack, report_path)


def test_formal_scan_excludes_reviewed_canary_before_source_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviewed_scene = "scene9000_00"
    fresh_scene = "scene9001_00"
    binding = {"scene_ids": [reviewed_scene]}
    monkeypatch.setattr(formal, "load_reviewed_canary_binding", lambda *a, **k: binding)
    monkeypatch.setattr(
        canary,
        "load_old_audit_binding",
        lambda path: {"scene_ids": [], "path": str(path)},
    )
    monkeypatch.setattr(sensor, "load_split_manifests", lambda path: ({}, {}))
    monkeypatch.setattr(sensor, "load_source_exclusions", lambda *a: (set(), {}))
    scenes = [
        SimpleNamespace(scene_id=reviewed_scene, official_split="val"),
        SimpleNamespace(scene_id=fresh_scene, official_split="val"),
    ]
    monkeypatch.setattr(sensor, "discover_scenes", lambda *a: scenes)
    monkeypatch.setattr(
        sensor, "load_canary_scene_manifest", lambda *a: {"scene_ids": []}
    )
    monkeypatch.setattr(sensor, "source_ready", lambda scene: True)
    monkeypatch.setattr(sensor, "load_label_map", lambda path: {})
    processed: list[str] = []

    def process(scene, *args, **kwargs):
        processed.append(scene.scene_id)
        return ([{"scene_id": scene.scene_id, "sample_id": scene.scene_id}], {}, {}, [])

    monkeypatch.setattr(sensor, "process_scene", process)
    monkeypatch.setattr(sensor, "validate_record", lambda *args: [])
    monkeypatch.setattr(canary, "candidate_features", lambda row: ({}, []))
    args = SimpleNamespace(
        reviewed_canary_pack=tmp_path / "pack",
        reviewed_canary_report=tmp_path / "report",
        old_pack=tmp_path / "old",
        split_root=tmp_path,
        source_exclusions=tmp_path / "excluded",
        scannet_root=tmp_path,
        openeqa_root=tmp_path,
        canary_scenes=tmp_path / "canary-scenes",
        label_map=tmp_path / "labels",
        additional_frame_root=[],
    )
    rows, inventory = formal._scan(args)
    assert processed == [fresh_scene]
    assert [row["scene_id"] for row in rows] == [fresh_scene]
    assert reviewed_scene in inventory["excluded_scene_ids"]


def test_formal_freeze_rejects_reviewed_canary_overlap_without_writing(
    tmp_path: Path,
) -> None:
    reviewed_scene = "scene9000_00"
    output = tmp_path / "must-not-exist"
    inventory = {"reviewed_rgb_canary_binding": {"scene_ids": [reviewed_scene]}}
    with pytest.raises(canary.PackError, match="overlaps reviewed"):
        formal._freeze_formal(
            output,
            [{"scene_id": reviewed_scene}],
            inventory,
        )
    assert not output.exists()


def test_formal_staging_failure_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "must-not-exist"
    inventory = {
        "reviewed_rgb_canary_binding": {"scene_ids": ["scene8000_00"]},
        "additional_frame_root_bindings": [],
    }

    def fail_mid_freeze(staged_pack: Path, *args, **kwargs):
        staged_pack.mkdir(parents=True)
        (staged_pack / "partial").write_text("partial", encoding="ascii")
        raise canary.PackError("injected staging failure")

    monkeypatch.setattr(canary, "_freeze", fail_mid_freeze)
    with pytest.raises(canary.PackError, match="injected staging failure"):
        formal._freeze_formal(
            output,
            [{"scene_id": "scene9000_00"}],
            inventory,
        )
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.formal.*")) == []


def test_formal_checker_live_revalidates_additional_root_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_root = {
        "path": str(tmp_path / "additional-root"),
        "manifest_path": str(tmp_path / "additional-root/manifest.json"),
        "manifest_sha256": "a" * 64,
        "manifest_payload_sha256": "b" * 64,
        "scene_count": 1,
        "scene_ids": ["scene7000_00"],
        "frame_count": 1,
        "stage": "batch",
        "training_materialization_eligible": False,
    }
    output, reviewed = _formal_pack_fixture(
        tmp_path, additional_bindings=[expected_root]
    )
    monkeypatch.setattr(
        checker, "load_reviewed_canary_binding", lambda *args, **kwargs: reviewed
    )
    monkeypatch.setattr(
        checker, "load_additional_frame_root", lambda path: dict(expected_root)
    )
    monkeypatch.setattr(sensor, "validate_record", lambda record: [])
    report = checker.validate_pack(output, rehash_source_artifacts=False)
    assert report["status"] == "PACK_VALID"

    monkeypatch.setattr(
        checker,
        "load_additional_frame_root",
        lambda path: {**expected_root, "manifest_sha256": "c" * 64},
    )
    with pytest.raises(checker.CheckError, match="frame-root binding drift"):
        checker.validate_pack(output, rehash_source_artifacts=False)


def test_checker_cli_formal_failure_is_clean_and_writes_no_report(
    tmp_path: Path,
) -> None:
    output, _ = _formal_pack_fixture(tmp_path)
    report_path = tmp_path / "must-not-exist-report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(checker.__file__).resolve()),
            "--pack",
            str(output),
            "--skip-source-artifact-rehash",
            "--out",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "formal reviewed-canary binding failed" in result.stderr
    assert not report_path.exists()


def test_old_formal_pack_without_reviewed_binding_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_scannet_rgb_auditable_depth_pack import _inventory, _record

    records = [_record(tmp_path, index) for index in range(canary.FORMAL_TARGET)]
    inventory = _inventory()
    inventory["strict_eligible_records"] = canary.FORMAL_TARGET
    inventory["strict_eligible_scenes"] = canary.FORMAL_TARGET
    output = tmp_path / "old-formal"
    canary._freeze(output, records, inventory, "formal")
    monkeypatch.setattr(sensor, "validate_record", lambda record: [])
    with pytest.raises(checker.CheckError, match="reviewed-canary binding"):
        checker.validate_pack(output, rehash_source_artifacts=False)
