from __future__ import annotations

import hashlib
import json
import struct
import sys
import zlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import extract_scannet_train_sens as sens  # noqa: E402
from tools.extract_scannet_val_frame_bundles import (  # noqa: E402
    DEFAULT_SOURCE_EXCLUSIONS,
    ExtractError,
    _frame_root_binding,
    canonical_hash,
    check_scene,
    inspect_scene,
    load_source_exclusions,
    main,
    materialize_bundles,
    stable_frame_indices,
)


SPLIT_ROOT = ROOT / "configs/scannet_splits"
VAL_SCENE = "scene0700_00"
SECOND_VAL_SCENE = "scene0700_01"
TRAIN_SCENE = "scene0191_00"


def _matrix_bytes(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype="<f4").tobytes()


def _jpeg_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _png_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_sens(path: Path, frame_count: int = 4) -> None:
    color_height, color_width = 6, 8
    depth_height, depth_width = 3, 4
    intrinsic_color = np.eye(4, dtype=np.float32)
    intrinsic_color[0, 0] = intrinsic_color[1, 1] = 2.0
    intrinsic_color[0, 2] = 3.5
    intrinsic_color[1, 2] = 2.5
    intrinsic_depth = np.eye(4, dtype=np.float32)
    intrinsic_depth[0, 0] = intrinsic_depth[1, 1] = 1.0
    intrinsic_depth[0, 2] = 1.5
    intrinsic_depth[1, 2] = 1.0
    identity = np.eye(4, dtype=np.float32)
    sensor_name = b"SyntheticSensor"
    with path.open("wb") as handle:
        handle.write(struct.pack("<I", 4))
        handle.write(struct.pack("<Q", len(sensor_name)))
        handle.write(sensor_name)
        for matrix in (intrinsic_color, identity, intrinsic_depth, identity):
            handle.write(_matrix_bytes(matrix))
        handle.write(
            struct.pack(
                "<iiIIIIfQ",
                2,
                1,
                color_width,
                color_height,
                depth_width,
                depth_height,
                1000.0,
                frame_count,
            )
        )
        for frame_index in range(frame_count):
            color = np.zeros((color_height, color_width, 3), dtype=np.uint8)
            color[..., 0] = 20 + frame_index
            color[..., 1] = np.arange(color_width, dtype=np.uint8)
            depth = np.full((depth_height, depth_width), 1000 + frame_index, dtype=np.uint16)
            color_bytes = _jpeg_bytes(color)
            depth_bytes = zlib.compress(depth.astype("<u2").tobytes())
            pose = identity.copy()
            pose[0, 3] = frame_index
            handle.write(_matrix_bytes(pose))
            handle.write(
                struct.pack(
                    "<QQQQ",
                    frame_index,
                    frame_index,
                    len(color_bytes),
                    len(depth_bytes),
                )
            )
            handle.write(color_bytes)
            handle.write(depth_bytes)


def _source_fixture(tmp_path: Path, scene_id: str = VAL_SCENE) -> Path:
    scannet_root = tmp_path / "ScanNet-official/scans"
    scene = scannet_root / scene_id
    scene.mkdir(parents=True)
    _write_sens(scene / f"{scene_id}.sens")
    (scene / f"{scene_id}.txt").write_text(
        "\n".join(
            (
                "colorHeight = 6",
                "colorWidth = 8",
                "depthHeight = 3",
                "depthWidth = 4",
                "numColorFrames = 4",
                "numDepthFrames = 4",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (scene / f"{scene_id}.aggregation.json").write_text(
        json.dumps(
            {
                "sceneId": f"scannet.{scene_id}",
                "segGroups": [{"objectId": 0, "label": "chair"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    instance = np.zeros((6, 8), dtype=np.uint8)
    instance[1:5, 2:6] = 1
    semantic = instance.astype(np.uint16) * 5
    with ZipFile(
        scene / f"{scene_id}_2d-instance-filt.zip", "w", ZIP_DEFLATED
    ) as archive:
        for index in (0, 1, 2, 3):
            archive.writestr(f"instance-filt/{index}.png", _png_bytes(instance))
    with ZipFile(
        scene / f"{scene_id}_2d-label-filt.zip", "w", ZIP_DEFLATED
    ) as archive:
        for index in (0, 2, 3):
            archive.writestr(f"label-filt/{index}.png", _png_bytes(semantic))
    return scannet_root


def _inspection(tmp_path: Path, max_frames: int = 2):
    scannet_root = _source_fixture(tmp_path)
    split_documents, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    source_exclusions, source_exclusions_binding = load_source_exclusions(
        DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    inspection = inspect_scene(
        scannet_root=scannet_root,
        scene_to_split=scene_to_split,
        source_exclusions=source_exclusions,
        scene_id=VAL_SCENE,
        max_frames=max_frames,
    )
    excluded_root = tmp_path / "existing_frame_root"
    excluded_root.mkdir()
    _, frame_root_binding = _frame_root_binding(excluded_root)
    return (
        scannet_root,
        split_documents,
        scene_to_split,
        inspection,
        source_exclusions_binding,
        [frame_root_binding],
    )


def test_selection_uses_only_common_members_and_is_stable(tmp_path: Path) -> None:
    _, _, _, inspection, _, _ = _inspection(tmp_path, max_frames=2)
    expected = stable_frame_indices(VAL_SCENE, {0, 2, 3}, 2)
    assert inspection.common_frame_count == 3
    assert inspection.selected_indices == expected
    assert 1 not in inspection.selected_indices
    assert set(inspection.instance_members) == set(expected)
    assert set(inspection.semantic_members) == set(expected)
    assert all(document["sha256"] for document in inspection.source_files.values())


def test_check_decodes_without_writing_and_forbids_training(tmp_path: Path) -> None:
    _, split_documents, _, inspection, source_binding, frame_bindings = _inspection(
        tmp_path
    )
    report = check_scene(
        inspection, split_documents, source_binding, frame_bindings
    )
    assert report["verdict"] == "CHECK_PASS_VAL_AUDIT_ONLY"
    assert report["write_count"] == 0
    assert report["training_materialization_eligible"] is False
    assert report["use_policy"]["official_split"] == "val"
    assert report["use_policy"]["may_enter_training"] is False
    assert len(report["decoded_frames"]) == 2
    assert report["source_identities_reverified_after_decode"] is True
    assert report["selection"]["requested_max_frames_per_scene"] == 2
    assert all(row["color_dtype"] == "uint8" for row in report["decoded_frames"])
    assert all(row["depth_dtype"] == "uint16" for row in report["decoded_frames"])


def test_canary_layout_manifest_and_hash_binding(tmp_path: Path) -> None:
    _, split_documents, _, inspection, source_binding, frame_bindings = _inspection(
        tmp_path
    )
    output = tmp_path / "expanded"
    result = materialize_bundles(
        inspections=[inspection],
        split_documents=split_documents,
        output_root=output,
        stage="canary",
        start_ordinal=7,
        scene_list_binding={"mode": "synthetic", "scene_count": 1},
        source_exclusions_binding=source_binding,
        frame_root_exclusions=frame_bindings,
    )
    scene_root = output / f"000007-scannet-{VAL_SCENE}"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checksum = (output / "manifest.sha256").read_text(encoding="ascii").split()[0]

    assert result["training_materialization_eligible"] is False
    assert manifest["training_materialization_eligible"] is False
    assert manifest["use_policy"]["held_out_benchmark_preparation_only"] is True
    assert manifest["use_policy"]["redistribution_eligible"] is False
    assert manifest["selection"]["requested_max_frames_per_scene"] == 2
    assert manifest["frame_root_exclusions"] == frame_bindings
    assert manifest["source_exclusions"]["sha256"]
    assert checksum == hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    integrity = manifest.pop("integrity")
    assert integrity["manifest_payload_sha256"] == canonical_hash(manifest)
    for name in (
        "intrinsic_color.txt",
        "intrinsic_depth.txt",
        "extrinsic_color.txt",
        "extrinsic_depth.txt",
    ):
        assert (scene_root / name).is_file()
    for index in inspection.selected_indices:
        assert (scene_root / f"{index:06d}-rgb.png").is_file()
        assert (scene_root / f"{index:06d}-depth.png").is_file()
        assert (scene_root / f"{index:06d}.txt").is_file()
        with Image.open(scene_root / f"{index:06d}-rgb.png") as image:
            rgb = np.asarray(image)
        with Image.open(scene_root / f"{index:06d}-depth.png") as image:
            depth = np.asarray(image)
        assert rgb.shape == (6, 8, 3) and rgb.dtype == np.uint8
        assert depth.shape == (3, 4) and depth.dtype in (np.uint16, np.int32)
    assert all(
        row["source_payloads"]["color_compressed_sha256"]
        for row in manifest["scenes"][0]["frames"]
    )


def test_nonempty_output_is_refused_without_changes(tmp_path: Path) -> None:
    _, split_documents, _, inspection, source_binding, frame_bindings = _inspection(
        tmp_path
    )
    output = tmp_path / "expanded"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(ExtractError, match="refusing nonempty"):
        materialize_bundles(
            inspections=[inspection],
            split_documents=split_documents,
            output_root=output,
            stage="canary",
            start_ordinal=0,
            scene_list_binding={"mode": "synthetic"},
            source_exclusions_binding=source_binding,
            frame_root_exclusions=frame_bindings,
        )
    assert marker.read_text(encoding="utf-8") == "preserve\n"


def test_train_scene_is_rejected_before_source_access(tmp_path: Path) -> None:
    _, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    source_exclusions, _ = load_source_exclusions(
        DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    with pytest.raises(ExtractError, match="not official val"):
        inspect_scene(
            scannet_root=tmp_path / "missing",
            scene_to_split=scene_to_split,
            source_exclusions=source_exclusions,
            scene_id=TRAIN_SCENE,
            max_frames=1,
        )


def test_max_frames_above_frozen_limit_is_rejected() -> None:
    with pytest.raises(ExtractError, match="max_frames"):
        stable_frame_indices(VAL_SCENE, range(100), 65)


def test_frozen_bad_source_is_rejected_before_source_access(tmp_path: Path) -> None:
    _, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    source_exclusions, binding = load_source_exclusions(
        DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    assert binding["sha256"]
    assert "scene0015_00" in source_exclusions
    with pytest.raises(ExtractError, match="frozen in source exclusions"):
        inspect_scene(
            scannet_root=tmp_path / "missing",
            scene_to_split=scene_to_split,
            source_exclusions=source_exclusions,
            scene_id="scene0015_00",
            max_frames=1,
        )


def test_selected_annotation_payload_must_be_valid_png(tmp_path: Path) -> None:
    scannet_root = _source_fixture(tmp_path)
    scene = scannet_root / VAL_SCENE
    with ZipFile(
        scene / f"{VAL_SCENE}_2d-label-filt.zip", "w", ZIP_DEFLATED
    ) as archive:
        for index in (0, 2, 3):
            archive.writestr(f"label-filt/{index}.png", b"not-a-png")
    _, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    source_exclusions, _ = load_source_exclusions(
        DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    with pytest.raises(ExtractError, match="invalid selected annotation member"):
        inspect_scene(
            scannet_root=scannet_root,
            scene_to_split=scene_to_split,
            source_exclusions=source_exclusions,
            scene_id=VAL_SCENE,
            max_frames=1,
        )


def test_source_drift_after_inspection_is_rejected(tmp_path: Path) -> None:
    scannet_root, split_documents, _, inspection, source_binding, frame_bindings = (
        _inspection(tmp_path, max_frames=1)
    )
    with (scannet_root / VAL_SCENE / f"{VAL_SCENE}.sens").open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(ExtractError, match="source identities changed"):
        check_scene(inspection, split_documents, source_binding, frame_bindings)


def test_materialization_requires_bound_frame_root_exclusion(tmp_path: Path) -> None:
    _, split_documents, _, inspection, source_binding, _ = _inspection(tmp_path)
    with pytest.raises(ExtractError, match="frame-root exclusion"):
        materialize_bundles(
            inspections=[inspection],
            split_documents=split_documents,
            output_root=tmp_path / "expanded",
            stage="canary",
            start_ordinal=0,
            scene_list_binding={"mode": "synthetic"},
            source_exclusions_binding=source_binding,
            frame_root_exclusions=[],
        )


def test_cli_canary_requires_exclusion_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scannet_root = _source_fixture(tmp_path)
    rc = main(
        [
            "--stage",
            "canary",
            "--scene-id",
            VAL_SCENE,
            "--scannet-root",
            str(scannet_root),
            "--split-root",
            str(SPLIT_ROOT),
            "--output-root",
            str(tmp_path / "expanded"),
            "--max-frames",
            "1",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert report["verdict"] == "BLOCKED"
    assert "exclude-frame-root" in report["error"]


def test_multiscene_batch_and_midrun_failure_are_atomic(tmp_path: Path) -> None:
    scannet_root = _source_fixture(tmp_path, VAL_SCENE)
    _source_fixture(tmp_path, SECOND_VAL_SCENE)
    split_documents, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    source_exclusions, source_binding = load_source_exclusions(
        DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    inspections = [
        inspect_scene(
            scannet_root=scannet_root,
            scene_to_split=scene_to_split,
            source_exclusions=source_exclusions,
            scene_id=scene_id,
            max_frames=1,
        )
        for scene_id in (VAL_SCENE, SECOND_VAL_SCENE)
    ]
    excluded_root = tmp_path / "excluded"
    excluded_root.mkdir()
    _, frame_binding = _frame_root_binding(excluded_root)
    binding = {
        "mode": "synthetic_batch",
        "scene_count": 2,
        "scene_ids_sha256": canonical_hash([VAL_SCENE, SECOND_VAL_SCENE]),
    }

    output = tmp_path / "batch"
    result = materialize_bundles(
        inspections=inspections,
        split_documents=split_documents,
        output_root=output,
        stage="batch",
        start_ordinal=20,
        scene_list_binding=binding,
        source_exclusions_binding=source_binding,
        frame_root_exclusions=[frame_binding],
    )
    assert result["scene_count"] == 2
    assert (output / f"000020-scannet-{VAL_SCENE}").is_dir()
    assert (output / f"000021-scannet-{SECOND_VAL_SCENE}").is_dir()

    failed_output = tmp_path / "failed_batch"
    second_sens = scannet_root / SECOND_VAL_SCENE / f"{SECOND_VAL_SCENE}.sens"
    with second_sens.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(ExtractError, match="source identities changed"):
        materialize_bundles(
            inspections=inspections,
            split_documents=split_documents,
            output_root=failed_output,
            stage="batch",
            start_ordinal=30,
            scene_list_binding=binding,
            source_exclusions_binding=source_binding,
            frame_root_exclusions=[frame_binding],
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed_batch.tmp-*"))


def test_cli_exclusion_root_blocks_overlap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scannet_root = _source_fixture(tmp_path)
    excluded = tmp_path / "existing"
    (excluded / f"001-scannet-{VAL_SCENE}").mkdir(parents=True)
    rc = main(
        [
            "--stage",
            "check",
            "--scene-id",
            VAL_SCENE,
            "--scannet-root",
            str(scannet_root),
            "--split-root",
            str(SPLIT_ROOT),
            "--exclude-frame-root",
            str(excluded),
            "--max-frames",
            "1",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert report["verdict"] == "BLOCKED"
    assert "overlap" in report["error"]
