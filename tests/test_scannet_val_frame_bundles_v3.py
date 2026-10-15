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
from tools import extract_scannet_val_frame_bundles as v1  # noqa: E402
from tools import extract_scannet_val_frame_bundles_v2 as v2  # noqa: E402
from tools import extract_scannet_val_frame_bundles_v3 as v3  # noqa: E402


SPLIT_ROOT = ROOT / "configs/scannet_splits"
VAL_SCENE = "scene0700_00"


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


def _write_sens(
    path: Path,
    *,
    frame_count: int = 4,
    depth_shift: float = 1000.0,
    bad_pose_modes: dict[int, str] | None = None,
    declared_color_sizes: dict[int, int] | None = None,
    imu_count: int | None = None,
) -> None:
    bad_pose_modes = bad_pose_modes or {}
    declared_color_sizes = declared_color_sizes or {}
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
                depth_shift,
                frame_count,
            )
        )
        for frame_index in range(frame_count):
            color = np.zeros((color_height, color_width, 3), dtype=np.uint8)
            color[..., 0] = 20 + frame_index
            color[..., 1] = np.arange(color_width, dtype=np.uint8)
            depth = np.full(
                (depth_height, depth_width), 1000 + frame_index, dtype=np.uint16
            )
            color_bytes = _jpeg_bytes(color)
            depth_bytes = zlib.compress(depth.astype("<u2").tobytes())
            pose = identity.copy()
            pose[0, 3] = frame_index
            mode = bad_pose_modes.get(frame_index)
            if mode == "non_finite":
                pose[0, 0] = np.nan
            elif mode == "invalid_homogeneous":
                pose[3, 3] = 0.0
            elif mode is not None:
                raise ValueError(f"unknown bad pose mode: {mode}")
            handle.write(_matrix_bytes(pose))
            handle.write(
                struct.pack(
                    "<QQQQ",
                    frame_index,
                    frame_index,
                    declared_color_sizes.get(frame_index, len(color_bytes)),
                    len(depth_bytes),
                )
            )
            handle.write(color_bytes)
            handle.write(depth_bytes)
        if imu_count is not None:
            handle.write(struct.pack("<Q", imu_count))
            imu_record = struct.pack("<" + "d" * 15 + "Q", *([0.0] * 15), 0)
            assert len(imu_record) == 128
            for _ in range(imu_count):
                handle.write(imu_record)


def _source_fixture(
    tmp_path: Path,
    *,
    color_frames: int = 4,
    depth_frames: int = 4,
    color_width: int = 8,
    depth_shift: float = 1000.0,
    bad_pose_modes: dict[int, str] | None = None,
    declared_color_sizes: dict[int, int] | None = None,
    imu_count: int | None = None,
) -> Path:
    scannet_root = tmp_path / "ScanNet-official/scans"
    scene = scannet_root / VAL_SCENE
    scene.mkdir(parents=True)
    _write_sens(
        scene / f"{VAL_SCENE}.sens",
        depth_shift=depth_shift,
        bad_pose_modes=bad_pose_modes,
        declared_color_sizes=declared_color_sizes,
        imu_count=imu_count,
    )
    metadata_lines = [
        "colorHeight = 6",
        f"colorWidth = {color_width}",
        "depthHeight = 3",
        "depthWidth = 4",
        f"numColorFrames = {color_frames}",
        f"numDepthFrames = {depth_frames}",
    ]
    if imu_count is not None:
        metadata_lines.append(f"numIMUmeasurements = {imu_count}")
    (scene / f"{VAL_SCENE}.txt").write_text(
        "\n".join(metadata_lines)
        + "\n",
        encoding="utf-8",
    )
    (scene / f"{VAL_SCENE}.aggregation.json").write_text(
        json.dumps(
            {
                "sceneId": f"scannet.{VAL_SCENE}",
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
        scene / f"{VAL_SCENE}_2d-instance-filt.zip", "w", ZIP_DEFLATED
    ) as archive:
        for index in (0, 1, 2, 3, 4):
            archive.writestr(f"instance-filt/{index}.png", _png_bytes(instance))
    with ZipFile(
        scene / f"{VAL_SCENE}_2d-label-filt.zip", "w", ZIP_DEFLATED
    ) as archive:
        for index in (0, 2, 3, 4):
            archive.writestr(f"label-filt/{index}.png", _png_bytes(semantic))
    return scannet_root


def _inspect(scannet_root: Path, max_frames: int = 4) -> v3.SceneInspection:
    _, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    exclusions, _ = v3.load_source_exclusions(
        v3.DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    return v3.inspect_scene(
        scannet_root=scannet_root,
        scene_to_split=scene_to_split,
        source_exclusions=exclusions,
        scene_id=VAL_SCENE,
        max_frames=max_frames,
    )


def _bindings(tmp_path: Path) -> tuple[dict[str, dict[str, object]], dict[str, object], list[dict[str, object]]]:
    split_documents, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    _, source_binding = v3.load_source_exclusions(
        v3.DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    excluded_root = tmp_path / "excluded"
    excluded_root.mkdir()
    _, frame_binding = v3._frame_root_binding(excluded_root)
    return split_documents, source_binding, [frame_binding]


@pytest.mark.parametrize(
    ("color_frames", "depth_frames"),
    ((4, 4), (5, 4), (4, 5), (5, 5)),
)
def test_v3_preserves_v2_metadata_gate(
    tmp_path: Path, color_frames: int, depth_frames: int
) -> None:
    inspection = _inspect(
        _source_fixture(
            tmp_path, color_frames=color_frames, depth_frames=depth_frames
        )
    )
    assert inspection.metadata_validation["observed_frame_counts"] == {
        "numColorFrames": color_frames,
        "numDepthFrames": depth_frames,
    }
    assert inspection.metadata_validation["allowed_metadata_frame_counts"] == [4, 5]
    assert inspection.common_frame_count == 3
    assert inspection.valid_common_frame_count == 3
    assert set(inspection.selected_indices) == {0, 2, 3}


def test_nonfinite_pose_before_selected_frame_is_safely_skipped(tmp_path: Path) -> None:
    inspection = _inspect(
        _source_fixture(tmp_path, bad_pose_modes={1: "non_finite"}),
        max_frames=1,
    )
    assert inspection.selected_indices == (3,)
    assert inspection.invalid_pose_rows == (
        {"frame_index": 1, "reason": "non_finite_pose"},
    )
    split_documents, source_binding, frame_bindings = _bindings(tmp_path)
    result = v3.check_scene(
        inspection, split_documents, source_binding, frame_bindings
    )
    assert result["verdict"] == "CHECK_PASS_VAL_AUDIT_ONLY"
    assert result["write_count"] == 0
    assert result["decoded_frames"][0]["pose_finite"] is True
    assert result["decoded_frames"][0]["pose_homogeneous"] is True


@pytest.mark.parametrize(
    ("mode", "reason"),
    (
        ("non_finite", "non_finite_pose"),
        ("invalid_homogeneous", "invalid_homogeneous_last_row"),
    ),
)
def test_pose_invalid_at_old_selected_frame_selects_next_valid_candidate(
    tmp_path: Path, mode: str, reason: str
) -> None:
    baseline = _inspect(_source_fixture(tmp_path / "baseline"), max_frames=1)
    assert baseline.selected_indices == (3,)
    inspection = _inspect(
        _source_fixture(tmp_path / "bad", bad_pose_modes={3: mode}),
        max_frames=1,
    )
    assert inspection.selected_indices == (0,)
    assert inspection.invalid_pose_rows == ({"frame_index": 3, "reason": reason},)
    assert 3 not in inspection.valid_pose_indices


def test_payload_size_and_exact_stream_boundary_remain_fail_closed(
    tmp_path: Path,
) -> None:
    oversized = _source_fixture(
        tmp_path / "oversized",
        declared_color_sizes={1: sens.MAX_FRAME_PAYLOAD_BYTES},
    )
    with pytest.raises(v3.ExtractError, match="payload exceeds"):
        _inspect(oversized)

    trailing = _source_fixture(tmp_path / "trailing")
    sens_path = trailing / VAL_SCENE / f"{VAL_SCENE}.sens"
    with sens_path.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(v3.ExtractError, match="IMU section|stream boundary mismatch"):
        _inspect(trailing)


def test_fixed_width_imu_section_is_strictly_counted_and_metadata_bound(
    tmp_path: Path,
) -> None:
    inspection = _inspect(_source_fixture(tmp_path / "valid", imu_count=2))
    assert inspection.sens_imu_section_present is True
    assert inspection.sens_imu_record_count == 2
    assert inspection.metadata_validation["imu_validation"] == {
        "policy": "metadata_count_equals_fixed_width_sens_imu_section",
        "section_present": True,
        "record_size_bytes": 128,
        "sens_record_count": 2,
        "metadata_record_count": 2,
        "stream_boundary_exact": True,
    }

    mismatched = _source_fixture(tmp_path / "mismatch", imu_count=2)
    metadata_path = mismatched / VAL_SCENE / f"{VAL_SCENE}.txt"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8").replace(
            "numIMUmeasurements = 2", "numIMUmeasurements = 1"
        ),
        encoding="utf-8",
    )
    with pytest.raises(sens.SourceError, match="numIMUmeasurements"):
        _inspect(mismatched)


def test_atomic_cleanup_removes_temp_root_after_decode_failure(tmp_path: Path) -> None:
    scannet_root = _source_fixture(tmp_path / "source")
    inspection = _inspect(scannet_root, max_frames=2)
    split_documents, source_binding, frame_bindings = _bindings(tmp_path)
    sens_path = scannet_root / VAL_SCENE / f"{VAL_SCENE}.sens"
    with sens_path.open("ab") as handle:
        handle.write(b"changed-after-inspection")
    output = tmp_path / "v3_output"

    with pytest.raises(v3.ExtractError, match="IMU stream boundary mismatch"):
        v3.materialize_bundles(
            inspections=[inspection],
            split_documents=split_documents,
            output_root=output,
            stage="canary",
            start_ordinal=7,
            scene_list_binding={"mode": "synthetic", "scene_count": 1},
            source_exclusions_binding=source_binding,
            frame_root_exclusions=frame_bindings,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".v3_output.tmp-*")) == []


def test_manifest_binds_v3_and_never_writes_invalid_pose_frame(tmp_path: Path) -> None:
    inspection = _inspect(
        _source_fixture(tmp_path / "source", bad_pose_modes={3: "non_finite"}),
        max_frames=2,
    )
    split_documents, source_binding, frame_bindings = _bindings(tmp_path)
    output = tmp_path / "v3_output"
    result = v3.materialize_bundles(
        inspections=[inspection],
        split_documents=split_documents,
        output_root=output,
        stage="canary",
        start_ordinal=7,
        scene_list_binding={"mode": "synthetic", "scene_count": 1},
        source_exclusions_binding=source_binding,
        frame_root_exclusions=frame_bindings,
    )
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    checksum = (output / "manifest.sha256").read_text(encoding="ascii").split()[0]
    extractor_path = Path(v3.__file__).resolve()

    assert result["schema"] == v3.SCHEMA
    assert manifest["schema"] == v3.SCHEMA
    assert manifest["protocol"] == v3.PROTOCOL
    assert manifest["output_contract"]["namespace"] == "scannet_val_frame_bundles_v3"
    assert manifest["extractor"]["extractor_path"] == str(extractor_path)
    assert manifest["extractor"]["extractor_sha256"] == hashlib.sha256(
        extractor_path.read_bytes()
    ).hexdigest()
    assert manifest["scenes"][0]["pose_validation"]["invalid_pose_rows"] == [
        {"frame_index": 3, "reason": "non_finite_pose"}
    ]
    written = {row["frame_index"] for row in manifest["scenes"][0]["frames"]}
    assert written == set(inspection.selected_indices)
    assert 3 not in written
    assert not list(output.rglob("000003*"))
    assert checksum == hashlib.sha256(manifest_bytes).hexdigest()
    integrity = manifest.pop("integrity")
    assert integrity["manifest_payload_sha256"] == v3.canonical_hash(manifest)


def test_main_check_only_with_bad_pose_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scannet_root = _source_fixture(
        tmp_path / "source", bad_pose_modes={1: "non_finite"}
    )
    output = tmp_path / "must_not_exist"
    rc = v3.main(
        [
            "--stage",
            "check",
            "--scene-id",
            VAL_SCENE,
            "--scannet-root",
            str(scannet_root),
            "--split-root",
            str(SPLIT_ROOT),
            "--source-exclusions",
            str(v3.DEFAULT_SOURCE_EXCLUSIONS),
            "--output-root",
            str(output),
            "--max-frames",
            "1",
        ]
    )
    document = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert document["stage"] == "check_only"
    assert document["write_count"] == 0
    assert document["scene"]["pose_validation"]["invalid_pose_frame_count"] == 1
    assert not output.exists()


def test_v3_launchers_are_isolated_and_queue_self_binds() -> None:
    launcher = (ROOT / "scripts/run_scannet_val_frame_bundles_v3.sh").read_text(
        encoding="utf-8"
    )
    queue = (
        ROOT / "scripts/queue_scannet_val_v52_frame_bundles_cpu_v3.sh"
    ).read_text(encoding="utf-8")
    assert "tools/extract_scannet_val_frame_bundles_v3.py" in launcher
    assert "data/scannet_val_frame_bundles_v3/canary_v3" in launcher
    assert "extract_scannet_val_frame_bundles_v3.py" in queue
    assert "data/scannet_val_frame_bundles_v3/$name" in queue
    assert "formal_pool_batch04a_v3" in queue
    assert "write_or_verify_binding" in queue
    assert "PREFLIGHT_ONLY" in queue
    assert "PREFLIGHT_PASS" in queue
    assert "queue_sha256=" in queue
    assert "execution_binding.json" in queue
    assert "tools/extract_scannet_val_frame_bundles.py" not in launcher
    assert "tools/extract_scannet_val_frame_bundles_v2.py" not in launcher
    assert v3.SELECTION_SEED == v1.SELECTION_SEED == v2.SELECTION_SEED
