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
    path: Path, *, frame_count: int = 4, depth_shift: float = 1000.0
) -> None:
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


def _source_fixture(
    tmp_path: Path,
    *,
    color_frames: int = 4,
    depth_frames: int = 4,
    color_width: int = 8,
    depth_shift: float = 1000.0,
) -> Path:
    scannet_root = tmp_path / "ScanNet-official/scans"
    scene = scannet_root / VAL_SCENE
    scene.mkdir(parents=True)
    _write_sens(scene / f"{VAL_SCENE}.sens", depth_shift=depth_shift)
    (scene / f"{VAL_SCENE}.txt").write_text(
        "\n".join(
            (
                "colorHeight = 6",
                f"colorWidth = {color_width}",
                "depthHeight = 3",
                "depthWidth = 4",
                f"numColorFrames = {color_frames}",
                f"numDepthFrames = {depth_frames}",
            )
        )
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


def _inspect(scannet_root: Path, max_frames: int = 4) -> v2.SceneInspection:
    _, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    exclusions, _ = v2.load_source_exclusions(
        v2.DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    return v2.inspect_scene(
        scannet_root=scannet_root,
        scene_to_split=scene_to_split,
        source_exclusions=exclusions,
        scene_id=VAL_SCENE,
        max_frames=max_frames,
    )


@pytest.mark.parametrize(
    ("color_frames", "depth_frames"),
    ((4, 4), (5, 4), (4, 5), (5, 5)),
)
def test_metadata_exact_or_independent_plus_one_is_accepted(
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
    assert set(inspection.selected_indices) == {0, 2, 3}
    assert all(index < inspection.header.num_frames for index in inspection.selected_indices)
    assert 4 not in inspection.selected_indices


def test_exact_metadata_preserves_v1_frame_selection(tmp_path: Path) -> None:
    inspection = _inspect(_source_fixture(tmp_path), max_frames=2)
    assert v2.SELECTION_SEED == v1.SELECTION_SEED
    assert inspection.selected_indices == v1.stable_frame_indices(
        VAL_SCENE, {0, 2, 3}, 2
    )


@pytest.mark.parametrize(
    ("color_frames", "depth_frames", "key"),
    ((6, 4, "numColorFrames"), (4, 6, "numDepthFrames")),
)
def test_metadata_plus_two_is_rejected(
    tmp_path: Path, color_frames: int, depth_frames: int, key: str
) -> None:
    scannet_root = _source_fixture(
        tmp_path, color_frames=color_frames, depth_frames=depth_frames
    )
    with pytest.raises(sens.SourceError, match=key):
        _inspect(scannet_root)


def test_dimensions_and_depth_shift_remain_exact(tmp_path: Path) -> None:
    with pytest.raises(sens.SourceError, match="colorWidth"):
        _inspect(_source_fixture(tmp_path / "dimension", color_width=9))
    with pytest.raises(v2.ExtractError, match="exact depth_shift=1000"):
        _inspect(_source_fixture(tmp_path / "shift", depth_shift=999.0))


def test_materialized_manifest_self_binds_v2_extractor(tmp_path: Path) -> None:
    inspection = _inspect(
        _source_fixture(tmp_path / "source", color_frames=5, depth_frames=4),
        max_frames=2,
    )
    split_documents, scene_to_split = sens.load_split_manifests(SPLIT_ROOT)
    _, source_binding = v2.load_source_exclusions(
        v2.DEFAULT_SOURCE_EXCLUSIONS, scene_to_split
    )
    excluded_root = tmp_path / "excluded"
    excluded_root.mkdir()
    _, frame_binding = v2._frame_root_binding(excluded_root)
    output = tmp_path / "v2_output"
    result = v2.materialize_bundles(
        inspections=[inspection],
        split_documents=split_documents,
        output_root=output,
        stage="canary",
        start_ordinal=7,
        scene_list_binding={"mode": "synthetic", "scene_count": 1},
        source_exclusions_binding=source_binding,
        frame_root_exclusions=[frame_binding],
    )
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    checksum = (output / "manifest.sha256").read_text(encoding="ascii").split()[0]
    extractor_path = Path(v2.__file__).resolve()
    v1_path = Path(v1.__file__).resolve()

    assert result["schema"] == v2.SCHEMA
    assert result["protocol"] == v2.PROTOCOL
    assert manifest["schema"] == v2.SCHEMA
    assert manifest["protocol"] == v2.PROTOCOL
    assert manifest["output_contract"] == {
        "root": str(output.resolve()),
        "namespace": "scannet_val_frame_bundles_v2",
    }
    assert Path(manifest["extractor"]["extractor_path"]) == extractor_path
    assert Path(manifest["extractor"]["extractor_path"]) != v1_path
    assert manifest["extractor"]["extractor_sha256"] == hashlib.sha256(
        extractor_path.read_bytes()
    ).hexdigest()
    dependencies = manifest["extractor"]["implementation_dependencies"]
    assert any(
        Path(row["path"]) == v1_path
        and row["sha256"] == hashlib.sha256(v1_path.read_bytes()).hexdigest()
        for row in dependencies
    )
    assert manifest["selection"][
        "selected_only_from_instance_semantic_member_intersection"
    ] is True
    assert manifest["selection"][
        "selected_indices_strictly_below_sens_num_frames"
    ] is True
    assert checksum == hashlib.sha256(manifest_bytes).hexdigest()
    integrity = manifest.pop("integrity")
    assert integrity["manifest_payload_sha256"] == v2.canonical_hash(manifest)


def test_v2_launchers_are_isolated_from_v1_output_roots() -> None:
    launcher = (ROOT / "scripts/run_scannet_val_frame_bundles_v2.sh").read_text(
        encoding="utf-8"
    )
    queue = (
        ROOT / "scripts/queue_scannet_val_v52_frame_bundles_cpu_v2.sh"
    ).read_text(encoding="utf-8")
    assert "tools/extract_scannet_val_frame_bundles_v2.py" in launcher
    assert "data/scannet_val_frame_bundles_v2/canary_v2" in launcher
    assert "extract_scannet_val_frame_bundles_v2.py" in queue
    assert "data/scannet_val_frame_bundles_v2/$name" in queue
    assert "formal_pool_batch04a_v2" in queue
    assert "tools/extract_scannet_val_frame_bundles.py" not in launcher
