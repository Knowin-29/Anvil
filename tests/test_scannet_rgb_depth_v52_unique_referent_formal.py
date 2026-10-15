from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import scannet_rgb_depth_v52_interface as interface
from tools import build_depth_v50_scannet_sensor as sensor
from tools import build_scannet_rgb_auditable_depth_pack as v51_filter
from tools import build_scannet_rgb_depth_v52_unique_referent_formal as builder
from tools import check_scannet_rgb_depth_v52_unique_referent_formal as checker
from tools import extract_scannet_val_frame_bundles as frame_v1
from tools import extract_scannet_val_frame_bundles_v3 as frame_v3


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifact(root: Path, relative: str, payload: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _complete_v3_root(tmp_path: Path) -> Path:
    root = (tmp_path / "formal_pool_v3").resolve()
    scene_id = "scene0700_00"
    ordinal = 300000
    directory = f"{ordinal:06d}-scannet-{scene_id}"
    matrix = b"1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    calibration = {
        name: _artifact(root, f"{directory}/{name}", matrix)
        for name in frame_v3.CALIBRATION_NAMES
    }
    outputs = {
        "rgb": _artifact(root, f"{directory}/000001-rgb.png", b"rgb"),
        "depth": _artifact(root, f"{directory}/000001-depth.png", b"depth"),
        "pose": _artifact(root, f"{directory}/000001.txt", matrix),
    }
    manifest = {
        "schema": frame_v3.SCHEMA,
        "protocol": frame_v3.PROTOCOL,
        "stage": "batch",
        "verdict": "BATCH_PASS_VAL_AUDIT_ONLY",
        "execution": {
            "cpu_only": True,
            "gpu_required": False,
            "network_calls": 0,
            "scene_count": 1,
            "frame_count": 1,
        },
        "output_contract": {
            "root": str(root),
            "namespace": "scannet_val_frame_bundles_v3",
        },
        "training_materialization_eligible": False,
        "use_policy": frame_v1._use_policy(),
        "extractor": frame_v3._binding(),
        "selection": {
            "seed": frame_v3.SELECTION_SEED,
            "maximum_frames_per_scene": frame_v3.MAX_FRAMES_PER_SCENE,
            "requested_max_frames_per_scene": 1,
            "candidate_set": (
                "instance_intersection_semantic_intersection_sens_range_"
                "intersection_finite_homogeneous_pose"
            ),
            "selected_indices_strictly_below_sens_num_frames": True,
            "scene_unique": True,
        },
        "scenes": [
            {
                "scene_id": scene_id,
                "official_split": "val",
                "ordinal": ordinal,
                "output_directory": directory,
                "sensor_header": {"num_frames": 2},
                "pose_validation": {
                    "policy": "finite_and_homogeneous_pose_required_before_stable_selection",
                    "sens_stream_boundary_exact": True,
                    "all_frame_payload_headers_validated": True,
                },
                "selected_frame_indices": [1],
                "selected_frame_indices_sha256": builder.canonical_hash([1]),
                "calibration_outputs": calibration,
                "frames": [{"frame_index": 1, "outputs": outputs}],
            }
        ],
    }
    manifest["integrity"] = {
        "manifest_payload_sha256": builder.canonical_hash(manifest),
        "scene_ids_sha256": builder.canonical_hash([scene_id]),
    }
    _write_json(root / "manifest.json", manifest)
    digest = builder.file_sha256(root / "manifest.json")
    (root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )
    return root


def _record(index: int, image_path: Path) -> dict:
    near_label = f"chair {index} [instance {2 * index + 1}]"
    far_label = f"table {index} [instance {2 * index + 2}]"
    return {
        "sample_id": f"source-{index:02d}",
        "scene_id": f"scene8{index:03d}_00",
        "frame_index": index,
        "official_split": "val",
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "payload": [
            {"label": near_label, "rank": 1, "relative_depth": 0.0},
            {"label": far_label, "rank": 2, "relative_depth": 1.0},
        ],
        "sensor_measurements": [
            {
                "label": near_label,
                "category": f"chair {index}",
                "instance_id": 2 * index + 1,
                "bbox_2d": [100, 400, 300, 900],
            },
            {
                "label": far_label,
                "category": f"table {index}",
                "instance_id": 2 * index + 2,
                "bbox_2d": [600, 100, 800, 350],
            },
        ],
    }


def _candidate(index: int, image_path: Path, *, reused: bool) -> dict:
    record = _record(index, image_path)
    evidence = {
        "method": builder.REFERENT_METHOD,
        "target_category_visible_instance_counts": [1, 1],
        "both_target_categories_unique": True,
    }
    candidate = {
        "lineage": "reused_v51" if reused else "new_v52",
        "record": record,
        "features": {"synthetic": index},
        "referent_evidence": evidence,
    }
    if reused:
        decision_a = {"case_id": f"old-{index}", "correct": True}
        decision_b = {"case_id": f"old-{index}", "correct": True}
        binding = {
            "v51_sample_id": f"old-{index}",
            "scene_id": record["scene_id"],
            "source_sample_id": record["sample_id"],
            "source_record_sha256": builder.canonical_hash(record),
            "image_sha256": record["image_sha256"],
            "payload_sha256": builder.canonical_hash(record["payload"]),
            "label_sha256": "a" * 64,
            "reviewer_a_decision_sha256": builder.canonical_hash(decision_a),
            "reviewer_b_decision_sha256": builder.canonical_hash(decision_b),
        }
        binding["decision_pair_sha256"] = builder.canonical_hash(
            [
                binding["reviewer_a_decision_sha256"],
                binding["reviewer_b_decision_sha256"],
            ]
        )
        candidate["reuse_binding"] = binding
        candidate["reused_decisions"] = {
            "reviewer_a": decision_a,
            "reviewer_b": decision_b,
        }
    else:
        candidate["frame_root_path"] = "/synthetic/v3"
    return candidate


def _v51_fixture(reused: list[dict]) -> dict:
    reused_scenes = [item["record"]["scene_id"] for item in reused]
    prior = [f"scene7{index:03d}_00" for index in range(96)]
    manifest = {
        "selected": [{"scene_id": scene} for scene in reused_scenes],
        "inventory": {"excluded_scene_ids": prior},
        "provenance": {
            "old_schema5_audit_binding": {"scene_ids": prior[:40]},
            "frozen_sensor_canary_binding": {"scene_ids": prior[40:50]},
            "reviewed_rgb_canary_binding": {"scene_ids": prior[50:62]},
        },
    }
    bindings = [item["reuse_binding"] for item in reused]
    return {
        "status": "FORMAL_PASS",
        "files": {},
        "manifest_payload_sha256": "a" * 64,
        "selected_sha256": "b" * 64,
        "reused_count": builder.REUSED_TARGET,
        "nonreused_count": builder.NEW_TARGET,
        "reused_scene_ids": sorted(reused_scenes),
        "nonreused_scene_ids": prior[62:95],
        "all_selected_scene_ids": sorted(reused_scenes + prior[62:95]),
        "reviewer_keys": {"a": "old-a", "b": "old-b"},
        "reused_decision_bindings": bindings,
        "reused_decision_bindings_sha256": builder.canonical_hash(bindings),
        "reusable": reused,
        "manifest": manifest,
    }


def _freeze_fixture(tmp_path: Path) -> Path:
    images = []
    for index in range(builder.FORMAL_TARGET):
        path = tmp_path / "images" / f"{index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{index}".encode("ascii"))
        images.append(path)
    reused = [_candidate(index, images[index], reused=True) for index in range(7)]
    new = [_candidate(index, images[index], reused=False) for index in range(7, 40)]
    v51 = _v51_fixture(reused)
    inventory = {
        "complete_v3_roots": [],
        "source_exclusions_binding": {},
        "unique_referent_eligible_scenes": builder.NEW_TARGET,
    }
    output = tmp_path / "pack"
    builder.freeze_formal_pack(
        output,
        reused,
        new,
        inventory,
        v51,
        {
            "selected_records_rehashed": 40,
            "source_artifact_errors": 0,
        },
    )
    return output


def _vote(label: dict, reviewer: str, *, correct: bool = True) -> dict:
    return {
        "annotation_id": f"decision-{reviewer}-{label['sample_id']}",
        "case_id": label["sample_id"],
        "annotator_key": reviewer,
        "timestamp": "2026-08-23T00:00:00+00:00",
        "correct": correct,
        "error_code": None if correct else "wrong_object",
        "comment": "" if correct else "The displayed box targets another object.",
        "label_snapshot": label,
    }


def test_strict_interface_accepts_exact_schema_without_oracle_repair() -> None:
    parsed = interface.parse_grounding_response(
        '{"B":[500,100,700,300],"A":[100,200,300,600]}'
    )
    assert parsed.valid
    assert parsed.boxes == {
        "A": (100, 200, 300, 600),
        "B": (500, 100, 700, 300),
    }
    assert not interface.parse_grounding_response(
        '{"A":[1,2,3,4],"B":[5,6,7,8],"C":[9,10,11,12]}'
    ).valid
    assert not interface.parse_grounding_response(
        '{"A":[1,2,1,4],"B":[5,6,7,8]}'
    ).valid
    assert not interface.parse_grounding_response(
        '{"A":[true,2,3,4],"B":[5,6,7,8]}'
    ).valid
    assert interface.parse_spatial_response(" A\n").choice == "A"
    assert not interface.parse_spatial_response("A because it is lower").valid


def test_public_grounding_projection_contains_no_private_boxes_or_s_answer(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    label = builder._target_label(_candidate(0, image, reused=False), "A")
    request = interface.format_grounding_request(label, image)
    assert set(request["targets"][0]) == {
        "slot",
        "category",
        "referring_expression",
    }
    assert "audit_targets" not in request
    assert "boxes" not in request
    assert "nearer_slot" not in request


def test_real_frozen_v51_binding_reproduces_exact_seven_reuse_cases() -> None:
    binding = builder.load_v51_reuse_binding(
        builder.DEFAULT_V51_PACK, builder.DEFAULT_V51_REPORT
    )
    assert binding["reused_count"] == 7
    assert binding["nonreused_count"] == 33
    assert binding["reused_scene_ids"] == [
        "scene0164_00",
        "scene0164_01",
        "scene0207_02",
        "scene0256_02",
        "scene0435_01",
        "scene0559_01",
        "scene0595_00",
    ]
    assert len(binding["reused_decision_bindings"]) == 7
    assert all(
        item["decision_pair_sha256"]
        == builder.canonical_hash(
            [
                item["reviewer_a_decision_sha256"],
                item["reviewer_b_decision_sha256"],
            ]
        )
        for item in binding["reused_decision_bindings"]
    )


def test_partial_v3_root_is_ignored_and_explicit_load_fails_before_traversal(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "parent/in_progress_v3"
    (partial / "scene-partial").mkdir(parents=True)
    (partial / "scene-partial/garbage").write_bytes(b"partial")
    complete, ignored = builder.discover_complete_v3_roots(tmp_path / "parent")
    assert complete == []
    assert ignored == [str(partial.resolve())]
    with pytest.raises(builder.PackError, match="incomplete v3 frame root"):
        builder.load_v3_frame_root(partial)


def test_complete_v3_root_is_manifest_and_artifact_bound(tmp_path: Path) -> None:
    root = _complete_v3_root(tmp_path)
    binding = builder.load_v3_frame_root(root)
    assert binding["scene_ids"] == ["scene0700_00"]
    assert binding["frame_count"] == 1
    rgb = next(root.rglob("*-rgb.png"))
    rgb.write_bytes(b"tampered")
    with pytest.raises(builder.PackError, match="artifact (size|hash) drift"):
        builder.load_v3_frame_root(root)


def test_new_selection_is_scene_unique_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "image"
    image.write_bytes(b"image")
    candidates = []
    for scene_index in range(4):
        for frame_index in range(2):
            item = _candidate(scene_index, image, reused=False)
            item["record"]["scene_id"] = f"scene9{scene_index:03d}_00"
            item["record"]["frame_index"] = frame_index
            item["record"]["sample_id"] = f"scene-{scene_index}-frame-{frame_index}"
            item["score"] = frame_index
            candidates.append(item)
    monkeypatch.setattr(
        v51_filter, "_scene_record_score", lambda record: (record["frame_index"],)
    )
    first, stats = builder.select_new_scene_unique(candidates, target=3)
    second, _ = builder.select_new_scene_unique(reversed(candidates), target=3)
    assert [item["record"]["sample_id"] for item in first] == [
        item["record"]["sample_id"] for item in second
    ]
    assert all(item["record"]["frame_index"] == 1 for item in first)
    assert stats["eligible_scenes"] == 4
    assert stats["selected_scenes"] == 3


def test_formal_capacity_shortfall_fails_before_any_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "must-not-exist"
    fake_v51 = {
        "reusable": [object()] * 7,
        "status": "FORMAL_PASS",
        "files": {},
        "reused_scene_ids": [],
        "nonreused_scene_ids": [],
        "reused_decision_bindings_sha256": "a" * 64,
    }
    monkeypatch.setattr(builder, "parse_args", lambda argv=None: SimpleNamespace(
        stage="formal",
        output_dir=output,
        frame_root=[],
        v3_root_parent=tmp_path / "roots",
        v51_pack=tmp_path / "v51",
        v51_report=tmp_path / "v51.json",
        split_root=tmp_path,
        source_exclusions=tmp_path / "excluded.json",
        scannet_root=tmp_path,
        label_map=tmp_path / "labels.tsv",
    ))
    monkeypatch.setattr(builder, "load_v51_reuse_binding", lambda *args: fake_v51)
    monkeypatch.setattr(
        builder,
        "scan_new_candidates",
        lambda *args: (
            [],
            {
                "unique_referent_eligible_scenes": 0,
                "complete_v3_roots": [],
                "source_exclusions_binding": {},
            },
        ),
    )
    assert builder.main([]) == 2
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.*")) == []


def test_freeze_creates_33_case_queues_and_zero_annotation_decisions(
    tmp_path: Path,
) -> None:
    pack = _freeze_fixture(tmp_path)
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"] == 40
    assert manifest["review_contract"]["new_decisions_required_total"] == 66
    assert manifest["review_contract"]["builder_wrote_annotation_decisions"] == 0
    labels_a = builder._read_jsonl(
        pack / "annotator_a/labels/GS_dependency/unique_pair.jsonl"
    )
    labels_b = builder._read_jsonl(
        pack / "annotator_b/labels/GS_dependency/unique_pair.jsonl"
    )
    assert len(labels_a) == len(labels_b) == 33
    assert labels_a == labels_b
    assert not (pack / "annotator_a/annotations/annotations.jsonl").exists()
    assert not (pack / "annotator_b/annotations/annotations.jsonl").exists()
    reused = builder._read_jsonl(pack / "provenance/reused_v51_decisions.jsonl")
    assert len(reused) == 7


def test_review_compiler_never_synthesizes_votes_and_requires_independent_keys(
    tmp_path: Path,
) -> None:
    pack = _freeze_fixture(tmp_path)
    base_report = {
        "status": checker.PACK_VALID,
        "source_artifacts_rehashed": True,
    }
    pending = checker.compile_reviews(pack, base_report)
    assert pending["status"] == checker.PENDING_REVIEW
    assert pending["review"]["a_complete"] == 0
    assert pending["review"]["b_complete"] == 0

    labels = builder._read_jsonl(
        pack / "annotator_a/labels/GS_dependency/unique_pair.jsonl"
    )
    _write_jsonl(
        pack / "annotator_a/annotations/annotations.jsonl",
        [_vote(label, "reviewer-a-v52") for label in labels],
    )
    _write_jsonl(
        pack / "annotator_b/annotations/annotations.jsonl",
        [_vote(label, "reviewer-a-v52") for label in labels],
    )
    with pytest.raises(checker.CheckError, match="must be independent"):
        checker.compile_reviews(pack, base_report)

    _write_jsonl(
        pack / "annotator_b/annotations/annotations.jsonl",
        [_vote(label, "reviewer-b-v52") for label in labels],
    )
    complete = checker.compile_reviews(pack, base_report)
    assert complete["status"] == checker.FORMAL_PASS
    assert complete["review"]["resolved_correct_count"] == 40
    assert complete["review"]["required_human_decisions_A_plus_B"] == 66


def test_reviewer_c_must_cover_exactly_all_disagreements(tmp_path: Path) -> None:
    pack = _freeze_fixture(tmp_path)
    labels = builder._read_jsonl(
        pack / "annotator_a/labels/GS_dependency/unique_pair.jsonl"
    )
    votes_a = [_vote(label, "reviewer-a-v52") for label in labels]
    votes_b = [_vote(label, "reviewer-b-v52") for label in labels]
    votes_b[0] = _vote(labels[0], "reviewer-b-v52", correct=False)
    _write_jsonl(pack / "annotator_a/annotations/annotations.jsonl", votes_a)
    _write_jsonl(pack / "annotator_b/annotations/annotations.jsonl", votes_b)
    report = checker.compile_reviews(
        pack, {"status": checker.PACK_VALID, "source_artifacts_rehashed": True}
    )
    assert report["status"] == checker.PENDING_ADJUDICATION
    assert report["review"]["required_reviewer_c_decisions"] == 1

    adjudication = tmp_path / "reviewer_c.jsonl"
    _write_jsonl(adjudication, [_vote(labels[0], "reviewer-c-v52")])
    resolved = checker.compile_reviews(
        pack,
        {"status": checker.PACK_VALID, "source_artifacts_rehashed": True},
        adjudication,
    )
    assert resolved["status"] == checker.FORMAL_PASS
