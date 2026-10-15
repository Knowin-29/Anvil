"""Independent geometry/source-binding regression tests, not model benchmarks."""
import copy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
import yaml


def test_specialist_disables_thinking_during_template_construction(monkeypatch):
    import sys
    from types import ModuleType
    from tools.train_prism_specialist import inference_template
    module = ModuleType('swift.template')
    calls = []
    def get_template(processor, **kwargs):
        calls.append((processor, kwargs))
        return object()
    module.get_template = get_template
    monkeypatch.setitem(sys.modules, 'swift.template', module)
    processor = object()
    inference_template(processor)
    assert calls == [(processor, {'template_type':'qwen3_5', 'max_length':32768,
                                  'truncation_strategy':'raise', 'enable_thinking':False})]

from tools.map_prism_native import (CONTRACT, EXCLUSION_CONTRACT, LENGTH_UNIT,
                                   SPLIT_CONTRACT, canonical, convert, digest, export_sft,
                                   image_fingerprint, map_record, partition_sft,
                                   validate_native_gripper, validate_sft_record)


@pytest.fixture
def source_rows():
    # Rz(90 degrees): gripper +x maps to camera +y, +z stays +z.
    pose = [[0., -1., 0., .1], [1., 0., 0., .2], [0., 0., 1., 1.], [0., 0., 0., 1.]]
    annot = {"object_category": "Cup", "object_id": "fixture", "annotation_id": "cup-1",
             "grasp_description": "grasp the rim", "grasp_id": 1}
    scene = "a" * 32
    view = {"scene_id": scene, "view_id": "view_0", "width": 8, "height": 6,
            "intrinsics": [[4., 0., 4.], [0., 4., 3.], [0., 0., 1.]],
            "object_names": ["Cup_fixture"], "raw_sha256": "0" * 64, "image_sha256": "1" * 64}
    grasp = {"scene_id": scene, "view_id": "view_0", "obs_id": "obs_0",
             "grasp_pose_camera": pose, "grasp_point_camera": [.25, .25, 1.],
             "grasp_point_px": [5., 4.], "annotation_yaml": yaml.safe_dump(annot)}
    task = {"scene_id": scene, "view_id": "view_0", "obs_id": "obs_0",
            "scene_path": scene + ".hdf5", "task": "Pick up the cup.",
            "matching_grasp_desc": "grasp the rim", "original_grasp_desc": "source text"}
    return view, grasp, task


def make_record(rows):
    return map_record(*rows, [0., 0., 1000., 1000.])


def test_known_rotation_and_contact_not_gripper_origin(source_rows):
    record = make_record(source_rows)
    answer = record["answer"]
    assert answer["jaw_axis_camera"] == [0., 1., 0.]
    assert answer["approach_axis_camera"] == [0., 0., 1.]
    assert answer["contact_point_camera"] == [.25, .25, 1.]
    assert answer["contact_point_camera"] != [row[3] for row in answer["camera_from_gripper"][:3]]
    assert answer["length_unit"] == LENGTH_UNIT
    assert record["split"] == "unassigned_source_only"
    assert not record["training_approved"] and not record["legacy_v3_admitted"]
    assert "wrist_deg" not in answer and "approach" not in answer
    assert record["id"] == make_record(source_rows)["id"]


def test_non_z_approach_rotation(source_rows):
    view, grasp, task = source_rows
    # Ry(90): +z approaches along camera +x; jaws follow camera -z.
    grasp["grasp_pose_camera"] = [[0., 0., 1., .1], [0., 1., 0., .2], [-1., 0., 0., 1.], [0., 0., 0., 1.]]
    answer = make_record(source_rows)["answer"]
    assert answer["approach_axis_camera"] == [1., 0., 0.]
    assert answer["jaw_axis_camera"] == [0., 0., -1.]


@pytest.mark.parametrize("mutation,match", [
    (lambda a: a["camera_from_gripper"][0].__setitem__(1, 1.), "rigid SE3"),
    (lambda a: a["camera_from_gripper"][3].__setitem__(0, .5), "rigid SE3"),
    (lambda a: a["camera_from_gripper"][0].__setitem__(3, float("nan")), "SE3 pose"),
    (lambda a: a.__setitem__("jaw_axis_camera", [1., 0., 0.]), "jaw_axis"),
    (lambda a: a.__setitem__("approach_axis_camera", [0., 0., -1.]), "approach_axis"),
    (lambda a: a.__setitem__("contact_point_px", [4., 4.]), "projection"),
    (lambda a: a.__setitem__("contact_point_px", [8., 4.]), "outside"),
    (lambda a: a.__setitem__("contact_point_camera", [.25, .25, -1.]), "outside"),
    (lambda a: a.__setitem__("length_unit", "m"), "units"),
    (lambda a: a.__setitem__("frame", "world"), "frame"),
    (lambda a: a.__setitem__("approach", "top"), "contract"),
])
def test_independent_validator_rejects_corrupted_geometry(source_rows, mutation, match):
    answer = make_record(source_rows)["answer"]
    mutation(answer)
    view = source_rows[0]
    with pytest.raises(ValueError, match=match):
        validate_native_gripper(answer, view["intrinsics"], view["width"], view["height"])


@pytest.mark.parametrize("field,value", [("scene_id", "b" * 32), ("view_id", "view_1"), ("obs_id", "obs_1")])
def test_no_cross_entity_join(source_rows, field, value):
    source_rows[2][field] = value
    with pytest.raises(ValueError, match="join"):
        make_record(source_rows)


def test_task_matching_description_and_legacy_contract_rejected(source_rows):
    from trains.flywheel.alignment_schema import Candidate
    record = make_record(source_rows)
    with pytest.raises(ValueError):
        Candidate.model_validate({"task_id": "dof", "form": "dof", "answer": [record["answer"]],
            "reasoning_trace": [{"evidence_ids": ["fixture"], "rule": "raw source", "conclusion": "fixture"}]})
    source_rows[2]["matching_grasp_desc"] = "unsupported invented grasp"
    with pytest.raises(ValueError, match="description mismatch"):
        make_record(source_rows)


def raw_fixture(tmp_path, rows, tasks=1):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "raw"
    source.mkdir()
    view, grasp, task = copy.deepcopy(rows)
    scene = source / view["scene_id"]
    scene.mkdir()
    raw = scene / (view["scene_id"] + ".hdf5")
    image = scene / "view_0.png"
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[:, :, 1] = 80
    Image.fromarray(rgb).save(image)
    with h5py.File(raw, "w") as f:
        v = f.create_group("view_0")
        v["rgb"] = rgb
        v["seg"] = np.ones((6, 8), dtype=np.uint8)
        v["cam_params"] = view["intrinsics"]
        v["object_names"] = [b"Cup_fixture"]
        obs = v.create_group("obs_0")
        obs["grasp_pose"] = grasp["grasp_pose_camera"]
        obs["grasp_point"] = grasp["grasp_point_camera"]
        obs["grasp_point_px"] = grasp["grasp_point_px"]
        obs["annot"] = grasp["annotation_yaml"]
    view.update(raw_hdf5=str(raw), raw_sha256=digest(raw), image=str(image), image_sha256=digest(image))
    grasp.update(raw_hdf5=str(raw), raw_sha256=digest(raw), image=str(image))
    for filename, records in [("views.jsonl", [view]), ("grasps.jsonl", [grasp]), ("tasks.jsonl", [task] * tasks)]:
        (source / filename).write_text("".join(canonical(row) + "\n" for row in records))
    binding = {"archive_sha256": "fixture-archive", "metadata_sha256": "fixture-csv",
               "files": {n: digest(source / n) for n in ["views.jsonl", "grasps.jsonl", "tasks.jsonl"]}}
    (source / "manifest.json").write_text(json.dumps(binding))
    return source


def test_full_raw_binding_and_no_local_path_leak(tmp_path, source_rows):
    source = raw_fixture(tmp_path, source_rows, tasks=2)
    output = tmp_path / "output"
    result = convert(source, output)
    assert result["counts"] == {"tasks_scanned": 2, "accepted_task_records": 1,
        "unique_scenes": 1, "unique_images": 1, "unique_grasp_observations": 1, "rejected_tasks": 1}
    text = (output / "records.jsonl").read_text()
    assert str(tmp_path) not in text
    record = json.loads(text)
    assert record["contract"] == CONTRACT
    assert record["input"]["entity"]["bbox_2d"] == [0., 0., 1000., 1000.]
    assert result["files"]["records.jsonl"] == digest(output / "records.jsonl")
    with pytest.raises(FileExistsError):
        convert(source, output)


def test_raw_corruption_fails_closed(tmp_path, source_rows):
    source = raw_fixture(tmp_path, source_rows)
    raw = next(source.glob("*/*.hdf5"))
    with raw.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="HDF5 checksum"):
        convert(source, tmp_path / "output")
    assert not (tmp_path / "output/manifest.json").exists()


def test_metadata_corruption_fails_before_output(tmp_path, source_rows):
    source = raw_fixture(tmp_path, source_rows)
    with (source / "tasks.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="manifest checksum"):
        convert(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_rehashed_metadata_cannot_forge_source_pose(tmp_path, source_rows):
    source = raw_fixture(tmp_path, source_rows)
    path = source / "grasps.jsonl"
    row = json.loads(path.read_text())
    row["grasp_pose_camera"][0][3] += .05
    path.write_text(canonical(row) + "\n")
    binding = json.loads((source / "manifest.json").read_text())
    binding["files"]["grasps.jsonl"] = digest(path)
    (source / "manifest.json").write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="disagrees with raw"):
        convert(source, tmp_path / "output")


def test_smoke_limit_is_explicit(tmp_path, source_rows):
    source = raw_fixture(tmp_path, source_rows, tasks=3)
    assert convert(source, tmp_path / "output", limit=1)["counts"]["tasks_scanned"] == 1


def training_fixture(tmp_path, source_rows):
    raw = raw_fixture(tmp_path, source_rows)
    native = tmp_path / "native"
    convert(raw, native)
    source_sha = digest(native / "records.jsonl")
    group = "PRISM:" + source_rows[0]["scene_id"]
    assignments = {"schema": SPLIT_CONTRACT, "contract": CONTRACT,
        "source_records_sha256": source_sha,
        "assignments": {group: {"split": "train", "training_approved": True}}}
    exclusions = {"schema": EXCLUSION_CONTRACT, "contract": CONTRACT,
        "source_records_sha256": source_sha, "protected_group_ids": [],
        "protected_image_sha256": [], "evidence_sha256": ["e" * 64]}
    assignment_path = tmp_path / "assignments.json"
    exclusion_path = tmp_path / "exclusions.json"
    assignment_path.write_text(json.dumps(assignments))
    exclusion_path.write_text(json.dumps(exclusions))
    return native, raw, assignment_path, exclusion_path


def test_explicit_sft_adapter_preserves_native_target_and_binding(tmp_path, source_rows):
    native, raw, assignments, exclusions = training_fixture(tmp_path, source_rows)
    out = tmp_path / "sft"
    result = export_sft(native, out, raw, assignments, exclusions, CONTRACT)
    assert result["counts"]["training_records"] == 1
    training = json.loads((out / "train.jsonl").read_text())
    source = json.loads((native / "records.jsonl").read_text())
    assert json.loads(training["messages"][1]["content"]) == source["answer"]
    assert training["images"] == [str(raw / source["images"][0])]
    assert training["anvil_meta"]["source_provenance"] == source["provenance"]
    assert training["anvil_meta"]["group_unit"] == "physical_scene"
    assert training["anvil_meta"]["training_approved"] is True
    assert training["anvil_meta"]["legacy_v3_admitted"] is False
    assert training["anvil_meta"]["public_release_eligible"] is False
    assert result["policy_bindings"]["scene_assignments_sha256"] == digest(assignments)
    assert json.loads(training["messages"][0]["content"].split("\n", 1)[1])["input"] == source["input"]
    # Independent admission check must reject a swapped scene, not trust metadata.
    training["anvil_meta"]["group_id"] = "PRISM:" + "b" * 32
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_sft_record(training, source, {"split": "train", "training_approved": True},
                            json.loads(exclusions.read_text()), result["policy_bindings"], raw / source["images"][0])


@pytest.mark.parametrize("kind", ["scene", "image"])
def test_eval_scene_and_image_exclusions_fail_closed(tmp_path, source_rows, kind):
    native, raw, assignments, exclusions = training_fixture(tmp_path, source_rows)
    source = json.loads((native / "records.jsonl").read_text())
    policy = json.loads(exclusions.read_text())
    if kind == "scene":
        policy["protected_group_ids"] = [source["group_id"]]
    else:
        policy["protected_image_sha256"] = [source["provenance"]["image_sha256"]]
    exclusions.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="held-out"):
        export_sft(native, tmp_path / "sft", raw, assignments, exclusions, CONTRACT)
    assert not (tmp_path / "sft").exists()


@pytest.mark.parametrize("policy_change,match", [
    (lambda p: p["assignments"]["PRISM:" + "a" * 32].__setitem__("training_approved", False), "explicit approval"),
    (lambda p: p["assignments"]["PRISM:" + "a" * 32].__setitem__("split", "test"), "held-out"),
    (lambda p: p.__setitem__("source_records_sha256", "f" * 64), "checksum mismatch"),
    (lambda p: p.__setitem__("contract", "anvil-gen-v3"), "contract"),
    (lambda p: p["assignments"].__setitem__("PRISM:" + "b" * 32,
        {"split": "test", "training_approved": False}), "cover exactly"),
])
def test_no_implicit_training_approval_or_wrong_partition(tmp_path, source_rows, policy_change, match):
    native, raw, assignments, exclusions = training_fixture(tmp_path, source_rows)
    policy = json.loads(assignments.read_text())
    policy_change(policy)
    assignments.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match=match):
        export_sft(native, tmp_path / "sft", raw, assignments, exclusions, CONTRACT)


def test_explicit_contract_opt_in_required(tmp_path, source_rows):
    native, raw, assignments, exclusions = training_fixture(tmp_path, source_rows)
    with pytest.raises(ValueError, match="opt-in"):
        export_sft(native, tmp_path / "sft", raw, assignments, exclusions, "anvil-gen-v3")


def test_geometry_revalidated_even_if_native_json_and_manifest_rehashed(tmp_path, source_rows):
    native, raw, assignments, exclusions = training_fixture(tmp_path, source_rows)
    path = native / "records.jsonl"
    source = json.loads(path.read_text())
    source["answer"]["jaw_axis_camera"] = [1., 0., 0.]
    path.write_text(canonical(source) + "\n")
    new_sha = digest(path)
    manifest = json.loads((native / "manifest.json").read_text())
    manifest["files"]["records.jsonl"] = new_sha
    (native / "manifest.json").write_text(json.dumps(manifest))
    for policy_path in [assignments, exclusions]:
        policy = json.loads(policy_path.read_text())
        policy["source_records_sha256"] = new_sha
        policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="jaw_axis"):
        export_sft(native, tmp_path / "sft", raw, assignments, exclusions, CONTRACT)


def test_heldout_assignment_is_not_promoted(tmp_path, source_rows):
    native, raw, assignments, exclusions = training_fixture(tmp_path, source_rows)
    policy = json.loads(assignments.read_text())
    policy["assignments"]["PRISM:" + "a" * 32] = {"split": "test", "training_approved": False}
    assignments.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="no explicitly approved"):
        export_sft(native, tmp_path / "sft", raw, assignments, exclusions, CONTRACT)
    assert not (tmp_path / "sft").exists()


def test_sft_image_hash_rechecked(tmp_path, source_rows):
    native, raw, assignments, exclusions = training_fixture(tmp_path, source_rows)
    image = next(raw.glob("*/*.png"))
    with image.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="image hash"):
        export_sft(native, tmp_path / "sft", raw, assignments, exclusions, CONTRACT)


def test_different_scene_labels_cannot_hide_heldout_image_reuse(tmp_path, source_rows):
    native, raw, assignments, exclusions = training_fixture(tmp_path, source_rows)
    path = native / "records.jsonl"
    first = json.loads(path.read_text())
    second = copy.deepcopy(first)
    old_scene, new_scene = "a" * 32, "b" * 32
    second["group_id"] = "PRISM:" + new_scene
    second["provenance"]["scene_id"] = new_scene
    second["images"] = [new_scene + "/view_0.png"]
    for item in [second["input"]["entity"], second["answer"]]:
        item["entity_id"] = item["entity_id"].replace(old_scene, new_scene)
    identity = {k: second["provenance"][k] for k in ["scene_id", "view_id", "obs_id"]}
    identity.update(task=second["input"]["task"], contract=CONTRACT)
    second["id"] = "prism-gripper-" + hashlib.sha256(canonical(identity).encode()).hexdigest()
    path.write_text(canonical(first) + "\n" + canonical(second) + "\n")
    new_sha = digest(path)
    manifest = json.loads((native / "manifest.json").read_text())
    manifest["files"]["records.jsonl"] = new_sha
    (native / "manifest.json").write_text(json.dumps(manifest))
    for policy_path in [assignments, exclusions]:
        policy = json.loads(policy_path.read_text())
        policy["source_records_sha256"] = new_sha
        if policy_path == assignments:
            policy["assignments"][second["group_id"]] = {"split": "test", "training_approved": False}
        policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="held-out scene/image"):
        export_sft(native, tmp_path / "sft", raw, assignments, exclusions, CONTRACT)
    assert not (tmp_path / "sft").exists()


def partition_fixture(tmp_path, source_rows):
    pytest.importorskip("cv2")
    native, raw = tmp_path / "native", tmp_path / "raw"
    native.mkdir(); raw.mkdir()
    records, views, images = [], [], []
    for i, token in enumerate("abcdef"):
        view, grasp, task = copy.deepcopy(source_rows)
        scene = token * 32
        for row in [view, grasp, task]:
            row["scene_id"] = scene
        task["scene_path"] = scene + ".hdf5"
        directory = raw / scene
        directory.mkdir()
        image = directory / "view_0.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (6, 8, 3), dtype=np.uint8)).save(image)
        view.update(image=str(image), image_sha256=digest(image))
        records.append(map_record(view, grasp, task, [0., 0., 1000., 1000.]))
        views.append(view); images.append(image)
    (native / "records.jsonl").write_text("".join(canonical(r) + "\n" for r in records))
    (native / "manifest.json").write_text(json.dumps({"contract": CONTRACT,
        "state": "source_native_extension_materialized", "files": {"records.jsonl": digest(native / "records.jsonl")}}))
    (raw / "views.jsonl").write_text("".join(canonical(r) + "\n" for r in views))
    (raw / "manifest.json").write_text(json.dumps({"files": {"views.jsonl": digest(raw / "views.jsonl")}}))
    protected_records, protected_fps, training_fps = [tmp_path / name for name in ["bench.jsonl", "eval_fps.jsonl", "train_fps.jsonl"]]
    protected_records.write_text(canonical({"images": [str(images[0])]}) + "\n")
    protected_fps.write_text(canonical(image_fingerprint(images[1])) + "\n")
    training_fps.write_text(canonical(image_fingerprint(images[2])) + "\n")
    return native, raw, protected_records, protected_fps, training_fps


def test_programmatic_partition_scans_real_images_and_freezes_scenes(tmp_path, source_rows):
    native, raw, bench, heldout_fps, history = partition_fixture(tmp_path, source_rows)
    out = tmp_path / "sft"
    result = partition_sft(native, out, raw, [bench], [heldout_fps], [history], CONTRACT, seed="test-seed")
    assert result["counts"]["training_records"] == 3
    assert result["counts"]["evaluation_records"] == 1
    assert result["counts"]["excluded_records"] == 2
    assert result["counts"]["total_source_scenes"] == 6
    audit = json.loads((tmp_path / "sft_policy/partition_audit.json").read_text())
    assignments = {r["group_id"]: r["split"] for r in audit["scene_decisions"]}
    assert assignments["PRISM:" + "a" * 32] == "exclude"
    assert assignments["PRISM:" + "b" * 32] == "exclude"
    assert assignments["PRISM:" + "c" * 32] == "train"
    assert audit["human_reviewed"] is False
    assert {e["sha256"] for e in audit["evidence"]} == {digest(p) for p in [bench, heldout_fps, history]}
    training = [json.loads(line) for line in (out / "train.jsonl").open()]
    evaluation = [json.loads(line) for line in (out / "evaluation.jsonl").open()]
    assert all(r["anvil_meta"]["training_approved"] is False for r in evaluation)
    assert not {r["anvil_meta"]["group_id"] for r in training} & {r["anvil_meta"]["group_id"] for r in evaluation}
    second = partition_sft(native, tmp_path / "repeat", raw, [bench], [heldout_fps], [history], CONTRACT, seed="test-seed")
    assert second["files"] == result["files"]


def test_all_source_views_checked_even_without_task_targets(tmp_path, source_rows):
    native, raw, bench, heldout_fps, history = partition_fixture(tmp_path, source_rows)
    views = [json.loads(line) for line in (raw / "views.jsonl").open()]
    extra = copy.deepcopy(views[3])
    extra["view_id"] = "view_1"
    image = raw / ("d" * 32) / "view_1.png"
    image.write_bytes((raw / ("a" * 32) / "view_0.png").read_bytes())
    extra.update(image=str(image), image_sha256=digest(image))
    (raw / "views.jsonl").write_text("".join(canonical(v) + "\n" for v in [*views, extra]))
    (raw / "manifest.json").write_text(json.dumps({"files": {"views.jsonl": digest(raw / "views.jsonl")}}))
    result = partition_sft(native, tmp_path / "sft", raw, [bench], [heldout_fps], [history], CONTRACT)
    assert result["counts"]["all_source_views_checked"] == 7
    audit = json.loads((tmp_path / "sft_policy/partition_audit.json").read_text())
    assignments = {r["group_id"]: r["split"] for r in audit["scene_decisions"]}
    assert assignments["PRISM:" + "d" * 32] == "exclude"


def test_partition_cannot_bypass_empty_protection_arguments(tmp_path, source_rows):
    native, raw, bench, heldout_fps, history = partition_fixture(tmp_path, source_rows)
    with pytest.raises(ValueError, match="all required"):
        partition_sft(native, tmp_path / "sft", raw, [], [heldout_fps], [history], CONTRACT)
    assert not (tmp_path / "sft").exists()


@pytest.fixture
def specialist_split(tmp_path, source_rows):
    native, raw, bench, heldout_fps, history = partition_fixture(tmp_path, source_rows)
    out = tmp_path / "sft"
    binding = partition_sft(native, out, raw, [bench], [heldout_fps], [history], CONTRACT, seed="test-seed")
    train = [json.loads(line) for line in (out / "train.jsonl").open()]
    heldout = [json.loads(line) for line in (out / "evaluation.jsonl").open()]
    return train, heldout, binding


def test_specialist_cpu_admission_and_independent_metric_fixtures(specialist_split):
    from tools.train_prism_specialist import metric_selftest, validate_dataset_split
    train, heldout, binding = specialist_split
    result = validate_dataset_split(train, heldout, binding)
    assert result == {"train_records": 3, "heldout_records": 1, "train_scenes": 3,
                      "heldout_scenes": 1, "images_checked": 4}
    metric_selftest()  # Known identity, 90-degree rotation, projection and reflection.


@pytest.mark.parametrize("field,value", [
    ("native_contract", "anvil-gen-v3"),
    ("group_id", "PRISM:" + "0" * 32),
    ("training_approved", False),
    ("policy_bindings", {}),
])
def test_specialist_rejects_relabelled_contract_or_scene(specialist_split, field, value):
    from tools.train_prism_specialist import validate_dataset_split
    train, heldout, binding = specialist_split
    train[0]["anvil_meta"][field] = value
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_dataset_split(train, heldout, binding)


def test_specialist_rejects_reused_eval_scene_even_with_new_row_id(specialist_split):
    from tools.train_prism_specialist import validate_dataset_split
    train, heldout, binding = specialist_split
    reused = copy.deepcopy(train[0])
    reused["id"] += "-heldout"
    reused["anvil_meta"].update(split="test", training_approved=False)
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_dataset_split(train, [reused], binding)


def test_specialist_rechecks_actual_image_bytes(specialist_split):
    from pathlib import Path
    from tools.train_prism_specialist import validate_dataset_split
    train, heldout, binding = specialist_split
    image = Path(train[0]["images"][0])
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="image bytes"):
        validate_dataset_split(train, heldout, binding)


def test_specialist_fixed_training_preset_has_no_heldout_tuning(monkeypatch, tmp_path):
    import tools.train_prism_specialist as specialist
    monkeypatch.setattr(specialist, "RUN", tmp_path / "run")
    monkeypatch.setattr(specialist, "BASE", tmp_path / "model")
    formal, smoke = specialist.training_command(), specialist.training_command(smoke=True)
    def value(command, flag):
        return command[command.index(flag) + 1]
    assert value(formal, "--max_steps") == "512"
    assert value(smoke, "--max_steps") == "2"
    assert value(formal, "--learning_rate") == "1e-4"
    assert value(formal, "--gradient_accumulation_steps") == "4"
    assert value(formal, "--seed") == "17"
    assert value(formal, "--freeze_vit") == "true"
    assert value(formal, "--freeze_aligner") == "true"
    assert value(formal, "--eval_strategy") == "no"
    assert "--resume_from_checkpoint" not in formal
    assert all("evaluation.jsonl" not in argument for argument in formal)


@pytest.mark.parametrize("memory,pid,utilization,idle", [
    (728, False, 0, True), (1024, False, 0, True), (1025, False, 0, False),
    (0, True, 0, False), (0, False, 1, False),
])
def test_specialist_gpu_idle_checks_do_not_evict_processes(memory, pid, utilization, idle):
    from tools.train_prism_specialist import is_gpu_idle
    assert is_gpu_idle({"memory_mib": memory, "compute_pid_present": pid,
                       "utilization_percent": utilization}) is idle


def test_specialist_gpu_allocation_double_samples_while_locked(monkeypatch, tmp_path):
    import fcntl
    from types import SimpleNamespace
    import tools.train_prism_specialist as specialist
    args = SimpleNamespace(lock_root=tmp_path / "locks", gpu="4")
    monkeypatch.setattr(specialist, "ARGS", args)
    monkeypatch.setattr(specialist, "RUN", tmp_path)
    monkeypatch.setattr(specialist.subprocess, "check_output", lambda *a, **k: "GPU-fixture\n")
    observed, sleeps = [], []
    def sample(uuid):
        with (args.lock_root / (uuid + ".lock")).open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed.append(uuid)
        return {"gpu_uuid": uuid, "memory_mib": 728, "compute_pid_present": False,
                "utilization_percent": 0, "sampled_at": len(observed)}
    monkeypatch.setattr(specialist, "gpu_idle_snapshot", sample)
    monkeypatch.setattr(specialist.time, "sleep", sleeps.append)
    lock, uuid = specialist.acquire_gpu()
    try:
        assert uuid == "GPU-fixture" and observed == [uuid, uuid] and sleeps == [2]
        assert len(json.loads((tmp_path / "gpu_allocation.json").read_text())["idle_checks_while_locked"]) == 2
    finally:
        lock.close()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_specialist_nonstandard_numeric_literals_are_not_json(token):
    from tools.train_prism_specialist import metrics
    assert metrics('{"field":' + token + '}', {})["json_valid"] is False


def test_specialist_children_use_frozen_code_not_mutable_entry(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import tools.train_prism_specialist as specialist
    monkeypatch.setattr(specialist, "RUN", tmp_path / "run")
    monkeypatch.setattr(specialist, "BASE", tmp_path / "model")
    monkeypatch.setattr(specialist, "TRAIN", tmp_path / "train.jsonl")
    monkeypatch.setattr(specialist, "EVALUATION", tmp_path / "evaluation.jsonl")
    monkeypatch.setattr(specialist, "DATA_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(specialist, "ARGS", SimpleNamespace(gpu="4", lock_root=tmp_path / "locks", runtime_parent=tmp_path))
    for stage in ["cpu-encode", "evaluate-base", "evaluate-specialist"]:
        command = specialist.child_command(stage)
        assert command[2] == str(tmp_path / "run/code/train_prism_specialist.py")
        assert command[-2:] == ["--stage", stage]
