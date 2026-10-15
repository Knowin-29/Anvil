"""Source-native region fixtures; production semantics require real source evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pytest

from tools import arxiv_source_scale as scale
from tools import map_sharerobot_benchmark as official_test


@pytest.fixture
def region_input(tmp_path):
    images = tmp_path / "images"
    relative = "batch/robot#episode_12/frame_0.png"
    image = images / relative
    image.parent.mkdir(parents=True)
    Image.fromarray(np.random.default_rng(71).integers(0, 256, (80, 100, 3), dtype=np.uint8)).save(image)
    official = {"id": 4, "image_path": relative,
                "meta_data": {"original_width": 100, "original_height": 80,
                              "original_dataset": "robot"},
                "instruction": "open the drawer",
                "affordance": {"x": 20., "y": 24., "width": 15., "height": 8.}}
    old_image = "/source/images/" + relative
    row = {"images": [old_image],
           "messages": [{"role": "user", "content": "<image>Open the drawer"},
                        {"role": "assistant", "content": "<bbox>[[200,300],[350,400]]</bbox>"}],
           "metadata": {"source": "sharerobot", "split": "train",
                        "output_type": "affordance_box", "src_id": "sharerobot_affordance_4",
                        "scene_id": str(Path(old_image).parent),
                        "src_meta": {"original_dataset": "robot", "original_size": [100, 80],
                                     "instruction": official["instruction"],
                                     "affordance_px": copy.deepcopy(official["affordance"]),
                                     "label_source": "human_annotated"}}}
    return images, image, official, row


def run_conversion(tmp_path, region_input, *, rows=None, protected=None, export=False):
    images, image, official, row = region_input
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in (rows or [row])))
    source_official = tmp_path / "official.json"
    source_official.write_text(json.dumps([official]))
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text(json.dumps(protected) + "\n" if protected else "")
    root = tmp_path / "output"
    result = scale.sharerobot_affordance(root, source, source_official, images,
        expected_source_sha256=scale.sha(source),
        expected_official_sha256=scale.sha(source_official),
        protected_fingerprints=[excluded], export=export)
    return result, root


def test_official_region_is_not_contact_point(region_input):
    images, _, official, row = region_input
    mapped = scale.sharerobot_region_reference(row, official, images)
    assert mapped["target"] == {"affordance": "open the drawer",
                               "region_bbox_2d": [200., 300., 350., 400.]}
    assert "contact_point_2d" not in mapped["target"]
    assert mapped["group_id"] == "ShareRobot:robot:robot#episode_12"


@pytest.mark.parametrize("fault", ["nontraining", "wrong_dataset", "changed_official",
                                  "wrong_id", "wrong_episode", "no_episode",
                                  "path_escape", "invalid_target", "nan", "zero_width"])
def test_malformed_or_unbound_source_is_rejected(region_input, fault):
    images, _, official, row = region_input
    row = copy.deepcopy(row); official = copy.deepcopy(official)
    if fault == "nontraining": row["metadata"]["split"] = "test"
    if fault == "wrong_dataset": row["metadata"]["src_meta"]["original_dataset"] = "other"
    if fault == "changed_official": official["instruction"] = "close the drawer"
    if fault == "wrong_id": official["id"] = 19
    if fault == "wrong_episode": row["metadata"]["scene_id"] = "/other/episode"
    if fault == "no_episode":
        official["image_path"] = "batch/robot/frame_0.png"
        row["images"][0] = "/source/images/" + official["image_path"]
        row["metadata"]["scene_id"] = str(Path(row["images"][0]).parent)
    if fault == "path_escape": official["image_path"] = "../../escape.png"
    if fault == "invalid_target": row["messages"][1]["content"] = "<bbox>[[0,0],[1,1]]</bbox>"
    if fault == "nan":
        official["affordance"]["x"] = float("nan")
        row["metadata"]["src_meta"]["affordance_px"]["x"] = float("nan")
    if fault == "zero_width":
        official["affordance"]["width"] = 0.
        row["metadata"]["src_meta"]["affordance_px"]["width"] = 0.
    with pytest.raises(ValueError):
        scale.sharerobot_region_reference(row, official, images)


def test_readonly_validation_creates_no_output(tmp_path, region_input):
    result, root = run_conversion(tmp_path, region_input)
    assert result["records"] == result["physical_episodes"] == 1
    assert result["state"] == "validated_no_write"
    assert not root.exists()


def test_export_is_immutable_train_only_and_hashed(tmp_path, region_input):
    result, root = run_conversion(tmp_path, region_input, export=True)
    assert result["train_sha256"] == scale.sha(root / "train.jsonl")
    record = next(scale.rows(root / "train.jsonl"))
    meta = record["anvil_meta"]
    assert meta["split"] == "train"
    assert meta["native_form"] == "affordance"
    assert meta["native_variant"] == scale.AFFORDANCE_REGION_CONTRACT
    assert meta["training_approved"]
    assert not meta["benchmark_eligible"]
    assert not meta["public_release_eligible"]
    assert not meta["legacy_full_contact_schema"]
    assert not meta["new_human_reviewed"]
    before = scale.sha(root / "train.jsonl")
    with pytest.raises(FileExistsError):
        run_conversion(tmp_path, region_input, export=True)
    assert scale.sha(root / "train.jsonl") == before


def test_protected_image_blocks_whole_episode(tmp_path, region_input):
    protected = scale.fingerprint(region_input[1])
    result, root = run_conversion(tmp_path, region_input, protected=protected, export=True)
    assert result["records"] == 0
    assert result["skipped"]["protected_evaluation_episode"] == 1
    assert (root / "train.jsonl").read_text() == ""


def test_duplicate_source_cannot_inflate_count(tmp_path, region_input):
    result, _ = run_conversion(tmp_path, region_input, rows=[region_input[3], region_input[3]])
    assert result["records"] == 1
    assert result["skipped"]["duplicate_source_id"] == 1


def test_wrong_image_dimensions_rejected(tmp_path, region_input):
    Image.new("RGB", (50, 40), "blue").save(region_input[1])
    result, _ = run_conversion(tmp_path, region_input)
    assert result["records"] == 0
    assert result["skipped"]["decoded_image_size_mismatch"] == 1


def test_explicit_eval_exclusions_required(tmp_path, region_input):
    with pytest.raises(ValueError, match="protected-evaluation"):
        scale.sharerobot_affordance(tmp_path / "output", "unused", "unused", region_input[0],
            expected_source_sha256="unused", expected_official_sha256="unused")


def test_changed_hash_not_admitted(tmp_path, region_input):
    images, _, official, row = region_input
    source = tmp_path / "source.jsonl"; source.write_text(json.dumps(row) + "\n")
    source_official = tmp_path / "official.json"; source_official.write_text(json.dumps([official]))
    excluded = tmp_path / "excluded.jsonl"; excluded.write_text("")
    with pytest.raises(ValueError, match="SHA256"):
        scale.sharerobot_affordance(tmp_path / "output", source, source_official, images,
            expected_source_sha256="0" * 64, expected_official_sha256=scale.sha(source_official),
            protected_fingerprints=[excluded])


@pytest.fixture
def official_test_input(tmp_path):
    root = tmp_path / "official"
    image = root / "affordance/images/example.png"
    image.parent.mkdir(parents=True)
    Image.fromarray(np.random.default_rng(211).integers(0, 256, (80, 100, 3), dtype=np.uint8)).save(image)
    row = {"id": 7, "image": "images/example.png", "conversations": [
        {"from": "human", "value": '<image>\nYou are a robot using the joint control. The task is "open the drawer". '
         'Please predict a possible affordance area of the end effector?'},
        {"from": "gpt", "value": "[20,24,35,32]"}]}
    return root, image, row, scale.fingerprint(image)


def test_official_test_preserves_region_and_test_assignment(official_test_input):
    root, image, row, fp = official_test_input
    mapped = official_test.native_row(row, root, "a" * 64, fp)
    assert json.loads(mapped["messages"][-1]["content"]) == {
        "affordance": "open the drawer", "region_bbox_2d": [200., 300., 350., 400.]}
    assert mapped["images"] == [str(image)]
    meta = mapped["anvil_meta"]
    assert meta["split"] == "test" and meta["benchmark_eligible"]
    assert not meta["training_approved"] and not meta["strong_model_verified"]
    assert not meta["new_human_reviewed"] and not meta["public_release_eligible"]
    assert meta["group_unit"] == "source_image" and meta["source_bbox_pixels"] == [20,24,35,32]
    assert "source_bbox_pixels" not in mapped["messages"][0]["content"]


@pytest.mark.parametrize("fault", ["escape", "absolute", "backslash", "wrong_role", "question", "nonint_id",
                                  "negative", "outside", "empty_box", "float", "nan", "unknown_format"])
def test_official_test_rejects_malformed_or_changed_contract(official_test_input, fault):
    root, _, row, fp = official_test_input
    if fault == "escape": row["image"] = "images/../../escape.png"
    if fault == "absolute": row["image"] = "/images/example.png"
    if fault == "backslash": row["image"] = "images/..\\escape.png"
    if fault == "wrong_role": row["conversations"][0]["from"] = "gpt"
    if fault == "question": row["conversations"][0]["value"] += "Please reason."
    if fault == "nonint_id": row["id"] = True
    if fault == "negative": row["conversations"][-1]["value"] = "[-1,24,35,32]"
    if fault == "outside": row["conversations"][-1]["value"] = "[20,24,101,32]"
    if fault == "empty_box": row["conversations"][-1]["value"] = "[35,24,35,32]"
    if fault == "float": row["conversations"][-1]["value"] = "[20.0,24,35,32]"
    if fault == "nan": row["conversations"][-1]["value"] = "[NaN,24,35,32]"
    if fault == "unknown_format": row["conversations"][-1]["value"] = '{"bbox":[20,24,35,32]}'
    with pytest.raises(ValueError): official_test.native_row(row, root, "a" * 64, fp)


@pytest.mark.parametrize("lfs", [True, False])
def test_official_file_verifies_actual_hub_object_identity(official_test_input, lfs):
    root, image, _, _ = official_test_input
    raw = image.read_bytes()
    entry = {"type": "file", "path": "affordance/images/example.png", "size": len(raw),
             "oid": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}
    if lfs: entry["lfs"] = {"oid": hashlib.sha256(raw).hexdigest()}
    assert official_test.verify_file(root, entry)["sha256"] == scale.sha(image)
    image.write_bytes(b"a" * len(raw))
    with pytest.raises(ValueError, match="mismatch"): official_test.verify_file(root, entry)


def test_official_source_symlink_cannot_escape_root(tmp_path):
    root = tmp_path / "official"; root.mkdir()
    (root / "images").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes root"):
        official_test.safe_path(root, "images/external.png")


@pytest.mark.parametrize("drift", [None, "index", "history", "inventory"])
def test_official_test_binds_current_and_historical_training(official_test_input, drift):
    root, _, _, fp = official_test_input
    history = root / "history.jsonl"; history.write_text(json.dumps(fp) + "\n")
    inventory = root / "inventory.json"
    inventory.write_text(json.dumps({"inputs": {"ANVIL_SOURCE_FINGERPRINTS": {
        "path": str(history), "sha256": scale.sha(history)}}}))
    index = root / "index.sqlite"
    with sqlite3.connect(index) as db:
        db.execute("CREATE TABLE images(file_sha256 TEXT, decoded_sha256 TEXT, phash64 TEXT, dhash64 TEXT, split TEXT)")
        db.execute("INSERT INTO images VALUES(?,?,?,?,'train')", [fp[k] for k in ("sha256","decoded_rgb_sha256","phash64","dhash64")])
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"selection_sqlite_sha256": scale.sha(index), "selection_complete": True,
        "eligibility": {"inventory": str(inventory), "inventory_sha256": scale.sha(inventory)}}))
    if drift:
        path = {"index": index, "history": history, "inventory": inventory}[drift]
        with path.open("ab") as stream: stream.write(b"\n")
        with pytest.raises(ValueError): official_test.load_training_protection(index, manifest)
    else:
        protected, binding = official_test.load_training_protection(index, manifest)
        assert protected.matches(fp)
        assert binding["pool_training_images"] == binding["historical_training_images"] == 1
