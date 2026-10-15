"""CPU-only contract tests for faithful ALFRED pre-action mapping."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SPEC = importlib.util.spec_from_file_location(
    "map_alfred_native", Path(__file__).resolve().parents[1] / "tools/map_alfred_native.py")
mapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapper)


class TestExplicitPrompt(unittest.TestCase):
    def test_default_is_byte_preserving(self):
        for prompt in ("<image>legacy task", "<image>\n" + mapper.SFT_INSTRUCTION + "把杯子放好"):
            self.assertEqual(mapper.render_sft_prompt(prompt), prompt)

    def test_explicit_contract_preserves_complete_goal(self):
        goal = 'put the mug away\nTask goal: do not edit this source text 把杯子放好'
        source = "<image>\n" + mapper.SFT_INSTRUCTION + goal
        rendered = mapper.render_sft_prompt(source, "explicit-schema-v1")
        self.assertTrue(rendered.endswith("\nTask goal: " + goal))
        self.assertEqual(rendered.count("<image>"), 1)
        schema_text = rendered.split("output schema: ", 1)[1].split("\n", 1)[0]
        self.assertEqual(json.loads(schema_text), mapper.SFT_OUTPUT_SCHEMA)
        self.assertIn("not a guarantee of executability", rendered)
        self.assertIn("never a dictionary", rendered)
        self.assertIn("not a high-level macro", rendered)

    def test_rejects_unknown_policy_or_modified_instruction(self):
        with self.assertRaisesRegex(ValueError, "Unknown ALFRED"):
            mapper.render_sft_prompt("task", "automatic-repair")
        for source in ("task", "<image>\n" + mapper.SFT_INSTRUCTION, "<image>\nchanged instruction"):
            with self.assertRaisesRegex(ValueError, "unchanged source"):
                mapper.render_sft_prompt(source, "explicit-schema-v1")
        rendered = mapper.render_sft_prompt("<image>\n" + mapper.SFT_INSTRUCTION + "task", "explicit-schema-v1")
        with self.assertRaisesRegex(ValueError, "unchanged source"):
            mapper.render_sft_prompt(rendered, "explicit-schema-v1")

    def test_schema_keeps_high_and_low_level_actions_distinct(self):
        fields = mapper.SFT_OUTPUT_SCHEMA["properties"]
        self.assertEqual(set(fields), set(mapper.SFT_OUTPUT_SCHEMA["required"]))
        self.assertFalse(mapper.SFT_OUTPUT_SCHEMA["additionalProperties"])
        self.assertEqual(set(fields["next_expert_action"]["enum"]), set(mapper.FRAME_COUNTS) - {"CoolObject"})
        self.assertNotIn("ToggleObject", fields["next_expert_action"]["enum"])
        self.assertEqual(set(fields["target_role"]["enum"]), {"interaction_object", "destination_receptacle"})
        steps = fields["remaining_high_level_actions"]
        self.assertEqual(steps["minItems"], 1)
        self.assertEqual(steps["items"]["properties"]["args"], {"type": "array", "items": {"type": "string"}})
        high = steps["items"]["properties"]["action"]["enum"]
        self.assertIn("ToggleObject", high)
        self.assertIn("NoOp", high)
        self.assertNotIn("ToggleObjectOn", high)


def fixture(action="PickupObject", high_action=None, count=15):
    low = {"high_idx": 0, "api_action": {"action": action, "objectId": "Mug|0|1|2"},
           "discrete_action": {"action": action, "args": {}}}
    frames = [{"low_idx": 0, "high_idx": 0, "image_name": f"{idx:09}.png"}
              for idx in range(10, 10 + count)]
    trajectory = {"plan": {"low_actions": [low], "high_pddl": [
        {"high_idx": 0, "planner_action": {"action": high_action or action}}]}, "images": frames}
    row = {"low_idx": 0, "high_idx": 0, "source_action": copy.deepcopy(low),
           "frames": [{"metadata": copy.deepcopy(frame), "member": frame["image_name"]}
                      for frame in frames]}
    return row, trajectory


class TemporalTests(unittest.TestCase):
    def test_selects_first_before_not_final_after(self):
        row, trajectory = fixture()
        selected, counts, mode = mapper.select_preaction(row, trajectory)
        self.assertEqual(selected["metadata"]["image_name"], "000000010.png")
        self.assertEqual(counts, (5, 0, 10))
        self.assertEqual(mode, "PickupObject")

    def test_cooling_close_uses_macro_counts(self):
        row, trajectory = fixture("CloseObject", "CoolObject", 38)
        target = row["source_action"]["api_action"]["objectId"]
        trajectory["plan"]["low_actions"] = [
            {"high_idx": 0, "api_action": {"action": "PutObject", "receptacleObjectId": target}},
            trajectory["plan"]["low_actions"][0],
            {"high_idx": 0, "api_action": {"action": "OpenObject", "objectId": target}},
        ]
        row["low_idx"] = 1
        for frame in trajectory["images"]:
            frame["low_idx"] = 1
        for frame in row["frames"]:
            frame["metadata"]["low_idx"] = 1
        selected, counts, mode = mapper.select_preaction(row, trajectory)
        self.assertEqual(selected["metadata"]["image_name"], "000000010.png")
        self.assertEqual(counts, (3, 30, 5))
        self.assertEqual(mode, "CoolObject")

    def test_final_fridge_close_does_not_use_cooling_counts(self):
        row, trajectory = fixture("CloseObject", "CoolObject", 4)
        self.assertEqual(mapper.select_preaction(row, trajectory)[1], (2, 0, 2))

    def test_all_supported_normal_counts(self):
        for action, counts in mapper.FRAME_COUNTS.items():
            if action != "CoolObject":
                with self.subTest(action=action):
                    row, trajectory = fixture(action, count=sum(counts))
                    self.assertEqual(mapper.select_preaction(row, trajectory)[1], counts)

    def test_rejects_removed_before_frames(self):
        row, trajectory = fixture()
        row["frames"] = row["frames"][5:]
        with self.assertRaisesRegex(ValueError, "complete source group"):
            mapper.select_preaction(row, trajectory)

    def test_rejects_reordered_materialized_frames(self):
        row, trajectory = fixture()
        row["frames"].reverse()
        with self.assertRaisesRegex(ValueError, "complete source group"):
            mapper.select_preaction(row, trajectory)

    def test_rejects_missing_frame_in_both_indices(self):
        row, trajectory = fixture()
        del trajectory["images"][3]
        del row["frames"][3]
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            mapper.select_preaction(row, trajectory)

    def test_rejects_wrong_count_even_if_both_indices_agree(self):
        row, trajectory = fixture(count=10)
        with self.assertRaisesRegex(ValueError, "frame count"):
            mapper.select_preaction(row, trajectory)

    def test_rejects_changed_action(self):
        row, trajectory = fixture()
        row["source_action"]["api_action"]["objectId"] = "Different"
        with self.assertRaisesRegex(ValueError, "original action"):
            mapper.select_preaction(row, trajectory)

    def test_rejects_different_high_step(self):
        row, trajectory = fixture()
        trajectory["images"][0]["high_idx"] = 1
        with self.assertRaisesRegex(ValueError, "image high_idx"):
            mapper.select_preaction(row, trajectory)

    def test_training_split_rejected_before_file_access(self):
        with self.assertRaisesRegex(ValueError, "held-out validation"):
            mapper.convert_record({"split": "train"}, {}, {}, Path("."))


class MaskTests(unittest.TestCase):
    def test_row_major_area(self):
        self.assertEqual(mapper.mask_area([[0, 3], [6, 2]], 3, 3), 5)

    def test_rejects_bad_masks(self):
        for runs in ([], [[0, 10]], [[0, 3], [2, 1]], [[-1, 2]], [[0, 0]], [[0.5, 1]]):
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                mapper.mask_area(runs, 3, 3)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.row, self.trajectory = fixture()
        self.root = Path("/anvil-test-fixture")
        self.row.update({"split": "valid_seen", "task_id": "trial_example", "scene_id": "FloorPlan1",
                         "trajectory": {"path": "/source/traj_data.json", "sha256": "f" * 64}})
        self.trajectory.update({"task_id": "trial_example", "task_type": "pick_and_place_simple",
                                "scene": {"scene_num": 1}, "turk_annotations": {"anns": [
                                    {"task_desc": "Move the mug to the table.",
                                     "high_descs": ["Pick up the mug."],
                                     "assignment_id": "SHOULD_NOT_BE_EXPORTED"}]}})
        self.row["source_action"]["discrete_action"]["args"] = {
            "bbox": [0, 0, 2, 1], "mask": [[0, 2]], "point": [1, 0]}
        self.trajectory["plan"]["low_actions"][0] = copy.deepcopy(self.row["source_action"])
        self.trajectory["plan"]["high_pddl"][0]["discrete_action"] = {
            "action": "PickupObject", "args": ["mug"]}
        for frame in self.row["frames"]:
            frame["member"] = ("full_2.1.0/valid_seen/example/trial_example/raw_images/"
                               + Path(frame["metadata"]["image_name"]).with_suffix(".jpg").name)
        member = self.row["frames"][0]["member"]
        self.images = {member: {"path": str(self.root / "frames" / member),
                                "sha256": "e" * 64, "size": [300, 300]}}

    def convert(self, image_hash="e" * 64):
        image = MagicMock()
        image.__enter__.return_value.size = (300, 300)
        with patch.object(mapper, "sha256", return_value=image_hash), patch.object(mapper.Image, "open", return_value=image):
            return mapper.convert_record(self.row, self.trajectory, self.images, self.root)

    def test_record_separates_answers_and_omits_contact_and_worker_id(self):
        record = self.convert()
        self.assertFalse(record["training_approved"])
        self.assertFalse(record["native_schema_complete"])
        self.assertIsNone(record["reference"]["preconditions"])
        self.assertIsNone(record["reference"]["failure_modes"])
        self.assertIsNone(record["reference"]["target_contact_point_2d"])
        self.assertNotIn("next_expert_api_action", record["input"])
        self.assertNotIn("assignment_id", json.dumps(record))
        self.assertNotIn('"point"', json.dumps(record))

    def test_put_mask_describes_receptacle_not_held_object(self):
        action = self.row["source_action"]
        action["api_action"].update({"action": "PutObject", "receptacleObjectId": "Table|1|2|3"})
        action["discrete_action"]["action"] = "PutObject"
        self.trajectory["plan"]["low_actions"][0] = copy.deepcopy(action)
        record = self.convert()
        self.assertEqual(record["reference"]["target_object_type"], "Table")
        self.assertEqual(record["reference"]["target_role"], "destination_receptacle")

    def test_image_hash_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "image hash mismatch"):
            self.convert("0" * 64)

    def test_source_member_traversal_fails_closed(self):
        self.row["frames"][0]["member"] = "../../outside.jpg"
        with self.assertRaisesRegex(ValueError, "source image member"):
            self.convert()

    def test_train_scene_requires_explicit_exclusion_result(self):
        self.row["split"] = "train"
        with self.assertRaisesRegex(ValueError, "training scene not explicitly authorized"):
            mapper.convert_record(self.row, self.trajectory, {}, self.root, training_scenes={"FloorPlan2"})

    def test_exact_training_member_rejects_duplicate_feature_trees(self):
        self.assertTrue(mapper.exact_train_trajectory("full_2.1.0/train/task-1/trial/traj_data.json"))
        for member in ("full_2.1.0/json_feat_2.1.0/train/task-1/trial/traj_data.json",
                       "full_2.1.0/json_lang_2.1.0/train/task-1/trial/traj_data.json",
                       "full_2.1.0/valid_seen/task-1/trial/traj_data.json",
                       "full_2.1.0/train/../trial/traj_data.json"):
            self.assertFalse(mapper.exact_train_trajectory(member))


class SFTExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "sft"
        self.train_path, self.eval_path = self.root / "train_records.jsonl", self.root / "records.jsonl"
        self.train_manifest, self.eval_manifest = self.root / "train_manifest.json", self.root / "eval_manifest.json"
        self.train = self.make_record("train", 1)
        self.evaluation = self.make_record("valid_seen", 2)
        self.write_sources()

    def make_record(self, split, scene):
        case = RecordTests()
        case.setUp()
        row, trajectory = copy.deepcopy(case.row), copy.deepcopy(case.trajectory)
        task = "trial_" + split
        row.update(split=split, task_id=task, scene_id=f"FloorPlan{scene}")
        trajectory.update(task_id=task, scene={"scene_num": scene})
        trajectory["turk_annotations"]["anns"][0]["high_descs"] = ["FUTURE_STEP_SECRET"]
        member_root = f"full_2.1.0/{split}/example-{scene}/{task}"
        trajectory_path = self.root / "source" / member_root / "traj_data.json"
        trajectory_path.parent.mkdir(parents=True)
        trajectory_path.write_text(json.dumps(trajectory))
        row["trajectory"] = {"path": str(trajectory_path), "sha256": mapper.sha256(trajectory_path)}
        for frame in row["frames"]:
            frame["member"] = member_root + "/raw_images/" + Path(frame["metadata"]["image_name"]).with_suffix(".jpg").name
        member = row["frames"][0]["member"]
        image_path = self.root / "frames" / member
        image_path.parent.mkdir(parents=True)
        mapper.Image.new("RGB", (300, 300), color=(scene * 60, 0, 0)).save(image_path)
        images = {member: {"path": str(image_path), "sha256": mapper.sha256(image_path), "size": [300, 300]}}
        return mapper.convert_record(row, trajectory, images, self.root,
                                     training_scenes={f"FloorPlan{scene}"} if split == "train" else None)

    def write_sources(self):
        self.train_path.write_text(json.dumps(self.train) + "\n")
        self.eval_path.write_text(json.dumps(self.evaluation) + "\n")
        self.eval_manifest.write_text(json.dumps({"state": "source_native_validation_records_converted",
            "records": 1, "records_sha256": mapper.sha256(self.eval_path),
            "source_splits": {self.evaluation["source_split"]: 1}}))
        self.train_manifest.write_text(json.dumps({"state": "scene_disjoint_source_native_training_records_converted",
            "records": 1, "records_path": str(self.train_path), "records_sha256": mapper.sha256(self.train_path),
            "heldout_records": str(self.eval_path), "heldout_records_sha256": mapper.sha256(self.eval_path),
            "training_scenes": [self.train["scene_group"]], "heldout_scenes": [self.evaluation["scene_group"]],
            "physical_scene_overlap": []}))

    def export(self, contract=mapper.SFT_CONTRACT):
        return mapper.export_sft(self.train_path, self.train_manifest, self.eval_path,
                                 self.eval_manifest, self.output, contract)

    def test_faithful_targets_and_future_labels_never_enter_prompt(self):
        receipt = self.export()
        row = next(mapper.read_jsonl(self.output / "train.jsonl"))
        target = json.loads(row["messages"][1]["content"])
        self.assertEqual(target, {"target_object_type": "Mug", "target_role": "interaction_object",
            "next_expert_action": "PickupObject", "remaining_high_level_actions": [{"action": "PickupObject", "args": ["mug"]}]})
        self.assertEqual(row["messages"][0]["content"], "<image>\n" + mapper.SFT_INSTRUCTION + self.train["input"]["goal"])
        for secret in ("FUTURE_STEP_SECRET", "Mug|0|1|2", '"bbox"', '"remaining_high_level_steps"'):
            self.assertNotIn(secret, json.dumps(row["messages"]))
        self.assertEqual(row["anvil_meta"]["source_target_object_id"], "Mug|0|1|2")
        self.assertEqual(row["anvil_meta"]["rights"], "local_only")
        self.assertEqual(row["anvil_meta"]["native_contract"], mapper.SFT_CONTRACT)
        self.assertEqual(row["anvil_meta"]["group_id"], "ALFRED:FloorPlan1")
        self.assertEqual(row["anvil_meta"]["native_form"], "action_cot")
        self.assertTrue(row["anvil_meta"]["verification"])
        self.assertFalse(row["anvil_meta"]["native_schema_complete"])
        heldout = next(mapper.read_jsonl(self.output / "evaluation.jsonl"))
        self.assertFalse(heldout["anvil_meta"]["training_approved"])
        self.assertEqual(heldout["anvil_meta"]["source_split"], "valid_seen")
        self.assertEqual(receipt["counts"]["train"]["records"], 1)
        self.assertEqual(receipt["files"]["train.jsonl"], mapper.sha256(self.output / "train.jsonl"))

    def test_heldout_record_cannot_be_relabelled_training(self):
        self.train["source_split"] = "valid_seen"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "partition mismatch"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_physical_scene_overlap_rejected(self):
        self.train["scene_group"] = self.evaluation["scene_group"]
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "heldout physical scene"):
            self.export()

    def test_cross_split_image_hash_overlap_rejected(self):
        self.train["input"]["image_sha256"] = self.evaluation["input"]["image_sha256"]
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "heldout image"):
            self.export()

    def test_source_record_drift_rejected(self):
        self.train_path.write_text(self.train_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source records hash mismatch"):
            self.export()

    def test_image_byte_drift_rejected(self):
        Path(self.train["input"]["image"]).write_bytes(b"changed image")
        with self.assertRaisesRegex(ValueError, "source image hash mismatch"):
            self.export()

    def test_source_trajectory_drift_rejected(self):
        path = Path(self.train["provenance"]["trajectory"]["path"])
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "trajectory hash mismatch"):
            self.export()

    def test_missing_steps_rejected_without_output(self):
        self.train["reference"]["remaining_high_level_steps"] = []
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "remaining source steps"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_target_type_cannot_be_invented(self):
        self.train["reference"]["target_object_type"] = "Chair"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "target identity/type"):
            self.export()

    def test_future_goal_injection_rejected_against_source(self):
        self.train["input"]["goal"] += " Then do FUTURE_STEP_SECRET"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "source goal mismatch"):
            self.export()

    def test_manifest_cannot_point_to_other_heldout(self):
        manifest = json.loads(self.train_manifest.read_text())
        manifest["heldout_records_sha256"] = "f" * 64
        self.train_manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "different heldout source"):
            self.export()

    def test_explicit_contract_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "explicit supported"):
            self.export("action_cot")
        self.export()
        digest = mapper.sha256(self.output / "train.jsonl")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.export()
        self.assertEqual(mapper.sha256(self.output / "train.jsonl"), digest)

    def test_source_empty_argument_is_preserved_not_invented(self):
        trajectory_path = Path(self.train["provenance"]["trajectory"]["path"])
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["plan"]["high_pddl"][0]["discrete_action"]["args"] = [""]
        trajectory_path.write_text(json.dumps(trajectory))
        self.train["provenance"]["trajectory"]["sha256"] = mapper.sha256(trajectory_path)
        self.train["reference"]["remaining_high_level_steps"][0]["discrete_action"]["args"] = [""]
        self.write_sources()
        receipt = self.export()
        row = next(mapper.read_jsonl(self.output / "train.jsonl"))
        target = json.loads(row["messages"][1]["content"])
        self.assertEqual(target["remaining_high_level_actions"][0]["args"], [""])
        self.assertEqual(row["anvil_meta"]["source_empty_argument_count"], 1)
        self.assertEqual(receipt["counts"]["train"]["rows_with_source_empty_arguments"], 1)


class ExtractionEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "fixture.7z"
        self.archive.write_bytes(b"fixture archive binding")
        self.destination = self.root / "source"
        self.journal = self.root / "journal.jsonl"
        self.member = "full_2.1.0/train/task-1/trial/traj_data.json"
        self.data = b'{"task_id":"trial"}'
        self.headers = [(self.member, len(self.data), 0o100000)]
        self.opened, self.copied, self.progress = [], [], []

    def add_verified_journal(self):
        path = self.destination / self.member
        path.parent.mkdir(parents=True)
        path.write_bytes(self.data)
        item = {"member": self.member, "path": str(path.resolve()), "bytes": len(self.data),
                "sha256": mapper.sha256(path)}
        self.journal.write_text(json.dumps(item) + "\n")
        return path, item

    def fake_reader(self, tail_guard=False, data_headers=None):
        owner = self
        class Reader:
            def __init__(self, archive):
                self.index = len(owner.opened)
                owner.opened.append(self)
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                pass
            def __iter__(self):
                headers = owner.headers if self.index == 0 or data_headers is None else data_headers
                for entry in headers:
                    self.member = entry[0]
                    yield entry
                if self.index > 0 and tail_guard:
                    raise AssertionError("unnecessary archive tail scan")
            def copy_current(self, path, expected_size):
                if self.index == 0:
                    raise AssertionError("header inventory must never read member payloads")
                owner.copied.append(self.member)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(owner.data)
                owner.assertEqual(len(owner.data), expected_size)
                return mapper.sha256(path)
        return Reader

    def run_extract(self, **kwargs):
        return mapper.extract_selected(self.archive, self.destination, self.journal,
            mapper.exact_train_trajectory, lambda **values: self.progress.append(values), **kwargs)

    def test_all_verified_members_skip_data_reader(self):
        self.add_verified_journal()
        with patch.object(mapper, "SevenZipReader", self.fake_reader()):
            rows = self.run_extract()
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(self.opened), 1)
        self.assertFalse(self.copied)
        self.assertTrue(self.progress[-1]["payload_scan_skipped"])

    def test_exact_targets_break_without_decoding_archive_tail(self):
        with patch.object(mapper, "SevenZipReader", self.fake_reader(tail_guard=True)):
            rows = self.run_extract(expected_members={self.member})
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.copied, [self.member])
        self.assertEqual(len(self.opened), 2)

    def test_duplicate_header_rejected_even_if_journal_is_complete(self):
        self.add_verified_journal()
        self.headers *= 2
        with patch.object(mapper, "SevenZipReader", self.fake_reader()):
            with self.assertRaisesRegex(ValueError, "duplicate selected archive member"):
                self.run_extract()
        self.assertFalse(self.copied)

    def test_missing_expected_header_fails_before_extracting_anything(self):
        with patch.object(mapper, "SevenZipReader", self.fake_reader()):
            with self.assertRaisesRegex(ValueError, "expected selected members missing"):
                self.run_extract(expected_members={self.member, "absent.jpg"})
        self.assertFalse(self.copied)

    def test_same_size_hash_drift_is_really_reextracted(self):
        path, item = self.add_verified_journal()
        path.write_bytes(b"x" * len(self.data))
        with patch.object(mapper, "SevenZipReader", self.fake_reader(tail_guard=True)):
            rows = self.run_extract()
        self.assertEqual(self.copied, [self.member])
        self.assertEqual(path.read_bytes(), self.data)
        self.assertEqual(rows[0]["sha256"], item["sha256"])
        self.assertEqual(len(self.journal.read_text().splitlines()), 1)

    def test_missing_actual_file_is_really_reextracted(self):
        path, _ = self.add_verified_journal()
        path.unlink()
        with patch.object(mapper, "SevenZipReader", self.fake_reader(tail_guard=True)):
            self.run_extract()
        self.assertEqual(self.copied, [self.member])
        self.assertEqual(path.read_bytes(), self.data)

    def test_selected_type_is_checked_from_full_header_index(self):
        self.add_verified_journal()
        self.headers = [(self.member, len(self.data), 0o120000)]
        with patch.object(mapper, "SevenZipReader", self.fake_reader()):
            with self.assertRaisesRegex(ValueError, "regular nonempty"):
                self.run_extract()

    def test_duplicate_journal_is_not_silently_collapsed(self):
        _, item = self.add_verified_journal()
        self.journal.write_text((json.dumps(item) + "\n") * 2)
        with patch.object(mapper, "SevenZipReader", self.fake_reader()):
            with self.assertRaisesRegex(ValueError, "duplicate extraction journal"):
                self.run_extract()

    def test_missing_payload_cannot_be_hidden_by_header_inventory(self):
        with patch.object(mapper, "SevenZipReader", self.fake_reader(data_headers=[])):
            with self.assertRaisesRegex(ValueError, "selected members missing"):
                self.run_extract()

    def test_header_size_drift_between_passes_fails_closed(self):
        changed = [(self.member, len(self.data) + 1, 0o100000)]
        with patch.object(mapper, "SevenZipReader", self.fake_reader(data_headers=changed)):
            with self.assertRaisesRegex(ValueError, "changed after header inventory"):
                self.run_extract()


if __name__ == "__main__":
    unittest.main()
