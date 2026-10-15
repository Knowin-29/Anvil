from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from trains.flywheel.contracts import (CONTRACT_VERSION, context_for,
                                      training_instruction, validate_task_columns)
from trains.flywheel.validators import validate
from trains.build_gsa_bench import normalize_frozen_test_record, stable_image_id

ROOT = Path(__file__).resolve().parents[1]


def load_reward_module():
    rewards = types.ModuleType("swift.rewards")
    rewards.ORM = type("ORM", (), {})
    rewards.orms = {}
    swift = types.ModuleType("swift")
    swift.rewards = rewards
    sys.modules["swift"] = swift
    sys.modules["swift.rewards"] = rewards

    spec = importlib.util.spec_from_file_location(
        "gsa_reward_plugin", ROOT / "scripts" / "gsa_reward_plugin.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.groundings = [
            {"object_class": "mug", "bbox": [100, 100, 400, 500]},
            {"object_class": "plate", "bbox": [500, 250, 850, 700]},
        ]

    def test_action_requires_spatial_context(self):
        context = context_for("action_cot", self.groundings)
        record = {
            "gsa_form": "action_cot",
            "gsa_contract_version": CONTRACT_VERSION,
            "gsa_context": context,
        }
        self.assertEqual(
            validate_task_columns(record), (False, "missing_spatial_context"))

    def test_training_grounding_names_referent_without_giving_reference_geometry(self):
        context = context_for("point", self.groundings)
        for form in ("bbox", "point", "mask"):
            question = training_instruction(form, context)
            self.assertIn('"mug"', question)
            self.assertNotIn("most prominent", question)
            self.assertNotIn(str(self.groundings[0]["bbox"]), question)

    def test_training_depth_excludes_its_answer_and_private_context(self):
        context = context_for("depth", self.groundings, {
            "depth_order": [{"label": "ANSWER_SENTINEL", "rank": 1}],
            "private_sensor_depth": "PRIVATE_SENTINEL",
        })
        question = training_instruction("depth", context)
        self.assertIn('"mug"', question)
        self.assertIn('"plate"', question)
        self.assertNotIn("ANSWER_SENTINEL", question)
        self.assertNotIn("PRIVATE_SENTINEL", question)

    def test_training_action_conditions_only_on_preceding_stages(self):
        context = context_for("action_cot", self.groundings, {
            "depth_order": [{"label": "DEPTH_SENTINEL", "rank": 1}],
            "affordance": [{"label": "CONTACT_SENTINEL"}],
            "dof": [{"label": "GRASP_SENTINEL"}],
            "action_cot": [{"goal": "ANSWER_SENTINEL"}],
            "private_sensor_depth": "PRIVATE_SENTINEL",
        })
        affordance = training_instruction("affordance", context)
        dof = training_instruction("dof", context)
        plan = training_instruction("action_cot", context)
        self.assertIn("DEPTH_SENTINEL", affordance)
        self.assertNotIn("CONTACT_SENTINEL", affordance)
        self.assertIn("CONTACT_SENTINEL", dof)
        self.assertNotIn("GRASP_SENTINEL", dof)
        self.assertIn("GRASP_SENTINEL", plan)
        self.assertNotIn("ANSWER_SENTINEL", plan)
        self.assertNotIn("PRIVATE_SENTINEL", plan)

    def test_split_identity_ignores_mount_prefix(self):
        a = "/old/node/data/bridge_data_v2/scene/frame.jpg"
        b = "/new/node/cache/bridge_data_v2/scene/frame.jpg"
        self.assertEqual(stable_image_id(a, "bridge_data_v2"),
                         stable_image_id(b, "bridge_data_v2"))

    def test_frozen_human_gt_target_is_not_rewritten(self):
        messages = [
            {"role": "user", "content": "<image> Point to the mug."},
            {"role": "assistant", "content": '[{"point_2d": [200, 300]}]'},
        ]
        record = {
            "messages": messages,
            "images": ["/data/robopoint_ref/example.jpg"],
            "gsa_meta": {
                "form": "point",
                "source": "robopoint_ref",
                "label_provenance": "human_gt",
            },
        }
        normalized = normalize_frozen_test_record(record)
        self.assertEqual(normalized["messages"], messages)
        self.assertEqual(normalized["gsa_form"], "point")
        self.assertEqual(normalized["gsa_family"], "G")
        self.assertEqual(normalized["gsa_meta"]["split"], "test")

    def test_frozen_model_label_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_frozen_test_record({
                "messages": [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": "a"}],
                "images": ["/data/image.jpg"],
                "gsa_meta": {"form": "bbox", "label_provenance": "model"},
            })

    def test_depth_is_multi_object_contiguous_and_monotonic(self):
        context = context_for("depth", self.groundings)
        good = [
            {"label": "mug", "rank": 1, "relative_depth": 0.2},
            {"label": "plate", "rank": 2, "relative_depth": 0.7},
        ]
        self.assertEqual(validate("depth", good, context), (True, ""))
        bad = [
            {"label": "mug", "rank": 1, "relative_depth": 0.8},
            {"label": "plate", "rank": 3, "relative_depth": 0.2},
        ]
        self.assertEqual(validate("depth", bad, context)[0], False)

    def test_dof_axis_and_wrist_must_agree(self):
        context = context_for(
            "dof", self.groundings,
            {"depth_order": [{"label": "mug", "rank": 1, "relative_depth": 0.2}]})
        good = [{
            "label": "mug",
            "grasp_point_2d": [250, 300],
            "jaw_axis_2d": [0.866, 0.5],
            "approach": "top",
            "wrist_deg": 30,
        }]
        self.assertEqual(validate("dof", good, context), (True, ""))
        template = [dict(good[0], jaw_axis_2d=[0.707, 0.707], wrist_deg=45)]
        self.assertEqual(validate("dof", template, context), (True, ""))
        for axis, wrist in (([-1, 0], 0), ([0, 1], -90), ([0, -1], 90)):
            self.assertEqual(validate("dof", [dict(good[0], jaw_axis_2d=axis, wrist_deg=wrist)], context), (True, ""))
        bad = [dict(good[0], jaw_axis_2d=[1, 0], wrist_deg=60)]
        self.assertEqual(validate("dof", bad, context)[0], False)
        for axis in ([0, 0], [5, 0], [float("nan"), 1]):
            self.assertFalse(validate("dof", [dict(good[0], jaw_axis_2d=axis)], context)[0])

    def test_affordance_rejects_background_part_label(self):
        context = context_for(
            "affordance", self.groundings,
            {"depth_order": [
                {"label": "mug", "rank": 1, "relative_depth": 0.2},
                {"label": "plate", "rank": 2, "relative_depth": 0.7},
            ]})
        payload = [{
            "label": "surface",
            "affordance": "place on",
            "bbox_2d": [150, 150, 300, 300],
            "contact_point_2d": [225, 225],
        }]
        self.assertEqual(
            validate("affordance", payload, context),
            (False, "background_label"))


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reward_module = load_reward_module()

    def test_declared_form_cannot_be_changed_by_completion(self):
        reward = self.reward_module.GSAReward()
        bbox_answer = json.dumps([{"label": "mug", "bbox_2d": [100, 100, 400, 500]}])
        scores = reward(
            [bbox_answer],
            gsa_form=["detection"],
            gsa_context=[context_for("detection", [])],
            gsa_contract_version=[CONTRACT_VERSION],
        )
        self.assertEqual(scores, [0.0])

    def test_missing_task_metadata_is_fatal(self):
        reward = self.reward_module.GSAReward()
        with self.assertRaises(ValueError):
            reward(["[]"])


if __name__ == "__main__":
    unittest.main()
