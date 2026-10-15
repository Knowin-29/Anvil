from __future__ import annotations

import unittest

from PIL import Image

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.robot_state.worker import RobotStateWorker


class FakeVLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_image_size(self, image_path):
        return (640, 480)

    def chat_json(self, prompt, image_path=None, max_tokens=1024, temperature=0.4):
        self.calls.append({"prompt": prompt, "image_path": image_path, "max_tokens": max_tokens})
        return self.result


def _sample():
    return Sample(
        sample_id="sample_1",
        source_type="image_dir",
        frames=[Image.new("RGB", (10, 10))],
        metadata={"image_path": "/tmp/robot.jpg", "filename": "robot.jpg"},
    )


class RobotStateWorkerTest(unittest.TestCase):
    def test_valid_robot_state_json_produces_annotation_schema(self):
        worker = RobotStateWorker(vlm_client=FakeVLM({
            "robot_part": "gripper",
            "tool_name": "parallel_jaw_gripper",
            "state_type": "holding_state",
            "state_value": "holding_object",
            "gripper_empty": False,
            "holding_object": "sponge",
            "contact_state": "in_contact",
            "object_between_fingers": True,
            "evidence_bbox": [120, 80, 300, 260],
            "confidence": 0.82,
            "confidence_reason": "Object is visibly centered between the two gripper fingers.",
        }))

        annotations = worker.process(_sample())

        self.assertEqual(len(annotations), 1)
        ann = annotations[0]
        self.assertEqual(ann["pipeline_type"], "A_robot_state")
        self.assertEqual(ann["sample_id"], "sample_1")
        self.assertEqual(ann["image_path"], "/tmp/robot.jpg")
        self.assertEqual(ann["robot_part"], "gripper")
        self.assertEqual(ann["state_value"], "holding_object")
        self.assertIs(ann["object_between_fingers"], True)

    def test_non_dict_or_truncated_result_returns_empty_annotations(self):
        self.assertEqual(RobotStateWorker(vlm_client=FakeVLM([])).process(_sample()), [])
        self.assertEqual(RobotStateWorker(vlm_client=FakeVLM("not-json")).process(_sample()), [])

    def test_prompt_is_json_only_and_uses_raised_token_budget(self):
        vlm = FakeVLM({"state_type": "gripper_state", "state_value": "empty"})
        RobotStateWorker(vlm_client=vlm).process(_sample())

        self.assertIn("JSON only", vlm.calls[0]["prompt"])
        self.assertIn("A_robot_state", vlm.calls[0]["prompt"])
        self.assertEqual(vlm.calls[0]["max_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
