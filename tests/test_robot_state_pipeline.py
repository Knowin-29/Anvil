from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from apps.data_flywheel.robot_state.pipeline import run_robot_state_pipeline, validate_robot_state_source_config


class FakeVLM:
    def get_image_size(self, image_path):
        return (32, 32)

    def chat_json(self, prompt, image_path=None, max_tokens=1024, temperature=0.4):
        return {
            "robot_part": "gripper",
            "tool_name": "parallel_jaw_gripper",
            "state_type": "holding_state",
            "state_value": "empty",
            "gripper_empty": True,
            "holding_object": None,
            "contact_state": "no_contact",
            "object_between_fingers": False,
            "evidence_bbox": [1, 2, 20, 21],
            "confidence": 0.77,
            "confidence_reason": "The gripper fingers are open and no object is visible between them.",
        }


class RobotStatePipelineTest(unittest.TestCase):
    def test_pipeline_writes_independent_robot_state_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "images"
            img_dir.mkdir()
            Image.new("RGB", (32, 32), color=(255, 255, 255)).save(img_dir / "sample.jpg")
            config = root / "robot_state.yaml"
            config.write_text(
                "sources:\n"
                f"  - path: {img_dir}\n"
                "    source_type: image_dir\n"
                "    recursive: true\n"
                "    max_images: 5\n"
                "    weight: 1.0\n",
                encoding="utf-8",
            )

            result = run_robot_state_pipeline(
                source_config=str(config),
                output_root=str(root / "out"),
                run_id="robot_test",
                vlm_client=FakeVLM(),
            )

            run_dir = root / "out" / "robot_test"
            annotations_path = run_dir / "annotations" / "A_robot_state.jsonl"
            self.assertEqual(result["total_samples"], 1)
            self.assertEqual(result["total_annotations"], 1)
            self.assertTrue((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "robot_state_stats.json").exists())
            self.assertTrue(annotations_path.exists())

            record = json.loads(annotations_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["pipeline_type"], "A_robot_state")
            self.assertEqual(record["state_value"], "empty")
            self.assertFalse((root / "out" / "robot_test" / "annotations" / "A" / "A.jsonl").exists())

    def test_source_config_validation_counts_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "images"
            img_dir.mkdir()
            Image.new("RGB", (32, 32), color=(255, 255, 255)).save(img_dir / "sample.jpg")
            config = root / "robot_state.yaml"
            config.write_text(
                "sources:\n"
                f"  - path: {img_dir}\n"
                "    source_type: image_dir\n"
                "    recursive: true\n"
                "    max_images: 5\n"
                "    weight: 1.0\n",
                encoding="utf-8",
            )

            result = validate_robot_state_source_config(str(config))

            self.assertTrue(result["ok"])
            self.assertEqual(result["available_sources"], 1)
            self.assertEqual(result["total_images"], 1)
            self.assertEqual(result["sources"][0]["num_images"], 1)


if __name__ == "__main__":
    unittest.main()
