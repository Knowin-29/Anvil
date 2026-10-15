from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from apps.data_flywheel.robot_state.pipeline import validate_robot_state_source_config
from apps.data_flywheel.sources.registry import get_source


def _write_sharerobot_fixture(root: Path) -> None:
    image_rel = "rtx_frames_success_0/demo_episode/frame_0.png"
    image_path = root / "trajectory" / "images" / image_rel
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 30), color=(255, 255, 255)).save(image_path)
    (root / "trajectory").mkdir(exist_ok=True)
    (root / "trajectory" / "trajectory.json").write_text(
        json.dumps([
            {
                "id": 7,
                "image_path": image_rel,
                "meta_data": {"original_dataset": "bridge", "original_width": 20, "original_height": 30},
                "instruction": "reach for the cloth",
                "trajectory": [[10, 11], [12, 13], [14, 15]],
            }
        ]),
        encoding="utf-8",
    )


class ShareRobotSourceTest(unittest.TestCase):
    def test_scans_trajectory_records_and_loads_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sharerobot_fixture(root)

            source = get_source("sharerobot", paths=[str(root)], subsets=["trajectory"], max_images=10)
            samples = list(source.iter_samples())

            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.source_type, "sharerobot")
            self.assertEqual(sample.metadata["dataset"], "sharerobot")
            self.assertEqual(sample.metadata["share_robot_subset"], "trajectory")
            self.assertEqual(sample.metadata["instruction"], "reach for the cloth")
            self.assertEqual(sample.metadata["annotation"]["trajectory"], [[10, 11], [12, 13], [14, 15]])
            self.assertTrue(sample.metadata["image_path"].endswith("frame_0.png"))

    def test_robot_state_config_validation_accepts_sharerobot_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sharerobot_fixture(root)
            config = root / "robot_state_sharerobot.yaml"
            config.write_text(
                "sources:\n"
                f"  - path: {root}\n"
                "    source_type: sharerobot\n"
                "    subsets: [trajectory]\n"
                "    max_images: 5\n"
                "    weight: 1.0\n",
                encoding="utf-8",
            )

            result = validate_robot_state_source_config(str(config))

            self.assertTrue(result["ok"])
            self.assertEqual(result["total_images"], 1)
            self.assertEqual(result["sources"][0]["source_type"], "sharerobot")


if __name__ == "__main__":
    unittest.main()
