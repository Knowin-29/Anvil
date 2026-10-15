from __future__ import annotations

import unittest

from PIL import Image

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.instruction_worker import InstructionWorker
from apps.data_flywheel.workers.planning_worker import PlanningWorker


class RecordingVLM:
    def __init__(self):
        self.calls = []

    def chat_json(self, prompt, image_path=None, max_tokens=1024, temperature=0.4):
        self.calls.append({"max_tokens": max_tokens, "prompt": prompt, "image_path": image_path})
        return []


def _sample():
    return Sample(
        sample_id="sample_1",
        source_type="image_dir",
        frames=[Image.new("RGB", (10, 10))],
        metadata={"image_path": "/tmp/fake.jpg"},
    )


def _grounding_data():
    return [{"name": "knob", "bbox": [1, 2, 3, 4]}]


class VLMWorkerTokenBudgetTest(unittest.TestCase):
    def test_planning_uses_raised_token_budget(self):
        vlm = RecordingVLM()

        PlanningWorker(vlm_client=vlm).process(_sample(), grounding_data=_grounding_data())

        self.assertEqual(vlm.calls[0]["max_tokens"], 3072)

    def test_instruction_uses_raised_token_budget(self):
        vlm = RecordingVLM()

        InstructionWorker(vlm_client=vlm).process(_sample(), grounding_data=_grounding_data())

        self.assertEqual(vlm.calls[0]["max_tokens"], 3072)


if __name__ == "__main__":
    unittest.main()
