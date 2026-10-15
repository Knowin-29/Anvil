from __future__ import annotations

import unittest
from unittest import mock

from PIL import Image

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.affordance_worker import AFFORDANCE_PROMPT, AffordanceWorker


class FakeVLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_image_size(self, image_path):
        return (640, 480)

    def chat_json(self, prompt, image_path=None, max_tokens=1024, temperature=0.4):
        self.calls.append({
            "prompt": prompt,
            "image_path": image_path,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return self.result


def _sample():
    return Sample(
        sample_id="sample_1",
        source_type="image_dir",
        frames=[Image.new("RGB", (10, 10))],
        metadata={"image_path": "/tmp/fake.jpg"},
    )


class AffordanceWorkerJsonTest(unittest.TestCase):
    def test_prompt_demands_json_only_and_short_reasoning(self):
        prompt = AFFORDANCE_PROMPT.format(width=640, height=480)

        self.assertIn("JSON only", prompt)
        self.assertIn("no markdown", prompt.lower())
        self.assertIn("12-20 words", prompt)
        self.assertIn("up to 3", prompt.lower())

    def test_valid_affordance_json_produces_annotation_with_raised_token_budget(self):
        vlm = FakeVLM([
            {
                "object": "rotary knob",
                "bbox": [100, 80, 180, 160],
                "action": "rotate",
                "contact_point": [140, 120],
                "approach_direction": "front",
                "reasoning": "Circular knob affords rotation for selecting settings.",
            }
        ])
        worker = AffordanceWorker(vlm_client=vlm)

        with mock.patch("random.uniform", return_value=0.8), mock.patch("random.randint", return_value=0):
            annotations = worker.process(_sample())

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].content["object_name"], "rotary knob")
        self.assertEqual(annotations[0].content["action"], "rotate")
        self.assertEqual(vlm.calls[0]["max_tokens"], 3072)

    def test_non_list_or_truncated_affordance_result_returns_empty_annotations(self):
        worker = AffordanceWorker(vlm_client=FakeVLM({}))

        self.assertEqual(worker.process(_sample()), [])


if __name__ == "__main__":
    unittest.main()
