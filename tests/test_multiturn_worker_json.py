from __future__ import annotations

import unittest
from unittest import mock

from PIL import Image

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.multiturn_worker import MULTITURN_PROMPT, MultiturnWorker


class FakeVLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def chat_json(self, prompt, image_path=None, max_tokens=1024, temperature=0.4):
        self.calls.append({"prompt": prompt, "image_path": image_path, "max_tokens": max_tokens})
        return self.result


def _sample():
    return Sample(
        sample_id="sample_1",
        source_type="image_dir",
        frames=[Image.new("RGB", (10, 10))],
        metadata={"image_path": "/tmp/fake.jpg"},
    )


def _grounding_data():
    return [{"name": "knob", "bbox": [1, 2, 3, 4]}]


class MultiturnWorkerJsonTest(unittest.TestCase):
    def test_prompt_demands_json_only_and_limited_dialogues(self):
        prompt = MULTITURN_PROMPT.format(objects_list="- knob")

        self.assertIn("JSON only", prompt)
        self.assertIn("no markdown", prompt.lower())
        self.assertIn("up to 2", prompt.lower())
        self.assertIn("2-4 turns", prompt)

    def test_valid_multiturn_json_produces_annotation_with_raised_token_budget(self):
        vlm = FakeVLM([
            {
                "turns": [
                    {"role": "user", "content": "What is the round object?"},
                    {"role": "assistant", "content": "It is a rotary knob."},
                    {"role": "user", "content": "What does it do?"},
                    {"role": "assistant", "content": "It selects settings by rotation."},
                ],
                "type": "followup",
                "key_object": "rotary knob",
            }
        ])
        worker = MultiturnWorker(vlm_client=vlm)

        with mock.patch("random.uniform", return_value=0.8), mock.patch("random.randint", return_value=0):
            annotations = worker.process(_sample(), grounding_data=_grounding_data())

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].content["type"], "followup")
        self.assertEqual(vlm.calls[0]["max_tokens"], 3072)

    def test_non_list_multiturn_result_returns_empty_annotations(self):
        worker = MultiturnWorker(vlm_client=FakeVLM({}))

        self.assertEqual(worker.process(_sample(), grounding_data=_grounding_data()), [])


if __name__ == "__main__":
    unittest.main()
