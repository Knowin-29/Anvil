from __future__ import annotations

import unittest

from apps.data_flywheel.workers.vlm_client import VLMClient


class VLMClientJsonParseTest(unittest.TestCase):
    def test_parses_unclosed_markdown_fenced_array(self):
        text = "```json\n[{\"object\": \"knob\", \"action\": \"rotate\"}]"

        self.assertEqual(
            VLMClient._parse_json(text),
            [{"object": "knob", "action": "rotate"}],
        )

    def test_truncated_array_keeps_complete_objects(self):
        text = (
            "```json\n"
            "[{\"object\": \"knob\", \"action\": \"rotate\"},"
            "{\"object\": \"button\", \"action\": \"press\"},"
            "{\"object\": \"door\", \"reasoning\": \"truncated"
        )

        self.assertEqual(
            VLMClient._parse_json(text),
            [
                {"object": "knob", "action": "rotate"},
                {"object": "button", "action": "press"},
            ],
        )

    def test_first_object_truncated_returns_empty_dict_and_labeled_warning(self):
        text = "```json\n[{\"object\": \"rotary dial\", \"reasoning\": \"The rotary di"

        with self.assertLogs("flywheel.vlm_client", level="WARNING") as logs:
            self.assertEqual(VLMClient._parse_json(text), {})

        self.assertTrue(
            any("json_parse_failed_or_truncated" in line for line in logs.output),
            logs.output,
        )


if __name__ == "__main__":
    unittest.main()
