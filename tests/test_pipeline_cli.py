from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from apps.data_flywheel import pipeline


class PipelineCliTest(unittest.TestCase):
    def test_v3_cli_forwards_vlm_endpoint_model_and_run_id(self):
        captured = {}

        def fake_run_v3_pipeline(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                run_id="cli_run",
                total_samples=0,
                total_annotations=0,
                per_capability={},
                elapsed_seconds=0.0,
            )

        argv = [
            "pipeline.py",
            "--v3",
            "--source-config",
            "scripts/configs/flywheel_anvil_quick.yaml",
            "--vlm-url",
            "http://localhost:10301",
            "--vlm-model",
            "qwen2.5-vl-32b-instruct",
            "--run-id",
            "cli_run",
        ]

        with mock.patch.object(pipeline, "run_v3_pipeline", fake_run_v3_pipeline):
            with mock.patch.object(sys, "argv", argv):
                pipeline.main()

        self.assertEqual(captured["source_config"], "scripts/configs/flywheel_anvil_quick.yaml")
        self.assertEqual(captured["vlm_url"], "http://localhost:10301")
        self.assertEqual(captured["vlm_model"], "qwen2.5-vl-32b-instruct")
        self.assertEqual(captured["run_id"], "cli_run")


if __name__ == "__main__":
    unittest.main()
