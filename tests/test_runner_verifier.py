from __future__ import annotations

import unittest
from unittest import mock

from apps.data_flywheel.orchestrator import runner


class RunnerVerifierTest(unittest.TestCase):
    def test_inline_verifier_is_skipped_when_transformers_missing(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertIsNone(runner._create_inline_verifier())


if __name__ == "__main__":
    unittest.main()
