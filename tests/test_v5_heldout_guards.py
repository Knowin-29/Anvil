from __future__ import annotations

import unittest

from trains.flywheel.to_viewer_format import adapt


class V5HeldOutGuardTests(unittest.TestCase):
    def test_generic_viewer_rejects_v5_protocol(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema-5 audit builder"):
            adapt(
                {
                    "protocol_version": "gsa-depth-v5.0-scannet-sensor-pairwise",
                    "task_form": "depth",
                    "payload": [],
                },
                "depth",
                1,
            )

    def test_generic_viewer_rejects_v5_source_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "generic viewer conversion is forbidden"):
            adapt(
                {
                    "source_artifact_schema": "gsa-scannet-sensor-depth-source-v1",
                    "task_form": "depth",
                    "payload": [],
                },
                "depth",
                1,
            )


if __name__ == "__main__":
    unittest.main()
