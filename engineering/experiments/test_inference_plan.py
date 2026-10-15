import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference_plan import (
    InferencePlanError,
    create_crossed_plan,
    exact_one_sided_sign_flip_pvalue,
    holm_adjust,
    seed_t_interval_95,
    validate_saved_plan,
    write_plan,
)


class InferencePlanTest(unittest.TestCase):
    def test_plan_is_deterministic_and_in_range(self):
        first = create_crossed_plan([17, 29, 43, 71], 5, 20, 99)
        second = create_crossed_plan([17, 29, 43, 71], 5, 20, 99)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[0].shape, (20, 4))
        self.assertEqual(first[1].shape, (20, 5))
        self.assertLess(int(first[0].max()), 4)
        self.assertLess(int(first[1].max()), 5)

    def test_seed_and_scene_draws_are_independent_stream_segments(self):
        seed_draws, scene_draws = create_crossed_plan([17, 29], 2, 50, 7)
        self.assertFalse(np.array_equal(seed_draws, scene_draws))

    def test_duplicate_seeds_are_rejected(self):
        with self.assertRaisesRegex(InferencePlanError, "unique"):
            create_crossed_plan([17, 17], 3, 10, 1)

    def test_too_few_scenes_are_rejected(self):
        with self.assertRaisesRegex(InferencePlanError, "scene clusters"):
            create_crossed_plan([17, 29], 1, 10, 1)

    def test_exact_sign_flip_for_eight_positive_equal_differences(self):
        self.assertEqual(exact_one_sided_sign_flip_pvalue([1.0] * 8), 1 / 256)

    def test_holm_adjustment_preserves_ordered_monotonicity(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_seed_t_interval_collapses_for_constant_differences(self):
        lower, upper = seed_t_interval_95([1.5] * 8)
        self.assertAlmostEqual(lower, 1.5)
        self.assertAlmostEqual(upper, 1.5)

    def test_saved_plan_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes.jsonl"
            scenes.write_text(
                "".join(json.dumps({"scene_id": f"s{i}"}) + "\n" for i in range(4)),
                encoding="utf-8",
            )
            output = root / "draws.npz"
            metadata = root / "draws.json"
            write_plan(output, metadata, scenes, [17, 29, 43], 25, 123)
            checked = validate_saved_plan(output, metadata)
            self.assertEqual(checked["scene_count"], 4)
            self.assertEqual(checked["replicates"], 25)

    def test_tampered_plan_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes.jsonl"
            scenes.write_text('{"scene_id":"s1"}\n{"scene_id":"s2"}\n', encoding="utf-8")
            output = root / "draws.npz"
            metadata = root / "draws.json"
            write_plan(output, metadata, scenes, [17, 29], 10, 3)
            output.write_bytes(output.read_bytes() + b"tamper")
            with self.assertRaisesRegex(InferencePlanError, "hash mismatch"):
                validate_saved_plan(output, metadata)

    def test_tampered_scene_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes.jsonl"
            scenes.write_text('{"scene_id":"s1"}\n{"scene_id":"s2"}\n', encoding="utf-8")
            output = root / "draws.npz"
            metadata = root / "draws.json"
            write_plan(output, metadata, scenes, [17, 29], 10, 3)
            scenes.write_text('{"scene_id":"s1"}\n{"scene_id":"s3"}\n', encoding="utf-8")
            with self.assertRaisesRegex(InferencePlanError, "scene-manifest hash mismatch"):
                validate_saved_plan(output, metadata, scenes)

    def test_output_suffix_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes.jsonl"
            scenes.write_text('{"scene_id":"s1"}\n{"scene_id":"s2"}\n', encoding="utf-8")
            with self.assertRaisesRegex(InferencePlanError, "must use .npz"):
                write_plan(root / "draws.bin", root / "draws.json", scenes, [17, 29], 10, 3)


if __name__ == "__main__":
    unittest.main()
