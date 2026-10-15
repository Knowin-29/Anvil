import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from observation_package import (
    ObservationPackageError,
    PackageIdentity,
    validate_package,
)


IDENTITY = PackageIdentity(
    run_id="M-ANVIL-S17",
    canonical_method_id="semantic_first_fault_mtd",
    protocol_id="anvil-eval-v1",
    split_revision="anvil-bench-v1",
    seed=17,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def manifest_rows() -> list[dict]:
    return [
        {
            "uid": "u1",
            "scene_id": "s1",
            "source_image_sha256": "a" * 64,
            "split_revision": "anvil-bench-v1",
            "form": "bbox",
        },
        {
            "uid": "u2",
            "scene_id": "s2",
            "source_image_sha256": "b" * 64,
            "split_revision": "anvil-bench-v1",
            "form": "relative_depth",
        },
    ]


def prediction(uid: str, scene: str, form: str, **overrides) -> dict:
    row = {
        "uid": uid,
        "scene_id": scene,
        "source_image_sha256": ("a" if uid == "u1" else "b") * 64,
        "run_id": "M-ANVIL-S17",
        "canonical_method_id": "semantic_first_fault_mtd",
        "protocol_id": "anvil-eval-v1",
        "split_revision": "anvil-bench-v1",
        "form": form,
        "seed": 17,
        "raw_prediction": {},
        "parse_state": "valid",
        "primary_score": 1.0,
        "malformed": False,
        "truncated": False,
        "missing": False,
    }
    row.update(overrides)
    return row


class ObservationPackageTest(unittest.TestCase):
    def validate(self, manifests: list[dict], predictions: list[dict], identity=IDENTITY):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.jsonl"
            prediction_path = root / "predictions.jsonl"
            write_jsonl(manifest_path, manifests)
            write_jsonl(prediction_path, predictions)
            return validate_package(manifest_path, prediction_path, identity)

    def test_complete_package_passes(self):
        result = self.validate(
            manifest_rows(),
            [prediction("u1", "s1", "bbox"), prediction("u2", "s2", "relative_depth")],
        )
        self.assertEqual(result.manifest_rows, 2)
        self.assertEqual(result.prediction_rows, 2)

    def test_malformed_output_must_score_zero(self):
        rows = [
            prediction("u1", "s1", "bbox", malformed=True, primary_score=0.5),
            prediction("u2", "s2", "relative_depth"),
        ]
        with self.assertRaisesRegex(ObservationPackageError, "must receive zero"):
            self.validate(manifest_rows(), rows)

    def test_malformed_zero_scored_output_passes(self):
        rows = [
            prediction("u1", "s1", "bbox", malformed=True, primary_score=0.0),
            prediction("u2", "s2", "relative_depth"),
        ]
        result = self.validate(manifest_rows(), rows)
        self.assertEqual(result.malformed_rows, 1)

    def test_missing_uid_is_rejected(self):
        with self.assertRaisesRegex(ObservationPackageError, "UID coverage mismatch"):
            self.validate(manifest_rows(), [prediction("u1", "s1", "bbox")])

    def test_extra_uid_is_rejected(self):
        rows = [
            prediction("u1", "s1", "bbox"),
            prediction("u2", "s2", "relative_depth"),
            prediction("u3", "s3", "point"),
        ]
        with self.assertRaisesRegex(ObservationPackageError, "UID coverage mismatch"):
            self.validate(manifest_rows(), rows)

    def test_duplicate_uid_is_rejected(self):
        rows = [prediction("u1", "s1", "bbox"), prediction("u1", "s1", "bbox")]
        with self.assertRaisesRegex(ObservationPackageError, "duplicate uid"):
            self.validate(manifest_rows(), rows)

    def test_uid_type_coercion_cannot_establish_coverage(self):
        manifests = manifest_rows()
        manifests[0]["uid"] = 1
        rows = [prediction("u1", "s1", "bbox"), prediction("u2", "s2", "relative_depth")]
        rows[0]["uid"] = "1"
        with self.assertRaisesRegex(ObservationPackageError, "uid must be a non-empty string"):
            self.validate(manifests, rows)

    def test_identity_fields_require_nonblank_strings_on_both_sides(self):
        fields = ("uid", "scene_id", "source_image_sha256", "split_revision", "form")
        for side in ("manifest", "prediction"):
            for field in fields:
                for value in (True, 1, 1.0, [], {}, "", " \t\r\n"):
                    with self.subTest(side=side, field=field, value=value):
                        manifests = manifest_rows()
                        rows = [prediction("u1", "s1", "bbox"), prediction("u2", "s2", "relative_depth")]
                        changed_rows = manifests if side == "manifest" else rows
                        changed_rows[0][field] = value
                        with self.assertRaisesRegex(
                            ObservationPackageError, f"{field} must be a non-empty string"
                        ):
                            self.validate(manifests, rows)

    def test_matching_blank_or_cross_typed_scene_ids_are_rejected(self):
        for manifest_scene, prediction_scene in (("", ""), (" \t", " \t"), (True, 1)):
            with self.subTest(manifest_scene=manifest_scene, prediction_scene=prediction_scene):
                manifests = manifest_rows()
                manifests[0]["scene_id"] = manifest_scene
                rows = [prediction("u1", prediction_scene, "bbox"), prediction("u2", "s2", "relative_depth")]
                with self.assertRaisesRegex(ObservationPackageError, "scene_id must be a non-empty string"):
                    self.validate(manifests, rows)

    def test_run_method_and_protocol_identities_require_nonblank_strings(self):
        for field in ("run_id", "canonical_method_id", "protocol_id"):
            for value in (True, 1, [], {}, "", " \t\n"):
                with self.subTest(field=field, value=value):
                    rows = [prediction("u1", "s1", "bbox", **{field: value}), prediction("u2", "s2", "relative_depth")]
                    with self.assertRaisesRegex(
                        ObservationPackageError, f"{field} must be a non-empty string"
                    ):
                        self.validate(manifest_rows(), rows)

    def test_seed_must_be_integer_before_identity_comparison(self):
        for supplied, expected in ((True, 1), (False, 0), (17.0, 17), ("17", 17)):
            with self.subTest(supplied=supplied, expected=expected):
                rows = [
                    prediction("u1", "s1", "bbox", seed=supplied),
                    prediction("u2", "s2", "relative_depth", seed=expected),
                ]
                with self.assertRaisesRegex(ObservationPackageError, "seed must be an integer"):
                    self.validate(manifest_rows(), rows, replace(IDENTITY, seed=expected))

    def test_valid_integer_seed_and_exact_uid_strings_are_preserved(self):
        rows = [
            prediction("u1", "s1", "bbox", seed=0),
            prediction("u2", "s2", "relative_depth", seed=0),
        ]
        self.assertEqual(
            self.validate(manifest_rows(), rows, replace(IDENTITY, seed=0)).prediction_rows, 2
        )
        rows[0]["uid"] = "u1 "
        with self.assertRaisesRegex(ObservationPackageError, "UID coverage mismatch"):
            self.validate(manifest_rows(), rows, replace(IDENTITY, seed=0))

    def test_each_invalid_output_flag_requires_zero_without_dropping_rows(self):
        for flag in ("malformed", "truncated", "missing"):
            with self.subTest(flag=flag):
                rows = [
                    prediction("u1", "s1", "bbox", **{flag: True}),
                    prediction("u2", "s2", "relative_depth"),
                ]
                with self.assertRaisesRegex(ObservationPackageError, "must receive zero"):
                    self.validate(manifest_rows(), rows)
                rows[0]["primary_score"] = 0.0
                result = self.validate(manifest_rows(), rows)
                self.assertEqual(result.manifest_rows, 2)
                self.assertEqual(result.prediction_rows, 2)
                self.assertEqual(getattr(result, f"{flag}_rows"), 1)

    def test_wrong_method_is_rejected(self):
        rows = [
            prediction("u1", "s1", "bbox", canonical_method_id="declared_family_route"),
            prediction("u2", "s2", "relative_depth"),
        ]
        with self.assertRaisesRegex(ObservationPackageError, "canonical_method_id"):
            self.validate(manifest_rows(), rows)

    def test_wrong_scene_is_rejected(self):
        rows = [prediction("u1", "other", "bbox"), prediction("u2", "s2", "relative_depth")]
        with self.assertRaisesRegex(ObservationPackageError, "scene_id"):
            self.validate(manifest_rows(), rows)

    def test_wrong_image_hash_is_rejected(self):
        rows = [
            prediction("u1", "s1", "bbox", source_image_sha256="c" * 64),
            prediction("u2", "s2", "relative_depth"),
        ]
        with self.assertRaisesRegex(ObservationPackageError, "source_image_sha256"):
            self.validate(manifest_rows(), rows)

    def test_unknown_form_is_rejected(self):
        manifests = manifest_rows()
        manifests[0]["form"] = "generic_vqa"
        with self.assertRaisesRegex(ObservationPackageError, "unknown form"):
            self.validate(
                manifests,
                [prediction("u1", "s1", "generic_vqa"), prediction("u2", "s2", "relative_depth")],
            )

    def test_non_finite_or_out_of_range_score_is_rejected(self):
        for bad_score in (float("nan"), float("inf"), -0.1, 1.1):
            rows = [
                prediction("u1", "s1", "bbox", primary_score=bad_score),
                prediction("u2", "s2", "relative_depth"),
            ]
            with self.subTest(bad_score=bad_score):
                with self.assertRaisesRegex(ObservationPackageError, "within \\[0, 1\\]"):
                    self.validate(manifest_rows(), rows)

    def test_non_boolean_health_flag_is_rejected(self):
        rows = [
            prediction("u1", "s1", "bbox", truncated=1),
            prediction("u2", "s2", "relative_depth"),
        ]
        with self.assertRaisesRegex(ObservationPackageError, "truncated must be boolean"):
            self.validate(manifest_rows(), rows)


if __name__ == "__main__":
    unittest.main()
