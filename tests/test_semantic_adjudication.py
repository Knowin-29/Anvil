from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from semantic_adjudication import AdjudicationStore, build_pack, content_hash, finalize  # noqa: E402


class SemanticAdjudicationTests(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "source"
        forms = ["bbox", "depth", "dof"]
        selected = []
        labels = []
        annotations = {"a": [], "b": []}
        votes = {"bbox": (True, False), "depth": (False, True), "dof": (False, True)}
        for index, target_form in enumerate(forms):
            image = root / f"image-{index}.jpg"
            image.write_bytes(f"image-{index}".encode())
            sample_ids = {form: f"sample-{index}-{form}" for form in forms}
            item = {
                "image_path": str(image),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "sample_ids": sample_ids,
                "label_hashes": {},
            }
            for form in forms:
                label = {
                    "sample_id": sample_ids[form],
                    "task_form": form,
                    "pipeline": "test",
                    "image_path": str(image),
                    "scene_type": "kitchen",
                    "source_bucket": "test",
                    "label_version": "v3",
                    "prompt_version": "v3",
                    "payload": [{"label": form, "bbox_2d": [100, 100, 500, 500]}],
                }
                labels.append(label)
                item["label_hashes"][form] = content_hash(label)
                pair = votes[target_form] if form == target_form else (True, True)
                for side, vote in zip(("a", "b"), pair):
                    annotations[side].append({
                        "annotator_key": f"reviewer-{side}",
                        "correct": vote,
                        "error_code": None if vote else "wrong_geometry",
                        "label_snapshot": label,
                    })
            selected.append(item)
        manifest = {
            "schema_version": 3,
            "audit_id": "audit-test",
            "source_fingerprint": "a" * 64,
            "images": len(selected),
            "forms": forms,
            "selected": selected,
        }
        source.mkdir()
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for side in ("a", "b"):
            label_path = source / f"annotator_{side}/labels/test/labels.jsonl"
            label_path.parent.mkdir(parents=True)
            label_path.write_text(
                "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
            )
            annotation_path = source / f"annotator_{side}/annotations/annotations.jsonl"
            annotation_path.parent.mkdir(parents=True)
            annotation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in annotations[side]), encoding="utf-8"
            )
        return source

    def make_pack(self, root: Path) -> Path:
        pack = root / "adjudication"
        build_pack(
            self.make_source(root), pack,
            excluded_forms={"depth"}, expected_count=2, reviewer_key="reviewer-c",
        )
        return pack

    def test_builder_is_non_depth_and_contains_no_ab_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = self.make_pack(Path(tmp))
            manifest_text = (pack / "manifest.json").read_text()
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["case_count"], 2)
            self.assertEqual({case["task_form"] for case in manifest["cases"]}, {"bbox", "dof"})
            self.assertNotIn('"correct"', manifest_text)
            self.assertNotIn('"comment"', manifest_text)

    def test_store_is_blind_and_negative_vote_requires_error_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AdjudicationStore(self.make_pack(Path(tmp)))
            case = store.case(0)
            self.assertEqual(case["annotator"], "C")
            self.assertEqual(len(case["forms"]), 1)
            self.assertNotIn("annotator_a", json.dumps(case))
            self.assertNotIn("annotator_b", json.dumps(case))
            with self.assertRaisesRegex(ValueError, "requires a valid error_code"):
                store.submit(0, [{"form": case["forms"][0]["form"], "correct": False}])

    def test_finalizer_is_complete_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self.make_pack(root)
            store = AdjudicationStore(pack)
            out = root / "canonical.jsonl"
            first = store.case(0)["forms"][0]
            store.submit(0, [{"form": first["form"], "correct": True}])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                finalize(pack, out)
            second = store.case(1)["forms"][0]
            store.submit(1, [{
                "form": second["form"], "correct": False,
                "error_code": "invalid_action_or_affordance",
            }])
            self.assertEqual(finalize(pack, out)["appended"], 2)
            self.assertEqual(finalize(pack, out)["appended"], 0)
            self.assertEqual(len(out.read_text().splitlines()), 2)

    def test_source_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = self.make_pack(Path(tmp))
            manifest = json.loads((pack / "manifest.json").read_text())
            source_a = Path(manifest["source_pack"]) / "annotator_a/annotations/annotations.jsonl"
            with source_a.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "annotator_a stream changed"):
                AdjudicationStore(pack)


if __name__ == "__main__":
    unittest.main()
