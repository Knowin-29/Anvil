from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "quick_semantic_audit_server",
    TOOLS / "quick_semantic_audit_server.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
AuditStore = MODULE.AuditStore
validate_bind_security = MODULE.core.validate_bind_security


class QuickAuditTests(unittest.TestCase):
    def make_pack(self, root: Path, with_triage: bool = False) -> Path:
        pack = root / "pack"
        image = root / "image.jpg"
        image.write_bytes(b"test-image")
        manifest = {
            "images": 1,
            "forms": ["bbox", "point"],
            "selected": [{
                "image_path": str(image),
                "sample_ids": {"bbox": "sample-b", "point": "sample-p"},
            }],
        }
        pack.mkdir()
        (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for annotator in ("annotator_a", "annotator_b"):
            labels = pack / annotator / "labels" / "G_grounding"
            labels.mkdir(parents=True)
            rows = [
                {
                    "sample_id": "sample-b",
                    "task_form": "bbox",
                    "pipeline": "G_grounding",
                    "image_path": str(image),
                    "payload": [],
                    "_verification": {"verdict": "verified"},
                },
                {
                    "sample_id": "sample-p",
                    "task_form": "point",
                    "pipeline": "G_grounding",
                    "image_path": str(image),
                    "payload": [],
                    "_verification": {"verdict": "review"},
                },
            ]
            (labels / "labels.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        if with_triage:
            (pack / "machine_triage.jsonl").write_text(
                json.dumps({
                    "sample_id": "sample-p",
                    "risk": "high",
                    "reason": "edge case",
                }) + "\n",
                encoding="utf-8",
            )
        return pack

    def test_writes_compiler_compatible_append_only_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditStore(self.make_pack(Path(tmp)), "a")
            result = store.submit(0, [
                {"form": "bbox", "correct": True, "comment": "visible"},
                {
                    "form": "point",
                    "correct": False,
                    "error_code": "wrong_geometry",
                    "comment": "off target",
                },
            ])
            self.assertEqual(result["saved"], 2)
            rows = [
                json.loads(line)
                for line in store.annotations_path.read_text().splitlines()
            ]
            self.assertEqual(rows[0]["label_snapshot"]["sample_id"], "sample-b")
            self.assertIs(rows[0]["correct"], True)
            self.assertIsNone(rows[0]["error_code"])
            self.assertEqual(rows[1]["error_code"], "wrong_geometry")
            self.assertEqual(store.stats()["complete_images"], 1)

    def test_latest_append_wins_without_changing_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditStore(self.make_pack(Path(tmp)), "b")
            label_path = next((store.root / "labels").rglob("*.jsonl"))
            labels_before = label_path.read_bytes()
            store.submit(0, [{
                "form": "bbox",
                "correct": False,
                "error_code": "wrong_geometry",
            }])
            store.submit(0, [{"form": "bbox", "correct": True}])
            reloaded = AuditStore(store.pack, "b")
            self.assertIs(
                reloaded.case(0)["forms"][0]["previous"]["correct"],
                True,
            )
            self.assertEqual(label_path.read_bytes(), labels_before)

    def test_blind_response_excludes_machine_metadata_and_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditStore(self.make_pack(Path(tmp), with_triage=True), "a")
            forms = {form["form"]: form for form in store.case(0)["forms"]}
            self.assertNotIn("verification", forms["bbox"])
            self.assertNotIn("machine_triage", forms["point"])
            self.assertIsNone(forms["point"]["previous"])
            self.assertFalse(store.annotations_path.exists())
            page = MODULE._page_bytes().decode()
            self.assertNotIn("_verification", page)
            self.assertNotIn("machine_triage", page)
            self.assertNotIn("Mark all supported", page)
            self.assertIn("Select an error type", page)

    def test_reads_legacy_v2_negative_vote_without_error_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditStore(self.make_pack(Path(tmp)), "a")
            store.annotations_path.parent.mkdir(parents=True)
            store.annotations_path.write_text(json.dumps({
                "correct": False,
                "comment": "legacy note",
                "label_snapshot": store.labels["sample-p"],
            }) + "\n", encoding="utf-8")
            reloaded = AuditStore(store.pack, "a")
            previous = {
                form["form"]: form["previous"]
                for form in reloaded.case(0)["forms"]
            }["point"]
            self.assertIs(previous["correct"], False)
            self.assertEqual(previous["error_code"], "")
            self.assertEqual(previous["comment"], "legacy note")

    def test_overlay_maps_thousandth_coordinates_to_display_extent(self):
        page = MODULE._page_bytes().decode()
        self.assertIn("var COORDINATE_EXTENT=1000", page)
        self.assertIn(
            "point[0]*width/COORDINATE_EXTENT,point[1]*height/COORDINATE_EXTENT",
            page,
        )
        self.assertNotIn("width/image.naturalWidth", page)
        self.assertNotIn("height/image.naturalHeight", page)
        self.assertIn('if(form.form==="depth")', page)
        self.assertIn("context.groundings", page)
        self.assertIn("var ACTION_FORMS={action_cot:true,dof:true,affordance:true}", page)
        self.assertIn("new ResizeObserver(drawOverlay)", page)
        self.assertIn("#1 and 0.0 are nearest", page)
        self.assertIn("Every adjacent near-to-far pair must have visible", page)
        self.assertIn("ties or ambiguous pairs are Unsupported", page)

    def test_depth_only_pack_uses_the_same_blind_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "pack"
            image = root / "image.jpg"
            image.write_bytes(b"image")
            manifest = {
                "schema_version": 4,
                "images": 1,
                "forms": ["depth"],
                "selected": [{
                    "image_path": str(image),
                    "sample_ids": {"depth": "v4-depth-only"},
                }],
            }
            pack.mkdir()
            (pack / "manifest.json").write_text(json.dumps(manifest))
            label = {
                "sample_id": "v4-depth-only",
                "task_form": "depth",
                "pipeline": "S_spatial",
                "image_path": str(image),
                "payload": [
                    {"label": "cup", "rank": 1, "relative_depth": 0.0},
                    {"label": "bowl", "rank": 2, "relative_depth": 1.0},
                ],
                "gsa_context": {"groundings": [
                    {"object_class": "cup", "bbox": [10, 10, 100, 100]},
                    {"object_class": "bowl", "bbox": [200, 200, 300, 300]},
                ]},
            }
            for annotator in ("annotator_a", "annotator_b"):
                labels = pack / annotator / "labels/S_spatial"
                labels.mkdir(parents=True)
                (labels / "depth.jsonl").write_text(json.dumps(label) + "\n")
            store = AuditStore(pack, "a")
            case = store.case(0)
            self.assertEqual([item["form"] for item in case["forms"]], ["depth"])
            self.assertIsNone(case["forms"][0]["previous"])
            result = store.submit(0, [{"form": "depth", "correct": True}])
            self.assertEqual(result["stats"]["expected_decisions"], 1)
            self.assertEqual(result["stats"]["complete_images"], 1)

    def test_embedded_javascript_has_valid_syntax(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is unavailable")
        page = MODULE._page_bytes().decode()
        match = re.search(r"<script>\s*(.*?)\s*</script>", page, flags=re.DOTALL)
        self.assertIsNotNone(match)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "quick_semantic_audit.js"
            script.write_text(match.group(1), encoding="utf-8")
            result = subprocess.run(
                [node, "--check", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_invalid_vote_and_unsafe_bind(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditStore(self.make_pack(Path(tmp)), "a")
            with self.assertRaisesRegex(ValueError, "Boolean"):
                store.submit(0, [{"form": "bbox", "correct": "yes"}])
            with self.assertRaisesRegex(ValueError, "requires one error_code"):
                store.submit(0, [{"form": "bbox", "correct": False}])
            with self.assertRaisesRegex(ValueError, "requires one error_code"):
                store.submit(0, [{
                    "form": "bbox",
                    "correct": False,
                    "error_code": "not-a-code",
                }])
            with self.assertRaisesRegex(ValueError, "must be empty"):
                store.submit(0, [{
                    "form": "bbox",
                    "correct": True,
                    "error_code": "wrong_geometry",
                }])
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_bind_security("0.0.0.0", None)
        validate_bind_security("0.0.0.0", "secret")
        validate_bind_security("127.0.0.1", None)


if __name__ == "__main__":
    unittest.main()
