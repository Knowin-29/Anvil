from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "eval" / "compile_semantic_audit.py"
BUILDER = ROOT / "trains" / "flywheel" / "build_semantic_audit_pack.py"
FORMS = [
    "action_cot", "affordance", "bbox", "depth",
    "detection", "dof", "mask", "point",
]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SemanticAuditTests(unittest.TestCase):
    def make_v2_pack(self, root: Path, vote_a: bool, vote_b: bool) -> Path:
        pack = root / "pack"
        manifest = {
            "images": 1,
            "forms": ["bbox"],
            "selected": [{"image_path": "/tmp/example.jpg", "sample_ids": {"bbox": "sample-1"}}],
        }
        pack.mkdir(parents=True)
        (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for name, vote in (("annotator_a", vote_a), ("annotator_b", vote_b)):
            annotation_dir = pack / name / "annotations"
            annotation_dir.mkdir(parents=True)
            row = {"correct": vote, "label_snapshot": {"sample_id": "sample-1", "task_form": "bbox"}}
            (annotation_dir / "annotations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return pack

    def make_v3_pack(self, root: Path) -> Path:
        pack = root / "v3_pack"
        selected: list[dict[str, Any]] = []
        labels: dict[str, list[dict[str, Any]]] = {form: [] for form in FORMS}
        form_pairs: dict[str, list[dict[str, str]]] = {form: [] for form in FORMS}
        for index in range(40):
            image = root / "images" / f"image-{index:02d}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{index}".encode())
            item: dict[str, Any] = {
                "image_path": str(image),
                "image_sha256": image_hash(image),
                "sample_ids": {},
                "label_hashes": {},
                "label_versions": {},
                "prompt_versions": {},
            }
            for form in FORMS:
                sample_id = f"v3-{form}-{index:02d}"
                record = {
                    "sample_id": sample_id,
                    "pipeline": f"pipeline-{form}",
                    "task_form": form,
                    "image_path": str(image),
                    "scene_type": f"scene-{index % 4}",
                    "label_version": "label-v3",
                    "prompt_version": "prompt-v3",
                    "payload": [{"value": index}],
                }
                digest = canonical_hash(record)
                labels[form].append(record)
                item["sample_ids"][form] = sample_id
                item["label_hashes"][form] = digest
                item["label_versions"][form] = "label-v3"
                item["prompt_versions"][form] = "prompt-v3"
                form_pairs[form].append({"sample_id": sample_id, "label_sha256": digest})
            selected.append(item)
        manifest = {
            "schema_version": 3,
            "round": "v3-test",
            "audit_id": "semantic-audit-v3-test",
            "source_fingerprint": "a" * 64,
            "images": 40,
            "forms": FORMS,
            "records_per_annotator": 320,
            "per_form_label_hashes": {
                form: canonical_hash(form_pairs[form]) for form in FORMS
            },
            "selected": selected,
        }
        pack.mkdir(parents=True)
        (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for root_name, key in (("annotator_a", "reviewer-a"), ("annotator_b", "reviewer-b")):
            annotations: list[dict[str, Any]] = []
            for form in FORMS:
                label_path = pack / root_name / "labels" / f"pipeline-{form}" / f"{form}.jsonl"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    "".join(json.dumps(record) + "\n" for record in labels[form]), encoding="utf-8"
                )
                annotations.extend({
                    "correct": True,
                    "annotator_key": key,
                    "label_snapshot": record,
                } for record in labels[form])
            annotation_path = pack / root_name / "annotations" / "annotations.jsonl"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in annotations), encoding="utf-8"
            )
        return pack

    def compile(
        self,
        pack: Path,
        out: Path,
        adjudication: Path | None = None,
        extra: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(COMPILER), "--pack", str(pack), "--out", str(out)]
        if adjudication is not None:
            command.extend(["--adjudication", str(adjudication)])
        command.extend(extra or [])
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def rewrite_annotations(self, pack: Path, root_name: str, update) -> None:
        path = pack / root_name / "annotations" / "annotations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        update(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_v2_matching_positive_votes_still_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.compile(self.make_v2_pack(root, True, True), root / "report.json")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((root / "report.json").read_text())["verdict"], "PASS")

    def test_v2_disagreement_blocks_without_adjudication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.compile(self.make_v2_pack(root, True, False), root / "report.json")
            self.assertEqual(result.returncode, 2)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["unresolved_sample_ids"], ["sample-1"])

    def test_v2_adjudication_remains_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self.make_v2_pack(root, True, False)
            adjudication = root / "adjudication.jsonl"
            adjudication.write_text(
                json.dumps({"sample_id": "sample-1", "correct": True, "reason": "visible target"}) + "\n",
                encoding="utf-8",
            )
            result = self.compile(pack, root / "report.json", adjudication, ["--min-agreement", "0"])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_v3_matching_votes_and_hashes_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.compile(self.make_v3_pack(root), root / "report.json")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["thresholds"], {"min_accuracy": 0.8, "min_agreement": 0.8})

    def test_v3_rejects_same_annotator_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self.make_v3_pack(root)
            self.rewrite_annotations(
                pack, "annotator_b", lambda rows: [row.update(annotator_key="reviewer-a") for row in rows]
            )
            result = self.compile(pack, root / "report.json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("must use different annotator_key", result.stdout)

    def test_v3_rejects_negative_vote_without_error_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self.make_v3_pack(root)
            self.rewrite_annotations(pack, "annotator_a", lambda rows: rows[0].update(correct=False))
            result = self.compile(pack, root / "report.json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("lacks error_code", result.stdout)

    def test_v3_rejects_annotation_label_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self.make_v3_pack(root)
            self.rewrite_annotations(
                pack, "annotator_a", lambda rows: rows[0]["label_snapshot"].update(payload=[{"value": 999}])
            )
            result = self.compile(pack, root / "report.json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("label snapshot hash drift", result.stdout)

    def test_adjudication_does_not_change_raw_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self.make_v3_pack(root)
            def disagree(rows):
                rows[0]["correct"] = False
                rows[0]["error_code"] = "wrong_object"
            self.rewrite_annotations(pack, "annotator_a", disagree)
            sample_id = json.loads(
                (pack / "annotator_a/annotations/annotations.jsonl").read_text().splitlines()[0]
            )["label_snapshot"]["sample_id"]
            adjudication = root / "adjudication.jsonl"
            adjudication.write_text(
                json.dumps({"sample_id": sample_id, "correct": True, "reason": "visible", "error_code": "none"}) + "\n",
                encoding="utf-8",
            )
            result = self.compile(pack, root / "report.json", adjudication)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["by_form"]["action_cot"]["raw_agreement"], 39 / 40)
            self.assertEqual(report["by_form"]["action_cot"]["adjudicated"], 1)


class SemanticAuditBuilderTests(unittest.TestCase):
    def make_source(self, root: Path, images: int = 40, omit_version: bool = False) -> Path:
        source = root / "source"
        for form in FORMS:
            rows = []
            for index in range(images):
                image = root / "source_images" / f"image-{index:03d}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                if not image.exists():
                    image.write_bytes(f"source-image-{index}".encode())
                row = {
                    "sample_id": f"source-{form}-{index:03d}",
                    "pipeline": f"pipeline-{form}",
                    "task_form": form,
                    "image_path": str(image),
                    "scene_type": f"scene-{index % 4}",
                    "source_bucket": "clean",
                    "label_version": "label-v3",
                    "prompt_version": "prompt-v3",
                    "payload": [{"value": index}],
                }
                if omit_version and form == FORMS[0] and index == 0:
                    row.pop("prompt_version")
                rows.append(row)
            path = source / "labels" / f"pipeline-{form}" / f"{form}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return source

    def build(self, source: Path, out: Path, extra: list[str] | None = None):
        return subprocess.run(
            [sys.executable, str(BUILDER), "--source", str(source), "--out", str(out), *(extra or [])],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_builder_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "pack"
            out.mkdir()
            (out / "keep.txt").write_text("do not overwrite")
            result = self.build(root / "unused", out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual((out / "keep.txt").read_text(), "do not overwrite")

    def test_builder_requires_label_and_prompt_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.build(self.make_source(root, omit_version=True), root / "pack")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing non-empty prompt_version", result.stderr)
            self.assertFalse((root / "pack").exists())

    def test_builder_excludes_prior_paths_and_content_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source(root, images=81)
            images = sorted((root / "source_images").glob("*.jpg"))
            # Image 40 is a byte-identical copy of excluded image 0 under a new path.
            images[40].write_bytes(images[0].read_bytes())
            exclude = root / "v2_manifest.json"
            exclude.write_text(json.dumps({
                "schema_version": 1,
                "selected": [{"image_path": str(path)} for path in images[:40]],
            }))
            out = root / "pack"
            result = self.build(source, out, ["--exclude-manifest", str(exclude)])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((out / "manifest.json").read_text())
            selected_paths = {item["image_path"] for item in manifest["selected"]}
            selected_hashes = {item["image_sha256"] for item in manifest["selected"]}
            self.assertEqual(len(selected_paths), 40)
            self.assertFalse(selected_paths & {str(path) for path in images[:41]})
            self.assertNotIn(image_hash(images[0]), selected_hashes)
            self.assertEqual(set(manifest["per_form_label_hashes"]), set(FORMS))


if __name__ == "__main__":
    unittest.main()
