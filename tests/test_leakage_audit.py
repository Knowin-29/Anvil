from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from trains import check_benchmark_leakage as audit


class LeakageAuditTests(unittest.TestCase):
    def run_audit(self, train: Path, bench: Path, out: Path) -> int:
        argv = [
            "check_benchmark_leakage.py",
            "--train", str(train),
            "--bench", str(bench),
            "--out", str(out),
            "--image-roots", str(train.parent),
        ]
        previous = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    audit.main()
            return int(raised.exception.code)
        finally:
            sys.argv = previous

    @staticmethod
    def write_record(path: Path, image: str, question: str, answer: str):
        record = {
            "images": [image],
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_exact_content_overlap_is_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.bin").write_bytes(b"same-image")
            (root / "bench.bin").write_bytes(b"same-image")
            train, bench, out = root / "train.jsonl", root / "bench.jsonl", root / "out.json"
            self.write_record(train, "train.bin", "train question", "train answer")
            self.write_record(bench, "bench.bin", "bench question", "bench answer")
            self.assertEqual(self.run_audit(train, bench, out), 3)
            self.assertEqual(json.loads(out.read_text())["verdict"], "LEAKAGE_DETECTED")

    def test_unresolved_image_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, bench, out = root / "train.jsonl", root / "bench.jsonl", root / "out.json"
            self.write_record(train, "missing-train.bin", "train question", "train answer")
            self.write_record(bench, "missing-bench.bin", "bench question", "bench answer")
            self.assertEqual(self.run_audit(train, bench, out), 4)
            self.assertEqual(json.loads(out.read_text())["verdict"], "INCOMPLETE")

    def test_resolved_disjoint_records_are_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.bin").write_bytes(b"train-image")
            (root / "bench.bin").write_bytes(b"bench-image")
            train, bench, out = root / "train.jsonl", root / "bench.jsonl", root / "out.json"
            self.write_record(train, "train.bin", "train question", "train answer")
            self.write_record(bench, "bench.bin", "bench question", "bench answer")
            self.assertEqual(self.run_audit(train, bench, out), 0)
            self.assertEqual(json.loads(out.read_text())["verdict"], "CLEAN")


if __name__ == "__main__":
    unittest.main()
