#!/usr/bin/env python3
"""Fail-closed third-reviewer adjudication for semantic-audit disagreements."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quick_semantic_audit import ERROR_CODE_LABELS


SCHEMA = "gsa-semantic-adjudication-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        if allow_missing:
            return []
        raise FileNotFoundError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected an object in {path}:{line_number}")
        rows.append(row)
    return rows


def latest_annotations(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), 1):
        snapshot = row.get("label_snapshot")
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_number}: missing label_snapshot.sample_id")
        latest[sample_id] = row
    return latest


def annotation_keys(rows: dict[str, dict[str, Any]], label: str) -> set[str]:
    keys = {
        row.get("annotator_key")
        for row in rows.values()
        if isinstance(row.get("annotator_key"), str) and row["annotator_key"].strip()
    }
    if len(keys) != 1:
        raise ValueError(f"{label} must contain exactly one non-empty annotator_key")
    return keys


class SourceAudit:
    """Validated, immutable view of an A/B semantic-audit pack."""

    def __init__(self, pack: Path):
        self.pack = pack.resolve()
        self.manifest_path = self.pack / "manifest.json"
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid source manifest {self.manifest_path}: {exc}") from exc
        if not isinstance(self.manifest, dict):
            raise ValueError("source manifest must be an object")
        schema_version = int(self.manifest.get("schema_version", -1))
        if schema_version not in {3, 5}:
            raise ValueError(
                "adjudication requires a secure schema-v3 or schema-v5 audit pack"
            )
        if schema_version == 5 and (
            self.manifest.get("forms") != ["depth"]
            or self.manifest.get("audit_contract") != "scannet_sensor_pairwise"
            or (self.manifest.get("training_materialization") or {}).get("eligible")
            is not False
        ):
            raise ValueError("schema-v5 adjudication source contract drift")
        self.schema_version = schema_version
        self.forms = self.manifest.get("forms")
        self.selected = self.manifest.get("selected")
        if not isinstance(self.forms, list) or not self.forms:
            raise ValueError("source manifest forms must be non-empty")
        if not isinstance(self.selected, list) or not self.selected:
            raise ValueError("source manifest selected must be non-empty")
        if self.manifest.get("images") != len(self.selected):
            raise ValueError("source manifest image count does not match selected")

        self.expected: dict[str, tuple[str, dict[str, Any]]] = {}
        for item_index, item in enumerate(self.selected):
            if not isinstance(item, dict):
                raise ValueError(f"source selected[{item_index}] is not an object")
            sample_ids = item.get("sample_ids")
            if not isinstance(sample_ids, dict) or set(sample_ids) != set(self.forms):
                raise ValueError(f"source selected[{item_index}] has incomplete sample_ids")
            image_path = item.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"source selected[{item_index}] lacks image_path")
            for form in self.forms:
                sample_id = sample_ids[form]
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"invalid sample ID for {image_path}/{form}")
                if sample_id in self.expected:
                    raise ValueError(f"duplicate source sample ID: {sample_id}")
                self.expected[sample_id] = (form, item)

        self.a_path = self.pack / "annotator_a/annotations/annotations.jsonl"
        self.b_path = self.pack / "annotator_b/annotations/annotations.jsonl"
        self.a = latest_annotations(self.a_path)
        self.b = latest_annotations(self.b_path)
        if set(self.a) != set(self.expected) or set(self.b) != set(self.expected):
            raise ValueError("A/B annotations must cover exactly every manifest sample")
        self.keys_a = annotation_keys(self.a, "annotator_a")
        self.keys_b = annotation_keys(self.b, "annotator_b")
        if self.keys_a & self.keys_b:
            raise ValueError("annotator_a and annotator_b must use different annotator keys")

        self.labels_a = self._load_labels("annotator_a")
        labels_b = self._load_labels("annotator_b")
        if set(self.labels_a) != set(self.expected) or set(labels_b) != set(self.expected):
            raise ValueError("A/B packed labels must cover exactly every manifest sample")
        for sample_id, (form, item) in self.expected.items():
            label = self.labels_a[sample_id]
            expected_hash = (item.get("label_hashes") or {}).get(form)
            if not isinstance(expected_hash, str) or content_hash(label) != expected_hash:
                raise ValueError(f"source label hash mismatch for {sample_id}")
            if content_hash(labels_b[sample_id]) != expected_hash:
                raise ValueError(f"annotator_b label hash mismatch for {sample_id}")
            for side, row in (("A", self.a[sample_id]), ("B", self.b[sample_id])):
                value = row.get("correct")
                if not isinstance(value, bool):
                    raise ValueError(f"reviewer {side} has non-Boolean vote for {sample_id}")
                if value is False and not isinstance(row.get("error_code"), str):
                    raise ValueError(f"reviewer {side} negative vote lacks error_code for {sample_id}")
                if content_hash(row.get("label_snapshot")) != expected_hash:
                    raise ValueError(f"reviewer {side} label snapshot drift for {sample_id}")

    def _load_labels(self, root_name: str) -> dict[str, dict[str, Any]]:
        label_root = self.pack / root_name / "labels"
        paths = sorted(label_root.rglob("*.jsonl")) if label_root.is_dir() else []
        if not paths:
            raise ValueError(f"{root_name} contains no packed labels")
        out: dict[str, dict[str, Any]] = {}
        for path in paths:
            for row in read_jsonl(path):
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"packed label lacks sample_id in {path}")
                if sample_id in out:
                    raise ValueError(f"duplicate packed label {sample_id} in {root_name}")
                out[sample_id] = row
        return out

    @property
    def reviewer_keys(self) -> set[str]:
        return self.keys_a | self.keys_b

    def disagreements(self, excluded_forms: set[str]) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for item in self.selected:
            for form in self.forms:
                sample_id = item["sample_ids"][form]
                if form in excluded_forms:
                    continue
                if self.a[sample_id]["correct"] == self.b[sample_id]["correct"]:
                    continue
                label = self.labels_a[sample_id]
                image_path = Path(item["image_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(f"missing source image: {image_path}")
                expected_image_hash = item.get("image_sha256")
                if not isinstance(expected_image_hash, str) or file_hash(image_path) != expected_image_hash:
                    raise ValueError(f"source image hash mismatch: {image_path}")
                cases.append({
                    "index": len(cases),
                    "sample_id": sample_id,
                    "task_form": form,
                    "image_path": str(image_path.resolve()),
                    "image_sha256": expected_image_hash,
                    "label_sha256": content_hash(label),
                    "scene_type": label.get("scene_type", ""),
                    "source_bucket": label.get("source_bucket", ""),
                })
        return cases


def build_pack(
    source_pack: Path,
    out: Path,
    *,
    excluded_forms: set[str],
    expected_count: int,
    reviewer_key: str,
) -> dict[str, Any]:
    source = SourceAudit(source_pack)
    reviewer_key = reviewer_key.strip()
    if not reviewer_key:
        raise ValueError("reviewer_key must be non-empty")
    if reviewer_key in source.reviewer_keys:
        raise ValueError("reviewer C key must differ from A and B")
    unknown_exclusions = excluded_forms - set(source.forms)
    if unknown_exclusions:
        raise ValueError(f"unknown excluded forms: {sorted(unknown_exclusions)}")
    cases = source.disagreements(excluded_forms)
    if len(cases) != expected_count:
        raise ValueError(
            f"expected {expected_count} disagreements after exclusions, found {len(cases)}"
        )

    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty adjudication pack: {out}")
    out.mkdir(parents=True, exist_ok=True)
    labels = [source.labels_a[case["sample_id"]] for case in cases]
    label_path = out / "labels/candidates.jsonl"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels),
        encoding="utf-8",
    )
    (out / "reviewer_c").mkdir(parents=True, exist_ok=True)

    case_fingerprint = content_hash([
        {key: case[key] for key in ("sample_id", "task_form", "image_sha256", "label_sha256")}
        for case in cases
    ])
    audit_id = source.manifest.get("audit_id")
    manifest = {
        "schema": SCHEMA,
        "adjudication_id": f"{audit_id}-c-{case_fingerprint[:12]}",
        "source_audit_id": audit_id,
        "source_fingerprint": source.manifest.get("source_fingerprint"),
        "source_pack": str(source.pack),
        "source_manifest_sha256": file_hash(source.manifest_path),
        "source_annotation_sha256": {
            "annotator_a": file_hash(source.a_path),
            "annotator_b": file_hash(source.b_path),
        },
        "reviewer_key": reviewer_key,
        "excluded_forms": sorted(excluded_forms),
        "case_count": len(cases),
        "case_fingerprint": case_fingerprint,
        "labels_sha256": file_hash(label_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest(pack: Path) -> dict[str, Any]:
    path = pack / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid adjudication manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"adjudication manifest must use schema {SCHEMA}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or manifest.get("case_count") != len(cases) or not cases:
        raise ValueError("adjudication manifest has an invalid case count")
    return manifest


def validate_pack(pack: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    label_path = pack / "labels/candidates.jsonl"
    if file_hash(label_path) != manifest.get("labels_sha256"):
        raise ValueError("adjudication candidate-label file has drifted")
    labels: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(label_path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in labels:
            raise ValueError(f"invalid or duplicate candidate sample_id: {sample_id!r}")
        labels[sample_id] = row
    expected_ids: set[str] = set()
    fingerprint_rows = []
    for expected_index, case in enumerate(manifest["cases"]):
        if not isinstance(case, dict) or case.get("index") != expected_index:
            raise ValueError(f"invalid adjudication case index {expected_index}")
        sample_id = case.get("sample_id")
        form = case.get("task_form")
        if not isinstance(sample_id, str) or sample_id in expected_ids or not isinstance(form, str):
            raise ValueError(f"invalid adjudication case {expected_index}")
        expected_ids.add(sample_id)
        label = labels.get(sample_id)
        if label is None or label.get("task_form") != form:
            raise ValueError(f"candidate identity mismatch for {sample_id}")
        if content_hash(label) != case.get("label_sha256"):
            raise ValueError(f"candidate label hash mismatch for {sample_id}")
        image_path = Path(str(case.get("image_path", "")))
        if not image_path.is_file() or file_hash(image_path) != case.get("image_sha256"):
            raise ValueError(f"candidate image hash mismatch for {sample_id}")
        fingerprint_rows.append({
            key: case[key]
            for key in ("sample_id", "task_form", "image_sha256", "label_sha256")
        })
    if set(labels) != expected_ids:
        raise ValueError("candidate labels do not exactly match adjudication cases")
    if content_hash(fingerprint_rows) != manifest.get("case_fingerprint"):
        raise ValueError("adjudication case fingerprint has drifted")
    return labels


def validate_frozen_source(manifest: dict[str, Any]) -> SourceAudit:
    source = SourceAudit(Path(manifest["source_pack"]))
    if file_hash(source.manifest_path) != manifest.get("source_manifest_sha256"):
        raise ValueError("source semantic-audit manifest changed after adjudication-pack creation")
    frozen = manifest.get("source_annotation_sha256") or {}
    if file_hash(source.a_path) != frozen.get("annotator_a"):
        raise ValueError("annotator_a stream changed after adjudication-pack creation")
    if file_hash(source.b_path) != frozen.get("annotator_b"):
        raise ValueError("annotator_b stream changed after adjudication-pack creation")
    reviewer_key = manifest.get("reviewer_key")
    if not isinstance(reviewer_key, str) or not reviewer_key.strip():
        raise ValueError("adjudication manifest lacks reviewer_key")
    if reviewer_key in source.reviewer_keys:
        raise ValueError("reviewer C key collides with an A/B reviewer key")
    recomputed = source.disagreements(set(manifest.get("excluded_forms") or []))
    fields = ("sample_id", "task_form", "image_path", "image_sha256", "label_sha256")
    current = [{key: row[key] for key in fields} for row in recomputed]
    frozen_cases = [{key: row[key] for key in fields} for row in manifest["cases"]]
    if current != frozen_cases:
        raise ValueError("current A/B disagreement set differs from the frozen C-review set")
    return source


class AdjudicationStore:
    """Blind UI store: candidate labels in, reviewer-C decisions out."""

    def __init__(self, pack: Path):
        self.pack = pack.resolve()
        self.manifest = load_manifest(self.pack)
        self.labels = validate_pack(self.pack, self.manifest)
        validate_frozen_source(self.manifest)
        self.cases = self.manifest["cases"]
        self.reviewer_key = self.manifest["reviewer_key"]
        self.annotations_path = self.pack / "reviewer_c/annotations.jsonl"
        self._lock = threading.Lock()
        self.latest = self._load_latest()

    def _load_latest(self) -> dict[str, dict[str, Any]]:
        expected = {case["sample_id"] for case in self.cases}
        by_id = {case["sample_id"]: case for case in self.cases}
        latest: dict[str, dict[str, Any]] = {}
        for line_number, row in enumerate(read_jsonl(self.annotations_path, allow_missing=True), 1):
            sample_id = row.get("sample_id")
            if sample_id not in expected:
                raise ValueError(f"unexpected reviewer-C sample at line {line_number}: {sample_id}")
            if row.get("annotator_key") != self.reviewer_key:
                raise ValueError(f"reviewer-C key mismatch at line {line_number}")
            value = row.get("correct")
            if not isinstance(value, bool):
                raise ValueError(f"reviewer-C vote is not Boolean at line {line_number}")
            error_code = row.get("error_code")
            if value and error_code not in (None, ""):
                raise ValueError(f"supported reviewer-C vote has error_code at line {line_number}")
            if not value and error_code not in ERROR_CODE_LABELS:
                raise ValueError(f"unsupported reviewer-C vote lacks valid error_code at line {line_number}")
            if content_hash(row.get("label_snapshot")) != by_id[sample_id]["label_sha256"]:
                raise ValueError(f"reviewer-C label snapshot drift at line {line_number}")
            latest[sample_id] = row
        return latest

    @property
    def image_count(self) -> int:
        return len(self.cases)

    def _case(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.image_count:
            raise IndexError(f"adjudication index {index} is out of range")
        return self.cases[index]

    def case(self, index: int) -> dict[str, Any]:
        case = self._case(index)
        label = self.labels[case["sample_id"]]
        previous = self.latest.get(case["sample_id"])
        form = {
            "form": case["task_form"],
            "sample_id": case["sample_id"],
            "audit_contract": label.get("audit_contract"),
            "depth_scope": label.get("depth_scope"),
            "pipeline": label.get("pipeline"),
            "object_class": label.get("object_class"),
            "payload": label.get("payload"),
            "gsa_context": label.get("gsa_context"),
            "bbox": label.get("bbox"),
            "point_2d": label.get("point_2d"),
            "contact_point_2d": label.get("contact_point_2d"),
            "grasp_point_2d": label.get("grasp_point_2d"),
            "polygon": label.get("polygon"),
            "previous": None if previous is None else {
                "correct": previous["correct"],
                "error_code": previous.get("error_code") or "",
                "comment": previous.get("comment", ""),
            },
        }
        return {
            "index": index,
            "position": index + 1,
            "total": self.image_count,
            "annotator": "C",
            "scene_type": case.get("scene_type", ""),
            "source_bucket": case.get("source_bucket", ""),
            "image_name": Path(case["image_path"]).name,
            "forms": [form],
            "error_codes": [
                {"code": code, "label": label}
                for code, label in ERROR_CODE_LABELS.items()
            ],
            "complete": previous is not None,
        }

    def image_path(self, index: int) -> Path:
        return Path(self._case(index)["image_path"])

    def stats(self) -> dict[str, Any]:
        by_form: dict[str, int] = {}
        for case in self.cases:
            form = case["task_form"]
            by_form.setdefault(form, 0)
            if case["sample_id"] in self.latest:
                by_form[form] += 1
        return {
            "annotator": "C",
            "images": self.image_count,
            "complete_images": len(self.latest),
            "decisions": len(self.latest),
            "expected_decisions": self.image_count,
            "by_form": by_form,
            "blind_to_ab": True,
        }

    def next_incomplete(self, after: int = -1) -> int:
        order = list(range(after + 1, self.image_count)) + list(range(0, after + 1))
        for index in order:
            if self.cases[index]["sample_id"] not in self.latest:
                return index
        return max(0, min(after, self.image_count - 1))

    def submit(self, index: int, judgments: list[dict[str, Any]]) -> dict[str, Any]:
        case = self._case(index)
        if not isinstance(judgments, list) or len(judgments) != 1:
            raise ValueError("reviewer C must submit exactly one judgment per case")
        judgment = judgments[0]
        if not isinstance(judgment, dict) or judgment.get("form") != case["task_form"]:
            raise ValueError("reviewer-C form does not match the frozen case")
        correct = judgment.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("reviewer-C correct must be Boolean")
        error_code = judgment.get("error_code")
        if correct:
            if error_code not in (None, ""):
                raise ValueError("a supported reviewer-C vote cannot contain error_code")
            error_code = None
        elif error_code not in ERROR_CODE_LABELS:
            raise ValueError("an unsupported reviewer-C vote requires a valid error_code")
        comment = judgment.get("comment", "")
        if not isinstance(comment, str) or len(comment) > 2000:
            raise ValueError("reviewer-C comment must be a string of at most 2000 characters")
        label = self.labels[case["sample_id"]]
        record = {
            "annotation_id": str(uuid.uuid4()),
            "sample_id": case["sample_id"],
            "annotator_key": self.reviewer_key,
            "annotator_name": "reviewer_c",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correct": correct,
            "error_code": error_code,
            "comment": comment.strip(),
            "label_snapshot": label,
        }
        with self._lock:
            self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
            with self.annotations_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.latest[case["sample_id"]] = record
        return {
            "saved": 1,
            "stats": self.stats(),
            "next_index": self.next_incomplete(index),
        }


def _canonical_row(manifest: dict[str, Any], case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": case["sample_id"],
        "correct": row["correct"],
        "error_code": row.get("error_code"),
        "comment": row.get("comment", ""),
        "reviewer_key": row["annotator_key"],
        "timestamp": row["timestamp"],
        "source_annotation_id": row["annotation_id"],
        "source_audit_id": manifest.get("source_audit_id"),
        "adjudication_id": manifest.get("adjudication_id"),
        "label_sha256": case["label_sha256"],
    }


def finalize(pack: Path, out: Path) -> dict[str, Any]:
    store = AdjudicationStore(pack)
    source = validate_frozen_source(store.manifest)
    if set(store.latest) != {case["sample_id"] for case in store.cases}:
        missing = [case["sample_id"] for case in store.cases if case["sample_id"] not in store.latest]
        raise ValueError(f"reviewer C is incomplete; missing {len(missing)} decisions")
    if store.reviewer_key in source.reviewer_keys:
        raise ValueError("reviewer C must be independent from A and B")
    rows = [
        _canonical_row(store.manifest, case, store.latest[case["sample_id"]])
        for case in store.cases
    ]
    expected_ids = {case["sample_id"] for case in store.cases}
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with out.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(handle.read().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid canonical adjudication JSON at line {line_number}") from exc
            sample_id = row.get("sample_id") if isinstance(row, dict) else None
            if sample_id not in expected_ids:
                raise ValueError(f"canonical adjudication contains unexpected sample_id {sample_id!r}")
            if sample_id in existing:
                raise ValueError(f"canonical adjudication contains duplicate sample_id {sample_id}")
            existing[sample_id] = row
        for row in rows:
            previous = existing.get(row["sample_id"])
            if previous is not None and previous != row:
                raise ValueError(f"canonical adjudication conflicts for {row['sample_id']}")
        missing_rows = [row for row in rows if row["sample_id"] not in existing]
        handle.seek(0, os.SEEK_END)
        for row in missing_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {
        "verdict": "READY",
        "pack": str(store.pack),
        "out": str(out),
        "expected": len(rows),
        "appended": appended,
        "total": len(rows),
        "excluded_forms": store.manifest.get("excluded_forms"),
        "source_audit_id": store.manifest.get("source_audit_id"),
    }
