#!/usr/bin/env python3
"""Blind QuickAudit adapter for the frozen ScanNet v5.2 G->S queues."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import quick_semantic_audit as core


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "data/semantic_audit_v52_scannet_unique_referent_gs_formal"
PACK_PROTOCOL = "gsa-depth-v5.2-scannet-unique-referent-gs-formal-v1"
AUDIT_CONTRACT = "scannet_sensor_unique_referent_grounding_to_depth"
DISPLAY_FORM = "depth"
NEW_TARGET = 33
FORMAL_TARGET = 40
ERROR_CODE_LABELS = {
    "wrong_object": "Wrong object or identity",
    "bbox_misaligned": "Bounding box is misaligned",
    "identity_ambiguous": "Object identity is ambiguous",
    "ambiguous_depth": "Near/far order is visually ambiguous",
    "wrong_geometry": "Wrong box geometry",
    "wrong_relation_or_order": "Wrong near/far order",
    "occlusion_or_truncation": "Occlusion or truncation prevents verification",
    "missing_content": "Required object or relation is missing",
    "unsuitable_source": "Source image is unsuitable",
    "schema_or_format": "Malformed or inconsistent label",
    "other": "Other concrete semantic error",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_relative_file(pack: Path, binding: Any, name: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise ValueError(f"invalid {name} binding")
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"invalid {name} path")
    path = (pack / relative).resolve()
    try:
        path.relative_to(pack)
    except ValueError as exc:
        raise ValueError(f"{name} path escapes the pack") from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != binding.get("bytes")
        or file_sha256(path) != binding.get("sha256")
    ):
        raise ValueError(f"{name} binding drift")
    return path


class V52QuickAuditStore(core.QuickAuditStore):
    """Expose only reviewer-visible v5.2 fields and append exact snapshots."""

    def __init__(self, pack: Path, annotator: str):
        if annotator not in {"a", "b"}:
            raise ValueError("annotator must be 'a' or 'b'")
        self.pack = pack.expanduser().resolve()
        self.annotator = annotator
        self.annotator_name = f"annotator_{annotator}"
        self.root = self.pack / self.annotator_name
        self.annotations_path = self.root / "annotations/annotations.jsonl"
        self._lock = threading.Lock()

        manifest_path = self.pack / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing regular manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        integrity = self.manifest.get("integrity") or {}
        manifest_payload = dict(self.manifest)
        manifest_payload.pop("integrity", None)
        if (
            self.manifest.get("protocol_version") != PACK_PROTOCOL
            or self.manifest.get("audit_contract") != AUDIT_CONTRACT
            or self.manifest.get("stage") != "formal_pending_review"
            or self.manifest.get("records") != FORMAL_TARGET
            or integrity.get("manifest_payload_sha256") != canonical_hash(manifest_payload)
        ):
            raise ValueError("not an intact frozen v5.2 unique-referent pack")

        provenance = self.manifest.get("provenance") or {}
        label_path = _verified_relative_file(
            self.pack,
            provenance.get(f"reviewer_{annotator}_labels"),
            f"reviewer {annotator.upper()} labels",
        )
        rows = core._read_jsonl(label_path)
        self.labels: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in self.labels:
                raise ValueError("v5.2 reviewer labels contain a missing or duplicate sample_id")
            self.labels[sample_id] = row

        selected = self.manifest.get("selected")
        if not isinstance(selected, list) or len(selected) != FORMAL_TARGET:
            raise ValueError("v5.2 selected manifest has the wrong size")
        new_rows = [row for row in selected if row.get("lineage") == "new_v52"]
        if len(new_rows) != NEW_TARGET or len(self.labels) != NEW_TARGET:
            raise ValueError("v5.2 reviewer queue must contain exactly 33 new cases")
        if {row.get("sample_id") for row in new_rows} != set(self.labels):
            raise ValueError("v5.2 reviewer labels do not match the new selection")

        self.audit_contract = AUDIT_CONTRACT
        self.forms = [DISPLAY_FORM]
        self.selected: list[dict[str, Any]] = []
        for row in new_rows:
            sample_id = row["sample_id"]
            label = self.labels[sample_id]
            image_path = Path(str(row.get("image_path", ""))).expanduser().resolve()
            if (
                canonical_hash(label) != row.get("label_sha256")
                or label.get("review_required_in_v52") is not True
                or label.get("lineage") != "new_v52"
                or Path(str(label.get("image_path", ""))).expanduser().resolve() != image_path
                or not image_path.is_file()
                or file_sha256(image_path) != row.get("image_sha256")
            ):
                raise ValueError(f"v5.2 label or image binding drift: {sample_id}")
            self.selected.append(
                {
                    "image_path": str(image_path),
                    "sample_ids": {DISPLAY_FORM: sample_id},
                }
            )
        self.latest = self._load_strict_latest_annotations()

    def _load_strict_latest_annotations(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        reviewer_keys: set[str] = set()
        for row in core._read_jsonl(self.annotations_path):
            snapshot = row.get("label_snapshot")
            sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
            if (
                sample_id not in self.labels
                or row.get("case_id") != sample_id
                or canonical_hash(snapshot) != canonical_hash(self.labels[sample_id])
                or not isinstance(row.get("correct"), bool)
            ):
                raise ValueError(f"invalid existing v5.2 annotation: {sample_id!r}")
            key = row.get("annotator_key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"existing v5.2 annotation lacks annotator_key: {sample_id}")
            reviewer_keys.add(key)
            correct = row["correct"]
            error_code = row.get("error_code")
            comment = row.get("comment")
            if correct and error_code not in (None, ""):
                raise ValueError(f"supported annotation has an error code: {sample_id}")
            if not correct and (
                error_code not in ERROR_CODE_LABELS
                or not isinstance(comment, str)
                or not comment.strip()
            ):
                raise ValueError(f"unsupported annotation lacks concrete evidence: {sample_id}")
            latest[sample_id] = row
        if len(reviewer_keys) > 1:
            raise ValueError("v5.2 annotation stream contains multiple reviewer keys")
        return latest

    @staticmethod
    def _display_fields(label: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        public_targets = {
            row["slot"]: row for row in label["public_interface"]["grounding_targets"]
        }
        audit_grounding = {
            row["slot"]: row for row in label["audit_targets"]["grounding"]
        }
        nearer = label["audit_targets"]["spatial"]["nearer_slot"]
        if set(public_targets) != {"A", "B"} or set(audit_grounding) != {"A", "B"}:
            raise ValueError("v5.2 display targets must contain exactly slots A and B")
        payload: list[dict[str, Any]] = []
        groundings: list[dict[str, Any]] = []
        for slot in ("A", "B"):
            expression = public_targets[slot]["referring_expression"]
            payload.append(
                {
                    "label": f"{slot}: {expression}",
                    "rank": 1 if slot == nearer else 2,
                    "relative_depth": 0.0 if slot == nearer else 1.0,
                }
            )
            groundings.append(
                {
                    "object_class": f"{slot}: {expression}",
                    "bbox": audit_grounding[slot]["bbox"],
                }
            )
        return payload, {"groundings": groundings}

    def case(self, index: int) -> dict[str, Any]:
        item = self._check_index(index)
        sample_id = item["sample_ids"][DISPLAY_FORM]
        label = self.labels[sample_id]
        previous = self.latest.get(sample_id)
        payload, context = self._display_fields(label)
        # The browser receives only the public proposal. Case/source identity
        # and the sensor-backed construction contract stay server-side and are
        # preserved only in the saved label snapshot.
        return {
            "index": index,
            "position": index + 1,
            "total": self.image_count,
            "annotator": self.annotator.upper(),
            "forms": [
                {
                    "form": DISPLAY_FORM,
                    "depth_scope": "pairwise",
                    "payload": payload,
                    "gsa_context": context,
                    "previous": None
                    if previous is None
                    else {
                        "correct": previous["correct"],
                        "error_code": previous.get("error_code") or "",
                        "comment": previous.get("comment", ""),
                    },
                }
            ],
            "error_codes": [
                {"code": code, "label": description}
                for code, description in ERROR_CODE_LABELS.items()
            ],
            "complete": sample_id in self.latest,
        }

    def submit(self, index: int, judgments: list[dict[str, Any]]) -> dict[str, Any]:
        item = self._check_index(index)
        if not isinstance(judgments, list) or len(judgments) != 1:
            raise ValueError("exactly one v5.2 judgment is required")
        judgment = judgments[0]
        if not isinstance(judgment, dict) or judgment.get("form") != DISPLAY_FORM:
            raise ValueError("the v5.2 display form must be depth")
        correct = judgment.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("correct must be Boolean")
        error_code = judgment.get("error_code")
        comment = judgment.get("comment", "")
        if not isinstance(comment, str) or len(comment) > 2000:
            raise ValueError("comment must be a string of at most 2000 characters")
        comment = comment.strip()
        if correct:
            if error_code not in (None, ""):
                raise ValueError("error_code must be empty for a supported judgment")
            error_code = None
        elif error_code not in ERROR_CODE_LABELS or not comment:
            raise ValueError(
                "an unsupported v5.2 judgment requires an error code and concrete comment"
            )

        sample_id = item["sample_ids"][DISPLAY_FORM]
        record = {
            "annotation_id": str(uuid.uuid4()),
            "case_id": sample_id,
            "annotator_key": f"quick-audit-v52-{self.annotator}",
            "annotator_name": self.annotator_name,
            "pipeline": "GS_dependency",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correct": correct,
            "error_code": error_code,
            "error_regions": [],
            "comment": comment,
            "label_snapshot": self.labels[sample_id],
        }
        with self._lock:
            self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
            with self.annotations_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.latest[sample_id] = record
        return {
            "saved": 1,
            "stats": self.stats(),
            "next_index": self.next_incomplete(index),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--annotator", required=True, choices=("a", "b"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", default=os.environ.get("GSA_QUICK_AUDIT_TOKEN"))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    core.validate_bind_security(args.host, args.token)
    store = V52QuickAuditStore(args.pack, args.annotator)
    if args.check:
        print(json.dumps(store.stats(), indent=2, sort_keys=True))
        return 0
    server = ThreadingHTTPServer((args.host, args.port), core.make_handler(store, args.token))
    auth_mode = "Bearer token" if args.token else "loopback-only"
    print(
        f"GSA v5.2 quick audit {store.annotator_name}: "
        f"http://{args.host}:{args.port}/ ({auth_mode})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
