#!/usr/bin/env python3
"""Image-centric, append-only UI for the fixed semantic audit pack.

The inherited annotation viewer moves source records after each judgment. This
tool instead treats labels as read-only and writes only compiler-compatible
annotation records.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "data/semantic_audit_v2"
HTML_PATH = Path(__file__).with_name("quick_semantic_audit.html")
MAX_REQUEST_BYTES = 64 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
ERROR_CODE_LABELS = {
    "wrong_object": "Wrong object or identity",
    "wrong_geometry": "Wrong box, point, mask, or pose",
    "wrong_relation_or_order": "Wrong spatial relation or depth order",
    "invalid_action_or_affordance": "Action, affordance, or DoF is not executable",
    "missing_content": "Required object, relation, or step is missing",
    "ambiguous_evidence": "The image does not support a reliable judgment",
    "unsuitable_source": "Multi-panel, derived, or otherwise unsuitable source",
    "schema_or_format": "Malformed or internally inconsistent output",
    "other": "Other semantic error",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected object in {path}:{line_number}")
        rows.append(row)
    return rows


def validate_bind_security(host: str, token: str | None) -> None:
    if host not in LOOPBACK_HOSTS and not token:
        raise ValueError("non-loopback binding requires --token or GSA_QUICK_AUDIT_TOKEN")


class QuickAuditStore:
    """Read the fixed pack and append judgments for exactly one annotator."""

    def __init__(self, pack: Path, annotator: str):
        if annotator not in {"a", "b"}:
            raise ValueError("annotator must be 'a' or 'b'")
        self.pack = pack.resolve()
        self.annotator = annotator
        self.annotator_name = f"annotator_{annotator}"
        self.root = self.pack / self.annotator_name
        self.annotations_path = self.root / "annotations/annotations.jsonl"
        self._lock = threading.Lock()

        manifest_path = self.pack / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.audit_contract = self.manifest.get("audit_contract")
        if (
            self.manifest.get("schema_version") == 5
            and self.audit_contract != "scannet_sensor_pairwise"
        ):
            raise ValueError("schema5 pack requires scannet_sensor_pairwise audit_contract")
        self.forms = list(self.manifest.get("forms") or [])
        self.selected = list(self.manifest.get("selected") or [])
        if not self.forms or not self.selected:
            raise ValueError("manifest must contain non-empty forms and selected")
        if int(self.manifest.get("images", -1)) != len(self.selected):
            raise ValueError("manifest images count does not match selected records")

        self.labels = self._load_labels()
        self._validate_manifest()
        self.latest = self._load_latest_annotations()

    def _load_labels(self) -> dict[str, dict[str, Any]]:
        labels_dir = self.root / "labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"missing label directory: {labels_dir}")
        labels: dict[str, dict[str, Any]] = {}
        for path in sorted(labels_dir.rglob("*.jsonl")):
            for row in _read_jsonl(path):
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"record without sample_id in {path}")
                if sample_id in labels:
                    raise ValueError(f"duplicate sample_id {sample_id}")
                labels[sample_id] = row
        return labels

    def _validate_manifest(self) -> None:
        seen: set[str] = set()
        for index, item in enumerate(self.selected):
            sample_ids = item.get("sample_ids") or {}
            if set(sample_ids) != set(self.forms):
                raise ValueError(f"manifest item {index} does not contain exactly the declared forms")
            image_path = Path(str(item.get("image_path", "")))
            if not image_path.is_file():
                raise FileNotFoundError(f"missing audit image: {image_path}")
            for form in self.forms:
                sample_id = sample_ids[form]
                if sample_id in seen:
                    raise ValueError(f"sample_id appears twice in manifest: {sample_id}")
                seen.add(sample_id)
                label = self.labels.get(sample_id)
                if label is None:
                    raise ValueError(f"manifest sample is absent from labels: {sample_id}")
                if label.get("task_form") != form:
                    raise ValueError(
                        f"sample {sample_id} has task_form={label.get('task_form')!r}, expected {form!r}"
                    )
                if Path(str(label.get("image_path", ""))).resolve() != image_path.resolve():
                    raise ValueError(f"sample {sample_id} points to a different image")

    def _load_latest_annotations(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        expected = {sid for item in self.selected for sid in item["sample_ids"].values()}
        for row in _read_jsonl(self.annotations_path):
            sample_id = (row.get("label_snapshot") or {}).get("sample_id")
            if sample_id in expected and isinstance(row.get("correct"), bool):
                latest[sample_id] = row
        return latest

    @property
    def image_count(self) -> int:
        return len(self.selected)

    def _check_index(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.image_count:
            raise IndexError(f"image index {index} is out of range")
        return self.selected[index]

    def case(self, index: int) -> dict[str, Any]:
        item = self._check_index(index)
        forms = []
        for form in self.forms:
            sample_id = item["sample_ids"][form]
            label = self.labels[sample_id]
            previous = self.latest.get(sample_id)
            forms.append(
                {
                    "form": form,
                    "sample_id": sample_id,
                    "audit_contract": (
                        self.audit_contract if form == "depth" else None
                    ),
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
                    "previous": None
                    if previous is None
                    else {
                        "correct": previous["correct"],
                        # v2 records predate structured error codes. An empty
                        # value keeps them readable while requiring a choice
                        # before an incorrect decision is re-saved.
                        "error_code": previous.get("error_code") or "",
                        "comment": previous.get("comment", ""),
                    },
                }
            )
        first = self.labels[item["sample_ids"][self.forms[0]]]
        return {
            "index": index,
            "position": index + 1,
            "total": self.image_count,
            "annotator": self.annotator.upper(),
            "scene_type": first.get("scene_type", ""),
            "source_bucket": first.get("source_bucket", ""),
            "image_name": Path(item["image_path"]).name,
            "forms": forms,
            "error_codes": [
                {"code": code, "label": label}
                for code, label in ERROR_CODE_LABELS.items()
            ],
            "complete": all(item["sample_ids"][form] in self.latest for form in self.forms),
        }

    def image_path(self, index: int) -> Path:
        return Path(self._check_index(index)["image_path"])

    def stats(self) -> dict[str, Any]:
        by_form = {form: 0 for form in self.forms}
        complete_images = 0
        for item in self.selected:
            is_complete = True
            for form, sample_id in item["sample_ids"].items():
                if sample_id in self.latest:
                    by_form[form] += 1
                else:
                    is_complete = False
            complete_images += int(is_complete)
        return {
            "annotator": self.annotator.upper(),
            "images": self.image_count,
            "complete_images": complete_images,
            "decisions": len(self.latest),
            "expected_decisions": self.image_count * len(self.forms),
            "by_form": by_form,
        }

    def next_incomplete(self, after: int = -1) -> int:
        order = list(range(after + 1, self.image_count)) + list(range(0, after + 1))
        for index in order:
            item = self.selected[index]
            if any(sample_id not in self.latest for sample_id in item["sample_ids"].values()):
                return index
        return max(0, min(after, self.image_count - 1))

    def submit(self, index: int, judgments: list[dict[str, Any]]) -> dict[str, Any]:
        item = self._check_index(index)
        if not isinstance(judgments, list) or not judgments:
            raise ValueError("judgments must be a non-empty list")
        supplied_forms: set[str] = set()
        records: list[dict[str, Any]] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for judgment in judgments:
            if not isinstance(judgment, dict):
                raise ValueError("each judgment must be an object")
            form = judgment.get("form")
            correct = judgment.get("correct")
            if form not in self.forms or form not in item["sample_ids"]:
                raise ValueError(f"unknown form for image {index}: {form!r}")
            if form in supplied_forms:
                raise ValueError(f"duplicate form: {form}")
            if not isinstance(correct, bool):
                raise ValueError(f"correct must be Boolean for form {form}")
            error_code = judgment.get("error_code")
            if correct:
                if error_code not in (None, ""):
                    raise ValueError(f"error_code must be empty for a supported form {form}")
                error_code = None
            elif not isinstance(error_code, str) or error_code not in ERROR_CODE_LABELS:
                allowed = ", ".join(ERROR_CODE_LABELS)
                raise ValueError(
                    f"unsupported form {form} requires one error_code from: {allowed}"
                )
            comment = judgment.get("comment", "")
            if not isinstance(comment, str) or len(comment) > 2000:
                raise ValueError(f"comment must be a string of at most 2000 characters for form {form}")
            supplied_forms.add(form)
            sample_id = item["sample_ids"][form]
            label = self.labels[sample_id]
            records.append(
                {
                    "annotation_id": str(uuid.uuid4()),
                    "case_id": sample_id,
                    "annotator_key": f"quick-audit-{self.annotator}",
                    "annotator_name": self.annotator_name,
                    "pipeline": label.get("pipeline", ""),
                    "timestamp": timestamp,
                    "correct": correct,
                    "error_code": error_code,
                    "error_regions": [],
                    "comment": comment.strip(),
                    "label_snapshot": label,
                }
            )

        with self._lock:
            self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
            with self.annotations_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            for record in records:
                self.latest[record["label_snapshot"]["sample_id"]] = record
        return {
            "saved": len(records),
            "stats": self.stats(),
            "next_index": self.next_incomplete(index),
        }


def make_handler(store: QuickAuditStore, token: str | None) -> type[BaseHTTPRequestHandler]:
    html = HTML_PATH.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "GSAQuickAudit/1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}")

        def _authorized(self) -> bool:
            if not token:
                return True
            supplied = self.headers.get("Authorization", "")
            return supplied.startswith("Bearer ") and secrets.compare_digest(supplied[7:], token)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' blob:; "
                "connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _api_auth(self) -> bool:
            if self._authorized():
                return True
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "token required"})
            return False

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, html, "text/html; charset=utf-8")
                return
            if not self._api_auth():
                return
            try:
                if path == "/api/stats":
                    self._json(HTTPStatus.OK, store.stats())
                elif path.startswith("/api/case/"):
                    self._json(HTTPStatus.OK, store.case(int(path.rsplit("/", 1)[1])))
                elif path.startswith("/api/image/"):
                    image_path = store.image_path(int(path.rsplit("/", 1)[1]))
                    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                    self._send(HTTPStatus.OK, image_path.read_bytes(), content_type)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (IndexError, ValueError):
                self._json(HTTPStatus.NOT_FOUND, {"error": "case not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._api_auth():
                return
            path = urlparse(self.path).path
            if path != "/api/submit":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                result = store.submit(int(payload["index"]), payload["judgments"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, result)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--annotator", required=True, choices=("a", "b"))
    parser.add_argument("--host", default=os.environ.get("GSA_QUICK_AUDIT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--token", default=os.environ.get("GSA_QUICK_AUDIT_TOKEN"))
    parser.add_argument("--check", action="store_true", help="validate the pack and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_bind_security(args.host, args.token)
    store = QuickAuditStore(args.pack, args.annotator)
    if args.check:
        print(json.dumps(store.stats(), indent=2))
        return 0
    port = args.port or (10320 if args.annotator == "a" else 10321)
    server = ThreadingHTTPServer((args.host, port), make_handler(store, args.token))
    auth_mode = "Bearer token" if args.token else "loopback-only"
    print(
        f"GSA quick audit {store.annotator_name}: http://{args.host}:{port}/ ({auth_mode})",
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
