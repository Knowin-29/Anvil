#!/usr/bin/env python3
"""Reviewer-C adjudication UI for completed ScanNet v5.2 A/B streams."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import quick_semantic_audit as core  # noqa: E402
import quick_scannet_rgb_depth_v52_unique_referent_audit as v52  # noqa: E402
from tools import check_scannet_rgb_depth_v52_unique_referent_formal as checker  # noqa: E402


DEFAULT_PACK = v52.DEFAULT_PACK
DEFAULT_PORT = 10325
ANNOTATOR_KEY = "quick-audit-v52-c"
WAITING = "WAITING_FOR_A_B"
NOT_REQUIRED = "NO_ADJUDICATION_REQUIRED"
READY = "READY_FOR_ADJUDICATION"


@dataclass(frozen=True)
class ReviewContext:
    pack: Path
    labels: dict[str, dict[str, Any]]
    queue: tuple[dict[str, Any], ...]
    votes_a: dict[str, dict[str, Any]]
    votes_b: dict[str, dict[str, Any]]
    key_a: str
    key_b: str
    stream_a_sha256: str
    stream_b_sha256: str
    disagreement_ids: tuple[str, ...]


def _public_vote(row: dict[str, Any]) -> dict[str, Any]:
    """Return only evidence reviewer C needs to compare the two decisions."""
    return {
        "decision": "Supported" if row["correct"] else "Unsupported",
        "error_type": row.get("error_code") or None,
        "comment": row.get("comment") or "",
    }


def _load_context(pack: Path) -> tuple[dict[str, Any], ReviewContext | None]:
    pack = pack.expanduser().resolve()
    store_a = v52.V52QuickAuditStore(pack, "a")
    store_b = v52.V52QuickAuditStore(pack, "b")
    if store_a.labels.keys() != store_b.labels.keys() or any(
        v52.canonical_hash(store_a.labels[sample_id])
        != v52.canonical_hash(store_b.labels[sample_id])
        for sample_id in store_a.labels
    ):
        raise ValueError("reviewer A/B frozen label queues differ")

    labels = store_a.labels
    path_a = pack / "annotator_a/annotations/annotations.jsonl"
    path_b = pack / "annotator_b/annotations/annotations.jsonl"
    votes_a = checker._latest_votes(path_a)
    votes_b = checker._latest_votes(path_b)
    if not set(votes_a) <= set(labels) or not set(votes_b) <= set(labels):
        raise ValueError("reviewer A/B streams contain a case outside the 33-case queue")
    for sample_id, vote in votes_a.items():
        checker._validate_vote(vote, labels[sample_id], "A")
    for sample_id, vote in votes_b.items():
        checker._validate_vote(vote, labels[sample_id], "B")
    key_a = checker._single_reviewer_key(votes_a, "A")
    key_b = checker._single_reviewer_key(votes_b, "B")
    if key_a is not None and key_b is not None and key_a == key_b:
        raise ValueError("reviewer A and B keys must be independent")

    common = {
        "a_complete": len(votes_a),
        "b_complete": len(votes_b),
        "expected_per_side": v52.NEW_TARGET,
        "service_ready": False,
    }
    if set(votes_a) != set(labels) or set(votes_b) != set(labels):
        c_path = pack / "reviewer_c/annotations/annotations.jsonl"
        if c_path.is_file() and checker._latest_votes(c_path):
            raise ValueError("reviewer C decisions exist before the A/B queues are complete")
        return {"state": WAITING, "disagreement_count": None, **common}, None

    assert key_a is not None and key_b is not None
    if ANNOTATOR_KEY in {key_a, key_b}:
        raise ValueError(f"reviewer A/B key conflicts with fixed reviewer C key {ANNOTATOR_KEY}")
    disagreement_ids = tuple(
        item["sample_ids"][v52.DISPLAY_FORM]
        for item in store_a.selected
        if votes_a[item["sample_ids"][v52.DISPLAY_FORM]]["correct"]
        != votes_b[item["sample_ids"][v52.DISPLAY_FORM]]["correct"]
    )
    if not disagreement_ids:
        c_path = pack / "reviewer_c/annotations/annotations.jsonl"
        if c_path.is_file() and checker._latest_votes(c_path):
            raise ValueError("reviewer C stream is non-empty but A/B have no disagreements")
        return {"state": NOT_REQUIRED, "disagreement_count": 0, **common}, None

    context = ReviewContext(
        pack=pack,
        labels=labels,
        queue=tuple(store_a.selected),
        votes_a=votes_a,
        votes_b=votes_b,
        key_a=key_a,
        key_b=key_b,
        stream_a_sha256=v52.file_sha256(path_a),
        stream_b_sha256=v52.file_sha256(path_b),
        disagreement_ids=disagreement_ids,
    )
    return {
        "state": READY,
        "disagreement_count": len(disagreement_ids),
        "expected_decisions": len(disagreement_ids),
        "service_ready": True,
        **{key: value for key, value in common.items() if key != "service_ready"},
    }, context


def adjudication_status(pack: Path) -> dict[str, Any]:
    status, context = _load_context(pack)
    if context is None:
        status["c_complete"] = 0
        return status
    store = V52AdjudicationStore._from_context(context)
    status["c_complete"] = len(store.latest)
    status["complete"] = len(store.latest) == len(context.disagreement_ids)
    return status


class V52AdjudicationStore(core.QuickAuditStore):
    """Expose exactly the completed A/B disagreement set to reviewer C."""

    def __init__(self, pack: Path):
        status, context = _load_context(pack)
        if context is None:
            raise ValueError(f"reviewer C service is unavailable: {status['state']}")
        self._initialize(context)

    @classmethod
    def _from_context(cls, context: ReviewContext) -> "V52AdjudicationStore":
        store = cls.__new__(cls)
        store._initialize(context)
        return store

    def _initialize(self, context: ReviewContext) -> None:
        self.pack = context.pack
        self.annotator = "c"
        self.annotator_name = "reviewer_c"
        self.root = self.pack / self.annotator_name
        self.annotations_path = self.root / "annotations/annotations.jsonl"
        self._lock = threading.Lock()
        self.audit_contract = v52.AUDIT_CONTRACT
        self.forms = [v52.DISPLAY_FORM]
        self.labels = context.labels
        self.votes_a = context.votes_a
        self.votes_b = context.votes_b
        self._ab_stream_bindings = {
            self.pack / "annotator_a/annotations/annotations.jsonl": context.stream_a_sha256,
            self.pack / "annotator_b/annotations/annotations.jsonl": context.stream_b_sha256,
        }
        wanted = set(context.disagreement_ids)
        self.selected = [
            item
            for item in context.queue
            if item["sample_ids"][v52.DISPLAY_FORM] in wanted
        ]
        if tuple(
            item["sample_ids"][v52.DISPLAY_FORM] for item in self.selected
        ) != context.disagreement_ids:
            raise ValueError("reviewer C disagreement order drift")
        self.latest = self._load_strict_latest_annotations()

    def _assert_ab_streams_unchanged(self) -> None:
        for path, expected in self._ab_stream_bindings.items():
            if not path.is_file() or v52.file_sha256(path) != expected:
                raise ValueError(
                    "reviewer A/B stream changed after reviewer C started; restart the launcher"
                )

    def _load_strict_latest_annotations(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        expected = {
            item["sample_ids"][v52.DISPLAY_FORM] for item in self.selected
        }
        for row in core._read_jsonl(self.annotations_path):
            snapshot = row.get("label_snapshot")
            sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
            if sample_id not in expected:
                raise ValueError("reviewer C stream contains a non-disagreement case")
            checker._validate_vote(row, self.labels[sample_id], "C")
            if row.get("annotator_key") != ANNOTATOR_KEY:
                raise ValueError(f"reviewer C key must be exactly {ANNOTATOR_KEY}")
            latest[sample_id] = row
        return latest

    def case(self, index: int) -> dict[str, Any]:
        self._assert_ab_streams_unchanged()
        item = self._check_index(index)
        sample_id = item["sample_ids"][v52.DISPLAY_FORM]
        label = self.labels[sample_id]
        previous = self.latest.get(sample_id)
        payload, context = v52.V52QuickAuditStore._display_fields(label)
        return {
            "index": index,
            "position": index + 1,
            "total": self.image_count,
            "annotator": "C",
            "forms": [
                {
                    "form": v52.DISPLAY_FORM,
                    "depth_scope": "pairwise",
                    "object_class": {
                        "reviewer_A": _public_vote(self.votes_a[sample_id]),
                        "reviewer_B": _public_vote(self.votes_b[sample_id]),
                    },
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
                for code, description in v52.ERROR_CODE_LABELS.items()
            ],
            "complete": sample_id in self.latest,
        }

    def image_path(self, index: int) -> Path:
        self._assert_ab_streams_unchanged()
        return super().image_path(index)

    def stats(self) -> dict[str, Any]:
        self._assert_ab_streams_unchanged()
        return super().stats()

    def submit(self, index: int, judgments: list[dict[str, Any]]) -> dict[str, Any]:
        self._assert_ab_streams_unchanged()
        item = self._check_index(index)
        if not isinstance(judgments, list) or len(judgments) != 1:
            raise ValueError("exactly one reviewer C judgment is required")
        judgment = judgments[0]
        if not isinstance(judgment, dict) or judgment.get("form") != v52.DISPLAY_FORM:
            raise ValueError("the reviewer C display form must be depth")
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
        elif error_code not in v52.ERROR_CODE_LABELS or not comment:
            raise ValueError(
                "an unsupported reviewer C judgment requires an error code and concrete comment"
            )

        sample_id = item["sample_ids"][v52.DISPLAY_FORM]
        record = {
            "annotation_id": str(uuid.uuid4()),
            "case_id": sample_id,
            "annotator_key": ANNOTATOR_KEY,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=os.environ.get("GSA_QUICK_AUDIT_TOKEN"))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        core.validate_bind_security(args.host, args.token)
        if args.check:
            print(json.dumps(adjudication_status(args.pack), indent=2, sort_keys=True))
            return 0
        store = V52AdjudicationStore(args.pack)
    except (ValueError, checker.CheckError) as exc:
        print(f"reviewer C preflight failed: {exc}", file=sys.stderr)
        return 2
    server = ThreadingHTTPServer((args.host, args.port), core.make_handler(store, args.token))
    auth_mode = "Bearer token" if args.token else "loopback-only"
    print(
        f"GSA v5.2 reviewer C: http://{args.host}:{args.port}/ ({auth_mode})",
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
