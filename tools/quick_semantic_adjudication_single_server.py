#!/usr/bin/env python3
"""Reviewer-C server with single-decision UI copy."""
from __future__ import annotations

import argparse
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

import quick_semantic_audit as core
import quick_semantic_audit_server as audit_server
from semantic_adjudication import AdjudicationStore


ROOT = Path(__file__).resolve().parents[1]


class StaticPage:
    def __init__(self, body: bytes):
        self.body = body

    def read_bytes(self) -> bytes:
        return self.body


def single_decision_handler(store: AdjudicationStore, token: str | None):
    body = core.HTML_PATH.read_bytes().replace(
        b"Save 8 decisions", b"Save 1 decision"
    ).replace(b"Update 8 decisions", b"Update 1 decision")
    original = audit_server.core.HTML_PATH
    audit_server.core.HTML_PATH = StaticPage(body)
    try:
        return audit_server.make_handler(store, token)
    finally:
        audit_server.core.HTML_PATH = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "data/semantic_adjudication_v3")
    parser.add_argument("--host", default=os.environ.get("GSA_QUICK_AUDIT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=10322)
    parser.add_argument("--token", default=os.environ.get("GSA_QUICK_AUDIT_TOKEN"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    core.validate_bind_security(args.host, args.token)
    store = AdjudicationStore(args.pack)
    if args.check:
        print(json.dumps(store.stats(), indent=2))
        return 0
    server = ThreadingHTTPServer((args.host, args.port), single_decision_handler(store, args.token))
    print(f"GSA blind adjudication reviewer_c: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
