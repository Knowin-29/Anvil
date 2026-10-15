#!/usr/bin/env python3
"""Serve the blind reviewer-C disagreement queue."""
from __future__ import annotations

import argparse
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

import quick_semantic_audit as core
from quick_semantic_audit_server import make_handler
from semantic_adjudication import AdjudicationStore


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "data/semantic_adjudication_v3")
    parser.add_argument("--host", default=os.environ.get("GSA_QUICK_AUDIT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=10322)
    parser.add_argument("--token", default=os.environ.get("GSA_QUICK_AUDIT_TOKEN"))
    parser.add_argument("--check", action="store_true", help="validate the pack and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    core.validate_bind_security(args.host, args.token)
    store = AdjudicationStore(args.pack)
    if args.check:
        print(json.dumps(store.stats(), indent=2))
        return 0
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.token))
    auth_mode = "Bearer token" if args.token else "loopback-only"
    print(
        f"GSA blind adjudication reviewer_c: http://{args.host}:{args.port}/ ({auth_mode})",
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
