#!/usr/bin/env python3
"""Launch the blind, image-centric semantic audit service."""
from __future__ import annotations

import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import quick_semantic_audit as core


class AuditStore(core.QuickAuditStore):
    """Alias retained for launcher and test compatibility."""


def _page_bytes() -> bytes:
    return core.HTML_PATH.read_bytes()


def make_handler(
    store: AuditStore,
    token: str | None,
) -> type[BaseHTTPRequestHandler]:
    html = _page_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "GSAQuickAudit/1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}")

        def _authorized(self) -> bool:
            if not token:
                return True
            supplied = self.headers.get("Authorization", "")
            return supplied.startswith("Bearer ") and secrets.compare_digest(
                supplied[7:],
                token,
            )

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
                    self._json(
                        HTTPStatus.OK,
                        store.case(int(path.rsplit("/", 1)[1])),
                    )
                elif path.startswith("/api/image/"):
                    image_path = store.image_path(int(path.rsplit("/", 1)[1]))
                    content_type = (
                        core.mimetypes.guess_type(image_path.name)[0]
                        or "application/octet-stream"
                    )
                    self._send(HTTPStatus.OK, image_path.read_bytes(), content_type)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (IndexError, ValueError):
                self._json(HTTPStatus.NOT_FOUND, {"error": "case not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._api_auth():
                return
            if urlparse(self.path).path != "/api/submit":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > core.MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                result = store.submit(
                    int(payload["index"]),
                    payload["judgments"],
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, result)

    return Handler


def main() -> int:
    args = core.parse_args()
    core.validate_bind_security(args.host, args.token)
    store = AuditStore(args.pack, args.annotator)
    if args.check:
        print(json.dumps(store.stats(), indent=2))
        return 0
    port = args.port or (10320 if args.annotator == "a" else 10321)
    server = ThreadingHTTPServer((args.host, port), make_handler(store, args.token))
    auth_mode = "Bearer token" if args.token else "loopback-only"
    print(
        f"GSA quick audit {store.annotator_name}: "
        f"http://{args.host}:{port}/ ({auth_mode})",
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
