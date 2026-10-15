from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

from tools.download_bounded_range import DownloadError, download


PAYLOAD = bytes(range(256)) * 4096


class RangeHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _headers(self, status: int, length: int, content_range: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", '"fixture"')
        if content_range:
            self.send_header("Content-Range", content_range)
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        self._headers(200, len(PAYLOAD))

    def do_GET(self) -> None:  # noqa: N802
        raw = self.headers.get("Range")
        if not raw or not raw.startswith("bytes="):
            self._headers(200, len(PAYLOAD))
            self.wfile.write(PAYLOAD)
            return
        start_raw, end_raw = raw.removeprefix("bytes=").split("-", 1)
        start, end = int(start_raw), int(end_raw)
        body = PAYLOAD[start : end + 1]
        self._headers(206, len(body), f"bytes {start}-{end}/{len(PAYLOAD)}")
        self.wfile.write(body)


@contextmanager
def range_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fixture.bin"
    finally:
        server.shutdown()
        thread.join()


def test_parallel_range_download_is_atomic_and_reusable(tmp_path: Path) -> None:
    destination = tmp_path / "payload.bin"
    with range_server() as url:
        first = download(
            url=url,
            output=destination,
            expected_bytes=len(PAYLOAD),
            expected_sha256=None,
            workers=4,
            chunk_bytes=65536,
            timeout=10,
            retries=0,
        )
        second = download(
            url=url,
            output=destination,
            expected_bytes=len(PAYLOAD),
            expected_sha256=first["sha256"],
            workers=4,
            chunk_bytes=65536,
            timeout=10,
            retries=0,
        )
    assert destination.read_bytes() == PAYLOAD
    assert first["reused"] is False
    assert second["reused"] is True
    assert not list(tmp_path.glob("*.assembling"))


def test_length_drift_fails_before_writing(tmp_path: Path) -> None:
    destination = tmp_path / "payload.bin"
    with range_server() as url:
        with pytest.raises(DownloadError, match="Content-Length"):
            download(
                url=url,
                output=destination,
                expected_bytes=len(PAYLOAD) + 1,
                expected_sha256=None,
                workers=2,
                chunk_bytes=65536,
                timeout=10,
                retries=0,
            )
    assert not destination.exists()
