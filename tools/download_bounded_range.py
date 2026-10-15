#!/usr/bin/env python3
"""Download one immutable HTTP object with bounded parallel range requests.

The downloader fails closed if the server's object length changes, a range response
does not describe the requested bytes, or an optional SHA-256 does not match. Chunks
are retained for resumability and the destination appears only after atomic assembly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class DownloadError(RuntimeError):
    """Raised when the remote object violates the frozen download contract."""


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    timeout: float,
):
    request = Request(
        url,
        method=method,
        headers={"User-Agent": "FirstFault-source-freezer/1.0", **(headers or {})},
    )
    return urlopen(request, timeout=timeout)  # noqa: S310 - caller supplies URL


def inspect_remote(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        with _request(url, method="HEAD", timeout=timeout) as response:
            headers = response.headers
            length_raw = headers.get("Content-Length")
            if length_raw is None:
                raise DownloadError("HEAD response has no Content-Length")
            length = int(length_raw)
            return {
                "url": response.geturl(),
                "content_length": length,
                "accept_ranges": (headers.get("Accept-Ranges") or "").lower(),
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
                "version_id": headers.get("x-amz-version-id"),
                "content_type": headers.get("Content-Type"),
            }
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise DownloadError(f"HEAD failed for {url}: {exc}") from exc


def _chunk_ranges(total: int, chunk_bytes: int) -> list[tuple[int, int, int]]:
    if total <= 0 or chunk_bytes <= 0:
        raise ValueError("total and chunk_bytes must be positive")
    return [
        (index, start, min(total - 1, start + chunk_bytes - 1))
        for index, start in enumerate(range(0, total, chunk_bytes))
    ]


def _download_chunk(
    *,
    url: str,
    chunk_dir: Path,
    index: int,
    start: int,
    end: int,
    total: int,
    timeout: float,
    retries: int,
) -> Path:
    expected = end - start + 1
    final = chunk_dir / f"chunk-{index:05d}-{start}-{end}.bin"
    if final.is_file() and final.stat().st_size == expected:
        return final
    partial = final.with_suffix(".partial")
    partial.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with _request(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=timeout,
            ) as response:
                if response.status != 206:
                    raise DownloadError(
                        f"range {start}-{end} returned HTTP {response.status}, expected 206"
                    )
                match = CONTENT_RANGE_RE.fullmatch(
                    (response.headers.get("Content-Range") or "").strip()
                )
                if not match:
                    raise DownloadError(f"invalid Content-Range for {start}-{end}")
                observed = tuple(int(value) for value in match.groups())
                if observed != (start, end, total):
                    raise DownloadError(
                        f"Content-Range {observed} != {(start, end, total)}"
                    )
                remaining = expected
                with partial.open("wb") as handle:
                    while remaining:
                        block = response.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise DownloadError(
                                f"early EOF for range {start}-{end}; {remaining} bytes missing"
                            )
                        handle.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise DownloadError(f"range {start}-{end} returned excess bytes")
            if partial.stat().st_size != expected:
                raise DownloadError(
                    f"chunk size {partial.stat().st_size} != expected {expected}"
                )
            os.replace(partial, final)
            return final
        except (DownloadError, HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise DownloadError(f"range {start}-{end} failed after {retries + 1} attempts: {last_error}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    *,
    url: str,
    output: Path,
    expected_bytes: int,
    expected_sha256: str | None,
    workers: int,
    chunk_bytes: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    remote = inspect_remote(url, timeout=timeout)
    if remote["content_length"] != expected_bytes:
        raise DownloadError(
            f"remote Content-Length {remote['content_length']} != expected {expected_bytes}"
        )
    if remote["accept_ranges"] != "bytes":
        raise DownloadError(f"remote does not advertise byte ranges: {remote['accept_ranges']!r}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        actual_sha = _sha256(output)
        if output.stat().st_size == expected_bytes and (
            expected_sha256 is None or actual_sha == expected_sha256
        ):
            return {**remote, "path": str(output), "sha256": actual_sha, "reused": True}
        raise DownloadError(f"existing destination does not satisfy contract: {output}")

    chunk_dir = output.with_name(output.name + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ranges = _chunk_ranges(expected_bytes, chunk_bytes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_chunk,
                url=url,
                chunk_dir=chunk_dir,
                index=index,
                start=start,
                end=end,
                total=expected_bytes,
                timeout=timeout,
                retries=retries,
            ): (index, start, end)
            for index, start, end in ranges
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"downloaded {completed}/{len(ranges)} ranges", file=sys.stderr, flush=True)

    assembling = output.with_name(output.name + ".assembling")
    digest = hashlib.sha256()
    with assembling.open("wb") as destination:
        for index, start, end in ranges:
            chunk = chunk_dir / f"chunk-{index:05d}-{start}-{end}.bin"
            with chunk.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
                    digest.update(block)
    actual_sha = digest.hexdigest()
    if assembling.stat().st_size != expected_bytes:
        raise DownloadError("assembled byte count does not match frozen object length")
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise DownloadError(f"sha256 {actual_sha} != expected {expected_sha256}")
    os.replace(assembling, output)
    shutil.rmtree(chunk_dir)
    return {**remote, "path": str(output), "sha256": actual_sha, "reused": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_mib < 1 or args.retries < 0:
        parser.error("workers/chunk-mib must be positive and retries must be nonnegative")
    report = download(
        url=args.url,
        output=args.output,
        expected_bytes=args.expected_bytes,
        expected_sha256=args.expected_sha256,
        workers=args.workers,
        chunk_bytes=args.chunk_mib * 1024 * 1024,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
