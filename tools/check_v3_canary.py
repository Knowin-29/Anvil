#!/usr/bin/env python3
"""Fail-closed provenance/coverage gate for an audit-candidate canary."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

FORMS = (
    "action_cot", "affordance", "bbox", "depth",
    "detection", "dof", "mask", "point",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/manifests/v3_generation_manifest.json"))
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--prompt-version", default="gsa-flywheel-v3.4")
    parser.add_argument("--label-version", default="gsa-label-v3.4")
    parser.add_argument("--min-common", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = manifest.get("selected", [])
    selected = selected[args.index_offset:args.index_offset + args.limit]
    allowed = {item["image_path"]: item.get("image_sha256") for item in selected}
    by_image: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    audit_only = Counter()

    for form in FORMS:
        path = args.data / "raw" / f"{form}.jsonl"
        if not path.is_file():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON ({exc.msg})")
                continue
            image = row.get("image_path")
            if image not in allowed:
                errors.append(f"{path}:{line_no}: image outside approved canary")
            if row.get("form") != form:
                errors.append(f"{path}:{line_no}: form/file mismatch")
            if row.get("prompt_version") != args.prompt_version:
                errors.append(f"{path}:{line_no}: prompt version drift")
            if row.get("label_version") != args.label_version:
                errors.append(f"{path}:{line_no}: label version drift")
            if row.get("image_sha256") != allowed.get(image):
                errors.append(f"{path}:{line_no}: image hash drift")
            if not isinstance(row.get("payload"), list) or not row["payload"]:
                errors.append(f"{path}:{line_no}: empty payload")
            verification = row.get("_verification", {})
            if verification.get("verdict") != "verified" and not row.get("_audit_only"):
                errors.append(f"{path}:{line_no}: rejected record lacks audit_only flag")
            if row.get("_audit_only"):
                if not isinstance(row.get("_audit_reason"), str) or not row["_audit_reason"].strip():
                    errors.append(f"{path}:{line_no}: audit_only lacks reason")
                audit_only[form] += 1
            by_image[image].add(form)
            counts[form] += 1

    request_count = 0
    request_images: set[str] = set()
    if not args.request_log.is_file():
        errors.append(f"missing request log: {args.request_log}")
    else:
        for line_no, line in enumerate(
            args.request_log.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            request_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{args.request_log}:{line_no}: invalid JSON ({exc.msg})")
                continue
            if any(key in row for key in ("api_key", "authorization", "base64", "image_b64")):
                errors.append(f"{args.request_log}:{line_no}: secret/payload field logged")
            image_hash = row.get("image_sha256")
            if isinstance(image_hash, str):
                request_images.add(image_hash)
                if image_hash not in set(allowed.values()):
                    errors.append(f"{args.request_log}:{line_no}: image hash outside canary")

    common = sorted(image for image, forms in by_image.items()
                   if set(forms) == set(FORMS))
    report = {
        "data": str(args.data),
        "approved_images": len(allowed),
        "records_by_form": dict(counts),
        "audit_only_by_form": dict(audit_only),
        "images_with_all_forms": common,
        "common_image_count": len(common),
        "request_count": request_count,
        "request_image_count": len(request_images),
        "errors": errors,
        "verdict": "PASS" if not errors and len(common) >= args.min_common else "BLOCKED",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
