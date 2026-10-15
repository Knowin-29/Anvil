#!/usr/bin/env python3
"""Prepare an offline, privacy-screened review bundle for the 500-image audit.

No network calls are made. The generated JPEGs are the exact bytes intended for
future upload, while the manifest retains the source path and content identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from trains.flywheel import prompts, prompts_ext

SCENES = ("kitchen", "exploration", "tabletop", "embodied_home")
EXCLUDED_PATH_PARTS = ("/flickr/",)
TARGET_PER_SCENE = 125
MAX_EDGE = 1280
THUMB_SIZE = (320, 220)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_bucket(path: str) -> str:
    lower = path.lower()
    for marker, name in (
        ("bridge_raw_work", "bridge"), ("changeit", "changeit"),
        ("robopoint-data", "robopoint"), ("refspatial", "refspatial"),
        ("dataflywheel/input/kitchen", "kitchen"),
        ("dataflywheel/input/navigation", "navigation"),
        ("dataflywheel/input/tabletop", "tabletop"),
        ("dataflywheel/input/household", "household"),
    ):
        if marker in lower:
            return name
    return "other"


def face_count(image: Image.Image, detectors) -> int:
    probe = image.copy()
    probe.thumbnail((720, 720), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.asarray(probe.convert("RGB")), cv2.COLOR_RGB2GRAY)
    total = 0
    for detector, profile in detectors:
        probes = (gray, cv2.flip(gray, 1)) if profile else (gray,)
        for candidate in probes:
            faces = detector.detectMultiScale(candidate, scaleFactor=1.08,
                                               minNeighbors=4, minSize=(24, 24))
            total += len(faces)
    return total


def prompt_markdown() -> str:
    placeholder_objects = [
        {"object_class": "<detected noun 1>", "bbox": ["x1", "y1", "x2", "y2"]},
        {"object_class": "<detected noun 2>", "bbox": ["x1", "y1", "x2", "y2"]},
    ]
    primary = placeholder_objects[0]
    spatial = {"depth_order": "<verified depth response>"}
    sections = [
        ("1. Detection (static, first request)", prompts_ext.DETECTION),
        ("2. Cross-check (instantiated after detection)",
         prompts.cross_check_set(placeholder_objects)),
        ("3. Mask (instantiated from verified primary object)",
         prompts_ext.mask_polygon(primary)),
        ("4. Depth (instantiated from verified object set)",
         prompts_ext.depth_order(placeholder_objects)),
        ("5. Affordance (instantiated from G + S context)",
         prompts_ext.affordance_region(primary, spatial)),
        ("6. DoF (instantiated from G + S context)",
         prompts_ext.dof_interface(primary, spatial)),
        ("7. Action plan (instantiated from G + S context)",
         prompts_ext.action_cot(primary, spatial)),
    ]
    lines = [
        "# Prompts and request scope for the 500-image audit", "",
        "No request has been sent while building this bundle. This P0 audit uses "
        "four multi-object scene families; the single-object-heavy ChangeIT pool is "
        "reserved for later object-state evaluation.", "",
        "## What one image triggers", "",
        "A first-pass successful cascade makes seven requests: detection, reverse "
        "cross-check, mask, depth, affordance, DoF, and action plan. Bbox and point "
        "records are derived locally from the verified detection. Invalid stages may "
        "retry once, so the worst case is fourteen requests for one image.", "",
        "The endpoint receives `model`, `max_tokens`, `temperature`, and one user "
        "message containing the base64 JPEG plus the prompt text. It does not receive "
        "the source path, scene tag, SHA, local manifest, prior training data, or API "
        "response cache. The Authorization header contains the configured bearer key.", "",
        "## Dynamic fields", "",
        "Prompts 2-7 cannot be fully instantiated before inference. Angle-bracket "
        "values below are replaced only with the endpoint's verified upstream object "
        "labels, boxes, and depth response for the same image.", "",
    ]
    for title, text in sections:
        lines.extend((f"## {title}", "", "```text", text, "```", ""))
    return "\n".join(lines)


def make_contact_sheet(rows: list[dict], output: Path, sheet_no: int) -> None:
    cols, rows_per = 5, 10
    cell_w, cell_h = 330, 258
    canvas = Image.new("RGB", (cols * cell_w, rows_per * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, row in enumerate(rows):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        with Image.open(output / row["thumbnail_path"]) as thumb:
            canvas.paste(thumb.convert("RGB"), (x + 5, y + 5))
        draw.text((x + 7, y + 229), f'{row["id"]}  {row["scene"]}', fill="black", font=font)
        draw.text((x + 7, y + 243), row["source_bucket"], fill="#555555", font=font)
    target = output / "contact_sheets" / f"sheet_{sheet_no:02d}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, "JPEG", quality=88, optimize=True)


def write_html(rows: list[dict], output: Path, prompts_md: str, summary: dict) -> None:
    cards = []
    for row in rows:
        cards.append(
            '<article class="card" data-scene="{scene}">'
            '<img loading="lazy" src="{thumb}" alt="{id}">'
            '<div><b>{id}</b> <span>{scene}</span></div>'
            '<div>{bucket} | {width}x{height}</div>'
            '<details><summary>Local provenance</summary><code>{source}</code><br>'
            '<code>source sha256: {source_sha}</code><br>'
            '<code>upload sha256: {upload_sha}</code></details></article>'.format(
                scene=html.escape(row["scene"]), thumb=html.escape(row["thumbnail_path"]),
                id=html.escape(row["id"]), bucket=html.escape(row["source_bucket"]),
                width=row["original_width"], height=row["original_height"],
                source=html.escape(row["source_path"]), source_sha=row["source_sha256"],
                upload_sha=row["upload_sha256"])
        )
    buttons = "".join(
        f'<button onclick="filterScene(\'{s}\')">{s}</button>' for s in ("all", *SCENES))
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>GSA 500-image upload review</title><style>
body{{font:14px Arial,sans-serif;margin:22px;color:#202124;background:#fff}}
h1{{font-size:25px}} .notice{{border-left:4px solid #d55e00;padding:10px 14px;background:#fbefea}}
.toolbar{{position:sticky;top:0;background:#fff;padding:10px 0;z-index:2}}
button{{margin-right:7px;padding:7px 10px;border:1px solid #888;background:#fff;cursor:pointer}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(225px,1fr));gap:10px}}
.card{{border:1px solid #c7cbd1;padding:7px}} .card img{{width:100%;aspect-ratio:16/11;object-fit:contain;background:#f5f6f7}}
.card span{{color:#0072b2}} code{{font-size:11px;overflow-wrap:anywhere}} pre{{white-space:pre-wrap}}
</style></head><body><h1>GSA 500-image upload review</h1>
<p class="notice"><b>No upload has occurred.</b> These sanitized JPEGs are the exact planned image bytes. The target endpoint is plaintext HTTP.</p>
<p>Selection: 125 images per multi-object scene; content deduplicated; AppleDouble, unreadable, Flickr, ChangeIT, and Haar-face-positive candidates excluded. Automated face screening is heuristic, so this gallery remains the manual approval surface. Manifest SHA-256: <code>{summary['manifest_sha256']}</code>.</p>
<div class="toolbar">{buttons}</div><div class="grid">{''.join(cards)}</div>
<h2>Prompt templates and transmitted fields</h2><pre>{html.escape(prompts_md)}</pre>
<script>function filterScene(s){{document.querySelectorAll('.card').forEach(x=>x.style.display=(s==='all'||x.dataset.scene===s)?'block':'none')}}</script>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="configs/data/embodied_images.index")
    parser.add_argument("--out", default="artifacts/upload_review_500_v2")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty review directory: {output}")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "thumbs").mkdir(parents=True, exist_ok=True)

    detectors = []
    for filename, profile in (
        ("haarcascade_frontalface_default.xml", False),
        ("haarcascade_frontalface_alt2.xml", False),
        ("haarcascade_profileface.xml", True),
    ):
        cascade_path = cv2.data.haarcascades + filename
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise RuntimeError(f"could not load face detector: {cascade_path}")
        detectors.append((detector, profile))

    selected: list[dict] = []
    counts = Counter()
    rejects = Counter()
    seen_sha = set()
    lines = Path(args.index).read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, 1):
        if not line.strip() or "\t" not in line:
            rejects["malformed_index"] += 1
            continue
        raw_path, scene = line.rsplit("\t", 1)
        scene = scene.strip()
        if scene not in SCENES or counts[scene] >= TARGET_PER_SCENE:
            continue
        path = Path(raw_path)
        lower = raw_path.lower()
        if path.name.startswith("._"):
            rejects["appledouble"] += 1
            continue
        if any(part in lower for part in EXCLUDED_PATH_PARTS):
            rejects["privacy_source_flickr"] += 1
            continue
        if not path.is_file():
            rejects["missing"] += 1
            continue
        try:
            source_data = path.read_bytes()
            source_sha = sha256_bytes(source_data)
            if source_sha in seen_sha:
                rejects["duplicate_content"] += 1
                continue
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            width, height = image.size
            if min(width, height) < 224:
                rejects["too_small"] += 1
                continue
            faces = face_count(image, detectors)
            if faces:
                rejects["face_detected"] += 1
                continue
        except Exception:
            rejects["unreadable"] += 1
            continue

        seen_sha.add(source_sha)
        counts[scene] += 1
        item_id = f"{scene[:3]}-{counts[scene]:03d}-{source_sha[:10]}"
        upload_rel = f"images/{item_id}.jpg"
        thumb_rel = f"thumbs/{item_id}.jpg"
        upload = image.copy()
        upload.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        upload.save(output / upload_rel, "JPEG", quality=88, optimize=True)
        thumb = ImageOps.contain(upload, THUMB_SIZE, Image.Resampling.LANCZOS)
        framed = Image.new("RGB", THUMB_SIZE, "#f5f6f7")
        framed.paste(thumb, ((THUMB_SIZE[0] - thumb.width) // 2,
                            (THUMB_SIZE[1] - thumb.height) // 2))
        framed.save(output / thumb_rel, "JPEG", quality=82, optimize=True)
        upload_data = (output / upload_rel).read_bytes()
        selected.append({
            "id": item_id, "scene": scene, "source_bucket": source_bucket(raw_path),
            "source_index_line": line_no, "source_path": raw_path,
            "source_sha256": source_sha, "source_bytes": len(source_data),
            "original_width": width, "original_height": height,
            "face_screen": "no_face_detected_three_haar_cascades_not_manual_clearance",
            "upload_path": upload_rel, "upload_sha256": sha256_bytes(upload_data),
            "upload_bytes": len(upload_data), "upload_width": upload.width,
            "upload_height": upload.height, "thumbnail_path": thumb_rel,
        })
        if all(counts[s] >= TARGET_PER_SCENE for s in SCENES):
            break

    if any(counts[s] != TARGET_PER_SCENE for s in SCENES):
        raise RuntimeError(f"could not fill scene quotas: {dict(counts)}")
    selected.sort(key=lambda row: (SCENES.index(row["scene"]), row["id"]))

    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in selected), encoding="utf-8")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    (output / "upload_images.index").write_text(
        "".join(f"{output / row['upload_path']}\t{row['scene']}\n" for row in selected),
        encoding="utf-8")

    prompts_md = prompt_markdown()
    (output / "prompts.md").write_text(prompts_md, encoding="utf-8")
    request_template = {
        "endpoint": "<explicitly approved chat/completions endpoint>",
        "headers": {"Content-Type": "application/json", "Authorization": "Bearer <REDACTED>"},
        "body": {"model": "Qwen3.5-397B-A17B", "max_tokens": "<stage-specific>",
                 "temperature": "<stage-specific>", "messages": [{"role": "user",
                 "content": [{"type": "image_url", "image_url": {
                     "url": "data:image/jpeg;base64,<EXACT REVIEW JPEG BYTES>"}},
                     {"type": "text", "text": "<one prompt from prompts.md>"}]}]},
    }
    (output / "request_template.json").write_text(
        json.dumps(request_template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for sheet in range(10):
        make_contact_sheet(selected[sheet * 50:(sheet + 1) * 50], output, sheet + 1)
    representative = []
    for scene in SCENES:
        group = [row for row in selected if row["scene"] == scene]
        representative.extend(group[i] for i in (0, 24, 49, 74, 99))
    make_contact_sheet(representative, output, 0)
    rep_path = output / "contact_sheets" / "sheet_00.jpg"
    rep_path.replace(output / "representative_25.jpg")

    manifest_sha = sha256_bytes(manifest.read_bytes())
    summary = {
        "schema_version": 2,
        "selection_policy": "index-order, 125 per multi-object scene; ChangeIT excluded",
        "selected": len(selected), "by_scene": dict(counts),
        "by_source_bucket": dict(Counter(row["source_bucket"] for row in selected)),
        "rejected_before_quotas_filled": dict(rejects),
        "source_bytes": sum(row["source_bytes"] for row in selected),
        "upload_bytes": sum(row["upload_bytes"] for row in selected),
        "manifest_sha256": manifest_sha,
        "network_calls_made": 0,
        "privacy_notes": [
            "Flickr and ChangeIT paths excluded",
            "three OpenCV Haar cascades (including mirrored profile) screen faces",
            "face screening is heuristic and requires gallery review",
            "JPEG derivatives remove EXIF", "visible text is not OCR-redacted",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html(selected, output, prompts_md, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
