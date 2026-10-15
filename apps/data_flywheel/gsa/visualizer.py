"""GSA Visualizer — Generate HTML reports with image + annotation overlay.

Renders bbox, points, regions directly on images using PIL,
then embeds them as base64 in the HTML report.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("gsa.visualizer")

# Colors per bucket
BUCKET_COLORS = {
    "G": (0, 200, 0),       # green
    "S": (0, 100, 255),     # blue
    "A_state": (255, 165, 0),  # orange (legacy)
    "A_action": (255, 0, 0),   # red (legacy)
    "A_identity_attr": (255, 165, 0),    # orange
    "A_obj_state": (255, 200, 50),       # gold
    "A_robot_state": (180, 0, 180),      # purple
    "A_action_precondition": (255, 80, 80),  # light red
    "A_action_interface": (255, 0, 0),   # red
}

SUBTYPE_COLORS = {
    "simple_grounding": (0, 200, 0),
    "descriptive_grounding": (0, 255, 100),
    "relational_grounding": (100, 255, 0),
    "counting": (200, 255, 0),
    "2d_relation": (0, 100, 255),
    "topology_relation": (0, 150, 255),
    "support_relation": (50, 50, 255),
    "occlusion_relation": (100, 0, 255),
    "containment_relation": (150, 0, 200),
    "stacking_relation": (0, 0, 200),
    "object_recognition": (255, 165, 0),
    "state_recognition": (255, 200, 0),
    "grasp_point_2d": (255, 0, 0),
    "push_contact_point_2d": (255, 50, 50),
    "place_region_2d": (200, 0, 50),
    # A_identity_attr subtypes
    "object_class": (255, 165, 0),
    "color": (255, 140, 0),
    "material": (255, 120, 0),
    "shape": (255, 100, 0),
    "size": (255, 80, 0),
    "multi_attr": (255, 60, 0),
    # A_obj_state subtypes
    "pose": (255, 200, 50),
    "open_close": (255, 210, 70),
    "occupancy": (255, 220, 90),
    "on_off": (255, 230, 110),
    # A_robot_state subtypes
    "gripper_empty": (180, 0, 180),
    "holding_object": (200, 0, 200),
    "object_between_fingers": (160, 0, 160),
    "contact_state": (140, 0, 140),
    "tool_ready": (120, 0, 120),
    # A_action_precondition subtypes
    "can_pick": (255, 80, 80),
    "can_push": (255, 100, 100),
    "can_place": (255, 120, 120),
    "blocked": (200, 0, 0),
    "need_pre_action": (180, 0, 0),
    "unreachable": (160, 0, 0),
    # A_action_interface subtypes
    "grasp_point": (255, 0, 0),
    "jaw_axis": (255, 30, 30),
    "push_contact": (255, 50, 50),
    "push_direction": (255, 70, 70),
    "place_region": (200, 0, 50),
    "reorient": (180, 0, 80),
}


def draw_annotations_on_image(
    image_path: str,
    labels: List[Dict[str, Any]],
    max_size: int = 800,
) -> Optional[str]:
    """Draw annotations on image and return base64 encoded PNG."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Cannot open image {image_path}: {e}")
        return None

    # Resize for display
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        scale = ratio
    else:
        scale = 1.0

    draw = ImageDraw.Draw(img)

    for label in labels:
        bucket = label.get("bucket", "")
        subtype = label.get("subtype", "")
        color = SUBTYPE_COLORS.get(subtype, BUCKET_COLORS.get(bucket, (128, 128, 128)))

        # Draw bbox
        answer = label.get("answer", {})
        bbox = None
        if isinstance(answer, dict):
            bbox = answer.get("bbox")
        if not bbox:
            bbox = label.get("bbox")

        if bbox and len(bbox) == 4:
            scaled_bbox = [int(b * scale) for b in bbox]
            draw.rectangle(scaled_bbox, outline=color, width=2)
            # Label text
            text = f"{subtype[:12]}"
            obj_name = ""
            if isinstance(answer, dict):
                obj_name = answer.get("object_name", "")
            if obj_name:
                text = f"{obj_name}"
            draw.text((scaled_bbox[0], max(0, scaled_bbox[1] - 14)), text,
                      fill=color)

        # Draw grasp point
        grasp_pt = label.get("grasp_point_2d")
        if grasp_pt and len(grasp_pt) == 2:
            x, y = int(grasp_pt[0] * scale), int(grasp_pt[1] * scale)
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 255))
            draw.text((x + 8, y - 6), "grasp", fill=(255, 0, 0))

        # Draw affordance point (star marker)
        aff_pt = label.get("affordance_point")
        if aff_pt and len(aff_pt) == 2:
            # Affordance points are in 0-1000 normalized coords
            img_w, img_h = img.size
            ax = int(aff_pt[0] / 1000 * img_w)
            ay = int(aff_pt[1] / 1000 * img_h)
            r = 8
            # Draw star shape
            for angle_offset in range(0, 360, 72):
                import math
                rad = math.radians(angle_offset)
                x1 = ax + int(r * math.cos(rad))
                y1 = ay + int(r * math.sin(rad))
                draw.line([(ax, ay), (x1, y1)], fill=(0, 255, 255), width=2)
            draw.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=(0, 255, 255))
            hint = label.get("grasp_hint", "")
            draw.text((ax + 10, ay - 6), f"aff:{hint}", fill=(0, 255, 255))

        # Draw grasp rectangle (4 corners)
        grasp_corners = label.get("grasp_rect_corners")
        if grasp_corners and len(grasp_corners) == 4:
            img_w, img_h = img.size
            scaled_corners = [
                (int(c[0] / 1000 * img_w), int(c[1] / 1000 * img_h))
                for c in grasp_corners
            ]
            for i in range(4):
                draw.line([scaled_corners[i], scaled_corners[(i + 1) % 4]],
                          fill=(255, 0, 255), width=2)

        # Draw approach direction arrow
        approach = label.get("preferred_approach", "")
        if approach and approach != "unknown" and aff_pt and len(aff_pt) == 2:
            img_w, img_h = img.size
            ax = int(aff_pt[0] / 1000 * img_w)
            ay = int(aff_pt[1] / 1000 * img_h)
            arrow_len = 30
            if approach == "top_down":
                draw.line([(ax, ay - arrow_len), (ax, ay)], fill=(255, 255, 0), width=2)
                draw.polygon([(ax, ay), (ax - 4, ay - 8), (ax + 4, ay - 8)], fill=(255, 255, 0))
            elif approach == "horizontal":
                draw.line([(ax - arrow_len, ay), (ax, ay)], fill=(255, 255, 0), width=2)
                draw.polygon([(ax, ay), (ax - 8, ay - 4), (ax - 8, ay + 4)], fill=(255, 255, 0))
            elif approach == "diagonal_down":
                dx = int(arrow_len * 0.7)
                draw.line([(ax - dx, ay - dx), (ax, ay)], fill=(255, 255, 0), width=2)

        # Draw push contact point
        push_pt = label.get("push_contact_point_2d")
        if push_pt and len(push_pt) == 2:
            x, y = int(push_pt[0] * scale), int(push_pt[1] * scale)
            r = 5
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 100, 0), outline=(255, 255, 255))
            # Draw direction arrow
            push_dir = label.get("push_direction_2d")
            if push_dir and len(push_dir) == 2:
                dx, dy = push_dir[0] * 30, push_dir[1] * 30
                draw.line([(x, y), (x + int(dx), y + int(dy))], fill=(255, 100, 0), width=2)

        # Draw place region
        place_region = label.get("place_region_2d")
        if place_region and isinstance(place_region, dict):
            points = place_region.get("points", [])
            if len(points) >= 3:
                scaled_pts = [(int(p[0] * scale), int(p[1] * scale)) for p in points]
                draw.polygon(scaled_pts, outline=(200, 0, 50), fill=None)

        # Draw spatial relation lines
        if bucket == "S" and label.get("subject_id") and label.get("object_id"):
            # We'd need object positions for this, skip for now
            pass

    # Encode to base64
    buf = io.BytesIO()
    img.save(buf, format="PNG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def generate_visual_report(
    labels: List[Dict[str, Any]],
    image_dir: str = "",
    output_path: str = "",
    run_id: str = "",
) -> str:
    """Generate a comprehensive HTML report with image + annotation visualization."""
    today = datetime.now().strftime("%Y-%m-%d")
    today_short = datetime.now().strftime("%Y%m%d")

    # Group labels by sample_id (image)
    by_sample: Dict[str, List[Dict]] = {}
    for label in labels:
        sid = label.get("sample_id", "unknown")
        by_sample.setdefault(sid, []).append(label)

    # Collect image paths from labels
    sample_images: Dict[str, str] = {}
    for label in labels:
        sid = label.get("sample_id", "")
        img = label.get("image_path", "")
        if img and sid:
            sample_images[sid] = img

    # Stats
    from collections import Counter
    bucket_counts = Counter(l.get("bucket") for l in labels)
    subtype_counts = Counter(l.get("subtype") for l in labels)

    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GSA Visual Report — {today}</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:0;background:#0d1117;color:#c9d1d9}}
.header{{background:linear-gradient(135deg,#161b22,#0d1117);padding:2em 3em;border-bottom:1px solid #30363d}}
h1{{color:#58a6ff;margin:0}}
.subtitle{{color:#8b949e;margin-top:0.5em}}
.container{{max-width:1400px;margin:0 auto;padding:2em}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1em;margin:1.5em 0}}
.stat{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1.2em;text-align:center}}
.stat .val{{font-size:2em;font-weight:700;color:#58a6ff}}
.stat .lbl{{color:#8b949e;font-size:0.85em;margin-top:0.3em}}
.sample-card{{background:#161b22;border:1px solid #30363d;border-radius:12px;margin:1.5em 0;overflow:hidden}}
.sample-header{{padding:1em 1.5em;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center}}
.sample-header h3{{margin:0;color:#f0f6fc}}
.sample-body{{display:grid;grid-template-columns:1fr 1fr;gap:0}}
.sample-img{{padding:1em;display:flex;align-items:center;justify-content:center;background:#0d1117}}
.sample-img img{{max-width:100%;border-radius:4px}}
.sample-labels{{padding:1em;max-height:500px;overflow-y:auto}}
.label-item{{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:0.8em;margin:0.5em 0;font-size:0.82em}}
.label-item pre{{margin:0;white-space:pre-wrap;color:#c9d1d9;font-family:'JetBrains Mono',monospace}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.75em;font-weight:600;margin-right:4px}}
.badge-G{{background:#1f6f2b;color:#7ee787}}
.badge-S{{background:#0c2d6b;color:#79c0ff}}
.badge-A_state{{background:#5c3d0e;color:#f0c674}}
.badge-A_action{{background:#5c0e0e;color:#f08080}}
table{{width:100%;border-collapse:collapse;margin:1em 0}}
th{{background:#21262d;color:#f0f6fc;padding:10px 12px;text-align:left;font-size:0.85em}}
td{{padding:8px 12px;border-bottom:1px solid #21262d;font-size:0.85em}}
.progress{{background:#21262d;border-radius:4px;height:6px;overflow:hidden}}
.progress-bar{{height:100%;border-radius:4px}}
footer{{text-align:center;padding:2em;color:#484f58;font-size:0.8em;border-top:1px solid #21262d;margin-top:3em}}
</style></head><body>
<div class="header">
<h1>GSA Data Flywheel v0 — Visual Report</h1>
<div class="subtitle">Run: {run_id} | Date: {today} | Branch: anvil_dev</div>
</div>
<div class="container">

<div class="stats-grid">
<div class="stat"><div class="val">{len(by_sample)}</div><div class="lbl">Images</div></div>
<div class="stat"><div class="val">{len(labels)}</div><div class="lbl">Total Labels</div></div>
<div class="stat"><div class="val">{bucket_counts.get('G',0)}</div><div class="lbl">G (Grounding)</div></div>
<div class="stat"><div class="val">{bucket_counts.get('S',0)}</div><div class="lbl">S (Spatial)</div></div>
<div class="stat"><div class="val">{bucket_counts.get('A_state',0)}</div><div class="lbl">A_state</div></div>
<div class="stat"><div class="val">{bucket_counts.get('A_action',0)}</div><div class="lbl">A_action</div></div>
</div>

<h2 style="color:#f0f6fc">Subtype Distribution (Viewer Tags)</h2>
<table><tr><th>Subtype</th><th>Bucket</th><th>Viewer Tag</th><th>Count</th><th>Coverage</th></tr>"""]

    from apps.data_flywheel.gsa.taxonomy import get_viewer_tag, SUBTYPE_TO_VIEWER_TAG

    for st, count in sorted(subtype_counts.items(), key=lambda x: -x[1]):
        pct = count / len(labels) * 100 if labels else 0
        bucket = ""
        for l in labels:
            if l.get("subtype") == st:
                bucket = l.get("bucket", "")
                break
        vtag = get_viewer_tag(bucket, st)
        badge_cls = f"badge-{bucket}"
        color = "#7ee787" if bucket == "G" else "#79c0ff" if bucket == "S" else "#f0c674" if bucket == "A_state" else "#f08080"
        html_parts.append(f'<tr><td>{st}</td><td><span class="badge {badge_cls}">{bucket}</span></td>'
                          f'<td><code>{vtag}</code></td>'
                          f'<td>{count}</td><td><div class="progress"><div class="progress-bar" '
                          f'style="width:{pct}%;background:{color}"></div></div></td></tr>')

    html_parts.append("</table>")
    html_parts.append('<h2 style="color:#f0f6fc">Sample Cases with Annotations</h2>')

    # Render each sample with image + annotations
    for sample_id, sample_labels in list(by_sample.items())[:20]:
        img_path = sample_images.get(sample_id, "")
        # Try to find image from label data
        if not img_path:
            for l in sample_labels:
                if l.get("image_path"):
                    img_path = l["image_path"]
                    break

        img_b64 = None
        if img_path and os.path.exists(img_path):
            img_b64 = draw_annotations_on_image(img_path, sample_labels)

        bucket_summary = Counter(l.get("bucket") for l in sample_labels)
        summary_str = " | ".join(f"{b}:{c}" for b, c in bucket_summary.items())

        html_parts.append(f"""
<div class="sample-card">
<div class="sample-header">
<h3>{sample_id}</h3>
<span style="color:#8b949e">{len(sample_labels)} labels | {summary_str}</span>
</div>
<div class="sample-body">
<div class="sample-img">""")

        if img_b64:
            html_parts.append(f'<img src="{img_b64}" alt="{sample_id}">')
        else:
            html_parts.append(f'<div style="color:#484f58;padding:2em">Image: {os.path.basename(img_path) if img_path else "N/A"}</div>')

        html_parts.append('</div><div class="sample-labels">')

        for label in sample_labels[:10]:
            bucket = label.get("bucket", "")
            badge_cls = f"badge-{bucket}"
            subtype = label.get("subtype", "")
            # Compact display
            display = {k: v for k, v in label.items()
                       if k not in ("sample_id", "source_provenance", "verifier_status", "image_path")
                       and v is not None}
            label_json = json.dumps(display, indent=1, ensure_ascii=False)[:400]
            html_parts.append(f"""<div class="label-item">
<span class="badge {badge_cls}">{bucket}</span> <span style="color:#8b949e">{subtype}</span>
<pre>{label_json}</pre></div>""")

        html_parts.append("</div></div></div>")

    html_parts.append(f"""
<footer>Generated by GSA Data Flywheel v0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Branch: anvil_dev</footer>
</div></body></html>""")

    html = "\n".join(html_parts)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"Visual report saved to {output_path}")

    return html
