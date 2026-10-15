"""Assemble section HTML fragments into a complete tabbed report."""

from __future__ import annotations

import base64
import os
from collections.abc import Iterable
from datetime import datetime
from io import BytesIO
from typing import Any, List, Optional, Tuple

from PIL import Image

from .assets import build_html_skeleton
from .section_ontology import render_ontology_section
from .section_placement import render_placement_section
from .section_perception import render_perception_section
from .section_scene import render_scene_section
from .section_training import render_training_section


def _image_to_base64(image_path: str, max_width: int = 800) -> str:
    """Load an image, resize to max_width if needed, and return a data URI."""

    image = Image.open(image_path).convert("RGB")
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)), Image.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _coerce_rejected_entries(validation_report: Optional[Any]) -> List[Any]:
    if validation_report is None:
        return []
    if isinstance(validation_report, (list, tuple)):
        return list(validation_report)
    if isinstance(validation_report, dict) and isinstance(validation_report.get("rejected_entries"), Iterable):
        return list(validation_report["rejected_entries"])
    rejected_entries = getattr(validation_report, "rejected_entries", None)
    if isinstance(rejected_entries, Iterable):
        return list(rejected_entries)
    return []


def generate_report(
    image_path: str,
    image_size: Tuple[int, int],
    ontology_labels: List[Any],
    detections: List[Any],
    segmentations: List[Any],
    scene_graph: Any,
    plan_results: Optional[List[Any]] = None,
    training_samples: Optional[List[Any]] = None,
    validation_report: Optional[Any] = None,
    output_dir: str = "data/output/reports",
) -> str:
    """Generate a self-contained HTML report and return its file path."""

    os.makedirs(output_dir, exist_ok=True)

    image_id = getattr(scene_graph, "image_id", None) or os.path.splitext(os.path.basename(image_path))[0] or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{image_id}_{timestamp}.html")

    image_base64 = _image_to_base64(image_path) if image_path and os.path.exists(image_path) else ""
    generated_at = datetime.now().isoformat(timespec="seconds")

    overview = f"""<section id="overview" class="active">
  <div class="card">
    <h3>Pipeline Report - {image_id}</h3>
    <p>Image: {image_path} ({image_size[0]}x{image_size[1]})</p>
    <p>Generated: {generated_at}</p>
    <div class="img-container"><img src="{image_base64}" alt="source image"></div>
  </div>
</section>"""

    m1 = f'<section id="m1">{render_ontology_section(ontology_labels, scene_graph=scene_graph)}</section>'
    m2 = f'<section id="m2">{render_perception_section(image_base64, image_size, detections, segmentations)}</section>'
    m3 = (
        f'<section id="m3">'
        f"{render_scene_section(scene_graph, image_base64=image_base64)}"
        f"{render_placement_section(scene_graph, training_samples or [], image_base64=image_base64)}"
        "</section>"
    )
    m4 = (
        f'<section id="m4">{render_training_section(plan_results or [], training_samples or [], _coerce_rejected_entries(validation_report))}</section>'
    )

    nav = """<nav>
  <button data-tab="overview" onclick="switchTab('overview', this)" class="active">Overview</button>
  <button data-tab="m1" onclick="switchTab('m1', this)">M1 Ontology</button>
  <button data-tab="m2" onclick="switchTab('m2', this)">M2 Perception</button>
  <button data-tab="m3" onclick="switchTab('m3', this)">M3 Scene</button>
  <button data-tab="m4" onclick="switchTab('m4', this)">M4 Training</button>
</nav>"""

    html = build_html_skeleton(
        title=f"Pipeline Report: {image_id}",
        body_sections=nav + overview + m1 + m2 + m3 + m4,
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)

    return output_path
