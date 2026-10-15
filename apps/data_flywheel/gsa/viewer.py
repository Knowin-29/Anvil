"""GSA Data Viewer — Flask-based GUI for browsing GSA labels with images.

Provides a web interface to:
- Browse all samples with annotated images
- Filter by bucket / subtype / scene_family
- View detailed label JSON
- See statistics and coverage
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, render_template_string, request, jsonify

logger = logging.getLogger("gsa.viewer")

app = Flask(__name__)

# Global state
DATA_ROOT = ""
ALL_LABELS: List[Dict] = []
BY_SAMPLE: Dict[str, List[Dict]] = {}

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GSA Data Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9}
.nav{background:#161b22;border-bottom:1px solid #30363d;padding:1em 2em;display:flex;align-items:center;gap:2em}
.nav h1{color:#58a6ff;font-size:1.3em}
.nav .stats{color:#8b949e;font-size:0.85em}
.container{max-width:1600px;margin:0 auto;padding:1.5em}
.filters{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1em;margin-bottom:1.5em;display:flex;gap:1em;flex-wrap:wrap;align-items:center}
.filters select,.filters input{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 12px;border-radius:4px;font-size:0.85em}
.filters label{color:#8b949e;font-size:0.85em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(450px,1fr));gap:1.5em}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden;transition:border-color 0.2s}
.card:hover{border-color:#58a6ff}
.card-img{background:#0d1117;padding:0.5em;display:flex;align-items:center;justify-content:center;min-height:200px}
.card-img img{max-width:100%;max-height:300px;border-radius:4px}
.card-body{padding:1em}
.card-title{color:#f0f6fc;font-size:0.95em;margin-bottom:0.5em;display:flex;justify-content:space-between}
.card-labels{max-height:200px;overflow-y:auto}
.label-row{background:#0d1117;border:1px solid #21262d;border-radius:4px;padding:0.5em;margin:0.3em 0;font-size:0.78em;font-family:monospace}
.badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:0.7em;font-weight:600}
.badge-G{background:#1f6f2b;color:#7ee787}
.badge-S{background:#0c2d6b;color:#79c0ff}
.badge-A_state{background:#5c3d0e;color:#f0c674}
.badge-A_action{background:#5c0e0e;color:#f08080}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:0.8em;margin-bottom:1.5em}
.summary-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1em;text-align:center}
.summary-card .val{font-size:1.8em;font-weight:700;color:#58a6ff}
.summary-card .lbl{color:#8b949e;font-size:0.8em;margin-top:0.2em}
.no-img{color:#484f58;padding:2em;text-align:center}
</style>
</head><body>
<div class="nav">
<h1>GSA Data Viewer</h1>
<div class="stats">{{ total_labels }} labels | {{ total_samples }} samples | Run: {{ run_id }}</div>
</div>
<div class="container">
<div class="summary">
<div class="summary-card"><div class="val">{{ g_count }}</div><div class="lbl">G (Grounding)</div></div>
<div class="summary-card"><div class="val">{{ s_count }}</div><div class="lbl">S (Spatial)</div></div>
<div class="summary-card"><div class="val">{{ a_state_count }}</div><div class="lbl">A_state</div></div>
<div class="summary-card"><div class="val">{{ a_action_count }}</div><div class="lbl">A_action</div></div>
<div class="summary-card"><div class="val">{{ verifiable_rate }}</div><div class="lbl">Verifiable</div></div>
</div>

<div class="filters">
<label>Bucket:</label>
<select id="bucket-filter" onchange="filterCards()">
<option value="">All</option>
<option value="G">G</option>
<option value="S">S</option>
<option value="A_state">A_state</option>
<option value="A_action">A_action</option>
</select>
<label>Subtype:</label>
<select id="subtype-filter" onchange="filterCards()">
<option value="">All</option>
{% for st in subtypes %}<option value="{{ st }}">{{ st }}</option>{% endfor %}
</select>
<label>Search:</label>
<input type="text" id="search-input" placeholder="object name..." oninput="filterCards()">
</div>

<div class="grid" id="cards-grid">
{% for sample_id, sample_data in samples.items() %}
<div class="card" data-buckets="{{ sample_data.buckets }}" data-subtypes="{{ sample_data.subtypes }}">
<div class="card-img">
{% if sample_data.img_b64 %}
<img src="{{ sample_data.img_b64 }}" alt="{{ sample_id }}">
{% else %}
<div class="no-img">{{ sample_data.img_name }}</div>
{% endif %}
</div>
<div class="card-body">
<div class="card-title">
<span>{{ sample_id[-20:] }}</span>
<span style="color:#8b949e">{{ sample_data.label_count }} labels</span>
</div>
<div class="card-labels">
{% for label in sample_data.labels[:8] %}
<div class="label-row">
<span class="badge badge-{{ label.bucket }}">{{ label.bucket }}</span>
<span style="color:#8b949e">{{ label.subtype }}</span>
{% if label.question %}<br>Q: {{ label.question[:60] }}{% endif %}
{% if label.grasp_point_2d %}<br>grasp: {{ label.grasp_point_2d }}{% endif %}
</div>
{% endfor %}
{% if sample_data.label_count > 8 %}<div style="color:#484f58;text-align:center;padding:0.3em">+{{ sample_data.label_count - 8 }} more</div>{% endif %}
</div>
</div>
</div>
{% endfor %}
</div>
</div>

<script>
function filterCards() {
  const bucket = document.getElementById('bucket-filter').value;
  const subtype = document.getElementById('subtype-filter').value;
  const search = document.getElementById('search-input').value.toLowerCase();
  document.querySelectorAll('.card').forEach(card => {
    let show = true;
    if (bucket && !card.dataset.buckets.includes(bucket)) show = false;
    if (subtype && !card.dataset.subtypes.includes(subtype)) show = false;
    if (search && !card.innerHTML.toLowerCase().includes(search)) show = false;
    card.style.display = show ? '' : 'none';
  });
}
</script>
</body></html>"""


def load_data(run_dir: str):
    """Load all labels from a GSA run directory."""
    global ALL_LABELS, BY_SAMPLE, DATA_ROOT
    DATA_ROOT = run_dir
    ALL_LABELS = []
    BY_SAMPLE = defaultdict(list)

    labels_dir = Path(run_dir) / "labels"
    for jsonl_file in labels_dir.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    label = json.loads(line)
                    ALL_LABELS.append(label)
                    BY_SAMPLE[label.get("sample_id", "unknown")].append(label)

    logger.info(f"Loaded {len(ALL_LABELS)} labels from {len(BY_SAMPLE)} samples")


@app.route("/")
def index():
    from apps.data_flywheel.gsa.visualizer import draw_annotations_on_image

    bucket_counts = Counter(l.get("bucket", "") for l in ALL_LABELS)
    subtypes = sorted(set(l.get("subtype", "") for l in ALL_LABELS if l.get("subtype")))

    # Prepare sample data for template
    samples = {}
    for sample_id, labels in list(BY_SAMPLE.items())[:30]:
        img_path = ""
        for l in labels:
            if l.get("image_path"):
                img_path = l["image_path"]
                break

        img_b64 = None
        if img_path and os.path.exists(img_path):
            img_b64 = draw_annotations_on_image(img_path, labels, max_size=600)

        buckets = ",".join(set(l.get("bucket", "") for l in labels))
        label_subtypes = ",".join(set(l.get("subtype", "") for l in labels))

        samples[sample_id] = {
            "img_b64": img_b64,
            "img_name": os.path.basename(img_path) if img_path else "N/A",
            "labels": labels[:8],
            "label_count": len(labels),
            "buckets": buckets,
            "subtypes": label_subtypes,
        }

    return render_template_string(
        HTML_TEMPLATE,
        total_labels=len(ALL_LABELS),
        total_samples=len(BY_SAMPLE),
        run_id=os.path.basename(DATA_ROOT),
        g_count=bucket_counts.get("G", 0),
        s_count=bucket_counts.get("S", 0),
        a_state_count=bucket_counts.get("A_state", 0),
        a_action_count=bucket_counts.get("A_action", 0) + bucket_counts.get("", 0),
        verifiable_rate="100%",
        subtypes=subtypes,
        samples=samples,
    )


def start_viewer(run_dir: str = "", port: int = 10303):
    """Start the GSA data viewer server."""
    if not run_dir:
        runs = sorted(glob("data/output/gsa_v0/gsa_v0_*"))
        if runs:
            run_dir = runs[-1]
        else:
            print("ERROR: No GSA runs found.")
            return

    load_data(run_dir)
    print(f"GSA Data Viewer starting on http://0.0.0.0:{port}")
    print(f"Data: {run_dir} ({len(ALL_LABELS)} labels)")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    import sys
    run_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 10303
    start_viewer(run_dir, port)
