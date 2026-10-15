"""Data Flywheel Viewer — Web 界面浏览标注数据 + 图片 + bbox 可视化。

启动:
    python -m apps.data_flywheel.viewer.app \
        --annotations-dir data/output/flywheel/v3_run_xxx/annotations \
        --port 7860 --host 0.0.0.0

功能:
- 按能力维度 (G/A/S/R/P/F/H/E/Aff) 筛选
- 按 sub_tag 筛选
- 图片 + bbox 可视化叠加
- 标注内容 JSON 展示
- 分页浏览
- 统计面板
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request, send_file
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("viewer")

app = Flask(__name__)

# Global state
ANNOTATIONS_DIR = ""
IMAGE_MANIFEST = {}  # sample_id -> image_path
ANNOTATIONS_CACHE: Dict[str, List[Dict]] = {}  # cap -> list of annotations
STATS_CACHE: Dict[str, Any] = {}

# Colors for different capabilities
CAP_COLORS = {
    "G": "#FF6B6B", "A": "#4ECDC4", "S": "#45B7D1",
    "R": "#96CEB4", "P": "#FFEAA7", "F": "#DDA0DD",
    "H": "#98D8C8", "E": "#F7DC6F", "Aff": "#BB8FCE",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rex-Embodied Data Flywheel Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; }
        .header { background: linear-gradient(135deg, #16213e, #0f3460); padding: 20px 30px; border-bottom: 2px solid #e94560; }
        .header h1 { font-size: 24px; color: #e94560; }
        .header p { color: #aaa; margin-top: 5px; }
        .container { display: flex; height: calc(100vh - 80px); }
        .sidebar { width: 280px; background: #16213e; padding: 20px; overflow-y: auto; border-right: 1px solid #333; }
        .main { flex: 1; padding: 20px; overflow-y: auto; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 20px; }
        .stat-card { background: #0f3460; border-radius: 8px; padding: 12px; text-align: center; }
        .stat-card .num { font-size: 20px; font-weight: bold; color: #e94560; }
        .stat-card .label { font-size: 11px; color: #aaa; margin-top: 4px; }
        .filter-section { margin-bottom: 20px; }
        .filter-section h3 { font-size: 13px; color: #aaa; margin-bottom: 8px; text-transform: uppercase; }
        .cap-btn { display: inline-block; padding: 6px 12px; margin: 3px; border-radius: 15px; cursor: pointer; font-size: 12px; font-weight: bold; border: 2px solid transparent; transition: all 0.2s; }
        .cap-btn:hover { transform: scale(1.05); }
        .cap-btn.active { border-color: #fff; box-shadow: 0 0 10px rgba(255,255,255,0.3); }
        .subtag-btn { display: inline-block; padding: 4px 10px; margin: 2px; border-radius: 10px; cursor: pointer; font-size: 11px; background: #333; color: #ccc; border: 1px solid #555; }
        .subtag-btn:hover { background: #444; }
        .subtag-btn.active { background: #e94560; color: #fff; border-color: #e94560; }
        .annotation-card { background: #16213e; border-radius: 12px; margin-bottom: 20px; overflow: hidden; border: 1px solid #333; }
        .annotation-card:hover { border-color: #e94560; }
        .card-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: #0f3460; }
        .card-header .cap-tag { padding: 4px 10px; border-radius: 10px; font-size: 11px; font-weight: bold; }
        .card-header .meta { font-size: 11px; color: #888; }
        .card-body { display: flex; }
        .card-image { width: 400px; min-height: 300px; position: relative; background: #000; display: flex; align-items: center; justify-content: center; }
        .card-image img { max-width: 100%; max-height: 400px; object-fit: contain; }
        .card-content { flex: 1; padding: 16px; overflow-x: auto; }
        .card-content pre { background: #0a0a1a; padding: 12px; border-radius: 8px; font-size: 12px; line-height: 1.5; overflow-x: auto; white-space: pre-wrap; color: #7fdbca; }
        .card-content .field { margin-bottom: 8px; }
        .card-content .field-key { color: #e94560; font-weight: bold; font-size: 12px; }
        .card-content .field-val { color: #ddd; font-size: 13px; margin-top: 2px; }
        .pagination { display: flex; justify-content: center; align-items: center; gap: 10px; padding: 20px; }
        .pagination button { padding: 8px 16px; border-radius: 6px; border: 1px solid #555; background: #16213e; color: #eee; cursor: pointer; }
        .pagination button:hover { background: #e94560; border-color: #e94560; }
        .pagination button:disabled { opacity: 0.3; cursor: not-allowed; }
        .pagination span { color: #aaa; }
        .no-data { text-align: center; padding: 60px; color: #666; font-size: 18px; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 8px; font-size: 10px; margin-left: 5px; }
        .loading { text-align: center; padding: 40px; color: #888; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Rex-Embodied Data Flywheel Viewer</h1>
        <p id="status-line">Loading...</p>
    </div>
    <div class="container">
        <div class="sidebar">
            <div class="stats-grid" id="stats-grid"></div>
            <div class="filter-section">
                <h3>Capability</h3>
                <div id="cap-filters"></div>
            </div>
            <div class="filter-section">
                <h3>Sub-tags</h3>
                <div id="subtag-filters"></div>
            </div>
            <div class="filter-section">
                <h3>Provenance</h3>
                <div id="prov-filters"></div>
            </div>
        </div>
        <div class="main" id="main-content">
            <div class="loading">Loading annotations...</div>
        </div>
    </div>

    <script>
    let allData = [];
    let filteredData = [];
    let currentPage = 0;
    const PAGE_SIZE = 10;
    let activeCap = null;
    let activeSubtag = null;
    let stats = {};

    const CAP_COLORS = {
        G: '#FF6B6B', A: '#4ECDC4', S: '#45B7D1',
        R: '#96CEB4', P: '#FFEAA7', F: '#DDA0DD',
        H: '#98D8C8', E: '#F7DC6F', Aff: '#BB8FCE'
    };

    async function init() {
        const resp = await fetch('/api/stats');
        stats = await resp.json();
        renderStats();
        renderCapFilters();
        loadAnnotations();
    }

    function renderStats() {
        const grid = document.getElementById('stats-grid');
        grid.innerHTML = `
            <div class="stat-card"><div class="num">${stats.total_annotations?.toLocaleString() || 0}</div><div class="label">Total Annotations</div></div>
            <div class="stat-card"><div class="num">${stats.total_samples?.toLocaleString() || 0}</div><div class="label">Unique Samples</div></div>
            <div class="stat-card"><div class="num">${stats.capabilities?.length || 0}</div><div class="label">Capabilities</div></div>
            <div class="stat-card"><div class="num">${stats.total_subtags || 0}</div><div class="label">Sub-tags</div></div>
        `;
        document.getElementById('status-line').textContent =
            `${stats.total_annotations?.toLocaleString() || 0} annotations across ${stats.capabilities?.length || 0} capabilities | ${stats.provenance_vlm || 0}% VLM-generated`;
    }

    function renderCapFilters() {
        const div = document.getElementById('cap-filters');
        div.innerHTML = '';
        (stats.capabilities || []).forEach(cap => {
            const btn = document.createElement('span');
            btn.className = 'cap-btn';
            btn.style.background = CAP_COLORS[cap] || '#666';
            btn.style.color = '#000';
            btn.textContent = `${cap} (${stats.per_cap?.[cap] || 0})`;
            btn.onclick = () => { activeCap = activeCap === cap ? null : cap; activeSubtag = null; updateFilters(); loadAnnotations(); };
            div.appendChild(btn);
        });
    }

    function updateFilters() {
        document.querySelectorAll('.cap-btn').forEach(btn => {
            const cap = btn.textContent.split(' ')[0];
            btn.classList.toggle('active', cap === activeCap);
        });
        // Update subtag filters
        const subDiv = document.getElementById('subtag-filters');
        subDiv.innerHTML = '';
        if (activeCap && stats.subtags_per_cap?.[activeCap]) {
            stats.subtags_per_cap[activeCap].forEach(st => {
                const btn = document.createElement('span');
                btn.className = 'subtag-btn' + (st === activeSubtag ? ' active' : '');
                btn.textContent = st;
                btn.onclick = () => { activeSubtag = activeSubtag === st ? null : st; loadAnnotations(); };
                subDiv.appendChild(btn);
            });
        }
    }

    async function loadAnnotations() {
        const main = document.getElementById('main-content');
        main.innerHTML = '<div class="loading">Loading...</div>';
        let url = `/api/annotations?page=${currentPage}&page_size=${PAGE_SIZE}`;
        if (activeCap) url += `&cap=${activeCap}`;
        if (activeSubtag) url += `&subtag=${activeSubtag}`;
        const resp = await fetch(url);
        const data = await resp.json();
        renderAnnotations(data);
    }

    function renderAnnotations(data) {
        const main = document.getElementById('main-content');
        if (!data.items || data.items.length === 0) {
            main.innerHTML = '<div class="no-data">No annotations found for current filters</div>';
            return;
        }
        let html = '';
        data.items.forEach(ann => {
            const cap = ann.capability_tag || '';
            const color = CAP_COLORS[cap] || '#666';
            const content = ann.content || {};
            const contentStr = JSON.stringify(content, null, 2);
            const imgUrl = ann.image_url ? `/api/image?path=${encodeURIComponent(ann.image_url)}&bbox=${encodeURIComponent(JSON.stringify(ann.bbox || []))}` : '';

            html += `<div class="annotation-card">
                <div class="card-header">
                    <div>
                        <span class="cap-tag" style="background:${color};color:#000">${cap}</span>
                        <span class="badge" style="background:#333">${ann.sub_tag || ''}</span>
                        <span class="badge" style="background:#1a5276;color:#85c1e9">diff:${ann.difficulty || '?'}</span>
                        <span class="badge" style="background:#145a32;color:#82e0aa">conf:${(ann.confidence || 0).toFixed(2)}</span>
                    </div>
                    <div class="meta">${ann.sample_id || ''} | ${ann.provenance || ''}</div>
                </div>
                <div class="card-body">
                    ${imgUrl ? `<div class="card-image"><img src="${imgUrl}" loading="lazy"/></div>` : ''}
                    <div class="card-content"><pre>${escapeHtml(contentStr)}</pre></div>
                </div>
            </div>`;
        });

        // Pagination
        const totalPages = Math.ceil(data.total / PAGE_SIZE);
        html += `<div class="pagination">
            <button onclick="prevPage()" ${currentPage === 0 ? 'disabled' : ''}>Prev</button>
            <span>Page ${currentPage + 1} / ${totalPages} (${data.total} total)</span>
            <button onclick="nextPage()" ${currentPage >= totalPages - 1 ? 'disabled' : ''}>Next</button>
        </div>`;
        main.innerHTML = html;
    }

    function prevPage() { if (currentPage > 0) { currentPage--; loadAnnotations(); } }
    function nextPage() { currentPage++; loadAnnotations(); }
    function escapeHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

    init();
    </script>
</body>
</html>"""


def load_manifest(annotations_dir: str):
    """加载 manifest 获取 sample_id -> image_path 映射。"""
    global IMAGE_MANIFEST
    parent = os.path.dirname(annotations_dir)
    manifest_path = os.path.join(parent, "manifest.jsonl")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    IMAGE_MANIFEST[obj.get("sample_id", "")] = obj.get("image_path", "")
                except:
                    pass
    logger.info(f"Loaded manifest: {len(IMAGE_MANIFEST)} entries")


def load_annotations(annotations_dir: str):
    """加载所有标注到内存。"""
    global ANNOTATIONS_CACHE, STATS_CACHE
    ANNOTATIONS_CACHE = {}
    total = 0
    samples = set()
    subtags_all = set()
    per_cap = {}
    subtags_per_cap = defaultdict(set)
    provenance_counter = Counter()

    for cap_dir in sorted(os.listdir(annotations_dir)):
        cap_path = os.path.join(annotations_dir, cap_dir)
        if not os.path.isdir(cap_path):
            continue
        jsonl_file = os.path.join(cap_path, f"{cap_dir}.jsonl")
        if not os.path.exists(jsonl_file):
            continue

        anns = []
        with open(jsonl_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ann = json.loads(line)
                    anns.append(ann)
                    samples.add(ann.get("sample_id", ""))
                    st = ann.get("sub_tag", "")
                    subtags_all.add(st)
                    subtags_per_cap[cap_dir].add(st)
                    provenance_counter[ann.get("provenance", "unknown")] += 1
                except:
                    pass
        ANNOTATIONS_CACHE[cap_dir] = anns
        per_cap[cap_dir] = len(anns)
        total += len(anns)

    vlm_pct = round(provenance_counter.get("vlm", 0) / max(total, 1) * 100, 1)
    STATS_CACHE = {
        "total_annotations": total,
        "total_samples": len(samples),
        "capabilities": sorted(ANNOTATIONS_CACHE.keys()),
        "total_subtags": len(subtags_all),
        "per_cap": per_cap,
        "subtags_per_cap": {k: sorted(v) for k, v in subtags_per_cap.items()},
        "provenance_vlm": vlm_pct,
    }
    logger.info(f"Loaded {total} annotations across {len(ANNOTATIONS_CACHE)} capabilities")


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/stats")
def api_stats():
    return jsonify(STATS_CACHE)


@app.route("/api/annotations")
def api_annotations():
    page = int(request.args.get("page", 0))
    page_size = int(request.args.get("page_size", 10))
    cap = request.args.get("cap", None)
    subtag = request.args.get("subtag", None)

    # Collect matching annotations
    items = []
    if cap and cap in ANNOTATIONS_CACHE:
        items = ANNOTATIONS_CACHE[cap]
    else:
        for anns in ANNOTATIONS_CACHE.values():
            items.extend(anns)

    if subtag:
        items = [a for a in items if a.get("sub_tag") == subtag]

    total = len(items)
    start = page * page_size
    end = start + page_size
    page_items = items[start:end]

    # Enrich with image_url
    result = []
    for ann in page_items:
        sample_id = ann.get("sample_id", "")
        content = ann.get("content", {})
        image_path = content.get("image_path") or IMAGE_MANIFEST.get(sample_id, "")
        bbox = content.get("bbox") or (content.get("boxes", [[]])[0] if content.get("boxes") else [])

        result.append({
            "sample_id": sample_id,
            "capability_tag": ann.get("capability_tag", ""),
            "sub_tag": ann.get("sub_tag", ""),
            "content": content,
            "confidence": ann.get("confidence", 0),
            "difficulty": ann.get("difficulty", 0),
            "provenance": ann.get("provenance", ""),
            "image_url": image_path,
            "bbox": bbox,
        })

    return jsonify({"items": result, "total": total, "page": page, "page_size": page_size})


@app.route("/api/image")
def api_image():
    """返回带 bbox 可视化的图片。"""
    image_path = request.args.get("path", "")
    bbox_str = request.args.get("bbox", "[]")

    if not image_path or not os.path.exists(image_path):
        return "Image not found", 404

    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        bbox = json.loads(bbox_str) if bbox_str else []
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=3)

        # Also draw contact_point if present
        cp_str = request.args.get("contact_point", "")
        if cp_str:
            cp = json.loads(cp_str)
            if cp and len(cp) >= 2:
                cx, cy = int(cp[0]), int(cp[1])
                draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill="#FF0000")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return f"Error: {e}", 500


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(description="Data Flywheel Viewer")
    parser.add_argument("--annotations-dir", required=True, help="Path to annotations directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global ANNOTATIONS_DIR
    ANNOTATIONS_DIR = args.annotations_dir

    load_manifest(args.annotations_dir)
    load_annotations(args.annotations_dir)

    print(f"\n{'='*60}")
    print(f"  Rex-Embodied Data Flywheel Viewer")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Annotations: {args.annotations_dir}")
    print(f"  Total: {STATS_CACHE.get('total_annotations', 0):,} annotations")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
