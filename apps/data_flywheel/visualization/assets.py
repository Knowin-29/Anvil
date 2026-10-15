"""Shared assets for pipeline visualization HTML reports."""

from __future__ import annotations

from html import escape

CDN_THREEJS = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"
CDN_THREEJS_CSS2D = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/renderers/CSS2DRenderer.js"
CDN_THREEJS_ORBIT = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js"
CDN_VISJS = "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js"
CDN_VISJS_CSS = "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.css"

OPERATION_COLORS = {
    "pick": "#4CAF50",
    "push": "#FF9800",
    "non_operable": "#9E9E9E",
    None: "#2196F3",
}

RELATION_COLORS = {
    "stacked-on": "#F44336",
    "supporting": "#F44336",
    "occluding": "#FFEB3B",
    "occluded-by": "#FFEB3B",
    "inside": "#2196F3",
    "outside": "#2196F3",
    "blocking": "#FF5722",
}

EDGE_SOURCE_COLORS = {
    "3d": "#e74c3c",
    "2d": "#95a5a6",
    "2d+3d": "#8e44ad",
    "derived": "#3498db",
}

REPORT_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f5f7fb;
  color: #1f2937;
}
nav {
  display: flex;
  gap: 0;
  background: #111827;
  position: sticky;
  top: 0;
  z-index: 100;
}
nav button {
  padding: 12px 18px;
  border: 0;
  background: transparent;
  color: #9ca3af;
  cursor: pointer;
  font-size: 14px;
}
nav button.active {
  color: #fff;
  background: #1f2937;
  border-bottom: 3px solid #60a5fa;
}
section {
  display: none;
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}
section.active {
  display: block;
}
.card {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  color: #fff;
  margin-right: 4px;
}
.badge-pick { background: #4CAF50; }
.badge-push { background: #FF9800; }
.badge-non_operable { background: #9E9E9E; }
.badge-unknown { background: #2196F3; }
.color-swatch {
  display: inline-block;
  width: 16px;
  height: 16px;
  border-radius: 3px;
  border: 1px solid #d1d5db;
  vertical-align: middle;
  margin-right: 4px;
}
"""

TAB_JS = """
function switchTab(tabId, button) {
  document.querySelectorAll('section').forEach((s) => s.classList.remove('active'));
  document.querySelectorAll('nav button').forEach((b) => b.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  if (button) button.classList.add('active');
}
document.addEventListener('DOMContentLoaded', () => {
  const firstButton = document.querySelector('nav button');
  if (firstButton) switchTab(firstButton.getAttribute('data-tab') || 'overview', firstButton);
});
"""


def build_html_skeleton(title: str, body_sections: str, inline_js: str = "") -> str:
    """Build a complete self-contained HTML document."""
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<link rel="stylesheet" href="{CDN_VISJS_CSS}">
<script src="{CDN_VISJS}"></script>
<script type="importmap">
{{
  "imports": {{
    "three": "{CDN_THREEJS.replace('.min.js', '.module.min.js')}",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<style>{REPORT_CSS}</style>
</head>
<body>
{body_sections}
<script>{TAB_JS}</script>
<script>{inline_js}</script>
</body>
</html>"""
