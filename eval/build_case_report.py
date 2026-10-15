#!/usr/bin/env python3
"""Build an HTML case-sampling report of GSA flywheel data for manual quality inspection.

Samples verified records per dimension (only those with locally-reachable images), renders
each as: the image (with GT bbox/point overlaid via CSS), the raw annotation JSON, and
automatic quality flags (e.g. degenerate/huge bbox, empty attributes, out-of-range coords,
suspicious state values). Also embeds a data-composition summary (scene/source distribution)
so the reviewer sees BOTH aggregate skew and concrete per-sample quality.

Usage: python eval/build_case_report.py --per-dim 12 --out eval/gsa_case_report.html
Images are referenced by absolute file:// path so the report opens locally.
"""
import argparse
import base64
import glob
import html
import json
import os
import random
from collections import Counter

FLYWHEEL = "/path/to/Anvil/data/output/gsa_flywheel_exp/verified"
DIMS = ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state",
        "A_action_precondition", "A_action_interface", "A_robot_state"]


def img_data_uri(path, max_bytes=900_000):
    """Return a data: URI for small images so the report is self-contained; else file path."""
    try:
        sz = os.path.getsize(path)
        if sz <= max_bytes:
            ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/{ext};base64,{b64}"
    except OSError:
        pass
    return "file://" + path


def flags_for(dim, d):
    """Automatic quality flags per record (thousandth-scale assumed for normalized coords)."""
    fl = []
    sz = d.get("image_size")
    b = d.get("bbox")
    if dim in ("G_grounding", "A_identity_attr", "A_action_interface") and b:
        if len(b) == 4:
            if b[2] <= b[0] or b[3] <= b[1]:
                fl.append("degenerate_bbox")
            if sz and (b[2] > sz[0] * 1.02 or b[3] > sz[1] * 1.02):
                fl.append("bbox_exceeds_image")
            if sz and sz[0] and sz[1]:
                area = (b[2]-b[0])*(b[3]-b[1])/(sz[0]*sz[1])
                if area > 0.9:
                    fl.append("bbox_covers_whole_image")
                if area < 0.002:
                    fl.append("bbox_tiny")
    if dim == "A_identity_attr":
        a = d.get("attributes")
        if not a:
            fl.append("empty_attributes")
        elif isinstance(a, dict) and len(a) < 2:
            fl.append("few_attributes")
    if dim == "A_obj_state" and not d.get("state_value"):
        fl.append("empty_state_value")
    if dim == "A_action_interface":
        gp = d.get("grasp_point_2d")
        if gp and b and len(b) == 4 and not (b[0] <= gp[0] <= b[2] and b[1] <= gp[1] <= b[3]):
            fl.append("grasp_point_outside_bbox")
    if dim == "S_spatial" and not (d.get("subject") and d.get("predicate") and d.get("object")):
        fl.append("incomplete_triple")
    return fl


def overlay_style(d, dim):
    """CSS box overlay for bbox in PIXEL coords (flywheel raw), positioned relative to image."""
    sz = d.get("image_size")
    b = d.get("bbox")
    if not (sz and b and len(b) == 4 and sz[0] and sz[1]):
        return ""
    l = 100*b[0]/sz[0]; t = 100*b[1]/sz[1]
    w = 100*(b[2]-b[0])/sz[0]; h = 100*(b[3]-b[1])/sz[1]
    return (f'<div class="bbox" style="left:{l:.1f}%;top:{t:.1f}%;'
            f'width:{w:.1f}%;height:{h:.1f}%"></div>')


def collect(dim, n, seed=7):
    files = sorted(glob.glob(os.path.join(FLYWHEEL, dim, "*.jsonl")))
    local = []
    scene = Counter(); dataset = Counter(); total = 0
    for fp in files:
        for l in open(fp):
            try:
                d = json.loads(l)
            except Exception:
                continue
            total += 1
            scene[d.get("scene_type", "?")] += 1
            dataset[d.get("dataset", "?")] += 1
            if os.path.exists(d.get("image_path", "")):
                local.append(d)
    random.Random(seed).shuffle(local)
    return local[:n], total, len(local), scene, dataset


def render_card(d, dim):
    uri = img_data_uri(d["image_path"])
    fl = flags_for(dim, d)
    flag_html = "".join(f'<span class="flag">{html.escape(f)}</span>' for f in fl) or '<span class="ok">clean</span>'
    ann = {k: v for k, v in d.items() if not k.startswith("_") and k not in
           ("pipeline", "sample_id", "object_id", "timestamp", "bucket")}
    ann_str = html.escape(json.dumps(ann, ensure_ascii=False, indent=2))
    return f"""<div class="card">
  <div class="imgwrap"><img src="{uri}" loading="lazy"/>{overlay_style(d, dim)}</div>
  <div class="meta"><div class="flags">{flag_html}</div>
  <pre>{ann_str}</pre></div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dim", type=int, default=12)
    ap.add_argument("--out", default="eval/gsa_case_report.html")
    args = ap.parse_args()

    sections = []
    summary_rows = []
    for dim in DIMS:
        cards, total, nloc, scene, dataset = collect(dim, args.per_dim)
        summary_rows.append(
            f"<tr><td>{dim}</td><td>{total:,}</td><td>{nloc:,} ({100*nloc/max(total,1):.0f}%)</td>"
            f"<td>{', '.join(f'{k}:{v}' for k,v in dataset.most_common(3))}</td></tr>")
        if not cards:
            sections.append(f'<h2>{dim}</h2><p class="warn">No locally-viewable images '
                            f'(total {total:,}, local 0). Images likely on a missing mount.</p>')
            continue
        sections.append(f'<h2>{dim} <span class="sub">(showing {len(cards)} of {nloc:,} local / '
                        f'{total:,} total)</span></h2><div class="grid">' +
                        "".join(render_card(d, dim) for d in cards) + "</div>")

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>GSA Data Case Report</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:24px}}
h1{{color:#58a6ff}} h2{{color:#79c0ff;border-bottom:1px solid #30363d;padding-top:20px}}
.sub{{color:#8b949e;font-size:.7em;font-weight:normal}}
.warn{{color:#f0883e}}
table{{border-collapse:collapse;margin:12px 0;width:100%;max-width:1000px}}
td,th{{border:1px solid #30363d;padding:6px 10px;text-align:left;font-size:.9em}}
th{{color:#8b949e}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}
.imgwrap{{position:relative;background:#000;text-align:center}}
.imgwrap img{{max-width:100%;max-height:300px;display:block;margin:auto}}
.bbox{{position:absolute;border:2px solid #3fb950;box-shadow:0 0 0 1px #000}}
.meta{{padding:10px}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:.72em;color:#adbac7;margin:6px 0 0}}
.flag{{display:inline-block;background:#5a1e02;color:#f0883e;padding:1px 6px;border-radius:4px;margin:1px;font-size:.72em}}
.ok{{color:#3fb950;font-size:.75em}}
.note{{background:#161b22;border-left:3px solid #f0883e;padding:10px 14px;margin:12px 0;max-width:1000px}}
</style></head><body>
<h1>GSA Flywheel Data — Case Sampling & Quality Report</h1>
<div class="note"><b>Data composition (verified set).</b> Sources are highly concentrated; embodied
manipulation diversity is thin. bbox overlays are drawn in raw pixel coords (flywheel native scale).</div>
<table><tr><th>Dimension</th><th>Total verified</th><th>Local images</th><th>Top datasets</th></tr>
{''.join(summary_rows)}</table>
{''.join(sections)}
<p class="sub">Generated by eval/build_case_report.py — flags are heuristic; inspect manually.</p>
</body></html>"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
