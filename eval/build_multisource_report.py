#!/usr/bin/env python3
"""HTML case report for the clean multi-source training set (per-source sampling).

Reads gsa_multisource_train.jsonl (relative image paths) + the images manifest (rel->abs) to
render, per source: the image with overlaid point/bbox (parsed from the assistant answer, in
thousandth scale), the user question, and the answer. Lets you visually confirm diversity +
coordinate correctness after the fix.

Usage: python eval/build_multisource_report.py --per-source 4 --out eval/multisource_report.html
"""
import argparse
import base64
import html
import json
import os
import re
from collections import defaultdict

PT_RE = re.compile(r'"point_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]')
BB_RE = re.compile(r'"bbox(?:_2d)?"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]')


def data_uri(path, max_bytes=700_000):
    try:
        if os.path.getsize(path) <= max_bytes:
            ext = os.path.splitext(path)[1].lstrip(".").lower() or "jpg"
            return f"data:image/{ext};base64," + base64.b64encode(open(path, "rb").read()).decode()
    except OSError:
        pass
    return "file://" + path


def overlays(ans):
    """Thousandth-scale point/bbox -> CSS % overlays."""
    h = []
    for x, y in PT_RE.findall(ans):
        h.append(f'<div class="pt" style="left:{float(x)/10:.1f}%;top:{float(y)/10:.1f}%"></div>')
    for x1, y1, x2, y2 in BB_RE.findall(ans):
        l, t = float(x1)/10, float(y1)/10
        w, ht = (float(x2)-float(x1))/10, (float(y2)-float(y1))/10
        h.append(f'<div class="bb" style="left:{l:.1f}%;top:{t:.1f}%;width:{w:.1f}%;height:{ht:.1f}%"></div>')
    return "".join(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="trains/gsa_multisource_train.jsonl")
    ap.add_argument("--manifest", default="trains/gsa_multisource_images.json")
    ap.add_argument("--per-source", type=int, default=4)
    ap.add_argument("--out", default="eval/multisource_report.html")
    args = ap.parse_args()

    img_map = json.load(open(args.manifest))["map"]
    by_src = defaultdict(list)
    for l in open(args.train):
        d = json.loads(l)
        src = d["images"][0].split("/")[1]
        if len(by_src[src]) < args.per_source:
            by_src[src].append(d)
        if all(len(v) >= args.per_source for v in by_src.values()) and len(by_src) >= 19:
            pass

    sections = []
    for src in sorted(by_src):
        cards = []
        for d in by_src[src]:
            rel = d["images"][0]
            ab = img_map.get(rel, rel)
            u = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
            a = next((m["content"] for m in d["messages"] if m["role"] == "assistant"), "")
            cards.append(f"""<div class="card"><div class="imgwrap">
<img src="{data_uri(ab)}" loading="lazy"/>{overlays(str(a))}</div>
<div class="meta"><div class="q">{html.escape(str(u).replace('<image>','')[:200])}</div>
<pre>{html.escape(str(a)[:400])}</pre></div></div>""")
        sections.append(f'<h2>{src} <span class="sub">({len(by_src[src])} shown)</span></h2>'
                        f'<div class="grid">{"".join(cards)}</div>')

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Multi-source GSA Data</title>
<style>
body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;margin:0;padding:24px}}
h1{{color:#58a6ff}} h2{{color:#79c0ff;border-bottom:1px solid #30363d}} .sub{{color:#8b949e;font-size:.7em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}
.imgwrap{{position:relative;background:#000;text-align:center}}
.imgwrap img{{max-width:100%;max-height:280px;display:block;margin:auto}}
.pt{{position:absolute;width:8px;height:8px;background:#3fb950;border:1px solid #000;border-radius:50%;transform:translate(-50%,-50%)}}
.bb{{position:absolute;border:2px solid #f0883e}}
.meta{{padding:8px}} .q{{color:#adbac7;font-size:.8em;margin-bottom:4px}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:.7em;color:#8b949e;margin:0}}
</style></head><body>
<h1>Clean Multi-Source GSA Data — Case Report</h1>
<p class="sub">19 real GSA sources, thousandth coords, relative image paths. Green dot=point_2d, orange box=bbox (overlaid from the answer).</p>
{''.join(sections)}
</body></html>"""
    open(args.out, "w").write(doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out)//1024} KB), {len(by_src)} sources")


if __name__ == "__main__":
    main()
