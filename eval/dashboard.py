#!/usr/bin/env python3
"""GSA-Forge data dashboard: browse generated samples with rendered annotations.

Serves a single-page UI so a human can audit whether the data is (a) actually
embodied-relevant and (b) correctly annotated. Overlays bbox / point / polygon /
grasp-axis on the real image, and shows the raw JSON + validator + Verified@k.

Run:  python eval/dashboard.py --port 10307 --data data/flywheel_mf
Then open http://<host>:10307/
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, request, Response

app = Flask(__name__)
STATE: dict = {"records": [], "by_form": Counter(), "root": ""}

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>GSA-Forge Data Dashboard</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e8eb}
header{padding:14px 20px;background:#171a21;border-bottom:1px solid #262b36;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
.pill{background:#232834;border:1px solid #313847;border-radius:99px;padding:3px 10px;font-size:12px}
.pill b{color:#7dd3fc}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:16px;padding:18px}
.card{background:#151922;border:1px solid #242a36;border-radius:10px;overflow:hidden}
.card img{width:100%;display:block;background:#000}
.meta{padding:10px 12px;font-size:12px;line-height:1.55}
.form{display:inline-block;background:#1e3a5f;color:#93c5fd;border-radius:5px;padding:1px 8px;font-size:11px;margin-right:6px}
.ok{color:#4ade80}.bad{color:#f87171}
pre{white-space:pre-wrap;word-break:break-all;background:#0d1017;border:1px solid #232834;
    border-radius:6px;padding:8px;font-size:11px;max-height:190px;overflow:auto;color:#cbd5e1}
select,button{background:#1c2130;color:#e6e8eb;border:1px solid #313847;border-radius:6px;padding:5px 10px;font-size:12px}
a{color:#7dd3fc}
</style></head><body>
<header>
  <h1>GSA-Forge Data Dashboard</h1>
  <span class=pill>total <b id=tot>-</b></span>
  <span id=forms></span>
  <select id=f></select>
  <button onclick=load()>Reshuffle</button>
  <span class=pill>audit: is it embodied? is the annotation right?</span>
</header>
<main id=g></main>
<script>
async function boot(){
  const s=await (await fetch('/api/stats')).json();
  document.getElementById('tot').textContent=s.total.toLocaleString();
  document.getElementById('forms').innerHTML=Object.entries(s.by_form)
    .map(([k,v])=>`<span class=pill>${k} <b>${v.toLocaleString()}</b></span>`).join(' ');
  const sel=document.getElementById('f');
  sel.innerHTML='<option value="">(all forms)</option>'+Object.keys(s.by_form)
    .map(k=>`<option value="${k}">${k}</option>`).join('');
  sel.onchange=load; load();
}
async function load(){
  const f=document.getElementById('f').value;
  const r=await (await fetch('/api/samples?n=12&form='+encodeURIComponent(f))).json();
  document.getElementById('g').innerHTML=r.map(x=>`
    <div class=card>
      <img src="/api/render?i=${x.idx}" loading=lazy>
      <div class=meta>
        <span class=form>${x.form}</span>
        <span class=pill>${x.source}</span>
        ${x.valid?'<span class=ok>&#10003; validated</span>':'<span class=bad>&#10007; invalid</span>'}
        ${x.scan?`<span class=pill>Verified@k <b>${x.scan}</b></span>`:''}
        <pre>${x.payload}</pre>
        <div style="opacity:.55;font-size:10px">${x.img}</div>
      </div>
    </div>`).join('');
}
boot();
</script></body></html>"""


def _load(data_dir: Path) -> None:
    recs = []
    raw_dir = data_dir / "raw"
    files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.is_dir() else sorted(data_dir.glob("**/*.jsonl"))
    for jf in files:
        form = jf.stem
        with jf.open(errors="ignore") as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                recs.append({"form": d.get("form", form), "image_path": d.get("image_path", ""),
                             "source": d.get("source", "?"),
                             "payload": d.get("payload", d),
                             "scan": d.get("_scan_count"),
                             "verif": d.get("_verification", {})})
    STATE["records"] = recs
    STATE["by_form"] = Counter(r["form"] for r in recs)
    print(f"dashboard loaded {len(recs)} records from {data_dir}", flush=True)


@app.get("/")
def index() -> Response:
    return Response(PAGE, mimetype="text/html")


@app.get("/api/stats")
def stats():
    return jsonify({"total": len(STATE["records"]), "by_form": dict(STATE["by_form"].most_common())})


@app.get("/api/samples")
def samples():
    n = int(request.args.get("n", 12))
    form = request.args.get("form") or ""
    idxs = [i for i, r in enumerate(STATE["records"]) if not form or r["form"] == form]
    random.shuffle(idxs)
    out = []
    for i in idxs[:n]:
        r = STATE["records"][i]
        out.append({"idx": i, "form": r["form"], "source": r["source"],
                    "img": r["image_path"],
                    "payload": json.dumps(r["payload"], ensure_ascii=False)[:1200],
                    "scan": r.get("scan"),
                    "valid": (r.get("verif") or {}).get("verdict", "verified") == "verified"})
    return jsonify(out)


@app.get("/api/render")
def render():
    from PIL import Image, ImageDraw
    i = int(request.args.get("i", 0))
    if not (0 <= i < len(STATE["records"])):
        return Response(status=404)
    r = STATE["records"][i]
    try:
        im = Image.open(r["image_path"]).convert("RGB")
    except Exception:
        return Response(status=404)
    W, H = im.size
    d = ImageDraw.Draw(im)

    def S(x, y):  # thousandth -> pixel
        return (x / 1000.0 * W, y / 1000.0 * H)

    items = r["payload"]
    items = items if isinstance(items, list) else [items]
    COLORS = ["#22d3ee", "#f97316", "#a3e635", "#f472b6", "#facc15", "#60a5fa"]
    for k, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        c = COLORS[k % len(COLORS)]
        b = it.get("bbox_2d") or it.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            d.rectangle([S(b[0], b[1]), S(b[2], b[3])], outline=c, width=3)
            lab = str(it.get("label") or it.get("object_class") or "")
            if lab:
                d.text(S(b[0] + 6, b[1] + 6), lab[:28], fill=c)
        poly = it.get("polygon")
        if isinstance(poly, list) and poly and isinstance(poly[0], (list, tuple)):
            d.line([S(p[0], p[1]) for p in poly] + [S(poly[0][0], poly[0][1])], fill=c, width=3)
        for key, rad in (("point_2d", 7), ("contact_point_2d", 7),
                         ("grasp_point_2d", 8), ("target_point_2d", 7)):
            p = it.get(key)
            if isinstance(p, (list, tuple)) and len(p) == 2:
                x, y = S(p[0], p[1])
                d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=c, width=3)
        ax = it.get("jaw_axis_2d")
        gp = it.get("grasp_point_2d")
        if (isinstance(ax, (list, tuple)) and len(ax) == 2
                and isinstance(gp, (list, tuple)) and len(gp) == 2):
            x, y = S(gp[0], gp[1]); L = 0.09 * max(W, H)
            d.line([(x - ax[0] * L, y - ax[1] * L), (x + ax[0] * L, y + ax[1] * L)], fill=c, width=4)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=86)
    return Response(buf.getvalue(), mimetype="image/jpeg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/flywheel_mf")
    ap.add_argument("--port", type=int, default=10307)
    ap.add_argument("--host", default="127.0.0.1", help="Bind address; use an authenticated proxy for remote access")
    a = ap.parse_args()
    _load(Path(a.data))
    app.run(host=a.host, port=a.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
