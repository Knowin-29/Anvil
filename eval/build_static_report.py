#!/usr/bin/env python3
"""Build a SELF-CONTAINED HTML audit report (images embedded as base64).

The live dashboard binds to a private IP that is not routable from the user's
machine, so this produces a single .html file that can be opened directly in the
IDE / browser with no server and no network.

  python eval/build_static_report.py --data data/flywheel_all --per-form 12
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = ["#22d3ee", "#f97316", "#a3e635", "#f472b6", "#facc15", "#60a5fa"]


def render(rec: dict, max_px: int = 560) -> str | None:
    try:
        im = Image.open(rec["image_path"]).convert("RGB")
    except Exception:
        return None
    W, H = im.size
    d = ImageDraw.Draw(im)

    def S(x, y):
        return (x / 1000.0 * W, y / 1000.0 * H)

    items = rec.get("payload")
    items = items if isinstance(items, list) else [items]
    for k, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        c = COLORS[k % len(COLORS)]
        b = it.get("bbox_2d") or it.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            d.rectangle([S(b[0], b[1]), S(b[2], b[3])], outline=c, width=3)
            lab = str(it.get("label") or it.get("object_class") or "")
            if lab:
                d.text(S(b[0] + 6, b[1] + 6), lab[:30], fill=c)
        poly = it.get("polygon")
        if isinstance(poly, list) and poly and isinstance(poly[0], (list, tuple)):
            pts = [S(p[0], p[1]) for p in poly]
            d.line(pts + [pts[0]], fill=c, width=3)
            for p in pts:
                d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=c)
        for key, rad in (("point_2d", 7), ("contact_point_2d", 7),
                         ("grasp_point_2d", 9), ("target_point_2d", 7)):
            p = it.get(key)
            if isinstance(p, (list, tuple)) and len(p) == 2:
                x, y = S(p[0], p[1])
                d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=c, width=3)
        ax, gp = it.get("jaw_axis_2d"), it.get("grasp_point_2d")
        if (isinstance(ax, (list, tuple)) and len(ax) == 2
                and isinstance(gp, (list, tuple)) and len(gp) == 2):
            x, y = S(gp[0], gp[1]); L = 0.10 * max(W, H)
            d.line([(x - ax[0] * L, y - ax[1] * L), (x + ax[0] * L, y + ax[1] * L)],
                   fill=c, width=4)
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/flywheel_all")
    ap.add_argument("--per-form", type=int, default=12)
    ap.add_argument("--out", default="eval/audit_report.html")
    a = ap.parse_args()

    raw = Path(a.data) / "raw"
    cards, counts = [], Counter()
    random.seed(0)
    for jf in sorted(raw.glob("*.jsonl")):
        form = jf.stem
        lines = jf.read_text(errors="ignore").splitlines()
        counts[form] = len(lines)
        random.shuffle(lines)
        picked = 0
        for ln in lines:
            if picked >= a.per_form:
                break
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            b64 = render(rec)
            if not b64:
                continue
            picked += 1
            cards.append({
                "form": form, "src": rec.get("source", "?"),
                "b64": b64,
                "payload": json.dumps(rec.get("payload"), ensure_ascii=False, indent=1)[:1100],
                "img": rec.get("image_path", ""),
                "scan": rec.get("_scan_count", ""),
                "verdict": (rec.get("_verification") or {}).get("verdict", ""),
            })
        print(f"  {form}: {picked} cards", flush=True)

    total = sum(counts.values())
    pills = " ".join(f'<span class=pill>{k} <b>{v:,}</b></span>' for k, v in counts.most_common())
    _q = chr(39)  # single quote, kept out of the f-string to avoid backslashes
    form_buttons = "".join(
        f'<button onclick="F({_q}{k}{_q})">{k}</button>' for k in counts)
    body = []
    for c in cards:
        body.append(f"""<div class="card" data-form="{c['form']}">
<img src="data:image/jpeg;base64,{c['b64']}">
<div class=meta><span class=form>{c['form']}</span><span class=pill>{c['src']}</span>
{'<span class=ok>&#10003; '+c['verdict']+'</span>' if c['verdict'] else ''}
{'<span class=pill>Verified@k <b>'+str(c['scan'])+'</b></span>' if c['scan'] else ''}
<pre>{c['payload'].replace('<','&lt;')}</pre>
<div class=path>{c['img']}</div></div></div>""")

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>GSA-Forge Data Audit</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e8eb}}
header{{padding:14px 20px;background:#171a21;border-bottom:1px solid #262b36;position:sticky;top:0;z-index:9}}
h1{{font-size:17px;margin:0 0 8px}}
.pill{{background:#232834;border:1px solid #313847;border-radius:99px;padding:3px 10px;font-size:12px;margin-right:5px;display:inline-block}}
.pill b{{color:#7dd3fc}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:16px;padding:18px}}
.card{{background:#151922;border:1px solid #242a36;border-radius:10px;overflow:hidden}}
.card img{{width:100%;display:block;background:#000}}
.meta{{padding:10px 12px;font-size:12px;line-height:1.6}}
.form{{display:inline-block;background:#1e3a5f;color:#93c5fd;border-radius:5px;padding:1px 8px;font-size:11px;margin-right:6px}}
.ok{{color:#4ade80}} .path{{opacity:.45;font-size:10px;margin-top:4px;word-break:break-all}}
pre{{white-space:pre-wrap;word-break:break-all;background:#0d1017;border:1px solid #232834;
   border-radius:6px;padding:8px;font-size:11px;max-height:200px;overflow:auto;color:#cbd5e1;margin:6px 0 0}}
button{{background:#1c2130;color:#e6e8eb;border:1px solid #313847;border-radius:6px;padding:5px 11px;font-size:12px;margin:2px;cursor:pointer}}
button:hover{{border-color:#7dd3fc}}
</style></head><body>
<header><h1>GSA-Forge Data Audit &mdash; {total:,} records, {len(cards)} rendered samples</h1>
<div>{pills}</div>
<div style="margin-top:8px">
<button onclick="F('')">all</button>
{form_buttons}
</div></header>
<main id=g>{''.join(body)}</main>
<script>function F(f){{document.querySelectorAll('.card').forEach(c=>
c.style.display=(!f||c.dataset.form===f)?'':'none')}}</script>
</body></html>"""
    Path(a.out).write_text(html)
    mb = Path(a.out).stat().st_size / 1e6
    print(f"\nwrote {a.out} ({mb:.1f} MB, {len(cards)} cards, {total:,} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
