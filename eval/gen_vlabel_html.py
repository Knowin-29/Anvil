#!/usr/bin/env python3
"""Generate HTML acceptance report from vlabel_score_*.json files."""
import json, glob, os

EVAL_DIR = "${GSA_FORGE_ROOT}/eval"
OUT_HTML = "${GSA_FORGE_ROOT}/trains/vlabel_eval_report.html"

# Load all score files
scores = {}
for f in sorted(glob.glob(os.path.join(EVAL_DIR, "vlabel_score_*.json"))):
    d = json.load(open(f))
    scores[d['checkpoint']] = d

if not scores:
    print("No score files found")
    exit(1)

# Order checkpoints logically
order = ['v4_5000', 'v5_7500', 'v7_3000', 'v7_4500']
ckpts = [c for c in order if c in scores] + [c for c in scores if c not in order]

ckpt_labels = {
    'v4_5000': 'v4-ckpt5000<br/>(???+desktop)',
    'v5_7500': 'v5-ckpt7500<br/>(??+??)',
    'v7_3000': 'v7-ckpt3000<br/>(??+refcoco)',
    'v7_4500': 'v7-ckpt4500<br/>(????)',
}

# All families
all_families = set()
for d in scores.values():
    all_families.update(d['by_family'].keys())
all_families = sorted(all_families)

def acc_cell(v):
    if v is None:
        return '<td class="num">-</td>'
    pct = v * 100
    color = '#4CAF50' if pct >= 80 else '#FF9800' if pct >= 60 else '#F44336'
    return f'<td class="num" style="color:{color};font-weight:600">{pct:.1f}%</td>'

# Overall cards
overall_rows = ""
for c in ckpts:
    ov = scores[c]['overall']
    overall_rows += f'<div class="card"><div class="v">{ov["acc"]*100:.1f}%</div><div class="l">{ckpt_labels.get(c,c)}</div><div class="sub">{ov["correct"]}/{ov["total"]}</div></div>\n'

# Family comparison table
fam_header = ''.join(f'<th>{ckpt_labels.get(c,c)}</th>' for c in ckpts)
fam_rows = ""
fam_label = {
    'bbox_grounding_single_or_few': 'bbox??(?/?)',
    'bbox_grounding_many_objects': 'bbox??(???)',
    'point_or_affordance': 'point/affordance',
    'spatial_relation_binary': '??????',
    'closed_choice_token': '????',
    'attribute_classification': '????',
    'state_classification': '????',
    'temporal_or_multiframe': '??/??',
    'robot_planning_json': '?????',
    'object_classification': '????',
    'untagged': '???',
    'other': '??',
}
for fam in all_families:
    cells = ''
    for c in ckpts:
        v = scores[c]['by_family'].get(fam)
        cells += acc_cell(v['acc'] if v else None)
    n = scores[ckpts[0]]['by_family'].get(fam, {}).get('total', '-')
    fam_rows += f'<tr><td>{fam_label.get(fam,fam)}</td><td class="num">{n}</td>{cells}</tr>\n'

# Capability table
cap_header = ''.join(f'<th>{ckpt_labels.get(c,c)}</th>' for c in ckpts)
cap_rows = ""
cap_label = {'G': 'G ?? Grounding', 'S': 'S ?? Spatial', 'A': 'A ??/?? Affordance'}
for cap in ['G', 'S', 'A']:
    cells = ''
    for c in ckpts:
        v = scores[c]['by_capability'].get(cap)
        cells += acc_cell(v['acc'] if v else None)
    cap_rows += f'<tr><td>{cap_label.get(cap,cap)}</td>{cells}</tr>\n'

html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>GSA_Forge Checkpoint ??????</title>
<style>
body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#eef1f5;color:#2c3e50;line-height:1.6}}
.wrap{{max-width:1200px;margin:0 auto;padding:24px}}
header{{background:linear-gradient(135deg,#1a237e,#3949ab);color:#fff;padding:30px;border-radius:14px;margin-bottom:24px}}
header h1{{margin:0 0 8px;font-size:24px}}
header .sub{{opacity:0.9;font-size:14px}}
h2{{color:#1a237e;border-bottom:3px solid #2196F3;padding-bottom:8px;margin-top:32px;font-size:19px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:18px 0}}
.card{{background:#fff;border-radius:12px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07)}}
.card .v{{font-size:30px;font-weight:700;color:#2196F3}}
.card .l{{font-size:12px;color:#666;margin-top:6px}}
.card .sub{{font-size:11px;color:#aaa;margin-top:2px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.07);margin:14px 0;font-size:13px}}
th{{background:#1a237e;color:#fff;padding:10px 8px;text-align:left;font-weight:500}}
td{{padding:9px 8px;border-bottom:1px solid #f0f0f0}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
tr:hover{{background:#f8f9ff}}
.box{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.07);margin:14px 0;font-size:14px}}
.alert{{padding:14px 18px;margin:12px 0;border-radius:8px;font-size:13px}}
.alert-ok{{background:#e8f5e9;border-left:4px solid #4CAF50}}
.alert-i{{background:#e3f2fd;border-left:4px solid #2196F3}}
.alert-w{{background:#fff8e1;border-left:4px solid #FF9800}}
</style></head><body><div class="wrap">

<header>
<h1>GSA_Forge Checkpoint ??????</h1>
<div class="sub">???: correct-balanced-train-test-v1 (verifier-correct ????) ? ???? 2212 ? ? {len(ckpts)} ? checkpoint ??</div>
</header>

<h2>???????</h2>
<div class="box">
<p><b>??:</b> ${DATA_ROOT}/verifier-labels/dataset.sqlite3 ??? test.jsonl (10?? verifier ??? correct ?????????)</p>
<p><b>??:</b> ?? label=correct,?"??????"????????? image+prompt ? replay,?????? verifier ????????</p>
<p><b>??:</b> 12 ? task_family(bbox??/????/point/??/??/??/???),? GSA ??(G/S/A)????????? Qwen3.5-397B?</p>
<p><b>????:</b> bbox ? IoU?0.5;????? pose_family ??;?????????;??/??? JSON ????;?????????</p>
</div>

<h2>?????????</h2>
<div class="cards">
{overall_rows}
</div>

<h2>???????(task_family)??</h2>
<table>
<tr><th>????</th><th>???</th>{fam_header}</tr>
{fam_rows}
</table>

<h2>??? GSA ??????</h2>
<table>
<tr><th>????</th>{cap_header}</tr>
{cap_rows}
</table>

<h2>????</h2>
<div class="box" id="conclusion">
<p>(???????????)</p>
</div>

</div></body></html>'''

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"HTML report: {OUT_HTML} ({len(html)} bytes)")
print(f"Checkpoints included: {ckpts}")
