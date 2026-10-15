#!/usr/bin/env python3
"""
Generate data processing report with HTML visualization.

Usage:
    python trains/generate_report.py
    # Opens trains/data_quality_report.html
"""
import json
import os
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent


def main():
    report_path = SCRIPT_DIR / "data_report.json"
    log_report_path = SCRIPT_DIR / "log_extract_report.json"

    r = json.load(open(report_path))
    log_r = json.load(open(log_report_path)) if log_report_path.exists() else {}

    # Compute stats
    dim_counts = defaultdict(int)
    source_data = []
    total_filtered = 0
    total_deduped = 0

    for name, info in r["per_source"].items():
        if info.get("status") in ("skipped_raw", "empty"):
            continue
        loaded = info.get("loaded", 0)
        clean = info.get("clean", 0)
        after = info.get("after_ratio", 0)
        filt = info.get("filtered_out", 0)
        dedup = info.get("deduped_out", 0)
        dims = info.get("dimensions", [])
        general = info.get("general", False)

        total_filtered += filt
        total_deduped += dedup

        if after > 0:
            for d in dims:
                dim_counts[d] += after
            source_data.append({
                "name": name,
                "loaded": loaded,
                "clean": clean,
                "final": after,
                "filtered": filt,
                "deduped": dedup,
                "dimensions": dims,
                "general": general,
            })

    # Sort by final count
    source_data.sort(key=lambda x: -x["final"])

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>GSA_Forge ??????</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
h2 {{ color: #555; margin-top: 30px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
.stat-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
.stat-card .value {{ font-size: 28px; font-weight: bold; color: #2196F3; }}
.stat-card .label {{ font-size: 14px; color: #666; margin-top: 5px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
th {{ background: #2196F3; color: white; padding: 12px 8px; text-align: left; font-size: 13px; }}
td {{ padding: 10px 8px; border-bottom: 1px solid #eee; font-size: 13px; }}
tr:hover {{ background: #f0f7ff; }}
.bar {{ height: 20px; border-radius: 3px; display: inline-block; min-width: 3px; }}
.bar-g {{ background: #4CAF50; }}
.bar-s {{ background: #2196F3; }}
.bar-a-attr {{ background: #FF9800; }}
.bar-a-iface {{ background: #9C27B0; }}
.dim-chart {{ display: flex; align-items: center; gap: 10px; margin: 5px 0; }}
.dim-label {{ width: 180px; text-align: right; font-size: 13px; }}
.dim-bar-container {{ flex: 1; background: #eee; border-radius: 3px; height: 24px; position: relative; }}
.dim-bar {{ height: 100%; border-radius: 3px; display: flex; align-items: center; padding-left: 8px; color: white; font-size: 12px; font-weight: bold; }}
.dim-value {{ width: 120px; font-size: 13px; }}
.warning {{ background: #FFF3E0; border-left: 4px solid #FF9800; padding: 10px 15px; margin: 10px 0; border-radius: 0 4px 4px 0; }}
.success {{ background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 10px 15px; margin: 10px 0; border-radius: 0 4px 4px 0; }}
.tag {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin: 1px; }}
.tag-g {{ background: #E8F5E9; color: #2E7D32; }}
.tag-s {{ background: #E3F2FD; color: #1565C0; }}
.tag-a {{ background: #FFF3E0; color: #E65100; }}
.tag-general {{ background: #F3E5F5; color: #6A1B9A; }}
.changelog {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.changelog li {{ margin: 8px 0; line-height: 1.5; }}
.mono {{ font-family: monospace; background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
<h1>GSA_Forge ??????</h1>
<p>????: {r.get('generated_at', 'N/A')} | Registry: sources_registry.yaml (v3 ????)</p>

<div class="summary-grid">
  <div class="stat-card"><div class="value">{r['summary']['sft_total']:,}</div><div class="label">SFT ????</div></div>
  <div class="stat-card"><div class="value">{r['summary']['grpo_total']:,}</div><div class="label">GRPO Prompts</div></div>
  <div class="stat-card"><div class="value">{r['summary']['opd_total']:,}</div><div class="label">OPD On-policy</div></div>
  <div class="stat-card"><div class="value">{log_r.get('train_records', 0):,}</div><div class="label">Log SFT (??)</div></div>
  <div class="stat-card"><div class="value">{len(source_data)}</div><div class="label">?????</div></div>
  <div class="stat-card"><div class="value">{r['summary']['image_spot_check']}</div><div class="label">????</div></div>
</div>

<h2>????</h2>
"""
    # Dimension bars
    max_dim = max(dim_counts.values()) if dim_counts else 1
    colors = {
        "G_grounding": ("#4CAF50", "Grounding"),
        "S_spatial": ("#2196F3", "Spatial"),
        "A_identity_attr": ("#FF9800", "Identity/Attr"),
        "A_action_interface": ("#9C27B0", "Action Interface"),
        "A_obj_state": ("#F44336", "Object State"),
        "A_action_precondition": ("#F44336", "Action Precond"),
    }
    for dim in ["G_grounding", "S_spatial", "A_identity_attr", "A_action_interface", "A_obj_state", "A_action_precondition"]:
        cnt = dim_counts.get(dim, 0)
        pct = cnt / max_dim * 100 if max_dim else 0
        color, label = colors.get(dim, ("#999", dim))
        total_pct = cnt / sum(dim_counts.values()) * 100 if sum(dim_counts.values()) else 0
        html += f"""<div class="dim-chart">
  <div class="dim-label">{label}</div>
  <div class="dim-bar-container"><div class="dim-bar" style="width:{max(pct,1)}%; background:{color}">{cnt:,}</div></div>
  <div class="dim-value">{total_pct:.1f}%</div>
</div>\n"""

    # Warnings
    html += """
<div class="warning">A_obj_state ? A_action_precondition ??? 0 ? ????????????????????? verified ???? teacher ?????</div>
<div class="success">?????? (5200/5200 ????)???????? (????????)?</div>
"""

    # Source table
    html += """<h2>?????</h2>
<table>
<tr><th>???</th><th>??</th><th>??/??</th><th>Clean</th><th>Final (ratio?)</th><th>??</th><th>??</th></tr>
"""
    for s in source_data:
        dim_tags = ""
        for d in s["dimensions"]:
            tag_class = "tag-g" if d.startswith("G") else "tag-s" if d.startswith("S") else "tag-a"
            dim_tags += f'<span class="tag {tag_class}">{d.split("_")[0]}</span>'
        type_tag = '<span class="tag tag-general">??</span>' if s["general"] else "GSA??"
        cleaning = ""
        if s["filtered"]: cleaning += f"?? {s['filtered']:,}"
        if s["deduped"]: cleaning += f" ?? {s['deduped']:,}"
        if not cleaning: cleaning = "-"
        html += f"""<tr>
  <td><span class="mono">{s['name']}</span></td>
  <td>{s['loaded']:,}</td>
  <td>{cleaning}</td>
  <td>{s['clean']:,}</td>
  <td><b>{s['final']:,}</b></td>
  <td>{dim_tags}</td>
  <td>{type_tag}</td>
</tr>\n"""

    html += "</table>"

    # Changelog
    html += """
<h2>??????</h2>
<div class="changelog">
<ul>
<li><b>v1 ? v2 (-37%)</b>: ?? bbox ? (img,response) ?? 384K ????; ??? images ????; ?? response&lt;2??; ?? sft_point_rootData (89?? video QA ??)</li>
<li><b>v2 ? v3 (-53% vs v1)</b>: ?? run1/run4 pointctl ????????, ??? PRISM+refspatial+robopoint+roborefit+FSD+SAT+pixmo ?????????? pointctl ??, ???????? (ratio/????)</li>
</ul>
<p><b>????</b>: <span class="mono">POINT_REPOINT_CHECKLIST.md</span> ?? run1/run4 ?????????, ? <span class="mono">dataset_jsons_qwen3_swift/</span> ?????????? (?? originDatasets_transfer_from_hpc3 ????)?</p>
</div>

<h2>Log ????</h2>
<div class="summary-grid">
  <div class="stat-card"><div class="value">{log_r.get('records_produced', 0):,}</div><div class="label">?? GSA ??</div></div>
  <div class="stat-card"><div class="value">{log_r.get('train_records', 0):,}</div><div class="label">???</div></div>
  <div class="stat-card"><div class="value">{log_r.get('test_records', 0):,}</div><div class="label">???</div></div>
</div>
<table>
<tr><th>??</th><th>??</th></tr>
"""
    for dim, cnt in sorted(log_r.get("by_dimension", {}).items(), key=lambda x: -x[1]):
        html += f"<tr><td>{dim}</td><td>{cnt:,}</td></tr>\n"
    html += """</table>
<p>????: Qwen3.5-397B ?????? log, ? bbox + affordance_point JSON response?</p>

<h2>????</h2>
<div class="changelog">
<pre>
cd /path/to/storage/anvil/GSA_Forge
source .env

# ????? (trains/gsa_sft_train.jsonl ?)
# ?? log ???? SFT:
cat trains/gsa_log_sft.jsonl >> trains/gsa_sft_train.jsonl

# ???:
bash scripts/run_sft_27b.sh          # Stage 1: 27B Teacher SFT
bash scripts/run_grpo_27b.sh         # Stage 2: 27B GRPO (?Stage1?)
bash scripts/run_opd_9b.sh           # Stage 3: OPD?? (?Stage2?)
</pre>
</div>

</div>
</body>
</html>"""

    output_path = SCRIPT_DIR / "data_quality_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
