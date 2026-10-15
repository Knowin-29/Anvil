#!/usr/bin/env python3
"""Build a static robot-state review page for one GSA run."""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _esc(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def _row(label: Dict[str, Any]) -> str:
    state = _esc(label.get("state_type"))
    value = _esc(label.get("state_value"))
    conf = label.get("confidence")
    conf_text = "" if conf is None else f"{float(conf):.2f}"
    flags = label.get("quality_flags") or []
    flag_html = "".join(f"<span class='flag'>{_esc(flag)}</span>" for flag in flags)
    status = _esc(label.get("verifier_status"))
    verifier = _esc(label.get("verifier_result"))
    reason = _esc(label.get("verifier_reason"))
    evidence = _esc(label.get("visual_evidence"))
    target = _esc(label.get("target_object_class"))
    return f"""
      <tr data-state="{state}">
        <td class="state">{state}</td>
        <td><span class="value value-{value}">{value}</span></td>
        <td>{target}</td>
        <td>{conf_text}</td>
        <td>{evidence}</td>
        <td><span class="verifier">{verifier}</span><div class="muted">{reason}</div></td>
        <td><span class="status">{status}</span><div class="flags">{flag_html}</div></td>
      </tr>
    """


def build(run_dir: Path, output: Path) -> None:
    label_path = run_dir / "labels" / "A_robot_state" / "A_robot_state.jsonl"
    labels = _read_jsonl(label_path)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for label in labels:
        key = label.get("image_path") or label.get("sample_id") or "unknown"
        groups[str(key)].append(label)

    cards = []
    for idx, (image_path, items) in enumerate(sorted(groups.items()), start=1):
        sample = _esc(items[0].get("sample_id"))
        image_src = _esc(image_path)
        rows = "\n".join(_row(item) for item in sorted(items, key=lambda x: x.get("state_type", "")))
        payload = _esc(json.dumps({"image_path": image_path, "labels": items}, ensure_ascii=False))
        cards.append(f"""
    <section class="card" data-index="{idx}" data-sample="{sample}">
      <div class="media">
        <img src="{image_src}" alt="{sample}" loading="lazy" />
      </div>
      <div class="panel">
        <div class="topline">
          <div><strong>{idx}. {sample}</strong><div class="path">{_esc(image_path)}</div></div>
          <div class="actions" data-payload="{payload}">
            <button type="button" data-vote="correct">对</button>
            <button type="button" data-vote="wrong">错</button>
            <button type="button" data-vote="unsure">不确定</button>
          </div>
        </div>
        <table>
          <thead><tr><th>state</th><th>value</th><th>target</th><th>conf</th><th>visual_evidence</th><th>verifier</th><th>quality</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <textarea placeholder="备注"></textarea>
      </div>
    </section>
        """)

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>A_robot_state Review</title>
<style>
:root {{ color-scheme: light; font-family: Arial, sans-serif; }}
body {{ margin: 0; background: #f5f6f8; color: #1d2430; }}
header {{ position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #d9dee7; padding: 12px 20px; display: flex; justify-content: space-between; gap: 16px; align-items: center; }}
main {{ padding: 18px 20px 40px; }}
button {{ border: 1px solid #bcc5d2; background: #fff; padding: 6px 10px; border-radius: 6px; cursor: pointer; }}
button.active {{ background: #17324d; color: #fff; border-color: #17324d; }}
.card {{ display: grid; grid-template-columns: minmax(320px, 42vw) 1fr; gap: 16px; background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; margin-bottom: 16px; }}
.media img {{ width: 100%; max-height: 72vh; object-fit: contain; background: #101820; border-radius: 6px; }}
.topline {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 10px; }}
.path {{ font-size: 12px; color: #667085; overflow-wrap: anywhere; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-top: 1px solid #e6e9ef; padding: 7px 6px; vertical-align: top; text-align: left; }}
th {{ background: #f8fafc; font-weight: 600; }}
.state {{ font-weight: 600; white-space: nowrap; }}
.value {{ border-radius: 4px; padding: 2px 6px; font-weight: 700; }}
.value-yes {{ background: #dff5e7; color: #116534; }}
.value-no {{ background: #eef2f7; color: #344054; }}
.value-unknown {{ background: #fff2cc; color: #8a5a00; }}
.flag {{ display: inline-block; margin: 2px; padding: 2px 5px; border-radius: 4px; background: #ffe4e4; color: #9b1c1c; font-size: 12px; }}
.muted {{ color: #667085; font-size: 12px; margin-top: 3px; }}
textarea {{ width: 100%; min-height: 58px; resize: vertical; margin-top: 10px; border: 1px solid #cfd6e0; border-radius: 6px; padding: 8px; box-sizing: border-box; }}
@media (max-width: 980px) {{ .card {{ grid-template-columns: 1fr; }} header {{ align-items: flex-start; flex-direction: column; }} }}
</style>
</head>
<body>
<header>
  <div><strong>A_robot_state Review</strong><div class="path">{_esc(str(label_path))} | images: {len(groups)} | labels: {len(labels)}</div></div>
  <button type="button" id="export">导出 JSON</button>
</header>
<main>
{''.join(cards)}
</main>
<script>
const storageKey = 'robot_state_review:' + location.pathname;
const state = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
function save() {{ localStorage.setItem(storageKey, JSON.stringify(state)); }}
document.querySelectorAll('.card').forEach(card => {{
  const id = card.dataset.sample + ':' + card.dataset.index;
  const note = card.querySelector('textarea');
  const saved = state[id] || {{}};
  if (saved.note) note.value = saved.note;
  if (saved.vote) card.querySelectorAll('button[data-vote]').forEach(b => b.classList.toggle('active', b.dataset.vote === saved.vote));
  card.querySelectorAll('button[data-vote]').forEach(button => button.addEventListener('click', () => {{
    state[id] = state[id] || {{}};
    state[id].vote = button.dataset.vote;
    state[id].sample = card.dataset.sample;
    state[id].payload = JSON.parse(card.querySelector('.actions').dataset.payload);
    card.querySelectorAll('button[data-vote]').forEach(b => b.classList.toggle('active', b === button));
    save();
  }}));
  note.addEventListener('input', () => {{
    state[id] = state[id] || {{}};
    state[id].sample = card.dataset.sample;
    state[id].note = note.value;
    state[id].payload = JSON.parse(card.querySelector('.actions').dataset.payload);
    save();
  }});
}});
document.getElementById('export').addEventListener('click', () => {{
  const blob = new Blob([JSON.stringify(Object.values(state), null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'robot_state_review_export.json';
  a.click();
  URL.revokeObjectURL(a.href);
}});
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "reports" / "robot_state_review.html"
    build(args.run_dir, output)
    print(output)


if __name__ == "__main__":
    main()
