#!/usr/bin/env python3
"""Generate and serve a robot-state label/image comparison page.

Default usage from repo root:
    python scripts/robot_state_label_viewer.py

It auto-detects the latest data/output/gsa_v0/* run that contains
labels/A_robot_state/A_robot_state.jsonl, builds a readable comparison HTML,
and serves it on http://127.0.0.1:10309/.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import socket
import sys
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is expected in the project env.
    Image = None  # type: ignore[assignment]

STATE_ORDER = [
    "gripper_empty",
    "holding_object",
    "object_between_fingers",
    "contact_state",
    "tool_ready",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def latest_run_dir(base_dir: Path) -> Path:
    candidates: List[Path] = []
    if base_dir.exists():
        for run_dir in base_dir.iterdir():
            label_file = run_dir / "labels" / "A_robot_state" / "A_robot_state.jsonl"
            if run_dir.is_dir() and label_file.exists():
                candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No run found under {base_dir} with labels/A_robot_state/A_robot_state.jsonl"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def state_sort_key(label: Dict[str, Any]) -> Tuple[int, str]:
    state_type = str(label.get("state_type") or "")
    try:
        return (STATE_ORDER.index(state_type), state_type)
    except ValueError:
        return (len(STATE_ORDER), state_type)


def esc(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value), quote=True)


def safe_float_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return esc(value)


def label_value_class(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in {"yes", "no", "unknown"} else "unknown"


def verifier_class(value: Any) -> str:
    text = str(value or "missing").strip().lower()
    return text if text in {"supported", "unsupported", "unsure"} else "missing"


def image_to_data_url(path: str, max_side: int, jpeg_quality: int) -> Tuple[str, str]:
    image_path = Path(path)
    if not image_path.exists():
        return "", "missing"

    if Image is not None:
        try:
            img = Image.open(image_path).convert("RGB")
            if max(img.size) > max_side:
                ratio = max_side / max(img.size)
                size = (max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio)))
                img = img.resize(size, Image.LANCZOS)
            from io import BytesIO

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            data = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{data}", f"{img.size[0]}x{img.size[1]} embedded"
        except Exception:
            pass

    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    try:
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}", "embedded-original"
    except Exception:
        return "", "unreadable"


def image_src(path: str, embed_images: bool, max_side: int, jpeg_quality: int) -> Tuple[str, str]:
    if not path:
        return "", "missing"
    if embed_images:
        return image_to_data_url(path, max_side=max_side, jpeg_quality=jpeg_quality)
    if Path(path).exists():
        return Path(path).resolve().as_uri(), "file-url"
    return path, "raw-path"


def flag_html(flags: Iterable[Any]) -> str:
    return "".join(f"<span class='flag'>{esc(flag)}</span>" for flag in flags)


def label_row(label: Dict[str, Any]) -> str:
    state_type = esc(label.get("state_type"))
    state_value = esc(label.get("state_value"))
    value_cls = label_value_class(label.get("state_value"))
    verifier = esc(label.get("verifier_result"))
    verifier_cls = verifier_class(label.get("verifier_result"))
    quality_flags = label.get("quality_flags") or []
    return f"""
      <tr>
        <td class="state">{state_type}</td>
        <td><span class="value value-{value_cls}">{state_value}</span></td>
        <td>{esc(label.get("target_object_class"))}</td>
        <td>{safe_float_text(label.get("confidence"))}</td>
        <td class="evidence">{esc(label.get("visual_evidence"))}</td>
        <td><span class="verifier verifier-{verifier_cls}">{verifier or "missing"}</span><div class="muted">{esc(label.get("verifier_reason"))}</div></td>
        <td><span class="status">{esc(label.get("verifier_status"))}</span><div class="flags">{flag_html(quality_flags)}</div></td>
      </tr>
    """


def stats(labels: List[Dict[str, Any]], image_count: int) -> Dict[str, int]:
    out = {
        "images": image_count,
        "labels": len(labels),
        "stable_training_label_count": 0,
        "auto_verifiable": 0,
        "review_required": 0,
    }
    for label in labels:
        if label.get("stable_training_label"):
            out["stable_training_label_count"] += 1
        status = str(label.get("verifier_status") or "")
        if status in out:
            out[status] += 1
    return out


def stat_cards(stat: Dict[str, int]) -> str:
    items = [
        ("Images", stat["images"]),
        ("Labels", stat["labels"]),
        ("Stable", stat["stable_training_label_count"]),
        ("Auto", stat["auto_verifiable"]),
        ("Review", stat["review_required"]),
    ]
    return "".join(f"<div class='stat'><b>{value}</b><span>{name}</span></div>" for name, value in items)


def build_html(
    labels_file: Path,
    output: Path,
    limit: Optional[int],
    embed_images: bool,
    max_side: int,
    jpeg_quality: int,
) -> None:
    labels = read_jsonl(labels_file)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for label in labels:
        key = str(label.get("image_path") or label.get("sample_id") or "missing-image")
        groups[key].append(label)

    group_items = sorted(groups.items(), key=lambda item: item[0])
    if limit is not None:
        group_items = group_items[:limit]

    visible_labels = [label for _, items in group_items for label in items]
    summary = stats(visible_labels, len(group_items))
    cards: List[str] = []
    for index, (path, items) in enumerate(group_items, start=1):
        sorted_items = sorted(items, key=state_sort_key)
        sample_id = sorted_items[0].get("sample_id", "") if sorted_items else ""
        src, image_note = image_src(path, embed_images=embed_images, max_side=max_side, jpeg_quality=jpeg_quality)
        rows = "\n".join(label_row(label) for label in sorted_items)
        raw_payload = esc(json.dumps({"image_path": path, "labels": sorted_items}, ensure_ascii=False))
        image_block = (
            f'<img src="{esc(src)}" alt="{esc(sample_id)}" loading="lazy" />'
            if src else '<div class="missing-image">图片不存在或无法读取</div>'
        )
        cards.append(f"""
    <section class="card" data-card="{index}" data-sample="{esc(sample_id)}">
      <div class="media">
        {image_block}
        <div class="image-meta">{esc(image_note)}</div>
      </div>
      <div class="panel">
        <div class="topline">
          <div>
            <strong>{index}. {esc(sample_id)}</strong>
            <div class="path">{esc(path)}</div>
          </div>
          <div class="actions" data-payload="{raw_payload}">
            <button type="button" data-vote="correct">对</button>
            <button type="button" data-vote="wrong">错</button>
            <button type="button" data-vote="unsure">不确定</button>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>state</th><th>value</th><th>target</th><th>conf</th><th>visual_evidence</th><th>verifier</th><th>quality</th></tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <textarea placeholder="备注，仅保存在浏览器 localStorage，可用右上角导出 JSON"></textarea>
      </div>
    </section>
        """)

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>A_robot_state Label Viewer</title>
<style>
:root {{ font-family: Arial, Helvetica, sans-serif; color-scheme: light; }}
body {{ margin: 0; background: #f4f6f9; color: #1f2937; }}
header {{ position: sticky; top: 0; z-index: 5; background: #fff; border-bottom: 1px solid #d7dde7; padding: 12px 18px; display: grid; grid-template-columns: 1fr auto; gap: 14px; align-items: center; }}
main {{ padding: 16px 18px 42px; }}
.title {{ font-size: 16px; font-weight: 700; }}
.path {{ color: #667085; font-size: 12px; overflow-wrap: anywhere; margin-top: 3px; }}
.stats {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
.stat {{ min-width: 70px; background: #f8fafc; border: 1px solid #dce2ea; border-radius: 6px; padding: 6px 8px; text-align: center; }}
.stat b {{ display: block; font-size: 16px; }}
.stat span {{ display: block; font-size: 11px; color: #667085; }}
.toolbar {{ grid-column: 1 / -1; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
button, select {{ border: 1px solid #b9c2cf; background: #fff; padding: 6px 10px; border-radius: 6px; cursor: pointer; }}
button.active {{ background: #17324d; color: #fff; border-color: #17324d; }}
.card {{ display: grid; grid-template-columns: minmax(360px, 43vw) 1fr; gap: 16px; background: #fff; border: 1px solid #d7dde7; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
.media {{ min-width: 0; }}
.media img {{ width: 100%; max-height: 74vh; object-fit: contain; background: #0f172a; border-radius: 6px; display: block; }}
.image-meta {{ margin-top: 5px; color: #667085; font-size: 12px; }}
.missing-image {{ height: 280px; display: grid; place-items: center; background: #111827; color: #fff; border-radius: 6px; }}
.topline {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 10px; }}
.actions {{ white-space: nowrap; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-top: 1px solid #e6eaf0; padding: 7px 6px; text-align: left; vertical-align: top; }}
th {{ background: #f8fafc; font-weight: 700; color: #334155; }}
.state {{ font-weight: 700; white-space: nowrap; }}
.evidence {{ min-width: 210px; }}
.value, .verifier {{ display: inline-block; border-radius: 4px; padding: 2px 6px; font-weight: 700; }}
.value-yes {{ background: #dcfce7; color: #166534; }}
.value-no {{ background: #e5e7eb; color: #374151; }}
.value-unknown {{ background: #fef3c7; color: #92400e; }}
.verifier-supported {{ background: #dbeafe; color: #1d4ed8; }}
.verifier-unsupported {{ background: #fee2e2; color: #991b1b; }}
.verifier-unsure, .verifier-missing {{ background: #fef3c7; color: #92400e; }}
.status {{ color: #475467; font-size: 12px; }}
.flag {{ display: inline-block; margin: 2px 2px 0 0; padding: 2px 5px; border-radius: 4px; background: #ffe4e6; color: #9f1239; font-size: 12px; }}
.muted {{ color: #667085; font-size: 12px; margin-top: 3px; }}
textarea {{ width: 100%; min-height: 58px; box-sizing: border-box; margin-top: 10px; border: 1px solid #cfd6e0; border-radius: 6px; padding: 8px; resize: vertical; }}
.card.hidden {{ display: none; }}
@media (max-width: 1060px) {{ .card {{ grid-template-columns: 1fr; }} header {{ grid-template-columns: 1fr; }} .stats {{ justify-content: flex-start; }} }}
</style>
</head>
<body>
<header>
  <div>
    <div class="title">A_robot_state Label / Image Viewer</div>
    <div class="path">labels: {esc(str(labels_file))}</div>
    <div class="path">html: {esc(str(output))}</div>
  </div>
  <div class="stats">{stat_cards(summary)}</div>
  <div class="toolbar">
    <label>Filter
      <select id="filter">
        <option value="all">全部图片</option>
        <option value="review">只看 review_required</option>
        <option value="unsupported">只看 verifier unsupported/unsure</option>
        <option value="unknown">只看 unknown value</option>
      </select>
    </label>
    <button type="button" id="export">导出人工抽样 JSON</button>
  </div>
</header>
<main>
{''.join(cards)}
</main>
<script>
const storageKey = 'robot_state_label_viewer:' + location.pathname;
const state = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
function save() {{ localStorage.setItem(storageKey, JSON.stringify(state)); }}
function cardPayload(card) {{ return JSON.parse(card.querySelector('.actions').dataset.payload); }}
function labelText(card) {{ return JSON.stringify(cardPayload(card).labels).toLowerCase(); }}
document.querySelectorAll('.card').forEach(card => {{
  const id = card.dataset.sample + ':' + card.dataset.card;
  const note = card.querySelector('textarea');
  const saved = state[id] || {{}};
  if (saved.note) note.value = saved.note;
  if (saved.vote) card.querySelectorAll('button[data-vote]').forEach(b => b.classList.toggle('active', b.dataset.vote === saved.vote));
  card.querySelectorAll('button[data-vote]').forEach(button => button.addEventListener('click', () => {{
    state[id] = state[id] || {{}};
    state[id].vote = button.dataset.vote;
    state[id].sample = card.dataset.sample;
    state[id].payload = cardPayload(card);
    card.querySelectorAll('button[data-vote]').forEach(b => b.classList.toggle('active', b === button));
    save();
  }}));
  note.addEventListener('input', () => {{
    state[id] = state[id] || {{}};
    state[id].sample = card.dataset.sample;
    state[id].note = note.value;
    state[id].payload = cardPayload(card);
    save();
  }});
}});
document.getElementById('filter').addEventListener('change', event => {{
  const mode = event.target.value;
  document.querySelectorAll('.card').forEach(card => {{
    const text = labelText(card);
    let show = true;
    if (mode === 'review') show = text.includes('review_required');
    if (mode === 'unsupported') show = text.includes('unsupported') || text.includes('unsure');
    if (mode === 'unknown') show = text.includes('"state_value":"unknown"') || text.includes('"state_value": "unknown"');
    card.classList.toggle('hidden', !show);
  }});
}});
document.getElementById('export').addEventListener('click', () => {{
  const blob = new Blob([JSON.stringify(Object.values(state), null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'robot_state_manual_sample_export.json';
  a.click();
  URL.revokeObjectURL(a.href);
}});
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")


def pick_port(host: str, preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred_port))
            return preferred_port
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def serve(directory: Path, filename: str, host: str, port: int) -> None:
    os.chdir(directory)
    handler = SimpleHTTPRequestHandler
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/{filename}"
    print(f"Serving: {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Build and optionally serve an A_robot_state label/image comparison page."
    )
    parser.add_argument("--run-dir", type=Path, help="GSA run directory. Default: latest data/output/gsa_v0 run.")
    parser.add_argument("--labels-file", type=Path, help="Direct path to A_robot_state.jsonl. Overrides --run-dir.")
    parser.add_argument("--output", type=Path, help="Output HTML path. Default: <run-dir>/reports/robot_state_label_viewer.html")
    parser.add_argument("--limit", type=int, help="Only render the first N images after grouping.")
    parser.add_argument("--no-embed-images", action="store_true", help="Use file:// image paths instead of embedding resized images.")
    parser.add_argument("--max-image-side", type=int, default=1100, help="Max embedded image side length. Default: 1100.")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="Embedded JPEG quality. Default: 82.")
    parser.add_argument("--no-serve", action="store_true", help="Only generate HTML; do not start local HTTP server.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host. Default: 127.0.0.1.")
    parser.add_argument("--port", type=int, default=10309, help="Preferred HTTP port. Default: 10309.")
    parser.set_defaults(base_dir=root / "data" / "output" / "gsa_v0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels_file:
        labels_file = args.labels_file.resolve()
        run_dir = labels_file.parents[2] if len(labels_file.parents) >= 3 else labels_file.parent
    else:
        run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir(args.base_dir).resolve()
        labels_file = run_dir / "labels" / "A_robot_state" / "A_robot_state.jsonl"

    if not labels_file.exists():
        raise FileNotFoundError(f"Label file not found: {labels_file}")

    output = args.output.resolve() if args.output else run_dir / "reports" / "robot_state_label_viewer.html"
    build_html(
        labels_file=labels_file,
        output=output,
        limit=args.limit,
        embed_images=not args.no_embed_images,
        max_side=args.max_image_side,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"Generated: {output}")

    if args.no_serve:
        return
    port = pick_port(args.host, args.port)
    serve(output.parent.resolve(), output.name, args.host, port)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
