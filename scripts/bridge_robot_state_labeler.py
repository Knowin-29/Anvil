#!/usr/bin/env python3
"""Bridge A_robot_state manual labeling UI.

Browse Bridge wrist images and record human labels. Decisions are appended to
review_decisions/bridge_review_decisions.jsonl under the selected GSA run.

Usage:
    cd /path/to/workspace/foundation/contributor/Rex-Embodied-a-robot-state
    export PYTHONPATH=.
    python scripts/bridge_robot_state_labeler.py --run-id gsa_v0_1780219771

Then open http://127.0.0.1:8765 (use SSH port forwarding if remote).
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request, send_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.taxonomy import ARobotStateSubtype

logger = logging.getLogger("bridge_robot_state_labeler")

STATE_TYPES = [s.value for s in ARobotStateSubtype]
STATE_VALUES = ["yes", "no"]

app = Flask(__name__)

RUN_DIR: Path = Path()
DECISIONS_PATH: Path = Path()
SAMPLES: List[Dict[str, Any]] = []
DECISIONS: Dict[str, Dict[str, Any]] = {}
REVIEWER_DEFAULT = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    if run_id:
        run_dir = output_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    runs = sorted(output_root.glob("gsa_v0_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No runs under {output_root}")
    return runs[-1]


def load_bridge_samples(run_dir: Path) -> List[Dict[str, Any]]:
    labels_path = run_dir / "labels/A_robot_state/A_robot_state.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    samples: List[Dict[str, Any]] = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            label = json.loads(line)
            if label.get("source_dataset") != "bridge_data_v2":
                continue
            image_path = label.get("image_path") or ""
            if not image_path or not Path(image_path).exists():
                logger.warning("Skip missing image: %s", label.get("sample_id"))
                continue
            samples.append(
                {
                    "sample_id": label.get("sample_id"),
                    "image_path": image_path,
                    "state_type": label.get("state_type"),
                    "state_value": label.get("state_value"),
                    "camera_view": label.get("camera_view", "wrist"),
                }
            )
    samples.sort(key=lambda s: s["sample_id"])
    return samples


def load_decisions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    decisions: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id:
                decisions[sample_id] = row
    return decisions


def append_decision(path: Path, decision: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")


def next_pending_index(start: int = 0) -> int:
    for idx in range(start, len(SAMPLES)):
        sid = SAMPLES[idx]["sample_id"]
        if sid not in DECISIONS:
            return idx
    for idx in range(0, start):
        sid = SAMPLES[idx]["sample_id"]
        if sid not in DECISIONS:
            return idx
    return max(0, len(SAMPLES) - 1)


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bridge A_robot_state 标注</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f1419;color:#e6edf3;min-height:100vh}
.top{background:#161b22;border-bottom:1px solid #30363d;padding:14px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.top h1{font-size:18px;color:#58a6ff}
.meta{color:#8b949e;font-size:13px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;padding:16px;max-width:1800px;margin:0 auto}
@media(max-width:960px){.layout{grid-template-columns:1fr}}
.panel{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px}
.img-wrap{background:#010409;border-radius:8px;min-height:420px;display:flex;align-items:center;justify-content:center;padding:8px}
.img-wrap img{max-width:100%;max-height:72vh;border-radius:6px}
.section{margin-bottom:16px}
.section h3{font-size:12px;color:#8b949e;text-transform:uppercase;margin-bottom:8px;letter-spacing:.04em}
.btn-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.btn,.value-btn,.nav-btn{border:1px solid #30363d;background:#21262d;color:#e6edf3;border-radius:8px;padding:10px 12px;cursor:pointer;font-size:13px;text-align:left}
.btn:hover,.value-btn:hover,.nav-btn:hover{border-color:#58a6ff}
.btn.active,.value-btn.active{background:#1f3a5f;border-color:#58a6ff}
.value-row{display:flex;gap:8px}
.value-btn{flex:1;text-align:center;font-weight:600}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.primary{background:#238636;border-color:#238636}
.primary:hover{background:#2ea043}
.warn{background:#6e3b12;border-color:#bd561d}
.skip{background:#30363d}
input[type=text]{width:100%;background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:8px 10px}
kbd{background:#30363d;border-radius:4px;padding:1px 5px;font-size:11px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}
.stat{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:10px;text-align:center}
.stat .n{font-size:20px;font-weight:700;color:#58a6ff}
.stat .l{font-size:11px;color:#8b949e;margin-top:2px}
.hint{font-size:12px;color:#8b949e;line-height:1.6;margin-top:12px}
.done{color:#3fb950;font-weight:600}
</style>
</head>
<body>
<div class="top">
  <h1>Bridge 手腕图 · A_robot_state 标注</h1>
  <div class="meta">Run: {{ run_id }} · 共 {{ total }} 张 · 已标 {{ done }} · 待标 {{ pending }}</div>
</div>
<div class="layout">
  <div class="panel">
    <div class="img-wrap"><img id="image" src="/api/image/{{ index }}" alt="bridge image"></div>
    <div class="hint" style="margin-top:12px">
      <div><strong id="sample-id">{{ sample.sample_id }}</strong></div>
      <div id="image-path" style="word-break:break-all;margin-top:4px">{{ sample.image_path }}</div>
    </div>
  </div>
  <div class="panel">
    <div class="stats">
      <div class="stat"><div class="n" id="stat-index">{{ index + 1 }}</div><div class="l">当前序号</div></div>
      <div class="stat"><div class="n" id="stat-done">{{ done }}</div><div class="l">已完成</div></div>
      <div class="stat"><div class="n" id="stat-pending">{{ pending }}</div><div class="l">待标注</div></div>
    </div>

    <div class="section">
      <h3>State Type <kbd>1-5</kbd></h3>
      <div class="btn-grid" id="type-buttons">
        {% for st in state_types %}
        <button class="btn type-btn" data-type="{{ st }}">{{ loop.index }}. {{ st }}</button>
        {% endfor %}
      </div>
    </div>

    <div class="section">
      <h3>State Value <kbd>y</kbd> / <kbd>n</kbd></h3>
      <div class="value-row">
        <button class="value-btn" data-value="yes">Yes</button>
        <button class="value-btn" data-value="no">No</button>
      </div>
    </div>

    <div class="section">
      <h3>Target Object（可选）</h3>
      <input type="text" id="target-object" placeholder="例如 mug, block, towel">
    </div>

    <div class="section">
      <h3>Reviewer</h3>
      <input type="text" id="reviewer" value="{{ reviewer }}">
    </div>

    <div class="actions">
      <button class="nav-btn" id="btn-prev">← 上一张</button>
      <button class="nav-btn" id="btn-next">下一张 →</button>
      <button class="primary nav-btn" id="btn-save">保存并下一张 <kbd>Enter</kbd></button>
      <button class="skip nav-btn" id="btn-skip">跳过 <kbd>s</kbd></button>
    </div>

    <div class="hint">
      快捷键：1-5 选 type，y/n 选 value，Enter 保存，s 跳过，←/→ 翻页。<br>
      保存后写入 <code>{{ decisions_name }}</code>
    </div>
    <div id="status" class="hint" style="margin-top:8px"></div>
  </div>
</div>
<script>
const stateTypes = {{ state_types_json | safe }};
let index = {{ index }};
let selectedType = "";
let selectedValue = "";

function setStatus(msg, ok=false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = ok ? "hint done" : "hint";
}

function highlightButtons() {
  document.querySelectorAll(".type-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.type === selectedType);
  });
  document.querySelectorAll(".value-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.value === selectedValue);
  });
}

async function loadSample(idx) {
  const res = await fetch(`/api/sample/${idx}`);
  const data = await res.json();
  index = data.index;
  selectedType = data.decision?.state_type || "";
  selectedValue = data.decision?.state_value || "";
  document.getElementById("image").src = `/api/image/${index}?t=${Date.now()}`;
  document.getElementById("sample-id").textContent = data.sample.sample_id;
  document.getElementById("image-path").textContent = data.sample.image_path;
  document.getElementById("stat-index").textContent = index + 1;
  document.getElementById("stat-done").textContent = data.done;
  document.getElementById("stat-pending").textContent = data.pending;
  document.getElementById("target-object").value = data.decision?.target_object_class || "";
  highlightButtons();
  setStatus(data.decision ? `已有标注: ${data.decision.review_decision}` : "待标注");
}

async function save(decision) {
  const payload = {
    index,
    review_decision: decision,
    state_type: selectedType,
    state_value: selectedValue,
    target_object_class: document.getElementById("target-object").value.trim(),
    reviewer: document.getElementById("reviewer").value.trim(),
    notes: "",
  };
  const res = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    setStatus(data.error || "保存失败");
    return false;
  }
  setStatus(`已保存: ${data.sample_id}`, true);
  await loadSample(data.next_index);
  return true;
}

document.querySelectorAll(".type-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    selectedType = btn.dataset.type;
    highlightButtons();
  });
});
document.querySelectorAll(".value-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    selectedValue = btn.dataset.value;
    highlightButtons();
  });
});
document.getElementById("btn-save").addEventListener("click", () => save("approve"));
document.getElementById("btn-skip").addEventListener("click", () => save("skip"));
document.getElementById("btn-prev").addEventListener("click", () => loadSample(Math.max(0, index - 1)));
document.getElementById("btn-next").addEventListener("click", () => loadSample(Math.min({{ total - 1 }}, index + 1)));

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key >= "1" && e.key <= "5") {
    selectedType = stateTypes[parseInt(e.key, 10) - 1] || "";
    highlightButtons();
  } else if (e.key === "y" || e.key === "Y") {
    selectedValue = "yes";
    highlightButtons();
  } else if (e.key === "n" || e.key === "N") {
    selectedValue = "no";
    highlightButtons();
  } else if (e.key === "Enter") {
    e.preventDefault();
    save("approve");
  } else if (e.key === "s" || e.key === "S") {
    save("skip");
  } else if (e.key === "ArrowLeft") {
    loadSample(Math.max(0, index - 1));
  } else if (e.key === "ArrowRight") {
    loadSample(Math.min({{ total - 1 }}, index + 1));
  }
});

loadSample(index);
</script>
</body></html>"""


@app.route("/")
def index_page():
    idx = request.args.get("index", type=int)
    if idx is None:
        idx = next_pending_index(0)
    idx = max(0, min(idx, len(SAMPLES) - 1 if SAMPLES else 0))
    done = len(DECISIONS)
    pending = len(SAMPLES) - done
    sample = SAMPLES[idx] if SAMPLES else {"sample_id": "", "image_path": ""}
    return render_template_string(
        HTML,
        run_id=RUN_DIR.name,
        total=len(SAMPLES),
        done=done,
        pending=pending,
        index=idx,
        sample=sample,
        state_types=STATE_TYPES,
        state_types_json=json.dumps(STATE_TYPES),
        reviewer=REVIEWER_DEFAULT,
        decisions_name=DECISIONS_PATH.name,
    )


@app.route("/api/sample/<int:idx>")
def api_sample(idx: int):
    idx = max(0, min(idx, len(SAMPLES) - 1))
    sample = SAMPLES[idx]
    decision = DECISIONS.get(sample["sample_id"])
    return jsonify(
        {
            "index": idx,
            "sample": sample,
            "decision": decision,
            "done": len(DECISIONS),
            "pending": len(SAMPLES) - len(DECISIONS),
        }
    )


@app.route("/api/image/<int:idx>")
def api_image(idx: int):
    if idx < 0 or idx >= len(SAMPLES):
        return "Not found", 404
    image_path = Path(SAMPLES[idx]["image_path"])
    if not image_path.exists():
        return "Image missing", 404
    mime, _ = mimetypes.guess_type(str(image_path))
    return send_file(image_path, mimetype=mime or "image/png")


@app.route("/api/save", methods=["POST"])
def api_save():
    global DECISIONS
    payload = request.get_json(force=True) or {}
    idx = int(payload.get("index", 0))
    idx = max(0, min(idx, len(SAMPLES) - 1))
    sample = SAMPLES[idx]
    decision_type = payload.get("review_decision", "approve")

    if decision_type == "approve":
        state_type = payload.get("state_type", "")
        state_value = payload.get("state_value", "")
        if state_type not in STATE_TYPES:
            return jsonify({"error": "请选择 state_type（1-5）"}), 400
        if state_value not in STATE_VALUES:
            return jsonify({"error": "请选择 state_value（y/n）"}), 400

    row = {
        "sample_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "review_decision": decision_type,
        "state_type": payload.get("state_type", ""),
        "state_value": payload.get("state_value", ""),
        "target_object_class": payload.get("target_object_class") or None,
        "reviewer": payload.get("reviewer") or REVIEWER_DEFAULT,
        "notes": payload.get("notes", ""),
        "created_at": _utc_now(),
    }
    append_decision(DECISIONS_PATH, row)
    DECISIONS[sample["sample_id"]] = row
    next_idx = next_pending_index(idx + 1)
    return jsonify({"ok": True, "sample_id": sample["sample_id"], "next_index": next_idx})


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge A_robot_state labeling UI")
    parser.add_argument("--output-root", default="data/output/gsa_v0")
    parser.add_argument("--run-id", default="", help="GSA run id; latest if empty")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reviewer", default="", help="Default reviewer name")
    parser.add_argument(
        "--decisions-file",
        default="",
        help="Override decisions jsonl path (default: {run}/review_decisions/bridge_review_decisions.jsonl)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    global RUN_DIR, DECISIONS_PATH, SAMPLES, DECISIONS, REVIEWER_DEFAULT
    RUN_DIR = resolve_run_dir(ROOT / args.output_root, args.run_id)
    DECISIONS_PATH = (
        Path(args.decisions_file)
        if args.decisions_file
        else RUN_DIR / "review_decisions/bridge_review_decisions.jsonl"
    )
    REVIEWER_DEFAULT = args.reviewer

    SAMPLES = load_bridge_samples(RUN_DIR)
    DECISIONS = load_decisions(DECISIONS_PATH)
    if not SAMPLES:
        logger.error("No bridge samples with existing images found in %s", RUN_DIR)
        return 1

    logger.info("Run: %s", RUN_DIR.name)
    logger.info("Bridge samples: %d", len(SAMPLES))
    logger.info("Existing decisions: %d", len(DECISIONS))
    logger.info("Decisions file: %s", DECISIONS_PATH)
    logger.info("Open http://127.0.0.1:%d (SSH: ssh -L %d:127.0.0.1:%d ...)", args.port, args.port, args.port)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
