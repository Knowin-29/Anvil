"""Rex-Embodied Gradio GUI — 数据飞轮 / 推理 / 抓取 / 标注。"""

import glob
import json
import os
import sys
import tempfile
import traceback
from html import escape
from pathlib import Path

import gradio as gr
import requests

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

API_BASE = os.getenv("REX_SERVICE_URL", "http://127.0.0.1:8101")
DEBUG_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "data", "input", "debug")
DEFAULT_IMAGE = DEBUG_IMAGE_DIR if os.path.isdir(DEBUG_IMAGE_DIR) else os.path.join(PROJECT_ROOT, "data", "input", "images")
FLYWHEEL_OUTPUT = os.path.join(PROJECT_ROOT, "data", "output", "flywheel")
REPORTS_OUTPUT = os.path.join(PROJECT_ROOT, "data", "output", "reports")

_ENV_PATH = Path(PROJECT_ROOT) / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# =====================================================================
# 工具函数
# =====================================================================

def _load_jsonl(path: str, max_lines: int = 0) -> list:
    if not path or not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
                if 0 < max_lines <= len(items):
                    break
    return items


def _service_available() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).ok
    except Exception:
        return False


def _post(path: str, payload: dict) -> dict:
    if not _service_available():
        return {"error": "后端服务未启动。请先运行: python -m services.app"}
    resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=1200)
    resp.raise_for_status()
    return resp.json()


def _safe_call(fn) -> str:
    try:
        return json.dumps(fn(), ensure_ascii=False, indent=2)
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "无法连接后端服务", "hint": "请先启动: python -m services.app"}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)


def _round_dir(round_idx: int) -> str:
    return os.path.join(FLYWHEEL_OUTPUT, f"round_{int(round_idx)}")


def _gradio_allowed_paths() -> list[str]:
    candidates = [
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(PROJECT_ROOT, "data", "input"),
        os.path.join(PROJECT_ROOT, "data", "output"),
        os.path.join(PROJECT_ROOT, "data", "data"),
        os.path.join(PROJECT_ROOT, "data", "data", "input"),
        DEBUG_IMAGE_DIR,
        FLYWHEEL_OUTPUT,
        REPORTS_OUTPUT,
        tempfile.gettempdir(),
    ]
    extra = os.getenv("REX_GRADIO_ALLOWED_PATHS", "")
    if extra:
        candidates.extend(path.strip() for path in extra.split(os.pathsep) if path.strip())
    allowed = []
    seen = set()
    for path in candidates:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        allowed.append(normalized)
    return allowed


def _report_files(output_dir: str) -> list[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return []
    files = glob.glob(os.path.join(output_dir, "*.html"))
    return sorted(files, key=lambda path: (os.path.getmtime(path), os.path.basename(path)), reverse=True)


def list_reports(output_dir: str = REPORTS_OUTPUT):
    files = _report_files(output_dir)
    choices = [os.path.basename(path) for path in files]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _render_report_iframe(report_html: str) -> str:
    # gr.HTML injects content via innerHTML, so embedded <script> tags in the
    # report will not execute. Wrapping the full document in iframe srcdoc gives
    # the report a real document context and preserves its JS/CSS behavior.
    escaped = escape(report_html, quote=True)
    return (
        '<iframe '
        'title="pipeline-report" '
        'sandbox="allow-scripts allow-same-origin" '
        'style="width:100%;height:1200px;border:1px solid #d1d5db;border-radius:8px;background:#ffffff;" '
        f'srcdoc="{escaped}"></iframe>'
    )


def load_report_html(report_name: str, output_dir: str = REPORTS_OUTPUT) -> str:
    if not report_name:
        return "<p style='color:#999;text-align:center;padding:40px;'>请选择一个报告文件</p>"
    report_path = report_name if os.path.isabs(report_name) else os.path.join(output_dir, report_name)
    if not os.path.exists(report_path):
        return f"<p style='color:#c00;text-align:center;padding:40px;'>报告不存在: {report_path}</p>"
    with open(report_path, "r", encoding="utf-8") as f:
        return _render_report_iframe(f.read())


def refresh_reports(output_dir: str = REPORTS_OUTPUT):
    selector = list_reports(output_dir)
    return selector, load_report_html(selector.get("value"), output_dir)


# =====================================================================
# 飞轮 — 输入预览 / API 测试
# =====================================================================

def preview_input_images(image_dir: str, max_images: int):
    if not image_dir or not os.path.isdir(image_dir):
        return [], f"目录不存在: {image_dir}"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in exts])
    total = len(imgs)
    imgs = imgs[:int(max_images)]
    return imgs, f"共 {total} 张图片，预览前 {len(imgs)} 张"


def _format_relation_source(source: str) -> str:
    mapping = {"2d": "2D", "3d": "3D", "2d+3d": "2D+3D"}
    return mapping.get(str(source).lower(), str(source))


def test_apis() -> str:
    try:
        from apps.data_flywheel.captioner import probe_apis, select_best_backend
        results = probe_apis()
        best = select_best_backend(results)
        lines = ["### API 可用性测试\n"]
        for name, info in results.items():
            status = "✅" if info.get("available") else "❌"
            if info.get("available"):
                detail = info.get("sample") or info.get("desc") or "可用"
            else:
                detail = info.get("reason") or info.get("error") or "不可用"
            lines.append(f"- {status} **{name}**: {detail}")
        lines.append(f"\n推荐后端: **{best or '无'}**")
        return "\n".join(lines)
    except Exception as e:
        return f"测试失败: {e}"


def estimate_time_display(max_images, use_api, en_aug, en_cap, en_gr, en_dep, en_sg, en_ref, en_pt, en_ev):
    from apps.data_flywheel.config import FlywheelConfig
    n = max(1, int(max_images or 10))
    steps = FlywheelConfig.estimate_time(
        n_images=n, use_api=bool(use_api),
        enable_augment=bool(en_aug), enable_caption=bool(en_cap),
        enable_grounding=bool(en_gr), enable_depth=bool(en_dep),
        enable_scene_graph=bool(en_sg), enable_referring=bool(en_ref),
        enable_pointing=bool(en_pt), enable_eval=bool(en_ev),
    )
    lines = [f"### ⏱ 预估耗时（{n} 张图，{'API' if use_api else '本地'}模式）\n",
             "| 步骤 | 时间 |", "|------|------|"]
    for name, secs in steps.items():
        t = f"{secs:.1f}s" if secs < 60 else f"{secs / 60:.1f}min"
        b = "**" if name == "合计" else ""
        lines.append(f"| {b}{name}{b} | {b}{t}{b} |")
    return "\n".join(lines)


# =====================================================================
# 飞轮 — 运行
# =====================================================================

def run_flywheel_local(image_dir, max_images, augment, rounds, caption_backend,
                       backend, use_api, detector_models_text, detector_primary, det_verifier_policy,
                       keep_only_verified, en_bbox_single, en_bbox_rel, en_point_single, en_point_rel,
                       en_aug, en_cap, en_gr, en_dep, en_sg, en_ref, en_pt, en_ev):
    try:
        from apps.data_flywheel.config import FlywheelConfig
        from apps.data_flywheel.pipeline import run_pipeline
        cfg = FlywheelConfig()
        if image_dir and image_dir.strip():
            cfg.image_dirs = [image_dir.strip()]
        cfg.rounds = int(rounds)
        cfg.max_images_per_round = int(max_images)
        cfg.augment_per_image = int(augment)
        cfg.filter_adjectives = True
        cfg.backend = backend
        cfg.caption_backend = caption_backend
        cfg.use_api = use_api
        detector_models = [x.strip() for x in (detector_models_text or "").replace("\n", ",").split(",") if x.strip()]
        if detector_models:
            cfg.detector_models = detector_models
        cfg.detector_primary = (detector_primary or "").strip()
        cfg.detection_verifier_policy = det_verifier_policy or "drop"
        cfg.keep_only_verified = bool(keep_only_verified)
        cfg.enable_bbox_single_qa = bool(en_bbox_single)
        cfg.enable_bbox_relational_qa = bool(en_bbox_rel)
        cfg.enable_point_single_qa = bool(en_point_single)
        cfg.enable_point_relational_qa = bool(en_point_rel)
        cfg.enable_augment = en_aug
        cfg.enable_caption = en_cap
        cfg.enable_grounding = en_gr
        cfg.enable_depth_estimation = en_dep
        cfg.enable_scene_graph = en_sg
        cfg.enable_referring = en_ref
        cfg.enable_pointing = en_pt
        cfg.enable_eval = en_ev
        result = run_pipeline(cfg)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2)


# =====================================================================
# 飞轮 — 结果浏览（分类加载）
# =====================================================================

def browse_overview(round_idx):
    rd = _round_dir(round_idx)
    if not os.path.isdir(rd):
        return "未找到该轮结果。请先运行飞轮。"
    sg = _load_jsonl(os.path.join(rd, "scene_graph.jsonl"))
    qa = _load_jsonl(os.path.join(rd, "spatial_qa.jsonl"))
    gr_data = _load_jsonl(os.path.join(rd, "grounding.jsonl"))
    dep = _load_jsonl(os.path.join(rd, "depth_info.jsonl"))
    ref = _load_jsonl(os.path.join(rd, "referring.jsonl"))
    pt = _load_jsonl(os.path.join(rd, "pointing.jsonl"))
    hard = _load_jsonl(os.path.join(rd, "hard_samples.jsonl"))
    exp = _load_jsonl(os.path.join(rd, "train_export.jsonl"))
    n_obj = sum(len(s.get("objects", [])) for s in sg)
    n_rel = sum(len(s.get("relations", [])) for s in sg)
    src, pred_cnt, qa_types = {}, {}, {}
    for s in sg:
        for r in s.get("relations", []):
            src[r.get("source", "2d")] = src.get(r.get("source", "2d"), 0) + 1
            pred = r["predicate"]
            pred_cnt.setdefault(pred, {})
            pred_cnt[pred][r.get("source", "2d")] = pred_cnt[pred].get(r.get("source", "2d"), 0) + 1
    for q in qa:
        t = q.get("type", "?")
        qa_types[t] = qa_types.get(t, 0) + 1
    dep_ok = sum(1 for d in dep if d.get("depth_valid"))
    detector_compare_path = os.path.join(rd, "detector_compare.json")
    detector_md = ""
    if os.path.exists(detector_compare_path):
        try:
            dc = json.load(open(detector_compare_path, "r", encoding="utf-8"))
            detector_md = "\n#### 🤖 检测器 A/B 对比\n| 检测器 | avg_iou | pass_rate | boxes | avg_boxes/phrase |\n|---|---:|---:|---:|---:|\n"
            for tag, info in dc.get("detectors", {}).items():
                detector_md += (
                    f"| {tag}{' (primary)' if tag == dc.get('primary') else ''} | "
                    f"{info.get('avg_iou', 0)} | {info.get('pass_rate', 0)} | {info.get('total_boxes', 0)} | "
                    f"{info.get('avg_boxes_per_phrase', 0)} |\n"
                )
        except Exception:
            detector_md = ""
    eval_path = os.path.join(rd, "eval_report.json")
    avg_iou, pass_rate = "N/A", "N/A"
    if os.path.exists(eval_path):
        ev = json.load(open(eval_path))
        avg_iou = ev.get("avg_iou", "N/A")
        pass_rate = ev.get("pass_rate", "N/A")
    timings_text = ""
    summary_path = os.path.join(FLYWHEEL_OUTPUT, "flywheel_summary.json")
    if os.path.exists(summary_path):
        sm = json.load(open(summary_path))
        for rpt in sm.get("reports", []):
            if rpt.get("round") == int(round_idx):
                st = rpt.get("step_timings", {})
                if st:
                    timings_text = "\n#### ⏱ 实际耗时\n| 步骤 | 耗时 |\n|------|------|\n"
                    for k, v in st.items():
                        timings_text += f"| {k} | {v:.2f}s |\n"
                break
    md = f"""### 📊 Round {int(round_idx)} 总览
| 指标 | 数值 |
|------|------|
| Grounding | {len(gr_data)} 张图 |
| 检测物体 | {n_obj} 个 |
| 空间关系 | {n_rel} 条 (2D: {src.get('2d',0)}, 3D: {src.get('3d',0)}) |
| Spatial QA | {len(qa)} 条 |
| 深度估计 | {dep_ok}/{len(dep)} 成功 |
| Referring | {len(ref)} 条 |
| Pointing | {len(pt)} 条 |
| 训练导出 | {len(exp)} 条 |
| avg_iou | {avg_iou} |
| pass_rate | {pass_rate} |
| 难例 | {len(hard)} 条 |
"""
    if pred_cnt:
        md += "\n#### 🔗 空间关系分布\n| 谓词 | 数量 | 来源 |\n|------|------|------|\n"
        for p, source_counts in sorted(pred_cnt.items(), key=lambda x: -sum(x[1].values())):
            count = sum(source_counts.values())
            source_labels = [
                _format_relation_source(source)
                for source, source_count in sorted(source_counts.items())
                if source_count
            ]
            source_label = " / ".join(source_labels) if source_labels else "2D"
            md += f"| {p} | {count} | {source_label} |\n"
    if qa_types:
        md += "\n#### ❓ QA 类型分布\n| 类型 | 数量 |\n|------|------|\n"
        for t, c in sorted(qa_types.items(), key=lambda x: -x[1]):
            md += f"| {t} | {c} |\n"
    md += detector_md + timings_text
    return md


def browse_visualizations(round_idx):
    rd = _round_dir(round_idx)
    vis_dir = os.path.join(rd, "visualizations")
    images = sorted(glob.glob(os.path.join(vis_dir, "*.jpg")) + glob.glob(os.path.join(vis_dir, "*.png")))
    if not images:
        images = _generate_vis(rd)
    return images if images else []


def _generate_vis(round_dir: str) -> list:
    sg_path = os.path.join(round_dir, "scene_graph.jsonl")
    gr_path = os.path.join(round_dir, "grounding.jsonl")
    dep_path = os.path.join(round_dir, "depth_info.jsonl")
    if not os.path.exists(sg_path):
        return []
    try:
        import numpy as np
        from apps.data_flywheel.visualizer import (
            draw_grounding, draw_scene_graph, draw_annotation,
            draw_depth_map, draw_depth_with_boxes, save_visualization,
        )
    except ImportError:
        return []
    vis_dir = os.path.join(round_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    sg_data, gr_data, dep_data = _load_jsonl(sg_path), _load_jsonl(gr_path), _load_jsonl(dep_path)
    gr_map = {g["image"]: g for g in gr_data}
    dep_map = {d["image"]: d for d in dep_data}
    generated = []
    for sg in sg_data:
        img_path = sg["image"]
        if not os.path.exists(img_path):
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        objs, rels = sg.get("objects", []), sg.get("relations", [])
        gr = gr_map.get(img_path, {})
        gr_objs = []
        for p in gr.get("phrases", []):
            for box in p.get("boxes", []):
                gr_objs.append({"name": p["phrase"], "bbox": box["bbox"],
                                "confidence": box.get("confidence", 0), "id": len(gr_objs)})
        if gr_objs:
            generated.append(save_visualization(draw_grounding(img_path, gr_objs),
                                                os.path.join(vis_dir, f"{stem}_grounding.jpg")))
        if objs:
            generated.append(save_visualization(draw_annotation(img_path, objs),
                                                os.path.join(vis_dir, f"{stem}_annotation.jpg")))
        if rels:
            generated.append(save_visualization(draw_scene_graph(img_path, objs, rels),
                                                os.path.join(vis_dir, f"{stem}_scene_graph.jpg")))
        di = dep_map.get(img_path)
        if di and di.get("xyz_map_path") and os.path.exists(di["xyz_map_path"]):
            try:
                import numpy as np
                xyz = np.load(di["xyz_map_path"])
                generated.append(save_visualization(draw_depth_map(xyz, img_path, 0.5),
                                                    os.path.join(vis_dir, f"{stem}_depth.jpg")))
                generated.append(save_visualization(draw_depth_with_boxes(xyz, objs, img_path),
                                                    os.path.join(vis_dir, f"{stem}_depth_bbox.jpg")))
            except Exception:
                pass
    return sorted(generated)


def browse_scene_graph(round_idx, max_show):
    rd = _round_dir(round_idx)
    data = _load_jsonl(os.path.join(rd, "scene_graph.jsonl"), max_lines=int(max_show))
    if not data:
        return "无 Scene Graph 数据"
    parts = []
    for i, sg in enumerate(data):
        img = os.path.basename(sg.get("image", ""))
        objs, rels = sg.get("objects", []), sg.get("relations", [])
        parts.append(f"#### 🖼 {i+1}. {img} — {len(objs)} 物体, {len(rels)} 关系\n")
        parts.append("| ID | 名称 | 置信度 | 深度 | pick | push | stack | container | surface |")
        parts.append("|---|---|---|---|---|---|---|---|---|")
        for o in objs:
            d = f"{o.get('mean_depth',0):.0f}" if o.get('mean_depth') else "-"
            parts.append(f"| {o['id']} | {o['name']} | {o.get('confidence',''):.2f} | {d} "
                         f"| {'✓' if o.get('can_pick') else '✗'} | {'✓' if o.get('can_push') else '✗'} "
                         f"| {'✓' if o.get('can_stack') else '✗'} | {'✓' if o.get('is_container') else '✗'} "
                         f"| {'✓' if o.get('is_surface') else '✗'} |")
        parts.append("\n| Subject | Predicate | Object | Source | Conf |")
        parts.append("|---|---|---|---|---|")
        for r in rels[:30]:
            sn = objs[r["subject"]]["name"] if r["subject"] < len(objs) else "?"
            on = objs[r["object"]]["name"] if r["object"] < len(objs) else "?"
            parts.append(f"| {sn} (#{r['subject']}) | **{r['predicate']}** | {on} (#{r['object']}) "
                         f"| {r.get('source','2d')} | {r.get('confidence','')} |")
        if len(rels) > 30:
            parts.append(f"| ... | 还有 {len(rels)-30} 条 | ... | | |")
        parts.append("\n---\n")
    return "\n".join(parts)


def browse_spatial_qa(round_idx, max_show):
    rd = _round_dir(round_idx)
    data = _load_jsonl(os.path.join(rd, "spatial_qa.jsonl"), max_lines=int(max_show))
    if not data:
        return "无 Spatial QA 数据"
    parts = []
    for i, qa in enumerate(data):
        img = os.path.basename(qa.get("image", ""))
        qtype = qa.get("type", "?")
        neg = " 🔴 负例" if qa.get("is_negative") else ""
        parts.append(f"**{i+1}. [{qtype}]{neg}** — {img}")
        parts.append(f"- **Q:** {qa.get('question', '')}")
        parts.append(f"- **A:** {qa.get('answer', '')}")
        if qa.get("answer_format"):
            parts.append(f"- **答案格式:** `{qa.get('answer_format')}`")
        if qa.get("answer_bbox") is not None:
            parts.append(f"- **answer_bbox:** `{qa.get('answer_bbox')}`")
        if qa.get("answer_point") is not None:
            parts.append(f"- **answer_point:** `{qa.get('answer_point')}`")
        if "verified" in qa:
            status = "✅" if qa.get("verified") else "❌"
            parts.append(f"- **Verifier:** {status} score={qa.get('verify_score', 0)} reasons={qa.get('verify_reasons', [])}")
        if qa.get("options"):
            parts.append(f"- **选项:** {qa['options']}")
        if qa.get("thinking"):
            t = qa["thinking"]
            parts.append(f"- 💭 *{t[:200]}{'...' if len(t) > 200 else ''}*")
        parts.append("")
    return "\n".join(parts)


def browse_depth(round_idx, max_show):
    rd = _round_dir(round_idx)
    data = _load_jsonl(os.path.join(rd, "depth_info.jsonl"), max_lines=int(max_show))
    if not data:
        return "无深度估计数据"
    parts = []
    for i, d in enumerate(data):
        img = os.path.basename(d.get("image", ""))
        valid = "✅" if d.get("depth_valid") else "❌"
        xyz_path = d.get("xyz_map_path", "")
        has_file = "✅" if xyz_path and os.path.exists(xyz_path) else "❌"
        parts.append(f"#### {i+1}. {img} — 深度: {valid} | xyz文件: {has_file}\n")
        objs = d.get("objects", [])
        if objs:
            parts.append("| 物体 | 深度 | 3D中心 | 3D尺寸 | valid |")
            parts.append("|---|---|---|---|---|")
            for o in objs:
                c = o.get("center_3d", [0, 0, 0])
                s = o.get("size_3d", [0, 0, 0])
                parts.append(f"| {o.get('name','')} | {o.get('mean_depth',0):.0f} "
                             f"| ({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) "
                             f"| ({s[0]:.0f},{s[1]:.0f},{s[2]:.0f}) | {'✓' if o.get('valid') else '✗'} |")
        parts.append("\n---\n")
    return "\n".join(parts)


def browse_grounding(round_idx, max_show):
    rd = _round_dir(round_idx)
    data = _load_jsonl(os.path.join(rd, "grounding.jsonl"), max_lines=int(max_show))
    if not data:
        return "无 Grounding 数据"
    parts = []
    for i, g in enumerate(data):
        img = os.path.basename(g.get("image", ""))
        phrases = g.get("phrases", [])
        n_boxes = sum(len(p.get("boxes", [])) for p in phrases)
        parts.append(f"#### {i+1}. {img} — {len(phrases)} 短语, {n_boxes} 框\n")
        parts.append("| 短语 | 框数 | 最高置信度 | BBox 示例 |")
        parts.append("|---|---|---|---|")
        for p in phrases:
            boxes = p.get("boxes", [])
            best = max((b.get("confidence", 0) for b in boxes), default=0)
            bbox_str = str([round(v, 1) for v in boxes[0]["bbox"]]) if boxes else "-"
            parts.append(f"| {p['phrase']} | {len(boxes)} | {best:.2f} | {bbox_str} |")
        parts.append("")
    return "\n".join(parts)


def browse_referring(round_idx, max_show):
    rd = _round_dir(round_idx)
    data = _load_jsonl(os.path.join(rd, "referring.jsonl"), max_lines=int(max_show))
    if not data:
        return "无 Referring 数据"
    return "\n".join(f"**{i+1}.** {os.path.basename(r.get('image',''))}: "
                     f"`{r.get('expression','')}` → bbox `{r.get('gt_bbox',[])}`"
                     for i, r in enumerate(data))


def browse_pointing(round_idx, max_show):
    rd = _round_dir(round_idx)
    data = _load_jsonl(os.path.join(rd, "pointing.jsonl"), max_lines=int(max_show))
    if not data:
        return "无 Pointing 数据"
    return "\n".join(f"**{i+1}.** {os.path.basename(p.get('image',''))}: "
                     f"`{p.get('phrase','')}` → point `{p.get('point',[])}`"
                     for i, p in enumerate(data))


def browse_hard_samples(round_idx, max_show):
    rd = _round_dir(round_idx)
    data = _load_jsonl(os.path.join(rd, "hard_samples.jsonl"), max_lines=int(max_show))
    if not data:
        return "无难例数据"
    return "\n".join(f"**{i+1}.** {os.path.basename(h.get('image',''))}: "
                     f"`{h.get('phrase','')}` — reason: **{h.get('reason','?')}** | iou: {h.get('best_iou','N/A')}"
                     for i, h in enumerate(data))


def browse_export(round_idx, max_show):
    rd = _round_dir(round_idx)
    data = _load_jsonl(os.path.join(rd, "train_export.jsonl"), max_lines=int(max_show))
    if not data:
        return "无导出数据"
    type_cnt = {}
    for d in data:
        t = d.get("type", "?")
        type_cnt[t] = type_cnt.get(t, 0) + 1
    parts = ["#### 导出类型分布\n| 类型 | 数量 |\n|---|---|"]
    for t, c in sorted(type_cnt.items(), key=lambda x: -x[1]):
        parts.append(f"| {t} | {c} |")
    parts.append(f"\n展示前 {len(data)} 条:\n")
    for i, d in enumerate(data):
        img = os.path.basename(d.get("image", ""))
        dtype = d.get("type", "?")
        if dtype.startswith("spatial_qa"):
            parts.append(f"**{i+1}. [{dtype}]** {img}: Q: `{d.get('question','')[:80]}` A: `{d.get('answer','')[:50]}`")
        elif dtype == "grounding":
            parts.append(f"**{i+1}. [{dtype}]** {img}: {d.get('caption','')[:80]}")
        elif dtype == "referring":
            parts.append(f"**{i+1}. [{dtype}]** {img}: `{d.get('expression','')[:80]}`")
        else:
            parts.append(f"**{i+1}. [{dtype}]** {img}")
    return "\n".join(parts)


# =====================================================================
# 推理 / 抓取 / 标注
# =====================================================================

def run_inference(image_path, task, categories, backend):
    payload = {
        "image_path": image_path,
        "backend": backend,
        "categories": [c.strip() for c in categories.split(",") if c.strip()],
    }
    if task == "detection":
        path = "/infer/detection"
    elif task == "pointing":
        path = "/infer/pointing"
    elif task == "segment":
        path = "/infer/segment"
    else:
        path = "/infer/detection"
    return _safe_call(lambda: _post(path, payload))


def run_grasp(image_path, target, strategy, backend):
    _ = strategy
    return _safe_call(lambda: _post("/grasp/optimize", {
        "image_path": image_path,
        "target_object": target,
        "candidate_objects": [],
        "robot_exec_mode": "mock",
        "backend": backend,
    }))


def run_auto_label(image_dir, categories, backend, max_images):
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    image_paths, _ = preview_input_images(image_dir, max_images)
    caption = ", ".join(cats)
    items = [{"image_path": path, "caption": caption} for path in image_paths]
    return _safe_call(lambda: _post("/engine/grounding", {
        "items": items,
        "backend": backend,
    }))


# =====================================================================
# UI 构建
# =====================================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Rex-Embodied") as demo:
        gr.Markdown("# 🤖 Rex-Embodied\n面向真实机械臂场景的 Rex-Omni 落地工程 — 感知 · 决策 · 标注 · 数据飞轮")

        # ==============================================================
        # Tab 1: 数据飞轮（首页）
        # ==============================================================
        with gr.Tab("🔄 数据飞轮"):
            gr.Markdown(
                "### 数据飞轮 — 图片 → Grounding / Scene Graph / Spatial QA / Referring / Pointing\n"
                "10 步自动化 pipeline，支持 2D+3D 分层空间关系推理。"
                " 💡 取消不需要的步骤可大幅加速。"
            )

            # ---- 输入 ----
            with gr.Accordion("📂 输入图片", open=True):
                with gr.Row():
                    fw_image_dir = gr.Textbox(label="图片目录", value=DEFAULT_IMAGE, scale=3)
                    fw_max_images = gr.Number(label="图片数", value=10, precision=0, scale=1)
                fw_preview_btn = gr.Button("预览", size="sm")
                fw_input_gallery = gr.Gallery(label="输入预览", columns=6, height=160)
                fw_input_info = gr.Markdown()
                fw_preview_btn.click(preview_input_images, [fw_image_dir, fw_max_images],
                                     [fw_input_gallery, fw_input_info])

            # ---- 配置 ----
            with gr.Accordion("⚙️ 配置", open=True):
                with gr.Row():
                    fw_augment = gr.Number(label="增强数/张", value=1, precision=0, info="0=不增强")
                    fw_rounds = gr.Number(label="轮数", value=1, precision=0, info="多轮=难例迭代")
                    fw_caption_backend = gr.Dropdown(
                        ["auto", "dashscope", "openai", "anthropic", "local"],
                        value="auto", label="Caption 后端", info="auto=最快可用 API")
                    fw_backend = gr.Dropdown(["transformers", "vllm"], value="transformers",
                                             label="检测后端")
                fw_use_api = gr.Checkbox(label="全 API 模式（无需 GPU，bbox 精度较低）", value=False)
                fw_detector_models = gr.Textbox(
                    label="检测器列表（逗号分隔，可填本地路径或 api）",
                    value="",
                    info="示例: /path/to/models/Rex-Omni-3B,/path/to/models/Rex-Omni-30B,api",
                )
                with gr.Row():
                    fw_detector_primary = gr.Textbox(label="主检测器tag（可留空）", value="")
                    fw_det_policy = gr.Dropdown(["drop", "rescore", "flag"], value="drop", label="检测Verifier策略")
                    fw_keep_verified = gr.Checkbox(label="导出仅保留 verified QA", value=False)
                with gr.Row():
                    fw_en_bbox_single = gr.Checkbox(label="启用 bbox_single QA", value=True)
                    fw_en_bbox_rel = gr.Checkbox(label="启用 bbox_relational QA", value=True)
                    fw_en_point_single = gr.Checkbox(label="启用 point_single QA", value=True)
                    fw_en_point_rel = gr.Checkbox(label="启用 point_relational QA", value=True)
                with gr.Row():
                    fw_test_api_btn = gr.Button("🔌 测试 API", size="sm")
                    fw_estimate_btn = gr.Button("⏱ 预估耗时", size="sm")
                fw_api_result = gr.Markdown()
                fw_test_api_btn.click(test_apis, [], [fw_api_result])

            # ---- 步骤开关 ----
            with gr.Accordion("🎛️ 步骤开关", open=False):
                gr.Markdown("取消勾选可跳过该步骤。Caption 和 Grounding 是最慢的步骤。")
                with gr.Row():
                    fw_en_aug = gr.Checkbox(label="① 增强 (~0.15s/张)", value=True)
                    fw_en_cap = gr.Checkbox(label="② Caption (~5.5s/张)", value=True)
                    fw_en_gr = gr.Checkbox(label="③ Grounding (~3s/张)", value=True)
                    fw_en_dep = gr.Checkbox(label="③.5 深度估计 (~0.5s/张)", value=True)
                with gr.Row():
                    fw_en_sg = gr.Checkbox(label="③.6+③.7 场景图+QA (~0.03s/张)", value=True)
                    fw_en_ref = gr.Checkbox(label="④ Referring (~2s/张)", value=True)
                    fw_en_pt = gr.Checkbox(label="⑤ Pointing (~1s/张)", value=True)
                    fw_en_ev = gr.Checkbox(label="⑦ 评估 (~3s/张)", value=True)

            _switch_inputs = [fw_max_images, fw_use_api, fw_en_aug, fw_en_cap, fw_en_gr,
                              fw_en_dep, fw_en_sg, fw_en_ref, fw_en_pt, fw_en_ev]
            fw_time_md = gr.Markdown()
            fw_estimate_btn.click(estimate_time_display, _switch_inputs, [fw_time_md])

            # ---- 启动 ----
            fw_run_btn = gr.Button("🚀 启动飞轮", variant="primary", size="lg")
            fw_log = gr.Code(label="运行日志", language="json")

            _run_inputs = [fw_image_dir, fw_max_images, fw_augment, fw_rounds,
                           fw_caption_backend, fw_backend, fw_use_api, fw_detector_models, fw_detector_primary,
                           fw_det_policy, fw_keep_verified, fw_en_bbox_single, fw_en_bbox_rel,
                           fw_en_point_single, fw_en_point_rel,
                           fw_en_aug, fw_en_cap, fw_en_gr, fw_en_dep, fw_en_sg,
                           fw_en_ref, fw_en_pt, fw_en_ev]
            fw_run_btn.click(run_flywheel_local, _run_inputs, [fw_log])

            # ---- 结果浏览 ----
            gr.Markdown("---\n### 📊 结果浏览")
            with gr.Row():
                fw_round = gr.Number(label="轮次", value=0, precision=0, scale=1)
                fw_max_show = gr.Number(label="每类最多展示", value=10, precision=0, scale=1,
                                        info="控制样本展示数量")
                fw_load_btn = gr.Button("加载结果", variant="secondary", scale=1)

            with gr.Tabs(elem_classes="result-tabs"):
                with gr.Tab("📊 总览"):
                    fw_overview = gr.Markdown()
                with gr.Tab("🖼️ 可视化"):
                    fw_gallery = gr.Gallery(label="可视化（Grounding / 深度图 / 场景图 / 标注）",
                                            columns=4, height=400)
                with gr.Tab("🔗 场景图"):
                    fw_sg_md = gr.Markdown()
                with gr.Tab("❓ 空间QA"):
                    fw_qa_md = gr.Markdown()
                with gr.Tab("📏 深度估计"):
                    fw_dep_md = gr.Markdown()
                with gr.Tab("🎯 Grounding"):
                    fw_gr_md = gr.Markdown()
                with gr.Tab("📝 Referring"):
                    fw_ref_md = gr.Markdown()
                with gr.Tab("👆 Pointing"):
                    fw_pt_md = gr.Markdown()
                with gr.Tab("⚠️ 难例"):
                    fw_hard_md = gr.Markdown()
                with gr.Tab("📦 导出"):
                    fw_exp_md = gr.Markdown()

            def _load_all(round_idx, max_show):
                r, m = int(round_idx), int(max_show)
                return (
                    browse_overview(r),
                    browse_visualizations(r),
                    browse_scene_graph(r, m),
                    browse_spatial_qa(r, m),
                    browse_depth(r, m),
                    browse_grounding(r, m),
                    browse_referring(r, m),
                    browse_pointing(r, m),
                    browse_hard_samples(r, m),
                    browse_export(r, m),
                )

            fw_load_btn.click(_load_all, [fw_round, fw_max_show],
                              [fw_overview, fw_gallery, fw_sg_md, fw_qa_md, fw_dep_md,
                               fw_gr_md, fw_ref_md, fw_pt_md, fw_hard_md, fw_exp_md])

        # ==============================================================
        # Tab 2: 推理工作台
        # ==============================================================
        with gr.Tab("🔍 推理工作台"):
            gr.Markdown("用 Rex-Omni 对图片做目标检测、定位或分割。\n> 需先启动后端: `python -m services.app`")
            with gr.Row():
                inf_image = gr.Textbox(label="图片路径",
                                       value=os.path.join(DEFAULT_IMAGE, "scene_01_kitchen_counter.jpg"), scale=3)
                inf_task = gr.Dropdown(["detection", "pointing", "segment"], value="detection",
                                       label="任务", scale=1)
            inf_cats = gr.Textbox(label="类别（逗号分隔）",
                                  value="cup, bottle, plate, bowl, spoon, fork, knife, mug")
            inf_backend = gr.Dropdown(["transformers", "vllm"], value="transformers", label="后端")
            inf_btn = gr.Button("运行推理", variant="primary")
            inf_output = gr.Code(label="结果 JSON", language="json")
            inf_btn.click(run_inference, [inf_image, inf_task, inf_cats, inf_backend], [inf_output])

        # ==============================================================
        # Tab 3: 抓取决策
        # ==============================================================
        with gr.Tab("🦾 抓取决策"):
            gr.Markdown("基于 Rex-Omni 检测结果，规划抓取策略。\n> 需先启动后端: `python -m services.app`")
            with gr.Row():
                grasp_image = gr.Textbox(label="图片路径",
                                         value=os.path.join(DEFAULT_IMAGE, "scene_01_kitchen_counter.jpg"), scale=3)
                grasp_target = gr.Textbox(label="目标物体", value="cup", scale=1)
            with gr.Row():
                grasp_strategy = gr.Dropdown(["nearest", "largest", "most_confident"],
                                             value="most_confident", label="策略")
                grasp_backend = gr.Dropdown(["transformers", "vllm"], value="transformers", label="后端")
            grasp_btn = gr.Button("规划抓取", variant="primary")
            grasp_output = gr.Code(label="结果 JSON", language="json")
            grasp_btn.click(run_grasp, [grasp_image, grasp_target, grasp_strategy, grasp_backend],
                            [grasp_output])

        # ==============================================================
        # Tab 4: 自动标注
        # ==============================================================
        with gr.Tab("🏷️ 自动标注"):
            gr.Markdown("批量自动标注图片目录。\n> 需先启动后端: `python -m services.app`")
            label_dir = gr.Textbox(label="图片目录", value=DEFAULT_IMAGE)
            label_cats = gr.Textbox(label="类别（逗号分隔）",
                                    value="cup, bottle, plate, bowl, spoon, fork, knife")
            with gr.Row():
                label_backend = gr.Dropdown(["transformers", "vllm"], value="transformers", label="后端")
                label_max = gr.Number(label="最大图片数", value=50, precision=0)
            label_btn = gr.Button("开始标注", variant="primary")
            label_output = gr.Code(label="结果 JSON", language="json")
            label_btn.click(run_auto_label, [label_dir, label_cats, label_backend, label_max],
                            [label_output])

        # ==============================================================
        # Tab 5: Pipeline Report
        # ==============================================================
        with gr.Tab("📊 Pipeline Report"):
            gr.Markdown("浏览由 M5 可视化模块生成的 HTML 报告。")
            with gr.Row():
                report_dir = gr.Textbox(label="报告目录", value=REPORTS_OUTPUT, scale=3)
                report_refresh = gr.Button("刷新", scale=1)
            report_selector = gr.Dropdown(label="报告文件", choices=[], scale=3)
            report_viewer = gr.HTML(value="<p style='color:#999;text-align:center;padding:40px;'>请选择一个报告文件</p>")

            report_refresh.click(fn=refresh_reports, inputs=[report_dir], outputs=[report_selector, report_viewer])
            report_selector.change(fn=load_report_html, inputs=[report_selector, report_dir], outputs=[report_viewer])

        # ==============================================================
        # Tab 5: 能力看板 (v3.0 新增)
        # ==============================================================
        with gr.Tab("📈 能力看板"):
            gr.Markdown("### v3.0 能力维度数据看板\n按 9 项能力维度展示数据分布和质量指标。")
            with gr.Row():
                cap_run_dir = gr.Textbox(
                    label="Run 输出目录",
                    value=os.path.join(FLYWHEEL_OUTPUT, ""),
                    scale=3,
                )
                cap_refresh_btn = gr.Button("刷新", scale=1)

            with gr.Tabs():
                with gr.Tab("分布概览"):
                    cap_stats_display = gr.JSON(label="能力维度统计")
                with gr.Tab("雷达图"):
                    gr.Markdown("当前数据集 capability 分布 vs 27B 缺口目标:")
                    cap_radar_plot = gr.Plot(label="能力雷达图")
                with gr.Tab("能力开关"):
                    gr.Markdown("选择要启用的能力维度:")
                    cap_checkboxes = gr.CheckboxGroup(
                        choices=["G-Grounding", "S-Spatial", "A-Attribute",
                                 "R-Reasoning", "P-Planning", "F-Instruction",
                                 "H-Hallucination", "E-MultiTurn", "Aff-Affordance"],
                        value=["G-Grounding", "S-Spatial", "A-Attribute",
                               "R-Reasoning", "P-Planning", "F-Instruction",
                               "H-Hallucination", "E-MultiTurn", "Aff-Affordance"],
                        label="启用的能力维度",
                    )
                with gr.Tab("数据源"):
                    gr.Markdown("配置数据源路径（每行一条）:")
                    cap_source_paths = gr.Textbox(
                        label="数据源路径",
                        value="./raw_data/test/",
                        lines=5,
                        placeholder="每行一个路径\n./raw_data/test/\n/mnt/data/oxe/",
                    )
                with gr.Tab("导出"):
                    gr.Markdown("一键导出 capability-balanced 训练 JSONL:")
                    cap_export_btn = gr.Button("导出 Balanced JSONL", variant="primary")
                    cap_export_output = gr.Textbox(label="导出路径", interactive=False)

            def _load_cap_stats(run_dir):
                stats_path = os.path.join(run_dir, "capability_stats.json")
                if os.path.exists(stats_path):
                    with open(stats_path) as f:
                        return json.load(f)
                return {"message": "未找到 capability_stats.json，请先运行 v3 飞轮"}

            def _render_radar(run_dir):
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import numpy as np

                    stats_path = os.path.join(run_dir, "capability_stats.json")
                    if not os.path.exists(stats_path):
                        return None

                    with open(stats_path) as f:
                        stats = json.load(f)

                    per_cap = stats.get("per_capability", {})
                    tags = ["G", "S", "A", "R", "P", "F", "H", "E", "Aff"]
                    target = [0.90, 0.85, 0.85, 0.80, 0.75, 0.85, 0.85, 0.80, 0.75]
                    total = sum(per_cap.values()) or 1
                    current = [per_cap.get(t, 0) / total for t in tags]

                    angles = np.linspace(0, 2 * np.pi, len(tags), endpoint=False).tolist()
                    angles += angles[:1]
                    current += current[:1]
                    target += target[:1]

                    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                    ax.plot(angles, current, "o-", linewidth=2, label="Current")
                    ax.fill(angles, current, alpha=0.25)
                    ax.plot(angles, target, "s--", linewidth=1.5, label="Target")
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(tags)
                    ax.set_title("Capability Distribution")
                    ax.legend(loc="upper right")
                    plt.tight_layout()
                    return fig
                except Exception:
                    return None

            def _export_balanced(run_dir):
                try:
                    from apps.data_flywheel.sampler import CapabilitySampler
                    sampler = CapabilitySampler()
                    output = sampler.sample_from_run(run_dir)
                    return output if output else "导出失败: 无有效数据"
                except Exception as e:
                    return f"导出失败: {e}"

            cap_refresh_btn.click(fn=_load_cap_stats, inputs=[cap_run_dir], outputs=[cap_stats_display])
            cap_refresh_btn.click(fn=_render_radar, inputs=[cap_run_dir], outputs=[cap_radar_plot])
            cap_export_btn.click(fn=_export_balanced, inputs=[cap_run_dir], outputs=[cap_export_output])

        # ==============================================================
        # Tab 6: 飞轮日志监控 (v3.0 新增)
        # ==============================================================
        with gr.Tab("📋 飞轮日志"):
            gr.Markdown("### 实时监控飞轮运行日志\n自动刷新最新事件。")
            with gr.Row():
                log_dir_input = gr.Textbox(
                    label="日志目录",
                    value="data/output",
                    scale=3,
                )
                log_refresh_btn = gr.Button("刷新日志列表", scale=1)

            log_selector = gr.Dropdown(label="选择日志文件", choices=[], interactive=True)
            log_tail_count = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="显示最近 N 条")
            log_display = gr.JSON(label="日志事件")

            with gr.Row():
                log_progress_display = gr.Textbox(label="运行进度", interactive=False)
                log_stats_display = gr.Textbox(label="能力统计", interactive=False)

            def _list_log_files(log_dir):
                import glob as _glob
                pattern = os.path.join(log_dir, "**/logs/flywheel_*.jsonl")
                files = sorted(_glob.glob(pattern, recursive=True), reverse=True)
                choices = [os.path.relpath(f, log_dir) for f in files]
                return gr.update(choices=choices, value=choices[0] if choices else None)

            def _read_log(log_file, log_dir, tail_n):
                if not log_file:
                    return [], "无日志", ""
                full_path = os.path.join(log_dir, log_file)
                from apps.data_flywheel.orchestrator.logger import FlywheelLogger
                events = FlywheelLogger.read_log(full_path, tail=int(tail_n))

                progress_str = "未开始"
                stats_str = ""
                for ev in reversed(events):
                    if ev.get("event") == "progress":
                        cur = ev.get("current", 0)
                        tot = ev.get("total", 0)
                        spd = ev.get("speed_samples_per_sec", 0)
                        progress_str = f"{cur}/{tot} samples | {spd:.2f} samples/s"
                        stats_str = json.dumps(ev.get("capability_stats", {}), indent=2)
                        break
                    elif ev.get("event") == "run_complete":
                        progress_str = f"完成: {ev.get('total_samples', 0)} samples in {ev.get('elapsed_seconds', 0):.1f}s"
                        stats_str = json.dumps(ev.get("result", {}).get("per_capability", {}), indent=2)
                        break

                return events, progress_str, stats_str

            log_refresh_btn.click(fn=_list_log_files, inputs=[log_dir_input], outputs=[log_selector])
            log_selector.change(
                fn=_read_log,
                inputs=[log_selector, log_dir_input, log_tail_count],
                outputs=[log_display, log_progress_display, log_stats_display],
            )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7862, allowed_paths=_gradio_allowed_paths())


if __name__ == "__main__":
    main()
