#!/usr/bin/env python3
"""空间关系理解模块 v2 端到端测试 — 5 张图片，输出到 data/output_test_0402/。

测试内容：
1. Grounding 标注（API 模式，无需本地模型）
2. 深度估计（FFS API）
3. Scene Graph 构建（2D + 3D 分层）
4. Spatial QA 生成（含思维链）
5. 导出 + 难例挖掘
6. 生成 HTML 可视化报告
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("test_spatial_v2")

# =====================================================================
# 测试配置
# =====================================================================

OUTPUT_ROOT = str(PROJECT_ROOT / "data" / "output_test_0402")
IMAGE_DIR = str(PROJECT_ROOT / "data" / "input" / "images")
MAX_IMAGES = 5
REPORT_HTML = os.path.join(OUTPUT_ROOT, "test_report.html")


def select_test_images(image_dir: str, n: int = 5) -> list:
    """选取前 n 张测试图片。"""
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    return images[:n]


# =====================================================================
# 模拟 Grounding（不依赖 Rex-Omni 模型，用 API 模式或 mock）
# =====================================================================

def run_mock_grounding(images: list, output_jsonl: str) -> dict:
    """用 VLM API 做 grounding，如果 API 不可用则用 mock 数据。"""
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    try:
        from apps.data_flywheel.captioner import generate_captions
        from apps.data_flywheel.grounding import run_grounding

        # 先生成 caption
        pairs = generate_captions(
            image_paths=images,
            caption_backend="auto",
            model_path="IDEA-Research/Rex-Omni",
            backend="transformers",
        )
        logger.info(f"Caption 生成: {len(pairs)} 条")

        # API grounding
        result = run_grounding(
            image_caption_pairs=pairs,
            output_jsonl=output_jsonl,
            vis_dir=os.path.join(os.path.dirname(output_jsonl), "visualizations"),
            visualize=False,
            filter_adjectives=True,
            use_api=True,
        )
        logger.info(f"API Grounding: {result['count']} 条")
        return result

    except Exception as e:
        logger.warning(f"API grounding 失败 ({e})，使用 mock 数据")
        return _write_mock_grounding(images, output_jsonl)


def _write_mock_grounding(images: list, output_jsonl: str) -> dict:
    """写入 mock grounding 数据用于测试后续模块。"""
    from PIL import Image
    import random

    mock_objects = [
        ["cup", "plate", "spoon", "bowl", "fork"],
        ["bottle", "box", "tray", "mug", "knife"],
        ["pan", "pot", "jar", "can", "basket"],
        ["book", "phone", "pen", "glass", "dish"],
        ["bag", "container", "lid", "board", "cloth"],
    ]

    records = []
    for idx, img_path in enumerate(images):
        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception:
            w, h = 640, 480

        objects = mock_objects[idx % len(mock_objects)]
        phrases = []
        for i, obj_name in enumerate(objects):
            x1 = random.randint(10, w // 2)
            y1 = random.randint(10, h // 2)
            x2 = x1 + random.randint(50, min(200, w - x1))
            y2 = y1 + random.randint(50, min(200, h - y1))
            phrases.append({
                "phrase": obj_name,
                "start_char": 0, "end_char": len(obj_name),
                "boxes": [{"bbox": [x1, y1, x2, y2], "confidence": round(random.uniform(0.6, 0.95), 2)}],
            })

        records.append({
            "image": img_path,
            "caption": f"A scene with {', '.join(objects)}.",
            "width": w, "height": h,
            "phrases": phrases,
        })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"count": len(records), "output_jsonl": output_jsonl}


# =====================================================================
# HTML 报告生成
# =====================================================================

def generate_html_report(
    output_root: str,
    grounding_jsonl: str,
    depth_info_jsonl: str,
    scene_graph_jsonl: str,
    spatial_qa_jsonl: str,
    hard_samples_jsonl: str,
    results: dict,
    elapsed: float,
) -> str:
    """生成 HTML 可视化报告 — 包含原图、深度图、bbox、场景图等可视化。"""
    import numpy as np
    from apps.data_flywheel.visualizer import (
        draw_grounding, draw_depth_map, draw_depth_with_boxes,
        draw_scene_graph, draw_annotation, render_to_base64,
    )

    def _load_jsonl(path):
        if not os.path.exists(path):
            return []
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    grounding_data = _load_jsonl(grounding_jsonl)
    depth_data = _load_jsonl(depth_info_jsonl)
    sg_data = _load_jsonl(scene_graph_jsonl)
    qa_data = _load_jsonl(spatial_qa_jsonl)
    hard_data = _load_jsonl(hard_samples_jsonl)

    # 建立 image → depth_info 映射
    depth_map = {d["image"]: d for d in depth_data}

    # 统计
    total_objects = sum(len(sg.get("objects", [])) for sg in sg_data)
    total_relations = sum(len(sg.get("relations", [])) for sg in sg_data)
    pred_counts = {}
    source_counts = {"2d": 0, "3d": 0}
    for sg in sg_data:
        for r in sg.get("relations", []):
            pred_counts[r["predicate"]] = pred_counts.get(r["predicate"], 0) + 1
            source_counts[r.get("source", "2d")] = source_counts.get(r.get("source", "2d"), 0) + 1

    qa_type_counts = {}
    for qa in qa_data:
        t = qa.get("type", "unknown")
        qa_type_counts[t] = qa_type_counts.get(t, 0) + 1

    depth_success = sum(1 for d in depth_data if d.get("depth_valid"))
    depth_failed = len(depth_data) - depth_success

    # 为每张图生成可视化 base64
    image_visuals = {}  # image_path -> {type: base64_uri}
    for idx, sg in enumerate(sg_data):
        img_path = sg["image"]
        if not os.path.exists(img_path):
            continue
        visuals = {}
        objects_sg = sg.get("objects", [])
        relations = sg.get("relations", [])

        # 原图 base64
        from PIL import Image as PILImage
        try:
            orig = PILImage.open(img_path).convert("RGB")
            visuals["original"] = render_to_base64(orig)
        except Exception:
            pass

        # Grounding bbox
        gr_objs = []
        for g in grounding_data:
            if g["image"] == img_path:
                for p in g.get("phrases", []):
                    for box in p.get("boxes", []):
                        gr_objs.append({"name": p["phrase"], "bbox": box["bbox"],
                                        "confidence": box.get("confidence", 0), "id": len(gr_objs)})
                break
        if gr_objs:
            visuals["grounding"] = render_to_base64(draw_grounding(img_path, gr_objs))

        # Annotation (bbox + 属性)
        if objects_sg:
            visuals["annotation"] = render_to_base64(draw_annotation(img_path, objects_sg))

        # Scene Graph (bbox + 关系箭头)
        if relations:
            visuals["scene_graph"] = render_to_base64(draw_scene_graph(img_path, objects_sg, relations))
            # 仅 3D 关系
            rels_3d = [r for r in relations if r.get("source") == "3d"]
            if rels_3d:
                visuals["scene_graph_3d"] = render_to_base64(
                    draw_scene_graph(img_path, objects_sg, rels_3d, show_3d_only=True))

        # 深度图
        di = depth_map.get(img_path)
        if di and di.get("xyz_map_path") and os.path.exists(di["xyz_map_path"]):
            try:
                xyz = np.load(di["xyz_map_path"])
                visuals["depth_heatmap"] = render_to_base64(draw_depth_map(xyz, img_path, alpha=0.5))
                visuals["depth_with_boxes"] = render_to_base64(draw_depth_with_boxes(xyz, objects_sg, img_path))
            except Exception as e:
                logger.warning(f"深度图可视化失败 {img_path}: {e}")

        image_visuals[img_path] = visuals

    # 构建 HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>空间关系理解模块 v2 测试报告</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }
.container { max-width: 1400px; margin: 0 auto; }
h1 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }
h2 { color: #333; margin-top: 30px; border-left: 4px solid #1a73e8; padding-left: 12px; }
h3 { color: #555; }
h4 { color: #666; margin-top: 15px; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }
.stat-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; padding: 20px; text-align: center; }
.stat-box .number { font-size: 2em; font-weight: bold; }
.stat-box .label { font-size: 0.9em; opacity: 0.9; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
th { background: #f8f9fa; font-weight: 600; }
tr:hover { background: #f0f7ff; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin: 2px; }
.tag-2d { background: #e3f2fd; color: #1565c0; }
.tag-3d { background: #fce4ec; color: #c62828; }
.tag-pass { background: #e8f5e9; color: #2e7d32; }
.tag-fail { background: #ffebee; color: #c62828; }
pre { background: #263238; color: #eeffff; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }
.qa-item { border-left: 3px solid #1a73e8; padding: 10px 15px; margin: 10px 0; background: #fafafa; border-radius: 0 6px 6px 0; }
.thinking { color: #666; font-style: italic; font-size: 0.9em; margin: 5px 0; }
.vis-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 15px; margin: 15px 0; }
.vis-item { text-align: center; }
.vis-item img { max-width: 100%; border-radius: 6px; border: 1px solid #ddd; }
.vis-item .vis-label { font-size: 0.85em; color: #666; margin-top: 5px; font-weight: 600; }
.vis-tabs { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.vis-tab { padding: 6px 14px; border-radius: 20px; cursor: pointer; font-size: 0.85em; border: 1px solid #ddd; background: white; }
.vis-tab.active { background: #1a73e8; color: white; border-color: #1a73e8; }
.vis-panel { display: none; }
.vis-panel.active { display: block; }
</style>
</head>
<body>
<div class="container">
""")

    html_parts.append(f"""
<h1>🔍 空间关系理解模块 v2 — 测试报告</h1>
<p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | 总耗时: {elapsed:.1f}s | 测试图片: {len(grounding_data)} 张</p>

<h2>📊 总体统计</h2>
<div class="stats-grid">
  <div class="stat-box"><div class="number">{len(grounding_data)}</div><div class="label">测试图片</div></div>
  <div class="stat-box"><div class="number">{total_objects}</div><div class="label">检测物体</div></div>
  <div class="stat-box"><div class="number">{total_relations}</div><div class="label">空间关系</div></div>
  <div class="stat-box"><div class="number">{len(qa_data)}</div><div class="label">QA 问答对</div></div>
  <div class="stat-box"><div class="number">{depth_success}/{len(depth_data)}</div><div class="label">深度估计成功</div></div>
  <div class="stat-box"><div class="number">{source_counts.get('3d',0)}</div><div class="label">3D 关系</div></div>
</div>
""")

    # 关系分布表
    html_parts.append("""<h2>🔗 空间关系分布</h2><div class="card"><table>
<tr><th>谓词 (Predicate)</th><th>数量</th><th>来源</th></tr>""")
    for pred, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
        # 判断该谓词主要来源
        pred_source = "3d" if pred in ("on-top-of","under","front","behind","inside","occluding","occluded-by") else "2d"
        tag_cls = "tag-3d" if pred_source == "3d" else "tag-2d"
        html_parts.append(f'<tr><td>{pred}</td><td>{cnt}</td><td><span class="tag {tag_cls}">{pred_source.upper()}</span></td></tr>')
    html_parts.append(f"""</table>
<p>来源分布: <span class="tag tag-2d">2D: {source_counts.get('2d', 0)}</span> <span class="tag tag-3d">3D: {source_counts.get('3d', 0)}</span></p>
</div>""")

    # QA 类型分布
    html_parts.append("""<h2>❓ Spatial QA 类型分布</h2><div class="card"><table>
<tr><th>QA 类型</th><th>数量</th></tr>""")
    for t, cnt in sorted(qa_type_counts.items(), key=lambda x: -x[1]):
        html_parts.append(f'<tr><td>{t}</td><td>{cnt}</td></tr>')
    html_parts.append("</table></div>")

    # 每张图的可视化 + 详细场景图
    html_parts.append("<h2>🖼️ 逐图可视化与场景图</h2>")

    # JavaScript for tab switching
    html_parts.append("""<script>
function switchTab(imgIdx, tabName) {
  var panels = document.querySelectorAll('#img-' + imgIdx + ' .vis-panel');
  var tabs = document.querySelectorAll('#img-' + imgIdx + ' .vis-tab');
  panels.forEach(function(p) { p.classList.remove('active'); });
  tabs.forEach(function(t) { t.classList.remove('active'); });
  document.getElementById('panel-' + imgIdx + '-' + tabName).classList.add('active');
  document.getElementById('tab-' + imgIdx + '-' + tabName).classList.add('active');
}
</script>""")

    for idx, sg in enumerate(sg_data):
        img_path = sg["image"]
        img_name = os.path.basename(img_path)
        objects = sg.get("objects", [])
        relations = sg.get("relations", [])
        visuals = image_visuals.get(img_path, {})

        html_parts.append(f'<div class="card" id="img-{idx}">')
        html_parts.append(f'<h3>图片 {idx+1}: {img_name}</h3>')
        html_parts.append(f'<p>物体: {len(objects)} 个 | 关系: {len(relations)} 条 | 尺寸: {sg.get("image_size", [])}</p>')

        # 可视化 Tab 切换
        vis_types = [
            ("original", "原图"),
            ("grounding", "Grounding BBox"),
            ("annotation", "标注+属性"),
            ("scene_graph", "场景图(全部)"),
            ("scene_graph_3d", "场景图(仅3D)"),
            ("depth_heatmap", "深度热力图"),
            ("depth_with_boxes", "深度+BBox"),
        ]
        available_tabs = [(k, label) for k, label in vis_types if k in visuals]

        if available_tabs:
            html_parts.append('<div class="vis-tabs">')
            for ti, (k, label) in enumerate(available_tabs):
                active = " active" if ti == 0 else ""
                html_parts.append(
                    f'<span class="vis-tab{active}" id="tab-{idx}-{k}" '
                    f'onclick="switchTab({idx},\'{k}\')">{label}</span>')
            html_parts.append('</div>')

            for ti, (k, label) in enumerate(available_tabs):
                active = " active" if ti == 0 else ""
                html_parts.append(
                    f'<div class="vis-panel{active}" id="panel-{idx}-{k}">'
                    f'<div class="vis-item"><img src="{visuals[k]}" alt="{label}">'
                    f'<div class="vis-label">{label}</div></div></div>')

        # 节点表
        html_parts.append("""<h4>节点 (Objects)</h4>
<table><tr><th>ID</th><th>名称</th><th>BBox</th><th>置信度</th><th>深度</th><th>can_pick</th><th>can_push</th><th>can_stack</th><th>is_container</th><th>is_surface</th></tr>""")

        for obj in objects:
            depth_str = f"{obj.get('mean_depth', 'N/A')}" if obj.get('mean_depth') else "N/A"
            html_parts.append(f"""<tr>
<td>{obj['id']}</td><td>{obj['name']}</td><td>{[round(v,1) for v in obj['bbox']]}</td>
<td>{obj.get('confidence', 'N/A')}</td><td>{depth_str}</td>
<td>{'✓' if obj.get('can_pick') else '✗'}</td>
<td>{'✓' if obj.get('can_push') else '✗'}</td>
<td>{'✓' if obj.get('can_stack') else '✗'}</td>
<td>{'✓' if obj.get('is_container') else '✗'}</td>
<td>{'✓' if obj.get('is_surface') else '✗'}</td>
</tr>""")

        html_parts.append("""</table><h4>边 (Relations)</h4>
<table><tr><th>Subject</th><th>Predicate</th><th>Object</th><th>Source</th><th>Confidence</th></tr>""")

        for rel in relations[:30]:
            s_name = objects[rel["subject"]]["name"] if rel["subject"] < len(objects) else "?"
            o_name = objects[rel["object"]]["name"] if rel["object"] < len(objects) else "?"
            src = rel.get("source", "2d")
            tag_cls = "tag-3d" if src == "3d" else "tag-2d"
            html_parts.append(f"""<tr>
<td>{s_name} (#{rel['subject']})</td><td><b>{rel['predicate']}</b></td>
<td>{o_name} (#{rel['object']})</td>
<td><span class="tag {tag_cls}">{src}</span></td>
<td>{rel.get('confidence', 'N/A')}</td></tr>""")

        if len(relations) > 30:
            html_parts.append(f'<tr><td colspan="5">... 还有 {len(relations)-30} 条关系</td></tr>')
        html_parts.append("</table></div>")

    # QA 样例
    html_parts.append("<h2>💬 Spatial QA 样例（前 20 条）</h2><div class='card'>")
    for qa in qa_data[:20]:
        neg_tag = ' <span class="tag tag-fail">负例</span>' if qa.get("is_negative") else ""
        html_parts.append(f"""<div class="qa-item">
<p><b>[{qa.get('type', '?')}]</b>{neg_tag} <b>Q:</b> {qa['question']}</p>
<p class="thinking">💭 {qa.get('thinking', 'N/A')}</p>
<p><b>A:</b> {qa['answer']}</p>
</div>""")
        if qa.get("answer_format"):
            html_parts.append(f"<p><b>answer_format:</b> {qa.get('answer_format')}</p>")
        if qa.get("answer_bbox") is not None:
            html_parts.append(f"<p><b>answer_bbox:</b> {qa.get('answer_bbox')}</p>")
        if qa.get("answer_point") is not None:
            html_parts.append(f"<p><b>answer_point:</b> {qa.get('answer_point')}</p>")
        if "verified" in qa:
            html_parts.append(f"<p><b>verified:</b> {qa.get('verified')} | score={qa.get('verify_score')} | reasons={qa.get('verify_reasons')}</p>")
    html_parts.append("</div>")

    # 难例
    if hard_data:
        html_parts.append("<h2>⚠️ 难例样本</h2><div class='card'><table><tr><th>图片</th><th>原因</th><th>详情</th></tr>")
        for h in hard_data:
            html_parts.append(f'<tr><td>{os.path.basename(h["image"])}</td><td><span class="tag tag-fail">{h["reason"]}</span></td><td>IoU={h.get("best_iou","N/A")}</td></tr>')
        html_parts.append("</table></div>")

    # 输出文件清单
    html_parts.append(f"""<h2>📁 输出文件</h2><div class="card"><pre>
{output_root}/
├── round_0/
│   ├── grounding.jsonl
│   ├── depth_info.jsonl
│   ├── depth_maps/          ← xyz_map numpy 文件
│   ├── scene_graph.jsonl
│   ├── spatial_qa.jsonl
│   ├── train_export.jsonl
│   ├── eval_report.json
│   └── hard_samples.jsonl
└── test_report.html  ← 本报告（含嵌入式可视化图片）
</pre></div>""")

    html_parts.append("</div></body></html>")

    report_path = os.path.join(output_root, "test_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    return report_path


# =====================================================================
# 主测试流程
# =====================================================================

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("空间关系理解模块 v2 — 端到端测试")
    logger.info("=" * 60)

    # 选取测试图片
    images = select_test_images(IMAGE_DIR, MAX_IMAGES)
    if not images:
        logger.error(f"未找到测试图片: {IMAGE_DIR}")
        sys.exit(1)
    logger.info(f"测试图片: {len(images)} 张")
    for img in images:
        logger.info(f"  {os.path.basename(img)}")

    round_dir = os.path.join(OUTPUT_ROOT, "round_0")
    os.makedirs(round_dir, exist_ok=True)

    results = {}

    # ---- Step 1: Grounding ----
    logger.info("\n[Step 1] Grounding 标注...")
    grounding_jsonl = os.path.join(round_dir, "grounding.jsonl")
    grounding_result = run_mock_grounding(images, grounding_jsonl)
    results["grounding"] = grounding_result
    logger.info(f"  → {grounding_result['count']} 条标注")

    # ---- Step 2: 深度估计 ----
    logger.info("\n[Step 2] 深度估计 (FFS API)...")
    depth_info_jsonl = os.path.join(round_dir, "depth_info.jsonl")
    try:
        from apps.data_flywheel.depth_client import estimate_depth_batch
        depth_result = estimate_depth_batch(
            grounding_jsonl=grounding_jsonl,
            output_jsonl=depth_info_jsonl,
            max_workers=2,
            timeout=30,
        )
        results["depth"] = depth_result
        logger.info(f"  → {depth_result['count']} 张图, 成功 {depth_result['success']}, 失败 {depth_result['failed']}")
    except Exception as e:
        logger.warning(f"  → 深度估计跳过: {e}")
        results["depth"] = {"count": 0, "success": 0, "failed": 0}
        # 写空文件
        with open(depth_info_jsonl, "w") as f:
            pass

    # ---- Step 3: Scene Graph ----
    logger.info("\n[Step 3] Scene Graph 构建 (2D + 3D)...")
    scene_graph_jsonl = os.path.join(round_dir, "scene_graph.jsonl")
    from apps.data_flywheel.scene_understanding.builder import (
        build_scene_graphs_from_grounding,
    )
    sg_result = build_scene_graphs_from_grounding(
        grounding_jsonl=grounding_jsonl,
        output_jsonl=scene_graph_jsonl,
        depth_info_jsonl=depth_info_jsonl,
        enable_3d=True,
    )
    results["scene_graph"] = sg_result
    logger.info(f"  → {sg_result['count']} 张图, {sg_result['total_objects']} 物体, {sg_result['total_relations']} 关系")
    logger.info(f"  → 关系分布: {sg_result.get('predicate_counts', {})}")
    logger.info(f"  → 来源分布: {sg_result.get('source_counts', {})}")

    # ---- Step 4: Spatial QA ----
    logger.info("\n[Step 4] Spatial QA 生成 (含思维链)...")
    spatial_qa_jsonl = os.path.join(round_dir, "spatial_qa.jsonl")
    from apps.data_flywheel.spatial_qa import generate_spatial_qa
    qa_result = generate_spatial_qa(
        scene_graph_jsonl=scene_graph_jsonl,
        output_jsonl=spatial_qa_jsonl,
        max_qa_per_image=20,
        include_negatives=True,
        negative_ratio=0.3,
        enable_thinking=True,
        lang="en",
        enable_bbox_single=True,
        enable_bbox_relational=True,
        enable_point_single=True,
        enable_point_relational=True,
        enable_verifier=True,
    )
    results["spatial_qa"] = qa_result
    logger.info(f"  → {qa_result['count']} 条 QA, 类型: {qa_result.get('type_counts', {})}")

    # ---- Step 5: 导出 ----
    logger.info("\n[Step 5] 训练集导出...")
    export_jsonl = os.path.join(round_dir, "train_export.jsonl")
    from apps.data_flywheel.exporter import export_dataset
    # 创建空的 referring/pointing 文件
    for fname in ["referring.jsonl", "pointing.jsonl"]:
        fpath = os.path.join(round_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                pass
    export_result = export_dataset(
        grounding_jsonl=grounding_jsonl,
        referring_jsonl=os.path.join(round_dir, "referring.jsonl"),
        pointing_jsonl=os.path.join(round_dir, "pointing.jsonl"),
        output_jsonl=export_jsonl,
        spatial_qa_jsonl=spatial_qa_jsonl,
        scene_graph_jsonl=scene_graph_jsonl,
    )
    results["export"] = export_result
    logger.info(f"  → {export_result['total']} 条, 分布: {export_result.get('stats', {})}")

    # ---- Step 6: 难例挖掘 ----
    logger.info("\n[Step 6] 难例挖掘...")
    hard_samples_jsonl = os.path.join(round_dir, "hard_samples.jsonl")
    # 创建空的 eval_report
    eval_report_path = os.path.join(round_dir, "eval_report.json")
    if not os.path.exists(eval_report_path):
        with open(eval_report_path, "w") as f:
            json.dump({"samples": []}, f)
    from apps.data_flywheel.hard_mining import mine_hard_samples
    hard_result = mine_hard_samples(
        eval_report_path=eval_report_path,
        output_jsonl=hard_samples_jsonl,
        scene_graph_jsonl=scene_graph_jsonl,
        depth_info_jsonl=depth_info_jsonl,
    )
    results["hard_mining"] = hard_result
    logger.info(f"  → {hard_result['hard_count']} 条难例, 分布: {hard_result.get('reason_counts', {})}")

    elapsed = time.time() - t0

    # ---- Step 7: HTML 报告 ----
    logger.info("\n[Step 7] 生成 HTML 可视化报告...")
    report_path = generate_html_report(
        OUTPUT_ROOT, grounding_jsonl, depth_info_jsonl,
        scene_graph_jsonl, spatial_qa_jsonl, hard_samples_jsonl,
        results, elapsed,
    )
    logger.info(f"  → 报告: {report_path}")

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ 测试完成! 耗时 {elapsed:.1f}s")
    logger.info(f"   输出目录: {OUTPUT_ROOT}")
    logger.info(f"   HTML 报告: {report_path}")
    logger.info("=" * 60)

    # 写 summary
    summary = {"elapsed": round(elapsed, 1), "results": results, "report": report_path}
    with open(os.path.join(OUTPUT_ROOT, "test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
