#!/usr/bin/env python3
"""Build data sources analysis report — render sources_priority.yaml to rich HTML.

Generates a comprehensive, self-contained HTML report with:
- Per-source data cards with priority/status/capabilities
- Sampled thumbnail images (base64 embedded) with bbox overlays
- Annotation statistics per source (count, dimension distribution, bbox area stats)
- Flywheel status badges (live vs registered/planned)
- Unified navigation bar

Usage:
    python scripts/build_data_sources_report.py
    python scripts/build_data_sources_report.py --output docs/report.html
    python scripts/build_data_sources_report.py --no-images  # skip image sampling
"""

import argparse
import base64
import html
import io
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ds_report")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_YAML = PROJECT_ROOT / "data" / "sources_priority.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "2026-06-12-\u6570\u636e\u98de\u8f6e\u6570\u636e\u6e90\u5206\u6790.html"

DATA_ROOT = Path("/path/to/datasets/GSA_volc")
IMAGES_ROOT = DATA_ROOT / "images"
ANNOTATIONS_ROOT = DATA_ROOT / "annotations"

LIVE_FLYWHEEL_SOURCES = frozenset([
    "bridge_data_v2", "benchmark", "visual_genome", "spatialsense",
    "embodied_reasoner", "egoobjects", "mit_states", "robosuite",
])

SOURCE_IMAGE_DIR_MAP = {
    "visual_genome": "visual_genome",
    "grasp_anything": "grasp_anything",
    "bridge_data_v2": "bridgev2",
    "bridge_data_v2_local": "bridgev2",
    "embodied_reasoner": "embodied_reasoner",
    "embodied_reasoner_local": "embodied_reasoner",
    "spatialsense": "navigation_spatialsense",
    "spatialsense_local": "navigation_spatialsense",
    "refspatial": "refspatial",
    "refspatial_cleaned": "refspatial",
    "changeit_frames_local": "changeit",
    "where2place_local": "household_where2place",
    "refcoco": "benchmark",
    "grefcoco": "benchmark",
    "cocostuff": "benchmark",
    "lvis": "benchmark",
    "coco_rem": "benchmark",
    "pixmo_points": "benchmark",
    "pixmo_cleaned": "benchmark",
    "the_cauldron": "vlm_archive",
    "finevision": "vlm_archive",
}

CATEGORY_MAP = {
    "the_cauldron": 0, "finevision": 0,
    "refcoco": 1, "grefcoco": 1, "cocostuff": 1, "visual_genome": 1,
    "open_images_v7": 1, "osprey_724k": 1, "lvis": 1, "coco_rem": 1,
    "pixmo_points": 2, "pixmo_cleaned": 2,
    "spatialsense": 3, "vsr": 3, "refspatial": 3, "refspatial_cleaned": 3,
    "scannet_v2": 4, "scannet_hf": 4, "ca1m": 4, "vlm_3r_data": 4, "taskonomy": 4,
    "grasp_anything": 5, "roboafford": 5, "sharerobot": 5, "paco_lvis": 5,
    "pascal_part": 5, "agd20k": 5, "ade20k": 5,
    "clevrer": 6,
    "videorefer_700k": 7, "molmo2_videopoint": 7, "llava_video_178k": 7,
    "droid": 8, "bridge_data_v2": 8, "robo2vlm_1": 8, "embodied_reasoner": 8,
    "agibot_world_2026": 8, "fsd_dataset": 8, "egoplan_it": 8, "interndata_a1": 8,
    "changeit_frames_local": 8, "embodied_reasoner_local": 8, "spatialsense_local": 3,
    "where2place_local": 8, "bridge_data_v2_local": 8,
}

CATEGORY_NAMES = {
    0: "通用 SFT 底座 · 多任务大汇总",
    1: "2D Grounding 主力库（框 · 掩码 · 指代表达）",
    2: "原生 Point 标注与高质量 Mask",
    3: "空间理解与度量推理（Spatial Understanding）",
    4: "3D 真值数据（深度 · 网格 · 3D 框 · 位姿）",
    5: "抓取与可操作性（Grasp + Affordance）",
    6: "时序与因果推理（合成视频物理）",
    7: "视频级 Pointing 与区域指代",
    8: "具身机器人轨迹与衍生 VQA（物理交互真值）",
}

COORD_LABELS = {
    "pixel_xywh": "像素 [x,y,w,h]",
    "pixel_xyxy": "像素 [x1,y1,x2,y2]",
    "norm_01": "归一化 [0,1]",
    "norm_01_xmin_xmax": "归一化 [0,1] (XMin/XMax/YMin/YMax)",
    "percent_100": "百分比 [0,100]",
    "thousandth_1000": "千分制 [0,1000)",
    "text_embedded": "文本内嵌",
}

PRIORITY_COLORS = {"P0": "#e85d75", "P1": "#f0a050", "P2": "#6b9bd1"}
STATUS_COLORS = {"ready": "#5cb85c", "downloading": "#f0ad4e", "pending": "#d9534f"}
TASK_COLORS = {"G": "#7ee787", "S": "#79c0ff", "A": "#f0c674", "P": "#d2a8ff"}

DETAIL_MAP = {
    "refcoco": {
        "content_structure": "COCO train2014 图像 + 指代表达三件套(RefCOCO/+/g)",
        "annotation_format": "parquet + refs(unc/google).p pickle + instances.json(COCO)",
        "key_fields": "sent/question(指代文本), bbox[x,y,w,h]像素, segmentation(polygon/RLE)",
        "capability_analysis": "G 金标准; 表达含左右等方位词可抽弱 S; bbox→中心点即成 P 数据",
    },
    "grefcoco": {
        "content_structure": "广义指代分割: 一条表达对应多目标或无目标(no-target)",
        "annotation_format": "grefs(unc).json + instances.json",
        "key_fields": "sentences[].raw, ann_id(list可空/-1), bbox/segmentation",
        "capability_analysis": "no-target 样本对 G 负样本训练价值高; 多目标表达常含空间限定词(弱 S)",
    },
    "visual_genome": {
        "content_structure": "10.8万图稠密标注: region caption/对象/属性/关系三元组",
        "annotation_format": "region_descriptions/objects/relationships/attributes.json",
        "key_fields": "regions[].phrase+x,y,w,h, relationships[].predicate/subject/object",
        "capability_analysis": "S 关系数据首选底座; 配单目深度可升级3D相对关系",
    },
    "open_images_v7": {
        "content_structure": "bbox(600类)/关系/实例mask + V7官方point-level标注",
        "annotation_format": "point-labels/bbox/vrd/segmentation CSV",
        "key_fields": "X,Y(point,归一0-1), XMin/XMax/YMin/YMax(bbox)",
        "capability_analysis": "V7 point标注=官方 P 金矿(含负点); vrd holds/wears 抽交互弱标",
    },
    "pixmo_points": {
        "content_structure": "原生 point 标注主力; 清洗版含8子集(47万)",
        "annotation_format": "原始百分比0-100; 清洗版internvl已转千分制",
        "key_fields": "point坐标(百分比/千分制), count, explanation文本",
        "capability_analysis": "P 直接监督主力; point2bbox 支持点↔框互转",
    },
    "refspatial": {
        "content_structure": "度量深度 + 空间推理; 332GB完整(含Simulator)",
        "annotation_format": "原始norm 0-1; 清洗版qwenvl/internvl/swift三套格式",
        "key_fields": "空间QA, 度量答案(米/厘米), point/bbox答案",
        "capability_analysis": "S 绝对度量主力(度量深度); 点答案服务 P",
    },
    "grasp_anything": {
        "content_structure": "大规模抓取标注: 抓取矩形(位置+角度+宽度)",
        "annotation_format": "grasp rectangles + 文本指令",
        "key_fields": "抓取矩形(cx,cy,angle,w,h), 物体类别",
        "capability_analysis": "A_action_interface 直接监督; 中心可转 P",
    },
    "bridge_data_v2": {
        "content_structure": "桌面操作轨迹; 腕相机视角; 25K图像",
        "annotation_format": "episode图像帧 + state/action轨迹",
        "key_fields": "episode帧, gripper state, 操作轨迹",
        "capability_analysis": "tabletop G/S/A 通用底座; 腕相机帧服务 A_robot_state",
    },
}


def load_sources():
    """Load sources from YAML."""
    try:
        import yaml
        with open(SOURCES_YAML) as f:
            data = yaml.safe_load(f)
        return data.get("sources", [])
    except ImportError:
        logger.error("PyYAML not available — pip install pyyaml")
        return []


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def get_image_dir(source_name: str):
    """Resolve local image directory for a source."""
    dir_name = SOURCE_IMAGE_DIR_MAP.get(source_name)
    if not dir_name:
        return None
    p = IMAGES_ROOT / dir_name
    return p if p.is_dir() else None


def sample_images(source_name: str, n: int = 4) -> list:
    """Sample up to n images from a source's image directory."""
    img_dir = get_image_dir(source_name)
    if not img_dir:
        return []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    candidates = []
    for ext in extensions:
        candidates.extend(img_dir.glob(f"*{ext}"))
        if len(candidates) > 500:
            break
    if not candidates:
        for ext in extensions:
            candidates.extend(img_dir.rglob(f"*{ext}"))
            if len(candidates) > 500:
                break
    if not candidates:
        return []
    random.seed(hash(source_name) % (2**32))
    return random.sample(candidates, min(n, len(candidates)))


def find_bbox_for_image(image_path: Path, source_name: str):
    """Try to find a bbox annotation for a given image from annotation files."""
    img_name = image_path.name
    search_dirs = [
        ANNOTATIONS_ROOT / "flywheel_verified",
        ANNOTATIONS_ROOT / "converted",
    ]
    for ann_dir in search_dirs:
        if not ann_dir.is_dir():
            continue
        for dim_dir in ann_dir.iterdir():
            if not dim_dir.is_dir():
                continue
            for jsonl_file in dim_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, "r") as f:
                        for line in f:
                            if img_name not in line:
                                continue
                            record = json.loads(line)
                            if img_name in str(record.get("image_path", "")):
                                bbox = record.get("bbox")
                                if bbox and isinstance(bbox, list) and len(bbox) == 4:
                                    return bbox
                except (json.JSONDecodeError, OSError):
                    continue
    return None


def image_to_base64_thumbnail(image_path: Path, max_width: int = 200, bbox=None):
    """Load image, resize, optionally draw bbox, return base64 data URI."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
    except (OSError, Exception):
        return None
    if bbox:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        if all(0 <= v <= 1.0 for v in bbox):
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        elif all(0 <= v <= 1000 for v in bbox):
            x1, y1, x2, y2 = x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000
        draw.rectangle([x1, y1, x2, y2], outline="#58a6ff", width=3)
    ratio = max_width / img.width if img.width > max_width else 1.0
    if ratio < 1.0:
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def compute_annotation_stats() -> dict:
    """Scan annotation dirs and compute per-source stats.

    Returns: {source_name: {total: int, by_dimension: Counter, bbox_areas: list}}
    """
    stats = defaultdict(lambda: {"total": 0, "by_dimension": Counter(), "bbox_areas": []})
    ann_dirs = [
        ("flywheel_verified", ANNOTATIONS_ROOT / "flywheel_verified"),
        ("converted", ANNOTATIONS_ROOT / "converted"),
    ]
    for label, ann_dir in ann_dirs:
        if not ann_dir.is_dir():
            continue
        for dim_dir in ann_dir.iterdir():
            if not dim_dir.is_dir():
                continue
            dim_name = dim_dir.name
            for jsonl_file in dim_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, "r") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            src = rec.get("source") or rec.get("dataset", "")
                            stats[src]["total"] += 1
                            stats[src]["by_dimension"][dim_name] += 1
                            bbox = rec.get("bbox")
                            if bbox and len(bbox) == 4:
                                try:
                                    area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                                    stats[src]["bbox_areas"].append(area)
                                except (TypeError, IndexError):
                                    pass
                except OSError:
                    continue

    bg1m = ANNOTATIONS_ROOT / "bbox_grounding_1m" / "sft_train.jsonl"
    if bg1m.is_file():
        try:
            with open(bg1m, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src = rec.get("source") or rec.get("dataset", "")
                    stats[src]["total"] += 1
                    stats[src]["by_dimension"]["bbox_grounding_1m"] += 1
        except OSError:
            pass
    return dict(stats)


def bbox_area_summary(areas: list) -> dict:
    """Compute min/max/mean/median for bbox area list."""
    if not areas:
        return {}
    areas_sorted = sorted(areas)
    n = len(areas_sorted)
    return {
        "count": n,
        "min": f"{areas_sorted[0]:.4f}",
        "max": f"{areas_sorted[-1]:.4f}",
        "mean": f"{sum(areas_sorted) / n:.4f}",
        "median": f"{areas_sorted[n // 2]:.4f}",
    }



HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GSA 数据飞轮 · 数据源分析与处理</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.6; padding-bottom: 60px;
  }
  .topnav {
    background: #010409; border-bottom: 1px solid #21262d; padding: 8px 40px;
    display: flex; gap: 24px; align-items: center;
  }
  .topnav a {
    color: #58a6ff; text-decoration: none; font-size: 13px; font-weight: 500;
    padding: 4px 10px; border-radius: 6px; transition: background .2s;
  }
  .topnav a:hover { background: #1f6feb33; }
  .topnav a.active { background: #1f6feb; color: #fff; }
  .hero {
    background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
    padding: 48px 40px 36px; border-bottom: 1px solid #30363d;
  }
  .hero h1 { font-size: 30px; color: #f0f6fc; margin-bottom: 10px; font-weight: 700; }
  .hero .sub { color: #8b949e; font-size: 14px; }
  .hero .meta { margin-top: 16px; display: flex; gap: 20px; flex-wrap: wrap; }
  .hero .meta span { font-size: 13px; color: #58a6ff; }
  nav {
    position: sticky; top: 0; z-index: 100; background: #161b22;
    border-bottom: 1px solid #30363d; padding: 10px 40px; overflow-x: auto;
    white-space: nowrap;
  }
  nav a {
    color: #8b949e; text-decoration: none; margin-right: 18px; font-size: 13px;
    padding: 4px 0; transition: color .2s;
  }
  nav a:hover { color: #58a6ff; }
  .container { max-width: 1280px; margin: 0 auto; padding: 32px 40px; }
  section { margin-bottom: 48px; }
  h2 {
    font-size: 22px; color: #f0f6fc; margin-bottom: 8px; padding-bottom: 10px;
    border-bottom: 2px solid #30363d; font-weight: 700;
  }
  h2 .cat-num {
    display: inline-block; width: 32px; height: 32px; line-height: 32px;
    text-align: center; background: #1f6feb; color: #fff; border-radius: 8px;
    font-size: 16px; margin-right: 12px;
  }
  .cat-desc { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
  .summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px; margin: 20px 0;
  }
  .stat-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 18px; text-align: center;
  }
  .stat-card .num { font-size: 28px; font-weight: 700; color: #58a6ff; }
  .stat-card .lbl { font-size: 12px; color: #8b949e; margin-top: 4px; }
  .ds-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px; transition: border-color .2s;
  }
  .ds-card:hover { border-color: #58a6ff; }
  .ds-header { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }
  .ds-name { font-size: 18px; font-weight: 700; color: #f0f6fc; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; color: #fff;
  }
  .badge-flywheel { background: #238636; }
  .badge-registered { background: #6e7681; }
  .task-pills { display: flex; gap: 4px; }
  .task-pill {
    width: 22px; height: 22px; line-height: 22px; text-align: center;
    border-radius: 6px; font-size: 12px; font-weight: 700; color: #0d1117;
  }
  .ds-path {
    font-family: "SF Mono", Consolas, monospace; font-size: 12px; color: #7d8590;
    background: #0d1117; padding: 6px 10px; border-radius: 6px; margin: 8px 0;
    word-break: break-all; border: 1px solid #21262d;
  }
  .ds-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px; margin-top: 12px;
  }
  .ds-field { background: #0d1117; border-radius: 8px; padding: 10px 12px; border: 1px solid #21262d; }
  .ds-field .k { font-size: 11px; color: #6e7681; text-transform: uppercase; letter-spacing: .5px; }
  .ds-field .v { font-size: 13px; color: #c9d1d9; margin-top: 3px; }
  .thumb-row {
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; padding: 10px;
    background: #0d1117; border-radius: 8px; border: 1px solid #21262d;
  }
  .thumb-row img {
    border-radius: 6px; max-height: 140px; object-fit: cover;
    border: 1px solid #30363d;
  }
  .ann-stats {
    background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
    padding: 12px 14px; margin-top: 10px; font-size: 12px;
  }
  .ann-stats .title { color: #58a6ff; font-weight: 600; margin-bottom: 6px; }
  .ann-stats table { width: 100%; font-size: 12px; }
  .ann-stats td { padding: 2px 8px; border: none; }
  .clean-box {
    background: #1c2128; border-left: 3px solid #3fb950; padding: 10px 14px;
    margin-top: 12px; border-radius: 0 6px 6px 0; font-size: 13px; color: #adbac7;
  }
  .clean-box .lbl { color: #3fb950; font-weight: 600; font-size: 11px; text-transform: uppercase; }
  table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }
  th { background: #161b22; color: #8b949e; font-weight: 600; }
  td { color: #c9d1d9; }
  .coord-warn {
    background: #2d1f12; border: 1px solid #9e6a03; border-radius: 8px;
    padding: 14px 18px; margin: 16px 0; color: #e3b341; font-size: 13px;
  }
  .pipeline-flow {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 20px; margin: 16px 0; font-size: 13px; color: #adbac7;
  }
  .flow-step { display: inline-block; background: #1f6feb; color: #fff; padding: 6px 12px;
    border-radius: 6px; margin: 4px; font-size: 12px; }
  .flow-arrow { color: #58a6ff; margin: 0 4px; }
  code { font-family: "SF Mono", Consolas, monospace; background: #21262d; padding: 1px 6px; border-radius: 4px; color: #79c0ff; }
</style>
</head>
<body>
"""

HTML_TAIL = """
</body>
</html>
"""


def render_topnav():
    return """
<div class="topnav">
  <a href="/data-sources" class="active">数据源 Data Sources</a>
  <a href="/flywheel">飞轮 Flywheel</a>
  <a href="/training-eval">训练评测 Training &amp; Eval</a>
</div>
"""


def render_hero(sources):
    total = len(sources)
    ready = sum(1 for s in sources if s.get("status") == "ready")
    p0 = sum(1 for s in sources if s.get("priority") == "P0")
    live = sum(1 for s in sources if s.get("name") in LIVE_FLYWHEEL_SOURCES)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
<div class="hero">
  <h1>GSA 数据飞轮 · 数据源分析与处理</h1>
  <div class="sub">按 9 大类组织 · 图像采样 / 标注统计 / 飞轮状态 · 生成于 {now}</div>
  <div class="meta">
    <span>● {total} 个数据源</span>
    <span>● {ready} 个就绪</span>
    <span>● {p0} 个 P0 主力</span>
    <span>● {live} 个已接入飞轮</span>
    <span>● 4 能力轴 G/S/A/P</span>
  </div>
</div>
"""


def render_nav():
    links = "".join(
        f'<a href="#cat{i}">{i} · {CATEGORY_NAMES[i].split("（")[0].split("·")[0].strip()}</a>'
        for i in range(9)
    )
    return (
        f'<nav><a href="#overview">概览</a><a href="#tasks">任务映射</a>'
        f'{links}<a href="#coord">坐标统一</a><a href="#pipeline">处理管线</a></nav>'
    )


def render_overview(sources):
    by_task = Counter()
    by_priority = Counter()
    by_status = Counter()
    for s in sources:
        for t in s.get("tasks", []):
            by_task[t] += 1
        by_priority[s.get("priority", "?")] += 1
        by_status[s.get("status", "?")] += 1

    cards = ""
    for label, count in [
        ("数据源总数", len(sources)),
        ("P0 主力源", by_priority.get("P0", 0)),
        ("就绪可用", by_status.get("ready", 0)),
        ("下载中", by_status.get("downloading", 0)),
        ("飞轮已接入", sum(1 for s in sources if s.get("name") in LIVE_FLYWHEEL_SOURCES)),
    ]:
        cards += f'<div class="stat-card"><div class="num">{count}</div><div class="lbl">{label}</div></div>'

    task_rows = ""
    task_names = {"G": "Grounding 定位", "S": "Spatial 空间", "A": "Affordance 可操作", "P": "Point 点指代"}
    for t in ["G", "S", "A", "P"]:
        task_rows += (
            f'<tr><td><span class="task-pill" style="background:{TASK_COLORS[t]}">{t}</span> '
            f'{task_names[t]}</td><td>{by_task.get(t, 0)} 个数据源</td></tr>'
        )

    return f"""
<section id="overview">
  <h2>数据源概览</h2>
  <div class="cat-desc">本报告系统梳理 GSA 数据飞轮接入的全部数据源，按 9 大类递进组织。</div>
  <div class="summary-grid">{cards}</div>
  <table>
    <tr><th>能力轴</th><th>覆盖数据源</th></tr>
    {task_rows}
  </table>
</section>
"""


def render_task_mapping():
    return """
<section id="tasks">
  <h2>三大任务 → 数据映射</h2>
  <table>
    <tr><th>任务目标</th><th>主力数据</th><th>次级来源</th><th>备注</th></tr>
    <tr><td>Grounding + 空间推理 + 绝对度量</td><td>RefSpatial · ScanNet v2 · CA-1M</td><td>VG · SpatialSense · VSR</td><td>3D 真值→投影→数值答案(米/厘米)</td></tr>
    <tr><td>Affordance</td><td>RoboAfford · ShareRobot · Grasp-Anything</td><td>PACO · DROID/Bridge</td><td>真机轨迹造数需相机标定</td></tr>
    <tr><td>Point Grounding</td><td>PixMo-Points · OIv7 · Molmo2-VideoPoint</td><td>RefSpatial点答案 · bbox/mask转点</td><td>坐标协议冲突需统一</td></tr>
  </table>
</section>
"""


def render_dataset_card(s, ann_stats: dict, include_images: bool = True):
    """Render a single dataset card with optional images and annotation stats."""
    name = s.get("name", "?")
    name_esc = esc(name)
    priority = s.get("priority", "P2")
    status = s.get("status", "ready")
    tasks = s.get("tasks", [])
    coord = s.get("coord_system", "")

    task_pills = "".join(
        f'<span class="task-pill" style="background:{TASK_COLORS.get(t, "#888")}">{t}</span>'
        for t in tasks
    )
    pcolor = PRIORITY_COLORS.get(priority, "#888")
    scolor = STATUS_COLORS.get(status, "#888")
    coord_label = COORD_LABELS.get(coord, coord)

    if name in LIVE_FLYWHEEL_SOURCES:
        flywheel_badge = '<span class="badge badge-flywheel">飞轮已接入</span>'
    else:
        flywheel_badge = '<span class="badge badge-registered">已注册/计划中</span>'

    dims = ", ".join(s.get("dimensions", []))
    weight = s.get("weight", "-")
    stage = s.get("stage", "-")
    scene = s.get("scene_tag", "-")

    detail = DETAIL_MAP.get(name, {})
    content_structure = s.get("content_structure") or detail.get("content_structure", "")
    annotation_format = s.get("annotation_format") or detail.get("annotation_format", "")
    key_fields = s.get("key_fields") or detail.get("key_fields", "")
    capability = s.get("capability_analysis") or detail.get("capability_analysis", "")

    detail_html = ""
    if content_structure:
        detail_html += f'<div class="ds-field"><div class="k">内容结构</div><div class="v">{esc(content_structure)}</div></div>'
    if annotation_format:
        detail_html += f'<div class="ds-field"><div class="k">标注格式</div><div class="v">{esc(annotation_format)}</div></div>'
    if key_fields:
        detail_html += f'<div class="ds-field"><div class="k">关键字段</div><div class="v">{esc(key_fields)}</div></div>'
    detail_block = f'<div class="ds-grid">{detail_html}</div>' if detail_html else ""

    cap_block = ""
    if capability:
        cap_block = (
            f'<div class="clean-box" style="border-left-color:#d2a8ff">'
            f'<span class="lbl" style="color:#d2a8ff">能力分析 · </span>{esc(capability)}</div>'
        )

    thumb_html = ""
    if include_images:
        images = sample_images(name, n=4)
        if images:
            img_tags = []
            for img_path in images:
                bbox = find_bbox_for_image(img_path, name)
                b64 = image_to_base64_thumbnail(img_path, max_width=200, bbox=bbox)
                if b64:
                    img_tags.append(f'<img src="{b64}" alt="{esc(img_path.name)}" title="{esc(img_path.name)}">')
            if img_tags:
                thumb_html = f'<div class="thumb-row">{"".join(img_tags)}</div>'

    ann_html = ""
    src_stats = ann_stats.get(name)
    if src_stats and src_stats["total"] > 0:
        dim_rows = "".join(
            f"<tr><td>{esc(d)}</td><td>{c:,}</td></tr>"
            for d, c in src_stats["by_dimension"].most_common(6)
        )
        area_info = bbox_area_summary(src_stats["bbox_areas"])
        area_str = ""
        if area_info:
            area_str = (
                f" | bbox面积: mean={area_info['mean']}, "
                f"median={area_info['median']}, n={area_info['count']}"
            )
        ann_html = f"""
<div class="ann-stats">
  <div class="title">标注统计 · 总计 {src_stats['total']:,} 条{area_str}</div>
  <table>{dim_rows}</table>
</div>"""

    return f"""
<div class="ds-card">
  <div class="ds-header">
    <span class="ds-name">{name_esc}</span>
    <span class="badge" style="background:{pcolor}">{priority}</span>
    <span class="badge" style="background:{scolor}">{esc(status)}</span>
    {flywheel_badge}
    <span class="task-pills">{task_pills}</span>
  </div>
  <div class="ds-path">{esc(s.get("path", ""))}</div>
  <div class="ds-grid">
    <div class="ds-field"><div class="k">飞轮维度</div><div class="v">{esc(dims)}</div></div>
    <div class="ds-field"><div class="k">场景标签</div><div class="v">{esc(scene)}</div></div>
    <div class="ds-field"><div class="k">坐标系</div><div class="v">{esc(coord_label)}</div></div>
    <div class="ds-field"><div class="k">采样权重</div><div class="v">{esc(weight)}</div></div>
    <div class="ds-field"><div class="k">训练阶段</div><div class="v">{esc(stage)}</div></div>
    <div class="ds-field"><div class="k">类型</div><div class="v">{esc(s.get("type", ""))}</div></div>
  </div>
  {detail_block}
  {thumb_html}
  {ann_html}
  <div class="clean-box"><span class="lbl">清洗策略 · </span>{esc(s.get("clean_strategy", ""))}</div>
  {cap_block}
  <div class="clean-box" style="border-left-color:#58a6ff"><span class="lbl" style="color:#58a6ff">来源备注 · </span>{esc(s.get("notes", ""))}</div>
</div>
"""


def render_categories(sources, ann_stats: dict, include_images: bool = True):
    by_cat = defaultdict(list)
    for s in sources:
        cat = CATEGORY_MAP.get(s.get("name"), 8)
        by_cat[cat].append(s)

    html_out = ""
    for cat in range(9):
        if cat not in by_cat:
            continue
        cards = "".join(
            render_dataset_card(s, ann_stats, include_images)
            for s in by_cat[cat]
        )
        html_out += f"""
<section id="cat{cat}">
  <h2><span class="cat-num">{cat}</span>{CATEGORY_NAMES[cat]}</h2>
  {cards}
</section>
"""
    return html_out


def render_coord_section():
    return """
<section id="coord">
  <h2>坐标协议统一</h2>
  <div class="coord-warn">
    <strong>系统性问题</strong>：各数据集坐标协议不统一 — 像素 xywh / 归一化 0-1 / 百分比 0-100 / 千分制 0-1000。
    入管线前必须统一到 <code>[0,1000)</code>。
  </div>
  <table>
    <tr><th>源坐标系</th><th>示例</th><th>转换 → [0,1000)</th></tr>
    <tr><td>pixel_xywh</td><td>[100,200,300,400] @1024x768</td><td>x·1000/W, y·1000/H</td></tr>
    <tr><td>norm_01</td><td>[0.1,0.2,0.5,0.8]</td><td>v · 1000</td></tr>
    <tr><td>percent_100</td><td>[50,75]</td><td>v · 10</td></tr>
    <tr><td>thousandth_1000</td><td>[100,200,500,800]</td><td>直通</td></tr>
  </table>
</section>
"""


def render_pipeline_section():
    return """
<section id="pipeline">
  <h2>数据处理管线总览</h2>
  <div class="pipeline-flow">
    <span class="flow-step">① 数据源注册</span><span class="flow-arrow">→</span>
    <span class="flow-step">② 坐标统一 [0,1000)</span><span class="flow-arrow">→</span>
    <span class="flow-step">③ VLM 生成式标注</span><span class="flow-arrow">→</span>
    <span class="flow-step">④ 三层校验</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑤ 清洗</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑥ 导出 GSA_volc</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑦ 上传 HF</span>
  </div>
</section>
"""


def main():
    parser = argparse.ArgumentParser(description="Build data sources HTML report")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output HTML file path")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image sampling for fast rebuild")
    args = parser.parse_args()

    include_images = not args.no_images

    print("[1/4] Loading sources from YAML...")
    sources = load_sources()
    if not sources:
        logger.error("No sources loaded from %s", SOURCES_YAML)
        sys.exit(1)
    print(f"       Loaded {len(sources)} sources")

    print("[2/4] Computing annotation statistics...")
    ann_stats = compute_annotation_stats()
    total_annotations = sum(v["total"] for v in ann_stats.values())
    print(f"       Found annotations for {len(ann_stats)} sources ({total_annotations:,} total records)")

    if include_images:
        print("[3/4] Sampling images and generating thumbnails...")
    else:
        print("[3/4] Skipping image sampling (--no-images)")

    print("[4/4] Rendering HTML report...")
    parts = [
        HTML_HEAD,
        render_topnav(),
        render_hero(sources),
        render_nav(),
        '<div class="container">',
        render_overview(sources),
        render_task_mapping(),
        render_categories(sources, ann_stats, include_images),
        render_coord_section(),
        render_pipeline_section(),
        "</div>",
        HTML_TAIL,
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    size_kb = output.stat().st_size / 1024
    print(f"\n✓ Report written: {output} ({size_kb:.1f} KB)")
    print(f"  Sources: {len(sources)} | Annotations: {total_annotations:,} | Images: {'enabled' if include_images else 'disabled'}")


if __name__ == "__main__":
    main()
