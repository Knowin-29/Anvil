"""可视化渲染模块 — 在图片上绘制 bbox、空间关系、深度热力图、场景图。

提供四种可视化：
1. draw_grounding  — 原图 + bbox + 类别标签
2. draw_depth_map  — 深度热力图（从 xyz_map 或 depth 数组）
3. draw_scene_graph — 原图 + bbox + 关系箭头 + 谓词标签
4. draw_annotation — 综合标注图（bbox + 属性 + 关系文字）
5. render_to_base64 — 将 PIL Image 转为 base64 data URI
"""

from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- 调色板 ----
# 20 种高对比度颜色，循环使用
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]

# 关系谓词 → 箭头颜色
RELATION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "on-top-of": (230, 25, 75),
    "under": (60, 180, 75),
    "front": (0, 130, 200),
    "behind": (255, 225, 25),
    "inside": (245, 130, 48),
    "contains": (245, 130, 48),
    "left-of": (145, 30, 180),
    "right-of": (70, 240, 240),
    "above": (240, 50, 230),
    "below": (210, 245, 60),
    "next-to": (128, 128, 128),
    "near": (170, 110, 40),
    "occluding": (128, 0, 0),
    "occluded-by": (0, 128, 128),
    "stacked-on": (255, 100, 100),
    "supporting": (100, 255, 100),
    "larger-than": (200, 200, 200),
    "smaller-than": (180, 180, 180),
    "overlapping": (255, 165, 0),
}


def _get_font(size: int = 14):
    """尝试加载字体，fallback 到默认。"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


# =====================================================================
# 1. Grounding 可视化 — 原图 + bbox + 标签
# =====================================================================

def draw_grounding(image_path: str, objects: List[Dict], title: str = "") -> Image.Image:
    """在原图上绘制检测框和类别标签。

    objects: [{"name": str, "bbox": [x1,y1,x2,y2], "confidence": float, "id": int}, ...]
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(14)
    font_sm = _get_font(11)

    for i, obj in enumerate(objects):
        color = PALETTE[i % len(PALETTE)]
        bbox = obj["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 画框（2px 粗）
        for offset in range(2):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

        # 标签背景
        label = f"{obj['name']}"
        if "confidence" in obj:
            label += f" {obj['confidence']:.2f}"
        tw = draw.textlength(label, font=font)
        th = 18
        label_y = max(0, y1 - th - 2)
        draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th], fill=color)
        draw.text((x1 + 3, label_y + 1), label, fill=(255, 255, 255), font=font)

    if title:
        draw.text((5, 5), title, fill=(255, 255, 255), font=_get_font(16))

    return img


# =====================================================================
# 2. 深度热力图
# =====================================================================

def draw_depth_map(
    depth_array: np.ndarray,
    image_path: Optional[str] = None,
    alpha: float = 0.5,
) -> Image.Image:
    """将深度数组渲染为热力图。

    depth_array: (H, W) float 深度值，或 (H, W, 3) xyz_map（取 z 通道）。
    image_path: 如果提供，将热力图叠加到原图上。
    """
    if depth_array.ndim == 3:
        depth = depth_array[:, :, 2]  # z 通道
    else:
        depth = depth_array

    # 处理无效值
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        # 全无效，返回灰色图
        h, w = depth.shape
        return Image.new("RGB", (w, h), (128, 128, 128))

    d_min = np.percentile(depth[valid], 2)
    d_max = np.percentile(depth[valid], 98)
    if d_max <= d_min:
        d_max = d_min + 1

    # 归一化到 0-255
    norm = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
    norm[~valid] = 0

    # 使用 turbo-like colormap（手动实现，不依赖 matplotlib）
    heatmap = _turbo_colormap(norm, valid)
    depth_img = Image.fromarray(heatmap)

    if image_path and os.path.exists(image_path):
        orig = Image.open(image_path).convert("RGB").resize(depth_img.size)
        blended = Image.blend(orig, depth_img, alpha)
        return blended

    return depth_img


def _turbo_colormap(norm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """简化版 turbo colormap: 蓝(近) → 绿 → 黄 → 红(远)。"""
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # 近=蓝, 中=绿/黄, 远=红
    r = np.clip(np.where(norm < 0.5, 0, (norm - 0.5) * 2 * 255), 0, 255).astype(np.uint8)
    g = np.clip(np.where(norm < 0.5, norm * 2 * 255, (1 - norm) * 2 * 255), 0, 255).astype(np.uint8)
    b = np.clip(np.where(norm < 0.5, (0.5 - norm) * 2 * 255, 0), 0, 255).astype(np.uint8)

    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    rgb[~valid] = [40, 40, 40]  # 无效区域深灰

    return rgb


def draw_depth_with_boxes(
    depth_array: np.ndarray,
    objects: List[Dict],
    image_path: Optional[str] = None,
) -> Image.Image:
    """深度热力图 + bbox 叠加。"""
    depth_img = draw_depth_map(depth_array, image_path, alpha=0.5)
    draw = ImageDraw.Draw(depth_img)
    font = _get_font(12)

    for i, obj in enumerate(objects):
        color = PALETTE[i % len(PALETTE)]
        bbox = obj["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        depth_val = obj.get("mean_depth")
        label = obj["name"]
        if depth_val and depth_val > 0:
            label += f" d={depth_val:.0f}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x1, y1 - 16, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), label, fill=(255, 255, 255), font=font)

    return depth_img


# =====================================================================
# 3. Scene Graph 可视化 — bbox + 关系箭头
# =====================================================================

def draw_scene_graph(
    image_path: str,
    objects: List[Dict],
    relations: List[Dict],
    max_relations: int = 40,
    show_3d_only: bool = False,
) -> Image.Image:
    """在原图上绘制场景图：bbox + 关系箭头 + 谓词标签。"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(13)
    font_sm = _get_font(10)

    # 画 bbox
    centers = {}
    for i, obj in enumerate(objects):
        color = PALETTE[i % len(PALETTE)]
        bbox = obj["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        for offset in range(2):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
        cx, cy = _bbox_center(bbox)
        centers[obj["id"]] = (cx, cy)

        # 标签
        label = f"#{obj['id']} {obj['name']}"
        tw = draw.textlength(label, font=font)
        label_y = max(0, y1 - 18)
        draw.rectangle([x1, label_y, x1 + tw + 6, label_y + 17], fill=color)
        draw.text((x1 + 3, label_y + 1), label, fill=(255, 255, 255), font=font)

    # 画关系箭头
    drawn = 0
    for rel in relations:
        if show_3d_only and rel.get("source") != "3d":
            continue
        if drawn >= max_relations:
            break

        s_id = rel["subject"]
        o_id = rel["object"]
        pred = rel["predicate"]

        if s_id not in centers or o_id not in centers:
            continue

        sx, sy = centers[s_id]
        ox, oy = centers[o_id]

        # 箭头颜色
        arrow_color = RELATION_COLORS.get(pred, (200, 200, 200))
        source_tag = rel.get("source", "2d")

        # 画线
        draw.line([(sx, sy), (ox, oy)], fill=arrow_color, width=2)

        # 箭头头部
        _draw_arrowhead(draw, sx, sy, ox, oy, arrow_color, size=8)

        # 谓词标签
        mx, my = (sx + ox) / 2, (sy + oy) / 2
        tag = f"{pred}"
        if source_tag == "3d":
            tag += " [3D]"
        tw = draw.textlength(tag, font=font_sm)
        draw.rectangle([mx - tw / 2 - 2, my - 8, mx + tw / 2 + 2, my + 8],
                        fill=(0, 0, 0, 180) if hasattr(draw, 'textbbox') else (0, 0, 0))
        draw.text((mx - tw / 2, my - 7), tag, fill=(255, 255, 255), font=font_sm)

        drawn += 1

    return img


def _draw_arrowhead(draw, x1, y1, x2, y2, color, size=8):
    """在 (x2, y2) 处画箭头头部。"""
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    # 箭头两翼
    a1 = angle + math.pi * 0.85
    a2 = angle - math.pi * 0.85
    p1 = (x2 + size * math.cos(a1), y2 + size * math.sin(a1))
    p2 = (x2 + size * math.cos(a2), y2 + size * math.sin(a2))
    draw.polygon([(x2, y2), p1, p2], fill=color)


# =====================================================================
# 4. 综合标注图 — bbox + 属性标签
# =====================================================================

def draw_annotation(
    image_path: str,
    objects: List[Dict],
    show_attributes: bool = True,
) -> Image.Image:
    """综合标注图：bbox + 属性（can_pick/push/stack 等）。"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    font_sm = _get_font(10)

    for i, obj in enumerate(objects):
        color = PALETTE[i % len(PALETTE)]
        bbox = obj["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        for offset in range(2):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

        # 名称 + 置信度
        label = f"{obj['name']} {obj.get('confidence', 0):.2f}"
        tw = draw.textlength(label, font=font)
        label_y = max(0, y1 - 16)
        draw.rectangle([x1, label_y, x1 + tw + 4, label_y + 15], fill=color)
        draw.text((x1 + 2, label_y), label, fill=(255, 255, 255), font=font)

        if show_attributes:
            # 属性标签
            attrs = []
            if obj.get("can_pick"):
                attrs.append("pick")
            if obj.get("can_push"):
                attrs.append("push")
            if obj.get("can_stack"):
                attrs.append("stack")
            if obj.get("is_container"):
                attrs.append("container")
            if obj.get("is_surface"):
                attrs.append("surface")
            if attrs:
                attr_text = " ".join(attrs)
                tw2 = draw.textlength(attr_text, font=font_sm)
                draw.rectangle([x1, y2 + 1, x1 + tw2 + 4, y2 + 14], fill=(0, 0, 0))
                draw.text((x1 + 2, y2 + 1), attr_text, fill=(200, 255, 200), font=font_sm)

            # 深度信息
            depth = obj.get("mean_depth")
            if depth and depth > 0:
                d_text = f"d={depth:.0f}"
                tw3 = draw.textlength(d_text, font=font_sm)
                draw.rectangle([x2 - tw3 - 4, y1, x2, y1 + 13], fill=(0, 0, 100))
                draw.text((x2 - tw3 - 2, y1), d_text, fill=(200, 200, 255), font=font_sm)

    return img


# =====================================================================
# 5. 工具函数
# =====================================================================

def render_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """PIL Image → base64 data URI。"""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def save_visualization(img: Image.Image, output_path: str, quality: int = 90):
    """保存可视化图片到磁盘。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".png"):
        img.save(output_path, format="PNG")
    else:
        img.save(output_path, format="JPEG", quality=quality)
    return output_path


def generate_all_visualizations(
    image_path: str,
    grounding_data: Dict,
    scene_graph_data: Dict,
    depth_info_data: Optional[Dict] = None,
    xyz_map: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Image.Image]:
    """为单张图片生成所有可视化，返回 {类型: PIL Image}。"""
    results = {}

    objects_sg = scene_graph_data.get("objects", [])
    objects_gr = []
    for p in grounding_data.get("phrases", []):
        for box in p.get("boxes", []):
            objects_gr.append({
                "name": p["phrase"],
                "bbox": box["bbox"],
                "confidence": box.get("confidence", 0),
                "id": len(objects_gr),
            })

    # 1. Grounding
    results["grounding"] = draw_grounding(image_path, objects_gr, title="Grounding")

    # 2. Annotation (bbox + 属性)
    results["annotation"] = draw_annotation(image_path, objects_sg)

    # 3. Scene Graph (bbox + 关系箭头)
    relations = scene_graph_data.get("relations", [])
    results["scene_graph"] = draw_scene_graph(image_path, objects_sg, relations)

    # 4. Scene Graph (仅 3D 关系)
    relations_3d = [r for r in relations if r.get("source") == "3d"]
    if relations_3d:
        results["scene_graph_3d"] = draw_scene_graph(
            image_path, objects_sg, relations_3d, show_3d_only=True
        )

    # 5. 深度图
    if xyz_map is not None:
        results["depth_heatmap"] = draw_depth_map(xyz_map, image_path, alpha=0.5)
        results["depth_with_boxes"] = draw_depth_with_boxes(xyz_map, objects_sg, image_path)
    elif depth_info_data:
        # 没有 xyz_map，用 object 深度信息画伪深度标注
        results["depth_annotation"] = draw_annotation(image_path, objects_sg, show_attributes=True)

    # 保存到磁盘
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_stem = os.path.splitext(os.path.basename(image_path))[0]
        for vis_type, vis_img in results.items():
            save_visualization(vis_img, os.path.join(output_dir, f"{img_stem}_{vis_type}.jpg"))

    return results
