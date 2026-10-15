"""空间关系查看器 — 按物体交互浏览 Scene Graph。

提供三个核心函数供 app.py 调用：
- sg_explorer_list_images: 列出图片
- sg_explorer_load_objects: 加载物体列表
- sg_explorer_render: 渲染聚焦视图 + 深度图 + 关系表 + 拓扑图
"""

from __future__ import annotations

import html as html_mod
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ---- 调色板 ----
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]

RELATION_COLORS = {
    "on-top-of": (230, 25, 75), "under": (60, 180, 75),
    "front": (0, 130, 200), "behind": (255, 225, 25),
    "inside": (245, 130, 48), "left-of": (145, 30, 180),
    "right-of": (70, 240, 240), "above": (240, 50, 230),
    "below": (210, 245, 60), "next-to": (128, 128, 128),
    "near": (170, 110, 40), "occluding": (128, 0, 0),
    "occluded-by": (0, 128, 128), "stacked-on": (255, 100, 100),
    "supporting": (100, 255, 100), "larger-than": (200, 200, 200),
    "smaller-than": (180, 180, 180), "overlapping": (255, 165, 0),
}


def _get_font(size: int = 14):
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _load_jsonl(path: str) -> list:
    if not path or not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _round_dir(output_root: str, round_idx: int) -> str:
    return os.path.join(output_root, f"round_{int(round_idx)}")


def _resolve_image_path(img_path: str, output_root: str) -> str:
    """解析图片路径：支持绝对路径、相对路径、以及 symlink 导致的路径偏移。"""
    if os.path.exists(img_path):
        return img_path
    # 相对路径：从 output_root 的各级父目录查找
    if not os.path.isabs(img_path):
        base = output_root
        for _ in range(5):
            full = os.path.join(base, img_path)
            if os.path.exists(full):
                return full
            base = os.path.dirname(base)
    # 绝对路径不存在：尝试在 output_root 的祖先中找同名文件
    # 例如 /proj/data/output/.../foo.npy → 在 output_root 对应的目录中查找
    if os.path.isabs(img_path):
        fname = os.path.basename(img_path)
        # 从 output_root 重建等效路径
        # e.g., old: .../data/output/flywheel/round_0/depth_maps/x.npy
        #        new: .../data/data/output/flywheel/round_0/depth_maps/x.npy
        # 提取相对于 "output/flywheel" 的子路径
        for marker in ("output/flywheel/", "output\\flywheel\\"):
            pos = img_path.find(marker)
            if pos >= 0:
                rel_suffix = img_path[pos + len(marker):]
                candidate = os.path.join(output_root, rel_suffix)
                if os.path.exists(candidate):
                    return candidate
    return img_path


def _load_scene(output_root: str, round_idx, image_choice):
    if not image_choice:
        return None, None
    idx = int(image_choice.split(":")[0])
    rd = _round_dir(output_root, int(round_idx))
    sg_data = _load_jsonl(os.path.join(rd, "scene_graph.jsonl"))
    dep_data = _load_jsonl(os.path.join(rd, "depth_info.jsonl"))
    dep_map = {d["image"]: d for d in dep_data}
    if idx >= len(sg_data):
        return None, None
    sg = sg_data[idx]
    orig_img = sg["image"]
    sg["image"] = _resolve_image_path(sg["image"], output_root)
    di = dep_map.get(orig_img)
    if di and di.get("xyz_map_path"):
        di["xyz_map_path"] = _resolve_image_path(di["xyz_map_path"], output_root)
    return sg, di


# =====================================================================
# 聚焦绘图：选中物体高亮，其余灰化
# =====================================================================

def _draw_focused(img_path, objs, rels, focus_id, source_filter):
    img = Image.open(img_path).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(14)
    font_sm = _get_font(11)

    # 过滤关系
    visible_rels = []
    related_ids = set()
    for r in rels:
        if source_filter == "3D only" and r.get("source") != "3d":
            continue
        if source_filter == "2D only" and r.get("source") != "2d":
            continue
        if focus_id is not None and r["subject"] != focus_id and r["object"] != focus_id:
            continue
        visible_rels.append(r)
        related_ids.add(r["subject"])
        related_ids.add(r["object"])
    if focus_id is not None:
        related_ids.add(focus_id)

    # 画 bbox
    centers = {}
    for i, obj in enumerate(objs):
        bbox = obj["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        centers[obj["id"]] = (cx, cy)

        is_active = focus_id is None or obj["id"] in related_ids
        if is_active:
            color = PALETTE[i % len(PALETTE)]
            if obj["id"] == focus_id:
                # 选中：粗框 + 半透明填充
                draw.rectangle([x1, y1, x2, y2], fill=(*color, 40))
                for off in range(3):
                    draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off],
                                   outline=(*color, 255))
            else:
                for off in range(2):
                    draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off],
                                   outline=(*color, 255))
            label = f"#{obj['id']} {obj['name']}"
            tw = draw.textlength(label, font=font)
            ly = max(0, y1 - 18)
            draw.rectangle([x1, ly, x1 + int(tw) + 6, ly + 17], fill=(*color, 220))
            draw.text((x1 + 3, ly + 1), label, fill=(255, 255, 255, 255), font=font)
        else:
            # 非相关：灰色细框
            draw.rectangle([x1, y1, x2, y2], outline=(150, 150, 150, 100))
            label = f"#{obj['id']}"
            tw = draw.textlength(label, font=font_sm)
            draw.rectangle([x1, y1 - 14, x1 + int(tw) + 4, y1], fill=(100, 100, 100, 150))
            draw.text((x1 + 2, y1 - 13), label, fill=(200, 200, 200, 200), font=font_sm)

    # 画关系箭头
    for rel in visible_rels:
        s_id, o_id = rel["subject"], rel["object"]
        if s_id not in centers or o_id not in centers:
            continue
        sx, sy = centers[s_id]
        ox, oy = centers[o_id]
        pred = rel["predicate"]
        base_color = RELATION_COLORS.get(pred, (200, 200, 200))
        is_3d = rel.get("source") == "3d"
        alpha = 230 if is_3d else 160
        ac = (*base_color, alpha)
        w = 2 if is_3d else 1

        draw.line([(sx, sy), (ox, oy)], fill=ac, width=w)

        # 箭头
        angle = math.atan2(oy - sy, ox - sx)
        sz = 9 if is_3d else 6
        a1, a2 = angle + math.pi * 0.85, angle - math.pi * 0.85
        p1 = (ox + sz * math.cos(a1), oy + sz * math.sin(a1))
        p2 = (ox + sz * math.cos(a2), oy + sz * math.sin(a2))
        draw.polygon([(ox, oy), p1, p2], fill=ac)

        # 谓词标签
        mx, my = (sx + ox) / 2, (sy + oy) / 2
        tag = f"{pred} [3D]" if is_3d else pred
        tw = draw.textlength(tag, font=font_sm)
        draw.rectangle([mx - tw / 2 - 2, my - 8, mx + tw / 2 + 2, my + 8], fill=(0, 0, 0, 180))
        draw.text((mx - tw / 2, my - 7), tag, fill=(255, 255, 255, 255), font=font_sm)

    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    return result.convert("RGB")


# =====================================================================
# vis.js 交互式拓扑图
# =====================================================================

def _build_visjs_html(objs, rels, source_filter, focus_id=None):
    filtered = []
    for r in rels:
        if source_filter == "3D only" and r.get("source") != "3d":
            continue
        if source_filter == "2D only" and r.get("source") != "2d":
            continue
        filtered.append(r)

    nodes_js = []
    for i, o in enumerate(objs):
        color = PALETTE[i % len(PALETTE)]
        hx = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        bc = "#ff0000" if o["id"] == focus_id else hx
        bw = 4 if o["id"] == focus_id else 2
        depth = o.get("mean_depth", 0)
        d_str = f"d={depth:.0f}" if depth else "no depth"
        label = html_mod.escape(f"#{o['id']} {o['name']}")

        attrs = []
        for k in ("can_pick", "can_push", "can_stack", "is_container", "is_surface"):
            if o.get(k):
                attrs.append(k.replace("can_", "").replace("is_", ""))
        n_rels = sum(1 for r in filtered if r["subject"] == o["id"] or r["object"] == o["id"])
        title = html_mod.escape(
            f"#{o['id']} {o['name']}\\n{d_str}\\nattrs: {','.join(attrs) or 'none'}\\nrels: {n_rels}"
        )
        nodes_js.append(
            f'{{id:{o["id"]},label:"{label}",color:{{background:"{hx}",border:"{bc}"}},'
            f'borderWidth:{bw},title:"{title}",font:{{color:"#fff",size:12}}}}'
        )

    # 合并同一对节点的多条边
    edge_map = {}
    for r in filtered:
        key = (r["subject"], r["object"])
        edge_map.setdefault(key, []).append(r)

    edges_js = []
    for (s, o), group in edge_map.items():
        labels, any_3d = [], False
        for r in group:
            tag = r["predicate"]
            if r.get("source") == "3d":
                tag += "*"
                any_3d = True
            labels.append(tag)
        label = html_mod.escape(", ".join(labels[:3]))
        if len(labels) > 3:
            label += f" +{len(labels) - 3}"
        ec = "#e74c3c" if any_3d else "#95a5a6"
        w = 2 if any_3d else 1
        edges_js.append(
            f'{{from:{s},to:{o},label:"{label}",arrows:"to",'
            f'color:{{color:"{ec}"}},width:{w},'
            f'font:{{size:9,color:"#333",strokeWidth:2,strokeColor:"#fff"}}}}'
        )

    return f"""
<div id="sg-net" style="width:100%;height:500px;border:1px solid #ddd;border-radius:8px;"></div>
<div style="margin-top:4px;font-size:12px;color:#666;">
  红边 = 3D关系(*) | 灰边 = 2D | 拖拽节点 | 滚轮缩放 | 悬停看详情
</div>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<script>
(function() {{
  var c = document.getElementById('sg-net');
  if (!c) return;
  var n = new vis.DataSet([{",".join(nodes_js)}]);
  var e = new vis.DataSet([{",".join(edges_js)}]);
  new vis.Network(c, {{nodes:n,edges:e}}, {{
    nodes: {{shape:'box',margin:6,font:{{size:12,color:'#fff'}},shadow:true}},
    edges: {{smooth:{{type:'curvedCW',roundness:0.15}},font:{{size:9,align:'middle'}}}},
    physics: {{solver:'forceAtlas2Based',
              forceAtlas2Based:{{gravitationalConstant:-40,springLength:120,springConstant:0.04}},
              stabilization:{{iterations:100}}}},
    interaction: {{hover:true,tooltipDelay:100}}
  }});
}})();
</script>
"""


# =====================================================================
# Gradio 回调函数
# =====================================================================

def list_images(output_root: str, round_idx):
    rd = _round_dir(output_root, int(round_idx))
    sg_data = _load_jsonl(os.path.join(rd, "scene_graph.jsonl"))
    if not sg_data:
        return gr.update(choices=[], value=None), "该轮次无 Scene Graph 数据。"
    choices = []
    for i, sg in enumerate(sg_data):
        img = os.path.basename(sg.get("image", ""))
        n_obj = len(sg.get("objects", []))
        n_rel = len(sg.get("relations", []))
        n_3d = sum(1 for r in sg.get("relations", []) if r.get("source") == "3d")
        choices.append(f"{i}: {img}  ({n_obj} obj, {n_rel} rel, {n_3d} 3D)")
    return gr.update(choices=choices, value=choices[0] if choices else None), f"共 {len(choices)} 张图"


def load_objects(output_root: str, round_idx, image_choice):
    sg, _ = _load_scene(output_root, round_idx, image_choice)
    if sg is None:
        return gr.update(choices=["(全部)"], value="(全部)")
    objs = sg.get("objects", [])
    rels = sg.get("relations", [])
    choices = ["(全部)"]
    for o in objs:
        depth = o.get("mean_depth", 0)
        d_str = f" d={depth:.0f}" if depth else ""
        n = sum(1 for r in rels if r["subject"] == o["id"] or r["object"] == o["id"])
        choices.append(f"#{o['id']} {o['name']}{d_str} ({n} rels)")
    return gr.update(choices=choices, value="(全部)")


def render(output_root: str, round_idx, image_choice, obj_choice, source_filter):
    """渲染聚焦图 + 深度图 + 关系 Markdown + vis.js 拓扑图 HTML。"""
    sg, dep_info = _load_scene(output_root, round_idx, image_choice)
    if sg is None:
        return None, None, "请先选择图片。", "<p style='color:#999'>无数据</p>"

    img_path = sg["image"]
    objs = sg.get("objects", [])
    rels = sg.get("relations", [])

    if not os.path.exists(img_path):
        return None, None, f"图片不存在: {img_path}", ""

    # 解析 focus_id
    focus_id = None
    if obj_choice and obj_choice != "(全部)":
        try:
            focus_id = int(obj_choice.split("#")[1].split(" ")[0])
        except (IndexError, ValueError):
            pass

    # 1. 聚焦可视化
    img_focused = _draw_focused(img_path, objs, rels, focus_id, source_filter)

    # 2. 深度图
    img_depth = None
    if dep_info and dep_info.get("xyz_map_path") and os.path.exists(dep_info["xyz_map_path"]):
        try:
            import numpy as np
            from apps.data_flywheel.visualizer import draw_depth_with_boxes
            xyz = np.load(dep_info["xyz_map_path"])
            img_depth = draw_depth_with_boxes(xyz, objs, img_path)
        except Exception:
            pass

    # 3. 关系 Markdown
    filtered = []
    for r in rels:
        if source_filter == "3D only" and r.get("source") != "3d":
            continue
        if source_filter == "2D only" and r.get("source") != "2d":
            continue
        if focus_id is not None and r["subject"] != focus_id and r["object"] != focus_id:
            continue
        filtered.append(r)

    parts = []
    if focus_id is not None:
        fo = next((o for o in objs if o["id"] == focus_id), None)
        if fo:
            depth = fo.get("mean_depth", 0)
            c3d = fo.get("center_3d")
            c3d_s = f"({c3d[0]:.1f}, {c3d[1]:.1f}, {c3d[2]:.1f})" if c3d and c3d != [0, 0, 0] else "-"
            attrs = []
            for k, short in [("can_pick", "pick"), ("can_push", "push"), ("can_stack", "stack"),
                              ("is_container", "container"), ("is_surface", "surface")]:
                if fo.get(k):
                    attrs.append(short)
            parts.append(f"### #{focus_id} {fo['name']}")
            parts.append(f"- **深度**: {depth:.0f}  |  **3D中心**: {c3d_s}")
            parts.append(f"- **属性**: {', '.join(attrs) or 'none'}")
            parts.append(f"- **关系**: {len(filtered)} 条 (filter: {source_filter})\n")
    else:
        n_3d = sum(1 for r in rels if r.get("source") == "3d")
        parts.append(f"### 全局 — {len(objs)} 物体, {len(filtered)} 关系 "
                     f"(filter: {source_filter}, 总: {len(rels)}, 3D: {n_3d})\n")

    # 按谓词分组
    pred_groups: Dict[str, list] = {}
    for r in filtered:
        pred_groups.setdefault(r["predicate"], []).append(r)

    for pred in sorted(pred_groups.keys()):
        group = pred_groups[pred]
        sources = set(r.get("source", "2d") for r in group)
        src_tag = " [3D]" if sources == {"3d"} else (" [2D]" if sources == {"2d"} else "")
        parts.append(f"**{pred}{src_tag}** ({len(group)})")
        for r in group:
            sn = objs[r["subject"]]["name"] if r["subject"] < len(objs) else "?"
            on = objs[r["object"]]["name"] if r["object"] < len(objs) else "?"
            arrow = ""
            if focus_id is not None:
                arrow = " ➜" if r["subject"] == focus_id else " ⬅"
            src = r.get("source", "2d").upper()
            parts.append(f"- {arrow} #{r['subject']} {sn} → #{r['object']} {on}  "
                         f"({src}, conf={r.get('confidence', '')})")
        parts.append("")

    # 全局模式下显示节点表
    if focus_id is None:
        parts.append("---\n#### 节点总览")
        parts.append("| ID | 名称 | 深度 | pick | push | stack | container | surface | rels |")
        parts.append("|---|---|---|---|---|---|---|---|---|")
        for o in objs:
            d = f"{o.get('mean_depth', 0):.0f}" if o.get('mean_depth') else "-"
            nr = sum(1 for r in filtered if r["subject"] == o["id"] or r["object"] == o["id"])
            parts.append(
                f"| {o['id']} | {o['name']} | {d} "
                f"| {'Y' if o.get('can_pick') else 'N'} "
                f"| {'Y' if o.get('can_push') else 'N'} "
                f"| {'Y' if o.get('can_stack') else 'N'} "
                f"| {'Y' if o.get('is_container') else 'N'} "
                f"| {'Y' if o.get('is_surface') else 'N'} "
                f"| {nr} |"
            )

    relation_md = "\n".join(parts)

    # 4. vis.js 拓扑图
    graph_html = _build_visjs_html(objs, rels, source_filter, focus_id)

    return img_focused, img_depth, relation_md, graph_html
