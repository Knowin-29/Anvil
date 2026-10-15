"""Step3.5: 深度估计客户端 — 调用 FFS 单目深度估计 API，获取 xyz_map 并提取物体 3D 信息。

调用流程：
1. 发送单张 RGB 图像 + 默认相机内参 → FFS 服务
2. 接收 xyz_map (H×W×3) numpy 数组
3. 根据 2D bbox 裁剪 xyz_map，计算物体 3D 中心/尺寸/平均深度
"""

from __future__ import annotations

import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger("flywheel.depth_client")

DEFAULT_FFS_URL = "YOUR_FFS_DEPTH_URL"

# 默认相机内参（用户说明不影响相对空间关系计算）
DEFAULT_FX = 500.0
DEFAULT_FY = 500.0
DEFAULT_CX = 640.0
DEFAULT_CY = 400.0


def _build_intrinsic_str(
    fx: float = DEFAULT_FX, fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX, cy: float = DEFAULT_CY,
) -> str:
    """构建 3×3 内参矩阵的空格分隔字符串。"""
    return f"{fx} 0 {cx} 0 {fy} {cy} 0 0 1"


def get_depth_map(
    image_path: str,
    server_url: str = DEFAULT_FFS_URL,
    fx: float = DEFAULT_FX, fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX, cy: float = DEFAULT_CY,
    timeout: int = 60,
) -> Optional[np.ndarray]:
    """单张图像 → xyz_map (H, W, 3)。

    FFS 服务接受单目图像，返回 xyz_map numpy 数组。
    对于单目模式，left_file 和 right_file 传同一张图，baseline 设为虚拟值 0.065m
    （FFS 需要非零 baseline 才能输出有效深度，单目模式下深度为相对值，不影响空间关系计算）。

    Returns:
        xyz_map: (H, W, 3) float32 数组，每个像素的 (x, y, z) 坐标；
                 失败时返回 None。
    """
    K_str = _build_intrinsic_str(fx, fy, cx, cy)
    try:
        with open(image_path, "rb") as f_img:
            img_bytes = f_img.read()

        files = {
            "left_file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg"),
            "right_file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg"),
        }
        data = {"K": K_str, "baseline": "0.065"}

        resp = requests.post(server_url, files=files, data=data, timeout=timeout)
        if resp.status_code != 200:
            logger.warning(f"FFS API 返回 {resp.status_code} for {image_path}: {resp.text[:200]}")
            return None

        xyz_map = np.load(io.BytesIO(resp.content))
        if xyz_map.ndim == 3 and xyz_map.shape[2] == 3:
            return xyz_map.astype(np.float32)

        logger.warning(f"FFS 返回异常形状 {xyz_map.shape} for {image_path}")
        return None

    except Exception as e:
        logger.warning(f"深度估计失败 {image_path}: {e}")
        return None


def get_object_3d_info(xyz_map: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
    """从 xyz_map 提取 bbox 区域的 3D 信息。

    Args:
        xyz_map: (H, W, 3) float32
        bbox: [x1, y1, x2, y2] 2D 边界框

    Returns:
        {"center_3d": [x,y,z], "size_3d": [dx,dy,dz], "mean_depth": float,
         "valid": bool}
    """
    h, w = xyz_map.shape[:2]
    x1 = max(0, int(round(bbox[0])))
    y1 = max(0, int(round(bbox[1])))
    x2 = min(w, int(round(bbox[2])))
    y2 = min(h, int(round(bbox[3])))

    if x2 <= x1 or y2 <= y1:
        return {"center_3d": [0, 0, 0], "size_3d": [0, 0, 0], "mean_depth": 0, "valid": False}

    crop = xyz_map[y1:y2, x1:x2]  # (crop_h, crop_w, 3)

    # 过滤无效点（深度为 0 或 NaN/Inf）
    z_vals = crop[:, :, 2]
    valid_mask = np.isfinite(z_vals) & (z_vals > 0)

    if valid_mask.sum() < 5:
        return {"center_3d": [0, 0, 0], "size_3d": [0, 0, 0], "mean_depth": 0, "valid": False}

    valid_points = crop[valid_mask]  # (N, 3)

    # 用中位数作为中心（比均值更鲁棒）
    center = np.median(valid_points, axis=0).tolist()

    # 尺寸：用 5%-95% 分位数范围
    p5 = np.percentile(valid_points, 5, axis=0)
    p95 = np.percentile(valid_points, 95, axis=0)
    size = (p95 - p5).tolist()

    mean_depth = float(np.median(valid_points[:, 2]))

    return {
        "center_3d": [round(v, 4) for v in center],
        "size_3d": [round(abs(v), 4) for v in size],
        "mean_depth": round(mean_depth, 4),
        "valid": True,
    }


def estimate_depth_for_image(
    image_path: str,
    objects: List[Dict],
    server_url: str = DEFAULT_FFS_URL,
    fx: float = DEFAULT_FX, fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX, cy: float = DEFAULT_CY,
    timeout: int = 60,
) -> Tuple[Optional[np.ndarray], List[Dict]]:
    """为单张图的所有物体计算 3D 信息。

    Returns:
        (xyz_map, enriched_objects): xyz_map 可能为 None（API 失败时），
        enriched_objects 是带 3D 信息的物体列表副本。
    """
    xyz_map = get_depth_map(image_path, server_url, fx, fy, cx, cy, timeout)

    enriched = []
    for obj in objects:
        obj_copy = dict(obj)
        if xyz_map is not None:
            info_3d = get_object_3d_info(xyz_map, obj["bbox"])
            obj_copy.update(info_3d)
        else:
            obj_copy.update({
                "center_3d": [0, 0, 0], "size_3d": [0, 0, 0],
                "mean_depth": 0, "valid": False,
            })
        enriched.append(obj_copy)

    return xyz_map, enriched


def estimate_depth_batch(
    grounding_jsonl: str,
    output_jsonl: str,
    server_url: str = DEFAULT_FFS_URL,
    fx: float = DEFAULT_FX, fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX, cy: float = DEFAULT_CY,
    timeout: int = 60,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """批量深度估计：读取 grounding JSONL，为每张图调用 FFS API，输出带 3D 信息的 JSONL。

    输出 JSONL 每行格式：
    {"image": "...", "objects": [...带3D信息...], "depth_valid": true/false}

    Returns:
        {"count": int, "success": int, "failed": int, "output_jsonl": str}
    """
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    grounding_data: List[Dict] = []
    with open(grounding_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                grounding_data.append(json.loads(line))

    if not grounding_data:
        with open(output_jsonl, "w") as f:
            pass
        return {"count": 0, "success": 0, "failed": 0, "output_jsonl": output_jsonl}

    # 提取每张图的物体列表
    image_tasks: List[Tuple[str, List[Dict], int, int]] = []
    for item in grounding_data:
        image_path = item["image"]
        img_w = item.get("width", 1280)
        img_h = item.get("height", 800)
        objects: List[Dict] = []
        for phrase_data in item.get("phrases", []):
            name = phrase_data["phrase"]
            for box_data in phrase_data.get("boxes", []):
                objects.append({
                    "id": len(objects),
                    "name": name,
                    "bbox": box_data["bbox"],
                    "confidence": box_data.get("confidence", box_data.get("score", 1.0)),
                })
        image_tasks.append((image_path, objects, img_w, img_h))

    # 并发调用 FFS API
    results: List[Dict] = [None] * len(image_tasks)  # type: ignore
    success_count = 0
    failed_count = 0

    def _process_one(idx: int) -> Tuple[int, Dict]:
        img_path, objs, img_w, img_h = image_tasks[idx]
        xyz_map, enriched_objs = estimate_depth_for_image(
            img_path, objs, server_url, fx, fy, cx, cy, timeout,
        )
        depth_valid = xyz_map is not None

        # 保存 xyz_map 到磁盘（用于后续可视化）
        xyz_map_path = None
        if xyz_map is not None:
            depth_dir = os.path.join(os.path.dirname(output_jsonl), "depth_maps")
            os.makedirs(depth_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            xyz_map_path = os.path.join(depth_dir, f"{stem}_xyz.npy")
            np.save(xyz_map_path, xyz_map)

        return idx, {
            "image": img_path,
            "image_size": [img_w, img_h],
            "objects": enriched_objs,
            "depth_valid": depth_valid,
            "xyz_map_path": xyz_map_path,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, i): i for i in range(len(image_tasks))}
        for fut in as_completed(futures):
            try:
                idx, result = fut.result()
                results[idx] = result
                if result["depth_valid"]:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                i = futures[fut]
                logger.warning(f"深度估计异常 idx={i}: {e}")
                img_path, objs, img_w, img_h = image_tasks[i]
                results[i] = {
                    "image": img_path,
                    "image_size": [img_w, img_h],
                    "objects": objs,
                    "depth_valid": False,
                }
                failed_count += 1

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"深度估计: {len(image_tasks)} 张图, 成功 {success_count}, 失败 {failed_count}")
    return {
        "count": len(image_tasks),
        "success": success_count,
        "failed": failed_count,
        "output_jsonl": output_jsonl,
    }
