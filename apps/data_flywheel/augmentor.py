"""Step1: 图片增强 — 随机裁剪/翻转/色彩抖动/遮挡合成。"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageEnhance, ImageFilter


def _random_crop(img: Image.Image, ratio: Tuple[float, float] = (0.7, 0.95)) -> Image.Image:
    w, h = img.size
    r = random.uniform(*ratio)
    nw, nh = int(w * r), int(h * r)
    x0 = random.randint(0, w - nw)
    y0 = random.randint(0, h - nh)
    return img.crop((x0, y0, x0 + nw, y0 + nh))


def _random_flip(img: Image.Image) -> Image.Image:
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _color_jitter(img: Image.Image) -> Image.Image:
    for enhancer_cls in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
        factor = random.uniform(0.7, 1.3)
        img = enhancer_cls(img).enhance(factor)
    return img


def _random_blur(img: Image.Image) -> Image.Image:
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img


def _random_occlusion(img: Image.Image) -> Image.Image:
    """在图片上随机放置一个灰色矩形模拟遮挡。"""
    if random.random() > 0.5:
        return img
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bw, bh = random.randint(w // 10, w // 4), random.randint(h // 10, h // 4)
    x0, y0 = random.randint(0, w - bw), random.randint(0, h - bh)
    gray = random.randint(80, 180)
    draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(gray, gray, gray))
    return img


def augment_single(img: Image.Image) -> Image.Image:
    """对单张图片做一次随机增强组合。"""
    img = _random_flip(img)
    img = _random_crop(img)
    img = _color_jitter(img)
    img = _random_blur(img)
    img = _random_occlusion(img)
    return img


def augment_directory(
    image_dir: str,
    output_dir: str,
    augment_per_image: int = 2,
    max_images: int = 50,
) -> List[str]:
    """对目录下的图片做增强，返回所有增强后图片路径（含原图）。"""
    os.makedirs(output_dir, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    src_paths = sorted([
        str(p) for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in exts
    ])[:max_images]

    all_paths: List[str] = []
    for src in src_paths:
        all_paths.append(src)  # 保留原图
        img = Image.open(src).convert("RGB")
        stem = Path(src).stem
        for i in range(augment_per_image):
            aug = augment_single(img.copy())
            out_name = f"{stem}_aug{i}.jpg"
            out_path = os.path.join(output_dir, out_name)
            aug.save(out_path, quality=90)
            all_paths.append(out_path)
    return all_paths
