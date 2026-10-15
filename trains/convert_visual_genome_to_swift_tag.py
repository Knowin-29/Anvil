#!/usr/bin/env python3
"""Visual Genome region_descriptions → RefCoco 风格 MS-Swift bbox grounding。

对齐 trains/convert_refcoco_to_swift_tag_v3.py 的输出格式:
  {
    "id": "vg_<region_id>",
    "messages": [
      {"role":"user","content":"<image>Locate \"<phrase>\" in the image."},
      {"role":"assistant","content":"<bbox>[[x1,y1],[x2,y2]]</bbox>"}
    ],
    "images": ["/abs/.../images/VG_100K/<image_id>.jpg"]
  }

坐标: pixel xywh -> xyxy -> norm1000 = round(coord/dim*1000), clamp [0,1000] (与 refcoco v3 一致)。
W/H 来源: 优先 image_data.json (无需打开图片); 缺失时 PIL fallback。
过滤: 无本地图片 / 退化框(w<=0 或 h<=0) / 空 phrase / 越界无法修复。

只读输入, 不覆盖任何已有文件。
用法: python3 trains/convert_visual_genome_to_swift_tag.py [--limit N]
"""
import json, os, sys

BASE = "${DATA_ROOT}/da/transferred_dataset/visual_genome"
ANNO_DIR = os.path.join(BASE, "annotations")
IMG_DIR = os.path.join(BASE, "images")
REGIONS = os.path.join(ANNO_DIR, "region_descriptions.json")
IMGDATA = os.path.join(ANNO_DIR, "image_data.json")
OUT = os.path.join(BASE, "vg_grounding_swift.jsonl")
MANIFEST = os.path.join(BASE, "vg_grounding_swift.manifest.json")

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

_PIL = None
def pil():
    global _PIL
    if _PIL is None:
        from PIL import Image
        _PIL = Image
    return _PIL


def folder_from_url(url):
    """https://cs.stanford.edu/people/rak248/VG_100K/2.jpg -> 'VG_100K'."""
    if not url:
        return None
    if "VG_100K_2" in url:
        return "VG_100K_2"
    if "VG_100K" in url:
        return "VG_100K"
    return None


def local_image_path(image_id, url):
    """优先用 url 推断子目录; 否则在两个子目录里探测。"""
    folder = folder_from_url(url)
    if folder:
        p = os.path.join(IMG_DIR, folder, f"{image_id}.jpg")
        if os.path.isfile(p):
            return p
    for folder in ("VG_100K", "VG_100K_2"):
        p = os.path.join(IMG_DIR, folder, f"{image_id}.jpg")
        if os.path.isfile(p):
            return p
    return None


def build_image_index():
    """image_id -> {"w","h","url"} from image_data.json。"""
    idx = {}
    data = json.load(open(IMGDATA))
    for im in data:
        iid = im.get("image_id") or im.get("id")
        if iid is None:
            continue
        idx[int(iid)] = {
            "w": im.get("width"),
            "h": im.get("height"),
            "url": im.get("url") or im.get("image_url"),
        }
    return idx


_dim_cache = {}
def img_wh(path):
    if path in _dim_cache:
        return _dim_cache[path]
    try:
        wh = pil().open(path).size
    except Exception:
        wh = None
    _dim_cache[path] = wh
    return wh


def norm1000(v, dim):
    return round(v / dim * 1000)


def main():
    img_idx = build_image_index()
    print(f"image_data: {len(img_idx)} images indexed")

    regions_data = json.load(open(REGIONS))
    print(f"region_descriptions: {len(regions_data)} image groups")

    n = written = 0
    skip_noimg = skip_nodim = skip_badbox = skip_nophrase = 0
    roundtrip = []

    with open(OUT, "w") as fo:
        for grp in regions_data:
            # 兼容 raw VG: 顶层 dict 有 "id"(=image_id) + "regions"
            for r in grp.get("regions", []):
                n += 1
                if LIMIT and n > LIMIT:
                    break
                region_id = r.get("region_id", r.get("id"))
                image_id = r.get("image_id", r.get("image", grp.get("id")))
                phrase = (r.get("phrase") or "").strip()
                if not phrase:
                    skip_nophrase += 1
                    continue
                try:
                    image_id = int(image_id)
                except (TypeError, ValueError):
                    skip_noimg += 1
                    continue

                meta = img_idx.get(image_id, {})
                path = local_image_path(image_id, meta.get("url"))
                if not path:
                    skip_noimg += 1
                    continue

                W = meta.get("w")
                H = meta.get("h")
                if not W or not H:
                    wh = img_wh(path)
                    if not wh:
                        skip_nodim += 1
                        continue
                    W, H = wh

                x, y, w, h = r.get("x"), r.get("y"), r.get("width"), r.get("height")
                if None in (x, y, w, h) or w <= 0 or h <= 0:
                    skip_badbox += 1
                    continue
                # clamp 像素框到图像内
                x1 = max(0, min(x, W))
                y1 = max(0, min(y, H))
                x2 = max(0, min(x + w, W))
                y2 = max(0, min(y + h, H))
                if x2 <= x1 or y2 <= y1:
                    skip_badbox += 1
                    continue

                nb = [norm1000(x1, W), norm1000(y1, H), norm1000(x2, W), norm1000(y2, H)]
                nb = [max(0, min(1000, v)) for v in nb]
                # norm1000 取整后可能塌成零宽/高(亚像素小框), 一并过滤
                if nb[2] <= nb[0] or nb[3] <= nb[1]:
                    skip_badbox += 1
                    continue
                box_str = f"<bbox>[[{nb[0]},{nb[1]}],[{nb[2]},{nb[3]}]]</bbox>"

                rec = {
                    "id": f"vg_{region_id}",
                    "messages": [
                        {"role": "user", "content": f'<image>Locate "{phrase}" in the image.'},
                        {"role": "assistant", "content": box_str},
                    ],
                    "images": [path],
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if len(roundtrip) < 8:
                    roundtrip.append({
                        "img": os.path.basename(path), "wh": [W, H],
                        "pixel_xywh": [x, y, w, h], "norm1000_xyxy": nb,
                        "phrase": phrase[:40],
                    })
            if LIMIT and n > LIMIT:
                break

    manifest = {
        "script": "trains/convert_visual_genome_to_swift_tag.py",
        "source": "ranjaykrishna/visual_genome :: region_descriptions",
        "input_regions": REGIONS, "input_image_data": IMGDATA,
        "image_dir": IMG_DIR, "output": OUT,
        "schema": "tag <bbox>[[x1,y1],[x2,y2]]</bbox>", "coord": "norm1000",
        "format_aligned_with": "trains/convert_refcoco_to_swift_tag_v3.py",
        "read_regions": n, "written": written,
        "skip_no_image": skip_noimg, "skip_no_dim": skip_nodim,
        "skip_bad_box": skip_badbox, "skip_no_phrase": skip_nophrase,
        "roundtrip_samples": roundtrip,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print(f"read={n} written={written} "
          f"skip(noimg={skip_noimg},nodim={skip_nodim},badbox={skip_badbox},nophrase={skip_nophrase})")
    print(f"-> {OUT}")
    for r in roundtrip:
        print(f"   {r['phrase']!r} {r['pixel_xywh']} @ {r['wh']} -> {r['norm1000_xyxy']}")


if __name__ == "__main__":
    main()
