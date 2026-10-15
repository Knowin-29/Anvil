#!/usr/bin/env python3
"""refcoco → tag 风格 <box> + 0-1000 重转 (Phase A, v2)。

v3 与 v2 的唯一区别: box 标签 <box> -> <bbox>。以下为 v2 相对 v1 的区别 (按用户 2026-06-18 指示):
  - 去掉 system 格式说明 (不再显式教 "bbox grounding mode / return as <box>...")。
  - user 消息改成自然的定位指令, 不提坐标/格式/JSON。
  让模型从 assistant 端的 <box> 输出里"默认学会"该格式, 而非靠指令钉死。

坐标与 v1 一致: norm1000 = round(coord/原图尺寸*1000), 已验证与 ms-swift 对齐。
标签 = <bbox>...</bbox> (用户 2026-06-18 指定, 取代 v2 的 <box>)。

不覆盖任何原文件:
  输入  trains/refcoco_grounding.jsonl                  (只读)
  输出  trains/refcoco_grounding_tag_norm1000_v3.jsonl  (新文件, 不动 v1)
  manifest trains/refcoco_grounding_tag_norm1000_v3.manifest.json

用法: python3 trains/convert_refcoco_to_swift_tag_v3.py [--limit N]
"""
import json, os, sys
from PIL import Image

ROOT = "${GSA_FORGE_ROOT}"
SRC = os.path.join(ROOT, "trains/refcoco_grounding.jsonl")
OUT = os.path.join(ROOT, "trains/refcoco_grounding_tag_norm1000_v3.jsonl")
MANIFEST = os.path.join(ROOT, "trains/refcoco_grounding_tag_norm1000_v3.manifest.json")

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

_dim_cache = {}
def img_wh(path):
    if path in _dim_cache:
        return _dim_cache[path]
    try:
        wh = Image.open(path).size
    except Exception:
        wh = None
    _dim_cache[path] = wh
    return wh

def norm1000(b, W, H):
    return [round(b[0]/W*1000), round(b[1]/H*1000), round(b[2]/W*1000), round(b[3]/H*1000)]

def main():
    n=0; written=0; skip_noimg=0; skip_badbox=0; skip_dim=0
    roundtrip=[]
    with open(SRC) as f, open(OUT, "w") as fo:
        for line in f:
            n += 1
            if LIMIT and n > LIMIT: break
            d = json.loads(line)
            imgs = d.get("images") or []
            objs = d.get("objects") or {}
            refs = objs.get("ref") or []
            bboxes = objs.get("bbox") or []
            if not imgs or not os.path.isfile(imgs[0]):
                skip_noimg += 1; continue
            if not bboxes or not refs:
                skip_badbox += 1; continue
            wh = img_wh(imgs[0])
            if not wh:
                skip_dim += 1; continue
            W, H = wh
            ref = str(refs[0]).strip()
            nb = norm1000(bboxes[0], W, H)
            nb = [max(0, min(1000, v)) for v in nb]
            box_str = f"<bbox>[[{nb[0]},{nb[1]}],[{nb[2]},{nb[3]}]]</bbox>"
            # 自然定位指令, 不含任何格式/坐标说明
            rec = {
                "id": d.get("id", f"refcoco_{n}"),
                "messages": [
                    {"role": "user", "content": f'<image>Locate "{ref}" in the image.'},
                    {"role": "assistant", "content": box_str},
                ],
                "images": imgs,
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if len(roundtrip) < 5:
                roundtrip.append({"img": imgs[0].split("/")[-1], "wh": [W, H],
                                  "pixel": bboxes[0], "norm1000": nb, "ref": ref})

    manifest = {
        "script": "trains/convert_refcoco_to_swift_tag_v3.py",
        "input": SRC, "output": OUT,
        "schema": "tag <bbox>[[x1,y1],[x2,y2]]</bbox>", "coord": "norm1000",
        "phase": "A", "version": "v2",
        "diff_from_v1": "去掉 system 格式说明 + user 改自然定位指令(不提格式/坐标)",
        "written": written, "skip_no_image": skip_noimg,
        "skip_bad_box": skip_badbox, "skip_dim_fail": skip_dim,
        "roundtrip_samples": roundtrip,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print(f"read={n} written={written} skip(noimg={skip_noimg},badbox={skip_badbox},dim={skip_dim})")
    print(f"-> {OUT}")
    for r in roundtrip:
        print(f"   {r['ref'][:30]!r} {r['pixel']} @ {r['wh']} -> norm1000 {r['norm1000']}")

if __name__ == "__main__":
    main()
