#!/usr/bin/env python3
"""all_correct → 训练集 (Phase B, v2: 修正坐标路由)。

v1 的两个 bug (用户 2026-06-18 指出"很多是json/其它格式"):
  (a) 把"含 bbox 键的复合答案"(planning/affordance/pose: 还带 goal_pixel/pose/
      confidence/reason/visible/state 等)也压成了 <bbox>, 破坏了内容。
  (b) 漏掉了裸坐标 [x,y,x,y] / (x,y,x,y) / 裸 point, 留作原文未转。

v2 规则 = **结构化判定, 只转纯 grounding**:
  - 解析答案(剥 ```json 围栏后 json.loads; 失败再退回裸元组正则)。
  - "纯 grounding" = 坐标 + 至多 label, 不含任何其它语义键(object_name/pose/
    confidence/goal_pixel/affordance_point/reason/visible/state/attributes/... 都算"其它")。
  - 纯 bbox(单/多框, 含裸数组/元组) → 每框一个 <bbox>[[x1,y1],[x2,y2]]</bbox> 0-1000。
  - 纯 point(单/多点) → <point>[[x,y],...]</point> 0-1000。
  - 其余一律 **原生保留**(复合 schema/pose/属性/选项/yes-no/自然语言/planning)。
  坐标基准: 像素(相对图实际 W×H) → round/dim*1000。丢 <FAILED>/空/无图。

不覆盖原文件:
  输入  ${DATA_ROOT}/datasets/correct-balanced-train-test-v1/train.jsonl   (只读)
  输出  trains/gsa_allcorrect_train_v2.jsonl   (不动 v1)
  manifest trains/gsa_allcorrect_train_v2.manifest.json

用法: python3 trains/build_allcorrect_train_v2.py [--limit N]
"""
import json, os, re, sys
from collections import Counter
from PIL import Image

ROOT = "${GSA_FORGE_ROOT}"
SRC = "${DATA_ROOT}/datasets/correct-balanced-train-test-v1/train.jsonl"
OUT = os.path.join(ROOT, "trains/gsa_allcorrect_train_v2.jsonl")
MANIFEST = os.path.join(ROOT, "trains/gsa_allcorrect_train_v2.manifest.json")
OLD = "${DATA_ROOT}/qwen-proxy-log-mirror"
NEW = "${DATA_ROOT}/datasets/qwen-proxy-log-mirror-fast"
BAD = {"<FAILED>", "FAILED", ""}

# 允许出现的"grounding 键"; 出现任何其它键 → 视为复合, 保持原生
ALLOWED_KEYS = {"bbox", "bbox_2d", "point_2d", "label"}
RE_FENCE = re.compile(r'^```(?:json)?\s*|\s*```$', re.S)
RE_BARE4 = re.compile(r'^[\(\[]\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]$')
RE_BARE2 = re.compile(r'^[\(\[]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]$')

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

_dim = {}
def img_wh(path):
    if path in _dim:
        return _dim[path]
    try:
        wh = Image.open(path).size
    except Exception:
        wh = None
    _dim[path] = wh
    return wh

def c1000(v, span):
    return max(0, min(1000, round(float(v) / span * 1000)))

def get_user(d):
    for m in (d.get("messages") or []):
        if m.get("role") == "user":
            u = str(m.get("content", ""))
            if u.strip():
                return u
    return str(d.get("prompt", ""))

def collect_keys(obj, keys):
    """递归收集所有 dict key。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            collect_keys(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            collect_keys(v, keys)

def collect_coords(obj, boxes, points):
    """递归提取 bbox(4) / point(2) 数值。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if kl in ("bbox", "bbox_2d") and isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
                boxes.append(v)
            elif kl == "point_2d" and isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                points.append(v)
            else:
                collect_coords(v, boxes, points)
    elif isinstance(obj, list):
        # 裸 [x,y,x,y] / [x,y]
        if len(obj) == 4 and all(isinstance(x, (int, float)) for x in obj):
            boxes.append(obj)
        elif len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            points.append(obj)
        else:
            for v in obj:
                collect_coords(v, boxes, points)

def classify(a):
    """返回 (kind, boxes, points)。kind in {bbox,point,native}。"""
    core = RE_FENCE.sub("", a.strip()).strip()
    # 1) 裸元组(json 解析不了的带括号四元组)
    m = RE_BARE4.match(core)
    if m:
        return "bbox", [[int(x) for x in m.groups()]], []
    m = RE_BARE2.match(core)
    if m:
        return "point", [], [[int(x) for x in m.groups()]]
    # 2) json 结构
    try:
        obj = json.loads(core)
    except Exception:
        return "native", [], []
    keys = set(); collect_keys(obj, keys)
    extra = {k for k in keys if k.lower() not in ALLOWED_KEYS}
    if extra:
        return "native", [], []          # 复合 schema → 原生
    boxes = []; points = []
    collect_coords(obj, boxes, points)
    if boxes and not points:
        return "bbox", boxes, []
    if points and not boxes:
        return "point", [], points
    return "native", [], []              # 混合或无坐标 → 原生

def main():
    n = 0; written = 0
    route = Counter(); skip = Counter()
    ex = []
    fo = open(OUT, "w")
    with open(SRC) as f:
        for line in f:
            n += 1
            if LIMIT and n > LIMIT: break
            d = json.loads(line)
            if d.get("overall") != "all_correct":
                skip["not_allcorrect"] += 1; continue
            a = str(d.get("model_answer", "")).strip()
            if a in BAD:
                skip["failed_empty"] += 1; continue
            imgs = [p.replace(OLD, NEW) for p in (d.get("images") or [])]
            if not imgs or not os.path.isfile(imgs[0]):
                skip["no_image"] += 1; continue

            kind, boxes, points = classify(a)
            if kind in ("bbox", "point"):
                wh = img_wh(imgs[0])
                if not wh:
                    skip["dim_fail"] += 1; continue
                W, H = wh
            if kind == "bbox":
                tags = []
                for b in boxes:
                    nb = [c1000(b[0], W), c1000(b[1], H), c1000(b[2], W), c1000(b[3], H)]
                    tags.append(f"<bbox>[[{nb[0]},{nb[1]}],[{nb[2]},{nb[3]}]]</bbox>")
                new_ans = "".join(tags)
            elif kind == "point":
                pts = [f"[{c1000(p[0], W)},{c1000(p[1], H)}]" for p in points]
                new_ans = f"<point>[{','.join(pts)}]</point>"
            else:
                new_ans = a   # native 原样

            u = get_user(d)
            if "<image>" not in u:
                u = "<image>\n" + u
            rec = {
                "id": d.get("id", f"allc_{n}"),
                "req_id": d.get("req_id"),
                "messages": [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": new_ans},
                ],
                "images": imgs,
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1; route[kind] += 1
            if len(ex) < 10 and kind != "native":
                ex.append({"kind": kind, "old": a[:60].replace("\n", " "), "new": new_ans[:70]})
    fo.close()

    manifest = {
        "script": "trains/build_allcorrect_train_v2.py",
        "input": SRC, "output": OUT,
        "filter": "overall=='all_correct' (strict) + 丢 <FAILED>/空/无图",
        "routing": "结构化判定: 只转纯grounding(坐标+至多label); 复合schema/pose/属性/planning 全部原生",
        "schema": "纯grounding -> <bbox>/<point> 0-1000; 其余原生", "coord": "norm1000(像素/图尺寸)",
        "fix_vs_v1": "(a)不再把含bbox键的复合答案压成<bbox>; (b)补齐裸数组/元组坐标",
        "phase": "B", "version": "v2",
        "read": n, "written": written,
        "route": dict(route), "skip": dict(skip),
        "samples": ex,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print(f"read={n} written={written}")
    print(f"  route={dict(route)}")
    print(f"  skip={dict(skip)}")
    print(f"-> {OUT}")
    for e in ex:
        print(f"   [{e['kind']}] {e['old']!r} -> {e['new']!r}")

if __name__ == "__main__":
    main()
