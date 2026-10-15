#!/usr/bin/env python3
"""all_correct → 训练集 (Phase B, native 版)。

用户 2026-06-18 决定: **保留 all_correct 原始答案的"奇怪格式"原样不动**, 当作额外的
指令遵循数据。 → 不做任何 bbox/point→tag 转换、不归一坐标、不剥 ```json 围栏。
只做:
  - 取 overall=='all_correct' (严格, 排除 inconclusive 噪声)。
  - 丢 <FAILED>/空/无图。
  - prompt 原样 + assistant = model_answer 逐字保留。
  - 图片路径 OLD→NEW 重映射。

(这版取代会转 tag 的 v1/v2; 那两份文件保留未删。)

不覆盖原文件:
  输入  ${DATA_ROOT}/datasets/correct-balanced-train-test-v1/train.jsonl   (只读)
  输出  trains/gsa_allcorrect_train_native_v1.jsonl
  manifest trains/gsa_allcorrect_train_native_v1.manifest.json

用法: python3 trains/build_allcorrect_train_native.py [--limit N]
"""
import json, os, re, sys
from collections import Counter

ROOT = "${GSA_FORGE_ROOT}"
SRC = "${DATA_ROOT}/datasets/correct-balanced-train-test-v1/train.jsonl"
OUT = os.path.join(ROOT, "trains/gsa_allcorrect_train_native_v1.jsonl")
MANIFEST = os.path.join(ROOT, "trains/gsa_allcorrect_train_native_v1.manifest.json")
OLD = "${DATA_ROOT}/qwen-proxy-log-mirror"
NEW = "${DATA_ROOT}/datasets/qwen-proxy-log-mirror-fast"
# 只丢真正的空答案; <FAILED> 是"目标不存在/不可定位"的合法拒答(grounding 负样本, 防乱画框), 保留。
BAD = {""}

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

def get_user(d):
    for m in (d.get("messages") or []):
        if m.get("role") == "user":
            u = str(m.get("content", ""))
            if u.strip():
                return u
    return str(d.get("prompt", ""))

def ans_shape(a):
    """仅用于统计(不改答案): 看原始格式分布。"""
    s = a.strip()
    if '"bbox' in s: return "json_bbox"
    if '"point' in s: return "json_point"
    if s.startswith("{") or s.startswith("[") or "```json" in s: return "other_json"
    if re.match(r'^[\(\[]?\s*\d+\s*,\s*\d+', s): return "tuple/array_coord"
    if re.fullmatch(r'\(?[A-Da-d]\)?\.?', s): return "choice"
    if re.fullmatch(r'(yes|no|true|false|是|否)', s, re.I): return "yes_no"
    if len(s) <= 40: return "short_text"
    return "long_text"

def main():
    n = 0; written = 0
    skip = Counter(); shapes = Counter()
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
            u = get_user(d)
            if "<image>" not in u:
                u = "<image>\n" + u
            rec = {
                "id": d.get("id", f"allc_{n}"),
                "req_id": d.get("req_id"),
                "messages": [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},   # 逐字保留原始答案
                ],
                "images": imgs,
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            shapes[ans_shape(a)] += 1
    fo.close()

    manifest = {
        "script": "trains/build_allcorrect_train_native.py",
        "input": SRC, "output": OUT,
        "policy": "保留原始答案格式逐字不动(指令遵循数据); 不转tag/不归一坐标",
        "filter": "overall=='all_correct' (strict) + 丢 <FAILED>/空/无图",
        "phase": "B", "version": "native_v1",
        "read": n, "written": written,
        "skip": dict(skip),
        "answer_shape_dist(原始,仅统计)": dict(shapes.most_common()),
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print(f"read={n} written={written}")
    print(f"  skip={dict(skip)}")
    print(f"  原始答案格式分布={dict(shapes.most_common())}")
    print(f"-> {OUT}")

if __name__ == "__main__":
    main()
