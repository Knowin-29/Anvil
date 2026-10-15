#!/usr/bin/env python3
"""desktop 短名集 + 长描述集「融合」版 (2026-06-18 决定)。

两个源是**同一批物体**的不同指代方式:
  SHORT = prompt_grounding_scheme_ab        (raw_name / human_name 短名)
  LONG  = longdesc_augmented_qwen3_6_plus   (qwen_long_description 长描述)

需求: 同一物体不要把短名+长描述+4模板的全量都塞进去(太多), 而是**按物体特性
  每物体只出 1 条 query**, 有的选长描述、有的选短名:
    - 短名在本图内**唯一**(能区分该物体) → 用**短名**(简洁)。
    - 短名在本图内**歧义**(同名指向 ≥2 个框) → 用**长描述**(消歧)。
  消歧用长描述时, 取该物体 ambiguity_risk 最低、query_rank 最优的一条。
  缺长描述的歧义物体 → 退回短名(并在 manifest 记数)。

输出 tag 统一 <bbox>[[x1,y1],[x2,y2]]</bbox> (system/user/assistant 全改)。
模板在语料层 round-robin 轮换(变相提问增广)。

不覆盖源文件:
  输出  trains/desktop_bbox_fused_{train,val}.jsonl
  manifest trains/desktop_bbox_fused.manifest.json

比例旋钮 --long_ratio (默认 0.25): 强制长描述(歧义/短名缺失)之外, 再从
  「短名唯一但也有长描述可用」的物体里按稳定 hash 提升一批到长描述, 使长描述
  占融合集的目标比例 (落在你要的 20-30%)。

用法: python3 trains/build_desktop_fused.py [--long_ratio 0.25]
"""
import json, os, re, sys, hashlib
from collections import Counter, defaultdict

LONG_RATIO = float(sys.argv[sys.argv.index("--long_ratio") + 1]) if "--long_ratio" in sys.argv else 0.25

BASE = "${DATA_ROOT}/prompt_grounding_table"
SHORT = {
    "train": f"{BASE}/prompt_grounding_scheme_ab_train_msswift_box2pt.jsonl",
    "val":   f"{BASE}/prompt_grounding_scheme_ab_val_msswift_box2pt.jsonl",
}
LONG = {
    "train": f"{BASE}/longdesc_augmented_qwen3_6_plus/prompt_grounding_longdesc_aug_train_msswift_box2pt.jsonl",
    "val":   f"{BASE}/longdesc_augmented_qwen3_6_plus/prompt_grounding_longdesc_aug_val_msswift_box2pt.jsonl",
}
OUT_DIR = "${GSA_FORGE_ROOT}/trains"
MANIFEST = os.path.join(OUT_DIR, "desktop_bbox_fused.manifest.json")
TEMPLATES = ["locate_bbox_v1", "find_bbox_v1", "ground_bbox_v1", "zh_bbox_v1"]


def to_bbox(d):
    for m in d.get("messages", []):
        c = m.get("content")
        if isinstance(c, str) and "<box>" in c:
            m["content"] = c.replace("<box>", "<bbox>").replace("</box>", "</bbox>")
    return d


def obj_key(md):
    return (md.get("image_name", ""), tuple(md.get("bbox") or []))


def pick_template(key_str):
    h = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
    return TEMPLATES[h % len(TEMPLATES)]


def num(x, default=1e9):
    try:
        return float(x)
    except Exception:
        return default


def process(split):
    short_path, long_path = SHORT[split], LONG[split]

    # ---- Pass A: SHORT 源 ----
    # canonical 短名 (raw_name 优先); 图内同名->框集合 (歧义判定); 每物体可选的短名记录(按模板分桶)
    short_records = defaultdict(dict)          # objkey -> {template_id: record}
    short_name = {}                            # objkey -> normalized short name (raw_name)
    img_name_boxes = defaultdict(set)          # (image, name_lower) -> set(bbox)
    n_short = 0
    with open(short_path) as f:
        for line in f:
            d = json.loads(line); md = d.get("metadata", {}) or {}
            ok = obj_key(md); n_short += 1
            tid = md.get("prompt_template_id")
            if md.get("prompt_source") == "raw_name":
                nm = (md.get("normalized_query") or md.get("raw_name") or "").strip()
                short_name[ok] = nm
                img_name_boxes[(ok[0], nm.lower())].add(ok[1])
            # 存所有短名候选(raw_name 优先覆盖 human_name)
            cur = short_records[ok].get(tid)
            if cur is None or md.get("prompt_source") == "raw_name":
                short_records[ok][tid] = d

    # ---- Pass B: LONG 源, 只取真长描述 qwen_long_description ----
    long_best = {}                             # objkey -> (sort_tuple, record)
    n_long = 0
    with open(long_path) as f:
        for line in f:
            d = json.loads(line); md = d.get("metadata", {}) or {}
            if md.get("prompt_source") != "qwen_long_description":
                continue
            n_long += 1
            ok = obj_key(md)
            # 选 ambiguity_risk 最低、query_rank 最优 的一条
            sort_t = (num(md.get("ambiguity_risk")), num(md.get("query_rank")))
            if ok not in long_best or sort_t < long_best[ok][0]:
                long_best[ok] = (sort_t, d)

    # ---- 分类每个物体 ----
    all_objs = set(short_records) | set(long_best)
    forced_long = []        # 歧义 或 短名缺失, 且有长描述 -> 必用长
    short_only = []         # 短名唯一, 但无长描述可用 -> 只能短
    promotable = []         # 短名唯一, 且也有长描述 -> 可被提升到长
    short_no_long_fallback = []  # 歧义但无长描述 -> 退回短
    for ok in all_objs:
        nm = short_name.get(ok, "")
        has_short = ok in short_records
        has_long = ok in long_best
        ambiguous = (len(img_name_boxes.get((ok[0], nm.lower()), set())) > 1) if nm else True
        if (ambiguous or not has_short) and has_long:
            forced_long.append(ok)
        elif ambiguous and not has_long and has_short:
            short_no_long_fallback.append(ok)
        elif has_short and has_long:
            promotable.append(ok)
        elif has_short:
            short_only.append(ok)
    T = len(forced_long) + len(short_only) + len(promotable) + len(short_no_long_fallback)
    target_long = round(LONG_RATIO * T)
    need_promote = max(0, target_long - len(forced_long))
    # 稳定挑选要提升的 promotable (按 hash 排序, 可复现)
    promotable.sort(key=lambda ok: hashlib.md5(str(ok).encode()).hexdigest())
    promote_set = set(promotable[:need_promote])

    # ---- 发射 ----
    out_path = os.path.join(OUT_DIR, f"desktop_bbox_fused_{split}.jsonl")
    fo = open(out_path, "w")
    route = Counter(); written = 0
    for ok in all_objs:
        if ok in promote_set or ok in forced_long:
            rec = long_best[ok][1]
            kind = "long(forced)" if ok in forced_long else "long(promoted)"
        elif ok in short_records:
            buckets = short_records[ok]
            want = pick_template(f"{ok}")
            rec = buckets.get(want) or buckets[sorted(k for k in buckets if k)[0]]
            kind = "short(no-long-fallback)" if ok in short_no_long_fallback else "short(unique)"
        else:
            continue
        rec = to_bbox(json.loads(json.dumps(rec)))   # copy + tag
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1; route[kind] += 1
    fo.close()
    long_out = route.get("long(forced)", 0) + route.get("long(promoted)", 0)
    return {
        "split": split, "output": out_path, "long_ratio_target": LONG_RATIO,
        "short_rows_in": n_short, "long_desc_rows_in": n_long,
        "short_objects": len(short_records), "long_objects": len(long_best),
        "promotable_objects": len(promotable), "promoted": len(promote_set),
        "fused_objects_out": written,
        "long_share_actual": round(long_out / max(written, 1), 4),
        "route": dict(route),
    }


def main():
    results = {}
    ex = None
    for split in ("train", "val"):
        if not (os.path.isfile(SHORT[split]) and os.path.isfile(LONG[split])):
            print(f"[skip] {split} missing source"); continue
        r = process(split); results[split] = r
        print(f"[{split}] fused={r['fused_objects_out']}  long_share={r['long_share_actual']:.1%} "
              f"(target {r['long_ratio_target']:.0%}, promotable={r['promotable_objects']}, promoted={r['promoted']})")
        print(f"        route={r['route']}")
    # 抽 3 条样例
    p = os.path.join(OUT_DIR, "desktop_bbox_fused_train.jsonl")
    samples = []
    if os.path.isfile(p):
        with open(p) as f:
            for i, line in enumerate(f):
                if i in (0, 1, 2):
                    d = json.loads(line)
                    samples.append({
                        "user": d["messages"][1]["content"].replace("\n", " ")[:110],
                        "asst": d["messages"][-1]["content"],
                        "src": d["metadata"].get("prompt_source"),
                    })
                if i > 2: break
    manifest = {
        "script": "trains/build_desktop_fused.py",
        "purpose": "短名集+长描述集按物体融合: 每物体1条query; 模板轮换; tag=<bbox>; long_ratio 控长描述占比",
        "long_ratio": LONG_RATIO,
        "rule": "forced-long: ambiguous/short-missing & has long; promote extra short-unique(with long) by stable hash to hit long_ratio; rest short; no-long ambiguous -> short fallback",
        "tag": "<box> -> <bbox>",
        "sources": {"short": SHORT, "long": LONG},
        "results": results,
        "samples": samples,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print(f"-> manifest {MANIFEST}")
    for s in samples:
        print(f"   [{s['src']}] {s['user']!r} -> {s['asst']}")


if __name__ == "__main__":
    main()
