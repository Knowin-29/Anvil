#!/usr/bin/env python3
"""desktop 短名集「提问方式增广」去重 (2026-06-18 决定)。

问题: scheme_ab 短名集对每个 (图,框,指代名) 单位都 ×4 个提问模板
  (locate/find/ground/zh)，造成 4× 提问方式重复。

设计: 不再每单位 ×4。改为**每单位只保留 1 条**，模板在语料层 round-robin
  轮换 → 4 种模板仍均匀出现在整库 = 变相增广，但每个目标不再重复 4 次。
  物体名称增广 (prompt_source=raw_name / human_name) **保留**（不同指代名是
  真实的语义多样性，不算冗余）。

去重单位 key = (image, bbox, normalized_query)  —— 同图同框同指代名的 4 个模板拷贝
轮换种子 = 该 key 的稳定 hash → 同 key 永远选同一模板（可复现、与顺序无关）。

不覆盖源文件:
  输入  prompt_grounding_scheme_ab_{train,val}_msswift_box2pt.jsonl  (只读)
  输出  trains/desktop_bbox_shortname_phrasingdedup_{train,val}.jsonl
  manifest trains/desktop_bbox_shortname_phrasingdedup.manifest.json

用法: python3 trains/dedup_desktop_shortname_phrasing.py
"""
import json, os, hashlib
from collections import Counter, defaultdict

SRC_DIR = "${DATA_ROOT}/prompt_grounding_table"
OUT_DIR = "${GSA_FORGE_ROOT}/trains"
SPLITS = {
    "train": "prompt_grounding_scheme_ab_train_msswift_box2pt.jsonl",
    "val":   "prompt_grounding_scheme_ab_val_msswift_box2pt.jsonl",
}
MANIFEST = os.path.join(OUT_DIR, "desktop_bbox_shortname_phrasingdedup.manifest.json")
TEMPLATES = ["locate_bbox_v1", "find_bbox_v1", "ground_bbox_v1", "zh_bbox_v1"]


def to_bbox_tag(d):
    """把 messages 里的 <box>...</box> 标签统一改成 <bbox>...</bbox> (含 system/user/assistant)。"""
    for m in d.get("messages", []):
        c = m.get("content")
        if isinstance(c, str) and "<box>" in c:
            m["content"] = c.replace("<box>", "<bbox>").replace("</box>", "</bbox>")
    return d


def unit_key(md):
    """同图同框同指代名 = 一个增广单位 (4 个模板拷贝共享此 key)。"""
    img = md.get("image_name", "")
    bbox = tuple(md.get("bbox") or [])
    ref = md.get("normalized_query") or md.get("query") or md.get("raw_name") or ""
    ps = md.get("prompt_source", "")  # raw_name / human_name 各算独立单位 (保留名称增广)
    return f"{img}|{bbox}|{ps}|{ref}"


def chosen_template(key):
    """稳定地为该单位选一个模板 (与读入顺序无关, 可复现)。"""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return TEMPLATES[h % len(TEMPLATES)]


def process(split, fname):
    path = os.path.join(SRC_DIR, fname)
    out_path = os.path.join(OUT_DIR, f"desktop_bbox_shortname_phrasingdedup_{split}.jsonl")
    # 1st pass: 收集每个 unit 实际拥有的模板 (有些 unit 未必 4 个都全)
    unit_templates = defaultdict(set)
    n = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            md = d.get("metadata", {}) or {}
            unit_templates[unit_key(md)].add(md.get("prompt_template_id"))
            n += 1
    # 决定每个 unit 保留哪个模板: 优先 round-robin 选中的; 若该 unit 没有该模板, 退回它拥有的第一个
    keep_tpl = {}
    for k, tpls in unit_templates.items():
        want = chosen_template(k)
        keep_tpl[k] = want if want in tpls else sorted(t for t in tpls if t)[0]
    # 2nd pass: 写出
    written = 0
    tpl_out = Counter(); src_out = Counter()
    fo = open(out_path, "w")
    seen = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            md = d.get("metadata", {}) or {}
            k = unit_key(md)
            if md.get("prompt_template_id") != keep_tpl.get(k):
                continue
            if k in seen:        # 同 unit 同模板理论上唯一; 防御性去重
                continue
            seen.add(k)
            d = to_bbox_tag(d)   # <box> -> <bbox>
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            written += 1
            tpl_out[md.get("prompt_template_id")] += 1
            src_out[md.get("prompt_source")] += 1
    fo.close()
    return {
        "split": split, "input": path, "output": out_path,
        "read": n, "units": len(unit_templates), "written": written,
        "reduction": f"{n}->{written} ({n/max(written,1):.2f}x)",
        "template_dist_after": dict(tpl_out),
        "prompt_source_after": dict(src_out),
    }


def main():
    results = {}
    for split, fname in SPLITS.items():
        if not os.path.isfile(os.path.join(SRC_DIR, fname)):
            print(f"[skip] missing {fname}"); continue
        r = process(split, fname)
        results[split] = r
        print(f"[{split}] read={r['read']} units={r['units']} written={r['written']} ({r['reduction']})")
        print(f"        templates_after={r['template_dist_after']}")
        print(f"        prompt_source_after={r['prompt_source_after']}")
    manifest = {
        "script": "trains/dedup_desktop_shortname_phrasing.py",
        "purpose": "短名集提问方式增广去重: 每(图,框,指代名)单位保留1条, 模板语料层round-robin轮换; 物体名称增广保留",
        "keep_axis": ["物体名称增广 raw_name/human_name (保留)"],
        "collapsed_axis": ["提问方式增广 4模板 -> 1模板/单位 (轮换)"],
        "untouched": "longdesc_augmented_qwen3_6_plus (长描述增广保留不动); state 不处理",
        "tag": "<box> -> <bbox> (system/user/assistant 全改)",
        "templates": TEMPLATES,
        "results": results,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print(f"-> manifest {MANIFEST}")


if __name__ == "__main__":
    main()
