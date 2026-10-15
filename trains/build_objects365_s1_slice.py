#!/usr/bin/env python3
"""Objects365 → Stage1 切片 (2026-06-21)。

从已清洗的 train_grounding_singleturn_1m.jsonl(1M, 带 objects.{type,negative} 元字段)
按用途重配比, 产出 Stage1 检测主干切片:
  - single (单目标定位, 最对口 eval) : 全留
  - all    (某类全部, 多目标)        : 全留
  - enum   (全图枚举, 冗长多框)      : 压到 ENUM_KEEP
  - neg    (拒答, 原 29%)            : cap 到输出的 NEG_FRAC

稳定 hash 采样(可复现, 与顺序无关)。不覆盖源。
输入  ${DATA_ROOT}/datasets/transferred_datasets/objects365_qa/train_grounding_singleturn_1m.jsonl
输出  trains/objects365_s1_slice.jsonl  + .manifest.json
"""
import json, os, hashlib
from collections import Counter

SRC = "${DATA_ROOT}/datasets/transferred_datasets/objects365_qa/train_grounding_singleturn_1m.jsonl"
OUT = "${GSA_FORGE_ROOT}/trains/objects365_s1_slice.jsonl"
MANIFEST = OUT.replace(".jsonl", ".manifest.json")

ENUM_KEEP = 0.30        # enum 保留比例
NEG_FRAC = 0.12         # 负样本占输出目标比例(cap)


def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def key(d):
    return (d.get("images") or [""])[0] + "|" + d["messages"][-1]["content"][:40]


def main():
    # pass1: 统计各 type 数量, 算 neg 保留比例
    cnt = Counter()
    with open(SRC) as f:
        for line in f:
            cnt[json.loads(line).get("objects", {}).get("type")] += 1
    kept_nonneg = cnt["single"] + cnt["all"] + round(cnt["enum"] * ENUM_KEEP)
    neg_target = round(kept_nonneg / (1 - NEG_FRAC) * NEG_FRAC)
    neg_keep_frac = min(1.0, neg_target / max(1, cnt["neg"]))
    print(f"源 type 分布: {dict(cnt)}")
    print(f"非neg保留≈{kept_nonneg:,}  neg目标≈{neg_target:,}  neg保留比={neg_keep_frac:.3f}")

    # pass2: 写出
    out = Counter(); n = 0
    with open(SRC) as fin, open(OUT, "w") as fo:
        for line in fin:
            d = json.loads(line); n += 1
            ty = d.get("objects", {}).get("type")
            k = key(d)
            keep = (ty in ("single", "all")
                    or (ty == "enum" and h01("e" + k) < ENUM_KEEP)
                    or (ty == "neg" and h01("n" + k) < neg_keep_frac))
            if keep:
                fo.write(line); out[ty] += 1
    total = sum(out.values())
    manifest = {
        "script": "trains/build_objects365_s1_slice.py", "src": SRC, "out": OUT,
        "params": {"ENUM_KEEP": ENUM_KEEP, "NEG_FRAC": NEG_FRAC},
        "src_type_dist": dict(cnt),
        "out_type_dist": dict(out), "out_total": total,
        "out_neg_frac": round(out["neg"] / max(1, total), 4),
    }
    json.dump(manifest, open(MANIFEST, "w"), ensure_ascii=False, indent=2)
    print(f"-> {OUT}  total={total:,}  type={dict(out)}  neg_frac={out['neg']/max(1,total):.1%}")


if __name__ == "__main__":
    main()
