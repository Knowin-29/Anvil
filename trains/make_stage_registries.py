#!/usr/bin/env python3
"""从基础 registry 派生两段式训练的 stage1/stage2 registry (只改 ratio, 其余原样)。

两段式 (用户 2026-06-18):
  Stage 1 = 地基(广、公开为主): in-domain all_correct 暂不引入。
  Stage 2 = in-domain 专精: gsa_allcorrect_train_native 为主力 + 少量 stage1 回放防遗忘。
  接受 tag(<bbox>/<point>) 与 json({"bbox"}) 两种格式共存(增强指令遵循)。

规则:
  - IN_DOMAIN = gsa_allcorrect_train_native: stage1=0, stage2=IN_DOMAIN_RATIO。
  - ALWAYS_OFF(被取代/排除): refcoco_grounding, pixmo_points(两处), gsa_volc_flywheel_sft,
    aobjstate_verifier → 两段都 0。
  - 其它源: stage1=基础 ratio; stage2=基础 ratio × REPLAY(薄回放)。

输出: data/sources_registry.stage1.yaml / data/sources_registry.stage2.yaml
运行后用: python trains/build_msswift_from_registry.py --registry data/sources_registry.stage1.yaml ...
"""
import os, yaml, copy

ROOT = "${GSA_FORGE_ROOT}"
BASE = os.path.join(ROOT, "data/sources_registry.yaml")
OUT1 = os.path.join(ROOT, "data/sources_registry.stage1.yaml")
OUT2 = os.path.join(ROOT, "data/sources_registry.stage2.yaml")

REPLAY = 0.15                     # stage2 对 stage1 curated 源的回放比例
# in-domain 源: stage1=0(暂不引入), stage2=指定 ratio。这些是干净的 in-domain 主力。
IN_DOMAIN_STAGE2 = {
    "gsa_allcorrect_train_native": 2.0,   # verifier 严格 all_correct, 26.7万
    "desktop_bbox_train": 1.0,            # 桌面人工标注 bbox+affordance, 48.2万(原 RESERVED, 现入 SFT stage2)
}
# 两段都 0: 被取代的旧源 + 排除的飞轮 + **raw 全量 proxy_log(=all_correct 的噪声超集, 只用其干净子集)**
ALWAYS_OFF = {"refcoco_grounding", "pixmo_points", "gsa_volc_flywheel_sft",
              "aobjstate_verifier", "proxy_log_sqlite"}
STAGE2_OFF_RAW = True             # stage2 跳过其余 format=='raw' 源(需extract、未验证)


def gen(base, stage):
    reg = copy.deepcopy(base)
    srcs = reg.get("sources", [])
    est1 = est2 = 0.0
    top1 = []; top2 = []
    for s in srcs:
        name = s.get("name", "")
        base_ratio = float(s.get("ratio", 0) or 0)
        est = float(s.get("est_samples", 0) or 0)
        is_raw = str(s.get("format", "")) == "raw"
        if name in ALWAYS_OFF:
            r = 0.0
        elif name in IN_DOMAIN_STAGE2:
            r = 0.0 if stage == 1 else IN_DOMAIN_STAGE2[name]
        elif stage == 2 and STAGE2_OFF_RAW and is_raw:
            r = 0.0                # raw 巨源不进 stage2 薄回放
        else:
            r = base_ratio if stage == 1 else round(base_ratio * REPLAY, 4)
        s["ratio"] = r
        eff = r * est
        if stage == 1: est1 += eff; (top1.append((name, eff)) if eff > 0 else None)
        else: est2 += eff; (top2.append((name, eff)) if eff > 0 else None)
    return reg, (est1 if stage == 1 else est2), (top1 if stage == 1 else top2)


def main():
    base = yaml.safe_load(open(BASE))
    for stage, out in [(1, OUT1), (2, OUT2)]:
        reg, est, top = gen(base, stage)
        with open(out, "w") as f:
            yaml.safe_dump(reg, f, allow_unicode=True, sort_keys=False, width=4096)
        top.sort(key=lambda x: -x[1])
        print(f"\n=== Stage {stage} -> {out}")
        print(f"  估算有效样本(Σ ratio×est) ≈ {est:,.0f}")
        idom = sum(e for n, e in top if n in IN_DOMAIN_STAGE2)
        if stage == 2:
            print(f"  in-domain 占比 ≈ {idom/est*100:.0f}%  (in-domain={idom:,.0f}, 回放={REPLAY})")
        print("  top10 源(有效量):")
        for n, e in top[:10]:
            print(f"    {e:12,.0f}  {n}")


if __name__ == "__main__":
    main()
