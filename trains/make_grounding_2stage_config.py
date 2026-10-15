#!/usr/bin/env python3
"""生成 grounding 两段式训练配置 (2026-06-21 修订: 并入 Objects365 + 格式条件化)。

生效源(6):
  objects365_s1_fmt        <bbox>/JSON 0-1000  开放域检测主干  (Stage1)   ★新增
  refcoco_grounding_fmt_v4 <bbox>/JSON 0-1000  开放域grounding (Stage1)
  pixmo_points_fmt_v4      <point>/JSON 0-1000+负样本 通用指点 (Stage1, 0.15→0.35上调)
  gsa_allcorrect_native    原生多格式(0-1000)  in-domain verifier (Stage2 主力)
  desktop_bbox_fused       <bbox> 0-1000       in-domain 桌面     (Stage2 主力)
  aobjstate_verifier       原生 pose/选项      in-domain 姿态     (Stage2, 放回补spatial)

格式: refcoco/pixmo/objects365 已做 prompt 格式条件化(tag 75% / JSON 25%, apply_format_directive.py);
      desktop 自带 <bbox> 说明; allcorrect 原生 JSON。tag/JSON 由 prompt 决定, 对齐 eval。

两段:
  Stage1 = 地基(广): objects365 + refcoco + pixmo; in-domain 三源=0。
  Stage2 = in-domain 专精(续训, LR≈1/3): allcorrect + desktop + aobjstate 为主 + 三个 stage1 源薄回放防遗忘。

产物: data/config_grounding_2stage.yaml (+ _stage1/_stage2.yaml)
注: attribute JSON(eval {color,material,shape}) 本轮缺源未补, 推迟新阶段(见 findings.md / docs)。
"""
import os, yaml

ROOT = "${GSA_FORGE_ROOT}"
T = ROOT + "/trains/"

# ---- 可调参数(比例为起点, 自行微调) ----
OBJ365_R = 1.0       # stage1 检测主干(624k 全量)
REFCOCO_R = 2.0      # stage1 过采样
PIXMO_R = 0.35       # stage1 (0.15→0.35: 加objects365后box反压point, 把点撑回)
ALLCORRECT_R = 1.0   # stage2 in-domain 主力
DESKTOP_R = 1.0      # stage2 in-domain 主力
AOBJSTATE_R = 1.0    # stage2 补 pose/朝向(spatial 目标), 量小
# stage2 薄回放(绝对比例, 因三源体量差异大故各自指定; 目标回放≈总量18-20%)
RP_OBJ365 = 0.08
RP_REFCOCO = 0.25
RP_PIXMO = 0.02

SRC = {
 "objects365_s1_fmt": dict(
    path=T+"objects365_s1_fmt.jsonl", est=624716,
    dimensions=["G_grounding"], scene_type="open_domain", license="open (Objects365)",
    note="检测主干; <bbox>/JSON 0-1000 格式条件化; single45%/all32%/enum11%/neg12%; 纯规则v5清洗99.8%准."),
 "refcoco_grounding_fmt_v4": dict(
    path=T+"refcoco_grounding_fmt_v4.jsonl", est=120624,
    dimensions=["G_grounding"], scene_type="indoor_general", license="open (RefCOCO)",
    note="开放域 referring grounding; <bbox>/JSON 0-1000 格式条件化(tag75/json25)."),
 "pixmo_points_fmt_v4": dict(
    path=T+"pixmo_points_fmt_v4.jsonl", est=1191851,
    dimensions=["G_grounding","A_identity_attr"], scene_type="mixed", license="open (Pixmo)",
    note="<point>/JSON 0-1000; 含196,761负样本; 格式条件化; 点的主锚(域外故ratio适中)."),
 "gsa_allcorrect_train_native": dict(
    path=T+"gsa_allcorrect_train_native_v1.jsonl", est=267163,
    dimensions=["G_grounding","S_spatial","A_identity_attr","A_obj_state","A_action_interface","A_action_precondition"],
    scene_type="robot_desktop", license="internal (verifier)",
    note="verifier严格all_correct; 原生多格式(bbox/点0-1000+选项/pose/文本/<FAILED>); in-domain; 原生JSON对齐eval."),
 "desktop_bbox_fused_train": dict(
    path=T+"desktop_bbox_fused_train.jsonl", est=114073,
    dimensions=["G_grounding","A_identity_attr","A_action_interface"],
    scene_type="desktop", license="internal",
    note="<bbox> 0-1000(自带格式说明); 短名+25%长描述融合; in-domain桌面人工标注.(val另存勿训)"),
 "aobjstate_verifier": dict(
    path=T+"aobjstate_verifier_v1.jsonl", est=18729,
    dimensions=["A_obj_state","S_spatial"], scene_type="robot_desktop", license="internal",
    note="pose/朝向/堆叠(瓶杯为主); 原生选项/pose JSON; 补spatial姿态(eval目标). 量小."),
}

ORDER = ["objects365_s1_fmt","refcoco_grounding_fmt_v4","pixmo_points_fmt_v4",
         "gsa_allcorrect_train_native","desktop_bbox_fused_train","aobjstate_verifier"]

STAGE_RATIO = {
 "stage1": {"objects365_s1_fmt":OBJ365_R, "refcoco_grounding_fmt_v4":REFCOCO_R, "pixmo_points_fmt_v4":PIXMO_R,
            "gsa_allcorrect_train_native":0.0, "desktop_bbox_fused_train":0.0, "aobjstate_verifier":0.0},
 "stage2": {"objects365_s1_fmt":RP_OBJ365, "refcoco_grounding_fmt_v4":RP_REFCOCO, "pixmo_points_fmt_v4":RP_PIXMO,
            "gsa_allcorrect_train_native":ALLCORRECT_R, "desktop_bbox_fused_train":DESKTOP_R, "aobjstate_verifier":AOBJSTATE_R},
}
STAGE_DESC = {
 "stage1": "地基(广域): objects365检测主干 + refcoco referring + pixmo指点(含负样本); 建立格式(tag/JSON条件化)+广义定位. in-domain三源=0.",
 "stage2": "in-domain专精(从stage1 ckpt续训, LR≈1/3): allcorrect+desktop+aobjstate为主 + objects365/refcoco/pixmo薄回放防遗忘.",
}
INDOMAIN = {"gsa_allcorrect_train_native","desktop_bbox_fused_train","aobjstate_verifier"}


def src_entry(name, ratio):
    s = SRC[name]
    return {"name": name, "path": s["path"], "format": "msswift_ready",
            "dimensions": s["dimensions"], "est_samples": s["est"], "ratio": ratio,
            "scene_type": s["scene_type"], "license": s["license"],
            "available_aliyun": True, "general": False, "notes": s["note"]}


def main():
    master = {
        "config": "grounding 6源两段式 (2026-06-21: +Objects365 +格式条件化)",
        "params": {"OBJ365_R":OBJ365_R,"REFCOCO_R":REFCOCO_R,"PIXMO_R":PIXMO_R,"ALLCORRECT_R":ALLCORRECT_R,
                   "DESKTOP_R":DESKTOP_R,"AOBJSTATE_R":AOBJSTATE_R,
                   "RP_OBJ365":RP_OBJ365,"RP_REFCOCO":RP_REFCOCO,"RP_PIXMO":RP_PIXMO},
        "deferred": "attribute JSON(eval {color,material,shape}) 缺源未补, 推迟新阶段",
        "run": ["python trains/build_msswift_from_registry.py --registry data/config_grounding_stage1.yaml",
                "python trains/build_msswift_from_registry.py --registry data/config_grounding_stage2.yaml  # resume stage1 ckpt, LR~1/3, 1-2ep"],
    }
    for st in ("stage1","stage2"):
        master[st] = {"description": STAGE_DESC[st],
                      "sources": [src_entry(n, STAGE_RATIO[st][n]) for n in ORDER if STAGE_RATIO[st][n] > 0]}
    yaml.safe_dump(master, open(ROOT+"/data/config_grounding_2stage.yaml","w"), allow_unicode=True, sort_keys=False, width=4096)
    for st in ("stage1","stage2"):
        flat = {"config_stage": st, "description": STAGE_DESC[st],
                "sources": [src_entry(n, STAGE_RATIO[st][n]) for n in ORDER if STAGE_RATIO[st][n] > 0]}
        yaml.safe_dump(flat, open(ROOT+f"/data/config_grounding_{st}.yaml","w"), allow_unicode=True, sort_keys=False, width=4096)

    print("生成: data/config_grounding_2stage.yaml (+ _stage1/_stage2.yaml)")
    for st in ("stage1","stage2"):
        eff = [(n, STAGE_RATIO[st][n]*SRC[n]["est"]) for n in ORDER if STAGE_RATIO[st][n] > 0]
        tot = sum(e for _, e in eff)
        idom = sum(e for n, e in eff if n in INDOMAIN)
        print(f"\n{st}: 有效≈{tot:,.0f}" + (f"  in-domain≈{idom/tot*100:.0f}%  回放≈{(tot-idom)/tot*100:.0f}%" if st=='stage2' else ""))
        for n, e in sorted(eff, key=lambda x:-x[1]):
            print(f"   {e:11,.0f}  {n}  (ratio={STAGE_RATIO[st][n]})")


if __name__ == "__main__":
    main()
