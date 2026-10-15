#!/usr/bin/env python3
"""A_obj_state 小批量提取(Phase A,提前自 Phase B)。

从 verifier train.jsonl(同库同域、verifier-correct)捞 pose/state(姿态/朝向/堆叠/开合)样本,
保留原生答案(选项/pose JSON,对齐 eval),转 ms-swift 新 jsonl。补 spatial/obj_state 根因。

筛选信号(任一命中,且答案非 bbox、非 <FAILED>):
  - prompt 关键词:朝向/正立/倒扣/横躺/开口朝/夹爪/standing|lying|upright|tipped 等(排除 bbox boilerplate)
  - scene_categories: CAT.2.1 朝向姿态 / CAT.3.2 状态分类
  - qc.task_family == state_classification

不覆盖原文件:
  输入  ${DATA_ROOT}/datasets/correct-balanced-train-test-v1/train.jsonl  (只读)
  输出  trains/aobjstate_verifier_v1.jsonl
  manifest trains/aobjstate_verifier_v1.manifest.json

用法: python3 trains/extract_aobjstate_from_verifier.py [--limit N] [--cap N]
"""
import json, os, re, sys

ROOT = "${GSA_FORGE_ROOT}"
SRC = "${DATA_ROOT}/datasets/correct-balanced-train-test-v1/train.jsonl"
OUT = os.path.join(ROOT, "trains/aobjstate_verifier_v1.jsonl")
MANIFEST = os.path.join(ROOT, "trains/aobjstate_verifier_v1.manifest.json")
OLD = "${DATA_ROOT}/qwen-proxy-log-mirror"
NEW = "${DATA_ROOT}/datasets/qwen-proxy-log-mirror-fast"

RE_POSE = re.compile(r'朝向|正立|倒扣|横躺|开口朝|夹爪|是否(开着|关着|为空|装满|倒)|开着还是关着|正放还是倒|'
                     r'站立还是|standing or lying|upright|tipped over|lying down|orientation of|pose of', re.I)
def is_bbox_ans(a): return '"bbox"' in a or '<box>' in a or 'bbox_2d' in a
def bad_ans(a): return (not a) or a.strip() in ('<FAILED>', 'FAILED', '')

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
CAP = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else None


def remap(p): return p.replace(OLD, NEW)


def main():
    n = 0; written = 0; skip_img = 0
    by_signal = {"keyword": 0, "scene": 0, "qc": 0}
    ans_kind = {"choice": 0, "json": 0, "text": 0}
    ex = []
    with open(SRC) as f, open(OUT, "w") as fo:
        for line in f:
            n += 1
            if LIMIT and n > LIMIT: break
            if CAP and written >= CAP: break
            d = json.loads(line)
            prompt = str(d.get("prompt", "")); ans = str(d.get("model_answer", ""))
            if bad_ans(ans) or is_bbox_ans(ans):
                continue
            tags = d.get("tags") or {}
            qc = (tags.get("qc") or {}) if isinstance(tags, dict) else {}
            scenes = tags.get("scene_categories") or []
            kw = bool(RE_POSE.search(prompt))
            sc = any("CAT.2.1" in str(s) or "CAT.3.2" in str(s) for s in scenes)
            qcs = isinstance(qc, dict) and qc.get("task_family") == "state_classification"
            if not (kw or sc or qcs):
                continue
            imgs = [remap(p) for p in (d.get("images") or [])]
            if not imgs or not os.path.isfile(imgs[0]):
                skip_img += 1; continue
            if kw: by_signal["keyword"] += 1
            if sc: by_signal["scene"] += 1
            if qcs: by_signal["qc"] += 1
            # answer kind
            if re.fullmatch(r'\s*\(?[A-D]\)?\.?\s*', ans): ans_kind["choice"] += 1
            elif ans.strip().startswith("{") or "```json" in ans: ans_kind["json"] += 1
            else: ans_kind["text"] += 1
            u = prompt
            if "<image>" not in u:
                u = "<image>\n" + u
            rec = {
                "id": d.get("id", f"aobj_{n}"),
                "messages": [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": ans},
                ],
                "images": imgs,
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if len(ex) < 6:
                ex.append({"q": prompt[:70].replace("\n", " "), "a": ans[:50].replace("\n", " ")})

    manifest = {
        "script": "trains/extract_aobjstate_from_verifier.py",
        "input": SRC, "output": OUT,
        "schema": "native (choice / pose JSON), 对齐 eval", "domain": "robot-desktop (verifier-correct)",
        "phase": "A (advanced from B)", "version": "v1",
        "total_read": n, "written": written, "skip_no_image": skip_img,
        "by_signal_hits": by_signal, "answer_kind": ans_kind,
        "samples": ex,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print(f"read={n} written={written} skip_no_image={skip_img}")
    print(f"  signal hits: {by_signal} | answer kind: {ans_kind}")
    print(f"-> {OUT}\n-> {MANIFEST}")
    for e in ex:
        print(f"   Q={e['q']!r} -> A={e['a']!r}")


if __name__ == "__main__":
    main()
