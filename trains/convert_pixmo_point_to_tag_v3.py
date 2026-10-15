#!/usr/bin/env python3
"""pixmo point → tag 风格 <point> (Phase A, v3)。

v3 与 v2 的唯一区别 (按用户 2026-06-18 指示):
  - **保留负样本**: 无 point 的 "There are none." 不再丢弃, 原样作为"答没有"的训练样本。
    (没有物体时模型必须会答"没有", 否则会乱点。)
  - 仅极少数(~140)坐标含负值的畸形 <point> 答案被 clamp/跳过。
其余同 v2: 无 system 格式说明、user 不加格式提示、正样本转 <point>[[x,y],...]</point> (0-1000)。

不覆盖任何原文件:
  输入  trains/pixmo_filtered.jsonl                     (只读)
  输出  trains/pixmo_points_tag_norm1000_v3.jsonl       (新文件, 不动 v1/v2)
  manifest trains/pixmo_points_tag_norm1000_v3.manifest.json

用法: python3 trains/convert_pixmo_point_to_tag_v3.py [--limit N]
"""
import json, os, re, sys

ROOT = "${GSA_FORGE_ROOT}"
SRC = os.path.join(ROOT, "trains/pixmo_filtered.jsonl")
OUT = os.path.join(ROOT, "trains/pixmo_points_tag_norm1000_v3.jsonl")
MANIFEST = os.path.join(ROOT, "trains/pixmo_points_tag_norm1000_v3.manifest.json")

RE_PT = re.compile(r'"point_2d"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]')
RE_JSON_HINT = re.compile(r'\.?\s*Output in JSON format\.?', re.I)
RE_NONE = re.compile(r'there\s+are\s+none|there\s+is\s+none|没有', re.I)

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
NEG_ANS = "There are none."


def clamp(v):
    return max(0, min(1000, int(round(v))))


def get_user(msgs):
    for m in msgs:
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def main():
    n = 0; pos = 0; neg = 0; skip_bad = 0; over1000 = 0
    ex = []
    with open(SRC) as f, open(OUT, "w") as fo:
        for line in f:
            n += 1
            if LIMIT and n > LIMIT: break
            d = json.loads(line)
            msgs = d.get("messages", [])
            a = "".join(str(m.get("content", "")) for m in msgs if m.get("role") == "assistant")
            pts = [(float(x), float(y)) for x, y in RE_PT.findall(a)]
            u = RE_JSON_HINT.sub("", get_user(msgs)).strip()
            if "<image>" not in u:
                u = "<image>\n" + u
            if pts:
                if any(x > 1000 or y > 1000 for x, y in pts):
                    over1000 += 1
                pts_str = ",".join(f"[{clamp(x)},{clamp(y)}]" for x, y in pts)
                ans = f"<point>[{pts_str}]</point>"
                pos += 1
            elif RE_NONE.search(a):
                ans = NEG_ANS          # 负样本: 统一成自然的"答没有"
                neg += 1
            else:
                skip_bad += 1; continue  # 畸形(如 <point>[[x,-1]]) 极少, 跳过
            rec = {
                "id": d.get("id", f"pixmo_{n}"),
                "messages": [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": ans},
                ],
                "images": d.get("images", []),
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if len(ex) < 6 and (neg <= 2 or pos <= 4):
                ex.append({"pos": bool(pts), "u": u[:70].replace("\n", " "), "a": ans[:60]})

    written = pos + neg
    manifest = {
        "script": "trains/convert_pixmo_point_to_tag_v3.py",
        "input": SRC, "output": OUT,
        "schema": "tag <point>[[x,y],...]</point> + 负样本 'There are none.'", "coord": "0-1000 (clamp)",
        "phase": "A", "version": "v3",
        "diff_from_v2": "保留 'There are none.' 负样本(教模型答没有)",
        "written": written, "positive": pos, "negative": neg,
        "skip_malformed": skip_bad, "over1000_clamped": over1000,
        "samples": ex,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print(f"read={n} written={written} (pos={pos} neg={neg}) skip_malformed={skip_bad} over1000_clamped={over1000}")
    print(f"-> {OUT}")
    for e in ex:
        print(f"   {'POS' if e['pos'] else 'NEG'} u={e['u']!r} -> {e['a']!r}")


if __name__ == "__main__":
    main()
