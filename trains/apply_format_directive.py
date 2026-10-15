#!/usr/bin/env python3
"""格式条件化后处理 (2026-06-21)：给 grounding 训练源的 prompt 加显式格式说明，
并按比例切出 JSON 变体，使「输出格式由 prompt 决定」(增强指令遵循 + 对齐 eval JSON)。

规则(每条按稳定 hash 决定 tag/JSON)：
  - JSON_FRAC(默认0.25) → JSON 形式 + prompt 要 JSON；其余 → tag 形式 + prompt 要 tag。
  - 几何重建为干净格式(剥离 NL 包装)：
      bbox 单:  tag `<bbox>[[x1,y1],[x2,y2]]</bbox>`        JSON `{"bbox":[x1,y1,x2,y2]}`
      bbox 多:  tag 多个 `<bbox>` 逗号连              JSON `[{"bbox":[..],"label":".."},..]`
      point 单: tag `<point>[[x,y]]</point>`               JSON `{"point_2d":[x,y]}`
      point 多: tag `<point>[[x,y],..]</point>`            JSON `{"points":[[x,y],..]}`
  - type=enum(全图枚举, NL 多类) : 保留 NL 答案, 只在 prompt 追加 tag 说明(不做 JSON)。
  - 负样本/无几何            : 答案保留, prompt 追加说明(含"若无则说明没有")。
  坐标 0-1000 不变。objects.type 优先用元字段, 否则按 tag 数推断。

用法:
  python trains/apply_format_directive.py --in X.jsonl --out Y.jsonl --geom bbox|point|auto [--json_frac 0.25]
  python trains/apply_format_directive.py --in X.jsonl --geom auto --sample 8   # 只打印 before/after, 不写
"""
import json, os, re, sys, hashlib, argparse

RE_BBOX = re.compile(r'<bbox>\[\[(-?\d+),(-?\d+)\],\[(-?\d+),(-?\d+)\]\]</bbox>')
RE_POINT_TAG = re.compile(r'<point>\[(.*?)\]</point>')
RE_XY = re.compile(r'\[(-?\d+),(-?\d+)\]')

SUF = {
    ("bbox", 1, "tag"):  " Respond with the box as <bbox>[[x1,y1],[x2,y2]]</bbox> (integers 0-1000).",
    ("bbox", 1, "json"): ' Respond in JSON: {"bbox":[x1,y1,x2,y2]} (integers 0-1000).',
    ("bbox", 2, "tag"):  " Respond with each box as <bbox>[[x1,y1],[x2,y2]]</bbox> (integers 0-1000); if none, say there are none.",
    ("bbox", 2, "json"): ' Respond in JSON: [{"bbox":[x1,y1,x2,y2],"label":"..."}, ...] (integers 0-1000); if none, return [].',
    ("point", 1, "tag"):  " Respond with the point as <point>[[x,y]]</point> (integers 0-1000).",
    ("point", 1, "json"): ' Respond in JSON: {"point_2d":[x,y]} (integers 0-1000).',
    ("point", 2, "tag"):  " Respond with the points as <point>[[x,y],...]</point> (integers 0-1000); if none, say there are none.",
    ("point", 2, "json"): ' Respond in JSON: {"points":[[x,y], ...]} (integers 0-1000); if none, return [].',
}


def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def boxes_of(a):
    return [[int(x) for x in m] for m in RE_BBOX.findall(a)]


def points_of(a):
    out = []
    for blk in RE_POINT_TAG.findall(a):
        out += [[int(x), int(y)] for x, y in RE_XY.findall(blk)]
    return out


def label_of(d, prompt):
    o = (d.get("objects") or {})
    lab = o.get("label")
    if lab and lab not in ("<all>", "<none>"):
        return str(lab)
    m = re.search(r'every (\w[\w ]*?) in', prompt) or re.search(r'all (\w[\w ]*?) in', prompt)
    return m.group(1).strip() if m else None


def strip_image(u):
    return u.replace("<image>", "").strip()


def process(d, geom, json_frac):
    msgs = d["messages"]
    u = next(m for m in msgs if m["role"] == "user")
    a = msgs[-1]
    user_txt = u["content"]
    ans = a["content"]
    ty = (d.get("objects") or {}).get("type")
    boxes = boxes_of(ans); points = points_of(ans)
    kind = "bbox" if boxes else ("point" if points else "none")
    if geom in ("bbox", "point"):
        kind = geom if (boxes or points) else kind
    want_json = h01(user_txt + ans) < json_frac

    # enum: 保留 NL, 仅加 tag 说明
    if ty == "enum":
        if "<bbox>" in ans and "0-1000" not in user_txt:
            u["content"] = user_txt + " Each box as <bbox>[[x1,y1],[x2,y2]]</bbox> (integers 0-1000)."
        return d, "enum_keepNL"

    # 负样本 / 无几何: 答案是自由文本拒答(格式无关), 一律用 tag 说明(避免"return []"与文本答案打架)
    if kind == "none" or (not boxes and not points):
        gk = geom if geom in ("bbox", "point") else "bbox"
        u["content"] = user_txt + SUF[(gk, 2, "tag")]
        return d, "neg"

    fmt = "json" if want_json else "tag"
    multi = 2 if (len(boxes) > 1 or len(points) > 1) else 1
    u["content"] = user_txt + SUF[(kind, multi, fmt)]

    if kind == "bbox":
        if fmt == "tag":
            a["content"] = ", ".join(f"<bbox>[[{b[0]},{b[1]}],[{b[2]},{b[3]}]]</bbox>" for b in boxes)
        else:
            if multi == 1:
                a["content"] = json.dumps({"bbox": boxes[0]}, ensure_ascii=False)
            else:
                lab = label_of(d, user_txt)
                a["content"] = json.dumps([{"bbox": b, **({"label": lab} if lab else {})} for b in boxes], ensure_ascii=False)
    else:  # point
        if fmt == "tag":
            inner = ",".join(f"[{p[0]},{p[1]}]" for p in points)
            a["content"] = f"<point>[{inner}]</point>"
        else:
            a["content"] = json.dumps({"point_2d": points[0]} if multi == 1 else {"points": points}, ensure_ascii=False)
    return d, f"{kind}{multi}_{fmt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out")
    ap.add_argument("--geom", default="auto", choices=["bbox", "point", "auto"])
    ap.add_argument("--json_frac", type=float, default=0.25)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    from collections import Counter
    route = Counter()
    if args.sample:
        shown = Counter()
        with open(args.inp) as f:
            for line in f:
                d = json.loads(line)
                before = (d["messages"][1 if d["messages"][0]["role"] != "user" else 0]["content"][:1],)  # noop
                u0 = next(m["content"] for m in d["messages"] if m["role"] == "user")
                a0 = d["messages"][-1]["content"]
                ty = (d.get("objects") or {}).get("type", "?")
                d2, r = process(json.loads(line), args.geom, args.json_frac)
                cat = r.split("_")[0]
                if shown[cat] < 2:
                    u1 = next(m["content"] for m in d2["messages"] if m["role"] == "user")
                    print(f"\n--- [{r}] type={ty} ---")
                    print("  U0:", u0.replace("\n", " ")[:120])
                    print("  U1:", u1.replace("\n", " ")[:150])
                    print("  A0:", a0.replace("\n", " ")[:120])
                    print("  A1:", d2['messages'][-1]['content'][:120])
                    shown[cat] += 1
                route[r] += 1
                if sum(shown.values()) >= args.sample:
                    break
        print("\nroute(sample):", dict(route))
        return

    n = 0
    with open(args.inp) as fin, open(args.out, "w") as fo:
        for line in fin:
            d, r = process(json.loads(line), args.geom, args.json_frac)
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            route[r] += 1; n += 1
    json.dump({"script": "apply_format_directive.py", "in": args.inp, "out": args.out,
               "geom": args.geom, "json_frac": args.json_frac, "n": n, "route": dict(route)},
              open(args.out.replace(".jsonl", ".fmt.manifest.json"), "w"), ensure_ascii=False, indent=2)
    print(f"-> {args.out}  n={n}  route={dict(route)}")


if __name__ == "__main__":
    main()
