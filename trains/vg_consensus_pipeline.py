#!/usr/bin/env python3
"""Visual Genome grounding 数据清洗 pipeline —— 三源一致裁决(consensus-of-3)。

经 500 抽样实测验证的最优方案:
  VG 人工标注框  ∩  Qwen3.6-27B(EAS)  ∩  Qwen3.5-397B(EAS)
  三方各自对 phrase 定位(均 norm1000), 两两 IoU≥T 才保留(留 VG 框)。
  关键: VG(人工)是与两个 Qwen 不同的源, 打破同源相关偏差。
  反面(脚本不采用, 已实测否决): 仅两个 Qwen 互相一致推翻 VG(ok仅37%); 单模型换框(净负)。

阶段:
  S0 几何预筛  -> 丢 area<min% / >max% / 退化框
  S1 grounding -> 27B + 397B 各定位一次 (EAS, 并发, 可选 self-host 27B 第二路)
  S2 consensus -> 三源两两 IoU≥T 才留
  S3 属性检查  -> (可选) 397B 判框内是否符合 phrase(含颜色/数量), 丢 "no"
  输出: 清洗后的 MS-Swift jsonl (沿用 <bbox>[[x1,y1],[x2,y2]]</bbox> norm1000) + manifest

凭证从环境变量读取(不落盘):
  EAS_27B_URL   EAS_27B_TOKEN       # 必需, 形如 http://<host>/api/predict/qwen3_6_27b
  EAS_397B_URL  EAS_397B_TOKEN      # 必需, 形如 http://<host>/api/predict/qwen3_5_397b_distill
  SELF_27B_URL                      # 可选, 27B 第二路(vLLM /v1), 分担吞吐
  EAS_27B_MODEL (默认 Qwen3.6-27B)  EAS_397B_MODEL (默认 Qwen3.5-397B)

用法:
  python3 trains/vg_consensus_pipeline.py \
      --input  trains/.../vg_grounding_swift.jsonl \
      --output out/vg_consensus.jsonl \
      --iou-threshold 0.85 --attr-check --workers 12
  # 断点续跑: grounding 结果缓存在 <output>.cache.jsonl, 重跑自动复用。
"""
import argparse, base64, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

BOX_PAT = re.compile(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]')
PHRASE_PAT = re.compile(r'Locate\s+"(.*?)"\s+in the image', re.S)
SWIFTBOX_PAT = re.compile(r'<bbox>\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]</bbox>')


# ---------------- 基础 ----------------
def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0


def b64img(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _post(url, tok, body, timeout=120):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": tok, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def eas_ground(url, tok, model, path, phrase, timeout=90):
    """OpenAI 兼容视觉定位, 返回 norm1000 box 或 None。"""
    body = {"model": model, "temperature": 0, "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64img(path)}"}},
            {"type": "text", "text": f'Locate "{phrase}" in the image. Output the bounding box.'}]}]}
    out = _post(url, tok, body, timeout)
    m = out["choices"][0]["message"]
    txt = m.get("content") or m.get("reasoning_content") or ""
    mm = BOX_PAT.search(txt)
    if not mm:
        return None
    box = [max(0, min(1000, int(round(float(x))))) for x in mm.groups()]
    return box if (box[2] > box[0] and box[3] > box[1]) else None


def eas_attr_match(url, tok, model, path, phrase, box, timeout=90):
    """属性一致性: 框内是否符合 phrase(含颜色/数量)。返回 yes/no/uncertain。"""
    body = {"model": model, "temperature": 0, "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64img(path)}"}},
            {"type": "text", "text": (
                f'The region of interest is the normalized box {box} (coords 0-1000 [x1,y1,x2,y2]).'
                f' Does the content inside that box fully match "{phrase}", including any color/attribute/count?'
                f' Answer strictly JSON: {{"match":"yes|no|uncertain"}}')}]}]}
    out = _post(url, tok, body, timeout)
    m = out["choices"][0]["message"]
    txt = m.get("content") or m.get("reasoning_content") or ""
    mm = re.search(r'"match"\s*:\s*"(yes|no|uncertain)"', txt)
    return mm.group(1) if mm else "uncertain"


# ---------------- 解析输入 ----------------
def parse_record(d):
    """从 MS-Swift 记录抽 (phrase, vg_box_norm1000, image_path)。"""
    msgs = d.get("messages", [])
    if len(msgs) < 2:
        return None
    pm = PHRASE_PAT.search(msgs[0].get("content", ""))
    bm = SWIFTBOX_PAT.search(msgs[1].get("content", ""))
    imgs = d.get("images") or []
    if not pm or not bm or not imgs:
        return None
    box = [int(bm.group(i)) for i in (1, 2, 3, 4)]
    return {"id": d.get("id"), "phrase": pm.group(1), "vg_box": box,
            "image": imgs[0], "raw": d}


def box_area_frac(b):
    return (b[2] - b[0]) * (b[3] - b[1]) / 1e6  # 占整图比例 (1000x1000)


# ---------------- 主流程 ----------------
def env(name, required=True, default=None):
    v = os.environ.get(name, default)
    if required and not v:
        sys.exit(f"[pipeline] 缺少环境变量 {name}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.85)
    ap.add_argument("--min-area", type=float, default=0.01, help="S0: 丢面积<此比例的框")
    ap.add_argument("--max-area", type=float, default=0.80, help="S0: 丢面积>此比例的框")
    ap.add_argument("--no-geo-filter", action="store_true")
    ap.add_argument("--attr-check", action="store_true", help="S3: 397B 属性一致性过滤")
    ap.add_argument("--two-source", action="store_true",
                    help="2源模式: 只 VG∩27B (跳过397B), 27B双路, ~2.3x 更快, 质量≈3源")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    eas27_url, eas27_tok = env("EAS_27B_URL"), env("EAS_27B_TOKEN")
    eas397_url, eas397_tok = env("EAS_397B_URL"), env("EAS_397B_TOKEN")
    self27_url = env("SELF_27B_URL", required=False)
    m27 = env("EAS_27B_MODEL", required=False, default="Qwen3.6-27B")
    m397 = env("EAS_397B_MODEL", required=False, default="Qwen3.5-397B")
    self27_tok = os.environ.get("SELF_27B_TOKEN", "")
    # 注意: 自部署 vLLM 的 model id 通常带前缀(Qwen/Qwen3.6-27B), 与 EAS(Qwen3.6-27B)不同
    self27_model = env("SELF_27B_MODEL", required=False, default="Qwen/Qwen3.6-27B")

    # 读取 + S0
    recs = []
    with open(args.input) as f:
        for n, line in enumerate(f):
            if args.limit and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            r = parse_record(json.loads(line))
            if r:
                recs.append(r)
    n_in = len(recs)
    geo = []
    s0_tiny = s0_full = s0_degen = 0
    for r in recs:
        a = box_area_frac(r["vg_box"])
        if r["vg_box"][2] <= r["vg_box"][0] or r["vg_box"][3] <= r["vg_box"][1]:
            s0_degen += 1; continue
        if not args.no_geo_filter:
            if a < args.min_area:
                s0_tiny += 1; continue
            if a > args.max_area:
                s0_full += 1; continue
        geo.append(r)
    print(f"[S0] 输入 {n_in} -> 几何预筛后 {len(geo)} (丢 tiny={s0_tiny} full={s0_full} degen={s0_degen})", flush=True)

    # S1 grounding (带缓存)
    cache_path = args.output + ".cache.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            c = json.loads(line)
            cache[c["id"]] = c
        print(f"[S1] 复用缓存 {len(cache)} 条", flush=True)

    def ground_one(idx, r):
        rid = r["id"]
        if rid in cache and cache[rid].get("b27") is not None and (args.two_source or cache[rid].get("b397") is not None):
            return cache[rid]
        # 27B 双路负载均衡 (各路 model id 不同)
        if self27_url and idx % 2 == 1:
            u27, t27, mod27 = self27_url, self27_tok, self27_model
        else:
            u27, t27, mod27 = eas27_url, eas27_tok, m27
        out = {"id": rid, "b27": None, "b397": None}
        for attempt in range(3):
            try:
                out["b27"] = eas_ground(u27, t27, mod27, r["image"], r["phrase"])
                break
            except Exception:
                time.sleep(2)
        if not args.two_source:
            for attempt in range(3):
                try:
                    out["b397"] = eas_ground(eas397_url, eas397_tok, m397, r["image"], r["phrase"])
                    break
                except Exception:
                    time.sleep(2)
        return out

    t0 = time.time()
    grounded = {}
    cf = open(cache_path, "a")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ground_one, i, r): r["id"] for i, r in enumerate(geo)}
        done = 0
        for fut in as_completed(futs):
            g = fut.result(); grounded[g["id"]] = g
            if g["id"] not in cache:
                cf.write(json.dumps(g, ensure_ascii=False) + "\n"); cf.flush()
            done += 1
            if done % 200 == 0:
                print(f"[S1] grounding {done}/{len(geo)} ({time.time()-t0:.0f}s)", flush=True)
    cf.close()

    # S2 consensus3
    T = args.iou_threshold
    survivors = []
    s2_nobox = s2_disagree = 0
    for r in geo:
        g = grounded.get(r["id"], {})
        b27, b397, vg = g.get("b27"), g.get("b397"), r["vg_box"]
        if args.two_source:
            if not b27:
                s2_nobox += 1; continue
            if iou(vg, b27) >= T:
                survivors.append(r)
            else:
                s2_disagree += 1
        else:
            if not b27 or not b397:
                s2_nobox += 1; continue
            if iou(b27, b397) >= T and iou(vg, b27) >= T and iou(vg, b397) >= T:
                survivors.append(r)
            else:
                s2_disagree += 1
    mode = "consensus2 (VG∩27B)" if args.two_source else "consensus3 (VG∩27B∩397B)"
    print(f"[S2] {mode}@{T}: {len(survivors)} 留 (丢 无框={s2_nobox} 不一致={s2_disagree})", flush=True)

    # S3 属性检查
    s3_dropped = 0
    if args.attr_check and survivors:
        keep = []
        def chk(r):
            for _ in range(2):
                try:
                    return r, eas_attr_match(eas397_url, eas397_tok, m397, r["image"], r["phrase"], r["vg_box"])
                except Exception:
                    time.sleep(2)
            return r, "uncertain"
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r, m in [f.result() for f in as_completed([ex.submit(chk, r) for r in survivors])]:
                if m == "no":
                    s3_dropped += 1
                else:
                    keep.append(r)
        survivors = keep
        print(f"[S3] 属性检查后 {len(survivors)} 留 (丢 no={s3_dropped})", flush=True)

    # 输出 (沿用原始 MS-Swift 记录, 不改框)
    survivors.sort(key=lambda r: str(r["id"]))
    with open(args.output, "w") as fo:
        for r in survivors:
            fo.write(json.dumps(r["raw"], ensure_ascii=False) + "\n")
    manifest = {
        "input": args.input, "output": args.output,
        "method": "consensus-of-2 (VG ∩ Qwen3.6-27B)" if args.two_source else "consensus-of-3 (VG ∩ Qwen3.6-27B ∩ Qwen3.5-397B)",
        "iou_threshold": T, "geo_filter": not args.no_geo_filter,
        "min_area": args.min_area, "max_area": args.max_area, "attr_check": args.attr_check,
        "n_input": n_in, "s0_kept": len(geo),
        "s0_drop_tiny": s0_tiny, "s0_drop_full": s0_full, "s0_drop_degen": s0_degen,
        "s2_no_box": s2_nobox, "s2_disagree": s2_disagree,
        "s3_attr_dropped": s3_dropped,
        "n_output": len(survivors),
        "keep_rate": round(len(survivors) / n_in * 100, 2) if n_in else 0,
        "models": {"loc1": m27, "loc2": m397},
    }
    json.dump(manifest, open(args.output + ".manifest.json", "w"), indent=2, ensure_ascii=False)
    print(f"[OUT] {len(survivors)}/{n_in} ({manifest['keep_rate']}%) -> {args.output}", flush=True)
    print(f"      manifest -> {args.output}.manifest.json", flush=True)


if __name__ == "__main__":
    main()
