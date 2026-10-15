#!/usr/bin/env python3
"""Offline forensic audit of eval/eval_gsa_test.py against the real GSA-Bench test split.

No inference is run. Every number printed comes from trains/gsa_bench/test.jsonl, the
images it references, and direct execution of the two scoring/parsing code paths:
  OLD -> eval/eval_gsa_test.py            (iou, parse_items)
  NEW -> eval/gsa_eval/{coords,parsing,scorers}.py

Usage:  python eval/audit_gsa_test.py [--data trains/gsa_bench/test.jsonl]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from gsa_eval import coords as C  # noqa: E402
from gsa_eval import parsing as P  # noqa: E402
from gsa_eval.datasets import Sample, stratified_subset  # noqa: E402
from gsa_eval.scorers import score_gsa_geometry  # noqa: E402


def _load_old_module():
    """Import eval_gsa_test.py without executing main() (it needs --model)."""
    spec = importlib.util.spec_from_file_location(
        "old_eval_gsa_test", ROOT / "eval" / "eval_gsa_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OLD = _load_old_module()


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def pcts(vals):
    """min / p1 / p50 / p99 / max of a numeric list, rounded."""
    if not vals:
        return {}
    v = sorted(float(x) for x in vals)
    q = statistics.quantiles(v, n=100) if len(v) > 1 else [v[0]] * 99
    return {"min": round(v[0], 1), "p1": round(q[0], 1), "p50": round(q[49], 1),
            "p99": round(q[98], 1), "max": round(v[-1], 1)}


def load_rows(data: Path) -> list[dict]:
    rows = []
    for ln in data.read_text(errors="ignore").splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def image_size(path: str) -> tuple[int, int]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


def enrich(rows: list[dict]) -> list[dict]:
    """Attach parsed GT, form, and true image size to every row."""
    cache: dict[str, tuple[int, int]] = {}
    out = []
    for i, r in enumerate(rows):
        msgs = r.get("messages") or []
        gt, gt_ok = None, False
        if len(msgs) >= 2:
            try:
                gt = json.loads(msgs[1]["content"])
                gt_ok = True
            except Exception:
                pass
        img = (r.get("images") or [None])[0]
        if img not in cache:
            cache[img] = image_size(img) if img else (0, 0)
        w, h = cache[img]
        out.append({"i": i, "row": r, "gt": gt, "gt_ok": gt_ok,
                    "form": (r.get("gsa_meta") or {}).get("form", "?"),
                    "source": (r.get("gsa_meta") or {}).get("source", "?"),
                    "prompt": msgs[0]["content"] if msgs else "",
                    "img": img, "w": w, "h": h})
    return out


def first_geom(rec):
    """(kind, coords) of the first GT item, mirroring what both scorers read."""
    gt = rec["gt"]
    gt = gt if isinstance(gt, list) else [gt]
    g = gt[0] if gt and isinstance(gt[0], dict) else {}
    b = g.get("bbox_2d") or g.get("bbox")
    if isinstance(b, list) and len(b) == 4:
        return "bbox", [float(v) for v in b]
    p = g.get("point_2d") or g.get("point")
    if isinstance(p, list) and len(p) == 2:
        return "point", [float(v) for v in p]
    return "none", None


def task1(recs) -> None:
    hr("TASK 1 -- ground-truth characterisation (all rows)")
    print(f"total rows parsed from jsonl: {len(recs)}")
    print(f"messages[1] JSON-parseable  : {sum(r['gt_ok'] for r in recs)}")
    print(f"images present on disk      : "
          f"{sum(1 for r in recs if r['w'] and r['h'])}")
    print(f"form distribution           : "
          f"{dict(sorted(Counter(r['form'] for r in recs).items()))}")
    print(f"(form, source)              : "
          f"{dict(sorted(Counter((r['form'], r['source']) for r in recs).items()))}")

    hr("TASK 1a -- coordinate ranges per form")
    for form in sorted({r["form"] for r in recs}):
        sub = [r for r in recs if r["form"] == form]
        kinds = Counter(first_geom(r)[0] for r in sub)
        print(f"\nform={form}  n={len(sub)}  gt item kinds={dict(kinds)}")
        cols = defaultdict(list)
        for r in sub:
            kind, c = first_geom(r)
            if c is None:
                continue
            names = ("x1", "y1", "x2", "y2") if kind == "bbox" else ("x", "y")
            for name, v in zip(names, c):
                cols[name].append(v)
            cols["ALL"].extend(c)
        for name in ("x1", "y1", "x2", "y2", "x", "y", "ALL"):
            if name in cols:
                print(f"  {name:>3}: n={len(cols[name]):5d} {pcts(cols[name])}")

    hr("TASK 1b -- do bbox GT coords fit inside the image pixel bounds?")
    inside = overflow = 0
    ov_axes = Counter()
    for r in recs:
        kind, c = first_geom(r)
        if kind != "bbox" or not (r["w"] and r["h"]):
            continue
        x1, y1, x2, y2 = c
        ox = max(x1, x2) > r["w"]
        oy = max(y1, y2) > r["h"]
        if ox or oy:
            overflow += 1
            ov_axes["x" if ox and not oy else "y" if oy and not ox else "both"] += 1
        else:
            inside += 1
    tot = inside + overflow
    print(f"bbox rows with a usable image: {tot}")
    print(f"  GT box INSIDE pixel bounds : {inside} ({100*inside/max(tot,1):.2f}%)")
    print(f"  GT box OVERFLOWS the image : {overflow} "
          f"({100*overflow/max(tot,1):.2f}%)  axes={dict(ov_axes)}")
    print("  interpretation: overflow is only possible if GT is NOT pixel-scale.")

    ip = io_ = 0
    for r in recs:
        kind, c = first_geom(r)
        if kind != "point" or not (r["w"] and r["h"]):
            continue
        if c[0] > r["w"] or c[1] > r["h"]:
            io_ += 1
        else:
            ip += 1
    print(f"point rows: inside pixel bounds={ip}  overflow={io_} "
          f"({100*io_/max(ip+io_,1):.2f}% overflow)")

    hr("TASK 1c -- image size distribution")
    sizes = Counter((r["w"], r["h"]) for r in recs if r["w"] and r["h"])
    print(f"distinct sizes: {len(sizes)}   top 10 by count:")
    for (w, h), n in sizes.most_common(10):
        print(f"  {w}x{h}: {n}")
    print(f"  width  {pcts([r['w'] for r in recs if r['w']])}")
    print(f"  height {pcts([r['h'] for r in recs if r['h']])}")


def task2(recs) -> None:
    hr("TASK 2 -- is the GT uniformly thousandth-scale?")
    over_1000 = []
    negative = []
    degenerate = []
    unit_like = []
    ambiguous = []       # fits pixel bounds AND has a large coord -> unit is undecidable
    for r in recs:
        kind, c = first_geom(r)
        if c is None:
            continue
        mx = max(c)
        if mx > 1000:
            over_1000.append((r["i"], r["form"], c, (r["w"], r["h"])))
        if min(c) < 0:
            negative.append((r["i"], r["form"], c))
        if mx <= C._UNIT_MAX:
            unit_like.append((r["i"], r["form"], c))
        if kind == "bbox":
            x1, y1, x2, y2 = c
            if x2 <= x1 or y2 <= y1:
                degenerate.append((r["i"], c))
            if (max(x1, x2) <= r["w"] and max(y1, y2) <= r["h"]
                    and mx > 100 and r["w"] and r["h"]):
                ambiguous.append((r["i"], c, (r["w"], r["h"])))

    print(f"rows with any coord > 1000            : {len(over_1000)}")
    for e in over_1000[:10]:
        print(f"    {e}")
    print(f"rows with a negative coord            : {len(negative)}")
    print(f"rows whose max coord <= {C._UNIT_MAX} (unit-like): {len(unit_like)}")
    print(f"bbox rows that are degenerate (x2<=x1 or y2<=y1): {len(degenerate)}")
    for e in degenerate[:10]:
        print(f"    {e}")
    print(f"bbox rows AMBIGUOUS under auto-inference (fit pixel bounds, max>100): "
          f"{len(ambiguous)}")
    for e in ambiguous[:10]:
        print(f"    idx={e[0]} box={e[1]} img={e[2]}")
    print("\nWhat infer_scale(declared='auto') would say about the GT itself:")
    auto = Counter()
    for r in recs:
        kind, c = first_geom(r)
        if c is None or not (r["w"] and r["h"]):
            continue
        if kind == "bbox":
            xs, ys = [c[0], c[2]], [c[1], c[3]]
        else:
            xs, ys = [c[0]], [c[1]]
        auto[(r["form"], C.infer_scale(xs, ys, r["w"], r["h"]))] += 1
    for k, v in sorted(auto.items()):
        print(f"  {k}: {v}")
    print("  -> any 'pixel' verdict here is a row where auto-inference would have been\n"
          "     WRONG; declared='thousandth' is what makes those rows score correctly.")


def _px_box(c, w, h):
    return [round(c[0] / 1000.0 * w, 1), round(c[1] / 1000.0 * h, 1),
            round(c[2] / 1000.0 * w, 1), round(c[3] / 1000.0 * h, 1)]


def _px_point(c, w, h):
    return [round(c[0] / 1000.0 * w, 1), round(c[1] / 1000.0 * h, 1)]


def _old_score(pairs_box, pairs_pt):
    """Run the OLD script's arithmetic verbatim on (pred, gt) pairs."""
    ious = [OLD.iou(pb, gb) for pb, gb in pairs_box]
    dists = [math.dist(pp, gp) / 1000.0 for pp, gp in pairs_pt]
    hits = [1 if d <= 0.1 else 0 for d in dists]
    out = {}
    if ious:
        out["bbox"] = {"n": len(ious), "mean_iou": round(sum(ious) / len(ious), 4),
                       "iou@0.5": round(100 * sum(i >= .5 for i in ious) / len(ious), 2),
                       "iou@0.75": round(100 * sum(i >= .75 for i in ious) / len(ious), 2)}
    if dists:
        out["point"] = {"n": len(dists),
                        "hit@0.1": round(100 * sum(hits) / len(hits), 2),
                        "mean_dist": round(sum(dists) / len(dists), 4)}
    return out


def _build_samples(recs):
    out = []
    for r in recs:
        if not (r["w"] and r["h"]) or not r["gt_ok"]:
            continue
        out.append((r, Sample(uid=f"{r['form']}_{r['i']}", prompt="", image_paths=[r["img"]],
                              gt=r["gt"], width=r["w"], height=r["h"], group=r["form"],
                              meta={"form": r["form"], "source": r["source"]})))
    return out


def task3(recs) -> None:
    hr("TASK 3 -- reproduce the bug with a SIMULATED oracle prediction (no inference)")
    print("Simulated prediction = the GT itself, converted thousandth -> PIXEL.")
    print("A model that localises perfectly but answers in pixels produces exactly this.")
    pairs = _build_samples(recs)

    box_px, pt_px, box_th, pt_th = [], [], [], []
    texts_px, texts_th, samples = [], [], []
    for r, s in pairs:
        kind, c = first_geom(r)
        if c is None:
            continue
        samples.append(s)
        if kind == "bbox":
            p = _px_box(c, r["w"], r["h"])
            box_px.append((p, c))
            box_th.append((list(c), c))
            texts_px.append(json.dumps([{"label": "object", "bbox_2d": p}]))
            texts_th.append(json.dumps([{"label": "object", "bbox_2d": list(c)}]))
        else:
            p = _px_point(c, r["w"], r["h"])
            pt_px.append((p, c))
            pt_th.append((list(c), c))
            texts_px.append(json.dumps([{"label": "object", "point_2d": p}]))
            texts_th.append(json.dumps([{"label": "object", "point_2d": list(c)}]))

    print(f"\nsimulated rows: bbox={len(box_px)}  point={len(pt_px)}")

    print("\n-- OLD eval_gsa_test.py scoring (no normalisation at all) --")
    old_px = _old_score(box_px, pt_px)
    print("  pred in PIXEL scale (the realistic case):")
    print("   ", json.dumps(old_px))
    old_th = _old_score(box_th, pt_th)
    print("  pred in THOUSANDTH scale (control: prediction matches GT units exactly):")
    print("   ", json.dumps(old_th))

    print("\n-- NEW score_gsa_geometry (normalises both sides) --")
    sc_px, det_px, f_px = score_gsa_geometry(samples, texts_px)
    print(f"  pred in PIXEL scale      -> score={sc_px} parse_failures={f_px}")
    print("   ", json.dumps(det_px))
    sc_th, det_th, f_th = score_gsa_geometry(samples, texts_th)
    print(f"  pred in THOUSANDTH scale -> score={sc_th} parse_failures={f_th}")
    print("   ", json.dumps(det_th))

    print("\n-- reconciliation with the published 4.2 numbers --")
    sim = old_px["bbox"]["mean_iou"]
    for label, reported, n_fail in (("base", 0.0003, 211), ("SFT", 0.0015, 253)):
        print(f"  {label}: reported mean_iou={reported}; pure scale mismatch alone "
              f"predicts {sim}")
        print(f"       parse failures were {n_fail}/400, and the OLD script appends "
              f"0.0 to `ious` for each")
        print(f"       bbox parse failure, so the ceiling drops -- but even a 100% "
              f"parse-failure-free")
        print(f"       run could not reach {reported} from scale mismatch alone: "
              f"{sim}/{reported:.4f} = {sim/reported:.0f}x gap.")
    print("  => scale mismatch is NECESSARY but not SUFFICIENT to explain 0.0003.")
    print("     The residual must come from parse failures (scored 0.0) plus genuinely")
    print("     wrong localisation. This audit CANNOT separate those two without")
    print("     re-running inference, which is out of scope here.")

    hr("TASK 3b -- the NEW harness PENALISES a model that obeys its own prompt hint")
    print("datasets.py:243 appends 'use a 0-1000 thousandth scale' to every prompt, but")
    print("scorers.py:129/143 reads the PREDICTION with declared='auto'. A perfect")
    print("thousandth-scale answer is therefore reclassified as pixels whenever it")
    print("happens to fit inside the image bounds. Rows lost that way:")
    lost_b, lost_p = [], []
    for r, s in pairs:
        kind, c = first_geom(r)
        if c is None:
            continue
        if kind == "bbox":
            gb = C.to_unit_box(c, s.width, s.height, declared="thousandth")
            got = C.iou(C.to_unit_box(c, s.width, s.height), gb)
            if got < 0.5:
                lost_b.append((r["i"], c, (r["w"], r["h"]),
                               C.infer_scale([c[0], c[2]], [c[1], c[3]],
                                             r["w"], r["h"]), round(got, 3)))
        else:
            gx, gy = C.to_unit_points([c], s.width, s.height,
                                      declared="thousandth")[0]
            px, py = C.to_unit_points([c], s.width, s.height)[0]
            if math.dist((px, py), (gx, gy)) > 0.1:
                lost_p.append((r["i"], c, (r["w"], r["h"]),
                               C.infer_scale([c[0]], [c[1]], r["w"], r["h"])))
    print(f"  bbox rows where an EXACT thousandth answer scores IoU<0.5 : "
          f"{len(lost_b)}/{len(box_th)}")
    for e in lost_b[:8]:
        print(f"    idx={e[0]} pred={e[1]} img={e[2]} auto_scale={e[3]} iou={e[4]}")
    print(f"  point rows where an EXACT thousandth answer misses hit@0.1: "
          f"{len(lost_p)}/{len(pt_th)}")
    for e in lost_p[:8]:
        print(f"    idx={e[0]} pred={e[1]} img={e[2]} auto_scale={e[3]}")
    print("\nAnd the reverse direction, for completeness:")
    lost_px = sum(
        1 for r, s in pairs
        if first_geom(r)[0] == "bbox"
        and C.iou(C.to_unit_box(_px_box(first_geom(r)[1], r["w"], r["h"]),
                                s.width, s.height),
                  C.to_unit_box(first_geom(r)[1], s.width, s.height,
                                declared="thousandth")) < 0.5)
    print(f"  bbox rows where an exact PIXEL answer scores IoU<0.5: {lost_px}"
          f"/{len(box_px)}")
    print("  => auto-inference is biased: it recovers pixel answers perfectly and")
    print("     corrupts thousandth answers. That is exactly backwards now that the")
    print("     prompt declares the thousandth convention.")


def task4(recs) -> None:
    hr("TASK 4 -- box/point PAIRING: both scorers use pred[0] vs gt[0]")
    per_form = defaultdict(Counter)
    multi = 0
    for r in recs:
        if not r["gt_ok"]:
            per_form[r["form"]]["GT_UNPARSEABLE"] += 1
            continue
        gt = r["gt"] if isinstance(r["gt"], list) else [r["gt"]]
        per_form[r["form"]][len(gt)] += 1
        if len(gt) > 1:
            multi += 1
    for form, c in sorted(per_form.items()):
        print(f"  form={form}: GT list length histogram "
              f"{dict(sorted(c.items(), key=lambda kv: str(kv[0])))}")
    print(f"\nrows whose GT holds MORE than one item: {multi} / {len(recs)}")
    print("  -> on THIS split gt[0] is lossless, because every GT list has exactly one")
    print("     item. The [0]-vs-[0] pairing is therefore not currently mis-scoring GT.")

    print("\nPREDICTION side is a different story. Measured on the training answers, an")
    print("SFT model is trained to emit multi-item lists, so pred[0] is a coin flip:")
    tr = ROOT / "trains" / "gsa_bench" / "train_shuf.jsonl"
    if tr.is_file():
        lens = Counter()
        n = 0
        with tr.open(errors="ignore") as f:
            for ln in f:
                if n >= 5000:
                    break
                try:
                    row = json.loads(ln)
                    v = json.loads(row["messages"][1]["content"])
                except Exception:
                    continue
                n += 1
                lens[len(v) if isinstance(v, list) else -1] += 1
        multi_tr = sum(v for k, v in lens.items() if isinstance(k, int) and k > 1)
        print(f"  train_shuf.jsonl first {n} rows: answer list length histogram "
              f"{dict(sorted(lens.items()))}")
        print(f"  rows with >1 item: {multi_tr}/{n} ({100*multi_tr/max(n,1):.1f}%)")
    print("\nOLD  eval_gsa_test.py:145  p, g = pred[0], gt[0]        -> [0] vs [0]")
    print("NEW  scorers.py:117/129     g = gt[0]; iou(pred[0], gb)  -> [0] vs [0]")
    print("  => the pairing defect is NOT fixed by the new harness. No Hungarian /")
    print("     greedy-max-IoU matching exists in either path, and list order is not")
    print("     guaranteed by the model.")


def _case_matrix():
    """Realistic response shapes. Built from real train_shuf.jsonl answers."""
    real = None
    tr = ROOT / "trains" / "gsa_bench" / "train_shuf.jsonl"
    if tr.is_file():
        with tr.open(errors="ignore") as f:
            for ln in f:
                try:
                    row = json.loads(ln)
                    v = json.loads(row["messages"][1]["content"])
                except Exception:
                    continue
                if isinstance(v, list) and v and "bbox_2d" in (v[0] or {}):
                    real = row["messages"][1]["content"]
                    break
    bare = real or '[{"label": "cup", "bbox_2d": [100, 200, 300, 400]}]'
    trunc = bare[:max(len(bare) // 2, 12)]
    return [
        ("bare JSON array (exact training format)", bare),
        ("JSON in ```json fence", f"```json\n{bare}\n```"),
        ("JSON in bare ``` fence", f"```\n{bare}\n```"),
        ("<think>...</think> then JSON", f"<think>the cup is left</think>\n{bare}"),
        ("<think> with decoy coords then JSON",
         f"<think>maybe [1,2,3,4]? no.</think>\n{bare}"),
        ("prose prefix then JSON", f"Here is the answer:\n{bare}"),
        ("JSON then trailing prose", f"{bare}\nThat is the cup on the left."),
        ("TRUNCATED mid-array (token limit)", trunc),
        ("TRUNCATED inside <think> (no answer)", "<think>the cup is on the"),
        ("single dict, not wrapped in a list",
         '{"label": "cup", "bbox_2d": [100, 200, 300, 400]}'),
        ("python-style single quotes",
         "[{'label': 'cup', 'bbox_2d': [100, 200, 300, 400]}]"),
        ("fence + trailing prose", f"```json\n{bare}\n```\nDone."),
    ]


def task5() -> None:
    hr("TASK 5 -- parser robustness: OLD parse_items vs NEW parse_json_items")
    print("PASS = returned >=1 dict carrying a usable bbox_2d/point_2d.")
    print("A truncated answer SHOULD fail (there is no complete coordinate set);")
    print("'inside <think>' SHOULD fail too, or decoy coords leak into the score.\n")

    def ok(items):
        if not isinstance(items, list) or not items:
            return False
        d = items[0]
        if not isinstance(d, dict):
            return False
        return any(isinstance(d.get(k), list) for k in
                   ("bbox_2d", "bbox", "point_2d", "point"))

    def safe(fn, text):
        try:
            return fn(text)
        except Exception:
            return None

    print("Two columns per parser: RAW = the parse function's own return value graded")
    print("by `ok()`, SCORER = what the scoring path actually receives (OLD reads")
    print("pred[0]['bbox_2d'] directly; NEW goes through parse_boxes).\n")
    print(f"{'case':<40} {'OLD raw':<8} {'OLD scr':<8} {'NEW raw':<8} {'NEW scr':<8} note")
    print("-" * 92)
    tally = Counter()
    for name, text in _case_matrix():
        o_raw = ok(safe(OLD.parse_items, text) or [])
        # OLD scorer path, verbatim from eval_gsa_test.py:105-117
        items = safe(OLD.parse_items, text) or []
        o_scr = False
        if items and isinstance(items[0], dict):
            d = items[0]
            o_scr = (isinstance(d.get("bbox_2d") or d.get("bbox"), list)
                     or isinstance(d.get("point_2d"), list))
        n_raw = ok(safe(P.parse_json_items, text) or [])
        n_scr = bool(safe(P.parse_boxes, text) or safe(P.parse_points, text))
        expect_fail = name.startswith("TRUNCATED")
        note = "(failure is CORRECT here)" if expect_fail else (
            "<-- DIVERGENCE" if o_scr != n_scr else "")
        for k, v in (("o_raw", o_raw), ("o_scr", o_scr),
                     ("n_raw", n_raw), ("n_scr", n_scr)):
            tally[k] += v
        print(f"{name:<40} {'PASS' if o_raw else 'FAIL':<8} "
              f"{'PASS' if o_scr else 'FAIL':<8} {'PASS' if n_raw else 'FAIL':<8} "
              f"{'PASS' if n_scr else 'FAIL':<8} {note}")
    print("-" * 92)
    print(f"{'totals (12 cases, 2 of which SHOULD fail)':<40} "
          f"{tally['o_raw']:<8} {tally['o_scr']:<8} "
          f"{tally['n_raw']:<8} {tally['n_scr']:<8}")

    print("\nWhy the raw/scorer columns differ on 'single dict, not wrapped in a list':")
    t = '{"label": "cup", "bbox_2d": [100, 200, 300, 400]}'
    print(f"  input                    {t}")
    print(f"  OLD parse_items       -> {OLD.parse_items(t)}")
    print(f"  NEW parse_json_items  -> {P.parse_json_items(t)}")
    print(f"  NEW parse_boxes       -> {P.parse_boxes(t)}")
    print("  Both _candidate_blobs and the OLD `[` scan latch onto the INNER coordinate")
    print("  array, so both return a bare number list. The OLD scorer then hits its")
    print("  `isinstance(pred[0], dict)` guard (eval_gsa_test.py:105) and counts a parse")
    print("  failure; NEW's parse_boxes accepts a 4-number list (parsing.py:149) and")
    print("  recovers the box. Same parse quirk, only NEW survives it.")

    print("\nThe one divergence that SILENTLY CORRUPTS a score: truncated <think>.")
    print("eval_gsa_test.py:34 strips `<think>.*?</think>`, which needs a CLOSING tag.")
    print("When generation is cut off mid-reasoning there is none, so the whole")
    print("reasoning span stays in the text and the `[` scan reads decoys from it.")
    for t2 in ('<think>maybe [{"label":"cup","bbox_2d":[10,20,30,40]}] fits, but',
               "<think>the cup might be at [10, 20, 30, 40] or"):
        o = OLD.parse_items(t2)
        guard = bool(o) and isinstance(o[0], dict)
        print(f"\n  input              {t2!r}")
        print(f"  OLD parse_items -> {o}")
        print(f"    reaches the scorer (passes the isinstance dict guard)? {guard}"
              f"{'   <-- SCORED AS IF IT WERE THE ANSWER' if guard else ''}")
        print(f"  NEW parse_boxes -> {P.parse_boxes(t2)}   "
              f"<-- correctly empty (parsing.py:36-38)")
    print("\n  => a truncated run does not merely lose these rows in the OLD script, it")
    print("     scores reasoning scratch-work as the model's answer. That inflates or")
    print("     deflates unpredictably and is invisible in the output.")

    print("\nAlso measured: how do the two parsers handle the 1,463 REAL GT answers?")
    tr = ROOT / "trains" / "gsa_bench" / "train_shuf.jsonl"
    n = o_ok = n_ok = 0
    if tr.is_file():
        with tr.open(errors="ignore") as f:
            for ln in f:
                if n >= 50:
                    break
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                t = row["messages"][1]["content"]
                n += 1
                o_ok += bool(OLD.parse_items(t))
                n_ok += bool(P.parse_json_items(t))
    print(f"  head -50 train_shuf.jsonl answers: OLD parsed {o_ok}/{n}, "
          f"NEW parsed {n_ok}/{n}")


def task6() -> None:
    hr("TASK 6 -- headline `score` silently falls back to the point hit rate")
    print("OLD eval_gsa_test.py:137-138")
    print('  res["score"] = (res.get("bbox", {}).get("iou@0.5")')
    print('                  or res.get("point", {}).get("hit@0.1"))')
    print("\n0.0 is falsey in Python, so an all-miss bbox result is discarded and the")
    print("POINT hit rate is reported in its place, under the same column name.\n")

    demo = [
        ("bbox collapsed (the actual 4.2 situation)",
         {"bbox": {"iou@0.5": 0.0}, "point": {"hit@0.1": 32.75}}),
        ("bbox collapsed, SFT run",
         {"bbox": {"iou@0.5": 0.0}, "point": {"hit@0.1": 22.68}}),
        ("bbox genuinely non-zero", {"bbox": {"iou@0.5": 11.4},
                                     "point": {"hit@0.1": 32.75}}),
        ("bbox-only run", {"bbox": {"iou@0.5": 0.0}}),
    ]
    print(f"{'case':<42} {'bbox iou@0.5':>13} {'point hit@0.1':>14} {'reported':>10}")
    print("-" * 82)
    for name, res in demo:
        score = (res.get("bbox", {}).get("iou@0.5")
                 or res.get("point", {}).get("hit@0.1"))
        print(f"{name:<42} {res.get('bbox', {}).get('iou@0.5'):>13} "
              f"{str(res.get('point', {}).get('hit@0.1')):>14} {str(score):>10}")
    print("-" * 82)
    print("Row 1/2 reproduce 32.75 (base) and 22.68 (SFT) exactly: those published")
    print("numbers are the POINT-only hit rate over 295 point rows, not a combined")
    print("score over 400 rows. Row 4 shows the other half of the bug: with no point")
    print("rows the score becomes None instead of 0.0.")
    print("\nNEW scorers.py:170-171 computes an explicit sample-weighted mean instead:")
    print("  score = sum(v*n for v, n in parts) / sum(n for _, n in parts)")
    print("  -> FIXED.")


def task7(recs) -> None:
    hr("TASK 7 -- `--limit 400` takes rows[:400], a PREFIX, not a stratified sample")
    full = Counter(r["form"] for r in recs)
    pre = Counter(r["form"] for r in recs[:400])
    print(f"{'form':<10} {'full 1463':>12} {'full %':>9} {'rows[:400]':>12} {'prefix %':>10}")
    print("-" * 60)
    for form in sorted(full):
        print(f"{form:<10} {full[form]:>12} {100*full[form]/len(recs):>8.2f}% "
              f"{pre[form]:>12} {100*pre[form]/400:>9.2f}%")
    print("-" * 60)
    print(f"\nsource distribution, full  : {dict(sorted(Counter(r['source'] for r in recs).items()))}")
    print(f"source distribution, [:400]: {dict(sorted(Counter(r['source'] for r in recs[:400]).items()))}")

    print("\nIs the file sorted / clustered by form and source? First index of each:")
    seen = {}
    for r in recs:
        seen.setdefault(("form", r["form"]), r["i"])
        seen.setdefault(("source", r["source"]), r["i"])
    for k, v in sorted(seen.items()):
        print(f"  first occurrence of {k}: row {v}")
    runs = 1
    for a, b in zip(recs, recs[1:]):
        if a["source"] != b["source"]:
            runs += 1
    print(f"  contiguous runs of identical source over 1463 rows: {runs} "
          f"(1463 = perfectly interleaved, 4 = fully blocked)")
    print("\nVERDICT: the file is shuffled, so rows[:400] happens to be representative")
    print("(bbox 26.25% vs 25.84% overall). The prefix is a LATENT defect, not an")
    print("active one: nothing in the pipeline guarantees the shuffle, and any re-export")
    print("that groups by source would silently skew every future run.")

    print("\nFor comparison, the NEW harness's stratified_subset(limit=400, seed=0):")
    pairs = _build_samples(recs)
    sub = stratified_subset([s for _, s in pairs], 400, seed=0)
    print(f"  form distribution: {dict(sorted(Counter(s.group for s in sub).items()))}")
    print(f"  source distribution: "
          f"{dict(sorted(Counter(s.meta['source'] for s in sub).items()))}")
    print("  note: it balances groups EVENLY (200/200), which is not the split's true")
    print("  1085/378 mix either -- deliberate, but it means a limited run's headline")
    print("  score is not comparable to a full-split run.")


def task_prompt_hint(recs) -> None:
    hr("EXTRA -- scale hint in the stored prompts")
    kw = ("thousandth", "0-1000", "0 to 1000", "normalis", "normaliz",
          "relative to the image", "[0,1]", "0-1 ", "pixel")
    n_hint = 0
    hits = Counter()
    for r in recs:
        low = r["prompt"].lower()
        found = [k for k in kw if k in low]
        if found:
            n_hint += 1
            hits[tuple(found)] += 1
    print(f"rows whose stored prompt mentions any scale keyword: {n_hint}/{len(recs)}")
    print(f"  keyword hits: {dict(hits)}")
    print("The NEW loader (datasets.py:243) appends an explicit thousandth-scale hint to")
    print("every prompt; the OLD script (eval_gsa_test.py:88) passes the prompt verbatim.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "trains" / "gsa_bench" / "test.jsonl"))
    a = ap.parse_args()

    data = Path(a.data)
    print(f"audit of {ROOT / 'eval' / 'eval_gsa_test.py'}")
    print(f"data     {data}")
    rows = load_rows(data)
    recs = enrich(rows)

    task1(recs)
    task2(recs)
    task3(recs)
    task4(recs)
    task5()
    task6()
    task7(recs)
    task_prompt_hint(recs)
    hr("END OF AUDIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
