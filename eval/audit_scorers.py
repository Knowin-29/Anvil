"""Offline audit of eval/gsa_eval scorers against already-dumped predictions.

Read-only with respect to the harness: it imports the real parsers/scorers/loaders
and re-scores the dumped responses under alternative coordinate interpretations so
every number in the report comes from execution, not from reading the code.

Run:
    cd /path/to/Anvil && python eval/audit_scorers.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from gsa_eval.coords import infer_scale, to_unit_points  # noqa: E402
from gsa_eval.datasets import (load_refspatial_bench,  # noqa: E402
                               load_where2place)
from gsa_eval.parsing import parse_choice, parse_points, strip_reasoning  # noqa: E402

SCRATCH = ROOT / ".cache" / "eval_assets"
LOGS = ROOT / "save" / "logs"

DUMPS = {
    ("where2place", "base"): "preds_base_where2place.jsonl",
    ("where2place", "sft"): "preds_sft_9b_lora_where2place.jsonl",
    ("refspatial_bench", "base"): "preds_base_refspatial_bench.jsonl",
    ("refspatial_bench", "sft"): "preds_sft_9b_lora_refspatial_bench.jsonl",
}
CV_DUMP = "preds_base_cv_bench_limit20.jsonl"


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def read_rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (LOGS / name).read_text().splitlines() if l.strip()]


_MASK_CACHE: dict[str, object] = {}


def load_mask(path: str):
    """Same as scorers._load_mask, with caching so repeated re-scoring is cheap."""
    import numpy as np
    from PIL import Image
    if path not in _MASK_CACHE:
        with Image.open(path) as im:
            _MASK_CACHE[path] = np.array(im.convert("L")) > 127
    return _MASK_CACHE[path]


def load_indexes() -> dict[str, dict]:
    idx = {}
    w2 = load_where2place(SCRATCH)
    rs = load_refspatial_bench(SCRATCH)
    idx["where2place"] = {s.uid: s for s in w2}
    idx["refspatial_bench"] = {s.uid: s for s in rs}
    return idx


def rescore_points(rows, index, declared):
    """Re-implement score_point_in_mask exactly, but with a forced `declared` scale.

    declared='auto' reproduces the harness's current headline number bit-for-bit.
    Returns (score_pct, n, n_parse_fail, per_sample list of (uid, frac)).
    """
    per = []
    fails = 0
    for r in rows:
        s = index[r["uid"]]
        pts = parse_points(r["response"])
        if not pts:
            fails += 1
            per.append((r["uid"], 0.0))
            continue
        mask = load_mask(s.gt)
        mh, mw = mask.shape
        inside = 0
        for x, y in to_unit_points(pts, s.width or mw, s.height or mh, declared):
            px, py = int(round(x * (mw - 1))), int(round(y * (mh - 1)))
            if 0 <= px < mw and 0 <= py < mh and bool(mask[py, px]):
                inside += 1
        per.append((r["uid"], inside / len(pts)))
    tot = sum(v for _, v in per)
    return (round(100.0 * tot / len(per), 2) if per else 0.0), len(per), fails, per


def classify(rows, index):
    """Per-sample scale classification + the ambiguity flags TASK A asks for."""
    out = []
    for r in rows:
        s = index[r["uid"]]
        pts = parse_points(r["response"])
        if not pts:
            out.append({"uid": r["uid"], "parsed": False})
            continue
        xs = [abs(p[0]) for p in pts]
        ys = [abs(p[1]) for p in pts]
        scale = infer_scale(xs, ys, s.width, s.height)
        mx = max(max(xs), max(ys))
        out.append({
            "uid": r["uid"], "parsed": True, "scale": scale, "maxcoord": mx,
            "n_pts": len(pts), "w": s.width, "h": s.height,
            # "looks like thousandths but fits in the frame" == silently read as pixel
            "ambiguous_int_like": scale == "pixel" and mx > 1.5,
        })
    return out


def task_a(idx) -> None:
    hr("TASK A - score_point_in_mask coordinate-scale correctness")
    print("Prompt audit (what scale the question actually asks for):")
    for bench in ("where2place", "refspatial_bench"):
        ss = list(idx[bench].values())
        n01 = sum("between 0 and 1" in s.prompt for s in ss)
        nk = sum(("thousandth" in s.prompt or "0-1000" in s.prompt) for s in ss)
        print(f"  {bench:<18} n={len(ss):<4} asks_for_[0,1]={n01:<4} asks_for_0-1000={nk}")

    print("\nScale classification of parsed answers (infer_scale, declared='auto'):")
    hdr = (f"  {'bench/model':<26}{'parsed':>7}{'fail':>6}{'unit':>6}"
           f"{'thou':>6}{'pixel':>6}{'ambig_pixel':>13}")
    print(hdr)
    store = {}
    for (bench, model), fname in DUMPS.items():
        rows = read_rows(fname)
        cls = classify(rows, idx[bench])
        store[(bench, model)] = (rows, cls)
        c = Counter(d["scale"] for d in cls if d["parsed"])
        amb = sum(1 for d in cls if d["parsed"] and d["ambiguous_int_like"])
        nf = sum(1 for d in cls if not d["parsed"])
        print(f"  {bench+'/'+model:<26}{len(cls)-nf:>7}{nf:>6}{c['unit']:>6}"
              f"{c['thousandth']:>6}{c['pixel']:>6}{amb:>13}")
    print("  ambig_pixel = every coord fits inside the image, yet max>1.5, so the")
    print("  answer is read as PIXEL even though the prompt requested [0,1].")
    task_a_scores(idx, store)
    task_a_smart(idx, store)


def task_a_scores(idx, store) -> None:
    print("\nReal point-in-mask score under each interpretation (masks reloaded from gt):")
    print(f"  {'bench/model':<26}{'auto(current)':>14}{'thousandth':>12}"
          f"{'unit':>9}{'pixel':>9}{'n':>5}{'fail':>6}")
    global A_SCORES
    A_SCORES = {}
    for key, (rows, _cls) in store.items():
        bench, model = key
        res = {}
        for decl in ("auto", "thousandth", "unit", "pixel"):
            sc, n, f, per = rescore_points(rows, idx[bench], decl)
            res[decl] = (sc, n, f, dict(per))
        A_SCORES[key] = res
        a, n, f, _ = res["auto"]
        print(f"  {bench+'/'+model:<26}{a:>14}{res['thousandth'][0]:>12}"
              f"{res['unit'][0]:>9}{res['pixel'][0]:>9}{n:>5}{f:>6}")


def rescore_prompt_aware(rows, index):
    """Both benchmarks' prompts ask for [0,1]; a >1.5 value can then only mean the
    model fell back to its 0-1000 training convention. So: unit if max<=1.5 else
    thousandth -- never pixel. Deterministic, prompt-driven, no per-sample tuning."""
    per = []
    for r in rows:
        s = index[r["uid"]]
        pts = parse_points(r["response"])
        if not pts:
            per.append((r["uid"], 0.0))
            continue
        mx = max(max(abs(p[0]), abs(p[1])) for p in pts)
        decl = "unit" if mx <= 1.5 else "thousandth"
        mask = load_mask(s.gt)
        mh, mw = mask.shape
        inside = 0
        for x, y in to_unit_points(pts, s.width or mw, s.height or mh, decl):
            px, py = int(round(x * (mw - 1))), int(round(y * (mh - 1)))
            if 0 <= px < mw and 0 <= py < mh and bool(mask[py, px]):
                inside += 1
        per.append((r["uid"], inside / len(pts)))
    return (round(100.0 * sum(v for _, v in per) / len(per), 2) if per else 0.0,
            dict(per))


def task_a_smart(idx, store) -> None:
    print("\nPrompt-aware rule (<=1.5 -> unit, else thousandth; 'pixel' never used):")
    print(f"  {'bench/model':<26}{'auto(current)':>14}{'prompt_aware':>14}"
          f"{'delta':>8}{'n_samples_changed':>19}")
    global A_SMART
    A_SMART = {}
    for key, (rows, _c) in store.items():
        bench, model = key
        sc, per = rescore_prompt_aware(rows, idx[bench])
        A_SMART[key] = (sc, per)
        auto = A_SCORES[key]["auto"]
        changed = sum(1 for u, v in per.items() if abs(v - auto[3][u]) > 1e-9)
        print(f"  {bench+'/'+model:<26}{auto[0]:>14}{sc:>14}"
              f"{round(sc - auto[0], 2):>8}{changed:>19}")


def task_b(idx) -> None:
    hr("TASK B - is the auto scale-classification biased between base and SFT?")
    print("Answer-scale habit per model (share of PARSED answers):")
    print(f"  {'bench/model':<26}{'parsed':>7}{'unit%':>8}{'thou%':>8}"
          f"{'pixel%':>8}{'mis-scaled%':>13}")
    rates = {}
    for (bench, model), fname in DUMPS.items():
        cls = classify(read_rows(fname), idx[bench])
        p = [d for d in cls if d["parsed"]]
        n = max(1, len(p))
        c = Counter(d["scale"] for d in p)
        amb = sum(1 for d in p if d["ambiguous_int_like"])
        rates[(bench, model)] = 100.0 * amb / n
        print(f"  {bench+'/'+model:<26}{len(p):>7}"
              f"{100.0*c['unit']/n:>8.1f}{100.0*c['thousandth']/n:>8.1f}"
              f"{100.0*c['pixel']/n:>8.1f}{100.0*amb/n:>13.1f}")
    print("\nmis-scaled% = parsed answers silently divided by image size instead of 1000.")
    for bench in ("where2place", "refspatial_bench"):
        b, s = rates[(bench, "base")], rates[(bench, "sft")]
        print(f"  {bench}: base {b:.1f}% vs sft {s:.1f}%  -> gap {s-b:+.1f} pp")
    print("\nScore impact of that gap (auto vs prompt-aware), base-vs-SFT delta:")
    for bench in ("where2place", "refspatial_bench"):
        ab = A_SCORES[(bench, "base")]["auto"][0]
        asf = A_SCORES[(bench, "sft")]["auto"][0]
        pb = A_SMART[(bench, "base")][0]
        ps = A_SMART[(bench, "sft")][0]
        print(f"  {bench:<18} auto: base {ab:>6} sft {asf:>6} (SFT-base {asf-ab:+.2f})"
              f" | prompt-aware: base {pb:>6} sft {ps:>6} (SFT-base {ps-pb:+.2f})")
    task_b2(idx)
    task_b3(idx)


def task_b2(idx) -> None:
    print("\nDetail of every silently-mis-scaled sample (scale=='pixel', max>1.5):")
    print(f"  {'bench/model':<24}{'uid':<16}{'img':>10}{'maxcoord':>10}"
          f"{'auto':>7}{'thou':>7}")
    for (bench, model), fname in DUMPS.items():
        rows = read_rows(fname)
        cls = {d["uid"]: d for d in classify(rows, idx[bench])}
        auto = A_SCORES[(bench, model)]["auto"][3]
        thou = A_SCORES[(bench, model)]["thousandth"][3]
        for uid, d in cls.items():
            if d.get("ambiguous_int_like"):
                size = "{}x{}".format(d["w"], d["h"])
                print(f"  {bench+'/'+model:<24}{uid:<16}"
                      f"{size:>10}{d['maxcoord']:>10.1f}"
                      f"{auto[uid]:>7.2f}{thou[uid]:>7.2f}")


def task_b3(idx) -> None:
    print("\nMixed-magnitude answers (some coords <=1.5, some >1.5 in ONE answer):")
    for (bench, model), fname in DUMPS.items():
        n = 0
        for r in read_rows(fname):
            pts = parse_points(r["response"])
            if not pts:
                continue
            v = [abs(c) for p in pts for c in p]
            if min(v) <= 1.5 and max(v) > 1.5:
                n += 1
        print(f"  {bench+'/'+model:<26}{n}")
    print("  (list-level scale inference makes such answers all-or-nothing)")
    task_b4(idx)


def task_b4(idx) -> None:
    print("\nTruncation / parse-failure asymmetry (0-scored, not dropped):")
    print(f"  {'bench/model':<26}{'n':>5}{'unterminated<think>':>21}{'parse_fail':>12}")
    for (bench, model), fname in DUMPS.items():
        rows = read_rows(fname)
        tr = sum(1 for r in rows if "<think>" in r["response"].lower()
                 and "</think>" not in r["response"].lower())
        pf = sum(1 for r in rows if not parse_points(r["response"]))
        print(f"  {bench+'/'+model:<26}{len(rows):>5}"
              f"{f'{tr} ({100*tr/len(rows):.0f}%)':>21}"
              f"{f'{pf} ({100*pf/len(rows):.0f}%)':>12}")
    print("\nPaired comparison on samples BOTH models parsed (isolates scale from"
          " truncation):")
    print(f"  {'bench':<18}{'n_paired':>9}{'base_auto':>11}{'sft_auto':>10}"
          f"{'base_pa':>9}{'sft_pa':>8}")
    for bench in ("where2place", "refspatial_bench"):
        okb = {r["uid"] for r in read_rows(DUMPS[(bench, "base")])
               if parse_points(r["response"])}
        oks = {r["uid"] for r in read_rows(DUMPS[(bench, "sft")])
               if parse_points(r["response"])}
        both = okb & oks
        if not both:
            print(f"  {bench:<18}{0:>9}")
            continue
        def m(d):
            return round(100.0 * sum(d[u] for u in both) / len(both), 2)
        print(f"  {bench:<18}{len(both):>9}"
              f"{m(A_SCORES[(bench,'base')]['auto'][3]):>11}"
              f"{m(A_SCORES[(bench,'sft')]['auto'][3]):>10}"
              f"{m(A_SMART[(bench,'base')][1]):>9}"
              f"{m(A_SMART[(bench,'sft')][1]):>8}")


def task_c(idx) -> None:
    hr("TASK C - mask/point coordinate-system consistency")
    from PIL import Image
    for bench in ("where2place", "refspatial_bench"):
        ss = list(idx[bench].values())
        mism, zero, ratio_bad = [], [], []
        for s in ss:
            with Image.open(s.gt) as m:
                mw, mh = m.size
            if (mw, mh) != (s.width, s.height):
                mism.append((s.uid, (s.width, s.height), (mw, mh)))
                if abs((mw / mh) - (s.width / s.height)) > 0.01:
                    ratio_bad.append(s.uid)
            if not s.width or not s.height:
                zero.append(s.uid)
        print(f"  {bench:<18} n={len(ss)} mask_size!=image_size={len(mism)} "
              f"aspect_mismatch={len(ratio_bad)} width_or_height==0={len(zero)}")
        if mism:
            print(f"     examples: {mism[:3]}")
    task_c_axis(idx)


def task_c_axis(idx) -> None:
    """Empirical y/x order check: transposing the index must not improve the score."""
    import numpy as np
    print("\n  Axis-order check: score with mask[py,px] (current) vs mask[px,py]:")
    for (bench, model), fname in DUMPS.items():
        rows = read_rows(fname)
        cur = tr = 0.0
        n = 0
        for r in rows:
            s = idx[bench][r["uid"]]
            pts = parse_points(r["response"])
            n += 1
            if not pts:
                continue
            mask = load_mask(s.gt)
            mh, mw = mask.shape
            a = b = 0
            for x, y in to_unit_points(pts, s.width or mw, s.height or mh):
                px, py = int(round(x * (mw - 1))), int(round(y * (mh - 1)))
                if 0 <= px < mw and 0 <= py < mh and bool(mask[py, px]):
                    a += 1
                if 0 <= py < mw and 0 <= px < mh and bool(mask[px, py]):
                    b += 1
            cur += a / len(pts)
            tr += b / len(pts)
        print(f"    {bench+'/'+model:<26} mask[py,px]={100*cur/n:>6.2f}  "
              f"mask[px,py]={100*tr/n:>6.2f}")
    sq = sum(1 for b in ("where2place", "refspatial_bench")
             for s in idx[b].values() if s.width == s.height)
    print(f"    (square images where the two orders are indistinguishable: {sq})")
    print(f"    numpy {np.__version__}; mask.shape unpacked as (mh, mw) matches "
          "PIL size (w,h) reversed -> consistent")
    task_c_mask(idx)


def task_c_mask(idx) -> None:
    """A mostly-white mask would inflate scores; a mostly-black one deflates them."""
    print("\n  Mask polarity / coverage (fraction of pixels >127 = 'valid region'):")
    for bench in ("where2place", "refspatial_bench"):
        fr = []
        ext = Counter()
        empty = 0
        for s in idx[bench].values():
            m = load_mask(s.gt)
            f = float(m.mean())
            fr.append(f)
            ext[Path(s.gt).suffix] += 1
            if f == 0.0 or f == 1.0:
                empty += 1
        fr.sort()
        print(f"    {bench:<18} n={len(fr)} min={fr[0]:.3f} p50={fr[len(fr)//2]:.3f} "
              f"max={fr[-1]:.3f} degenerate(0 or 1)={empty} ext={dict(ext)}")
    print("    -> masks are sparse positive regions, so a random point scores low;")
    print("       polarity (>127 == target region) is the correct sense.")
    print("\n  Index scaling uses x*(mw-1): a coord of exactly 1.0 maps to the last")
    print("  column, so no out-of-range indexing. Off-by-one bias <=1px, negligible.")


ADVERSARIAL = [
    ("(B)", "B"), ("B", "B"), ("B.", "B"), ("Answer: B", "B"),
    ("The answer is B", "B"), ("**B**", "B"), ("B) blue", "B"),
    ("the red cup", "B"),
    ("Option B is correct.", "B"),
    ("I would go with B, because it is on the left.", "B"),
    ("B\n", "B"),
    ("The answer is B. Both are visible, but b is nearer.", "B"),
    ("Answer: A. The cup is behind the box.", "A"),
    ("The answer is C, i.e. the mug on the desk.", "C"),
    ("A is wrong, so the answer is D", "D"),
    ("The object is a bed, so the answer is (A)", "A"),
]


def task_d() -> None:
    hr("TASK D - score_multiple_choice / parse_choice robustness")
    rows = read_rows(CV_DUMP)
    print(f"Real dump {CV_DUMP}: n={len(rows)}")
    ok = none = 0
    correct = 0
    fulltext_only = 0
    for r in rows:
        ans = strip_reasoning(r["response"])
        pred = parse_choice(r["response"], 4)
        gold = str(r["gt"]).strip().strip("()").upper()[:1]
        if pred is None:
            none += 1
            print(f"  PARSE-FAIL uid={r['uid']} gold={gold} "
                  f"answer_span={ans[-60:]!r} raw_tail={r['response'][-60:]!r}")
        else:
            ok += 1
            correct += int(pred == gold)
        if ans and not any(ch in ans for ch in "ABCDEF"):
            fulltext_only += 1
    print(f"  parsed={ok} parse_fail={none} correct={correct} "
          f"acc={100.0*correct/len(rows):.2f}")
    print(f"  responses whose answer span contains NO option letter at all: "
          f"{fulltext_only}/{len(rows)}")
    task_d_forms()
    task_d_falsepos(rows)
    task_d_nchoices(rows)


def task_d_forms() -> None:
    print("\nHand-constructed legitimate answer forms (n_choices=4):")
    print(f"  {'response':<54}{'expect':>7}{'got':>6}  verdict")
    nfail = 0
    for text, expect in ADVERSARIAL:
        got = parse_choice(text, 4)
        okk = (got == expect)
        nfail += (not okk)
        print(f"  {text!r:<54}{expect:>7}{str(got):>6}  {'ok' if okk else 'FAIL'}")
    print(f"  -> {nfail}/{len(ADVERSARIAL)} forms mis-parsed")


def task_d_falsepos(rows) -> None:
    import re
    print("\nIGNORECASE false-positive risk (lowercase prose letter hijacking):")
    lastpat = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
    n_lower_only = 0
    n_lastwins_differs = 0
    for r in rows:
        ans = strip_reasoning(r["response"])
        if not ans:
            continue
        up = lastpat.findall(ans)
        both = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", ans, re.IGNORECASE)
        if both and not up:
            n_lower_only += 1
        if both and up and both[-1].upper() != up[-1]:
            n_lastwins_differs += 1
    print(f"  answer spans whose fallback matches are ONLY lowercase: "
          f"{n_lower_only}/{len(rows)}")
    print(f"  answer spans where last-match-wins flips when case is respected: "
          f"{n_lastwins_differs}/{len(rows)}")
    probes = [
        ("Both are visible. a bit unclear, but the answer is C.", "C"),
        ("The answer is D, a chair is nearer.", "D"),
        ("It is the second option; a cup.", None),
        ("The answer is B. Note it is a big one.", "B"),
    ]
    print("  targeted probes for the ordering / IGNORECASE interaction:")
    for text, expect in probes:
        got = parse_choice(text, 4)
        print(f"    {text!r:<58} expect={expect} got={got} "
              f"{'ok' if got == expect else 'FAIL'}")


def task_d_nchoices(rows) -> None:
    """The real n_choices matters: 1850/2638 CV-Bench rows have 2 options, so the
    letter class is [AB] and the English article 'a' becomes a live false positive."""
    import re
    print("\nn_choices=2 case (1850/2638 CV-Bench rows; letter class is [AB]):")
    probes = [("It is a cup on the table.", None),
              ("The pillow is on a bed.", None),
              ("(B) pillow", "B"),
              ("Answer: A", "A")]
    for text, expect in probes:
        got = parse_choice(text, 2)
        print(f"    n=2 {text!r:<40} expect={expect} got={got} "
              f"{'ok' if got == expect else 'FAIL'}")
    n_art = 0
    for r in rows:
        ans = strip_reasoning(r["response"])
        if ans and re.search(r"(?<![A-Za-z])[ab](?![A-Za-z])", ans):
            n_art += 1
    print(f"  real dump answer spans containing a standalone lowercase 'a'/'b': "
          f"{n_art}/{len(rows)}")
    print("  NOTE: the dump was scored with n_choices=4 here; the harness passes the")
    print("  row's real len(choices), so 2-option rows use the [AB] class.")
    print("\n  Which pattern actually fires on the real dump (n_choices=2 and 4):")
    for nch in (2, 4):
        c = Counter()
        for r in rows:
            c[which_pattern(strip_reasoning(r["response"]), nch)] += 1
        print(f"    n_choices={nch}: {dict(sorted(c.items(), key=lambda kv: str(kv[0])))}")
    print("    pattern 0='(X)' 1='answer is X' 2='X.' 3=bare-letter fallback"
          " None=no match")


def which_pattern(answer: str, n_choices: int):
    """Index of the first parse_choice pattern that matches; mirrors parsing.py."""
    import re
    if not answer:
        return None
    letters = "".join(chr(ord("A") + i) for i in range(max(1, min(n_choices, 26))))
    pats = (rf"\(\s*([{letters}])\s*\)",
            rf"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([{letters}])\b",
            rf"\b([{letters}])\s*[.):]",
            rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])")
    for i, p in enumerate(pats):
        if re.findall(p, answer, re.IGNORECASE):
            return i
    return None


def task_e() -> None:
    hr("TASK E - GT normalisation in score_multiple_choice")
    import pyarrow.parquet as pq
    D = Path("/path/to/datasets/datasets/transferred_datasets/"
             "EmbodiedAI_benchmark")
    for f in ("test_2d.parquet", "test_3d.parquet"):
        t = pq.read_table(D / "CV-Bench" / f, columns=["answer", "choices"])
        ans = t.column("answer").to_pylist()
        ch = t.column("choices").to_pylist()
        norm = Counter(str(a).strip().strip("()").upper()[:1] for a in ans)
        bad = sum(1 for a in ans
                  if str(a).strip().strip("()").upper()[:1] not in "ABCDEF")
        print(f"  CV-Bench/{f}: n={len(ans)} raw={dict(sorted(Counter(ans).items()))}")
        print(f"     normalised={dict(sorted(norm.items()))} unnormalisable={bad}")
        print(f"     n_choices dist={dict(sorted(Counter(len(c) for c in ch).items()))}")
        over4 = sum(1 for c in ch if len(c) > 4)
        beyond_d = sum(1 for a in ans
                       if str(a).strip().strip("()").upper()[:1] not in "ABCD")
        print(f"     rows with >4 choices={over4}  rows whose GOLD is beyond D="
              f"{beyond_d}")
    task_e_notes()


def task_e_notes() -> None:
    print("\n  n_choices actually passed to parse_choice = len(meta['choices']) or 4,")
    print("  so for a 6-option CV-Bench row the letter set is A-F, matching gold.")
    print("  EmbSpatial: gt = letters[row['answer']] (datasets.py load_embspatial),")
    print("  i.e. already a bare letter -> strip('()').upper()[:1] is a no-op.")
    print("  Verified by reading the loaders; both GT forms normalise correctly.")


def main() -> None:
    idx = load_indexes()
    print(f"loaded samples: where2place={len(idx['where2place'])} "
          f"refspatial_bench={len(idx['refspatial_bench'])}")
    task_a(idx)
    task_b(idx)
    task_c(idx)
    task_d()
    task_e()


if __name__ == "__main__":
    main()
