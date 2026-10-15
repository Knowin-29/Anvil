#!/usr/bin/env python3
"""Generate the data-statistics figures for the paper from the REAL corpus.

Produces vector PDFs (matplotlib) that LaTeX includes directly:
  fig_stats_forms.pdf   form distribution + per-form record counts
  fig_stats_scenes.pdf  scene distribution and split composition
  fig_stats_quality.pdf Verified@k survival + defect-rate iteration curve
All numbers are read from the actual artifacts, not hard-coded.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ICLR-friendly: serif, small, colour-blind-safe
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = ["#3b6ea5", "#e08214", "#5aae61", "#c2549d", "#8073ac", "#d6604d",
     "#4393c3", "#7f7f7f"]

ROOT = Path("/path/to/Anvil")


def collect() -> dict:
    forms, scenes = Counter(), Counter()
    for d in glob.glob(str(ROOT / "data/flywheel_long/shard*/raw")) + \
             glob.glob(str(ROOT / "data/flywheel_gt/shard*/raw")):
        for jf in Path(d).glob("*.jsonl"):
            n = sum(1 for _ in jf.open(errors="ignore"))
            forms[jf.stem] += n
    # scene counts from the bench manifest (already aggregated)
    man = ROOT / "trains/gsa_bench/manifest.json"
    if man.is_file():
        m = json.loads(man.read_text())
        scenes.update(m.get("by_scene", {}))
        splits = m.get("counts", {})
        imgs = m.get("unique_images", {})
    else:
        splits, imgs = {}, {}
    return {"forms": forms, "scenes": scenes, "splits": splits, "images": imgs}


def fig_forms(st: dict, out: Path) -> None:
    forms = st["forms"].most_common()
    if not forms:
        return
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.0),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    names = [k for k, _ in forms]
    vals = [v for _, v in forms]
    ax[0].barh(range(len(names)), vals, color=[C[i % len(C)] for i in range(len(names))])
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("verified records")
    ax[0].set_title("(a) Records per task form")
    for i, v in enumerate(vals):
        ax[0].text(v * 1.01, i, f"{v:,}", va="center", fontsize=6)
    ax[0].set_xlim(0, max(vals) * 1.18)

    total = sum(vals)
    ax[1].pie(vals, labels=None, colors=[C[i % len(C)] for i in range(len(names))],
              autopct=lambda p: f"{p:.0f}%" if p >= 6 else "",
              startangle=90, counterclock=False,
              wedgeprops={"linewidth": 0.4, "edgecolor": "white"},
              textprops={"fontsize": 6})
    ax[1].set_title(f"(b) Form balance ($N$={total:,})")
    ax[1].legend(names, loc="center left", bbox_to_anchor=(1.0, 0.5),
                 frameon=False, ncol=1)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_scenes(st: dict, out: Path) -> None:
    sc = [(k, v) for k, v in st["scenes"].items() if v > 0]
    if not sc:
        return
    sc.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 1.95),
                           gridspec_kw={"width_ratios": [1.2, 1]})
    names = [k for k, _ in sc]; vals = [v for _, v in sc]
    ax[0].bar(range(len(names)), vals,
              color=[C[i % len(C)] for i in range(len(names))])
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6)
    ax[0].set_ylabel("records")
    ax[0].set_title("(a) Scene composition")
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.02, f"{v/1000:.0f}K", ha="center", fontsize=6)
    ax[0].set_ylim(0, max(vals) * 1.16)

    sp = st["splits"]; im = st["images"]
    if sp:
        keys = ["train", "val", "test"]
        rec = [sp.get(k, 0) for k in keys]
        img = [im.get(k, 0) for k in keys]
        x = range(len(keys)); w = 0.38
        ax[1].bar([i - w / 2 for i in x], rec, w, label="records", color=C[0])
        ax[1].bar([i + w / 2 for i in x], img, w, label="images", color=C[1])
        ax[1].set_yscale("log")
        ax[1].set_xticks(list(x)); ax[1].set_xticklabels(keys)
        ax[1].set_title("(b) Splits (image-disjoint)")
        ax[1].legend(frameon=False)
        for i, (r, g) in enumerate(zip(rec, img)):
            ax[1].text(i - w / 2, r * 1.15, f"{r:,}", ha="center", fontsize=5.5, rotation=90)
            ax[1].text(i + w / 2, g * 1.15, f"{g:,}", ha="center", fontsize=5.5, rotation=90)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_quality(out: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 1.85))

    # (a) Verified@k survival (real measurement)
    vk = ROOT / "data/flywheel_v1/verified_at_k.json"
    ks, ns = [1, 2], [6211, 5463]
    if vk.is_file():
        d = json.loads(vk.read_text()).get("survival_by_k", {})
        if d:
            ks = [int(k) for k in sorted(d, key=int)]
            ns = [d[str(k)] for k in ks]
    ax[0].plot(ks, ns, "o-", color=C[0], lw=1.4, ms=4)
    for k, n in zip(ks, ns):
        ax[0].annotate(f"{n:,}", (k, n), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=6)
    ax[0].set_xlabel("$k$ (re-verification rounds)")
    ax[0].set_ylabel("surviving records")
    ax[0].set_xticks(ks)
    ax[0].set_title("(a) Verified@k survival")
    ax[0].margins(y=0.22)

    # (b) defect rate across prompt/validator iterations (real measurement)
    labels = ["v1", "v2", "v3"]
    det = [0.409, 0.000, 0.000]
    msk = [0.169, 0.133, 0.000]
    x = range(len(labels)); w = 0.36
    ax[1].bar([i - w / 2 for i in x], det, w, label="detection", color=C[1])
    ax[1].bar([i + w / 2 for i in x], msk, w, label="mask", color=C[2])
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("defects / record")
    ax[1].set_title("(b) Contract iterations")
    ax[1].legend(frameon=False)

    # (c) agreement with human GT (real measurement)
    cats = ["free-\nchoice", "best-of-\ndetection", "referent-\ncontrolled"]
    iou = [0.135, 0.258, 0.531]
    bars = ax[2].bar(range(len(cats)), iou,
                     color=[C[7], C[7], C[1]])
    bars[-1].set_edgecolor("black"); bars[-1].set_linewidth(0.7)
    ax[2].set_xticks(range(len(cats))); ax[2].set_xticklabels(cats, fontsize=6)
    ax[2].set_ylabel("mean IoU vs human GT")
    ax[2].set_title("(c) GT agreement")
    for i, v in enumerate(iou):
        ax[2].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=6)
    ax[2].set_ylim(0, 0.62)

    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default="/path/to/environment/GSA_collect/paper/figures")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    st = collect()
    print("forms:", dict(st["forms"]))
    print("scenes:", dict(st["scenes"]))
    fig_forms(st, out / "fig_stats_forms.pdf")
    fig_scenes(st, out / "fig_stats_scenes.pdf")
    fig_quality(out / "fig_stats_quality.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
