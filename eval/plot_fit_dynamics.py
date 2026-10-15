#!/usr/bin/env python3
"""Plot the measured training dynamics: mixed 8-form vs single-form at equal volume.

This answers a concrete objection -- that heterogeneous multi-form supervision is hard to
fit, so a single model cannot absorb all eight output conventions. The controlled run says
otherwise: at identical sample count, the mixed corpus converges to the same loss and
token accuracy as a single-form corpus.

Reads the real swift logs, so the figure cannot drift from what was actually run.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"'loss': '([\d.]+)'.*?'token_acc': '([\d.]+)'"
                 r".*?'global_step/max_steps': '(\d+)/(\d+)'")


def read(log: Path):
    rows = []
    if not log.is_file():
        return rows
    for ln in log.open(errors="ignore"):
        m = PAT.search(ln)
        if m:
            rows.append((int(m.group(3)), float(m.group(1)), float(m.group(2))))
    rows.sort()
    # de-duplicate repeated step logs, keep last
    out, seen = [], set()
    for s, l, a in reversed(rows):
        if s not in seen:
            seen.add(s); out.append((s, l, a))
    return sorted(out)


def ema(xs, beta=0.6):
    out, m = [], None
    for x in xs:
        m = x if m is None else beta * m + (1 - beta) * x
        out.append(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "GSA_collect/paper/figures"
                                        / "fig_fit_dynamics.pdf"))
    a = ap.parse_args()

    runs = [("mixed 8 forms", "fit_mixed", "#3b6ea5"),
            ("single form (bbox)", "fit_bbox", "#e08214")]
    data = {lbl: read(ROOT / "logs" / f"{f}.log") for lbl, f, _ in runs}
    n = min(len(v) for v in data.values() if v) if all(data.values()) else 0
    if not n:
        print("insufficient data"); return 1

    fig, ax = plt.subplots(1, 2, figsize=(5.2, 1.6))
    for lbl, _, col in runs:
        r = data[lbl]
        s = [x[0] for x in r]; L = [x[1] for x in r]; A = [x[2] for x in r]
        ax[0].plot(s, L, color=col, alpha=0.25, lw=0.8)
        ax[0].plot(s, ema(L), color=col, lw=1.6, label=lbl)
        ax[1].plot(s, A, color=col, alpha=0.25, lw=0.8)
        ax[1].plot(s, ema(A), color=col, lw=1.6, label=lbl)
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("training loss")
    ax[0].set_title("(a) Loss"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("optimizer step"); ax[1].set_ylabel("token accuracy")
    ax[1].set_title("(b) Token accuracy"); ax[1].legend(frameon=False, loc="lower right")

    # annotate the converged means over the common step range
    lim = min(max(x[0] for x in data[l]) for l, _, _ in runs)
    txt = []
    for lbl, _, _ in runs:
        tail = [x for x in data[lbl] if x[0] > lim * 0.6]
        if tail:
            txt.append(f"{lbl}: loss {sum(t[1] for t in tail)/len(tail):.3f}, "
                       f"acc {sum(t[2] for t in tail)/len(tail):.3f}")
    print(" | ".join(txt))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
