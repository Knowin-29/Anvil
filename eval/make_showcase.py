#!/usr/bin/env python3
"""Per-form showcase figure: each task form gets its OWN visual language.

The previous version drew whatever geometry it found, so every panel looked like
"image + box" even though the forms carry very different information. Concretely:
  bbox        -> ONE box, labelled
  detection   -> MANY boxes, each a different colour + label (the point is multiplicity)
  mask        -> filled translucent polygon + vertex markers (the point is the outline,
                 which must visibly differ from a rectangle)
  point       -> crosshair marker, no box (the point is that there is no extent)
  depth       -> no 2D geometry exists in the payload; render the rank/relative-depth
                 ordering as an overlaid bar strip instead of faking a box
  affordance  -> object box (thin, grey) + contact SUB-PART box (bold) + contact dot,
                 so the containment relation is the visual
  dof         -> grasp point + jaw axis as an explicit two-jaw gripper glyph + wrist angle
  action_cot  -> numbered step list rendered beside the image, with the target point
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "figure.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    # Keep labels selectable in the canonical SVG and editable as embedded
    # TrueType text in the derived vector PDF.
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GSA_RUNTIME_DATA_ROOT", str(REPO_ROOT)))
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
ACC = "#D55E00"
RAW_DIRS: list[Path] = []
EXCLUDED_IMAGE_IDS: set[str] = set()


def load(form: str, want: int = 40) -> list[dict]:
    out = []
    dirs = RAW_DIRS or [Path(d) for d in (
        glob.glob(str(ROOT / "data/flywheel_v2/full500/raw"))
        + glob.glob(str(ROOT / "data/flywheel_long/shard*/raw"))
        + glob.glob(str(ROOT / "data/flywheel_gt/shard*/raw")))]
    for d in dirs:
        p = d / f"{form}.jsonl"
        if not p.is_file():
            continue
        lines = [l for l in p.read_text(errors="ignore").splitlines()[:3000] if l.strip()]
        random.shuffle(lines)
        for l in lines:
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("image_sha256") in EXCLUDED_IMAGE_IDS:
                continue
            if Path(r.get("image_path", "")).is_file():
                out.append(r)
                if len(out) >= want:
                    return out
    return out


def load_exact(form: str, image_sha256: str) -> dict:
    """Recover one frozen showcase record without rerunning visual selection."""
    dirs = RAW_DIRS or [Path(d) for d in (
        glob.glob(str(ROOT / "data/flywheel_v2/full500/raw"))
        + glob.glob(str(ROOT / "data/flywheel_long/shard*/raw"))
        + glob.glob(str(ROOT / "data/flywheel_gt/shard*/raw")))]
    matches = []
    for directory in dirs:
        path = directory / f"{form}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            if image_sha256 not in line:
                continue
            record = json.loads(line)
            if record.get("image_sha256") == image_sha256:
                matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"frozen showcase selection for {form} resolves to {len(matches)} records"
        )
    if image_sha256 in EXCLUDED_IMAGE_IDS:
        raise ValueError(f"frozen showcase selection for {form} is now excluded")
    return matches[0]


def items(r: dict) -> list:
    p = r.get("payload")
    return p if isinstance(p, list) else ([p] if p else [])


def show(ax, r: dict):
    im = Image.open(r["image_path"]).convert("RGB")
    im.thumbnail((420, 420))
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#999")
    return im.size


def S(b, W, H):
    return [b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H]


def P(p, W, H):
    return p[0] / 1000 * W, p[1] / 1000 * H


# ---------------- per-form panels ----------------

def panel_bbox(ax, r):
    W, H = show(ax, r)
    it = items(r)[0]
    x1, y1, x2, y2 = S(it.get("bbox_2d") or it.get("bbox"), W, H)
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    ec=ACC, lw=2.0))
    ax.text(x1, y1 - 3, it.get("label", ""), color="white", fontsize=6,
            va="bottom", bbox=dict(fc=ACC, ec="none", pad=1.0))
    ax.set_title("bbox: one referent, one box", fontsize=6.5, pad=2)


def panel_detection(ax, r):
    W, H = show(ax, r)
    its = [i for i in items(r) if isinstance(i, dict)][:6]
    for k, it in enumerate(its):
        c = PAL[k % len(PAL)]
        x1, y1, x2, y2 = S(it.get("bbox_2d") or it.get("bbox"), W, H)
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                        ec=c, lw=1.5))
        ax.text(x1, y1 - 2, it.get("label", "")[:14], color="white", fontsize=5,
                va="bottom", bbox=dict(fc=c, ec="none", pad=0.7))
    ax.set_title(f"detection: {len(its)} distinct objects", fontsize=6.5, pad=2)


def panel_mask(ax, r):
    W, H = show(ax, r)
    it = items(r)[0]
    poly = [P(p, W, H) for p in it["polygon"]]
    ax.add_patch(mpatches.Polygon(poly, closed=True, fc=ACC, ec=ACC,
                                  alpha=0.30, lw=1.8))
    xs, ys = zip(*poly)
    ax.plot(xs + (xs[0],), ys + (ys[0],), "-", color=ACC, lw=1.6)
    ax.plot(xs, ys, "o", color="white", mec=ACC, mew=1.0, ms=3.0)
    ax.set_title(f"mask: {len(poly)}-vertex outline", fontsize=6.5, pad=2)


def panel_point(ax, r):
    W, H = show(ax, r)
    it = items(r)[0]
    x, y = P(it["point_2d"], W, H)
    ax.plot([x], [y], marker="+", color=ACC, ms=13, mew=2.2)
    ax.add_patch(mpatches.Circle((x, y), 0.045 * max(W, H), fill=False,
                                 ec=ACC, lw=1.4))
    ax.text(x + 0.05 * W, y, it.get("label", ""), color="white", fontsize=6,
            va="center", bbox=dict(fc=ACC, ec="none", pad=1.0))
    ax.set_title("point: location without extent", fontsize=6.5, pad=2)


def panel_depth(ax, r):
    """The depth payload has NO 2D geometry -- show the ordering it does carry.

    Rendered on an opaque strip at the bottom so the bars stay readable regardless
    of how dark the underlying photo is.
    """
    W, H = show(ax, r)
    its = [i for i in items(r) if isinstance(i, dict)]
    its.sort(key=lambda d: d.get("rank", 99))
    its = its[:3]
    strip_h = (0.10 * len(its) + 0.06) * H
    ax.add_patch(mpatches.Rectangle((0, H - strip_h), W, strip_h,
                                    fc="white", alpha=0.86, ec="none"))
    for k, it in enumerate(its):
        d = float(it.get("relative_depth", 0.0))
        bw = (0.10 + 0.80 * (1.0 - d)) * W * 0.90
        yy = H - strip_h + 0.03 * H + k * 0.10 * H
        ax.add_patch(mpatches.Rectangle((0.03 * W, yy), bw, 0.062 * H,
                                        fc=PAL[k % len(PAL)], ec="none", alpha=0.95))
        ax.text(0.045 * W, yy + 0.031 * H,
                f"#{it.get('rank','?')} {str(it.get('label',''))[:14]} ({d:.2f})",
                color="white", fontsize=5.0, va="center", fontweight="bold")
    ax.set_title("depth: near$\\rightarrow$far ordering", fontsize=6.5, pad=2)


def panel_affordance(ax, r):
    W, H = show(ax, r)
    it = items(r)[0]
    context = r.get("gsa_context") or r.get("g_context") or {}
    primary = context.get("primary") if isinstance(context.get("primary"), dict) else context
    g = primary.get("bbox")
    if g:
        ox1, oy1, ox2, oy2 = S(g, W, H)
        ax.add_patch(mpatches.Rectangle((ox1, oy1), ox2 - ox1, oy2 - oy1, fill=False,
                                        ec="#9aa0a6", lw=1.0, ls=(0, (3, 2))))
        ax.text(ox1, oy1 - 2, "object", color="#5f6368", fontsize=5, va="bottom")
    x1, y1, x2, y2 = S(it.get("bbox_2d") or it.get("bbox"), W, H)
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True,
                                    fc=ACC, alpha=0.22, ec=ACC, lw=2.0))
    cx, cy = P(it["contact_point_2d"], W, H)
    ax.plot([cx], [cy], "o", color="white", mec=ACC, mew=2.0, ms=6)
    ax.text(x1, y2 + 4, f"{it.get('affordance','')} @ {it.get('label','')}"[:22],
            color="white", fontsize=5.4, va="top",
            bbox=dict(fc=ACC, ec="none", pad=0.8))
    ax.set_title("affordance: sub-part $\\subset$ object", fontsize=6.5, pad=2)


def panel_dof(ax, r):
    W, H = show(ax, r)
    it = items(r)[0]
    gx, gy = P(it["grasp_point_2d"], W, H)
    ax_, ay = it["jaw_axis_2d"]
    n = math.hypot(ax_, ay) or 1.0
    ux, uy = ax_ / n, ay / n
    L = 0.10 * max(W, H)
    # jaw axis (the line along which the two jaws close)
    ax.plot([gx - ux * L, gx + ux * L], [gy - uy * L, gy + uy * L],
            "-", color=ACC, lw=1.6)
    # two jaw plates, perpendicular to the axis at each end
    px, py = -uy, ux
    for s in (-1, 1):
        jx, jy = gx + ux * L * s, gy + uy * L * s
        ax.plot([jx - px * L * 0.38, jx + px * L * 0.38],
                [jy - py * L * 0.38, jy + py * L * 0.38],
                "-", color=ACC, lw=2.6, solid_capstyle="butt")
    ax.plot([gx], [gy], "o", color="white", mec=ACC, mew=2.0, ms=5)
    ax.text(0.03 * W, 0.97 * H,
            f"{it.get('approach','?')}, wrist {it.get('wrist_deg','?')}$^\\circ$",
            color="white", fontsize=5.4, va="bottom",
            bbox=dict(fc=ACC, ec="none", pad=0.8))
    ax.set_title("dof: jaw axis + approach", fontsize=6.5, pad=2)


def panel_action_cot(ax, r):
    W, H = show(ax, r)
    it = items(r)[0]
    tp = it.get("target_point_2d")
    if isinstance(tp, (list, tuple)) and len(tp) == 2:
        x, y = P(tp, W, H)
        ax.plot([x], [y], marker="*", color=ACC, ms=13, mec="white", mew=0.7)
    steps = it.get("steps", [])[:3]
    lines = ["goal: " + textwrap.shorten(str(it.get("goal", "")), 28, placeholder="...")]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. " + textwrap.shorten(str(s), 30, placeholder="..."))
    ax.text(0.03 * W, 0.03 * H, "\n".join(lines), fontsize=5.0, va="top",
            color="white", linespacing=1.5,
            bbox=dict(fc="black", alpha=0.70, ec="none", pad=2.2))
    ax.set_title(f"action plan: {len(it.get('steps',[]))}-step plan $+$ target",
                 fontsize=6.5, pad=2)


def rank_mask(r) -> float:
    """Prefer masks whose polygon is genuinely NON-rectangular and multi-vertex,
    since a rectangle-looking mask fails to show what the form contributes."""
    it = items(r)[0] if items(r) else {}
    poly = it.get("polygon")
    if not (isinstance(poly, list) and len(poly) >= 6):
        return -1.0
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    hull = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if hull <= 0:
        return -1.0
    n = len(poly)
    sh = abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                 for i in range(n))) / 2.0
    fill = sh / hull                      # 1.0 == rectangle
    area = hull / 1e6                     # want a reasonably large object
    if area < 0.04:
        return -1.0
    return (1.0 - fill) * 2.0 + min(n, 14) / 14.0


def rank_detection(r) -> float:
    """Prefer many objects with distinct labels and modest overlap."""
    its = [i for i in items(r) if isinstance(i, dict)]
    labs = {str(i.get("label", "")).lower() for i in its}
    return len(its) + 1.5 * len(labs)


def rank_affordance(r) -> float:
    """Prefer cases where the sub-part is clearly smaller than the object, so the
    containment relation is visible."""
    it = items(r)[0] if items(r) else {}
    b = it.get("bbox_2d") or it.get("bbox")
    context = r.get("gsa_context") or r.get("g_context") or {}
    primary = context.get("primary") if isinstance(context.get("primary"), dict) else context
    g = primary.get("bbox")
    if not (isinstance(b, list) and isinstance(g, list) and len(b) == 4 and len(g) == 4):
        return -1.0
    ab = (b[2] - b[0]) * (b[3] - b[1])
    ag = (g[2] - g[0]) * (g[3] - g[1])
    if ag <= 0 or ab <= 0:
        return -1.0
    ratio = ab / ag
    if not (0.05 < ratio < 0.6):
        return -1.0
    return (0.6 - ratio) + min(ag / 1e6, 0.4)


def rank_depth(r) -> float:
    return float(len([i for i in items(r) if isinstance(i, dict)]))


def rank_dof(r) -> float:
    """Prefer a clearly non-axis-aligned jaw axis so the glyph is informative."""
    it = items(r)[0] if items(r) else {}
    a = it.get("jaw_axis_2d")
    if not (isinstance(a, list) and len(a) == 2):
        return -1.0
    ang = abs(math.degrees(math.atan2(a[1], a[0]))) % 90
    return min(ang, 90 - ang)


def rank_cot(r) -> float:
    it = items(r)[0] if items(r) else {}
    return len(it.get("steps", [])) + (1.0 if it.get("target_point_2d") else 0.0)


PANELS = [
    ("bbox", panel_bbox, None),
    ("detection", panel_detection, rank_detection),
    ("mask", panel_mask, rank_mask),
    ("point", panel_point, None),
    ("depth", panel_depth, rank_depth),
    ("affordance", panel_affordance, rank_affordance),
    ("dof", panel_dof, rank_dof),
    ("action_cot", panel_action_cot, rank_cot),
]


def main() -> int:
    global RAW_DIRS, EXCLUDED_IMAGE_IDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "paper/figures/fig_showcase.pdf"))
    ap.add_argument("--raw-dir", action="append", default=[],
                    help="raw flywheel directory; repeatable")
    ap.add_argument("--source-manifest", default="")
    ap.add_argument("--exclude-source-bucket", action="append", default=[])
    ap.add_argument("--selection-out", default="",
                    help="JSON manifest of the eight selected records")
    ap.add_argument("--selection-in", default="",
                    help="freeze the rendered records to an existing selection manifest")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    random.seed(args.seed)
    RAW_DIRS = [Path(d) for d in args.raw_dir]
    excluded_buckets = set(args.exclude_source_bucket)
    if args.source_manifest:
        for line in Path(args.source_manifest).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_bucket") in excluded_buckets:
                EXCLUDED_IMAGE_IDS.add(row.get("upload_sha256", ""))

    frozen_selection = {}
    if args.selection_in:
        rows = json.loads(Path(args.selection_in).read_text())
        if not isinstance(rows, list):
            raise ValueError("selection input must be a JSON list")
        frozen_selection = {
            row.get("form"): row.get("image_sha256")
            for row in rows
            if isinstance(row, dict)
        }
        expected_forms = {form for form, _, _ in PANELS}
        if set(frozen_selection) != expected_forms or any(
            not isinstance(value, str) or len(value) != 64
            for value in frozen_selection.values()
        ):
            raise ValueError("selection input must bind exactly one image hash per form")

    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.05))
    selected = []
    for ax, (form, fn, ranker) in zip(axes.ravel(), PANELS):
        if frozen_selection:
            recs = [load_exact(form, frozen_selection[form])]
        else:
            recs = load(form, want=120 if ranker else 40)
        if ranker and not frozen_selection:
            scored = [(ranker(r), r) for r in recs]
            scored = [(s, r) for s, r in scored if s > 0]
            scored.sort(key=lambda t: -t[0])
            recs = [r for _, r in scored] or recs
        drawn = False
        for r in recs:
            try:
                fn(ax, r)
                with Image.open(r["image_path"]) as source_image:
                    source_width, source_height = source_image.size
                selected.append({
                    "form": form,
                    "image_path": r.get("image_path"),
                    "image_sha256": r.get("image_sha256"),
                    "image_width": source_width,
                    "image_height": source_height,
                    "scene": r.get("scene"),
                    "payload": r.get("payload"),
                })
                drawn = True
                break
            except Exception:
                ax.clear()
        if not drawn:
            ax.axis("off")
            ax.set_title(f"{form}: no renderable sample", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.35)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    stem = Path(args.out).with_suffix("")
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    selection_out = Path(args.selection_out) if args.selection_out else stem.with_name(
        stem.name + "_selection.json")
    selection_out.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
    print("wrote", args.out)
    print("wrote", stem.with_suffix(".svg"))
    print("wrote", stem.with_suffix(".png"))
    print("wrote", selection_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
