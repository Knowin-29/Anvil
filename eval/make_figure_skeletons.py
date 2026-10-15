#!/usr/bin/env python3
"""Build the EDITABLE PowerPoint skeleton for every hand-drawn GSA-Forge figure.

One PPTX, one slide per figure, every element a real editable shape (no rasters):

  1  fig1_teaser            the measured capability gap + the four-stage loop
  2  fig2_framework         sources -> cascade -> verify -> GSA-Bench -> train -> reflow
  3  fig_signal_three_uses  Verified@k as filter AND reward AND teacher-assignment
  4  fig_multiteacher_opd   why three teachers, why one student, why on-policy

Two-file discipline (do not break it):

  docs/figures_editable_generated.pptx   <- THIS script writes it. Overwritten freely.
  docs/figures_editable.pptx             <- the human-edited master. Never touched here.

`scripts/export_figures.sh` exports from the MASTER, so regenerating the skeleton can
never clobber manual layout work. See docs/figures_workflow.md.

Every number comes from eval/figure_data.json, which carries a `_src` provenance field
per block. Change a number there, not here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_JSON = HERE / "figure_data.json"

SLIDE_W, SLIDE_H = 10.0, 5.63          # inches, 16:9
FONT = "Times New Roman"

# ICLR's text block is 5.5in (measured: \textwidth = 397.485pt). A figure exported at
# ~9.4in of ink and placed at \linewidth is therefore scaled by ~0.59, and EVERY font
# shrinks with it. Only the ratio size/ink_width matters, so the floor below is what
# keeps the smallest text readable in print; eval/check_figure_layout.py reports the
# effective printed size for every run so this is measured, not assumed.
ICLR_TEXTWIDTH_IN = 5.5
MIN_PRINT_PT = 5.0          # below this, detail text stops being legible on paper
# 5.5in / 9.4in of ink = 0.585, so authored sizes below ~8.6pt print under the floor.
# Text is kept short enough that this floor does not force wrapping.
SIZE_FLOOR = 8.6            # smallest size any run may be authored at

# Palette chosen so the four accent colours keep distinct greyscale luminance, and so
# every colour-coded group ALSO carries a text label or a dash pattern. Colour is
# redundant encoding, never the only channel.
INK = RGBColor(0x1C, 0x22, 0x2E)       # near-black text
MUT = RGBColor(0x64, 0x6C, 0x7A)       # muted caption text
BOX = RGBColor(0xF2, 0xF4, 0xF8)       # light panel fill
TRK = RGBColor(0xE4, 0xE8, 0xEF)       # bar-chart track
EDGE = RGBColor(0xBF, 0xC6, 0xD2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x2F, 0x5F, 0x96)      # generation stages
ORNG = RGBColor(0xC9, 0x7B, 0x0A)      # verification stages
GRN = RGBColor(0x46, 0x7F, 0x37)       # training stages / saturated capability
RED = RGBColor(0x8E, 0x24, 0x1C)       # failure + reflow (always ALSO dashed)


def IN(v: float) -> Emu:
    return Emu(int(round(v * 914400)))


def _runs(tf, text, *, size, bold, color, align, italic=False, sub_size=None,
          sub_color=None):
    """Fill a text frame line-by-line; line 1 is the heading, the rest are detail."""
    tf.word_wrap = True
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = ln
        raw = size if i == 0 else (sub_size or max(size - 1.5, SIZE_FLOOR))
        r.font.size = Pt(max(raw, SIZE_FLOOR))
        r.font.bold = bold if i == 0 else False
        r.font.italic = italic
        r.font.color.rgb = color if i == 0 else (sub_color or MUT)
        r.font.name = FONT


def box(slide, x, y, w, h, text="", *, fill=BOX, line=EDGE, size=10, bold=False,
        color=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, align=PP_ALIGN.CENTER,
        line_w=1.0, sub_size=None, sub_color=None):
    s = slide.shapes.add_shape(shape, IN(x), IN(y), IN(w), IN(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    tf = s.text_frame
    tf.margin_left = tf.margin_right = IN(0.055)
    tf.margin_top = tf.margin_bottom = IN(0.028)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _runs(tf, text, size=size, bold=bold, color=color, align=align,
          sub_size=sub_size, sub_color=sub_color)
    return s


def label(slide, x, y, w, text, *, size=10, bold=False, color=INK,
          align=PP_ALIGN.LEFT, italic=False, h=None):
    """Free text. Unlike box(), every line keeps the SAME size/colour/weight.

    Height defaults to the actual line count so eval/check_figure_layout.py sees the
    real footprint instead of a fixed 0.3in stub that fakes overlaps.
    """
    if h is None:
        h = len(text.split("\n")) * size * 1.24 / 72.0
    tb = slide.shapes.add_textbox(IN(x), IN(y), IN(w), IN(h))
    tb.text_frame.margin_left = tb.text_frame.margin_right = 0
    tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    _runs(tb.text_frame, text, size=size, bold=bold, color=color, align=align,
          italic=italic, sub_size=size, sub_color=color)
    return tb


def legend(slide, x, y, entries, *, size=7.5, gap=0.11, pitch=0.052):
    """Swatch + text pairs laid out left to right, sized so they cannot collide."""
    cx = x
    for text, col in entries:
        box(slide, cx, y + 0.045, 0.12, 0.10, fill=col, line=None,
            shape=MSO_SHAPE.RECTANGLE)
        label(slide, cx + 0.17, y, len(text) * pitch + 0.12, text, size=size,
              color=MUT)
        cx += 0.17 + len(text) * pitch + gap
    return cx - x


def arrow(slide, x1, y1, x2, y2, *, color=MUT, width=1.5, dashed=False, head=True):
    """Straight connector with a real arrowhead on the far end."""
    c = slide.shapes.add_connector(2, IN(x1), IN(y1), IN(x2), IN(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dashed:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(parse_xml(
            '<a:tailEnd %s type="triangle" w="med" len="med"/>' % nsdecls("a")))
    return c


def title(slide, text, subtitle):
    label(slide, 0.32, 0.15, 9.4, text, size=15, bold=True)
    label(slide, 0.32, 0.49, 9.4, subtitle, size=8.5, color=MUT, italic=True)


def bar(slide, x, y, w, h, frac, col):
    """Track + filled bar. Length is the primary channel so greyscale still reads."""
    box(slide, x, y, w, h, fill=TRK, line=None, shape=MSO_SHAPE.RECTANGLE)
    box(slide, x, y, max(w * frac, 0.10), h, fill=col, line=None,
        shape=MSO_SHAPE.RECTANGLE)


def load_data(path: Path | None = None) -> dict:
    d = json.loads((path or DATA_JSON).read_text(encoding="utf-8"))
    t = d["teachers"]
    got = t["G"]["rows"] + t["S"]["rows"] + t["A"]["rows"]
    if got != t["total_check"]:
        raise SystemExit(
            f"figure_data.json inconsistent: G+S+A={got} != train={t['total_check']}")
    return d


def _k(n: int) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------- slide 1
def slide_teaser(prs, D):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Figure 1   Grounding is already solved; what composes over it is not.",
          "EDITABLE SKELETON  \u00b7  measured per-dimension probe of the base VLM; "
          "every number traces to eval/figure_data.json")

    label(s, 0.32, 0.92, 4.6, "(a) Measured capability profile", size=11, bold=True)
    cmap = {"saturated": GRN, "headroom": ORNG, "gap": RED}
    y = 1.30
    for r in D["base_profile"]["rows"]:
        label(s, 0.32, y - 0.015, 2.30,
              f"{r['layer']}   {r['name']}", size=8.5)
        bar(s, 2.72, y, 1.95, 0.225, r["value"] / 100.0, cmap[r["status"]])
        label(s, 4.74, y - 0.015, 0.66, f"~{r['value']:.0f}%", size=8.5, bold=True)
        y += 0.345
    legend(s, 0.32, y + 0.06,
           [("saturated", GRN), ("partial headroom", ORNG), ("collapsed", RED)])

    label(s, 0.32, y + 0.44, 4.95,
          "Grounding is near-saturated, so more grounding data buys nothing.\n"
          "Every capability that must reason OVER grounded objects collapses,\n"
          "and open corpora supply almost no action-level supervision \u2014 the\n"
          "gap cannot be closed by collecting more of the same.",
          size=8.5, color=MUT)
    label(s, 5.55, 0.92, 4.15, "(b) The four-stage loop that closes it",
          size=11, bold=True)
    v = D["verified_at_k"]
    xa, xb, r1, r2, bw, bh = 5.55, 7.75, 1.28, 2.60, 1.85, 0.95
    stages = [
        (xa, r1, BLUE, "1  Open images",
         "no human labels needed\n"
         f"{_k(D['corpus']['images']['embodied_index'])} embodied frames"),
        (xb, r1, BLUE, "2  Cascaded generation",
         f"G \u2192 S \u2192 A, {D['bench']['n_forms']} forms\n"
         "grounding is the anchor"),
        (xb, r2, ORNG, "3  Verify",
         "form contracts + anchor\ncross-check, then Verified@k"),
        (xa, r2, ORNG, "4  Verified@k survival",
         f"{v['survival_pct']}% of k=1 survive k=2\n"
         f"({_k(v['k1_records'])} \u2192 {_k(v['k2_records'])})"),
    ]
    for x, yy, col, head, sub in stages:
        box(s, x, yy, bw, bh, f"{head}\n{sub}", fill=WHITE, line=col, size=9.5,
            bold=True, line_w=1.25, sub_size=7.5)
    my1, my2 = r1 + bh / 2, r2 + bh / 2
    arrow(s, xa + bw + 0.02, my1, xb - 0.02, my1)                  # 1 -> 2
    arrow(s, xb + bw / 2, r1 + bh + 0.02, xb + bw / 2, r2 - 0.02)  # 2 -> 3
    arrow(s, xb - 0.02, my2, xa + bw + 0.02, my2)                  # 3 -> 4
    arrow(s, xa + bw / 2, r2 - 0.02, xa + bw / 2, r1 + bh + 0.02)  # 4 -> 1

    ty = r2 + bh + 0.30
    box(s, xa, ty, 4.05, 0.62,
        "Train:  SFT \u2192 RL (same verifier as reward) \u2192 multi-teacher OPD\n"
        "the student's contract failures select what to regenerate next round",
        fill=BOX, line=GRN, size=9, bold=True, line_w=1.25, sub_size=7.5)
    arrow(s, xa + 2.90, r2 + bh + 0.02, xa + 2.90, ty - 0.02, color=GRN)
    arrow(s, xa + 0.85, ty - 0.02, xa + 0.85, r2 + bh + 0.02, color=GRN, dashed=True)
    label(s, xa + 0.95, ty - 0.29, 1.9, "failure reflow", size=7, color=GRN,
          italic=True)
    label(s, xa, ty + 0.70, 4.15,
          "One quantity, three jobs (Fig. 3): survival under independent\n"
          "re-verification is filter, reward and teacher-assignment at once.",
          size=8.5, color=MUT)
    return s


# ---------------------------------------------------------------- slide 2
def slide_framework(prs, D):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Figure 2   GSA-Forge architecture: one verifier closes the loop.",
          "EDITABLE SKELETON  \u00b7  five stages left to right, failure reflow along "
          "the bottom; counts are the frozen corpus numbers")

    c, b, t, o = D["corpus"], D["bench"], D["teachers"], D["opd"]
    v, lk = D["verified_at_k"], D["leakage"]
    cols = [
        (0.32, "1  Sources", BLUE, [
            (f"{c['n_sources']} open sources", f"{_k(c['open_source_records'])} records"),
            (f"{_k(c['images']['embodied_index'])} images",
             f"{b['n_scenes']} scene buckets"),
            ("QA is sourced, not made",
             f"{_k(c['qa_records'])} QA ({c['qa_share_of_open_pct']}%)")]),
        (2.24, "2  Cascaded generation", BLUE, [
            ("G  anchor", "point, bbox,\ndetection, mask"),
            ("S  spatial", "depth ordering,\non the anchor"),
            ("A  action", "affordance, DoF,\naction CoT")]),
        (4.16, "3  Verification", ORNG, [
            (f"{len(D['contracts']['checks'])} contracts",
             "bbox_too_large,\nbad_jaw_axis, \u2026"),
            ("Anchor cross-check", "kill before the\ncascade spreads"),
            (f"Verified@k: {v['survival_pct']}% at k=2",
             f"{_k(v['k1_records'])} \u2192 {_k(v['k2_records'])}")]),
        (6.08, "4  GSA-Bench", GRN, [
            (f"train {_k(b['train']['records'])}",
             f"{_k(b['train']['images'])} images\n0 QA records"),
            (f"val {_k(b['val']['records'])}", "image-disjoint\nfrom train"),
            (f"test {_k(b['test']['records'])}",
             "HUMAN GT only,\nnever model output")]),
        (8.00, "5  Training", GRN, [
            ("SFT per dimension",
             f"G {_k(t['G']['rows'])} / S {_k(t['S']['rows'])}\nA {_k(t['A']['rows'])}"),
            ("RL: verifier IS reward",
             f"R = {v['reward_weight_contract']}\u00b7contract\n"
             f"+ {v['reward_weight_quality']}\u00b7k/k_max"),
            ("On-policy distillation", "3 teachers \u2192\n1 student")]),
    ]
    cw, hy, hh, bxh = 1.68, 0.90, 0.30, 0.86
    for x, head, col, items in cols:
        box(s, x, hy, cw, hh, head, fill=col, line=None, size=9.5, bold=True,
            color=WHITE)
        y = hy + hh + 0.12
        for main, sub in items:
            box(s, x, y, cw, bxh, f"{main}\n{sub}", fill=WHITE, line=col, size=8.8,
                sub_size=8.6)
            y += bxh + 0.11
        if x < 7.9:
            arrow(s, x + cw + 0.02, hy + hh + 1.58, x + cw + 0.22,
                  hy + hh + 1.58, width=1.9)
    ry = 4.24
    box(s, 2.24, ry, 5.44, 0.42,
        "Failure reflow: student outputs violating the SAME contracts select which "
        "(image, form) pairs the next round regenerates",
        fill=BOX, line=RED, size=8.8, bold=True, line_w=1.25)
    for seg in (((8.84, ry - 0.24), (8.84, ry + 0.21)),
                ((8.84, ry + 0.21), (7.72, ry + 0.21)),
                ((2.22, ry + 0.21), (1.16, ry + 0.21)),
                ((1.16, ry + 0.21), (1.16, ry - 0.24))):
        (x1, y1), (x2, y2) = seg
        arrow(s, x1, y1, x2, y2, color=RED, width=1.8, dashed=True)

    label(s, 0.32, 4.80, 9.4,
        "Every edge is auditable. The test split carries human GT and is content-hash "
        "deduplicated against the "
        f"{lk['benchmarks_checked']} public benchmarks "
        f"({lk['content_md5_collisions']} MD5 collisions from "
        f"{lk['basename_collisions']} coincidental\nbasenames); train/val/test image "
        f"overlap is {b['overlap']['train_val']}/{b['overlap']['train_test']}/"
        f"{b['overlap']['val_test']}; and the RL reward imports the production "
        "validators rather than a separate reward model, so\nfilter and reward cannot "
        f"drift. G is {o['g_share_of_train_pct']}% of train, which is why the teachers "
        "are split by dimension (Fig. 4).",
        size=8.6, color=MUT)
    return s


# ---------------------------------------------------------------- slide 3
def slide_signal_three_uses(prs, D):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Figure 3   One signal, three uses: Verified@k is filter, reward and "
             "teacher-assignment.",
          "EDITABLE SKELETON  \u00b7  the same scalar drives all three arms, which is "
          "what makes this a loop rather than three stages")

    v, tu = D["verified_at_k"], D["three_uses"]
    box(s, 2.62, 0.90, 4.76, 0.72,
        "Verified@k   \u00b7   survival under independent stochastic re-verification\n"
        f"measured non-degenerate: {v['survival_pct']}% of records accepted at k=1 "
        f"survive to k=2  ({_k(v['k1_records'])} \u2192 {_k(v['k2_records'])})",
        fill=BOX, line=ORNG, size=11, bold=True, line_w=1.6, sub_size=8)

    xs = [0.42, 3.62, 6.82]
    aw, ay, ah = 2.76, 2.16, 1.02
    arms = [
        ("1  Data filter", BLUE, "drop records failing the form\ncontract, or dying at k = 1"),
        ("2  Process reward", ORNG,
         f"R = {v['reward_weight_contract']}\u00b7R_contract "
         f"+ {v['reward_weight_quality']}\u00b7(k / k_max)\nshapes RL directly"),
        ("3  Teacher assignment", GRN, "low, high-variance survival marks\n"
                                       "the forms needing own capacity"),
    ]
    for x, (head, col, sub) in zip(xs, arms):
        box(s, x, ay, aw, ah, f"{head}\n{sub}", fill=WHITE, line=col, size=10,
            bold=True, line_w=1.4, sub_size=8)
        arrow(s, 5.00, 1.64, x + aw / 2, ay - 0.02, color=col, width=1.6)
    dp = D["discriminative_power"]
    label(s, 0.42, 3.38, 4.5,
          "Measured discriminative power of that reward", size=10, bold=True)
    gx, bx, bwid = 1.40, 3.02, 0.92
    label(s, gx, 3.66, 0.88, "well-formed", size=7, color=GRN, bold=True)
    label(s, bx, 3.66, 0.88, "degenerate", size=7, color=RED, bold=True)
    y = 3.88
    for p in dp["pairs"]:
        label(s, 0.42, y, 0.90, p["form"], size=8, h=0.135)
        bar(s, gx, y + 0.012, bwid, 0.165, p["good"], GRN)
        label(s, gx + bwid + 0.05, y, 0.50, f"{p['good']:.3f}", size=7.5, color=MUT,
             h=0.135)
        bar(s, bx, y + 0.012, bwid, 0.165, p["bad"], RED)
        label(s, bx + bwid + 0.05, y, 0.50, f"{p['bad']:.3f}", size=7.5, color=MUT,
             h=0.135)
        y += 0.27
    label(s, 0.42, y + 0.03, 4.55,
          f"illegal JSON scores {dp['invalid_json']:.3f}.  Bar length, not colour, "
          "carries the value.", size=7.5, color=MUT)
    box(s, 5.20, 3.38, 4.42, 0.66,
        "No-drift guarantee\n"
        "the reward imports the production validators.py, so\n"
        "the RL reward and the data filter cannot diverge in code",
        fill=BOX, line=GRN, size=9.2, bold=True, align=PP_ALIGN.LEFT, sub_size=8.6)
    box(s, 5.20, 4.12, 4.42, 0.66,
        "Why arm 2 is load-bearing\n"
        "OpenThoughts (ICLR'26 Oral): answer-filtering often fails\n"
        "to beat no filtering, so a filter-only verifier is a liability",
        fill=BOX, line=RED, size=9.2, bold=True, align=PP_ALIGN.LEFT, sub_size=8.6)
    label(s, 5.20, 4.88, 4.42,
          "Precedent: Math-Shepherd's process signal is literally\n"
          "survival-under-resampling.", size=8.6, color=MUT)
    return s


# ---------------------------------------------------------------- slide 4
def slide_multiteacher_opd(prs, D):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Figure 4   Multi-Teacher OPD is forced by the hierarchy, not chosen for "
             "novelty.",
          "EDITABLE SKELETON  \u00b7  incompatible optimisation pressure forces the "
          "split; single-model inference forces the recombination")

    t, o = D["teachers"], D["opd"]
    label(s, 0.32, 0.86, 4.9,
          "(a) Why split: the dimensions pull in different directions", size=10.5,
          bold=True, h=0.20)
    label(s, 0.32, 1.12, 5.0,
          f"G \u2192 S \u2192 A is a dependency chain: S is ill-posed until its "
          f"arguments are grounded, A until the geometry is established.",
          size=7.5, color=MUT)

    rows = [("G", t["G"]["rows"], GRN,
             "near-saturated: small gradients, mostly output format"),
            ("S", t["S"]["rows"], ORNG, "partial headroom"),
            ("A", t["A"]["rows"], RED,
             "far from saturated: needs large updates")]
    y = 1.46
    for name, n, col, why in rows:
        frac = n / t["total_check"]
        label(s, 0.32, y, 0.26, name, size=11, bold=True, color=col, h=0.20)
        bar(s, 0.62, y + 0.025, 1.55, 0.19, frac, col)
        label(s, 2.24, y, 0.70, _k(n), size=8, bold=True, align=PP_ALIGN.RIGHT,
              h=0.16)
        label(s, 3.00, y, 0.50, f"{100 * frac:.1f}%", size=8, color=MUT, h=0.16)
        label(s, 3.58, y - 0.02, 1.72, why, size=6.8, color=MUT, h=0.30)
        y += 0.50
    label(s, 0.32, y + 0.04, 5.05,
          f"G: {', '.join(t['G']['forms'])}     "
          f"S: {', '.join(t['S']['forms'])}     "
          f"A: {', '.join(t['A']['forms'])}",
          size=7, color=MUT)
    box(s, 0.32, y + 0.34, 5.05, 0.58,
        "Joint training dilutes the A signal\n"
        f"a {o['g_share_of_train_pct']}%-share G bucket with little left to learn "
        "averages away the updates A still needs; per-dimension teachers let each "
        "layer take its own step size.  Falsifiable via E4.",
        fill=BOX, line=RED, size=8.5, bold=True, align=PP_ALIGN.LEFT, sub_size=7.2)
    label(s, 5.62, 0.88, 4.05,
          "(b) Why recombine: inference must be one model", size=10.5, bold=True)
    tw, th, ty = 1.06, 0.62, 1.30
    for i, (name, col) in enumerate((("G", GRN), ("S", ORNG), ("A", RED))):
        tx = 5.62 + i * 1.20
        box(s, tx, ty, tw, th, f"Teacher {name}\n27B, own step size", fill=WHITE,
            line=col, size=9, bold=True, sub_size=6.5)
        arrow(s, tx + tw / 2, ty + th + 0.02, 7.62, 2.28, color=col, width=1.4)

    box(s, 6.24, 2.30, 2.78, 0.52,
        "On-policy distillation (GKD)\nstudent samples; teachers correct in place",
        fill=BOX, line=BLUE, size=9, bold=True, line_w=1.4, sub_size=7)
    arrow(s, 7.62, 2.84, 7.62, 3.06, color=BLUE, width=1.6)
    box(s, 6.24, 3.08, 2.78, 0.50,
        f"One student  \u00b7  all {D['bench']['n_forms']} forms\nwe cannot ship three "
        "models",
        fill=WHITE, line=BLUE, size=9, bold=True, line_w=1.4, sub_size=7)
    box(s, 5.62, 3.70, 4.05, 0.66,
        "Why on-policy specifically\n"
        "offline KD trains on inputs the student never produces,\n"
        "but a student emitting a rectangular mask must be\n"
        "corrected ON that rectangle.",
        fill=BOX, line=ORNG, size=9, bold=True, align=PP_ALIGN.LEFT, sub_size=8.6)
    box(s, 5.62, 4.44, 4.05, 0.66,
        "Checkable by the same verifier\n"
        "degenerate-output rates (" + ", ".join(o["degenerate_checks"][:2]) + ",\n"
        f"{o['degenerate_checks'][2]}) use the SAME contracts\n"
        "that filtered the data.  Falsifiable via E5 / E5b.",
        fill=BOX, line=GRN, size=9, bold=True, align=PP_ALIGN.LEFT, sub_size=8.6)
    label(s, 0.32, 5.20, 9.35,
          "Distillation here is not post-hoc compression: it is the only route from "
          "per-dimension specialists to one deployable policy, which is why OPD belongs "
          "inside the loop rather than in a serving-cost appendix.",
          size=8.6, color=MUT)
    return s


# Slide order defines the export order; export_figures.sh reads these stems.
SLIDES = [
    ("fig1_teaser", slide_teaser),
    ("fig2_framework", slide_framework),
    ("fig_signal_three_uses", slide_signal_three_uses),
    ("fig_multiteacher_opd", slide_multiteacher_opd),
]


def build(D) -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = IN(SLIDE_W), IN(SLIDE_H)
    for _, fn in SLIDES:
        fn(prs, D)
    return prs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "docs" / "figures_editable_generated.pptx"),
                    help="skeleton path; NEVER point this at figures_editable.pptx")
    ap.add_argument("--data", default=str(DATA_JSON))
    ap.add_argument("--list", action="store_true", help="print slide order and exit")
    a = ap.parse_args()

    if a.list:
        for i, (stem, _) in enumerate(SLIDES, 1):
            print(f"{i}\t{stem}")
        return 0

    out = Path(a.out)
    if out.name == "figures_editable.pptx":
        raise SystemExit(
            "refusing to write figures_editable.pptx: that is the hand-edited master.\n"
            "Write the skeleton to figures_editable_generated.pptx instead.")

    D = load_data(Path(a.data))
    prs = build(D)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    n = len(prs.slides._sldIdLst)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {n} slides)")
    for i, (stem, _) in enumerate(SLIDES, 1):
        print(f"  slide {i} -> {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
