"""Coordinate-space normalisation.

Motivating bug (docs/experiment_findings.md section 4.2): gsa_bench_test scored
IoU@0.5 = 0.00 for both base and SFT, which reads as "no spatial ability" but is
actually a unit mismatch. GSA-Bench ground truth is thousandth-scale (0-1000) while
the 1,463 test prompts carried no scale hint at all (measured: 1463/1463 without one),
so the model answers in pixels and every IoU collapses to ~0.

Fix: score in a single normalised [0,1] space, converting each side out of its own
unit. Ground truth uses its documented convention (`declared="thousandth"`, confirmed
by measurement: 120/120 sampled GT values fall under 1000 while 85/120 overflow their
image bounds). Predictions can be auto-classified, which is necessarily heuristic, so
the scorer reports the declared-convention number as the headline and the
auto-classified one as a diagnostic instead of silently picking whichever looks better.
"""
from __future__ import annotations

# A prediction in [0,1] units is indistinguishable from a degenerate pixel answer
# clustered in the top-left corner. Values at or below this bound are read as
# already-normalised; the risk is confined to the first ~1.5% of the frame.
_UNIT_MAX = 1.5


def infer_scale(xs, ys, width: int, height: int, declared: str = "auto") -> str:
    """Classify a coordinate list's unit: 'unit', 'thousandth' or 'pixel'.

    Axis-aware on purpose: a box like [140, 43, 344, 662] on an 800x600 image is
    thousandth-scale because y=662 overflows the 600px height, even though 662 stays
    under the 800px width. Comparing against max(width, height) missed exactly this
    case. When nothing overflows, the value is ambiguous and 'pixel' is returned, so
    callers that know the convention should pass it in explicitly.
    """
    if declared != "auto":
        return declared
    xs = [abs(float(v)) for v in xs]
    ys = [abs(float(v)) for v in ys]
    allv = xs + ys
    if not allv:
        return "unit"
    if max(allv) <= _UNIT_MAX:
        return "unit"
    over_x = width and xs and max(xs) > width
    over_y = height and ys and max(ys) > height
    if over_x or over_y:
        return "thousandth"
    return "pixel"


AMBIGUOUS = "ambiguous"


def classify_scale(xs, ys, width: int, height: int) -> str:
    """Like `infer_scale`, but says AMBIGUOUS instead of guessing 'pixel'.

    Measured need (docs/audit_gsa_test_report.md defect 2): on 640x480 images a
    thousandth-scale value below 480 is also a legal pixel value, so 97/378 bbox and
    387/1085 point rows of the ground truth itself -- known to be thousandth -- were
    classified 'pixel'. Feeding a perfect thousandth answer through that path scored
    67.26 instead of 100, while a perfect pixel answer scored 100: the guess is biased
    in one direction. Callers resolve AMBIGUOUS from the model's own unambiguous rows.
    """
    xs = [abs(float(v)) for v in xs]
    ys = [abs(float(v)) for v in ys]
    allv = xs + ys
    if not allv:
        return "unit"
    if max(allv) <= _UNIT_MAX:
        return "unit"
    if max(allv) > 1000.0:
        return "pixel"
    if (width and xs and max(xs) > width) or (height and ys and max(ys) > height):
        return "thousandth"
    return AMBIGUOUS


def resolve_corpus_scale(labels) -> str:
    """Pick the convention a model actually uses, from its unambiguous rows only.

    Decided by the model's own output distribution, never by which reading scores
    higher: base answers 46 unit / 8 thousandth and SFT 2 unit / 40 thousandth on
    where2place, because SFT was trained on thousandth JSON while the prompt asks for
    [0,1]. Ties and an all-ambiguous corpus fall back to 'pixel', the old behaviour.
    """
    from collections import Counter
    c = Counter(l for l in labels if l != AMBIGUOUS)
    if not c:
        return "pixel"
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "pixel"
    return top[0][0]


def _apply(xs, ys, width: int, height: int, scale: str):
    if scale == AMBIGUOUS:
        scale = "pixel"
    if scale == "unit":
        return list(xs), list(ys)
    if scale == "thousandth":
        return [v / 1000.0 for v in xs], [v / 1000.0 for v in ys]
    fx = width or 1
    fy = height or 1
    return [v / fx for v in xs], [v / fy for v in ys]


def to_unit_points(points, width: int, height: int, declared: str = "auto"):
    """Map a point list into [0,1].

    The unit is decided for the list as a whole: an answer like [(12, 640)] is
    unambiguously pixel-scale, but 12 on its own would read as normalised.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if not pts:
        return []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    scale = infer_scale(xs, ys, width, height, declared)
    nx, ny = _apply(xs, ys, width, height, scale)
    return list(zip(nx, ny))


def to_unit_box(box, width: int, height: int, declared: str = "auto"):
    """Map [x1, y1, x2, y2] into [0,1], oriented so x1<=x2 and y1<=y2."""
    if box is None or len(box) != 4:
        return None
    b = [float(v) for v in box]
    xs, ys = [b[0], b[2]], [b[1], b[3]]
    scale = infer_scale(xs, ys, width, height, declared)
    nx, ny = _apply(xs, ys, width, height, scale)
    x1, x2 = sorted(nx)
    y1, y2 = sorted(ny)
    return [x1, y1, x2, y2]


def iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes in a shared coordinate space."""
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0
