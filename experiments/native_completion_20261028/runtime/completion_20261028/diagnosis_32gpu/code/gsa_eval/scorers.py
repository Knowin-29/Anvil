"""Per-benchmark scoring.

Every scorer returns (score, detail, n_parse_failures). Unparseable responses are
counted separately and scored 0 rather than dropped, so a run that mostly failed to
produce the requested format cannot masquerade as a high accuracy on a small subset.
"""
from __future__ import annotations

from collections import defaultdict

from .coords import (AMBIGUOUS, classify_scale, iou, resolve_corpus_scale,
                     to_unit_box, to_unit_points)
from .parsing import parse_boxes, parse_choice, parse_points


def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _group_table(per_group: dict[str, list[int]]) -> dict[str, dict]:
    return {k: {"n": len(v), "acc": _pct(sum(v), len(v))}
            for k, v in sorted(per_group.items())}


def score_multiple_choice(samples, responses) -> tuple[float, dict, int]:
    """Accuracy over letter answers (CV-Bench, EmbSpatial).

    Ground truth arrives as '(C)' in CV-Bench and as a letter in EmbSpatial, so it is
    normalised to a bare letter on both sides.
    """
    fails = 0
    per_group: dict[str, list[int]] = defaultdict(list)
    for s, text in zip(samples, responses):
        n_choices = len(s.meta.get("choices") or []) or 4
        pred = parse_choice(text, n_choices)
        gold = str(s.gt).strip().strip("()").upper()[:1]
        if pred is None:
            fails += 1
            per_group[s.group].append(0)
            continue
        per_group[s.group].append(int(pred == gold))
    flat = [v for vs in per_group.values() for v in vs]
    detail = {"per_group": _group_table(per_group),
              "n_correct": sum(flat)}
    return _pct(sum(flat), len(flat)), detail, fails


def _load_mask(path: str):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("L")) > 127


_UNIT_BOUND = 1.5


def _prompt_driven_scale(pts) -> str:
    """Read a point list under a prompt that explicitly asks for [0,1].

    Fixes audit defects F1 and F2. `infer_scale`/`classify_scale` admit a third 'pixel'
    reading, chosen whenever nothing overflows the image bounds -- so on a 640x480 image
    a thousandth-scale (568, 300) was silently divided by 640x480 and landed half a frame
    away. Measured effect: 2/56 base and 9/51 SFT where2place answers were misread, and
    20/103 vs 27/121 on refspatial_bench, i.e. the penalty was 14.1 pp heavier on the
    model that answers in thousandths. Since both benchmarks' prompts state "between 0
    and 1" in 100/100 and 277/277 rows and never mention 0-1000, a value above 1.5 can
    only be the model falling back to its training convention. 'pixel' is therefore not
    an admissible reading here.

    Both remaining readings are strict, so this narrows the interpretation rather than
    relaxing it: the [0,1] and thousandth readings differ only by a factor of 1000.
    """
    mx = max(max(abs(x), abs(y)) for x, y in pts)
    return "unit" if mx <= _UNIT_BOUND else "thousandth"


def score_point_in_mask(samples, responses, *, spec=None) -> tuple[float, dict, int]:
    """Where2Place / RefSpatial: fraction of predicted points inside the GT mask.

    Follows RoboPoint's protocol: a question's score is the share of its own predicted
    points that land in the mask, then averaged over questions, so emitting many
    scattered points is not rewarded.

    Scale handling depends on what the prompt declares. When the benchmark's prompts ask
    for [0,1] (both point benchmarks here do, measured on every row), each answer is read
    by magnitude alone -- see `_prompt_driven_scale`. Otherwise the previous corpus-vote
    heuristic is kept.
    """
    fails = 0
    per_group: dict[str, list[float]] = defaultdict(list)
    hit_any: dict[str, list[int]] = defaultdict(list)
    n_points: dict[str, list[int]] = defaultdict(list)
    parsed = [parse_points(t) for t in responses]
    prompt_scale = getattr(spec, "prompt_coord_scale", "") if spec else ""
    # Diagnostic only: what each answer WOULD have been called by the old auto path, so
    # the misread count stays visible instead of disappearing with the bug.
    labels = [classify_scale([p[0] for p in pts], [p[1] for p in pts],
                             s.width, s.height)
              for s, pts in zip(samples, parsed) if pts]
    corpus = resolve_corpus_scale(labels)
    n_ambiguous = sum(1 for l in labels if l == AMBIGUOUS)
    used: dict[str, int] = defaultdict(int)
    n_reinterpreted = 0
    for s, pts in zip(samples, parsed):
        if not pts:
            fails += 1
            per_group[s.group].append(0.0)
            hit_any[s.group].append(0)
            continue
        try:
            mask = _load_mask(s.gt)
        except Exception:
            fails += 1
            per_group[s.group].append(0.0)
            hit_any[s.group].append(0)
            continue
        mh, mw = mask.shape
        inside = 0
        own = classify_scale([p[0] for p in pts], [p[1] for p in pts],
                             s.width or mw, s.height or mh)
        if prompt_scale == "unit":
            declared = _prompt_driven_scale(pts)
        else:
            declared = corpus if own == AMBIGUOUS else own
        old = corpus if own == AMBIGUOUS else own
        if declared != old:
            n_reinterpreted += 1
        used[declared] += 1
        for x, y in to_unit_points(pts, s.width or mw, s.height or mh,
                                   declared=declared):
            px, py = int(round(x * (mw - 1))), int(round(y * (mh - 1)))
            if 0 <= px < mw and 0 <= py < mh and bool(mask[py, px]):
                inside += 1
        per_group[s.group].append(inside / len(pts))
        hit_any[s.group].append(int(inside > 0))
        n_points[s.group].append(len(pts))
    flat = [v for vs in per_group.values() for v in vs]
    all_np = sorted(v for vs in n_points.values() for v in vs)
    detail = {
        "per_group": {k: {"n": len(v), "mean_in_mask": _pct(sum(v), len(v))}
                      for k, v in sorted(per_group.items())},
        "any_point_hit": _pct(sum(v for vs in hit_any.values() for v in vs), len(flat)),
        # Emitting many points pushes the in-mask fraction towards the mask's area share,
        # a near-random baseline; recording the distribution keeps that visible so a
        # scatter-heavy model cannot be read as well localised. Measured: base answers a
        # median of 3 points but a mean of 39.8 and a max of 492, SFT a median of 1.
        "points_per_question": {
            "n": len(all_np),
            "median": all_np[len(all_np) // 2] if all_np else 0,
            "mean": round(sum(all_np) / len(all_np), 2) if all_np else 0.0,
            "max": max(all_np) if all_np else 0,
            "note": ("many points drive the in-mask fraction towards the mask area "
                     "share, which is close to random; compare against "
                     "points_per_question before reading a high score as localisation"),
        },
        "coord_scale": {
            "rule": ("prompt-driven: the prompt states [0,1], so >1.5 is read as the "
                     "model's thousandth training habit and 'pixel' is inadmissible"
                     if prompt_scale == "unit" else
                     "corpus vote over the model's own unambiguous rows"),
            "used": dict(sorted(used.items())),
            "n_reinterpreted_vs_auto": n_reinterpreted,
            "auto_would_have_said": {k: labels.count(k) for k in sorted(set(labels))},
            "n_ambiguous_under_auto": n_ambiguous,
            "corpus_vote_under_auto": corpus,
        },
        "referent_type": getattr(spec, "referent_type", "") if spec else "",
        "referent_by_group": (getattr(spec, "referent_by_group", {}) or {}) if spec else {},
    }
    if spec is not None and getattr(spec, "referent_by_group", None):
        # Split the headline by referent so an object-referring subset can be compared
        # against the training task without the vacancy rows averaged in.
        by_ref: dict[str, list[float]] = defaultdict(list)
        for k, v in per_group.items():
            by_ref[spec.referent_by_group.get(k, "unknown")].extend(v)
        detail["by_referent_type"] = {
            k: {"n": len(v), "mean_in_mask": _pct(sum(v), len(v))}
            for k, v in sorted(by_ref.items())}
    return _pct(sum(flat), len(flat)), detail, fails


def _resolve_pred_scale(samples, responses) -> dict:
    """Classify the coordinate convention a model uses, from its unambiguous rows."""
    labels: list[str] = []
    for s, text in zip(samples, responses):
        if s.meta.get("form") == "bbox":
            vals = [b for b in parse_boxes(text) if len(b) == 4]
            xs = [v for b in vals for v in (b[0], b[2])]
            ys = [v for b in vals for v in (b[1], b[3])]
        else:
            pts = parse_points(text)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
        if xs and ys:
            labels.append(classify_scale(xs, ys, s.width, s.height))
    corpus = resolve_corpus_scale(labels)
    return {"resolved_for_ambiguous": corpus,
            "n_ambiguous": sum(1 for l in labels if l == AMBIGUOUS),
            "observed": {k: labels.count(k) for k in sorted(set(labels))},
            "declared_by_prompt": "thousandth"}


def score_gsa_geometry(samples, responses) -> tuple[float, dict, int]:
    """GSA-Bench test: bbox IoU and point hit rate in a shared [0,1] space.

    Two defects from the findings report are fixed here. Both sides are converted out
    of their native units before comparison (the old scorer compared thousandth-scale
    GT against pixel-scale predictions, forcing IoU@0.5 to 0.00). And the headline
    score is an explicit sample-weighted combination instead of
    `bbox.iou@0.5 or point.hit@0.1`, whose falsey fallback silently reported the point
    hit rate as if it were a combined score.
    """
    fails = 0
    ious: list[float] = []
    hits: list[int] = []
    dists: list[float] = []
    # Diagnostic: same predictions read as thousandth-scale. The prompt now states the
    # convention, so a large gap versus the auto-classified reading means the model is
    # ignoring the stated scale rather than mislocating the object.
    hits_auto: list[int] = []
    ious_auto: list[float] = []
    n_multi = 0
    # Localisation and unit-convention compliance are different abilities, and reading
    # predictions on one fixed scale conflates them: the prompt declares 0-1000, so a
    # model trained on thousandth JSON is read correctly while a model answering in
    # pixels collapses to ~1.3 even when perfectly located (measured both directions
    # with GT-as-prediction). The headline therefore reads each model in the convention
    # its own unambiguous rows reveal -- symmetric across models, decided by output
    # distribution and never by which reading scores higher -- and the declared-scale
    # compliance rate is reported separately so ignoring the prompt stays visible.
    pred_scale = _resolve_pred_scale(samples, responses)
    for s, text in zip(samples, responses):
        gt = s.gt if isinstance(s.gt, list) else [s.gt]
        g = gt[0] if gt and isinstance(gt[0], dict) else {}
        form = s.meta.get("form")
        if form == "bbox":
            # GT convention is documented and measured, so it is declared, not guessed.
            gb = to_unit_box(g.get("bbox_2d") or g.get("bbox"), s.width, s.height,
                             declared="thousandth")
            pred = parse_boxes(text)
            if not pred or gb is None:
                fails += 1
                ious.append(0.0)
                ious_auto.append(0.0)
                continue
            if len(pred) > 1:
                n_multi += 1
            xs = [v for b in pred if len(b) == 4 for v in (b[0], b[2])]
            ys = [v for b in pred if len(b) == 4 for v in (b[1], b[3])]
            own = classify_scale(xs, ys, s.width, s.height)
            decl = (pred_scale["resolved_for_ambiguous"]
                    if own == AMBIGUOUS else own)
            # Best match over the predicted list, not pred[0]: SFT taught the model to
            # emit multi-target lists (measured 700/5000 = 14% of training answers),
            # and list order carries no meaning, so pred[0] scored an arbitrary box.
            # GT is single-target on this split (measured 0/1463 multi rows), so this
            # cannot inflate: it removes an ordering artefact, it does not relax.
            ious.append(max(iou(to_unit_box(p, s.width, s.height, declared=decl), gb)
                            for p in pred))
            ious_auto.append(max(iou(to_unit_box(p, s.width, s.height,
                                                 declared="thousandth"), gb)
                                 for p in pred))
        else:
            gp = g.get("point_2d") or g.get("point")
            pred = parse_points(text)
            if not pred or not isinstance(gp, (list, tuple)) or len(gp) != 2:
                fails += 1
                hits.append(0)
                hits_auto.append(0)
                dists.append(1.0)
                continue
            if len(pred) > 1:
                n_multi += 1
            gx, gy = to_unit_points([gp], s.width, s.height,
                                    declared="thousandth")[0]

            def _closest(decl):
                best = 1.0
                for px, py in to_unit_points(pred, s.width, s.height,
                                             declared=decl):
                    best = min(best, ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)
                return best

            own = classify_scale([p[0] for p in pred], [p[1] for p in pred],
                                 s.width, s.height)
            decl = (pred_scale["resolved_for_ambiguous"]
                    if own == AMBIGUOUS else own)
            d = _closest(decl)
            dists.append(min(d, 1.0))
            hits.append(int(d <= 0.1))
            hits_auto.append(int(_closest("thousandth") <= 0.1))
    detail: dict = {}
    parts: list[tuple[float, int]] = []
    if ious:
        detail["bbox"] = {
            "n": len(ious),
            "mean_iou": round(sum(ious) / len(ious), 4),
            "iou@0.5": _pct(sum(i >= .5 for i in ious), len(ious)),
            "iou@0.75": _pct(sum(i >= .75 for i in ious), len(ious)),
            "iou@0.5_pred_as_declared_thousandth": _pct(
                sum(i >= .5 for i in ious_auto), len(ious_auto))}
        parts.append((detail["bbox"]["iou@0.5"], len(ious)))
    if dists:
        detail["point"] = {
            "n": len(dists),
            "hit@0.1": _pct(sum(hits), len(hits)),
            "mean_dist": round(sum(dists) / len(dists), 4),
            "hit@0.1_pred_as_declared_thousandth": _pct(sum(hits_auto),
                                                        len(hits_auto))}
        parts.append((detail["point"]["hit@0.1"], len(dists)))
    total = sum(n for _, n in parts)
    score = round(sum(v * n for v, n in parts) / total, 2) if total else 0.0
    detail["n_multi_target_predictions"] = n_multi
    detail["coord_scale"] = pred_scale
    detail["score_definition"] = (
        "sample-weighted mean of bbox IoU@0.5 and point hit@0.1, both in normalised "
        "[0,1] coordinates; predictions read in the convention the model's own "
        "unambiguous answers reveal, best match over the predicted list against "
        "single-target GT. *_pred_as_declared_thousandth measures compliance with the "
        "prompt's stated 0-1000 scale and is a separate ability from localisation")
    return score, detail, fails


def score_point_in_box(samples, responses, *, spec=None) -> tuple[float, dict, int]:
    """VABench-Point: fraction of questions whose predicted point lands in the GT box.

    Ground truth is a single normalised [0,1] bounding box (``normalized_bbox``). A
    question scores 1 when any predicted point falls inside that box and 0 otherwise;
    an unparseable answer is a parse failure scored 0, matching the other scorers. The
    prompt states a [0,1] convention, so a value above 1.5 is read as the model's
    thousandth training habit rather than admitting an image-size-dependent pixel
    reading (same rule as ``score_point_in_mask``).
    """
    fails = 0
    per_group: dict[str, list[int]] = defaultdict(list)
    parsed = [parse_points(t) for t in responses]
    used: dict[str, int] = defaultdict(int)
    for s, pts in zip(samples, parsed):
        if not pts:
            fails += 1
            per_group[s.group].append(0)
            continue
        gb = s.gt if isinstance(s.gt, (list, tuple)) and len(s.gt) == 4 else None
        if gb is None:
            fails += 1
            per_group[s.group].append(0)
            continue
        declared = _prompt_driven_scale(pts)
        used[declared] += 1
        x0, y0, x1, y1 = (min(gb[0], gb[2]), min(gb[1], gb[3]),
                          max(gb[0], gb[2]), max(gb[1], gb[3]))
        hit = 0
        for x, y in to_unit_points(pts, s.width, s.height, declared=declared):
            if x0 <= x <= x1 and y0 <= y <= y1:
                hit = 1
                break
        per_group[s.group].append(hit)
    flat = [v for vs in per_group.values() for v in vs]
    detail = {
        "per_group": {k: {"n": len(v), "acc": _pct(sum(v), len(v))}
                      for k, v in sorted(per_group.items())},
        "n_correct": sum(flat),
        "coord_scale": {
            "rule": ("prompt-driven: the prompt states [0,1], so >1.5 is read as the "
                     "model's thousandth training habit and 'pixel' is inadmissible"),
            "used": dict(sorted(used.items())),
        },
    }
    return _pct(sum(flat), len(flat)), detail, fails
