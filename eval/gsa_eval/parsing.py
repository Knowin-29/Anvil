"""Response parsing shared by every benchmark scorer.

The model under test is a thinking model: it emits a <think> span before the answer,
and that span routinely contains decoy letters and coordinates. Every extractor here
therefore strips reasoning first and reads the answer from the tail of the response.
"""
from __future__ import annotations

import ast
import json
import re

_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|python)?\s*(.*?)\s*```", re.DOTALL)
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_PAIR = re.compile(rf"[\(\[]\s*({_NUM})\s*,\s*({_NUM})\s*[\)\]]")

# Keys that carry a single (x, y) location, most-canonical first. The first four are
# the generic ones every VLM uses. The rest are the GSA-Bench corpus's OWN answer
# schema, which is why omitting them mis-scored our own model on our own format:
# a `dof` answer names its point `grasp_point_2d`, an `affordance` answer
# `contact_point_2d`, an `action_cot` answer `target_point_2d` (measured over
# 40,000 training rows: 5,075 / 4,802 / 398 occurrences, and `point_2d` appears
# only in the `point` form). Recognising them is not leniency -- when the prompt is
# "Point to the grasp that would accomplish X", `grasp_point_2d` IS the answer.
POINT_KEYS = ("point_2d", "point", "coordinates", "coord",
              "grasp_point_2d", "contact_point_2d", "target_point_2d",
              "coordinate_2d", "coordinate")

# Same idea for boxes: the corpus writes `bbox_2d`, with `box_2d` as a rare variant.
BOX_KEYS = ("bbox_2d", "bbox", "box", "bounding_box", "box_2d")


def strip_reasoning(text: str) -> str:
    """Return the answer span, dropping any <think> reasoning.

    Handles the truncated case too: if generation stopped inside <think> there is no
    closing tag, and keeping the fragment would let decoy coordinates leak into
    scoring, so the answer is treated as empty instead.
    """
    if not text:
        return ""
    text = _THINK_BLOCK.sub(" ", text)
    m = None
    for m in _THINK_CLOSE.finditer(text):
        pass
    if m is not None:
        return text[m.end():].strip()
    if _THINK_OPEN.search(text):
        return ""
    return text.strip()


def _candidate_blobs(text: str):
    for m in _FENCE.finditer(text):
        yield m.group(1)
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        while start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        yield text[start:i + 1]
                        break
            start = text.find(opener, start + 1)
    yield text


def parse_json_items(text: str) -> list:
    """Extract a list of dicts from a response, tolerating fences and stray prose."""
    for blob in _candidate_blobs(strip_reasoning(text)):
        blob = blob.strip()
        if not blob:
            continue
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(blob)
            except Exception:
                continue
            if isinstance(v, dict):
                return [v]
            if isinstance(v, list):
                return v
    return []


def parse_choice(text: str, n_choices: int) -> str | None:
    """Extract a multiple-choice letter, reading the answer span right-to-left.

    Accepts '(B)', 'B.', 'Answer: B' and a bare trailing 'B'. Returns None rather
    than guessing when nothing matches, so parse failures stay visible.

    Audit defect F5: the last, bare-letter pattern used to run case-insensitively, so an
    English article in 'It is a cup on the table.' matched as choice A. CV-Bench is
    1850/2638 two-choice rows where the letter class is [AB] and 'a' falls inside it, so
    the misfire biases towards A -- and the two models differ in how often they answer in
    prose, which would make that bias asymmetric. The option letters are upper-case in
    every prompt, so the fallback is now case-sensitive; the three higher-priority
    patterns keep their tolerance for lower case. If the fallback sees several DIFFERENT
    letters it reports a parse failure instead of taking the last one, matching this
    module's existing rule of not guessing.
    """
    answer = strip_reasoning(text)
    if not answer:
        return None
    letters = "".join(chr(ord("A") + i) for i in range(max(1, min(n_choices, 26))))
    patterns = (
        rf"\(\s*([{letters}])\s*\)",
        rf"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([{letters}])\b",
        rf"\b([{letters}])\s*[.):]",
    )
    for pat in patterns:
        hits = re.findall(pat, answer, re.IGNORECASE)
        if hits:
            return hits[-1].upper()
    hits = re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", answer)
    if hits and len({h.upper() for h in hits}) == 1:
        return hits[-1].upper()
    return None


def parse_points(text: str) -> list[tuple[float, float]]:
    """Extract (x, y) pairs from the answer span.

    Reads plain tuple lists, JSON dicts carrying any of `POINT_KEYS`, and Qwen-style
    <point>x y</point> tags, since responses mix all three.
    """
    answer = strip_reasoning(text)
    if not answer:
        return []
    pts: list[tuple[float, float]] = []
    for item in parse_json_items(answer):
        if isinstance(item, dict):
            for key in POINT_KEYS:
                v = item.get(key)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    try:
                        pts.append((float(v[0]), float(v[1])))
                    except (TypeError, ValueError):
                        pass
                    break
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                pass
    if pts:
        return pts
    for m in re.finditer(rf"<point>\s*({_NUM})[\s,]+({_NUM})\s*</point>", answer):
        pts.append((float(m.group(1)), float(m.group(2))))
    if pts:
        return pts
    return [(float(a), float(b)) for a, b in _PAIR.findall(answer)]


def parse_boxes(text: str) -> list[list[float]]:
    """Extract [x1, y1, x2, y2] boxes from the answer span."""
    answer = strip_reasoning(text)
    if not answer:
        return []
    boxes: list[list[float]] = []
    for item in parse_json_items(answer):
        cand = None
        if isinstance(item, dict):
            for key in BOX_KEYS:
                v = item.get(key)
                if isinstance(v, (list, tuple)) and len(v) == 4:
                    cand = v
                    break
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            cand = item
        if cand is not None:
            try:
                boxes.append([float(x) for x in cand])
            except (TypeError, ValueError):
                pass
    if boxes:
        return boxes
    nums = re.findall(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]",
                      answer)
    return [[float(x) for x in g] for g in nums]
