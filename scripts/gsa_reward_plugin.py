"""GSA reward plugin for ms-swift GRPO.

The point of this file is theoretical, not just practical: the reward is computed by the
**same per-form contracts that filtered the training corpus**. That is what makes the
verifier a process reward rather than a one-off data filter, and it is what separates this
work from filter-only pipelines (which, per OpenThoughts, frequently fail to beat no
filtering at all).

Reward:
    R = 0.3 * R_contract + 0.7 * R_quality

  R_contract in {0,1}: does the completion satisfy the hard structural/geometric contract
    for its task form (valid JSON, coordinates in range, non-degenerate geometry,
    sub-part strictly inside its object, non-axis-aligned jaw axis, ...)?
  R_quality in [0,1]: graded credit for the properties a binary contract cannot express
    (outline non-rectangularity, label specificity, step substance).

Both terms are label-free: nothing here needs a human annotation or a separately trained
reward model.

Usage:
  swift rlhf --rlhf_type grpo \
    --external_plugins scripts/gsa_reward_plugin.py \
    --reward_funcs gsa_reward format
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import List

from swift.rewards import ORM, orms

# Reuse the production validators so the reward can never drift from the data filter.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel.contracts import CONTRACT_VERSION, FORM_SPECS  # noqa: E402
from trains.flywheel.validators import BACKGROUND, GENERIC  # noqa: E402
from trains.flywheel.validators import validate as _validate  # noqa: E402


# ---------------------------------------------------------------- parsing
def parse_payload(text: str):
    """Extract the JSON array/object a GSA answer should contain."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()
    # prefer the outermost array, since every form answers with a list
    start = text.find("[")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            depth += (ch == "[") - (ch == "]")
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    break
    try:
        return json.loads(text)
    except Exception:
        return None



# ---------------------------------------------------------------- graded quality
def _poly_fill_ratio(poly) -> float:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    hull = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if hull <= 0:
        return 1.0
    n = len(poly)
    sh = abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                 for i in range(n))) / 2.0
    return sh / hull


def quality(form: str, items) -> float:
    """Graded credit for what a binary contract cannot express."""
    if not isinstance(items, list) or not items:
        return 0.0
    first = items[0] if isinstance(items[0], dict) else {}
    lab = str(first.get("label") or first.get("object_class") or "").strip().lower()
    generic_penalty = 0.0 if (lab and lab not in GENERIC and lab not in BACKGROUND) else 0.5

    if form == "mask":
        poly = first.get("polygon")
        if not (isinstance(poly, list) and poly and isinstance(poly[0], (list, tuple))):
            return 0.0
        # reward a real outline: penalise both a rectangle (ratio->1) and a sliver
        r = _poly_fill_ratio(poly)
        shape = max(0.0, 1.0 - abs(r - 0.65) / 0.65)
        verts = min(len(poly), 16) / 16.0
        return max(0.0, (0.6 * shape + 0.4 * verts) - generic_penalty)

    if form == "detection":
        labs = [str(i.get("label", "")).strip().lower()
                for i in items if isinstance(i, dict)]
        if not labs:
            return 0.0
        distinct = len(set(labs)) / len(labs)          # monolabel enumerations score low
        clean = sum(1 for l in labs if l and l not in BACKGROUND and l not in GENERIC) / len(labs)
        count = min(len(labs), 5) / 5.0
        return max(0.0, 0.4 * distinct + 0.4 * clean + 0.2 * count)

    if form == "dof":
        ax = first.get("jaw_axis_2d")
        if not (isinstance(ax, (list, tuple)) and len(ax) == 2):
            return 0.0
        # axis-aligned defaults ([1,0]/[0,1]) carry no geometric information
        ang = abs(math.degrees(math.atan2(ax[1], ax[0]))) % 90.0
        diversity = min(ang, 90.0 - ang) / 45.0
        w = first.get("wrist_deg")
        wrist = 1.0 if isinstance(w, (int, float)) and -90 <= w <= 90 else 0.0
        return max(0.0, (0.7 * diversity + 0.3 * wrist) - generic_penalty)

    if form == "affordance":
        b = first.get("bbox_2d") or first.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            return 0.0
        area = (b[2] - b[0]) * (b[3] - b[1]) / 1e6
        # a contact sub-part should be small; a part equal to the object is uninformative
        part = max(0.0, 1.0 - abs(area - 0.06) / 0.30)
        verb = 1.0 if first.get("affordance") else 0.0
        return max(0.0, (0.6 * part + 0.4 * verb) - generic_penalty)

    if form == "action_cot":
        steps = first.get("steps") or []
        if not isinstance(steps, list) or not steps:
            return 0.0
        n = min(len(steps), 4) / 4.0
        substance = sum(1 for s in steps
                        if isinstance(s, str) and len(s.split()) >= 4) / max(len(steps), 1)
        pre = 1.0 if isinstance(first.get("preconditions"), list) else 0.0
        return 0.4 * n + 0.4 * substance + 0.2 * pre

    if form == "depth":
        ranks = [i.get("rank") for i in items if isinstance(i, dict)]
        ds = [i.get("relative_depth") for i in items if isinstance(i, dict)]
        if not ranks or len(set(r for r in ranks if isinstance(r, int))) != len(ranks):
            return 0.0
        spread = 0.0
        vals = [d for d in ds if isinstance(d, (int, float))]
        if len(vals) >= 2:
            spread = min(1.0, (max(vals) - min(vals)) / 0.5)
        return 0.5 * min(len(items), 4) / 4.0 + 0.5 * spread

    # bbox / point: contract already covers most of it; credit label specificity
    return max(0.0, 1.0 - generic_penalty)


class GSAReward(ORM):
    """Task-locked contract reward. Dataset metadata selects the validator."""

    W_CONTRACT = 0.3
    W_QUALITY = 0.7

    def __call__(
        self,
        completions: List[str],
        gsa_form=None,
        gsa_context=None,
        gsa_contract_version=None,
        **kwargs,
    ) -> List[float]:
        n = len(completions)
        columns = {
            "gsa_form": gsa_form,
            "gsa_context": gsa_context,
            "gsa_contract_version": gsa_contract_version,
        }
        for name, values in columns.items():
            if not isinstance(values, (list, tuple)) or len(values) != n:
                raise ValueError(
                    f"GSAReward requires a length-{n} dataset column '{name}'; "
                    "run scripts/validate_grpo_dataset.py before training"
                )

        out: List[float] = []
        for text, form, context, version in zip(
                completions, gsa_form, gsa_context, gsa_contract_version):
            if version != CONTRACT_VERSION or form not in FORM_SPECS:
                raise ValueError(
                    f"unsupported GSA task metadata: form={form!r}, version={version!r}"
                )
            items = parse_payload(text)
            if items is None:
                out.append(0.0)
                continue
            if isinstance(items, dict):
                items = [items]

            try:
                ok, _ = _validate(form, items, context)
                contract = 1.0 if ok else 0.0
            except Exception:
                contract = 0.0

            try:
                q = max(0.0, min(1.0, quality(form, items)))
            except Exception:
                q = 0.0
            out.append(contract * (self.W_CONTRACT + self.W_QUALITY * q))
        return out


orms["gsa_reward"] = GSAReward
