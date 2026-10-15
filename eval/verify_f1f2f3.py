#!/usr/bin/env python3
"""Check whether audit defects F1, F2, F3 and F5 are actually landed in the code.

The previous session was interrupted mid-edit, so "the report recommends X" and "X is
in the tree" had drifted apart. This asserts the behaviour rather than grepping for
comments: each check drives the real functions and fails loudly if the fix is absent.

Read-only with respect to the harness; no GPU, no network, deterministic.

    python eval/verify_f1f2f3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gsa_eval import runner as R  # noqa: E402
from eval.gsa_eval.coords import AMBIGUOUS, classify_scale, infer_scale  # noqa: E402
from eval.gsa_eval.parsing import parse_choice  # noqa: E402
from eval.gsa_eval.registry import get_spec  # noqa: E402
from eval.gsa_eval.scorers import _prompt_driven_scale, score_point_in_mask  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = "") -> None:
    results.append((name, bool(ok), note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note else ""))


def main() -> int:
    # F1: the ambiguous case must be reported as ambiguous, not guessed as 'pixel'.
    # (568, 300) on 640x480 is a legal pixel value AND a legal thousandth value.
    amb = classify_scale([568], [300], 640, 480)
    old = infer_scale([568], [300], 640, 480)
    check("F1 classify_scale reports AMBIGUOUS instead of guessing pixel",
          amb == AMBIGUOUS and old == "pixel",
          f"classify={amb!r} legacy_infer={old!r}")

    # F1/F2: under a prompt that states [0,1], 'pixel' must be inadmissible.
    check("F1/F2 prompt-driven rule admits only unit|thousandth",
          _prompt_driven_scale([(568, 300)]) == "thousandth"
          and _prompt_driven_scale([(0.5, 0.7)]) == "unit")

    # F2: both point benchmarks must declare the prompt's scale so the rule is reached.
    declared = {b: get_spec(b).prompt_coord_scale
                for b in ("where2place", "refspatial_bench")}
    check("F2 point benchmarks declare prompt_coord_scale='unit'",
          all(v == "unit" for v in declared.values()), json.dumps(declared))

    # F2 end-to-end: a thousandth answer on a 640x480 frame must be read as thousandth.
    class _S:
        def __init__(self, gt, w, h, g="all"):
            self.gt, self.width, self.height, self.group = gt, w, h, g
            self.meta = {}

    class _Spec:
        prompt_coord_scale = "unit"
        referent_type = ""
        referent_by_group: dict = {}

    import numpy as np
    from PIL import Image
    tmp = ROOT / ".cache" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    mp = tmp / "_verify_mask.png"
    m = np.zeros((480, 640), dtype=np.uint8)
    # (568, 300) is the exact ambiguous case from the audit: legal as a thousandth value
    # AND as a pixel coordinate on a 640x480 frame. Read as thousandth it lands at pixel
    # (0.568*639, 0.300*479) = (363, 143); read as pixel at (0.8875*639, 0.625*479) =
    # (567, 299). Only the thousandth reading may hit the mask.
    m[133:153, 353:373] = 255
    Image.fromarray(m).save(mp)
    answer = '[{"point_2d": [568, 300]}]'
    s_new, d_new, _ = score_point_in_mask(
        [_S(str(mp), 640, 480)], [answer], spec=_Spec())

    class _SpecOld(_Spec):
        prompt_coord_scale = ""
    s_old, _, _ = score_point_in_mask(
        [_S(str(mp), 640, 480)], [answer], spec=_SpecOld())
    check("F2 thousandth answer scores 1.0 under prompt-driven, 0.0 under old auto",
          s_new == 100.0 and s_old == 0.0,
          f"prompt_driven={s_new} auto={s_old} used={d_new['coord_scale']['used']}")

    # F3: the trust gate and the per-uid parse record must both exist.
    check("F3 trust gate thresholds present",
          R.TRUNCATION_LIMIT == 15.0 and R.PARSE_FAILURE_LIMIT == 25.0,
          f"trunc={R.TRUNCATION_LIMIT} parse={R.PARSE_FAILURE_LIMIT}")
    check("F3 paired-subset support (_parse_success) present",
          callable(getattr(R, "_parse_success", None)))

    # refspatial split reporting, the finding that must stay reproducible.
    rb = get_spec("refspatial_bench")
    check("refspatial_bench reports per referent split",
          rb.referent_by_group.get("location") == "object"
          and rb.referent_by_group.get("placement") == "vacancy",
          json.dumps(rb.referent_by_group))

    # Engine cache.
    check("engine cache present", isinstance(getattr(R, "_ENGINE_CACHE", None), dict)
          and callable(getattr(R, "get_engine", None)))

    # F5: the bare-letter fallback must not fire on an English article.
    got = parse_choice("It is a cup on the table.", 2)
    check("F5 bare-letter fallback does not read article 'a' as choice A",
          got is None, f"got={got!r}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
