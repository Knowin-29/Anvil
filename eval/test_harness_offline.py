#!/usr/bin/env python3
"""Offline self-test: loaders and scorers, no GPU and no model.

Verifies that each dataset actually parses and that each scorer returns the expected
value on hand-built cases, including the reasoning-truncation and coordinate-scale
failure modes that previously produced misleading numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gsa_eval import datasets as ds  # noqa: E402
from eval.gsa_eval import scorers as sc  # noqa: E402
from eval.gsa_eval.coords import iou, to_unit_box, to_unit_points  # noqa: E402
from eval.gsa_eval.parsing import parse_choice, parse_points, strip_reasoning  # noqa: E402
from eval.gsa_eval.registry import get_spec  # noqa: E402
from eval.gsa_eval.runner import EVALUATION_PROTOCOL, RESULT_SCHEMA  # noqa: E402

FAILED: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")
    if not ok:
        FAILED.append(label)


def test_parsing() -> None:
    print("== parsing ==")
    check("think block stripped",
          parse_choice("<think>maybe (A) or (D)</think>Answer: (B)", 4), "B")
    check("truncated think -> no answer",
          parse_choice("<think>the answer is likely (A) because", 4), None)
    check("truncated think -> no points",
          parse_points("<think>maybe (0.5, 0.5) works"), [])
    check("bare trailing letter", parse_choice("<think>x</think> C", 4), "C")
    check("json point", parse_points('</think>[{"point_2d": [120, 340]}]'),
          [(120.0, 340.0)])
    check("tuple list", parse_points("</think>[(0.1, 0.2), (0.3, 0.4)]"),
          [(0.1, 0.2), (0.3, 0.4)])
    check("no reasoning tags passthrough", strip_reasoning("(A)"), "(A)")


def test_coords() -> None:
    print("== coords (the gsa_bench_test defect) ==")
    # 800x600 image, GT [140,43,344,662] is thousandth-scale: y2=662 overflows the
    # 600px height even though it stays under the 800px width, which is exactly the
    # case a max(width,height) comparison missed.
    check("axis-aware auto-detect finds thousandth",
          [round(v, 3) for v in to_unit_box([140, 43, 344, 662], 800, 600)],
          [0.14, 0.043, 0.344, 0.662])
    gt = to_unit_box([140, 43, 344, 662], 800, 600, declared="thousandth")
    check("declared thousandth GT normalised", [round(v, 3) for v in gt],
          [0.14, 0.043, 0.344, 0.662])
    # A pixel-scale prediction covering the same region must now score high.
    pred = to_unit_box([112, 26, 275, 397], 800, 600)
    check("pixel pred vs thousandth GT overlaps after fix", iou(pred, gt) > 0.9, True)
    check("unambiguous pixel box stays pixel",
          [round(v, 3) for v in to_unit_box([80, 60, 400, 300], 800, 600)],
          [0.1, 0.1, 0.5, 0.5])
    check("unit-scale passthrough", to_unit_points([(0.5, 0.5)], 800, 600),
          [(0.5, 0.5)])
    check("per-list unit inference (mixed magnitude)",
          [tuple(round(v, 3) for v in p)
           for p in to_unit_points([(12, 480)], 800, 600)], [(0.015, 0.8)])


def test_scorers() -> None:
    print("== scorers ==")
    S = ds.Sample
    samples = [S("a", "", [], "(B)", 100, 100, "2D", {"choices": list("wxyz")}),
               S("b", "", [], "(A)", 100, 100, "3D", {"choices": list("wxyz")})]
    score, detail, fails = sc.score_multiple_choice(
        samples, ["</think>(B)", "<think>truncated"])
    check("mc acc 1/2", score, 50.0)
    check("mc parse failure counted", fails, 1)
    check("mc per-group reported", sorted(detail["per_group"]), ["2D", "3D"])

    gsa = [S("p", "", [], [{"point_2d": [500, 500]}], 800, 600, "point",
             {"form": "point"}),
           S("q", "", [], [{"bbox_2d": [140, 43, 344, 662]}], 800, 600, "bbox",
             {"form": "bbox"})]
    score, detail, fails = sc.score_gsa_geometry(
        gsa, ['</think>[{"point_2d": [400, 300]}]',
              '</think>[{"bbox_2d": [112, 26, 275, 397]}]'])
    check("pixel point answer hits thousandth GT", detail["point"]["hit@0.1"], 100.0)
    check("bbox IoU@0.5 no longer collapses to 0", detail["bbox"]["iou@0.5"], 100.0)
    check("combined score is weighted, not falsey-fallback", score, 100.0)
    check("gsa parse failures", fails, 0)


def test_datasets() -> None:
    print("== datasets (real files) ==")
    scratch = Path(__file__).resolve().parents[1] / ".cache" / "eval_assets"
    scratch.mkdir(parents=True, exist_ok=True)
    for name, loader, want in (
            ("where2place", ds.load_where2place, 100),
            ("refspatial_bench", ds.load_refspatial_bench, 277),
            ("gsa_bench_test", ds.load_gsa_bench_test, 1463)):
        rows = loader(scratch)
        check(f"{name} count", len(rows), want)
        r = rows[0]
        assert r.prompt and r.image_paths, name
        check(f"{name} image exists", Path(r.image_paths[0]).is_file(), True)
        check(f"{name} has size", bool(r.width and r.height), True)


def test_protocol() -> None:
    print("== protocol ==")
    check("result schema", RESULT_SCHEMA, 7)
    check("protocol name", EVALUATION_PROTOCOL, "gsa-eval-v7")
    check("CV concise budget", get_spec("cv_bench").max_tokens, 128)
    check("CV direct-answer suffix",
          ds.load_cv_bench(Path("/tmp/gsa-eval-scratch"))[0].prompt.endswith(
              "do not provide reasoning or any other text."), True)
    check("EmbSpatial concise budget", get_spec("embspatial").max_tokens, 128)
    for name in ("where2place", "refspatial_bench", "gsa_bench_test"):
        check(f"{name} structured budget", get_spec(name).max_tokens, 256)


def main() -> int:
    test_parsing()
    test_coords()
    test_scorers()
    test_datasets()
    test_protocol()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("all offline checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
