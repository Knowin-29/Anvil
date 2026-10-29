"""Benchmark registry: one entry per benchmark, no external runner involved.

`available=False` entries are declared but refuse to produce a number, so the summary
distinguishes "not supported yet" from "measured badly". LLM-judge benchmarks stay
unavailable until a judge is configured rather than falling back to string overlap,
which would not be comparable to any published figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from . import datasets as ds
from . import scorers as sc


@dataclass
class BenchmarkSpec:
    name: str
    loader: Optional[Callable] = None
    scorer: Optional[Callable] = None
    metric: str = ""
    n_expected: int = 0
    available: bool = True
    skip_reason: str = ""
    max_tokens: int = 256
    # What the prompt asks the model to point AT. Measured per sample from the shipped
    # prompts (see eval/audit_referent_type.py), because it decides whether a point
    # benchmark tests the same task the training corpus taught:
    #   'object'   -- point at a thing        (training: "Point to the most prominent
    #                                          manipulable object", 100% single-point)
    #   'vacancy'  -- point at empty space    (opposite task; the model was never
    #                                          trained to find free space)
    #   'mixed'    -- both, split by group
    #   ''         -- not a pointing benchmark
    referent_type: str = ""
    # Per-group referent type, for the 'mixed' case. Keys are Sample.group values.
    referent_by_group: dict = field(default_factory=dict)
    # The coordinate convention the shipped prompt asks for. 'unit' means the prompt
    # states [0,1]; the scorer then reads an out-of-range answer as the model's
    # thousandth habit rather than admitting a third 'pixel' reading that depends on
    # image size by coincidence (see docs/audit_scorers_report.md F1/F2).
    prompt_coord_scale: str = ""
    # Whether the prompt asks for one point or several. Where2Place asks for "several
    # spots" in 100/100 prompts while the training answers are single points, so the
    # cardinality mismatch is part of the protocol gap, not a model failure.
    prompt_cardinality: str = ""
    notes: str = ""


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "cv_bench": BenchmarkSpec(
        "cv_bench", ds.load_cv_bench, sc.score_multiple_choice,
        metric="multiple-choice acc (2D+3D)", n_expected=2638, max_tokens=128),
    "embspatial": BenchmarkSpec(
        "embspatial", ds.load_embspatial, sc.score_multiple_choice,
        metric="multiple-choice acc (6 relations)", n_expected=3640, max_tokens=128),
    # These are structured-output tasks, not free-form reasoning tasks. Under the
    # non-thinking protocol a valid answer needs far fewer than 256 tokens. A model
    # that continues for thousands of tokens has failed concise task compliance; a
    # 4096-token budget previously turned one runaway sample into whole-batch tail
    # latency and made formal 9B runs approach the worker timeout.
    # Measured referent audit (eval/audit_referent_type.py, over the shipped prompts):
    # where2place is 100/100 vacancy and 100/100 asks for SEVERAL points, whereas the E1
    # training corpus is 100% "Point to the most prominent manipulable object" with a
    # single-point answer. So this benchmark asks for the opposite referent AND a
    # different cardinality; its base-vs-SFT gap is not evidence about geometry transfer.
    # It is kept and reported, but tagged so the paper cannot use it as such.
    "where2place": BenchmarkSpec(
        "where2place", ds.load_where2place, sc.score_point_in_mask,
        metric="point-in-mask", n_expected=100, max_tokens=256,
        referent_type="vacancy", prompt_coord_scale="unit",
        prompt_cardinality="several",
        notes="100/100 prompts ask for several points in VACANT space; the training "
              "corpus only ever asked for a single point on an OBJECT, so this cell "
              "measures protocol mismatch, not localisation ability"),
    # RefSpatial-Bench splits cleanly by referent (measured per prompt):
    #   location  100/100 object   ("Please point out the orange box.")
    #   placement 100/100 vacancy  ("... the free space on the top of the wooden shelf")
    #   unseen     21 object / 56 vacancy
    # `location` is therefore the ONLY published point subset that asks the same kind of
    # question the training data taught, which is why it is the one the paper may use
    # for an aligned comparison.
    "refspatial_bench": BenchmarkSpec(
        "refspatial_bench", ds.load_refspatial_bench, sc.score_point_in_mask,
        max_tokens=256,
        metric="point acc (point-in-mask)", n_expected=277,
        referent_type="mixed",
        referent_by_group={"location": "object", "placement": "vacancy",
                           "unseen": "mixed"},
        prompt_coord_scale="unit", prompt_cardinality="one",
        notes="report per split: only `location` is object-referring and thus aligned "
              "with the training task"),
    "gsa_bench_test": BenchmarkSpec(
        "gsa_bench_test", ds.load_gsa_bench_test, sc.score_gsa_geometry,
        metric="bbox IoU@0.5 + point hit@0.1 (normalised)", n_expected=1463,
        max_tokens=256),
    # ERQA ships a converted JSON (erqa_converted.json) alongside the tfrecord, so the
    # old "no tensorflow" blocker no longer applies. Only the single-image subset is
    # self-hosted; multi-image rows still need interleaved input and are skipped in the
    # loader, so n_expected reflects the scored single-image count.
    "erqa": BenchmarkSpec(
        "erqa", ds.load_erqa, sc.score_multiple_choice,
        metric="multiple-choice acc (single-image subset)", n_expected=287,
        max_tokens=128),
    # BLINK's Relative-Depth task is single-image two-choice MC, so it is self-hosted
    # here even though the wider BLINK suite needs multi-image input. It directly
    # exercises the S (relative-depth) family the manuscript flags as thin.
    "blink_relative_depth": BenchmarkSpec(
        "blink_relative_depth", ds.load_blink_relative_depth, sc.score_multiple_choice,
        metric="multiple-choice acc (relative depth)", n_expected=124, max_tokens=128),
    # RoboSpatial-Home yes/no relation questions (compatibility + configuration).
    "robospatial_home": BenchmarkSpec(
        "robospatial_home", ds.load_robospatial_home, sc.score_multiple_choice,
        metric="binary MC acc (compatibility+configuration)", n_expected=228,
        max_tokens=128),
    # RoboSpatial-Home context pointing subset: several points in vacant space, scored
    # point-in-mask under the unit-scale rule. Vacancy-referring, tagged so it is not
    # read as object localisation.
    "robospatial_home_context": BenchmarkSpec(
        "robospatial_home_context", ds.load_robospatial_home_context,
        sc.score_point_in_mask, metric="point-in-mask (vacant space)", n_expected=122,
        max_tokens=256, referent_type="vacancy", prompt_coord_scale="unit",
        prompt_cardinality="several"),
    # VABench-Point: manipulation-target pointing, scored point-in-box against a single
    # normalised [0,1] target box.
    "vabench_point": BenchmarkSpec(
        "vabench_point", ds.load_vabench_point, sc.score_point_in_box,
        metric="point-in-box acc", n_expected=300, max_tokens=256,
        referent_type="object", prompt_coord_scale="unit", prompt_cardinality="one"),
    # Declared but not yet self-hosted. Reasons are the measured blockers from
    # docs/experiment_findings.md, not guesses.
    "vsibench": BenchmarkSpec(
        "vsibench", metric="MRA+Acc", n_expected=5130, available=False,
        skip_reason="video input; frame sampling for 5,130 clips not implemented here"),
    "blink": BenchmarkSpec(
        "blink", metric="multiple-choice acc", available=False,
        skip_reason="multi-image interleaved input not implemented here (Relative_Depth "
                    "is self-hosted as blink_relative_depth)"),
    "robovqa": BenchmarkSpec(
        "robovqa", metric="LLM-judge", n_expected=1921, available=False,
        skip_reason="no judge available"),
    "openeqa": BenchmarkSpec(
        "openeqa", metric="LLM-judge 1-5", n_expected=1636, available=False,
        skip_reason="no judge available"),
}

# Benchmarks that are self-hosted and verified end-to-end.
PRIORITY = ["cv_bench", "embspatial", "where2place", "refspatial_bench"]


def get_spec(name: str) -> BenchmarkSpec:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]
