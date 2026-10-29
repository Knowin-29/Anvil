"""Dataset loaders.

Each loader yields Sample objects with a uniform shape regardless of the on-disk
format (parquet with inline image bytes, one giant JSON with base64 images, or jsonl
plus an image directory). Images are materialised into a scratch directory as files so
they can be handed to swift's engine by path.

Deterministic subsampling: --limit takes a seeded stratified subset, not a prefix, so
a limited run stays comparable across checkpoints and is not dominated by whichever
task happens to sort first.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ.get(
    "ANVIL_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2] / "data/benchmarks")
)).expanduser().resolve()


@dataclass
class Sample:
    """One benchmark question, normalised across source formats."""
    uid: str
    prompt: str
    image_paths: list[str]
    gt: Any
    width: int = 0
    height: int = 0
    group: str = "all"          # stratification key / per-subset reporting
    meta: dict = field(default_factory=dict)


def _scratch(bench: str, scratch_root: Path) -> Path:
    d = scratch_root / bench
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_image(data: bytes, dest: Path) -> tuple[int, int]:
    """Write image bytes to dest (idempotent) and return its (width, height)."""
    from PIL import Image
    if not dest.is_file():
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    with Image.open(dest) as im:
        return im.size


def stratified_subset(samples: list[Sample], limit: int, seed: int = 0) -> list[Sample]:
    """Take `limit` samples spread evenly over groups, deterministically."""
    if not limit or limit <= 0 or limit >= len(samples):
        return samples
    buckets: dict[str, list[Sample]] = {}
    for s in samples:
        buckets.setdefault(s.group, []).append(s)
    rng = random.Random(seed)
    for key in buckets:
        buckets[key] = sorted(buckets[key], key=lambda s: s.uid)
        rng.shuffle(buckets[key])
    out: list[Sample] = []
    keys = sorted(buckets)
    i = 0
    while len(out) < limit:
        progressed = False
        for k in keys:
            if i < len(buckets[k]):
                out.append(buckets[k][i])
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
        i += 1
    return out


def _image_size(path: str) -> tuple[int, int]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)
# --- loaders ---


def load_cv_bench(scratch_root: Path) -> list[Sample]:
    """CV-Bench: 1,438 2D + 1,200 3D multiple-choice rows.

    Images live only inside the parquet files (the sibling jsonl has no usable image
    column), so they are extracted to scratch on first use.
    """
    import pyarrow.parquet as pq
    out: list[Sample] = []
    sc = _scratch("cv_bench", scratch_root)
    for kind, fname in (("2D", "test_2d.parquet"), ("3D", "test_3d.parquet")):
        path = DATA_ROOT / "CV-Bench" / fname
        if not path.is_file():
            raise FileNotFoundError(path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                img = row.get("image") or {}
                data = img.get("bytes")
                if not data:
                    continue
                uid = f"{kind}_{row['idx']}"
                dest = sc / f"{uid}.png"
                w, h = _write_image(data, dest)
                prompt = (row["prompt"].rstrip() + "\n"
                          "Answer with only the option letter from the given choices; "
                          "do not provide reasoning or any other text.")
                out.append(Sample(
                    uid=uid, prompt=prompt, image_paths=[str(dest)],
                    gt=row["answer"], width=w, height=h,
                    group=f"{kind}/{row.get('task', '?')}",
                    meta={"choices": list(row.get("choices") or []),
                          "type": kind, "task": row.get("task")}))
    return out


def load_embspatial(scratch_root: Path) -> list[Sample]:
    """EmbSpatial-Bench: 3,640 rows in one 242 MB JSON with base64 images.

    The prompt is assembled here because the source ships question + option list
    separately; the answer is a zero-based index into answer_options.
    """
    path = DATA_ROOT / "EmbSpatial-Bench" / "embspatial_bench.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    sc = _scratch("embspatial", scratch_root)
    with path.open() as f:
        rows = json.load(f)
    out: list[Sample] = []
    for row in rows:
        opts = list(row.get("answer_options") or [])
        if not opts or not isinstance(row.get("answer"), int):
            continue
        uid = str(row.get("question_id"))
        dest = sc / f"{uid}.jpg"
        try:
            w, h = _write_image(base64.b64decode(row["image"]), dest)
        except Exception:
            continue
        letters = [chr(ord("A") + i) for i in range(len(opts))]
        block = "\n".join(f"({l}) {o}" for l, o in zip(letters, opts))
        prompt = (f"{row['question']} Select from the following choices.\n{block}\n"
                  "Answer with the option's letter from the given choices directly.")
        out.append(Sample(
            uid=uid, prompt=prompt, image_paths=[str(dest)],
            gt=letters[row["answer"]], width=w, height=h,
            group=str(row.get("relation", "?")),
            meta={"choices": opts, "relation": row.get("relation"),
                  "data_source": row.get("data_source")}))
    return out


def load_where2place(scratch_root: Path) -> list[Sample]:
    """Where2Place: 100 free-space questions, scored point-in-mask.

    Images and masks are already files on disk; the mask is the ground truth. The
    masks directory holds 131 files: a `.jpg` for all 100 questions plus a `.png` for
    31 of them. The two agree on coverage (measured identical >127 fractions), but the
    PNG is losslessly binary (2 grey levels) whereas the JPEG carries compression
    ringing (33-59 levels), so the PNG is preferred and the JPEG is the fallback.
    Requiring PNG alone silently dropped 69 of the 100 questions.
    """
    base = DATA_ROOT / "where2place"
    qfile = base / "point_questions.jsonl"
    if not qfile.is_file():
        raise FileNotFoundError(qfile)
    out: list[Sample] = []
    for line in qfile.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        img = base / "images" / row["image"]
        stem = Path(row["image"]).stem
        mask = next((m for ext in (".png", ".jpg")
                     if (m := base / "masks" / (stem + ext)).is_file()), None)
        if not img.is_file() or mask is None:
            continue
        w, h = _image_size(str(img))
        out.append(Sample(
            uid=str(row["question_id"]), prompt=row["text"],
            image_paths=[str(img)], gt=str(mask), width=w, height=h,
            group=str(row.get("category", "all"))))
    return out


def load_refspatial_bench(scratch_root: Path) -> list[Sample]:
    """RefSpatial-Bench: 100 Location + 100 Placement + 77 Unseen = 277 rows.

    Both image and mask are inline parquet bytes. The shipped `suffix` already asks
    for normalised [0,1] tuples, so it is kept verbatim to stay comparable with the
    published protocol.
    """
    import pyarrow.parquet as pq
    out: list[Sample] = []
    sc = _scratch("refspatial_bench", scratch_root)
    for split in ("location", "placement", "unseen"):
        path = DATA_ROOT / "RefSpatial-Bench" / "data" / f"{split}-00000-of-00001.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                img_b = (row.get("image") or {}).get("bytes")
                mask_b = (row.get("mask") or {}).get("bytes")
                if not img_b or not mask_b:
                    continue
                uid = f"{split}_{row['id']}"
                img_dest = sc / f"{uid}.jpg"
                mask_dest = sc / f"{uid}_mask.png"
                w, h = _write_image(img_b, img_dest)
                _write_image(mask_b, mask_dest)
                prompt = f"{row['prompt']} {row.get('suffix', '')}".strip()
                out.append(Sample(
                    uid=uid, prompt=prompt, image_paths=[str(img_dest)],
                    gt=str(mask_dest), width=w, height=h, group=split,
                    meta={"step": row.get("step"), "object": row.get("object")}))
    return out


#: What each GSA-Bench form's ground truth actually looks like, so the prompt can ask
#: for the thing the scorer checks. Measured over trains/gsa_bench/test.jsonl:
#: 378/378 bbox rows carry exactly one `bbox_2d`, 1085/1085 point rows exactly one
#: `point_2d`.
_FORM_SCHEMA = {
    "bbox": '[{"label": "<name>", "bbox_2d": [x1, y1, x2, y2]}]',
    "point": '[{"label": "<name>", "point_2d": [x, y]}]',
}


def load_gsa_bench_test(scratch_root: Path, data: str | None = None, *,
                        declare_form: bool = True) -> list[Sample]:
    """GSA-Bench held-out test split: 1,085 point + 378 bbox rows, human GT.

    Two prompt defects are fixed here rather than in the scorer, because both are
    cases of the harness scoring something it never asked for.

    1. Coordinate scale. The stored prompts state no convention (measured 1463/1463)
       while the GT is thousandth-scale, so the model answers in pixels and IoU
       collapses. An explicit convention is appended.

    2. Output form. The bbox rows' stored prompt is `Locate "X" in the image.` in
       378/378 cases and never names a geometric form, yet `score_gsa_geometry`
       requires a box for them. Both models therefore answered with a point and were
       booked as parse failures: 105/300 for base and 232/300 for the SFT checkpoint
       in the recorded dump, which is the whole of that run's 18.00% vs 38.83% parse
       failure gap (measured, eval/audit_gsa_form_mismatch.py). The asymmetry is not
       noise -- the E1 corpus taught bbox under a different template
       (`Detect the main manipulable object and output its bounding box in JSON`,
       0/378 test prompts match it) while teaching `point` as the default way to
       indicate one salient object, so the under-specified prompt pushes the SFT model
       towards a point much harder than it pushes base. Declaring the form removes a
       harness ambiguity; it does not make a wrong answer count as right.

    `declare_form=False` restores the old under-specified prompt so the defect can be
    reproduced.
    """
    path = Path(data) if data else (Path(__file__).resolve().parents[2]
                                    / "trains" / "gsa_bench" / "test.jsonl")
    if not path.is_file():
        raise FileNotFoundError(path)
    scale_hint = (" Express every coordinate on a 0-1000 thousandth scale "
                  "relative to the image size (not raw pixels).")
    out: list[Sample] = []
    for i, line in enumerate(path.read_text(errors="ignore").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        msgs = row.get("messages") or []
        if len(msgs) < 2 or not row.get("images"):
            continue
        try:
            gt = json.loads(msgs[1]["content"])
        except Exception:
            continue
        form = (row.get("gsa_meta") or {}).get("form", "?")
        img = row["images"][0]
        w, h = _image_size(img)
        if not w or not h:
            continue
        schema = _FORM_SCHEMA.get(form) if declare_form else None
        form_hint = (f" Answer with JSON of the form {schema}." if schema
                     else " Answer in JSON.")
        out.append(Sample(
            uid=f"{form}_{i}", prompt=msgs[0]["content"] + form_hint + scale_hint,
            image_paths=[img], gt=gt, width=w, height=h, group=form,
            meta={"form": form, "source": (row.get("gsa_meta") or {}).get("source")}))
    return out


def load_erqa(scratch_root: Path) -> list[Sample]:
    """ERQA: 400 embodied-reasoning MC questions from the converted JSON.

    Only single-image items are scored here (287/400 rows); multi-image questions need
    interleaved input the harness does not support and are skipped, mirroring the way
    other multi-image benchmarks stay declared-but-out. The prompt already ends with an
    answer-only instruction and embeds the lettered choices, so the letter parser reads
    it directly and the zero-based visual-index images resolve to files on disk.
    """
    base = DATA_ROOT / "ERQA"
    path = base / "erqa_converted.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = json.load(path.open())
    out: list[Sample] = []
    for key, row in rows.items():
        paths = row.get("image_path") or []
        if len(paths) != 1:
            continue
        img = base / paths[0]
        if not img.is_file():
            continue
        w, h = _image_size(str(img))
        gold = str(row.get("original_answer", "")).strip().upper()[:1]
        if not gold:
            continue
        n_choices = len(re.findall(r"(?:^|\s)([A-H])\.\s", row.get("question", "")))
        out.append(Sample(
            uid=str(key), prompt=row["question"], image_paths=[str(img)],
            gt=gold, width=w, height=h,
            group=str(row.get("task_type", "?")),
            meta={"choices": [chr(ord("A") + i) for i in range(max(2, n_choices))],
                  "task_type": row.get("task_type")}))
    return out


def load_blink_relative_depth(scratch_root: Path) -> list[Sample]:
    """BLINK Relative-Depth (val): 124 two-choice depth-ordering questions.

    Single-image MC. The parquet ships the choices as a list and the answer as a
    bracketed letter like ``(A)``; both are normalised to bare letters so the shared
    multiple-choice scorer applies without a bespoke path.
    """
    import pyarrow.parquet as pq
    path = DATA_ROOT / "BLINK" / "Relative_Depth" / "val-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    sc = _scratch("blink_relative_depth", scratch_root)
    out: list[Sample] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            img = (row.get("image_1") or {})
            data = img.get("bytes")
            if not data:
                continue
            uid = str(row["idx"])
            dest = sc / f"{uid}.jpg"
            w, h = _write_image(data, dest)
            choices = list(row.get("choices") or [])
            letters = [chr(ord("A") + i) for i in range(len(choices))]
            block = "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
            prompt = (f"{row['question']}\n{block}\n"
                      "Answer with the option's letter from the given choices directly.")
            gold = str(row.get("answer", "")).strip().strip("()").upper()[:1]
            if not gold:
                continue
            out.append(Sample(
                uid=uid, prompt=prompt, image_paths=[str(dest)],
                gt=gold, width=w, height=h, group="relative_depth",
                meta={"choices": choices, "sub_task": row.get("sub_task")}))
    return out


def _robospatial_yesno(scratch_root: Path) -> list[Sample]:
    import pyarrow.parquet as pq
    sc = _scratch("robospatial_home", scratch_root)
    out: list[Sample] = []
    for cat in ("compatibility", "configuration"):
        path = DATA_ROOT / "RoboSpatial-Home" / "data" / f"{cat}-00000-of-00001.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        pf = pq.ParquetFile(path)
        i = 0
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                data = (row.get("img") or {}).get("bytes")
                ans = str(row.get("answer", "")).strip().lower()
                if not data or ans not in ("yes", "no"):
                    continue
                uid = f"{cat}_{i}"; i += 1
                dest = sc / f"{uid}.jpg"
                w, h = _write_image(data, dest)
                q = re.sub(r"\s*Answer yes or no\.?\s*$", "", row["question"].strip())
                prompt = (f"{q}\n(A) yes\n(B) no\n"
                          "Answer with the option's letter from the given choices directly.")
                out.append(Sample(
                    uid=uid, prompt=prompt, image_paths=[str(dest)],
                    gt="A" if ans == "yes" else "B", width=w, height=h, group=cat,
                    meta={"choices": ["yes", "no"], "category": cat}))
    return out


def load_robospatial_home(scratch_root: Path) -> list[Sample]:
    """RoboSpatial-Home: 228 yes/no spatial-relation questions (compatibility+config).

    The two yes/no subsets share one binary MC contract, so they are converted to a
    two-choice ``(A) yes / (B) no`` prompt and scored by the shared multiple-choice
    scorer. The pointing ``context`` subset is a different task (several points in
    vacant space) and is registered separately as ``robospatial_home_context``.
    """
    return _robospatial_yesno(scratch_root)


def load_robospatial_home_context(scratch_root: Path) -> list[Sample]:
    """RoboSpatial-Home context: 122 vacant-space pointing questions, point-in-mask.

    Each row carries a PNG mask of the valid free space; the prompt already asks for a
    list of [0,1] tuples, so it is scored by the shared point-in-mask scorer under the
    unit-scale rule. This is a vacancy-referring task, tagged so it is not read as
    object localisation.
    """
    import pyarrow.parquet as pq
    sc = _scratch("robospatial_home_context", scratch_root)
    path = DATA_ROOT / "RoboSpatial-Home" / "data" / "context-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    out: list[Sample] = []
    pf = pq.ParquetFile(path)
    i = 0
    for batch in pf.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            data = (row.get("img") or {}).get("bytes")
            mask = (row.get("mask") or {}).get("bytes")
            if not data or not mask:
                continue
            uid = f"context_{i}"; i += 1
            dest = sc / f"{uid}.jpg"
            w, h = _write_image(data, dest)
            mpath = sc / f"{uid}_mask.png"
            if not mpath.is_file():
                mpath.write_bytes(mask)
            out.append(Sample(
                uid=uid, prompt=row["question"], image_paths=[str(dest)],
                gt=str(mpath), width=w, height=h, group="context"))
    return out


def load_vabench_point(scratch_root: Path) -> list[Sample]:
    """VABench-Point (test): 300 manipulation-target pointing questions, point-in-box.

    Ground truth is a single normalised [0,1] target box; a predicted point counts when
    it lands inside that box. The shipped ``problem`` opens with a literal ``<image>``
    placeholder that is stripped so it does not reach the model as text.
    """
    import pyarrow.parquet as pq
    sc = _scratch("vabench_point", scratch_root)
    path = DATA_ROOT / "VABench-Point" / "data" / "test-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    out: list[Sample] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            data = (row.get("image") or {}).get("bytes")
            gb = row.get("normalized_bbox")
            if not data or gb is None or len(gb) != 4:
                continue
            uid = str(row["idx"])
            dest = sc / f"{uid}.png"
            w, h = _write_image(data, dest)
            prompt = str(row["problem"]).replace("<image>", "").strip()
            out.append(Sample(
                uid=uid, prompt=prompt, image_paths=[str(dest)],
                gt=[float(v) for v in gb], width=w, height=h, group="point",
                meta={"prompt_coord_scale": "unit"}))
    return out
