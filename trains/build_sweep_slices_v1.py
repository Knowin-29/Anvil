#!/usr/bin/env python3
"""Build the training-configuration sweep slices from corpus_source_clean_v1/p4.

Reads P4 read-only and writes every artefact under trains/sweep_slices_v1/.

Design constraints this file encodes:

* image-disjoint train/val: the split is decided by a hash of the sample's image
  path, so no image can appear on both sides. Row-level sampling would leak,
  because P4 holds 263,322 rows over only 159,497 images.
* per-form proportions preserved: the sampler draws each form in its P4 share so
  a slice is a scale model of the corpus rather than a reweighting of it.
* coordinate-convention arms are pure re-encodings of the SAME rows. Arms A/B/C
  share one row set and one row order; only the numeals (and, for C, an explicit
  scale instruction) differ. Any score gap is therefore attributable to the
  convention and not to a different data draw.
* the unit conversion is key-allowlisted, never a blanket "divide every number".
  `jaw_axis_2d` is already a unit vector, `relative_depth` is already 0-1,
  `confidence`, `wrist_deg` and `rank` are not coordinates; dividing those would
  silently corrupt four of the nine forms.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "trains" / "corpus_source_clean_v1" / "p4" / "train.jsonl"
P4_SHA256 = "1afebc5a30ebdb98d3d9173c53d1b5e350f2b9d06d2693015ee35726516b6f82"
P4_ROWS = 263322
OUT_ROOT = ROOT / "trains" / "sweep_slices_v1"

# Keys whose values are pixel-ish coordinates on the thousandth grid.
POINT_KEYS = ("point_2d", "contact_point_2d", "grasp_point_2d")
BOX_KEYS = ("bbox_2d",)
# Keys that look numeric but are NOT thousandth coordinates.
NON_COORD_KEYS = ("jaw_axis_2d", "relative_depth", "confidence", "wrist_deg", "rank")

SCALE_HINT = {
    "thousandth": "Express all coordinates on a 0-1000 integer grid.",
    "unit": "Express all coordinates as decimals normalized to [0, 1].",
}


def _num(v: float) -> str:
    """Thousandth -> unit with 1/1000 resolution, i.e. no information is lost."""
    return f"{round(v / 1000.0, 3):g}"


def _conv_list(body: str) -> str:
    return ", ".join(_num(float(t)) for t in re.split(r"\s*,\s*", body.strip()) if t)


def to_unit(text: str) -> str:
    """Re-encode thousandth coordinates as [0,1] decimals, key by key.

    Returns the text unchanged where no allowlisted key matches, so forms that
    carry no coordinates (`qa`) pass through byte-for-byte.
    """
    out = text
    for key in POINT_KEYS + BOX_KEYS:
        out = re.sub(
            r'("%s"\s*:\s*\[)([^\[\]]*?)(\])' % re.escape(key),
            lambda m: m.group(1) + _conv_list(m.group(2)) + m.group(3),
            out,
        )
    # polygon: [[x, y], [x, y], ...] -- convert the inner pairs only.
    def _poly(m: re.Match) -> str:
        inner = re.sub(r"\[([^\[\]]*?)\]",
                       lambda p: "[" + _conv_list(p.group(1)) + "]", m.group(2))
        return m.group(1) + inner + m.group(3)

    out = re.sub(r'("polygon"\s*:\s*\[)(.*?)(\]\s*\})', _poly, out, flags=re.S)
    # RefSpatial's own tag form: <point>[[x,y]]</point>
    out = re.sub(
        r"(<point>\s*\[\[)([^\[\]]*?)(\]\]\s*</point>)",
        lambda m: m.group(1) + _conv_list(m.group(2)) + m.group(3),
        out,
    )
    # action_cot embeds a box inside prose: "Pick up the mug at [367, 323, 419, 359]"
    out = re.sub(
        r"(at\s+\[)(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){3})(\])",
        lambda m: m.group(1) + _conv_list(m.group(2)) + m.group(3),
        out,
    )
    return out


def n_points(answer: str) -> int:
    return len(re.findall(r'"point_2d"\s*:\s*\[', answer)) + \
        len(re.findall(r"<point>\s*\[\[", answer))


def image_bucket(path: str) -> int:
    return int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) % 1000


def load_pool(limit_scan: int = 0) -> list[dict]:
    rows = []
    with P4.open() as f:
        for i, line in enumerate(f):
            if limit_scan and i >= limit_scan:
                break
            d = json.loads(line)
            imgs = d.get("images") or []
            if not imgs:
                continue
            d["_img"] = imgs[0]
            d["_bucket"] = image_bucket(imgs[0])
            d["_form"] = d.get("gsa_meta", {}).get("form", "unknown")
            d["_npt"] = n_points(d["messages"][-1]["content"])
            rows.append(d)
    return rows


def stratified_draw(pool: list[dict], n: int, rng: random.Random,
                    forms: dict[str, int]) -> list[dict]:
    """Draw `n` rows keeping each form's share of `forms` (the P4 distribution)."""
    total = sum(forms.values())
    by_form: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        by_form[r["_form"]].append(r)
    picked: list[dict] = []
    for form, count in sorted(forms.items()):
        want = max(1, round(n * count / total))
        avail = by_form.get(form, [])
        rng.shuffle(avail)
        picked.extend(avail[:want])
    rng.shuffle(picked)
    return picked[:n]


def write_arm(rows: list[dict], path: Path, convention: str,
              hint: bool) -> dict:
    """Emit one arm. `convention` in {thousandth, unit, mixed}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260831)
    counts = collections.Counter()
    with path.open("w") as f:
        for r in rows:
            msgs = [dict(m) for m in r["messages"]]
            if convention == "mixed":
                # Deterministic per row so the arm is reproducible, and balanced
                # 50/50 so neither convention is the majority the model can
                # collapse onto while ignoring the instruction.
                conv = "unit" if rng.random() < 0.5 else "thousandth"
            else:
                conv = convention
            counts[conv] += 1
            if conv == "unit":
                for m in msgs:
                    m["content"] = to_unit(m["content"])
            if hint:
                msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n" + SCALE_HINT[conv]
            out = {"messages": msgs, "images": r["images"],
                   "gsa_meta": r["gsa_meta"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return {"path": str(path), "rows": len(rows),
            "convention": convention, "prompt_scale_hint": hint,
            "convention_counts": dict(counts),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-rows", type=int, default=6000)
    ap.add_argument("--val-rows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--val-buckets", type=int, default=30,
                    help="image buckets (of 1000) reserved for validation")
    a = ap.parse_args()

    assert P4.is_file(), f"missing {P4}"
    rng = random.Random(a.seed)

    print(f"[slices] scanning {P4}", flush=True)
    pool = load_pool()
    assert len(pool) <= P4_ROWS, (len(pool), P4_ROWS)
    print(f"[slices] pool {len(pool)} rows, "
          f"{len({r['_img'] for r in pool})} images", flush=True)

    val_pool = [r for r in pool if r["_bucket"] < a.val_buckets]
    train_pool = [r for r in pool if r["_bucket"] >= a.val_buckets]
    forms = collections.Counter(r["_form"] for r in pool)

    train_rows = stratified_draw(train_pool, a.train_rows, rng, forms)
    val_rows = stratified_draw(val_pool, a.val_rows, rng, forms)

    tr_imgs = {r["_img"] for r in train_rows}
    va_imgs = {r["_img"] for r in val_rows}
    assert not (tr_imgs & va_imgs), "train/val share images"

    # Cardinality arms reuse the same non-point rows and swap only the point rows,
    # so a gap between them isolates cardinality rather than form mix.
    pt = [r for r in train_rows if r["_form"] == "point"]
    other = [r for r in train_rows if r["_form"] != "point"]
    single_pool = [r for r in train_pool
                   if r["_form"] == "point" and r["_npt"] == 1
                   and r["_img"] not in va_imgs]
    rng.shuffle(single_pool)
    single_only = single_pool[:len(pt)]
    assert len(single_only) == len(pt), (len(single_only), len(pt))
    card_single = other + single_only
    rng.shuffle(card_single)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    arms = {
        "A_thousandth": write_arm(train_rows, OUT_ROOT / "train_A_thousandth.jsonl",
                                  "thousandth", False),
        "B_unit": write_arm(train_rows, OUT_ROOT / "train_B_unit.jsonl",
                            "unit", False),
        "C_mixed_hint": write_arm(train_rows, OUT_ROOT / "train_C_mixed_hint.jsonl",
                                  "mixed", True),
    }
    val = {
        "A_thousandth": write_arm(val_rows, OUT_ROOT / "val_A_thousandth.jsonl",
                                 "thousandth", False),
        "B_unit": write_arm(val_rows, OUT_ROOT / "val_B_unit.jsonl", "unit", False),
        "C_mixed_hint": write_arm(val_rows, OUT_ROOT / "val_C_mixed_hint.jsonl",
                                  "mixed", True),
    }

    card = {}
    for name, rows in (("single", card_single),):
        for arm, (conv, hint) in (("A_thousandth", ("thousandth", False)),
                                  ("B_unit", ("unit", False))):
            key = f"{arm}_pt_{name}"
            card[key] = write_arm(rows, OUT_ROOT / f"train_{key}.jsonl", conv, hint)

    pt_card = collections.Counter(r["_npt"] for r in pt)
    manifest = {
        "builder": {"path": str(Path(__file__).resolve()),
                    "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "source_corpus": {"path": str(P4), "expected_rows": P4_ROWS,
                          "expected_sha256": P4_SHA256,
                          "observed_rows": len(pool)},
        "seed": a.seed,
        "split": {"rule": "sha256(image_path) %% 1000",
                  "val_buckets": f"<{a.val_buckets}",
                  "train_rows": len(train_rows), "val_rows": len(val_rows),
                  "train_images": len(tr_imgs), "val_images": len(va_imgs),
                  "image_disjoint": True},
        "form_distribution_source": dict(forms),
        "form_distribution_train": dict(collections.Counter(
            r["_form"] for r in train_rows)),
        "form_distribution_val": dict(collections.Counter(
            r["_form"] for r in val_rows)),
        "point_cardinality_train_natural": dict(sorted(pt_card.items())),
        "coordinate_arms": arms,
        "validation_arms": val,
        "cardinality_arms": card,
        "notes": [
            "arms A/B/C hold the identical row set and order; only the coordinate "
            "encoding differs (C additionally declares the scale in the prompt, "
            "which is what makes a mixed convention learnable rather than noise)",
            "unit conversion is key-allowlisted; jaw_axis_2d / relative_depth / "
            "confidence / wrist_deg / rank are left untouched",
            "cardinality arms swap only the point rows and keep every other form "
            "row identical to the coordinate arms",
        ],
    }
    mpath = OUT_ROOT / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("coordinate_arms", "validation_arms",
                                   "cardinality_arms")},
                     indent=2, ensure_ascii=False))
    print(f"[slices] wrote {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
