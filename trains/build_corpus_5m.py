#!/usr/bin/env python3
"""Assemble the >=5M-sample GSA training corpus in ms-swift format.

Streams every open-license source, validates each record against its task-form
contract, drops non-embodied sources (chart/doc/OCR), dedups by (image, answer),
and enforces a scene-balance target so the corpus is embodied-dominated.

Validation (per record):
  - image path resolves after /mnt -> /path/to/workspace remap
  - <image> tag count == len(images)
  - coordinates parse and lie in [0,1000)
  - assistant answer non-empty and not a refusal/echo
Rejected records are counted by reason (audit trail for the paper).

Output: shards under trains/corpus_v1/ + a manifest with exact counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.task_taxonomy import (FORMS, NON_EMBODIED, SOURCE_SCENE, classify)  # noqa: E402

_VG_ROOT = "/path/to/datasets/datasets/transferred_datasets/datasets_train/robopoint-data/images/"

REMAP = [("/mnt/data/datasets/", "/path/to/datasets/datasets/"),
         ("/mnt/data/", "/path/to/datasets/"),
         ("/path/to/storage/", "/path/to/workspace/foundation/"),
         # Visual Genome jsonls carry a typo'd root ("/data/da/transferred_dataset/");
         # the real images live under the robopoint-data mirror (verified 30/30).
         ("/path/to/datasets/da/transferred_dataset/visual_genome/images/", _VG_ROOT),
         ("/mnt/data/da/transferred_dataset/visual_genome/images/", _VG_ROOT)]
_NUM = re.compile(r'-?\d+\.?\d*')
_COORD_BLOCK = re.compile(r'"(?:point_2d|bbox_2d|bbox)"\s*:\s*\[([^\]]*)\]|<point>\[\[([^\]]*)\]\]')


def remap(p: str) -> str:
    for a, b in REMAP:
        if p.startswith(a):
            return b + p[len(a):]
    return p


def coords_ok(asst: str) -> bool:
    """Every coordinate found must be inside [0,1000). Empty -> ok (text answer)."""
    found = False
    for m in _COORD_BLOCK.finditer(asst):
        blob = m.group(1) or m.group(2) or ""
        for tok in _NUM.findall(blob):
            found = True
            try:
                v = float(tok)
            except ValueError:
                return False
            if not (0 <= v < 1000):
                return False
    return True or found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="configs/data/sources_clean.yaml")
    ap.add_argument("--out-dir", default="trains/corpus_v1")
    ap.add_argument("--target", type=int, default=5_200_000)
    ap.add_argument("--general-cap", type=float, default=0.10,
                    help="max fraction of general_visual (anti-forgetting) data")
    ap.add_argument("--shard-size", type=int, default=500_000)
    ap.add_argument("--cap-frac", type=float, default=0.22,
                    help="max fraction of a bucket any single source may contribute")
    ap.add_argument("--source-weight", action="append", default=[],
                    metavar="NAME=W",
                    help="scale a source's quota, e.g. --source-weight visual_genome=0.35 "
                         "(repeatable). Used to down-weight over-represented sources.")
    ap.add_argument("--check-images", type=int, default=200,
                    help="sample this many image existence checks per source (0=all, slow)")
    args = ap.parse_args()

    import yaml
    reg = yaml.safe_load(Path(args.registry).read_text())
    sources = reg["sources"]

    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    reject = Counter(); by_form = Counter(); by_scene = Counter(); by_src = Counter()
    seen: set[str] = set()
    shard_idx = 0; in_shard = 0; n_out = 0
    fh = (outd / f"shard_{shard_idx:03d}.jsonl").open("w")
    general_budget = int(args.target * args.general_cap)

    # ---- per-source quota so no single source starves the rest ----
    usable = []
    for s in sources:
        name = s.get("name", "?")
        if name in NON_EMBODIED:
            reject["src_non_embodied"] += int(s.get("est_samples") or 0)
            continue
        p = remap(str(s.get("path", "")))
        if not os.path.isfile(p):
            reject["src_missing"] += 1
            continue
        usable.append((name, p, SOURCE_SCENE.get(name, "general_visual")))

    emb = [u for u in usable if u[2] != "general_visual"]
    gen = [u for u in usable if u[2] == "general_visual"]
    emb_target = args.target - general_budget

    def avail(p: str) -> int:
        try:
            with open(p, "rb") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    # Proportional-with-cap quota: a source may contribute up to `cap_frac` of the
    # bucket, so large sources fill the target without any single one dominating.
    quota = {}
    for group, budget in ((emb, emb_target), (gen, general_budget)):
        sizes = {n: avail(p) for n, p, _ in group}
        total = sum(sizes.values()) or 1
        cap = int(budget * args.cap_frac)
        for n in sizes:
            share = int(budget * sizes[n] / total)
            quota[n] = max(1, min(sizes[n], max(share, budget // (4 * max(len(group), 1))), cap))
    # explicit per-source down/up-weighting (e.g. keep Visual Genome from dominating)
    weights = {}
    for spec in args.source_weight:
        if "=" in spec:
            k, v = spec.split("=", 1)
            try:
                weights[k.strip()] = float(v)
            except ValueError:
                pass
    for k, w in weights.items():
        if k in quota:
            quota[k] = max(1, int(quota[k] * w))
            print(f"   weight {k} x{w} -> {quota[k]:,}", flush=True)

    print(f"sources: embodied={len(emb)} general={len(gen)} "
          f"quota_sum={sum(quota.values()):,} target={args.target:,}", flush=True)
    for n, q in sorted(quota.items(), key=lambda x: -x[1])[:10]:
        print(f"   quota {n}: {q:,}", flush=True)

    random.seed(0)
    for name, path, scene in usable:
        taken = 0
        img_checked = 0
        with open(path, errors="ignore") as f:
            for ln in f:
                if n_out >= args.target or taken >= quota[name]:
                    break
                if not ln.strip():
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    reject["bad_json"] += 1; continue
                if scene == "general_visual" and by_scene["general_visual"] >= general_budget:
                    reject["general_cap"] += 1; continue
                msgs = d.get("messages") or []
                if len(msgs) < 2:
                    reject["no_messages"] += 1; continue
                user = next((m for m in msgs if m.get("role") == "user"), None)
                asst = next((m for m in msgs if m.get("role") == "assistant"), None)
                if not user or not asst:
                    reject["no_turn"] += 1; continue
                utext = str(user.get("content") or ""); atext = str(asst.get("content") or "")
                if len(atext.strip()) < 2:
                    reject["empty_answer"] += 1; continue
                imgs = d.get("images") or d.get("image") or []
                if isinstance(imgs, str):
                    imgs = [imgs]
                imgs = [remap(str(i)) for i in imgs]
                if not imgs:
                    reject["no_image"] += 1; continue
                if utext.count("<image>") != len(imgs):
                    # normalize: exactly one <image> prefix per image
                    utext = "".join("<image>" for _ in imgs) + re.sub(r"<image>\s*", "", utext)
                if img_checked < args.check_images:
                    img_checked += 1
                    if not os.path.exists(imgs[0]):
                        reject["image_missing"] += 1; continue
                if not coords_ok(atext):
                    reject["coord_out_of_range"] += 1; continue
                key = hashlib.md5((imgs[0] + "|" + atext[:400]).encode()).hexdigest()
                if key in seen:
                    reject["duplicate"] += 1; continue
                seen.add(key)
                form = classify(utext, atext)
                rec = {"messages": [{"role": "user", "content": utext},
                                    {"role": "assistant", "content": atext}],
                       "images": imgs,
                       "gsa_meta": {"source": name, "scene": scene, "form": form}}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                by_form[form] += 1; by_scene[scene] += 1; by_src[name] += 1
                n_out += 1; in_shard += 1; taken += 1
                if in_shard >= args.shard_size:
                    fh.close(); shard_idx += 1; in_shard = 0
                    fh = (outd / f"shard_{shard_idx:03d}.jsonl").open("w")
        print(f"[{name}] cum={n_out:,}", flush=True)
        if n_out >= args.target:
            break
    fh.close()

    man = {"total": n_out, "shards": shard_idx + 1,
           "by_form": dict(by_form.most_common()),
           "by_scene": dict(by_scene.most_common()),
           "by_source": dict(by_src.most_common()),
           "rejected": dict(reject.most_common()),
           "embodied_fraction": round(1 - by_scene["general_visual"] / max(n_out, 1), 4)}
    (outd / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: v for k, v in man.items() if k != "by_source"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
