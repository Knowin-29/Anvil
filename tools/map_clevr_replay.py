#!/usr/bin/env python3
"""Recover CLEVR QA identities from official annotations; export train-only replay.

Local conversation order/image indices are not official IDs. Match the complete
question/answer multiset to one official training image, then preserve the local
image association with explicit byte/RGB fingerprints. This does not claim that
transcoded image bytes equal the original official PNG. QA remains auxiliary
replay, not a newly generated native spatial form or a human-verifier result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.arxiv_source_scale import Fingerprints, fingerprint, sha
from trains.flywheel.native_training import canonical, strict_json

VERSION = "anvil-clevr-official-qa-replay-v1"
QUESTION_MEMBER = "CLEVR_v1.0/questions/CLEVR_train_questions.json"
OFFICIAL_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip"
SUFFIXES = {
    "Offer a very short reply.", "Your answer should be compact.",
    "Your answer should be very brief.", "Ensure brevity in your answer.",
    "Provide a short and direct response.", "Your response must be concise.",
    "Keep it short and to the point.", "Make the answer very short.",
    "Provide a succinct answer.", "Give a very brief answer.", "Offer a terse response.",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def bag_key(pairs):
    return hashlib.sha256(canonical(sorted(pairs)).encode()).hexdigest()


def local_pairs(row):
    messages = row.get("messages")
    require(isinstance(messages, list) and messages and len(messages) % 2 == 0,
            "source must contain complete QA pairs")
    pairs = []
    for index in range(0, len(messages), 2):
        user, assistant = messages[index:index + 2]
        require(isinstance(user, dict) and isinstance(assistant, dict)
                and user.get("role") == "user" and assistant.get("role") == "assistant",
                "nonalternating source conversation")
        question, answer = user.get("content"), assistant.get("content")
        require(isinstance(question, str) and isinstance(answer, str), "nontext QA")
        if index == 0:
            require(question.startswith("<image>\n"), "missing first image placeholder")
            question = question[len("<image>\n"):]
        question, separator, suffix = question.rpartition("\n")
        require(separator and suffix in SUFFIXES and question and "<image>" not in question,
                "unknown converted question wrapper")
        require(answer.endswith(".") and answer[:-1], "unknown converted answer wrapper")
        normalized = answer[:-1].lower()
        require(answer == normalized.capitalize() + ".", "noncanonical converted answer")
        pairs.append((question, normalized))
    return pairs


def official_index(payload):
    groups = defaultdict(list)
    ids = set()
    for source in payload["questions"]:
        image_index, question_index = source.get("image_index"), source.get("question_index")
        require(type(image_index) is int and image_index >= 0
                and type(question_index) is int and question_index >= 0, "invalid official IDs")
        require(source.get("split") == "train" and source.get("image_filename") ==
                f"CLEVR_train_{image_index:06d}.png", "official nontraining or inconsistent image ID")
        require(question_index not in ids, "duplicate official question ID")
        ids.add(question_index)
        require(isinstance(source.get("question"), str) and source["question"]
                and isinstance(source.get("answer"), str) and source["answer"], "invalid official QA")
        groups[image_index].append({key: source[key] for key in
                                  ("question_index", "image_index", "image_filename", "question", "answer")})
    index = defaultdict(list)
    for image_index, questions in groups.items():
        key = bag_key([(q["question"], q["answer"]) for q in questions])
        index[key].append(questions)
    return index, len(groups), len(ids)


def match_official(row, index):
    pairs = local_pairs(row)
    matches = index.get(bag_key(pairs), [])
    require(len(matches) == 1, "unmatched or ambiguous complete official QA group")
    questions = matches[0]
    require(Counter(pairs) == Counter((q["question"], q["answer"]) for q in questions),
            "question multiset mismatch")
    return questions


def source_image(row, image_root):
    require(isinstance(row.get("id"), str) and re.fullmatch(r"clevr_[0-9]{6}", row["id"]),
            "unexpected local image identity")
    images = row.get("images")
    require(isinstance(images, list) and len(images) == 1 and isinstance(images[0], str),
            "one local source image required")
    name = row["id"] + ".jpg"
    require(Path(images[0]).name == name, "local image and record ID mismatch")
    path = (image_root / name).resolve(strict=True)
    require(path.is_relative_to(image_root.resolve()), "local image escapes source root")
    return path


def read_official(path, expected):
    require(sha(path) == expected, "official archive SHA256 mismatch")
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist().count(QUESTION_MEMBER) == 1, "missing/duplicate official train member")
        with archive.open(QUESTION_MEMBER) as member:
            member_sha = hashlib.file_digest(member, "sha256").hexdigest()
        with archive.open(QUESTION_MEMBER) as member:
            payload = json.load(member)
    index, images, questions = official_index(payload)
    return index, {"archive_sha256": expected, "member": QUESTION_MEMBER,
                   "member_sha256": member_sha, "images": images, "questions": questions,
                   "url": OFFICIAL_URL}


def protected_image_path(image, remaps):
    original = Path(image)
    if original.is_file():
        return original.resolve()
    candidates = []
    for old, new in remaps:
        if original.is_relative_to(old):
            root = new.resolve(strict=True)
            candidate = (root / original.relative_to(old)).resolve(strict=True)
            require(candidate.is_relative_to(root) and candidate.is_file(), "protected image remap escapes root")
            candidates.append(candidate)
    require(len(set(candidates)) == 1, "missing or ambiguous protected image path")
    return candidates[0]


def add_protection(args):
    protected = Fingerprints()
    bindings = []
    manifest = json.loads(args.pool_manifest.read_text())
    require(sha(args.pool_index) == manifest["selection_sqlite_sha256"], "pool index SHA mismatch")
    db = sqlite3.connect(args.pool_index.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    for raw, decoded, phash, dhash in db.execute(
            "SELECT file_sha256,decoded_sha256,phash64,dhash64 FROM images WHERE split='benchmark'"):
        protected.add({"sha256": raw, "decoded_rgb_sha256": decoded, "phash64": phash, "dhash64": dhash})
    db.close()
    bindings.append({"kind": "fixed_pool_benchmark", "path": str(args.pool_manifest.resolve()),
                     "sha256": sha(args.pool_manifest), "index_sha256": manifest["selection_sqlite_sha256"]})
    for path in args.protected_fingerprints:
        with path.open() as stream:
            for line in stream:
                protected.add(json.loads(line))
        bindings.append({"kind": "protected_fingerprints", "path": str(path.resolve()), "sha256": sha(path)})
    seen = set()
    for path in args.protected_data:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                images = row.get("images") or [row.get("image_path")]
                require(images and all(isinstance(p, str) and p for p in images),
                        "protected record missing image path")
                for image in images:
                    if image not in seen:
                        resolved = protected_image_path(image, args.protected_image_prefix)
                        protected.add(fingerprint(resolved)); seen.add(image)
        bindings.append({"kind": "protected_dataset_images", "path": str(path.resolve()), "sha256": sha(path)})
    return protected, bindings


def replay_record(source, question, fp, image, source_line, line_sha, source_sha, official):
    return {"id": "clevr-qa:" + str(question["question_index"]), "images": [str(image)],
            "messages": [{"role": "user", "content": "<image>\n" + question["question"]},
                         {"role": "assistant", "content": question["answer"]}],
            "anvil_meta": {"schema": VERSION, "record_type": "qa_replay", "source": "CLEVR",
                "split": "train", "training_approved": True, "group_unit": "official_image",
                "group_id": "CLEVR:train:" + str(question["image_index"]),
                "official_question_id": question["question_index"], "official_image_id": question["image_index"],
                "official_image_filename": question["image_filename"], "official_questions_sha256": official["member_sha256"],
                "official_archive_sha256": official["archive_sha256"], "source_sha256": source_sha,
                "source_line": source_line, "source_line_sha256": line_sha, "source_record_id": source["id"],
                "image_sha256": fp["sha256"], "decoded_rgb_sha256": fp["decoded_rgb_sha256"],
                "verification": "unique_complete_official_train_QA_multiset_and_local_image_fingerprint",
                "independent_question": True, "official_original_image_bytes_verified": False,
                "human_reviewed": False, "strong_model_verified": False, "model_calls": 0,
                "public_release_eligible": False, "rights": "local_only_pending_image_provenance_release_audit"}}


def build(args):
    start = time.monotonic()
    code_sha = sha(Path(__file__))
    require(not args.output.exists(), "output must be a new directory")
    require(sha(args.source) == args.source_sha256, "converted source SHA256 mismatch")
    protected, protections = add_protection(args)
    print(canonical({"phase": "historical_protection_loaded", "input_files": len(protections)}), flush=True)
    index, official = read_official(args.official_zip, args.official_sha256)
    print(canonical({"phase": "official_index_loaded", "images": official["images"],
                     "questions": official["questions"]}), flush=True)
    args.output.mkdir(parents=True)
    counts = Counter(); seen_ids = set(); seen_images = set()
    rows = []
    with args.source.open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            row = strict_json(raw)
            rows.append((line_no, hashlib.sha256(raw).hexdigest(), row))
    def prepare(item):
        line_no, line_sha, source = item
        questions = match_official(source, index)
        image = source_image(source, args.image_root)
        fp = fingerprint(image)
        require(source.get("width") == [fp["width"]] and source.get("height") == [fp["height"]],
                "local image dimensions disagree with source")
        return line_no, line_sha, source, questions, image, fp
    candidate = args.output / "train_candidates.jsonl"
    image_file = args.output / "images.jsonl"
    with candidate.open("x") as out, image_file.open("x") as images, \
            ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Bound queued work: all records are matched; image decoding stays bounded.
        for offset in range(0, len(rows), 512):
            for line_no, line_sha, source, questions, image, fp in executor.map(prepare, rows[offset:offset + 512]):
                counts["source_images_checked"] += 1
                if protected.matches(fp):
                    counts["protected_images"] += 1; counts["protected_qa"] += len(questions)
                    continue
                require(fp["decoded_rgb_sha256"] not in seen_images, "duplicate decoded training image")
                seen_images.add(fp["decoded_rgb_sha256"])
                for question in questions:
                    require(question["question_index"] not in seen_ids, "duplicate official question identity")
                    seen_ids.add(question["question_index"])
                    record = replay_record(source, question, fp, image, line_no, line_sha, args.source_sha256, official)
                    out.write(canonical(record) + "\n"); counts["train_candidate_qa"] += 1
                images.write(canonical({**fp, "path": str(image), "split": "train", "qa_count": len(questions),
                    "source": "CLEVR", "official_image_id": questions[0]["image_index"],
                    "group_id": "CLEVR:train:" + str(questions[0]["image_index"])}) + "\n")
                counts["train_candidate_images"] += 1
            out.flush(); images.flush()
            print(canonical({"phase": "official_match_and_protection", **counts}), flush=True)
    require(sha(args.source) == args.source_sha256 and sha(args.official_zip) == args.official_sha256,
            "source changed while converting")
    require(sha(Path(__file__)) == code_sha, "converter changed while running")
    for protection in protections:
        require(sha(protection["path"]) == protection["sha256"], "protection input changed while converting")
    require(sha(args.pool_index) == protections[0]["index_sha256"], "fixed pool changed while converting")
    result = {"schema": VERSION, "state": "official_QA_matched_local_image_candidates",
              "counts": dict(counts), "official": official,
              "source": {"path": str(args.source.resolve()), "sha256": args.source_sha256},
              "protections": protections,
              "protected_image_prefix_remaps": [[str(a), str(b)] for a, b in args.protected_image_prefix],
              "files": {p.name: sha(p) for p in (candidate, image_file)},
              "qa_replay_not_native_form": True, "paper_pool_complete": False,
              "official_original_image_bytes_verified": False, "public_release_eligible": False,
              "model_calls": 0, "new_human_reviews": 0, "seconds": time.monotonic() - start}
    result["converter_sha256"] = code_sha
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "official-zip", "image-root", "pool-index", "pool-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha256", "official-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protected-fingerprints", type=Path, action="append", default=[])
    parser.add_argument("--protected-data", type=Path, action="append", default=[])
    parser.add_argument("--protected-image-prefix", type=Path, nargs=2, action="append", default=[],
                        metavar=("OLD_ROOT", "LOCAL_ROOT"), help="Explicit relocation of missing historical image paths")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require(1 <= args.workers <= 32, "worker count outside 1..32")
    require(args.protected_fingerprints and args.protected_data, "historical protection inputs required")
    print(canonical(build(args)), flush=True)


if __name__ == "__main__":
    main()
