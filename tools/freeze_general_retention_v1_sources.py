#!/usr/bin/env python3
"""Freeze primary sources and prove each evaluation TSV row maps back to them."""
from __future__ import annotations

import argparse
import ast
import base64
import binascii
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import string
import sys
from typing import Any, Iterable
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml


LETTERS = string.ascii_uppercase


class SourceError(RuntimeError):
    """Raised when a frozen source or primary-adapter crosswalk drifts."""


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hamming64(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _distance_summary(values: list[int]) -> dict[str, int] | None:
    if not values:
        return None
    ordered = sorted(values)

    def nearest(percentile: float) -> int:
        return ordered[round((len(ordered) - 1) * percentile)]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": nearest(0.50),
        "p90": nearest(0.90),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "max": ordered[-1],
    }


def _float_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)

    def nearest(percentile: float) -> float:
        return ordered[round((len(ordered) - 1) * percentile)]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": nearest(0.50),
        "p90": nearest(0.90),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "max": ordered[-1],
    }


def _pixel_gate_sensitivity(
    maes: list[float],
    correlations: list[float],
    gates: list[dict[str, float]],
) -> list[dict[str, Any]]:
    if len(maes) != len(correlations):
        raise SourceError("paired pixel-distance vectors have different lengths")
    total = len(maes)
    return [
        {
            **gate,
            "matched": sum(
                mae <= float(gate["max_normalized_rgb_mae"])
                and correlation >= float(gate["min_grayscale_correlation"])
                for mae, correlation in zip(maes, correlations, strict=True)
            ),
            "total": total,
        }
        for gate in gates
    ]


def _paired_threshold_sensitivity(
    phash_distances: list[int],
    dhash_distances: list[int],
    thresholds: tuple[int, ...] = (0, 2, 4, 8, 16, 32),
) -> list[dict[str, Any]]:
    if len(phash_distances) != len(dhash_distances):
        raise SourceError("paired hash-distance vectors have different lengths")
    total = len(phash_distances)
    return [
        {
            "max_phash64_hamming": threshold,
            "max_dhash64_hamming": threshold,
            "matched": sum(
                phash <= threshold and dhash <= threshold
                for phash, dhash in zip(phash_distances, dhash_distances, strict=True)
            ),
            "total": total,
        }
        for threshold in thresholds
    ]


def _ai2d_adapter_index(image_id: str, question_suffix: int) -> str:
    return str(int(image_id) * 100 + question_suffix)


def _ai2d_punctuation_process(value: str) -> str:
    punctuation = [
        ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
        "_", "-", ">", "<", "@", "`", ",", "?", "!",
    ]
    comma_strip = re.compile(r"(\d)(,)(\d)")
    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    output = value
    for mark in punctuation:
        if mark + " " in value or " " + mark in value or comma_strip.search(value):
            output = output.replace(mark, "")
        else:
            output = output.replace(mark, " ")
    return period_strip.sub("", output)


def _ai2d_choice_is_character_like(value: str) -> bool:
    value = str(value)
    if value in ("{}", "Both", "None of above"):
        return True
    if value.startswith("Stage "):
        return _ai2d_choice_is_character_like(value[6:])
    if " and " in value:
        return all(_ai2d_choice_is_character_like(part) for part in value.split(" and "))
    if len(value) <= 2:
        return True
    words = _ai2d_punctuation_process(value).split()
    if len(words) > 1:
        return all(_ai2d_choice_is_character_like(word) for word in words)
    return False


def _ai2d_uses_labeled_derivative(
    abc_label: bool,
    choices: list[str],
    question: str,
) -> bool:
    if not abc_label:
        return False
    if sum(_ai2d_choice_is_character_like(choice) for choice in choices) >= 3:
        return True
    words = _ai2d_punctuation_process(question).split()
    return any(character in words for character in string.ascii_letters)


def verify_file(root: Path, entry: dict[str, Any], label: str) -> dict[str, Any]:
    path = root / entry["path"]
    if not path.is_file():
        raise SourceError(f"missing {label}: {path}")
    size = path.stat().st_size
    if size != int(entry["bytes"]):
        raise SourceError(f"{label} bytes {size} != {entry['bytes']}")
    observed = {"bytes": size, "sha256": file_hash(path, "sha256")}
    if observed["sha256"] != str(entry["sha256"]).lower():
        raise SourceError(f"{label} sha256 drift")
    if entry.get("md5"):
        observed["md5"] = file_hash(path, "md5")  # noqa: S324 - release identity
        if observed["md5"] != str(entry["md5"]).lower():
            raise SourceError(f"{label} md5 drift")
    return {**entry, **observed, "path": str(path.resolve())}


def _vlmeval_default_jpeg(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as opened:
        transport_image = opened.copy()
    if transport_image.mode not in ("1", "L", "RGB", "CMYK"):
        transport_image = transport_image.convert("RGB")
    output = io.BytesIO()
    transport_image.save(output, format="JPEG")
    return output.getvalue()


def _rgb64(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(
            image.resize((64, 64), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )


def _rgb64_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(left - right)) / 255.0)
    left_gray = np.mean(left, axis=2).reshape(-1)
    right_gray = np.mean(right, axis=2).reshape(-1)
    if float(np.std(left_gray)) == 0.0 or float(np.std(right_gray)) == 0.0:
        correlation = 1.0 if np.array_equal(left_gray, right_gray) else 0.0
    else:
        correlation = float(np.corrcoef(left_gray, right_gray)[0, 1])
    return mae, correlation


def _proper_ai2d_font_size(font_path: Path, width_height: tuple[int, int], text: str) -> int:
    font_size = 2
    while True:
        font = ImageFont.truetype(str(font_path), font_size)
        box = font.getbbox(text)
        rendered = (box[2] - box[0], box[3] - box[1])
        if rendered[0] > width_height[0] or rendered[1] > width_height[1]:
            return font_size
        font_size += 1


def _ai2d_labeled_replay(raw: bytes, annotation: dict[str, Any], font_path: Path) -> bytes:
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.copy()
    draw = ImageDraw.Draw(image)
    for item in annotation["text"].values():
        start = tuple(item["rectangle"][0])
        end = tuple(item["rectangle"][1])
        replacement = str(item["replacementText"])
        draw.rectangle((start, end), fill="white")
        font_size = _proper_ai2d_font_size(
            font_path,
            (end[0] - start[0], end[1] - start[1]),
            replacement,
        )
        font = ImageFont.truetype(str(font_path), font_size)
        box = font.getbbox(replacement)
        text_width, text_height = box[2] - box[0], box[3] - box[1]
        center_x, center_y = (start[0] + end[0]) // 2, start[1]
        draw.text(
            (center_x - text_width // 2, center_y - text_height // 2),
            replacement,
            font=font,
            fill="black",
        )
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _image_features(raw: bytes) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            image_format = str(opened.format or "unknown").lower()
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
    except (OSError, ValueError) as exc:
        raise SourceError(f"undecodable image: {exc}") from exc
    grayscale = image.convert("L")
    dh = np.asarray(grayscale.resize((9, 8), Image.Resampling.LANCZOS))
    dh_bits = (dh[:, :-1] > dh[:, 1:]).reshape(-1)
    dhash = sum(int(value) << index for index, value in enumerate(dh_bits))
    ph = np.asarray(grayscale.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(ph)[:8, :8].reshape(-1)
    threshold = float(np.median(low[1:]))
    phash = sum(int(value > threshold) << index for index, value in enumerate(low))
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "md5": hashlib.md5(raw).hexdigest(),  # noqa: S324 - content identity
        "decoded_rgb_sha256": hashlib.sha256(
            f"{image.width}x{image.height}:RGB:".encode("ascii") + image.tobytes()
        ).hexdigest(),
        "pillow_default_jpeg_sha256": hashlib.sha256(_vlmeval_default_jpeg(raw)).hexdigest(),
        "format": image_format,
        "width": image.width,
        "height": image.height,
        "phash64": f"{phash:016x}",
        "dhash64": f"{dhash:016x}",
    }


def _extract(
    raw: bytes,
    destination: Path,
    *,
    write: bool,
) -> dict[str, Any]:
    features = _image_features(raw)
    if write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if file_hash(destination, "sha256") != features["sha256"]:
                raise SourceError(f"existing extracted image drift: {destination}")
        else:
            partial = destination.with_name(destination.name + ".partial")
            with partial.open("wb") as handle:
                handle.write(raw)
            os.replace(partial, destination)
    return {**features, "path": str(destination.resolve())}


def _jsonl_payload(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
    os.replace(partial, path)


def _sample(
    *,
    dataset: str,
    uid: str,
    image_id: str,
    question: str,
    answer: Any,
    choices: list[str] | None,
    image: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "uid": uid,
        "image_id": image_id,
        "question": question,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "answer": answer,
        "choices": choices,
        "source_image_path": image["path"],
        "source_image_sha256": image["sha256"],
    }


def load_ai2d(
    root: Path,
    config: dict[str, Any],
    extraction_root: Path,
    *,
    write: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    archive = root / config["files"]["archive"]["path"]
    ids_path = root / config["files"]["test_ids"]["path"]
    test_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    if len(test_ids) != len(set(test_ids)):
        raise SourceError("AI2D test IDs are not unique")
    samples: list[dict[str, Any]] = []
    images: dict[str, dict[str, Any]] = {}
    missing_question_files: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        license_evidence = config["license_evidence"]
        license_raw = bundle.read(license_evidence["archive_member"])
        if len(license_raw) != int(license_evidence["bytes"]):
            raise SourceError("AI2D archive license bytes drift")
        if hashlib.sha256(license_raw).hexdigest() != license_evidence["sha256"]:
            raise SourceError("AI2D archive license sha256 drift")
        license_text = license_raw.decode("utf-8")
        if "License for Non-Commercial Use" not in license_text:
            raise SourceError("AI2D archive license text drift")
        for image_id in test_ids:
            question_member = f"ai2d/questions/{image_id}.png.json"
            if question_member not in names:
                missing_question_files.append(question_member)
                continue
            image_member = f"ai2d/images/{image_id}.png"
            if image_member not in names:
                raise SourceError(f"AI2D image missing for questions: {image_member}")
            raw_image = bundle.read(image_member)
            image = _extract(
                raw_image,
                extraction_root / "ai2d_test" / f"{image_id}.png",
                write=write,
            )
            image.update({"dataset": "ai2d_test", "image_id": image_id, "member": image_member})
            images[image_id] = image
            question_payload = json.loads(bundle.read(question_member))
            image["_ai2d_raw_replay_rgb64"] = _rgb64(_vlmeval_default_jpeg(raw_image))
            requires_labeled_replay = any(
                _ai2d_uses_labeled_derivative(
                    bool(item.get("abcLabel")),
                    [str(value) for value in item["answerTexts"]],
                    question,
                )
                for question, item in question_payload["questions"].items()
            )
            if requires_labeled_replay:
                annotation_member = f"ai2d/annotations/{image_id}.png.json"
                if annotation_member not in names:
                    raise SourceError(f"AI2D annotation missing: {annotation_member}")
                font_path = root / config["crosswalk_policy"]["ai2d_notebook_replay"]["font_path"]
                labeled_replay = _ai2d_labeled_replay(
                    raw_image,
                    json.loads(bundle.read(annotation_member)),
                    font_path,
                )
                image["_ai2d_labeled_replay_rgb64"] = _rgb64(labeled_replay)
            for question, item in question_payload["questions"].items():
                qid = str(item["questionId"])
                suffix = int(qid.rsplit("-", 1)[1])
                adapter_index = _ai2d_adapter_index(image_id, suffix)
                choices = [str(value) for value in item["answerTexts"]]
                answer = LETTERS[int(item["correctAnswer"])]
                uses_labeled_derivative = _ai2d_uses_labeled_derivative(
                    bool(item.get("abcLabel")),
                    choices,
                    question,
                )
                expected_adapter_image_path = (
                    f"ai2d/images_abc/{image_id}.png"
                    if uses_labeled_derivative
                    else image_member
                )
                samples.append(
                    {
                        **_sample(
                            dataset="ai2d_test",
                            uid=qid,
                            image_id=image_id,
                            question=question,
                            answer=answer,
                            choices=choices,
                            image=image,
                        ),
                        "adapter_index": adapter_index,
                        "primary_image_path": image_member,
                        "expected_adapter_image_path": expected_adapter_image_path,
                        "abc_label": bool(item.get("abcLabel")),
                        "upstream_is_abc": uses_labeled_derivative,
                    }
                )
    if len(samples) != int(config["expected_samples"]):
        raise SourceError(f"AI2D sample count {len(samples)} != {config['expected_samples']}")
    return samples, images, {
        "official_test_ids": len(test_ids),
        "test_ids_with_questions": len(images),
        "test_ids_without_question_file": len(missing_question_files),
        "samples": len(samples),
    }


def load_scienceqa(
    root: Path,
    config: dict[str, Any],
    extraction_root: Path,
    *,
    write: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    problems = json.loads((root / config["files"]["problems"]["path"]).read_text())
    splits = json.loads((root / config["files"]["pid_splits"]["path"]).read_text())
    if set(splits["test"]) != {qid for qid, item in problems.items() if item["split"] == "test"}:
        raise SourceError("ScienceQA problems.json and pid_splits.json test sets disagree")
    selected = [
        (qid, item)
        for qid, item in problems.items()
        if item["split"] == "test" and item.get("image") is not None
    ]
    samples: list[dict[str, Any]] = []
    images: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(root / config["files"]["images"]["path"]) as bundle:
        names = set(bundle.namelist())
        for position, (qid, item) in enumerate(selected, start=1):
            member = f"test/{qid}/{item['image']}"
            if member not in names:
                raise SourceError(f"ScienceQA test image missing: {member}")
            raw_image = bundle.read(member)
            suffix = Path(str(item["image"])).suffix.lower() or ".png"
            image = _extract(
                raw_image,
                extraction_root / "scienceqa_test" / f"{qid}{suffix}",
                write=write,
            )
            image.update({"dataset": "scienceqa_test", "image_id": qid, "member": member})
            images[qid] = image
            choices = [str(value) for value in item["choices"]]
            samples.append(
                {
                    **_sample(
                        dataset="scienceqa_test",
                        uid=qid,
                        image_id=qid,
                        question=str(item["question"]),
                        answer=LETTERS[int(item["answer"])],
                        choices=choices,
                        image=image,
                    ),
                    "adapter_index": str(position),
                }
            )
    if len(samples) != int(config["expected_samples"]):
        raise SourceError(
            f"ScienceQA image-test count {len(samples)} != {config['expected_samples']}"
        )
    return samples, images, {
        "official_test_all": len(splits["test"]),
        "official_test_image_subset": len(samples),
        "samples": len(samples),
    }


def load_textvqa(
    root: Path,
    config: dict[str, Any],
    extraction_root: Path,
    *,
    write: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads((root / config["files"]["annotations"]["path"]).read_text())
    if str(payload.get("dataset_version")) != str(config["version"]):
        raise SourceError("TextVQA dataset version drift")
    rows = payload["data"]
    samples: list[dict[str, Any]] = []
    images: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(root / config["files"]["images"]["path"]) as bundle:
        names = set(bundle.namelist())
        for item in rows:
            if item.get("set_name") != "val":
                raise SourceError("TextVQA annotation outside val split")
            image_id = str(item["image_id"])
            member = f"train_images/{image_id}.jpg"
            if image_id not in images:
                if member not in names:
                    raise SourceError(f"TextVQA val image missing: {member}")
                image = _extract(
                    bundle.read(member),
                    extraction_root / "textvqa_val" / f"{image_id}.jpg",
                    write=write,
                )
                image.update({"dataset": "textvqa_val", "image_id": image_id, "member": member})
                images[image_id] = image
            samples.append(
                {
                    **_sample(
                        dataset="textvqa_val",
                        uid=str(item["question_id"]),
                        image_id=image_id,
                        question=str(item["question"]),
                        answer=[str(value) for value in item["answers"]],
                        choices=None,
                        image=images[image_id],
                    ),
                    "adapter_index": str(item["question_id"]),
                }
            )
    if len(samples) != int(config["expected_samples"]):
        raise SourceError(f"TextVQA val count {len(samples)} != {config['expected_samples']}")
    return samples, images, {
        "official_val_questions": len(samples),
        "official_val_unique_images": len(images),
        "samples": len(samples),
    }


def _adapter_image(raw: str) -> tuple[bytes, dict[str, Any]]:
    try:
        decoded = base64.b64decode(raw, validate=True)
        return decoded, _image_features(decoded)
    except (ValueError, binascii.Error) as exc:
        raise SourceError(f"invalid adapter base64 image: {exc}") from exc


def verify_adapter(
    root: Path,
    dataset: str,
    entry: dict[str, Any],
    samples: list[dict[str, Any]],
    images: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    extraction_root: Path,
    *,
    write: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_index = {sample["adapter_index"]: sample for sample in samples}
    if len(by_index) != len(samples):
        raise SourceError(f"{dataset} primary adapter indices are not unique")
    path = root / entry["path"]
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    seen: set[str] = set()
    adapter_images: dict[str, dict[str, Any]] = {}
    adapter_primary_ids: dict[str, str] = {}
    adapter_rows_by_index: dict[str, tuple[bytes, dict[str, Any]]] = {}
    exact_byte_images: set[str] = set()
    transcoded_identity_images: set[str] = set()
    labeled_derivative_images: set[str] = set()
    transcode_phash_distances: list[int] = []
    transcode_dhash_distances: list[int] = []
    derivative_phash_distances: list[int] = []
    derivative_dhash_distances: list[int] = []
    raw_replay_maes: list[float] = []
    raw_replay_correlations: list[float] = []
    labeled_replay_maes: list[float] = []
    labeled_replay_correlations: list[float] = []
    adapter_image_rows: list[dict[str, Any]] = []
    image_reference_rows = 0
    row_mapping_digest = hashlib.sha256()
    transcode_policy = policy["exact_or_transcoded_image"]
    derivative_policy = policy["ai2d_labeled_derivative"]
    replay_policy = policy["ai2d_notebook_replay"]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            index = str(row.get("index", ""))
            if index in seen or index not in by_index:
                raise SourceError(f"{dataset} bad adapter index at TSV row {row_number}: {index}")
            seen.add(index)
            primary = by_index[index]
            if row["question"] != primary["question"]:
                raise SourceError(f"{dataset} question drift at index {index}")
            if dataset == "textvqa_val":
                try:
                    answers = [str(value) for value in ast.literal_eval(row["answer"])]
                except (ValueError, SyntaxError, TypeError) as exc:
                    raise SourceError(f"TextVQA answer parse failure at {index}: {exc}") from exc
                if answers != primary["answer"]:
                    raise SourceError(f"TextVQA answers drift at {index}")
            else:
                if row["answer"] != primary["answer"]:
                    raise SourceError(f"{dataset} answer drift at index {index}")
                for offset, expected in enumerate(primary["choices"] or []):
                    if row.get(LETTERS[offset], "") != expected:
                        raise SourceError(f"{dataset} choice drift at index {index}")
            if dataset == "ai2d_test":
                adapter_abc_label = row.get("abcLabel", "").casefold() == "true"
                if adapter_abc_label != primary["abc_label"]:
                    raise SourceError(f"AI2D abcLabel drift at index {index}")
                if row.get("image_path") != primary["expected_adapter_image_path"]:
                    raise SourceError(f"AI2D image_path drift at index {index}")

            image_id = primary["image_id"]
            adapter_image_id = row.get("image_path") or image_id
            image_reference_index: str | None = None
            if row["image"] in adapter_rows_by_index:
                image_reference_index = row["image"]
                image_reference_rows += 1
                adapter_raw, observed = adapter_rows_by_index[image_reference_index]
            else:
                adapter_raw, observed = _adapter_image(row["image"])
            adapter_rows_by_index[index] = (adapter_raw, observed)
            first_adapter_image = adapter_image_id not in adapter_images
            if not first_adapter_image:
                if observed["sha256"] != adapter_images[adapter_image_id]["sha256"]:
                    raise SourceError(
                        f"{dataset} repeated image bytes drift for {adapter_image_id}"
                    )
                if adapter_primary_ids[adapter_image_id] != image_id:
                    raise SourceError(
                        f"{dataset} adapter image maps to multiple primary IDs: "
                        f"{adapter_image_id}"
                    )
            adapter_images[adapter_image_id] = observed
            adapter_primary_ids[adapter_image_id] = image_id
            source = images[image_id]
            same_dimensions = (
                observed["width"] == source["width"]
                and observed["height"] == source["height"]
            )
            phash_distance = _hamming64(observed["phash64"], source["phash64"])
            dhash_distance = _hamming64(observed["dhash64"], source["dhash64"])
            is_labeled_derivative = (
                dataset == "ai2d_test"
                and adapter_image_id.startswith("ai2d/images_abc/")
            )
            replay_mae: float | None = None
            replay_correlation: float | None = None
            if dataset == "ai2d_test":
                replay_key = (
                    "_ai2d_labeled_replay_rgb64"
                    if is_labeled_derivative
                    else "_ai2d_raw_replay_rgb64"
                )
                replay_mae, replay_correlation = _rgb64_distance(
                    source[replay_key],
                    _rgb64(adapter_raw),
                )
                if (
                    replay_mae > float(replay_policy["max_normalized_rgb_mae"])
                    or replay_correlation
                    < float(replay_policy["min_grayscale_correlation"])
                ):
                    raise SourceError(
                        f"{dataset} notebook replay mismatch at {adapter_image_id}: "
                        f"mae={replay_mae:.6f}, correlation={replay_correlation:.6f}"
                    )
            if is_labeled_derivative:
                if not primary["abc_label"]:
                    raise SourceError(
                        f"{dataset} labeled derivative used by abcLabel=false row {index}"
                    )
                if derivative_policy["acceptance"] != "metadata_and_pinned_notebook_replay":
                    raise SourceError("unsupported AI2D derivative crosswalk policy")
                if derivative_policy["require_equal_dimensions"] and not same_dimensions:
                    raise SourceError(
                        f"{dataset} labeled derivative dimensions drift at {image_id}"
                    )
                labeled_derivative_images.add(adapter_image_id)
                if first_adapter_image:
                    derivative_phash_distances.append(phash_distance)
                    derivative_dhash_distances.append(dhash_distance)
                    labeled_replay_maes.append(float(replay_mae))
                    labeled_replay_correlations.append(float(replay_correlation))
                mapping_method = "pinned_notebook_labeled_derivative_replay"
            elif observed["sha256"] == source["sha256"]:
                exact_byte_images.add(adapter_image_id)
                mapping_method = "exact_bytes"
            elif (
                transcode_policy["acceptance"]
                == "exact_bytes_or_exact_pillow_default_jpeg"
                and (not transcode_policy["require_equal_dimensions"] or same_dimensions)
                and observed["sha256"] == source["pillow_default_jpeg_sha256"]
            ):
                transcoded_identity_images.add(adapter_image_id)
                if first_adapter_image:
                    transcode_phash_distances.append(phash_distance)
                    transcode_dhash_distances.append(dhash_distance)
                mapping_method = "exact_pillow_default_jpeg_transform"
            elif dataset == "ai2d_test":
                transcoded_identity_images.add(adapter_image_id)
                if first_adapter_image:
                    transcode_phash_distances.append(phash_distance)
                    transcode_dhash_distances.append(dhash_distance)
                    raw_replay_maes.append(float(replay_mae))
                    raw_replay_correlations.append(float(replay_correlation))
                mapping_method = "pinned_notebook_raw_codec_replay"
            else:
                raise SourceError(f"{dataset} adapter image does not map to primary image {image_id}")
            if first_adapter_image:
                suffix = {
                    "jpeg": ".jpg",
                    "png": ".png",
                    "webp": ".webp",
                    "gif": ".gif",
                }.get(observed["format"], ".img")
                destination = extraction_root / dataset / f"{observed['sha256']}{suffix}"
                extracted = _extract(adapter_raw, destination, write=write)
                adapter_image_rows.append(
                    {
                        **extracted,
                        "dataset": dataset,
                        "image_id": adapter_image_id,
                        "adapter_image_id": adapter_image_id,
                        "primary_image_id": image_id,
                        "primary_image_sha256": source["sha256"],
                        "evaluation_role": "adapter_transport",
                        "mapping_method": mapping_method,
                    }
                )
            row_binding = {
                "adapter_index": index,
                "primary_uid": primary["uid"],
                "primary_image_id": image_id,
                "adapter_image_id": adapter_image_id,
                "adapter_image_sha256": observed["sha256"],
                "primary_image_sha256": source["sha256"],
                "mapping_method": mapping_method,
                "primary_abc_label": primary.get("abc_label"),
                "upstream_is_abc": primary.get("upstream_is_abc"),
                "image_reference_index": image_reference_index,
            }
            row_mapping_digest.update(
                json.dumps(
                    row_binding,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
            )
    missing = sorted(set(by_index) - seen)
    if missing:
        raise SourceError(f"{dataset} adapter missing {len(missing)} primary samples")
    summary = {
        "dataset": dataset,
        "rows": len(seen),
        "unique_adapter_images": len(adapter_images),
        "adapter_image_reference_rows": image_reference_rows,
        "exact_byte_images": len(exact_byte_images),
        "deterministic_transcode_images": len(transcoded_identity_images),
        "pinned_notebook_labeled_derivatives": len(labeled_derivative_images),
        "transcode_distance_summary": {
            "phash64_hamming": _distance_summary(transcode_phash_distances),
            "dhash64_hamming": _distance_summary(transcode_dhash_distances),
        },
        "transcode_paired_threshold_sensitivity_descriptive_only": (
            _paired_threshold_sensitivity(
                transcode_phash_distances,
                transcode_dhash_distances,
            )
        ),
        "transcode_acceptance": transcode_policy,
        "labeled_derivative_distance_summary_descriptive_only": {
            "phash64_hamming": _distance_summary(derivative_phash_distances),
            "dhash64_hamming": _distance_summary(derivative_dhash_distances),
        },
        "labeled_derivative_paired_threshold_sensitivity_descriptive_only": (
            _paired_threshold_sensitivity(
                derivative_phash_distances,
                derivative_dhash_distances,
            )
        ),
        "labeled_derivative_acceptance": (
            derivative_policy if labeled_derivative_images else None
        ),
        "ai2d_notebook_replay_protocol": replay_policy if dataset == "ai2d_test" else None,
        "ai2d_raw_codec_replay_distance_summary": (
            {
                "normalized_rgb_mae": _float_summary(raw_replay_maes),
                "grayscale_correlation": _float_summary(raw_replay_correlations),
            }
            if dataset == "ai2d_test"
            else None
        ),
        "ai2d_raw_codec_replay_sensitivity": (
            _pixel_gate_sensitivity(
                raw_replay_maes,
                raw_replay_correlations,
                replay_policy["sensitivity_gates"],
            )
            if dataset == "ai2d_test"
            else None
        ),
        "ai2d_labeled_replay_distance_summary": (
            {
                "normalized_rgb_mae": _float_summary(labeled_replay_maes),
                "grayscale_correlation": _float_summary(labeled_replay_correlations),
            }
            if dataset == "ai2d_test"
            else None
        ),
        "ai2d_labeled_replay_sensitivity": (
            _pixel_gate_sensitivity(
                labeled_replay_maes,
                labeled_replay_correlations,
                replay_policy["sensitivity_gates"],
            )
            if dataset == "ai2d_test"
            else None
        ),
        "adapter_file_sha256": entry["sha256"],
        "row_mapping_sha256": row_mapping_digest.hexdigest(),
        "all_rows_primary_mapped": True,
    }
    return summary, adapter_image_rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/data/general_retention_v1_sources.yaml",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != 1:
        raise SourceError("unsupported source registry schema")

    verified_files: dict[str, Any] = {"datasets": {}, "adapter": {}}
    for dataset, item in config["datasets"].items():
        verified_files["datasets"][dataset] = {
            role: verify_file(root, entry, f"{dataset}.{role}")
            for role, entry in item["files"].items()
        }
    textvqa_license_evidence = config["datasets"]["textvqa_val"]["license_evidence"]
    textvqa_bundle = root / config["datasets"]["textvqa_val"]["files"][
        textvqa_license_evidence["file_role"]
    ]["path"]
    if textvqa_license_evidence["required_text"] not in textvqa_bundle.read_text(
        encoding="utf-8"
    ):
        raise SourceError("TextVQA official license declaration drift")
    for dataset, entry in config["adapter"]["files"].items():
        verified_files["adapter"][dataset] = verify_file(root, entry, f"adapter.{dataset}")
    for role, entry in config["adapter"]["implementation_files"].items():
        verified_files["adapter"][role] = verify_file(
            root,
            {**entry, "bytes": (root / entry["path"]).stat().st_size},
            f"adapter_implementation.{role}",
        )

    extraction_root = root / config["outputs"]["extraction_root"]
    all_samples: list[dict[str, Any]] = []
    all_images: list[dict[str, Any]] = []
    dataset_samples: dict[str, list[dict[str, Any]]] = {}
    dataset_images: dict[str, dict[str, dict[str, Any]]] = {}
    dataset_stats: dict[str, Any] = {}
    loaders = {
        "ai2d_test": load_ai2d,
        "scienceqa_test": load_scienceqa,
        "textvqa_val": load_textvqa,
    }
    for dataset, loader in loaders.items():
        print(f"loading primary {dataset}", file=sys.stderr, flush=True)
        dataset_config = {
            **config["datasets"][dataset],
            "crosswalk_policy": config["adapter"]["crosswalk_policy"],
        }
        samples, images, stats = loader(
            root,
            dataset_config,
            extraction_root,
            write=not args.check_only,
        )
        dataset_samples[dataset] = samples
        dataset_images[dataset] = images
        dataset_stats[dataset] = stats
        all_samples.extend(samples)
        all_images.extend(images.values())

    crosswalks = []
    all_adapter_images: list[dict[str, Any]] = []
    adapter_extraction_root = root / config["outputs"]["adapter_extraction_root"]
    for dataset in loaders:
        print(f"crosswalking adapter {dataset}", file=sys.stderr, flush=True)
        crosswalk_summary, adapter_image_rows = verify_adapter(
            root,
            dataset,
            config["adapter"]["files"][dataset],
            dataset_samples[dataset],
            dataset_images[dataset],
            config["adapter"]["crosswalk_policy"],
            adapter_extraction_root,
            write=not args.check_only,
        )
        crosswalks.append(crosswalk_summary)
        all_adapter_images.extend(adapter_image_rows)

    sample_rows = sorted(all_samples, key=lambda row: (row["dataset"], row["uid"]))
    image_rows = sorted(
        (
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in all_images
        ),
        key=lambda row: (row["dataset"], row["image_id"]),
    )
    sample_payload = _jsonl_payload(sample_rows)
    image_payload = _jsonl_payload(image_rows)
    adapter_image_rows = sorted(
        all_adapter_images,
        key=lambda row: (row["dataset"], row["adapter_image_id"]),
    )
    adapter_image_payload = _jsonl_payload(adapter_image_rows)
    crosswalk = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "adapter_revision": config["adapter"]["revision"],
        "state": "PASS",
        "datasets": crosswalks,
    }
    crosswalk["identity_sha256"] = canonical_hash(crosswalk)
    source_manifest = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "state": "SOURCE_AND_ADAPTER_FROZEN",
        "formal_gpu_ready": False,
        "formal_gpu_blockers": [
            "source preparation does not provide the same-engine runner or official scorer",
            "formal GPU admission is delegated to scripts/preflight_general_specialist.py",
        ],
        "source_registry": str(args.config.resolve()),
        "source_registry_sha256": file_hash(args.config, "sha256"),
        "frozen_at_utc": config["frozen_at_utc"],
        "dataset_stats": dataset_stats,
        "total_samples": len(sample_rows),
        "total_unique_images": len(image_rows),
        "total_unique_adapter_images": len(adapter_image_rows),
        "verified_files": verified_files,
        "preparation_implementation": {
            name: {
                "path": str((root / relative).resolve()),
                "sha256": file_hash(root / relative, "sha256"),
            }
            for name, relative in {
                "source_freezer": "tools/freeze_general_retention_v1_sources.py",
                "bounded_downloader": "tools/download_bounded_range.py",
                "leakage_auditor": "tools/audit_general_retention_v1_leakage.py",
            }.items()
        },
        "sample_manifest_sha256": hashlib.sha256(sample_payload).hexdigest(),
        "image_manifest_sha256": hashlib.sha256(image_payload).hexdigest(),
        "adapter_image_manifest_sha256": hashlib.sha256(
            adapter_image_payload
        ).hexdigest(),
        "crosswalk_identity_sha256": crosswalk["identity_sha256"],
        "license_provenance": {
            key: {
                "license": value["license"],
                "official_page": value.get("official_page") or value.get("official_repo"),
                "revision": value.get("revision"),
                "evidence": value["license_evidence"],
            }
            for key, value in config["datasets"].items()
        },
    }
    source_manifest["identity_sha256"] = canonical_hash(source_manifest)

    if args.check_only:
        print(json.dumps({"source_manifest": source_manifest, "crosswalk": crosswalk}, indent=2))
        return 0
    _atomic_write(root / config["outputs"]["sample_manifest"], sample_payload)
    _atomic_write(root / config["outputs"]["image_manifest"], image_payload)
    _atomic_write(
        root / config["outputs"]["adapter_image_manifest"],
        adapter_image_payload,
    )
    _atomic_write(
        root / config["outputs"]["crosswalk_manifest"],
        (json.dumps(crosswalk, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(
        root / config["outputs"]["source_manifest"],
        (json.dumps(source_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "state": source_manifest["state"],
        "samples": len(sample_rows),
        "images": len(image_rows),
        "identity_sha256": source_manifest["identity_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
