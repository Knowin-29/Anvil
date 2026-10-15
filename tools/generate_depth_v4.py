#!/usr/bin/env python3
"""Generate corrected depth-v4 labels from frozen v3.4 grounding records.

This tool makes external requests only after the guarded launcher has obtained
explicit upload authorization. A generator must justify every adjacent depth
pair with visible evidence, then an independent falsifier must fail to find a
counterexample. Only labels passing both deterministic validators are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trains.flywheel.build_depth_semantic_audit_pack import (  # noqa: E402
    content_hash,
    ensure_empty_output,
    file_hash,
    load_exclusions,
    normalized_path,
    stable_key,
)
from trains.flywheel import prompts, prompts_ext  # noqa: E402
from trains.flywheel.contracts import grounding  # noqa: E402
from trains.flywheel.validators import validate, validate_depth_v4_record  # noqa: E402

PROMPT_VERSION = "gsa-depth-v4.1"
LABEL_VERSION = "gsa-label-v4-depth"
TARGET = 40
MIN_CONFIDENCE = 0.85
ALLOWED_CUES = {"occlusion", "contact_layering", "support_plane_perspective"}
BANNED_EVIDENCE = (
    "left of", "right of", "higher in the image", "lower in the image",
    "larger", "smaller", "image height", "looks bigger", "looks smaller",
)


def _read_jsonl(path: Path, form: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict) or row.get("task_form") != form:
            raise SystemExit(f"{path}:{line_number}: expected task_form={form}")
        rows.append(row)
    return rows


def _bbox(value: Any) -> list[int] | None:
    if (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and 0 <= value[0] < value[2] <= 1000
        and 0 <= value[1] < value[3] <= 1000
    ):
        return list(value)
    return None


def _entity(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    label = item.get("label")
    box = _bbox(item.get("bbox_2d") or item.get("bbox"))
    if not isinstance(label, str) or not label.strip() or box is None:
        return None
    return {"label": label.strip(), "bbox_2d": box}


def load_candidates(
    detection_path: Path | None,
    bbox_path: Path | None,
    index_path: Path,
    exclusion_manifests: list[Path],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_paths, excluded_hashes, _, exclusion_provenance = load_exclusions(
        exclusion_manifests
    )
    detections = _read_jsonl(detection_path, "detection") if detection_path else []
    boxes = _read_jsonl(bbox_path, "bbox") if bbox_path else []
    bbox_by_path = {normalized_path(str(row.get("image_path", ""))): row for row in boxes}
    rejected: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for detection in detections:
        image = detection.get("image_path")
        if not isinstance(image, str) or not Path(image).is_file():
            rejected["missing_image"] += 1
            continue
        path_key = normalized_path(image)
        bbox_row = bbox_by_path.get(path_key)
        if bbox_row is None:
            rejected["missing_bbox_pair"] += 1
            continue
        image_sha = file_hash(Path(image))
        if path_key in excluded_paths:
            rejected["prior_path"] += 1
            continue
        if image_sha in excluded_hashes:
            rejected["prior_sha256"] += 1
            continue
        if image_sha in seen_hashes:
            rejected["duplicate_image_sha256"] += 1
            continue

        raw_entities = detection.get("payload")
        if not isinstance(raw_entities, list) or len(raw_entities) < 2:
            rejected["fewer_than_two_entities"] += 1
            continue
        entities = [_entity(item) for item in raw_entities[:4]]
        if any(entity is None for entity in entities):
            rejected["invalid_entity"] += 1
            continue
        exact_entities = [entity for entity in entities if entity is not None]
        labels = [entity["label"].casefold() for entity in exact_entities]
        if len(set(labels)) != len(labels):
            rejected["duplicate_entity_label"] += 1
            continue
        primary = _entity((bbox_row.get("payload") or [None])[0])
        if (
            primary is None
            or primary["label"] != exact_entities[0]["label"]
            or primary["bbox_2d"] != exact_entities[0]["bbox_2d"]
        ):
            rejected["bbox_detection_mismatch"] += 1
            continue
        seen_hashes.add(image_sha)
        candidates.append({
            "image_path": image,
            "image_sha256": image_sha,
            "entities": exact_entities,
            "detection": detection,
            "bbox": bbox_row,
            "scene_type": detection.get("scene_type", "unknown"),
        })

    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split("\t", 1)
        image = parts[0]
        scene = parts[1].strip() if len(parts) == 2 else "unknown"
        image_path = Path(image)
        if not image_path.is_file():
            rejected["index_missing_image"] += 1
            continue
        path_key = normalized_path(image)
        image_sha = file_hash(image_path)
        if path_key in excluded_paths:
            rejected["index_prior_path"] += 1
            continue
        if image_sha in excluded_hashes:
            rejected["index_prior_sha256"] += 1
            continue
        if image_sha in seen_hashes:
            continue
        seen_hashes.add(image_sha)
        candidates.append({
            "image_path": image,
            "image_sha256": image_sha,
            "entities": None,
            "detection": None,
            "bbox": None,
            "scene_type": scene,
            "index_line": line_number,
        })

    candidates.sort(key=lambda row: stable_key(seed, row["image_sha256"]))
    provenance = {
        "index_path": str(index_path.resolve()),
        "index_sha256": file_hash(index_path),
        "detection_path": str(detection_path.resolve()) if detection_path else None,
        "detection_sha256": file_hash(detection_path) if detection_path else None,
        "bbox_path": str(bbox_path.resolve()) if bbox_path else None,
        "bbox_sha256": file_hash(bbox_path) if bbox_path else None,
        "exclusion_manifests": exclusion_provenance,
        "rejected_candidates": dict(sorted(rejected.items())),
    }
    return candidates, provenance


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


def _source_bucket(image_path: str) -> str:
    name = Path(image_path).name
    if name.startswith("emb-"):
        return "bridge"
    if name.startswith("kit-"):
        return "kitchen"
    if name.startswith("exp-"):
        return "navigation"
    if name.startswith("tab-"):
        return "tabletop"
    return "unknown"


def ground_candidate(
    candidate: dict[str, Any],
    client: Callable[..., Any],
) -> tuple[dict[str, Any] | None, str]:
    if candidate.get("entities") is not None:
        return candidate, ""
    image = candidate["image_path"]
    last_reason = "grounding_failed"
    for attempt in range(2):
        response = client(
            image,
            prompts_ext.DETECTION,
            max_tokens=700,
            temperature=0.1 + 0.1 * attempt,
            use_cache=(attempt == 0),
        )
        raw = _as_list(response)
        valid, reason = validate("detection", raw, None)
        if not valid:
            last_reason = f"detection:{reason}"
            continue
        proposed = [
            grounding(item["label"], item.get("bbox_2d") or item.get("bbox"))
            for item in raw
        ]
        checked = _as_list(client(
            image,
            prompts.cross_check_set(proposed),
            max_tokens=420,
            temperature=0.0,
            use_cache=(attempt == 0),
        ))
        by_index = {
            item.get("index"): item
            for item in checked
            if isinstance(item.get("index"), int)
        }
        keep = [
            index
            for index in range(len(raw))
            if (
                index in by_index
                and by_index[index].get("match") is True
                and _confidence(by_index[index].get("confidence"))
            )
        ][:4]
        selected = [raw[index] for index in keep]
        valid, reason = validate("detection", selected, None)
        if not valid or not 2 <= len(selected) <= 4:
            last_reason = f"cross_check:{reason or 'entity_cardinality'}"
            continue
        entities = [_entity(item) for item in selected]
        if any(item is None for item in entities):
            last_reason = "cross_check:invalid_entity"
            continue
        exact_entities = [item for item in entities if item is not None]
        groundings = [
            {"object_class": item["label"], "bbox": item["bbox_2d"]}
            for item in exact_entities
        ]
        identity = candidate["image_sha256"][:20]
        source_bucket = _source_bucket(image)
        common = {
            "pipeline": "G_grounding",
            "dimension": "G_grounding",
            "image_path": image,
            "source": "upload_review_500_v2",
            "dataset": "upload_review_500_v2",
            "scene_type": candidate["scene_type"],
            "source_bucket": source_bucket,
            "object_class": groundings[0]["object_class"],
            "confidence": min(float(by_index[index]["confidence"]) for index in keep),
            "label_version": LABEL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "gsa_context": {
                "contract_version": "gsa-contract-v2",
                "primary": groundings[0],
                "groundings": groundings,
                "spatial": {},
            },
            "bbox": groundings[0]["bbox"],
            "_cross_check": [by_index[index] for index in keep],
        }
        detection = {
            **common,
            "sample_id": f"v4_detection_{identity}",
            "task_form": "detection",
            "payload": selected,
        }
        bbox = {
            **common,
            "sample_id": f"v4_bbox_{identity}",
            "task_form": "bbox",
            "payload": [selected[0]],
        }
        return {
            **candidate,
            "entities": exact_entities,
            "detection": detection,
            "bbox": bbox,
        }, ""
    return None, last_reason


def generator_prompt(entities: list[dict[str, Any]]) -> str:
    return (
        "You are the primary camera-depth annotator. The listed entities and boxes "
        "are frozen; use every exact label once and never add, rename, merge, or split "
        f"an entity. Entities: {json.dumps(entities, ensure_ascii=False)}. Decide first "
        "whether all entities can be strictly ordered by physical camera z-depth. "
        "Orderability requires a visible cue for EVERY adjacent pair: direct occlusion, "
        "contact/layering, or support-plane perspective. Image height, apparent size, "
        "and left/right position alone are invalid. Ties or ambiguity make the whole "
        "case unorderable. If unorderable, return exactly "
        '{"orderable":false,"reason_code":"ambiguous_pair","confidence":0.0,'
        '"order":[],"adjacent_evidence":[]}. If orderable, return exactly one JSON '
        "object with orderable=true, reason_code=supported, confidence in [0,1], "
        '"order":[{"label":"<exact label>","rank":1,"relative_depth":0.0},...], '
        "using contiguous ranks and strictly increasing depths with first=0.0 and "
        'last=1.0, plus "adjacent_evidence":[{"nearer":"<exact label>",'
        '"farther":"<exact label>","cue_type":"occlusion|contact_layering|'
        'support_plane_perspective","evidence":"a concrete visible observation naming '
        'both exact labels"},...]. Return JSON only.'
    )


def falsifier_prompt(
    entities: list[dict[str, Any]], proposal: dict[str, Any]
) -> str:
    return (
        "Act as an independent falsifier, not a confirmer. Re-read the image and try "
        "to disprove each adjacent nearest-to-farthest pair. Frozen entities: "
        f"{json.dumps(entities, ensure_ascii=False)}. Proposal: "
        f"{json.dumps(proposal, ensure_ascii=False)}. A pair is supported only by "
        "direct occlusion, contact/layering, or support-plane perspective. Reject "
        "apparent size, image height, and left/right position; reject a tie, ambiguous "
        "pair, reversed pair, wrong entity, or unsupported evidence. Use confidence in [0.85,1.0] for the overall result and every pair check. Return exactly "
        '{"valid":true,"reason_code":"no_counterexample","confidence":0.95,'
        '"pair_checks":[{"nearer":"<exact label>","farther":"<exact label>",'
        '"supported":true,"confidence":0.95,"counterevidence":""},...]} or valid=false '
        "with a specific reason and counterevidence. Include every adjacent pair once. "
        "Return JSON only."
    )


def _confidence(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and MIN_CONFIDENCE <= float(value) <= 1.0
    )


def validate_generator_response(
    response: Any, entities: list[dict[str, Any]]
) -> tuple[bool, str]:
    if not isinstance(response, dict):
        return False, "generator_not_object"
    if response.get("orderable") is not True:
        reason = response.get("reason_code")
        if isinstance(reason, str) and reason:
            return False, f"not_orderable:{reason}"
        return False, "generator_missing_orderability"
    if response.get("reason_code") != "supported" or not _confidence(response.get("confidence")):
        return False, "generator_gate"
    order = response.get("order")
    evidence = response.get("adjacent_evidence")
    if not isinstance(order, list) or len(order) != len(entities):
        return False, "order_cardinality"
    if not isinstance(evidence, list) or len(evidence) != len(entities) - 1:
        return False, "evidence_cardinality"

    expected_labels = [entity["label"] for entity in entities]
    labels: list[str] = []
    depths: list[float] = []
    for rank, item in enumerate(order, 1):
        if not isinstance(item, dict):
            return False, "order_item"
        label = item.get("label")
        depth = item.get("relative_depth")
        if (
            not isinstance(label, str)
            or item.get("rank") != rank
            or not isinstance(depth, (int, float))
            or isinstance(depth, bool)
            or not 0 <= float(depth) <= 1
        ):
            return False, "order_schema"
        labels.append(label)
        depths.append(float(depth))
    if len(set(labels)) != len(labels) or set(labels) != set(expected_labels):
        return False, "labels_not_exact"
    if depths[0] != 0.0 or depths[-1] != 1.0:
        return False, "depth_endpoints"
    if any(left >= right for left, right in zip(depths, depths[1:])):
        return False, "depth_not_strict"

    expected_pairs = list(zip(labels, labels[1:]))
    for item, (nearer, farther) in zip(evidence, expected_pairs):
        if not isinstance(item, dict):
            return False, "evidence_item"
        text = item.get("evidence")
        if (
            item.get("nearer") != nearer
            or item.get("farther") != farther
            or item.get("cue_type") not in ALLOWED_CUES
            or not isinstance(text, str)
            or len(text.strip()) < 20
        ):
            return False, "evidence_schema"
        folded = text.casefold()
        if nearer.casefold() not in folded or farther.casefold() not in folded:
            return False, "evidence_missing_exact_labels"
        if any(term in folded for term in BANNED_EVIDENCE):
            return False, "unsupported_depth_heuristic"
    return True, ""


def validate_falsifier_response(
    response: Any, order: list[dict[str, Any]]
) -> tuple[bool, str]:
    if not isinstance(response, dict):
        return False, "falsifier_not_object"
    if (
        response.get("valid") is not True
        or response.get("reason_code") != "no_counterexample"
        or not _confidence(response.get("confidence"))
    ):
        return False, f"falsifier_rejected:{response.get('reason_code', 'invalid')}"
    checks = response.get("pair_checks")
    if not isinstance(checks, list) or len(checks) != len(order) - 1:
        return False, "falsifier_cardinality"
    labels = [item["label"] for item in order]
    for check, nearer, farther in zip(checks, labels, labels[1:]):
        if (
            not isinstance(check, dict)
            or check.get("nearer") != nearer
            or check.get("farther") != farther
            or check.get("supported") is not True
            or not _confidence(check.get("confidence"))
            or check.get("counterevidence") not in ("", None)
        ):
            return False, "falsifier_pair"
    return True, ""


def _build_record(
    candidate: dict[str, Any],
    generator: dict[str, Any],
    falsifier: dict[str, Any],
) -> dict[str, Any]:
    detection = candidate["detection"]
    entities = candidate["entities"]
    groundings = [
        {"object_class": entity["label"], "bbox": entity["bbox_2d"]}
        for entity in entities
    ]
    payload = generator["order"]
    identity = content_hash({
        "image_sha256": candidate["image_sha256"],
        "payload": payload,
        "prompt_version": PROMPT_VERSION,
    })
    context = {
        "contract_version": "gsa-contract-v2",
        "primary": groundings[0],
        "groundings": groundings,
        "spatial": {
            "depth_order": payload,
            "adjacent_depth_evidence": generator["adjacent_evidence"],
        },
    }
    return {
        "sample_id": f"v4_depth_{identity[:20]}",
        "pipeline": "S_spatial",
        "dimension": "S_spatial",
        "task_form": "depth",
        "image_path": candidate["image_path"],
        "source": detection.get("source", "upload_review_500_v2"),
        "dataset": detection.get("dataset", "upload_review_500_v2"),
        "scene_type": detection.get("scene_type", "unknown"),
        "source_bucket": detection.get("source_bucket", "unknown"),
        "object_class": groundings[0]["object_class"],
        "confidence": min(float(generator["confidence"]), float(falsifier["confidence"])),
        "label_version": LABEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "payload": payload,
        "gsa_context": context,
        "bbox": groundings[0]["bbox"],
        "_v4_sources": {
            "detection_sample_id": detection.get("sample_id"),
            "detection_label_sha256": content_hash(detection),
            "bbox_sample_id": candidate["bbox"].get("sample_id"),
            "bbox_label_sha256": content_hash(candidate["bbox"]),
            "image_sha256": candidate["image_sha256"],
        },
        "_semantic_gate": {
            "depth_v4": {
                "generator": generator,
                "falsifier": falsifier,
            }
        },
        "_verification": {
            "contract_version": "gsa-contract-v2",
            "form": "depth",
            "family": "S",
            "signals": {
                "exact_entity_identity": 1.0,
                "adjacent_visible_evidence": 1.0,
                "independent_falsifier": 1.0,
            },
            "verdict": "verified",
        },
    }


def generate_one(
    candidate: dict[str, Any],
    client: Callable[..., Any],
) -> tuple[dict[str, Any] | None, str]:
    grounded, grounding_reason = ground_candidate(candidate, client)
    if grounded is None:
        return None, grounding_reason
    candidate = grounded
    entities = candidate["entities"]
    prompt = generator_prompt(entities)
    generator: dict[str, Any] | None = None
    last_reason = "generator_failed"
    for attempt in range(2):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                f" Your previous response failed deterministic validation ({last_reason}). "
                "Return a corrected response without weakening any evidence rule."
            )
        response = client(
            candidate["image_path"],
            attempt_prompt,
            max_tokens=900,
            temperature=0.1 + 0.1 * attempt,
            use_cache=(attempt == 0),
        )
        valid, reason = validate_generator_response(response, entities)
        if valid:
            generator = response
            break
        last_reason = reason
        if reason.startswith("not_orderable:"):
            return None, reason
    if generator is None:
        return None, last_reason

    falsifier = client(
        candidate["image_path"],
        falsifier_prompt(entities, generator),
        max_tokens=700,
        temperature=0.0,
        use_cache=True,
    )
    valid, reason = validate_falsifier_response(falsifier, generator["order"])
    if not valid:
        return None, reason
    record = _build_record(candidate, generator, falsifier)
    valid, reason = validate_depth_v4_record(record)
    if not valid:
        return None, f"record:{reason}"
    return record, ""


def run_generation(
    candidates: list[dict[str, Any]],
    provenance: dict[str, Any],
    out: Path,
    workers: int,
    client: Callable[..., Any],
    minimum_audit_images: int = TARGET,
    protocol_artifact: bool = True,
    stage: str = "full",
) -> tuple[dict[str, Any], int]:
    accepted_by_sha: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    attempted = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(candidates), workers):
            batch = candidates[start:start + workers]
            attempted += len(batch)
            futures = {
                executor.submit(generate_one, candidate, client): candidate
                for candidate in batch
            }
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    record, reason = future.result()
                except Exception as exc:  # fail one candidate, not the provenance run
                    rejected[f"runtime:{type(exc).__name__}"] += 1
                    continue
                if record is None:
                    rejected[reason or "unknown"] += 1
                    continue
                accepted_by_sha[candidate["image_sha256"]] = record

    accepted_pool = [
        accepted_by_sha[candidate["image_sha256"]]
        for candidate in candidates
        if candidate["image_sha256"] in accepted_by_sha
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "depth-v4-generation",
        "prompt_version": PROMPT_VERSION,
        "label_version": LABEL_VERSION,
        "stage": stage,
        "protocol_artifact": protocol_artifact,
        "artifact_kind": "generation_pool" if protocol_artifact else "canary_preview",
        "candidate_limit_images": 8 if stage == "canary" else None,
        "minimum_audit_images": minimum_audit_images,
        "candidate_images": len(candidates),
        "attempted_images": attempted,
        "accepted_pool_images": len(accepted_pool),
        "emitted_records": len(accepted_pool),
        "audit_pool_eligible": (
            protocol_artifact and len(accepted_pool) >= minimum_audit_images
        ),
        "rejections": dict(sorted(rejected.items())),
        "source_provenance": provenance,
        "prompt_hashes": {
            "generator_template_sha256": hashlib.sha256(
                generator_prompt([
                    {"label": "<entity-a>", "bbox_2d": [0, 0, 1, 1]},
                    {"label": "<entity-b>", "bbox_2d": [1, 1, 2, 2]},
                ]).encode("utf-8")
            ).hexdigest(),
            "falsifier_protocol_sha256": hashlib.sha256(
                falsifier_prompt([], {}).encode("utf-8")
            ).hexdigest(),
        },
        "accepted_pool": [
            {
                "image_path": record["image_path"],
                "image_sha256": record["_v4_sources"]["image_sha256"],
                "sample_id": record["sample_id"],
                "label_sha256": content_hash(record),
            }
            for record in accepted_pool
        ],
    }
    label_path = out / "labels/S_spatial/depth.jsonl"
    label_path.parent.mkdir(parents=True, exist_ok=False)
    label_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in accepted_pool
        ),
        encoding="utf-8",
    )
    report["labels_sha256"] = file_hash(label_path)
    report["accepted_pool_sha256"] = content_hash(report["accepted_pool"])
    (out / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report, (
        2
        if protocol_artifact and len(accepted_pool) < minimum_audit_images
        else 0
    )


def append_stage_event(stage: str, event: str, **fields: Any) -> None:
    request_log = os.environ.get("GSA_VLM_REQUEST_LOG")
    if not request_log:
        return
    path = Path(request_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "depth_v4_stage": stage,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--detection", type=Path)
    parser.add_argument("--bbox", type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("data/depth_v4/generated"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--canary", type=int, choices=(0, 8), default=0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target != TARGET:
        raise SystemExit(f"depth-v4 protocol requires --target {TARGET}")
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be in [1,16]")
    if (args.detection is None) != (args.bbox is None):
        raise SystemExit("--detection and --bbox must be supplied together")
    stage = "canary" if args.canary == 8 else "full"
    if not args.check_only:
        if os.environ.get("GSA_DEPTH_V4_EXTERNAL_UPLOAD_AUTHORIZED") != "1":
            raise SystemExit(
                "external upload is disabled; use the guarded depth-v4 launcher"
            )
        if (
            stage == "full"
            and os.environ.get("GSA_DEPTH_V4_CANARY_APPROVED") != "1"
        ):
            raise SystemExit("full depth-v4 generation requires canary approval")
    candidates, provenance = load_candidates(
        args.detection, args.bbox, args.index, args.exclude_manifest, args.seed
    )
    if args.canary:
        candidates = candidates[:args.canary]
    candidate_limit = args.canary or None
    plan = {
        "protocol": "depth-v4-generation",
        "stage": stage,
        "protocol_artifact": args.canary == 0,
        "artifact_kind": "generation_pool" if args.canary == 0 else "canary_preview",
        "candidate_limit_images": candidate_limit,
        "minimum_audit_images": args.target,
        "eligible_images": len(candidates),
        "cached_groundings": sum(candidate["entities"] is not None for candidate in candidates),
        "groundings_to_generate": sum(candidate["entities"] is None for candidate in candidates),
        "image_paths": [candidate["image_path"] for candidate in candidates],
        "grounding_prompt": prompts_ext.DETECTION,
        "generator_prompt_example": generator_prompt(candidates[0]["entities"])
        if candidates and candidates[0]["entities"] else None,
        "falsifier_prompt_protocol": falsifier_prompt([], {}),
        "source_provenance": provenance,
    }
    if args.check_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        if args.canary:
            return 0
        return 0 if len(candidates) >= args.target else 2
    ensure_empty_output(args.out)
    from trains.flywheel.vlm_client import chat_json

    append_stage_event(
        stage,
        "stage_start",
        minimum_audit_images=args.target,
        candidates=len(candidates),
    )
    report, return_code = run_generation(
        candidates,
        provenance,
        args.out,
        args.workers,
        chat_json,
        minimum_audit_images=args.target,
        protocol_artifact=args.canary == 0,
        stage=stage,
    )
    append_stage_event(
        stage,
        "stage_end",
        return_code=return_code,
        attempted_images=report["attempted_images"],
        accepted_pool_images=report["accepted_pool_images"],
        audit_pool_eligible=report["audit_pool_eligible"],
    )
    print(json.dumps(report, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
