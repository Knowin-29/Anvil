#!/usr/bin/env python3
"""Build an immutable, blind A/B depth-only semantic-audit pack."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trains.flywheel.validators import validate_depth_v4_record

FORM = "depth"
REQUIRED_IMAGES = 40
SCHEMA_VERSION = 4
TOTAL_ORDER_SPEC = {
    "depth_scope": "total_order",
    "prompt_version": "gsa-depth-v4.1",
    "label_version": "gsa-label-v4-depth",
    "protocol": "depth-only-calibration",
    "round": "v4-depth",
    "audit_id_prefix": "semantic-audit-v4-depth",
}
PAIRWISE_V42_SPEC = {
    "depth_scope": "pairwise",
    "prompt_version": "gsa-depth-v4.2-pairwise",
    "label_version": "gsa-label-v4-depth-pairwise",
    "protocol": "depth-pairwise-calibration-v1",
    "round": "v4-depth-pairwise",
    "audit_id_prefix": "semantic-audit-v4-depth-pairwise",
}
PAIRWISE_V43_SPEC = {
    "depth_scope": "pairwise",
    "prompt_version": "gsa-depth-v4.3-pairwise",
    "label_version": "gsa-label-v4.3-depth-pairwise",
    "protocol": "depth-v4.3-pairwise-semantic-audit-v1",
    "round": "v4.3-depth-pairwise",
    "audit_id_prefix": "semantic-audit-v43-depth-pairwise",
}
PAIRWISE_SPECS = (PAIRWISE_V42_SPEC, PAIRWISE_V43_SPEC)
PAIRWISE_CUES = ("occlusion", "contact_layering", "support_plane_perspective")
PAIRWISE_MIN_CUE_COUNTS = {"occlusion": 8, "contact_layering": 8}
PAIRWISE_GENERATION_CONTRACTS = {
    "gsa-depth-v4.2-pairwise": {
        "protocol": "depth-v4.2-pairwise-generation",
        "label_version": "gsa-label-v4-depth-pairwise",
        "depth_scope": "pairwise",
        "prompt_hashes": {
            "generator_template_sha256": (
                "b42e4a696d363b5f634842ec0c1a0cbceffe52ff642e60c3a18ca79922afa751"
            ),
            "falsifier_protocol_sha256": (
                "3fbf3161547157cc3db1fbd543e2180b4d3024573a9ef14831f5b323893d170e"
            ),
        },
    },
    "gsa-depth-v4.3-pairwise": {
        "protocol": "depth-v4.3-pairwise-generation",
        "label_version": "gsa-label-v4.3-depth-pairwise",
        "depth_scope": "pairwise",
        "prompt_hashes": {
            "generator_template_sha256": (
                "406f17b12199664fc2cba409dd41d36597714d04be9538e30b4a7f27cb39bf79"
            ),
            "verifier_protocol_sha256": (
                "c4fecdbd704ff235b7af90e6bd57be8ec82b53aa32a327284bc30ef94ab68adc"
            ),
        },
        "requires_run_audit": True,
        "requires_human_canary": True,
        "human_review_protocol": "depth-v4.3-pairwise-human-canary-review",
        "minimum_canary_accepted": 6,
        "canary_candidate_set_sha256": (
            "51f1a9f013a48d7444e5499b2d63ec52e7b476a86aaaf41779bacb1753645ea2"
        ),
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def generation_run_evidence_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Hash runtime evidence separately from the report's accepted-pool body."""
    return {
        key: value
        for key, value in sorted(report.items())
        if key != "accepted_pool"
    }


def generation_run_evidence_hash(report: dict[str, Any]) -> str:
    return content_hash(generation_run_evidence_payload(report))


def normalized_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def ensure_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"refusing to overwrite non-empty audit output: {path}")


def resolve_source_files(sources: list[Path]) -> list[Path]:
    files: list[Path] = []
    for source in sources:
        if source.is_file():
            files.append(source.resolve())
        elif source.is_dir():
            files.extend(path.resolve() for path in sorted(source.rglob("*.jsonl")))
        else:
            raise SystemExit(f"depth source does not exist: {source}")
    files = sorted(set(files))
    if not files:
        raise SystemExit("no source JSONL files found")
    return files


def iter_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number}: expected a JSON object")
            yield path, line_number, row


def load_depth_records(source_files: list[Path]) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for source_path, line_number, row in iter_jsonl(source_files):
        if row.get("task_form") != FORM:
            continue
        image = row.get("image_path")
        sample_id = row.get("sample_id")
        if not isinstance(image, str) or not image or not isinstance(sample_id, str) or not sample_id:
            rejected["missing_identity"] += 1
            continue
        if not Path(image).is_file():
            rejected["missing_image"] += 1
            continue
        valid, reason = validate_depth_v4_record(row)
        if not valid:
            rejected[f"strict:{reason}"] += 1
            continue
        candidate = copy.deepcopy(row)
        candidate["_v4_source"] = {
            "jsonl_path": str(source_path),
            "line_number": line_number,
            "sample_id": sample_id,
            "label_sha256": content_hash(row),
        }
        records.append(candidate)
    if not records:
        raise SystemExit("no structurally valid depth records found")
    return records, rejected


def depth_protocol_spec(records: list[dict[str, Any]]) -> dict[str, str]:
    """Bind one source pool to exactly one versioned depth semantics."""
    specs: set[str] = set()
    specs_by_protocol = {
        spec["protocol"]: spec for spec in (TOTAL_ORDER_SPEC, *PAIRWISE_SPECS)
    }
    for record in records:
        if (
            record.get("prompt_version") == TOTAL_ORDER_SPEC["prompt_version"]
            and record.get("label_version") == TOTAL_ORDER_SPEC["label_version"]
            and record.get("depth_scope") in (None, "total_order")
        ):
            specs.add(TOTAL_ORDER_SPEC["protocol"])
        else:
            matches = [
                spec
                for spec in PAIRWISE_SPECS
                if record.get("prompt_version") == spec["prompt_version"]
                and record.get("label_version") == spec["label_version"]
                and record.get("depth_scope") == spec["depth_scope"]
            ]
            if len(matches) != 1:
                raise SystemExit(
                    f"validated depth record has an unsupported protocol identity: "
                    f"{record.get('sample_id')}"
                )
            specs.add(matches[0]["protocol"])
    if len(specs) != 1:
        raise SystemExit("depth audit source mixes total-order and pairwise protocols")
    return specs_by_protocol[next(iter(specs))]


def load_exclusions(manifests: list[Path]) -> tuple[set[str], set[str], set[str], list[dict[str, Any]]]:
    if len(manifests) < 2:
        raise SystemExit("v4 protocol requires at least the v2 and v3 exclusion manifests")
    paths: set[str] = set()
    hashes: set[str] = set()
    sample_ids: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for manifest_path in manifests:
        resolved = manifest_path.resolve()
        if not resolved.is_file():
            raise SystemExit(f"exclude manifest does not exist: {manifest_path}")
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
        selected = manifest.get("selected")
        if not isinstance(selected, list):
            raise SystemExit(f"exclude manifest has no selected list: {resolved}")
        for item in selected:
            if not isinstance(item, dict):
                raise SystemExit(f"exclude manifest contains a non-object item: {resolved}")
            image = item.get("image_path")
            if not isinstance(image, str) or not image:
                raise SystemExit(f"exclude manifest item lacks image_path: {resolved}")
            paths.add(normalized_path(image))
            digest = item.get("image_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                image_path = Path(image)
                if not image_path.is_file():
                    raise SystemExit(f"cannot hash excluded image absent from disk: {image}")
                digest = file_hash(image_path)
            hashes.add(digest)
            ids = item.get("sample_ids", {})
            if isinstance(ids, dict):
                sample_ids.update(value for value in ids.values() if isinstance(value, str))
        provenance.append({
            "path": str(resolved),
            "manifest_sha256": file_hash(resolved),
            "schema_version": manifest.get("schema_version", 1),
            "round": manifest.get("round"),
            "audit_id": manifest.get("audit_id"),
            "selected_images": len(selected),
        })
    return paths, hashes, sample_ids, provenance


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def select_records(
    records: list[dict[str, Any]],
    excluded_paths: set[str],
    excluded_hashes: set[str],
    seed: int,
    depth_scope: str = "total_order",
) -> tuple[
    list[tuple[dict[str, Any], str]],
    Counter[str],
    Counter[str],
]:
    rejected: Counter[str] = Counter()
    candidates_by_hash: dict[str, dict[str, Any]] = {}
    for record in records:
        image = str(record["image_path"])
        digest = file_hash(Path(image))
        if normalized_path(image) in excluded_paths:
            rejected["prior_path"] += 1
            continue
        if digest in excluded_hashes:
            rejected["prior_sha256"] += 1
            continue
        if digest in candidates_by_hash:
            rejected["duplicate_image_sha256"] += 1
            continue
        candidates_by_hash[digest] = record
    if len(candidates_by_hash) < REQUIRED_IMAGES:
        raise SystemExit(
            f"only {len(candidates_by_hash)} fresh unique depth images remain; "
            f"v4 requires exactly {REQUIRED_IMAGES}"
        )

    if depth_scope == "pairwise":
        selected = _select_pairwise_records(candidates_by_hash, seed)
        candidate_cue_counts = Counter(
            _pairwise_cue(record) for record in candidates_by_hash.values()
        )
        return selected, rejected, candidate_cue_counts

    strata: dict[tuple[str, str], list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for digest, record in candidates_by_hash.items():
        key = (
            str(record.get("scene_type") or "unknown"),
            str(record.get("source_bucket") or "unknown"),
        )
        strata[key].append((record, digest))
    for values in strata.values():
        values.sort(key=lambda pair: stable_key(seed, f"{pair[1]}:{pair[0]['image_path']}"))
    selected: list[tuple[dict[str, Any], str]] = []
    keys = sorted(strata)
    while len(selected) < REQUIRED_IMAGES:
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < REQUIRED_IMAGES:
                selected.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected, rejected, Counter()


def _pairwise_cue(record: dict[str, Any]) -> str:
    spatial = (record.get("gsa_context") or {}).get("spatial") or {}
    evidence = spatial.get("adjacent_depth_evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        raise SystemExit(f"pairwise record lacks one depth cue: {record.get('sample_id')}")
    cue = evidence[0].get("cue_type") if isinstance(evidence[0], dict) else None
    if cue not in PAIRWISE_CUES:
        raise SystemExit(f"pairwise record has unsupported depth cue: {record.get('sample_id')}")
    return cue


def _source_record(record: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(record)
    value.pop("_v4_source", None)
    return value


def _jsonl_rows(path: Path, kind: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return rows, [f"{kind} is missing: {path}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{kind}:{line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            errors.append(f"{kind}:{line_number}: expected an object")
            continue
        rows.append(row)
    return rows, errors


def _bound_relative_artifact(report_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    root = report_path.parent.resolve()
    artifact = (root / value).resolve()
    return artifact if artifact == root or root in artifact.parents else None


def _run_audit_artifact_errors(
    report: dict[str, Any],
    report_path: Path,
    protocol: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    audit = report.get("run_audit")
    if not isinstance(audit, dict):
        return ["generation report run_audit is missing"], {}
    run_id = audit.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        errors.append("generation run_audit has no run_id")

    request = audit.get("request_log_slice")
    request_hash = None
    if not isinstance(request, dict):
        errors.append("generation run_audit request_log_slice is missing")
    else:
        request_path_value = request.get("path")
        request_path = (
            Path(request_path_value).resolve()
            if isinstance(request_path_value, str) and request_path_value
            else None
        )
        start = request.get("start_byte")
        end = request.get("end_byte_exclusive")
        if (
            request_path is None
            or not request_path.is_file()
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > request_path.stat().st_size
        ):
            errors.append("generation request-log slice is invalid")
        else:
            payload = request_path.read_bytes()[start:end]
            request_hash = hashlib.sha256(payload).hexdigest()
            lines = [line for line in payload.splitlines() if line.strip()]
            try:
                rows = [json.loads(line) for line in lines]
            except json.JSONDecodeError:
                rows = []
                errors.append("generation request-log slice contains invalid JSON")
            if (
                request.get("sha256") != request_hash
                or request.get("row_count") != len(lines)
                or not rows
                or not isinstance(rows[0], dict)
                or not isinstance(rows[-1], dict)
                or rows[0].get("event") != "stage_start"
                or rows[-1].get("event") != "stage_end"
                or rows[0].get("run_id") != run_id
                or rows[-1].get("run_id") != run_id
            ):
                errors.append("generation request-log slice hash/boundaries drift")

    response = audit.get("response_audit")
    response_hash = None
    response_rows: list[dict[str, Any]] = []
    pairs_attempted = report.get("pairs_attempted")
    valid_pairs_attempted = type(pairs_attempted) is int and pairs_attempted >= 0
    if not isinstance(response, dict):
        errors.append("generation run_audit response_audit is missing")
    else:
        response_path = _bound_relative_artifact(report_path, response.get("path"))
        if response_path is None:
            errors.append("generation response audit path is invalid")
        else:
            response_rows, row_errors = _jsonl_rows(response_path, "response audit")
            errors.extend(row_errors)
            if response_path.is_file():
                response_hash = file_hash(response_path)
            if (
                response_hash != response.get("sha256")
                or response.get("row_count") != len(response_rows)
                or not valid_pairs_attempted
                or len(response_rows) < pairs_attempted
            ):
                errors.append("generation response audit hash/count drift")

    cache = audit.get("cache_manifest")
    cache_hash = None
    cache_entries: list[dict[str, Any]] = []
    if not isinstance(cache, dict):
        errors.append("generation run_audit cache_manifest is missing")
    else:
        cache_path = _bound_relative_artifact(report_path, cache.get("path"))
        cache_document: Any = None
        if cache_path is None or not cache_path.is_file():
            errors.append("generation cache manifest path is invalid")
        else:
            cache_hash = file_hash(cache_path)
            try:
                cache_document = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                errors.append("generation cache manifest is invalid JSON")
        if isinstance(cache_document, dict):
            entries = cache_document.get("entries")
            if isinstance(entries, list) and all(isinstance(item, dict) for item in entries):
                cache_entries = entries
            else:
                errors.append("generation cache manifest entries are invalid")
            if (
                cache_document.get("protocol") != protocol
                or cache_document.get("run_id") != run_id
                or cache_document.get("entries_sha256") != content_hash(cache_entries)
            ):
                errors.append("generation cache manifest protocol/run binding drift")
        if (
            cache_hash != cache.get("sha256")
            or cache.get("entry_count") != len(cache_entries)
            or cache.get("entries_sha256") != content_hash(cache_entries)
        ):
            errors.append("generation cache manifest hash/count drift")

    if response_rows and cache_entries:
        response_by_call = {row.get("call_index"): row for row in response_rows}
        cache_by_call = {row.get("call_index"): row for row in cache_entries}
        if (
            len(response_by_call) != len(response_rows)
            or len(cache_by_call) != len(cache_entries)
            or set(response_by_call) != set(cache_by_call)
            or any(
                response_by_call[index].get("response_sha256")
                != cache_by_call[index].get("response_sha256")
                for index in response_by_call
            )
        ):
            errors.append("generation response/cache audit is not one-to-one")

    return errors, {
        "run_id": run_id,
        "request_log_slice_sha256": request_hash,
        "response_audit_sha256": response_hash,
        "cache_manifest_sha256": cache_hash,
    }


def _v43_canary_binding_errors(
    report: dict[str, Any],
    report_path: Path,
    contract: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    provenance = report.get("source_provenance")
    binding = provenance.get("canary_binding") if isinstance(provenance, dict) else None
    if not isinstance(binding, dict):
        return ["v4.3 full report lacks source_provenance.canary_binding"]
    if report.get("canary_binding") != binding:
        errors.append("v4.3 top-level/source_provenance canary binding drift")
    expected_human_gate = {
        "required_for_full": True,
        "protocol": contract["human_review_protocol"],
        "minimum_correct_records": contract["minimum_canary_accepted"],
        "requires_every_emitted_record_reviewed": True,
        "requires_zero_critical_errors": True,
        "passed": True,
    }
    if report.get("human_canary_gate") != expected_human_gate:
        errors.append("v4.3 full report human_canary_gate did not pass")
    canary_path_value = binding.get("path")
    if not isinstance(canary_path_value, str) or not canary_path_value:
        return ["v4.3 canary binding has no report path"]
    canary_path = Path(canary_path_value).resolve()
    if not canary_path.is_file():
        return [f"v4.3 bound canary report is missing: {canary_path}"]
    if file_hash(canary_path) != binding.get("report_sha256"):
        errors.append("v4.3 bound canary report hash drift")
    try:
        canary = json.loads(canary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return errors + ["v4.3 bound canary report is invalid JSON"]
    if not isinstance(canary, dict):
        return errors + ["v4.3 bound canary report is not an object"]

    expected = {
        "protocol": contract["protocol"],
        "depth_scope": "pairwise",
        "prompt_version": report.get("prompt_version"),
        "label_version": contract["label_version"],
        "stage": "canary",
        "protocol_artifact": False,
        "artifact_kind": "canary_preview",
        "candidate_limit_images": 8,
        "pair_selection_rule": "stable_hash_after_threshold",
        "role_order_counterbalanced": True,
        "cue_agreement_required": True,
    }
    for field, value in expected.items():
        if canary.get(field) != value:
            errors.append(f"v4.3 bound canary {field} drift")
    if canary.get("prompt_hashes") != contract["prompt_hashes"]:
        errors.append("v4.3 bound canary prompt hashes drift")
    if (
        canary.get("candidate_set_sha256")
        != contract["canary_candidate_set_sha256"]
        or binding.get("candidate_set_sha256") != canary.get("candidate_set_sha256")
    ):
        errors.append("v4.3 bound canary candidate set drift")
    accepted = canary.get("accepted_pool_images")
    canary_gate = canary.get("canary_gate")
    canary_gate = canary_gate if isinstance(canary_gate, dict) else {}
    system_errors = canary.get("system_errors")
    system_errors = system_errors if isinstance(system_errors, dict) else {}
    if (
        type(accepted) is not int
        or accepted < contract["minimum_canary_accepted"]
        or canary.get("emitted_records") != accepted
        or canary_gate.get("passed") is not True
        or system_errors.get("total") != 0
    ):
        errors.append("v4.3 bound canary machine gate did not pass")

    labels_path = canary_path.parent / "labels/S_spatial/depth.jsonl"
    labels, label_errors = _jsonl_rows(labels_path, "v4.3 canary labels")
    errors.extend(label_errors)
    if (
        not labels_path.is_file()
        or canary.get("labels_sha256") != file_hash(labels_path)
        or binding.get("labels_sha256") != canary.get("labels_sha256")
        or len(labels) != accepted
    ):
        errors.append("v4.3 bound canary labels hash/count drift")
    pool = canary.get("accepted_pool")
    if (
        not isinstance(pool, list)
        or canary.get("accepted_pool_sha256") != content_hash(pool)
        or len(pool) != len(labels)
    ):
        errors.append("v4.3 bound canary accepted pool drift")
    else:
        for item, label in zip(pool, labels):
            sources = label.get("_v4_sources")
            image_sha = sources.get("image_sha256") if isinstance(sources, dict) else None
            if (
                not isinstance(item, dict)
                or item.get("sample_id") != label.get("sample_id")
                or item.get("label_sha256") != content_hash(label)
                or item.get("image_path") != label.get("image_path")
                or item.get("image_sha256") != image_sha
                or not validate_depth_v4_record(label)[0]
            ):
                errors.append("v4.3 bound canary accepted pool/labels mismatch")
                break

    run_errors, run_binding = _run_audit_artifact_errors(
        canary, canary_path, contract["protocol"]
    )
    errors.extend(f"v4.3 canary {error}" for error in run_errors)
    if binding.get("run_audit") != run_binding:
        errors.append("v4.3 canary run-audit summary drift")

    human = binding.get("human_review")
    if not isinstance(human, dict):
        return errors + ["v4.3 canary human_review binding is missing"]
    review_path_value = human.get("path")
    review_path = (
        Path(review_path_value).resolve()
        if isinstance(review_path_value, str) and review_path_value
        else None
    )
    review: Any = None
    if review_path is None or not review_path.is_file():
        errors.append("v4.3 canary human review file is missing")
    else:
        if file_hash(review_path) != human.get("sha256"):
            errors.append("v4.3 canary human review hash drift")
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("v4.3 canary human review is invalid JSON")
    rows = review.get("reviews") if isinstance(review, dict) else None
    if (
        not isinstance(review, dict)
        or review.get("schema_version") != 1
        or review.get("protocol") != contract["human_review_protocol"]
        or review.get("generation_protocol") != contract["protocol"]
        or review.get("prompt_version") != report.get("prompt_version")
        or review.get("label_version") != report.get("label_version")
        or not isinstance(review.get("reviewer_id"), str)
        or not review.get("reviewer_id", "").strip()
        or not isinstance(review.get("reviewed_at_utc"), str)
        or not review.get("reviewed_at_utc", "").strip()
        or review.get("canary_report_sha256") != binding.get("report_sha256")
        or review.get("labels_sha256") != canary.get("labels_sha256")
        or review.get("candidate_set_sha256") != canary.get("candidate_set_sha256")
        or review.get("request_log_slice_sha256")
        != run_binding.get("request_log_slice_sha256")
        or review.get("response_audit_sha256")
        != run_binding.get("response_audit_sha256")
        or review.get("cache_manifest_sha256")
        != run_binding.get("cache_manifest_sha256")
        or not isinstance(rows, list)
        or len(rows) != len(labels)
    ):
        errors.append("v4.3 canary human review artifact binding drift")
        rows = []
    expected_labels = {
        label.get("sample_id"): (
            content_hash(label),
            (label.get("_v4_sources") or {}).get("image_sha256"),
        )
        for label in labels
        if isinstance(label.get("_v4_sources"), dict)
    }
    correct = 0
    critical = 0
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id") if isinstance(row, dict) else None
        expected_hashes = expected_labels.get(sample_id)
        votes = (
            [
                row.get("direction_correct"),
                row.get("cue_consistent"),
                row.get("evidence_entails_direction"),
                row.get("geometry_supported"),
            ]
            if isinstance(row, dict)
            else []
        )
        row_correct = len(votes) == 4 and all(type(vote) is bool for vote in votes) and all(votes)
        if (
            expected_hashes is None
            or sample_id in seen
            or row.get("label_sha256") != expected_hashes[0]
            or row.get("image_sha256") != expected_hashes[1]
            or row.get("correct") is not row_correct
            or type(row.get("critical_error")) is not bool
        ):
            errors.append("v4.3 canary human review row drift")
            continue
        seen.add(sample_id)
        correct += int(row_correct)
        critical += int(row["critical_error"])
    if (
        set(expected_labels) != seen
        or correct < contract["minimum_canary_accepted"]
        or critical != 0
        or human.get("protocol") != contract["human_review_protocol"]
        or human.get("reviewed_records") != len(rows)
        or human.get("correct_records") != correct
        or human.get("critical_errors") != critical
        or human.get("passed") is not True
    ):
        errors.append("v4.3 canary human review gate did not pass")
    return errors


def pairwise_generation_report_errors(
    report: dict[str, Any],
    report_path: Path,
    source_files: list[Path],
    records: list[dict[str, Any]],
    protocol_spec: dict[str, str],
) -> list[str]:
    """Validate a full pairwise generation run against its emitted source labels."""
    errors: list[str] = []
    prompt_version = protocol_spec["prompt_version"]
    contract = PAIRWISE_GENERATION_CONTRACTS.get(prompt_version)
    if contract is None:
        return [f"unregistered pairwise generation prompt: {prompt_version}"]

    label_path = (report_path.parent / "labels/S_spatial/depth.jsonl").resolve()
    resolved_sources = [path.resolve() for path in source_files]
    if resolved_sources != [label_path]:
        errors.append(
            "pairwise source must be exactly the generation report's "
            "labels/S_spatial/depth.jsonl"
        )
    if not label_path.is_file():
        errors.append(f"generation source labels are missing: {label_path}")

    expected_identity = {
        "schema_version": 1,
        "protocol": contract["protocol"],
        "depth_scope": contract["depth_scope"],
        "prompt_version": prompt_version,
        "label_version": contract["label_version"],
        "stage": "full",
        "protocol_artifact": True,
        "artifact_kind": "generation_pool",
        "candidate_limit_images": None,
        "minimum_audit_images": REQUIRED_IMAGES,
        "audit_pool_eligible": True,
        "pair_selection_rule": "stable_hash_after_threshold",
        "role_order_counterbalanced": True,
    }
    for field, expected in expected_identity.items():
        if report.get(field) != expected:
            errors.append(
                f"generation report {field}={report.get(field)!r}, expected {expected!r}"
            )
    if report.get("prompt_hashes") != contract["prompt_hashes"]:
        errors.append("generation report prompt_hashes do not match the registered protocol")
    if contract.get("requires_human_canary"):
        geometry_gate = report.get("geometry_gate")
        if report.get("cue_agreement_required") is not True:
            errors.append("v4.3 generation report does not require cue agreement")
        if not isinstance(geometry_gate, dict) or geometry_gate.get("required") is not True:
            errors.append("v4.3 generation report geometry gate is not required")

    system_errors = report.get("system_errors")
    if not isinstance(system_errors, dict):
        errors.append("generation report system_errors must be an object")
    else:
        parser_errors = system_errors.get("parser_rejections")
        runtime_errors = system_errors.get("runtime_or_record_rejections")
        total_errors = system_errors.get("total")
        values = (parser_errors, runtime_errors, total_errors)
        if not all(type(value) is int for value in values) or (
            parser_errors != 0
            or runtime_errors != 0
            or total_errors != 0
            or total_errors != parser_errors + runtime_errors
        ):
            errors.append("generation report requires zero internally consistent system_errors")

    cue_gate = report.get("cue_gate")
    if not isinstance(cue_gate, dict):
        errors.append("generation report cue_gate must be an object")
    else:
        if cue_gate.get("minimum_images") != PAIRWISE_MIN_CUE_COUNTS:
            errors.append("generation report cue_gate minimums do not match the protocol")
        if cue_gate.get("passed") is not True:
            errors.append("generation report cue_gate did not pass")

    source_records = [_source_record(record) for record in records]
    source_count = len(source_records)
    for field in ("accepted_pool_images", "emitted_records"):
        if report.get(field) != source_count:
            errors.append(
                f"generation report {field}={report.get(field)!r}, "
                f"but source labels contain {source_count} records"
            )
    if source_count < REQUIRED_IMAGES:
        errors.append(
            f"generation source has {source_count} accepted records; requires at least {REQUIRED_IMAGES}"
        )
    if label_path.is_file() and report.get("labels_sha256") != file_hash(label_path):
        errors.append("generation report labels_sha256 does not match source labels")

    accepted_pool = report.get("accepted_pool")
    if not isinstance(accepted_pool, list):
        errors.append("generation report accepted_pool must be a list")
        accepted_pool = []
    if report.get("accepted_pool_sha256") != content_hash(accepted_pool):
        errors.append("generation report accepted_pool_sha256 drift")
    if len(accepted_pool) != source_count:
        errors.append("generation report accepted_pool length does not match source labels")

    accepted_hashes: list[str] = []
    source_cues: list[str] = []
    for index, (item, record) in enumerate(zip(accepted_pool, source_records)):
        if not isinstance(item, dict):
            errors.append(f"generation accepted_pool[{index}] is not an object")
            continue
        context = record.get("gsa_context")
        spatial = context.get("spatial") if isinstance(context, dict) else None
        spatial = spatial if isinstance(spatial, dict) else {}
        evidence = spatial.get("adjacent_depth_evidence")
        cue = (
            evidence[0].get("cue_type")
            if isinstance(evidence, list)
            and len(evidence) == 1
            and isinstance(evidence[0], dict)
            else None
        )
        if cue in PAIRWISE_CUES:
            source_cues.append(cue)
        else:
            errors.append(f"generation source label {index} has an invalid depth cue")
        payload = record.get("payload")
        selected_pair = (
            [entry.get("label") for entry in payload]
            if isinstance(payload, list)
            and len(payload) == 2
            and all(isinstance(entry, dict) for entry in payload)
            else None
        )
        sources = record.get("_v4_sources")
        image_sha256 = (
            sources.get("image_sha256") if isinstance(sources, dict) else None
        )
        expected_item = {
            "image_path": record.get("image_path"),
            "image_sha256": image_sha256,
            "sample_id": record.get("sample_id"),
            "label_sha256": content_hash(record),
            "selected_pair": selected_pair,
            "selection_score": record.get("confidence"),
            "cue_type": cue,
        }
        for field, expected in expected_item.items():
            if item.get(field) != expected:
                errors.append(
                    f"generation accepted_pool[{index}].{field} does not match source label"
                )
        if is_sha256(image_sha256):
            accepted_hashes.append(image_sha256)
        else:
            errors.append(f"generation source label {index} lacks a valid image hash")
    if len(accepted_hashes) != len(set(accepted_hashes)):
        errors.append("generation accepted pool contains duplicate image hashes")

    cue_counts = Counter(source_cues)
    scene_counts = Counter(
        str(record.get("scene_type") or "unknown") for record in source_records
    )
    source_counts = Counter(
        str(record.get("source_bucket") or "unknown") for record in source_records
    )
    for field, counts in (
        ("accepted_by_cue", cue_counts),
        ("accepted_by_scene", scene_counts),
        ("accepted_by_source", source_counts),
    ):
        if report.get(field) != dict(sorted(counts.items())):
            errors.append(f"generation report {field} does not match source labels")
    for cue, minimum in PAIRWISE_MIN_CUE_COUNTS.items():
        if cue_counts[cue] < minimum:
            errors.append(
                f"generation source requires at least {minimum} {cue} records"
            )

    candidate_hashes = report.get("candidate_image_sha256s")
    if (
        not isinstance(candidate_hashes, list)
        or not all(is_sha256(value) for value in candidate_hashes)
        or len(candidate_hashes) != len(set(candidate_hashes))
    ):
        errors.append("generation report candidate_image_sha256s are invalid")
    else:
        if report.get("candidate_images") != len(candidate_hashes):
            errors.append("generation report candidate_images count drift")
        if report.get("attempted_images") != len(candidate_hashes):
            errors.append("generation report did not attempt every candidate image")
        if report.get("candidate_set_sha256") != content_hash(candidate_hashes):
            errors.append("generation report candidate_set_sha256 drift")
        positions = {digest: index for index, digest in enumerate(candidate_hashes)}
        if any(digest not in positions for digest in accepted_hashes):
            errors.append("generation accepted pool contains a non-candidate image")
        else:
            accepted_positions = [positions[digest] for digest in accepted_hashes]
            if accepted_positions != sorted(accepted_positions):
                errors.append("generation accepted pool order differs from candidate order")

    if not isinstance(report.get("source_provenance"), dict):
        errors.append("generation report source_provenance must be an object")
    if contract.get("requires_run_audit"):
        run_errors, _ = _run_audit_artifact_errors(
            report, report_path, contract["protocol"]
        )
        errors.extend(run_errors)
    if contract.get("requires_human_canary"):
        errors.extend(_v43_canary_binding_errors(report, report_path, contract))
    return errors


def load_pairwise_generation_run(
    report_path: Path,
    source_files: list[Path],
    records: list[dict[str, Any]],
    protocol_spec: dict[str, str],
) -> dict[str, Any]:
    resolved = report_path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"pairwise generation report does not exist: {report_path}")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid pairwise generation report: {resolved}: {exc}") from exc
    if not isinstance(report, dict):
        raise SystemExit(f"pairwise generation report is not an object: {resolved}")
    errors = pairwise_generation_report_errors(
        report, resolved, source_files, records, protocol_spec
    )
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"pairwise full generation report binding failed:\n{details}")
    label_path = (resolved.parent / "labels/S_spatial/depth.jsonl").resolve()
    generation_run = {
        "report_path": str(resolved),
        "report_sha256": file_hash(resolved),
        "run_evidence_sha256": generation_run_evidence_hash(report),
        "source_labels_path": str(label_path),
        "source_labels_sha256": file_hash(label_path),
        "accepted_pool_sha256": report["accepted_pool_sha256"],
        "accepted_pool_images": report["accepted_pool_images"],
        "protocol": report["protocol"],
        "depth_scope": report["depth_scope"],
        "prompt_version": report["prompt_version"],
        "label_version": report["label_version"],
        "prompt_hashes": report["prompt_hashes"],
        "stage": report["stage"],
    }
    if report.get("run_audit") is not None:
        generation_run["run_audit_sha256"] = content_hash(report["run_audit"])
    provenance = report.get("source_provenance")
    canary_binding = (
        provenance.get("canary_binding") if isinstance(provenance, dict) else None
    )
    if canary_binding is not None:
        generation_run["canary_binding_sha256"] = content_hash(canary_binding)
    return generation_run


def _select_pairwise_records(
    candidates_by_hash: dict[str, dict[str, Any]],
    seed: int,
) -> list[tuple[dict[str, Any], str]]:
    """Select a cue-balanced audit set while retaining scene/source diversity."""
    cue_strata: dict[
        str,
        dict[tuple[str, str], list[tuple[dict[str, Any], str]]],
    ] = {cue: defaultdict(list) for cue in PAIRWISE_CUES}
    for digest, record in candidates_by_hash.items():
        cue = _pairwise_cue(record)
        stratum = (
            str(record.get("scene_type") or "unknown"),
            str(record.get("source_bucket") or "unknown"),
        )
        cue_strata[cue][stratum].append((record, digest))

    candidate_counts = Counter({
        cue: sum(len(values) for values in cue_strata[cue].values())
        for cue in PAIRWISE_CUES
    })
    deficits = {
        cue: minimum - candidate_counts[cue]
        for cue, minimum in PAIRWISE_MIN_CUE_COUNTS.items()
        if candidate_counts[cue] < minimum
    }
    if deficits:
        details = ", ".join(
            f"{cue}={candidate_counts[cue]}<{PAIRWISE_MIN_CUE_COUNTS[cue]}"
            for cue in PAIRWISE_MIN_CUE_COUNTS
            if cue in deficits
        )
        raise SystemExit(f"pairwise cue coverage is insufficient: {details}")

    queues: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for cue in PAIRWISE_CUES:
        strata = cue_strata[cue]
        keys = sorted(strata)
        for key, values in strata.items():
            values.sort(
                key=lambda pair: stable_key(
                    seed, f"{cue}:{key[0]}:{key[1]}:{pair[1]}:{pair[0]['image_path']}"
                )
            )
        queue: list[tuple[dict[str, Any], str]] = []
        while any(strata.values()):
            for key in keys:
                if strata[key]:
                    queue.append(strata[key].pop(0))
        queues[cue] = queue

    selected: list[tuple[dict[str, Any], str]] = []
    selected_counts: Counter[str] = Counter()
    for cue, minimum in PAIRWISE_MIN_CUE_COUNTS.items():
        for _ in range(minimum):
            selected.append(queues[cue].pop(0))
            selected_counts[cue] += 1

    cue_order = {cue: index for index, cue in enumerate(PAIRWISE_CUES)}
    while len(selected) < REQUIRED_IMAGES:
        available = [cue for cue in PAIRWISE_CUES if queues[cue]]
        if not available:
            break
        cue = min(available, key=lambda value: (selected_counts[value], cue_order[value]))
        selected.append(queues[cue].pop(0))
        selected_counts[cue] += 1
    if len(selected) != REQUIRED_IMAGES:
        raise SystemExit(
            f"only {len(selected)} pairwise samples could be selected; "
            f"v4 requires exactly {REQUIRED_IMAGES}"
        )
    return selected


def fresh_label(
    record: dict[str, Any],
    image_sha256: str,
    seed: int,
    audit_round: str,
) -> dict[str, Any]:
    source = record.pop("_v4_source")
    identity = content_hash({
        "seed": seed,
        "image_sha256": image_sha256,
        "source_label_sha256": source["label_sha256"],
    })
    label = copy.deepcopy(record)
    label["sample_id"] = f"v4_depth_{identity[:20]}"
    label["audit_schema_version"] = SCHEMA_VERSION
    label["audit_round"] = audit_round
    label["source_sample_id"] = source["sample_id"]
    label["source_label_sha256"] = source["label_sha256"]
    label["source_jsonl_path"] = source["jsonl_path"]
    label["source_jsonl_line"] = source["line_number"]
    label["label_version"] = str(label.get("label_version") or "legacy-unversioned")
    label["prompt_version"] = str(label.get("prompt_version") or "legacy-unversioned")
    return label


def manifest_payload_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("integrity", None)
    return content_hash(payload)


def write_root(
    root: Path,
    labels: list[dict[str, Any]],
    audit_id: str,
    protocol: str,
    depth_scope: str,
) -> None:
    label_path = root / "labels/S_spatial/depth.jsonl"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "".join(json.dumps(label, ensure_ascii=False) + "\n" for label in labels),
        encoding="utf-8",
    )
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "adapter_stats.json").write_text(json.dumps({
        "semantic_audit": True,
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "depth_scope": depth_scope,
        "audit_id": audit_id,
        "images": REQUIRED_IMAGES,
        "forms": 1,
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("data/semantic_audit_v4_depth"))
    parser.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--generation-report",
        type=Path,
        help="required full generation_report.json for pairwise depth sources",
    )
    parser.add_argument("--images", type=int, default=REQUIRED_IMAGES)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--round")
    parser.add_argument("--audit-id")
    args = parser.parse_args()

    if args.images != REQUIRED_IMAGES:
        raise SystemExit(f"v4 depth protocol requires exactly {REQUIRED_IMAGES} images")
    ensure_empty_output(args.out)
    source_files = resolve_source_files(args.source)
    records, structural_rejections = load_depth_records(source_files)
    protocol_spec = depth_protocol_spec(records)
    generation_run: dict[str, Any] | None = None
    if protocol_spec["depth_scope"] == "pairwise":
        if args.generation_report is None:
            raise SystemExit("pairwise depth audit requires --generation-report")
        generation_run = load_pairwise_generation_run(
            args.generation_report,
            source_files,
            records,
            protocol_spec,
        )
    elif args.generation_report is not None:
        raise SystemExit(
            "--generation-report binding is currently registered for pairwise depth only"
        )
    audit_round = args.round or protocol_spec["round"]
    if audit_round != protocol_spec["round"]:
        raise SystemExit(
            f"{protocol_spec['protocol']} requires --round {protocol_spec['round']}"
        )
    excluded_paths, excluded_hashes, prior_sample_ids, exclusion_provenance = load_exclusions(
        args.exclude_manifest
    )
    selected_records, selection_rejections, candidate_cue_counts = select_records(
        records,
        excluded_paths,
        excluded_hashes,
        args.seed,
        protocol_spec["depth_scope"],
    )

    source_files_provenance = [
        {"path": str(path), "sha256": file_hash(path)} for path in source_files
    ]
    source_fingerprint = content_hash(source_files_provenance)
    audit_id = args.audit_id or (
        f"{protocol_spec['audit_id_prefix']}-{source_fingerprint[:12]}"
    )
    labels = [
        fresh_label(copy.deepcopy(record), digest, args.seed, audit_round)
        for record, digest in selected_records
    ]
    sample_ids = [label["sample_id"] for label in labels]
    if len(set(sample_ids)) != REQUIRED_IMAGES:
        raise SystemExit("fresh v4 sample IDs are not unique")
    overlap = set(sample_ids) & prior_sample_ids
    if overlap:
        raise SystemExit(f"fresh v4 sample IDs overlap prior audit IDs: {sorted(overlap)}")

    selected: list[dict[str, Any]] = []
    pairs: list[dict[str, str]] = []
    for label, (_, image_sha256) in zip(labels, selected_records):
        digest = content_hash(label)
        selected.append({
            "image_path": label["image_path"],
            "image_sha256": image_sha256,
            "sample_ids": {FORM: label["sample_id"]},
            "source_sample_ids": {FORM: label["source_sample_id"]},
            "label_hashes": {FORM: digest},
            "label_versions": {FORM: label["label_version"]},
            "prompt_versions": {FORM: label["prompt_version"]},
            "depth_scopes": {FORM: protocol_spec["depth_scope"]},
            "depth_cues": ({FORM: _pairwise_cue(label)}
                           if protocol_spec["depth_scope"] == "pairwise" else {}),
        })
        pairs.append({"sample_id": label["sample_id"], "label_sha256": digest})

    scene_counts = Counter(str(label.get("scene_type") or "unknown") for label in labels)
    source_counts = Counter(str(label.get("source_bucket") or "unknown") for label in labels)
    cue_counts = Counter(
        _pairwise_cue(label) for label in labels
    ) if protocol_spec["depth_scope"] == "pairwise" else Counter()
    roots = [args.out / "annotator_a", args.out / "annotator_b"]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol_spec["protocol"],
        "round": audit_round,
        "depth_scope": protocol_spec["depth_scope"],
        "audit_id": audit_id,
        "seed": args.seed,
        "source_fingerprint": source_fingerprint,
        "source_files": source_files_provenance,
        "exclusion_manifests": exclusion_provenance,
        "images": REQUIRED_IMAGES,
        "forms": [FORM],
        "records_per_annotator": REQUIRED_IMAGES,
        "roots": [str(root) for root in roots],
        "per_form_label_hashes": {FORM: content_hash(pairs)},
        "scene_counts": dict(sorted(scene_counts.items())),
        "source_bucket_counts": dict(sorted(source_counts.items())),
        "candidate_cue_counts": dict(sorted(candidate_cue_counts.items())),
        "cue_counts": dict(sorted(cue_counts.items())),
        "rejected_source_records": dict(
            sorted((structural_rejections + selection_rejections).items())
        ),
        "selected": selected,
    }
    if generation_run is not None:
        manifest["generation_run"] = generation_run
    manifest["integrity"] = {
        "selected_sha256": content_hash(selected),
        "manifest_payload_sha256": manifest_payload_hash(manifest),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for root in roots:
        write_root(
            root,
            labels,
            audit_id,
            protocol_spec["protocol"],
            protocol_spec["depth_scope"],
        )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "audit_id": audit_id,
        "schema_version": SCHEMA_VERSION,
        "images": REQUIRED_IMAGES,
        "forms": [FORM],
        "protocol": protocol_spec["protocol"],
        "depth_scope": protocol_spec["depth_scope"],
        "scene_counts": manifest["scene_counts"],
        "source_bucket_counts": manifest["source_bucket_counts"],
        "selected_sha256": manifest["integrity"]["selected_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
