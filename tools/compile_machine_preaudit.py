#!/usr/bin/env python3
"""Compile two independent machine pre-audits into a human review queue."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "sample_id",
    "reviewer",
    "supported",
    "error_code",
    "concise_reason",
    "confidence",
    "reviewed_at_utc",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            extra = set(row) - REQUIRED_FIELDS
            if extra:
                raise ValueError(
                    f"{path}:{line_number}: unexpected fields may leak private data: {sorted(extra)}"
                )
            if not isinstance(row["sample_id"], str) or not row["sample_id"]:
                raise ValueError(f"{path}:{line_number}: invalid sample_id")
            if not isinstance(row["reviewer"], str) or not row["reviewer"]:
                raise ValueError(f"{path}:{line_number}: invalid reviewer")
            if not isinstance(row["supported"], bool):
                raise ValueError(f"{path}:{line_number}: supported must be Boolean")
            error_code = row["error_code"]
            if error_code is not None and (
                not isinstance(error_code, str) or not error_code.strip()
            ):
                raise ValueError(f"{path}:{line_number}: invalid error_code")
            if row["supported"] and error_code is not None:
                raise ValueError(f"{path}:{line_number}: supported rows cannot have an error_code")
            if not row["supported"] and error_code is None:
                raise ValueError(f"{path}:{line_number}: unsupported rows require an error_code")
            if not isinstance(row["concise_reason"], str) or not row["concise_reason"].strip():
                raise ValueError(f"{path}:{line_number}: concise_reason is required")
            if not isinstance(row["reviewed_at_utc"], str) or not row["reviewed_at_utc"].strip():
                raise ValueError(f"{path}:{line_number}: reviewed_at_utc is required")
            confidence = row["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError(f"{path}:{line_number}: confidence must be numeric")
            if not 0 <= float(confidence) <= 1:
                raise ValueError(f"{path}:{line_number}: confidence must be in [0,1]")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


PUBLIC_PAYLOAD_FIELDS = {
    "label", "object_class", "bbox", "bbox_2d", "point_2d", "polygon", "rank",
    "relative_depth", "affordance", "part", "part_label", "action", "contact_point_2d",
    "grasp_point_2d", "jaw_axis_2d", "approach", "wrist_deg", "target_point_2d",
    "goal", "steps", "preconditions", "failure_modes", "reasoning",
}


def visible_upstream_context(form: str, context: dict[str, Any]) -> dict[str, Any]:
    """Expose only task-provided predecessors, never the current answer or private gates."""
    keys = {
        "affordance": ("depth_order",),
        "dof": ("depth_order", "affordance"),
        "action_cot": ("depth_order", "affordance", "dof"),
    }.get(form, ())
    spatial = context.get("spatial") or {}
    return {
        key: [{field: value for field, value in item.items() if field in PUBLIC_PAYLOAD_FIELDS}
              for item in spatial[key]]
        for key in keys if isinstance(spatial.get(key), list)
        and all(isinstance(item, dict) for item in spatial[key])
    }


def visible_native_groundings(form: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    # Native G prompts expose the named referent, never its reference box. The
    # private gsa_context may contain that box for scoring and must stay hidden.
    if form in {"bbox", "detection", "mask", "point"}:
        return []
    return [{key: item[key] for key in ("object_class", "bbox") if key in item}
            for item in context.get("groundings", [])]


def prepare_flywheel_inputs(raw: Path, output: Path) -> int:
    """Freeze visible proposals without exposing generator verdicts to reviewers."""
    if output.exists():
        raise ValueError(f"refusing existing public input: {output}")
    rows = []
    seen = set()
    image_hashes = {}
    for path in sorted(raw.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source = json.loads(line)
            form = source["form"]
            if form != path.stem:
                raise ValueError(f"form/file mismatch: {path}")
            image_path = Path(source["image_path"]).resolve()
            if image_path not in image_hashes:
                image_hashes[image_path] = sha256(image_path)
            image_sha = image_hashes[image_path]
            if image_sha != source["image_sha256"]:
                raise ValueError(f"image binding drift: {image_path}")
            sample_id = image_sha + ":" + form
            if sample_id in seen:
                raise ValueError(f"duplicate public sample: {sample_id}")
            seen.add(sample_id)
            payload = source["payload"]
            if not isinstance(payload, list) or not all(isinstance(x, dict) for x in payload):
                raise ValueError(f"invalid visible payload: {sample_id}")
            unknown = set().union(*(set(x) for x in payload)) - PUBLIC_PAYLOAD_FIELDS
            if unknown:
                raise ValueError(f"unreviewed payload fields for {sample_id}: {sorted(unknown)}")
            context = source.get("gsa_context") or {}
            rows.append({
                "sample_id": sample_id, "form": form, "image_path": str(image_path),
                "image_sha256": image_sha, "payload": payload,
                "groundings": [
                    {key: item[key] for key in ("object_class", "bbox") if key in item}
                    for item in context.get("groundings", [])
                ],
                "spatial_context": visible_upstream_context(form, context),
            })
    if not rows:
        raise ValueError(f"no proposals found: {raw}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def load_public_inputs(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {row["sample_id"]: row for row in rows}
    if not indexed or len(indexed) != len(rows):
        raise ValueError("public input must contain nonempty, unique sample IDs")
    image_hashes = {}
    for row in rows:
        image_path = Path(row["image_path"])
        if image_path not in image_hashes:
            image_hashes[image_path] = sha256(image_path)
        if image_hashes[image_path] != row["image_sha256"]:
            raise ValueError(f"image binding drift: {row['sample_id']}")
    return indexed


def prepare_native_predictions(source: Path, predictions: list[tuple[str, Path]], output: Path) -> int:
    """Bind native inference by exact prompt/image, hide reference answers and model identity."""
    if output.exists():
        raise ValueError(f"refusing existing public input: {output}")

    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def key(row):
        images = tuple(str(Path(x["path"] if isinstance(x, dict) else x).resolve())
                       for x in row["images"])
        prompt = tuple((m["role"], m["content"]) for m in row["messages"]
                       if m["role"] != "assistant")
        return images, prompt

    originals = rows(source)
    indexed = {key(row): row for row in originals}
    if not originals or len(indexed) != len(originals):
        raise ValueError("native source must have unique nonempty prompt/image identities")
    public = []
    families = set()
    for family, path in predictions:
        if family not in {"G", "S", "A"} or family in families:
            raise ValueError("each native family must appear once and be G, S or A")
        families.add(family)
        inferred = rows(path)
        joined = {key(row): row for row in inferred}
        if len(joined) != len(inferred) or set(joined) != set(indexed):
            raise ValueError("native predictions must cover every source prompt/image exactly once")
        prediction_sha = sha256(path)
        for identity, original in indexed.items():
            prediction = joined[identity]
            reference = [m["content"] for m in original["messages"] if m["role"] == "assistant"]
            if len(reference) != 1 or prediction.get("labels") != reference[0]:
                raise ValueError("native inference reference binding drift")
            if original["gsa_family"] != family:
                continue
            response = re.sub(r"^\s*<think>\s*</think>\s*", "", prediction["response"]).strip()
            prefix = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}[family]
            if not response.startswith(prefix):
                raise ValueError("native prediction missing expected stage prefix")
            payload = json.loads(response[len(prefix):])
            if not isinstance(payload, list) or not payload or not all(isinstance(x, dict) for x in payload):
                raise ValueError("native prediction must be a nonempty JSON array of objects")
            if set().union(*(set(x) for x in payload)) - PUBLIC_PAYLOAD_FIELDS:
                raise ValueError("native prediction has unreviewed payload fields")
            if len(identity[0]) != 1:
                raise ValueError("offline native review requires exactly one bound image")
            image_path = Path(identity[0][0])
            image_sha = sha256(image_path)
            if image_sha != original["gsa_meta"]["image_sha256"]:
                raise ValueError("native source image binding drift")
            context = original.get("gsa_context") or {}
            public.append({
                "sample_id": hashlib.sha256((prediction_sha + original["sample_id"]).encode()).hexdigest(),
                "form": original["gsa_form"], "image_path": str(image_path),
                "image_sha256": image_sha, "payload": payload,
                "instruction": "\n".join(content for role, content in identity[1] if role == "user"),
                "groundings": visible_native_groundings(original["gsa_form"], context),
                "spatial_context": visible_upstream_context(original["gsa_form"], context),
            })
    if not public:
        raise ValueError("no native predictions selected")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in sorted(public, key=lambda row: row["sample_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(public)


def prepare_native_candidates(source: Path, output: Path, *, per_form: int,
                              seed: str, exclude: list[Path], selected_forms: list[str] | None = None) -> dict[str, Any]:
    """Freeze a label-blind stratified sample, excluding prior images and scene groups."""
    if output.exists():
        raise ValueError(f"refusing existing public input: {output}")
    if per_form < 1:
        raise ValueError("candidate sample size per form must be positive")
    originals = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    ids = [row["sample_id"] for row in originals]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("native candidates require nonempty unique sample IDs")
    excluded_images = {row["image_sha256"] for path in exclude
                       for row in load_public_inputs(path).values()}

    def group(row):
        meta = row["gsa_meta"]
        return meta.get("scene_group") or meta["image_sha256"]

    excluded_groups = {group(row) for row in originals
                       if row["gsa_meta"]["image_sha256"] in excluded_images}
    eligible = [row for row in originals if group(row) not in excluded_groups
                and row["gsa_meta"]["image_sha256"] not in excluded_images]
    forms = sorted({row["gsa_form"] for row in originals})
    if selected_forms is not None:
        if not selected_forms or len(set(selected_forms)) != len(selected_forms) or set(selected_forms) - set(forms):
            raise ValueError("selected review forms must be unique and present in the source")
        forms = sorted(selected_forms)
    selected = []
    for form in forms:
        pool = [row for row in eligible if row["gsa_form"] == form]
        if len(pool) < per_form:
            raise ValueError(f"insufficient fresh {form} candidates: {len(pool)} < {per_form}")
        selected.extend(sorted(pool, key=lambda row: hashlib.sha256(
            (seed + ":" + row["sample_id"]).encode()).hexdigest())[:per_form])
    public = []
    for row in selected:
        payload = row["gsa_payload"]
        if not isinstance(payload, list) or not payload or not all(isinstance(x, dict) for x in payload):
            raise ValueError("native candidate must contain a nonempty object array")
        if set().union(*(set(x) for x in payload)) - PUBLIC_PAYLOAD_FIELDS:
            raise ValueError("native candidate has unreviewed payload fields")
        if len(row["images"]) != 1:
            raise ValueError("native review requires exactly one image")
        image_path = Path(row["images"][0]).resolve()
        image_sha = sha256(image_path)
        if image_sha != row["gsa_meta"]["image_sha256"]:
            raise ValueError("native candidate image binding drift")
        context = row.get("gsa_context") or {}
        public.append({
            "sample_id": row["sample_id"], "form": row["gsa_form"],
            "image_path": str(image_path), "image_sha256": image_sha, "payload": payload,
            "instruction": "\n".join(m["content"] for m in row["messages"] if m["role"] == "user"),
            "groundings": visible_native_groundings(row["gsa_form"], context),
            "spatial_context": visible_upstream_context(row["gsa_form"], context),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in sorted(public, key=lambda row: row["sample_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "status": "MACHINE_REVIEW_PENDING", "source_sha256": sha256(source),
        "public_inputs_sha256": sha256(output), "sampling_seed": seed,
        "selected_forms": forms,
        "selection": "SHA256(seed:sample_id) ascending within each form; no label-quality filtering",
        "source_rows": len(originals), "eligible_rows": len(eligible),
        "excluded_scene_groups": len(excluded_groups),
        "exclusion_inputs": [{"path": str(path), "sha256": sha256(path)} for path in exclude],
        "records": len(public), "per_form": dict(Counter(row["form"] for row in public)),
        "images": len({row["image_sha256"] for row in public}),
        "machine_only": True, "may_populate_formal_human_metrics": False,
    }


def index_stream(
    path: Path, *, expected_count: int
) -> tuple[str, dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} rows, found {len(rows)}")
    reviewers = {row["reviewer"] for row in rows}
    if len(reviewers) != 1:
        raise ValueError(f"{path}: expected one reviewer identity, found {sorted(reviewers)}")
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        duplicates = sorted(
            sample_id
            for sample_id, count in Counter(row["sample_id"] for row in rows).items()
            if count > 1
        )
        raise ValueError(f"{path}: duplicate sample IDs: {duplicates}")
    return next(iter(reviewers)), indexed


def expected_ids(pack: Path) -> set[str]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    return {
        row["sample_id"]
        for row in manifest["selected"]
        if row.get("lineage") == "new_v52"
    }


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_true = sum(left for left, _ in pairs) / len(pairs)
    right_true = sum(right for _, right in pairs) / len(pairs)
    expected = left_true * right_true + (1 - left_true) * (1 - right_true)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 6)


def compile_streams(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reviewer_a, stream_a = index_stream(args.reviewer_a, expected_count=args.expected_count)
    reviewer_b, stream_b = index_stream(args.reviewer_b, expected_count=args.expected_count)
    if reviewer_a == reviewer_b:
        raise ValueError("reviewer identities must be distinct")
    if set(stream_a) != set(stream_b):
        raise ValueError("reviewer streams have different sample IDs")
    if args.pack:
        expected = expected_ids(args.pack)
        if set(stream_a) != expected:
            missing = sorted(expected - set(stream_a))
            extra = sorted(set(stream_a) - expected)
            raise ValueError(f"stream IDs do not match pack; missing={missing}, extra={extra}")
    public_inputs = getattr(args, "public_inputs", None)
    public_rows = load_public_inputs(public_inputs) if public_inputs else {}
    if public_inputs and set(public_rows) != set(stream_a):
        raise ValueError("stream IDs do not match frozen public inputs")

    previous_report = getattr(args, "previous_report", None)
    rereviewed_form = getattr(args, "rereviewed_form", None)
    calibration = None
    if bool(previous_report) != bool(rereviewed_form):
        raise ValueError("calibration requires both --previous-report and --rereviewed-form")
    if previous_report:
        if not public_inputs:
            raise ValueError("calibration requires frozen public inputs")
        previous = json.loads(previous_report.read_text(encoding="utf-8"))
        for binding in previous["provenance"].values():
            if sha256(Path(binding["path"])) != binding["sha256"]:
                raise ValueError("previous preaudit provenance drift")
        if previous["provenance"]["public_inputs"]["sha256"] != sha256(public_inputs):
            raise ValueError("calibration must use the same frozen public inputs")
        reviewed_ids = {key for key, row in public_rows.items() if row["form"] == rereviewed_form}
        if not reviewed_ids:
            raise ValueError("rereviewed form is absent from the frozen public inputs")
        old_streams = []
        for side, current in (("a", stream_a), ("b", stream_b)):
            _, old = index_stream(Path(previous["provenance"][f"reviewer_{side}"]["path"]),
                                  expected_count=args.expected_count)
            if set(old) != set(current):
                raise ValueError("calibration changed the review denominator")
            for key in current:
                if key not in reviewed_ids and any(
                    current[key][field] != old[key][field] for field in REQUIRED_FIELDS - {"reviewer"}
                ):
                    raise ValueError("calibration changed a carried record outside the rereviewed form")
                if key in reviewed_ids and current[key]["reviewed_at_utc"] == old[key]["reviewed_at_utc"]:
                    raise ValueError("every record in the rereviewed form requires a new review timestamp")
            old_streams.append(old)
        old_agreements = sum(old_streams[0][key]["supported"] == old_streams[1][key]["supported"]
                             for key in stream_a)
        if (previous["records"] != len(stream_a) or previous["decision_agreements"] != old_agreements
                or previous["raw_decision_agreement"] != round(old_agreements / len(stream_a), 6)):
            raise ValueError("previous preaudit report disagrees with its original streams")
        calibration = {
            "kind": "full_form_rubric_clarification_with_other_forms_carried_unchanged",
            "previous_raw_decision_agreement": previous["raw_decision_agreement"],
            "previous_decision_agreements": old_agreements,
            "rereviewed_form": rereviewed_form, "rereviewed_records_per_reviewer": len(reviewed_ids),
            "carried_records_per_reviewer": len(stream_a) - len(reviewed_ids),
            "fresh_full_sample_independent_pass": False,
        }

    ids = sorted(stream_a)
    pairs = [(stream_a[key]["supported"], stream_b[key]["supported"]) for key in ids]
    decision_agreements = sum(left == right for left, right in pairs)
    exact_agreements = sum(
        (stream_a[key]["supported"], stream_a[key]["error_code"])
        == (stream_b[key]["supported"], stream_b[key]["error_code"])
        for key in ids
    )
    raw_agreement = decision_agreements / len(ids)
    gate_passed = raw_agreement > args.min_agreement
    kappa = cohen_kappa(pairs)

    queue: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for sample_id in ids:
        left, right = stream_a[sample_id], stream_b[sample_id]
        agrees = left["supported"] == right["supported"]
        low_confidence = min(float(left["confidence"]), float(right["confidence"])) < args.min_confidence
        ambiguous = any(
            "ambig" in str(row.get("error_code") or "").lower() for row in (left, right)
        )
        if not agrees:
            priority, reason = 0, "machine_disagreement"
        elif low_confidence or ambiguous:
            priority, reason = 1, "low_confidence_or_ambiguity"
        else:
            priority, reason = 2, "consensus_verification"
        reason_counts[reason] += 1
        queue.append(
            {
                "sample_id": sample_id,
                "priority": priority,
                "review_reason": reason,
                "requires_human_review": True,
                "machine_suggestion": left["supported"] if agrees else None,
                "reviewer_a": {
                    key: left[key]
                    for key in ("reviewer", "supported", "error_code", "concise_reason", "confidence")
                },
                "reviewer_b": {
                    key: right[key]
                    for key in ("reviewer", "supported", "error_code", "concise_reason", "confidence")
                },
            }
        )
    queue.sort(key=lambda row: (row["priority"], row["sample_id"]))

    report = {
        "schema": "anvil-machine-preaudit-v1",
        "status": "PREAUDIT_READY_FOR_HUMAN_REVIEW" if gate_passed else "PREAUDIT_BELOW_GATE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine_only": True,
        "may_populate_formal_human_metrics": False,
        "reviewers": {"a": reviewer_a, "b": reviewer_b},
        "records": len(ids),
        "decision_agreements": decision_agreements,
        "decision_disagreements": len(ids) - decision_agreements,
        "raw_decision_agreement": round(raw_agreement, 6),
        "exact_decision_error_agreement": round(exact_agreements / len(ids), 6),
        "cohen_kappa": kappa,
        "cohen_kappa_note": (
            "undefined under degenerate reviewer marginals" if kappa is None else None
        ),
        "thresholds": {
            "raw_decision_agreement_strictly_greater_than": args.min_agreement,
            "low_confidence_below": args.min_confidence,
        },
        "gate_passed": gate_passed,
        "human_review_queue": dict(reason_counts),
        "provenance": {
            "reviewer_a": {"path": str(args.reviewer_a), "sha256": sha256(args.reviewer_a)},
            "reviewer_b": {"path": str(args.reviewer_b), "sha256": sha256(args.reviewer_b)},
        },
        "limitations": [
            "Machine-machine agreement is not human agreement.",
            "Every queued record requires human confirmation or correction.",
            "Formal paper values remain needcheck until a human-signed report is compiled.",
        ],
    }
    if public_inputs:
        report["provenance"]["public_inputs"] = {
            "path": str(public_inputs), "sha256": sha256(public_inputs),
        }
        report["per_form"] = {}
        for form in sorted({row["form"] for row in public_rows.values()}):
            form_ids = [key for key in ids if public_rows[key]["form"] == form]
            form_pairs = [(stream_a[key]["supported"], stream_b[key]["supported"]) for key in form_ids]
            agreements = sum(a == b for a, b in form_pairs)
            report["per_form"][form] = {
                "records": len(form_ids), "decision_agreements": agreements,
                "raw_decision_agreement": round(agreements / len(form_ids), 6),
                "cohen_kappa": cohen_kappa(form_pairs),
                "both_supported": sum(a and b for a, b in form_pairs),
                "both_unsupported": sum(not a and not b for a, b in form_pairs),
            }
    if calibration:
        report["calibration"] = calibration
        report["provenance"]["previous_report"] = {
            "path": str(previous_report), "sha256": sha256(previous_report),
        }
        report["limitations"].append("Calibrated agreement carries unchanged votes outside the fully rereviewed form; it is not a fresh full-sample first pass.")
    return report, queue


def render_human_html(pack: Path, report: dict[str, Any], queue: list[dict[str, Any]]) -> str:
    """Package the public RGB proposals for offline, explicitly machine-aided review."""
    from quick_scannet_rgb_depth_v52_unique_referent_audit import V52QuickAuditStore

    store = V52QuickAuditStore(pack, "a")
    manifest_sha = sha256(pack / "manifest.json")
    positions = {item["sample_ids"]["depth"]: i for i, item in enumerate(store.selected)}
    cases = []
    for queued in queue:
        index = positions[queued["sample_id"]]
        form = store.case(index)["forms"][0]
        image_path = Path(store.selected[index]["image_path"])
        image_bytes = image_path.read_bytes()
        public = {"proposal": form["payload"], "groundings": form["gsa_context"]["groundings"]}
        public_sha = hashlib.sha256(
            image_bytes + json.dumps(public, sort_keys=True).encode("utf-8")
        ).hexdigest()
        # The full source/sample identity remains in the local frozen pack.
        opaque_id = hashlib.sha256((manifest_sha + queued["sample_id"]).encode()).hexdigest()[:24]
        mime = mimetypes.guess_type(image_path.name)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"unsupported offline image format: {image_path.suffix}")
        cases.append({
            "id": opaque_id, "public_sha256": public_sha, **public,
            "review_reason": queued.get("review_reason"),
            "image": f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii"),
            "machine": [queued["reviewer_a"], queued["reviewer_b"]],
        })
    payload = {
        "schema": "anvil-offline-human-review-v1", "pack_sha256": manifest_sha,
        "machine_stream_sha256": {k: report["provenance"][k]["sha256"] for k in ("reviewer_a", "reviewer_b")},
        "cases": cases, "error_codes": store.case(0)["error_codes"],
        "preaudit_agreement": report.get("raw_decision_agreement"),
        "preaudit_gate_passed": report.get("gate_passed"),
    }
    # Escape script terminators, including those inside an untrusted annotation reason.
    serialized = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    return OFFLINE_HTML.replace("__ANVIL_PAYLOAD__", serialized)


def render_public_human_html(public_inputs: Path, report: dict[str, Any], queue: list[dict[str, Any]]) -> str:
    indexed = load_public_inputs(public_inputs)
    if {row["sample_id"] for row in queue} != set(indexed):
        raise ValueError("queue IDs do not match public inputs")
    pack_sha = sha256(public_inputs)
    cases = []
    images = {}
    image_buffers = {}
    for queued in queue:
        row = indexed[queued["sample_id"]]
        image_path = Path(row["image_path"])
        if row["image_sha256"] not in image_buffers:
            image_buffers[row["image_sha256"]] = image_path.read_bytes()
        image_bytes = image_buffers[row["image_sha256"]]
        public = {"form": row["form"], "proposal": row["payload"], "groundings": row["groundings"]}
        if row.get("instruction"):
            public["instruction"] = row["instruction"]
        if row.get("spatial_context"):
            public["spatial_context"] = row["spatial_context"]
        public_sha = hashlib.sha256(image_bytes + json.dumps(public, sort_keys=True).encode()).hexdigest()
        mime = mimetypes.guess_type(image_path.name)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"unsupported offline image format: {image_path.suffix}")
        if row["image_sha256"] not in images:
            images[row["image_sha256"]] = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
        cases.append({
            "id": hashlib.sha256((pack_sha + row["sample_id"]).encode()).hexdigest()[:24],
            "public_sha256": public_sha, **public,
            "review_reason": queued.get("review_reason"),
            "image_id": row["image_sha256"],
            "machine": [queued["reviewer_a"], queued["reviewer_b"]],
        })
    errors = {
        "missing_object": "对象不可见", "wrong_label": "对象名称错误",
        "not_visible": "目标缺少可见证据",
        "bbox_misaligned": "边界框明显错位", "point_off_object": "点未落在对象上",
        "mask_inaccurate": "轮廓明显不准确", "depth_reversed": "远近顺序错误",
        "mask_misaligned": "轮廓偏移，同时漏掉目标并包含背景",
        "mask_omission": "轮廓漏掉明显可见的目标部分",
        "mask_excess": "轮廓包含明显不属于目标的区域",
        "mask_boundary_ambiguous": "轮廓边界缺少清晰可见证据",
        "ambiguous_depth": "缺少可判断远近的证据", "ambiguous_referent": "对象指代不明确",
        "action_unsupported": "动作缺少可见证据", "other": "其他问题（请说明）",
    }
    payload = {
        "schema": "anvil-offline-human-review-v1", "pack_sha256": pack_sha,
        "machine_stream_sha256": {k: report["provenance"][k]["sha256"] for k in ("reviewer_a", "reviewer_b")},
        "cases": cases, "images": images,
        "calibration": report.get("calibration"),
        "preaudit_agreement": report.get("raw_decision_agreement"),
        "preaudit_gate_passed": report.get("gate_passed"),
        "error_codes": [{"code": key, "label": label} for key, label in errors.items()],
    }
    return OFFLINE_HTML.replace("__ANVIL_PAYLOAD__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c"))


def validate_human_export(export: Path, html: Path) -> dict[str, Any]:
    """Bind returned decisions to the issued page without admitting paper metrics."""
    marker = '<script id="data" type="application/json">'
    content = html.read_text(encoding="utf-8")
    if content.count(marker) != 1:
        raise ValueError("expected one bound payload in the issued human-review page")
    issued = json.loads(content.split(marker, 1)[1].split("</script>", 1)[0])
    returned = json.loads(export.read_text(encoding="utf-8"))
    for key in ("schema", "pack_sha256", "machine_stream_sha256"):
        if returned.get(key) != issued[key]:
            raise ValueError(f"human export binding drift: {key}")
    if (returned.get("machine_aided_review") is not True
            or returned.get("independent_human_agreement_eligible") is not False
            or returned.get("may_populate_paper_metrics") is not False):
        raise ValueError("human export must preserve machine-aided provenance")
    if not isinstance(returned.get("reviewer"), str) or not returned["reviewer"].strip():
        raise ValueError("human export requires a reviewer identity")
    expected = {case["id"]: case for case in issued["cases"]}
    rows = returned.get("records")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("human export must retain every issued case, including pending cases")
    seen = set()
    complete = supported = exposed_before_decision = 0
    errors = {item["code"] for item in issued["error_codes"]}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("human export record must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected or case_id in seen:
            raise ValueError("human export contains an unknown or duplicate case ID")
        seen.add(case_id)
        if row.get("public_sha256") != expected[case_id]["public_sha256"]:
            raise ValueError(f"human export payload binding drift: {case_id}")
        decision = row.get("supported")
        comment = row.get("comment", "")
        if "supported" in row and not isinstance(decision, bool):
            raise ValueError("human export completion flag requires a Boolean decision")
        if not isinstance(comment, str):
            raise ValueError("human review comment must be text")
        if row.get("error_code") is not None and not isinstance(row["error_code"], str):
            raise ValueError("human review error code must be text or null")
        valid = isinstance(decision, bool) and (
            decision or (row.get("error_code") in errors and bool(comment.strip())))
        if row.get("complete") is not valid:
            raise ValueError(f"human export completion flag disagrees with decision: {case_id}")
        for flag in ("machine_suggestion_viewed", "machine_seen_before_valid_decision"):
            if flag in row and not isinstance(row[flag], bool):
                raise ValueError(f"human export provenance flag must be Boolean: {flag}")
        if row.get("machine_seen_before_valid_decision") is True:
            if row.get("machine_suggestion_viewed") is not True:
                raise ValueError("inconsistent machine-suggestion viewing history")
            exposed_before_decision += 1
        if valid:
            if decision and row.get("error_code") is not None:
                raise ValueError("supported human decisions cannot have an error code")
            try:
                reviewed = datetime.fromisoformat(row["reviewed_at_utc"].replace("Z", "+00:00"))
            except (KeyError, AttributeError, ValueError) as exc:
                raise ValueError("complete human decisions require an ISO timestamp") from exc
            if reviewed.tzinfo is None:
                raise ValueError("human review timestamp must include a time zone")
            complete += 1
            supported += int(decision)
    expected_status = ("HUMAN_REVIEW_PENDING_VERIFICATION" if complete == len(expected)
                       else "PARTIAL_HUMAN_REVIEW")
    if returned.get("status") != expected_status:
        raise ValueError("human export status disagrees with completed case count")
    return {
        "status": "HUMAN_EXPORT_BINDINGS_VALIDATED", "reviewer": returned["reviewer"],
        "records": len(expected), "complete": complete, "pending": len(expected) - complete,
        "supported": supported, "unsupported": complete - supported,
        "machine_seen_before_valid_decision": exposed_before_decision,
        "machine_aided_review": True, "independent_human_agreement_eligible": False,
        "may_populate_paper_metrics": False,
        "note": "Binding and format validation only; substantive adjudication remains required.",
        "provenance": {"export_sha256": sha256(export), "issued_html_sha256": sha256(html),
                       "pack_sha256": issued["pack_sha256"],
                       "machine_stream_sha256": issued["machine_stream_sha256"]},
    }


OFFLINE_HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anvil · 离线人工复核</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#17212c;font:16px/1.6 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #ccd4dc;padding:12px 4%;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:20px;margin:0}button,input,select,textarea{font:inherit}button{cursor:pointer;padding:8px 16px;border:1px solid #1459ac;border-radius:6px;background:#1459ac;color:white}
input,select,textarea{border:1px solid #aebac6;border-radius:5px;padding:6px}input{max-width:180px}
main{max-width:1200px;margin:auto;padding:20px}article{background:white;border:1px solid #d0d8e0;border-radius:10px;margin:22px 0;padding:20px;scroll-margin-top:130px}
h2{margin:0 0 12px;font-size:19px}.row{display:grid;grid-template-columns:minmax(0,3fr) minmax(270px,2fr);gap:20px}
.picture{position:relative;align-self:start}.picture img{width:100%;display:block}.picture svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.vote{display:flex;gap:20px;margin:15px 0}.vote input{margin-right:5px}textarea{width:100%;min-height:80px;margin-top:10px}
select{width:100%}details{margin-top:16px;padding-top:10px;border-top:1px solid #dbe0e6}summary{cursor:pointer}.muted{color:#526474;font-size:14px}
#notice{color:#a31d26}.proposal{padding:10px;background:#f2f5f9}.a{color:#c91d3d}.b{color:#1358b6}p{margin:8px 0}
.task-context{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;max-height:280px;overflow:auto}
@media(max-width:750px){.row{grid-template-columns:1fr}header{position:static}main{padding:12px}article{padding:12px}}
</style>
<header><h1>Anvil · 离线人工复核</h1><label>复核人代号 <input id="reviewer" maxlength="80" autocomplete="off"></label>
<strong id="progress"></strong><button id="export">导出复核结果</button><span id="notice" role="status"></span></header>
<main><p id="preaudit-status"></p><p>请逐条检查对象名称、边界框、点或轮廓是否符合图像；深度任务按 rank 从近到远检查可见证据。不能仅凭画面上下位置推断远近；不确定时选“不支持”并说明。叠加坐标范围为 0–1000。</p>
<p>轮廓任务请先隐藏图形标记，根据任务说明和原图确定对象。多个不同的可见对象都符合名称、且任务未说明具体是哪一个时，选择“对象指代不明确”；候选轮廓圈住其中一个不算消除歧义，仅凭大小、前景位置或显眼程度也不算。同一物体的组成部件（如剪刀的两片刀刃）不是多个独立对象。轮廓应包含明显可见的目标部分及附属部件；不要求补出被遮挡的不可见部分，允许轻微的边界近似。</p>
<p>操作任务同时检查接触部位、抓取姿态和动作计划。黄色点为候选接触、抓取或目标点；青色框为候选接触部位；青色线为夹爪闭合方向，长度仅用于显示，方向相差 180° 等价。抓取点须落在可见实体材料上，不能落在把手或环形结构的孔洞中心；闭合方向须横跨接触处的局部长轴。请结合图像检查，不能只按水平、垂直或斜向判断对错。</p>
<p class="muted">图片已内嵌，无需联网或访问端口。建议先判断，再展开机器建议。所有条目均需人工复核；机器建议可能出错。此页面用于辅助复核，其结果不计作独立、盲审的人际一致率。填写会尝试保存在当前浏览器，关闭前请导出 JSON；导出的结果仍需核验后才能替换论文占位指标。</p><div id="cases"></div></main>
<script id="data" type="application/json">__ANVIL_PAYLOAD__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("data").textContent);
const storageKey="anvil-review-"+data.pack_sha256+"-"+data.machine_stream_sha256.reviewer_a+"-"+data.machine_stream_sha256.reviewer_b;
let saved={reviewer:"",decisions:{}};
try{const prior=JSON.parse(localStorage.getItem(storageKey));if(prior&&typeof prior.reviewer==="string"&&prior.decisions&&typeof prior.decisions==="object")saved=prior}catch(e){}
const $=id=>document.getElementById(id);
if(typeof data.preaudit_agreement==='number')$("preaudit-status").textContent='机器预审原始一致率：'+(100*data.preaudit_agreement).toFixed(2)+'%（'+(data.preaudit_gate_passed?'达到':'未达到')+'本轮门槛）。分歧和歧义项已排在前面；所有条目仍需人工复核。';
if(data.calibration){const c=data.calibration;$("preaudit-status").textContent='首轮机器一致率 '+(100*c.previous_raw_decision_agreement).toFixed(2)+'%；统一 '+c.rereviewed_form+' 定义后为 '+(100*data.preaudit_agreement).toFixed(2)+'%（'+(data.preaudit_gate_passed?'达到':'未达到')+'本轮门槛）。每路重新查看该形式全部 '+c.rereviewed_records_per_reviewer+' 条，其余 '+c.carried_records_per_reviewer+' 条原判断不变；这不是新一轮全量首审。所有条目仍需人工复核。'}
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const valid=d=>!!d&&typeof d.supported==="boolean"&&(d.supported||(data.error_codes.some(x=>x.code===d.error_code)&&!!(d.comment||"").trim()));
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(saved));$("notice").textContent=""}catch(e){$("notice").textContent="本地保存不可用，请及时导出"}updateProgress()}
function updateProgress(){$("progress").textContent=data.cases.filter(c=>valid(saved.decisions[c.id])).length+" / "+data.cases.length+" 已复核"}
$("reviewer").value=saved.reviewer;$("reviewer").oninput=()=>{saved.reviewer=$("reviewer").value.trim();persist()};
$("cases").innerHTML=data.cases.map((c,i)=>{
 const d=saved.decisions[c.id]||{};
 const boxItems=c.form==='bbox'||c.form==='detection'?c.proposal.map(p=>({bbox:p.bbox_2d||p.bbox})):c.groundings;
 const boxes=boxItems.filter(g=>Array.isArray(g.bbox)&&g.bbox.length===4).map((g,j)=>{const [x,y,x2,y2]=g.bbox.map(Number);return '<rect x="'+x+'" y="'+y+'" width="'+(x2-x)+'" height="'+(y2-y)+'" fill="none" stroke="'+(j?'#2675ff':'#fa3056')+'" stroke-width="4"/>'}).join("");
 const shapes=c.proposal.map(p=>{
  const point=p.point_2d||p.contact_point_2d||p.grasp_point_2d||p.target_point_2d;
  let out=Array.isArray(point)?'<circle cx="'+Number(point[0])+'" cy="'+Number(point[1])+'" r="9" fill="#ffe600" stroke="#111" stroke-width="2"/>':'';
  if(c.form==='affordance'&&Array.isArray(p.bbox_2d||p.bbox)){const [x,y,x2,y2]=(p.bbox_2d||p.bbox).map(Number);out+='<rect x="'+x+'" y="'+y+'" width="'+(x2-x)+'" height="'+(y2-y)+'" fill="none" stroke="#00d5d5" stroke-width="5"/>'}
  if(Array.isArray(point)&&Array.isArray(p.jaw_axis_2d)){const [dx,dy]=p.jaw_axis_2d.map(Number),n=Math.hypot(dx,dy);if(n>0&&Number.isFinite(n)){const x=Number(point[0]),y=Number(point[1]);out+='<line x1="'+(x-55*dx/n)+'" y1="'+(y-55*dy/n)+'" x2="'+(x+55*dx/n)+'" y2="'+(y+55*dy/n)+'" stroke="#00d5d5" stroke-width="6"/>'}}
  if(Array.isArray(p.polygon))out+='<polygon points="'+p.polygon.map(x=>x.map(Number).join(',')).join(' ')+'" fill="#ffe60033" stroke="#ffe600" stroke-width="4"/>';
  return out;
 }).join('');
 const options=data.error_codes.map(x=>'<option value="'+esc(x.code)+'" '+(d.error_code===x.code?'selected':'')+'>'+esc(x.label)+'</option>').join("");
 const proposals='<button type="button" class="overlay-toggle" aria-pressed="true">隐藏图形标记</button>'+(c.form?'<p><strong>'+esc(c.form)+'</strong></p>':'')+c.proposal.map((p,j)=>'<p class="'+(j?'b':'a')+'">'+esc(p.label||p.object_class)+' '+(p.rank!=null?'· rank '+esc(p.rank)+'（从近到远）':'')+'</p>'+(c.form&&c.form!=='depth'?'<pre style="white-space:pre-wrap">'+esc(JSON.stringify(p,null,2))+'</pre>':'')).join("");
 const context=c.instruction?'<p><strong>任务说明与输入上下文</strong></p><div class="task-context">'+esc(c.instruction)+'</div>':c.spatial_context?'<p><strong>任务提供的上游上下文</strong></p><pre class="task-context">'+esc(JSON.stringify(c.spatial_context,null,2))+'</pre>':'';
 const hints=c.machine.map((m,j)=>'<p><strong>Agent '+(j?'B':'A')+'：'+(m.supported?'支持':'不支持')+'</strong> '+esc(m.error_code||'')+'</p><p>'+esc(m.concise_reason)+'</p>').join("");
 const priority={machine_disagreement:'优先复查：两位 agent 判断不同',low_confidence_or_ambiguity:'优先复查：置信度较低或存在歧义',consensus_verification:'核对两位 agent 的一致判断'}[c.review_reason]||'';
 return '<article data-id="'+c.id+'"><h2>复核 '+(i+1)+'</h2><p class="muted">'+priority+'</p><div class="row"><div class="picture"><img loading="lazy" src="'+(c.image||(data.images||{})[c.image_id])+'" alt="复核图像 '+(i+1)+'"><svg viewBox="0 0 1000 1000" preserveAspectRatio="none">'+boxes+shapes+'</svg></div><div>'+context+'<div class="proposal">'+proposals+'</div><div class="vote"><label><input type="radio" name="'+c.id+'" value="true" '+(d.supported===true?'checked':'')+'>支持</label><label><input type="radio" name="'+c.id+'" value="false" '+(d.supported===false?'checked':'')+'>不支持</label></div><select aria-label="错误类型" '+(d.supported===false?'':'disabled')+'><option value="">不支持时，请选择错误类型</option>'+options+'</select><textarea maxlength="2000" aria-label="复核说明" placeholder="不支持时，请填写具体可见问题">'+esc(d.comment||'')+'</textarea><details><summary>查看两位 agent 的建议</summary>'+hints+'</details></div></div></article>';
}).join("");
document.querySelectorAll("article").forEach(card=>{
 const id=card.dataset.id,select=card.querySelector("select"),comment=card.querySelector("textarea");
 const overlayButton=card.querySelector('.overlay-toggle');overlayButton.onclick=()=>{const show=overlayButton.getAttribute('aria-pressed')!=='true';card.querySelector('svg').style.visibility=show?'visible':'hidden';overlayButton.setAttribute('aria-pressed',String(show));overlayButton.textContent=show?'隐藏图形标记':'显示图形标记'};
 function record(){const d=saved.decisions[id]||(saved.decisions[id]={});const checked=card.querySelector("input:checked");if(checked)d.supported=checked.value==="true";select.disabled=d.supported!==false;d.error_code=d.supported===false?select.value:null;d.comment=comment.value.trim();d.reviewed_at_utc=new Date().toISOString();persist()}
 card.querySelectorAll("input").forEach(input=>input.onchange=record);select.onchange=record;comment.oninput=record;
 card.querySelector("details").ontoggle=event=>{if(event.target.open){const d=saved.decisions[id]||(saved.decisions[id]={});d.machine_suggestion_viewed=true;if(!valid(d))d.machine_seen_before_valid_decision=true;persist()}};
});
$("export").onclick=()=>{
 if(!saved.reviewer){$("notice").textContent="请先填写复核人代号";$("reviewer").focus();return}
 const records=data.cases.map(c=>({case_id:c.id,public_sha256:c.public_sha256,...(saved.decisions[c.id]||{}),complete:valid(saved.decisions[c.id])}));
 const result={schema:data.schema,pack_sha256:data.pack_sha256,machine_stream_sha256:data.machine_stream_sha256,reviewer:saved.reviewer,exported_at_utc:new Date().toISOString(),status:records.every(r=>r.complete)?"HUMAN_REVIEW_PENDING_VERIFICATION":"PARTIAL_HUMAN_REVIEW",machine_aided_review:true,independent_human_agreement_eligible:false,may_populate_paper_metrics:false,records};
 const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+"\n"],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="anvil-human-review-"+new Date().toISOString().slice(0,10)+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
updateProgress();
</script></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--pack", type=Path)
    inputs.add_argument("--public-inputs", type=Path, help="Frozen visible proposal JSONL")
    parser.add_argument("--prepare-flywheel", type=Path, help="Prepare --public-inputs from a raw/ directory, then exit")
    parser.add_argument("--prepare-native-candidates", type=Path, help="Sample frozen native candidate labels for review")
    parser.add_argument("--sample-per-form", type=int, default=24)
    parser.add_argument("--sample-forms", nargs="+", help="Review only these new forms, preserving the full source binding")
    parser.add_argument("--sample-seed", default="anvil-native-review-v1")
    parser.add_argument("--exclude-public-inputs", type=Path, action="append", default=[],
                        help="Exclude prior review images and their source scene groups")
    parser.add_argument("--native-source", type=Path, help="Frozen native source JSONL for prediction preaudit")
    parser.add_argument("--native-predictions", nargs=2, action="append", metavar=("FAMILY", "JSONL"),
                        help="Select a specialist's own family; repeat for other families")
    parser.add_argument("--expected-count", type=int, default=33)
    parser.add_argument("--min-agreement", type=float, default=0.90)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--previous-report", type=Path, help="Bind the unchanged prior report for a transparent calibration round")
    parser.add_argument("--rereviewed-form", help="Entire form independently reassessed; other forms must remain unchanged")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--human-queue", type=Path)
    parser.add_argument("--human-html", type=Path, help="Optional self-contained offline human review page")
    parser.add_argument("--validate-human-export", type=Path,
                        help="Read-only validation of returned JSON against the issued --human-html")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_native_candidates:
        if not args.public_inputs or any((args.prepare_flywheel, args.native_source,
                args.native_predictions, args.reviewer_a, args.reviewer_b, args.report,
                args.human_queue, args.human_html, args.validate_human_export)):
            raise ValueError("candidate preparation requires only sampling options and public output")
        print(json.dumps(prepare_native_candidates(
            args.prepare_native_candidates, args.public_inputs, per_form=args.sample_per_form,
            seed=args.sample_seed, exclude=args.exclude_public_inputs, selected_forms=args.sample_forms), indent=2, sort_keys=True))
        return 0
    if args.sample_forms:
        raise ValueError("--sample-forms requires --prepare-native-candidates")
    if args.exclude_public_inputs:
        raise ValueError("--exclude-public-inputs requires --prepare-native-candidates")
    if args.validate_human_export:
        if not args.human_html or any((args.reviewer_a, args.reviewer_b, args.pack,
                                      args.public_inputs, args.prepare_flywheel,
                                      args.native_source, args.native_predictions,
                                      args.report, args.human_queue)):
            raise ValueError("--validate-human-export requires only the issued --human-html")
        print(json.dumps(validate_human_export(args.validate_human_export, args.human_html),
                         indent=2, sort_keys=True))
        return 0
    if args.native_source or args.native_predictions:
        if not all((args.native_source, args.native_predictions, args.public_inputs)) or any((
                args.prepare_flywheel, args.reviewer_a, args.reviewer_b, args.report,
                args.human_queue, args.human_html)):
            raise ValueError("native preparation requires only source, predictions and public output")
        records = prepare_native_predictions(
            args.native_source, [(family, Path(path)) for family, path in args.native_predictions],
            args.public_inputs)
        print(json.dumps({"records": records, "public_inputs_sha256": sha256(args.public_inputs),
                          "source_sha256": sha256(args.native_source),
                          "predictions_sha256": {family: sha256(Path(path))
                                                 for family, path in args.native_predictions}}))
        return 0
    if args.prepare_flywheel:
        if not args.public_inputs or any((args.reviewer_a, args.reviewer_b, args.report, args.human_queue, args.human_html)):
            raise ValueError("--prepare-flywheel requires only --public-inputs output")
        records = prepare_flywheel_inputs(args.prepare_flywheel, args.public_inputs)
        print(json.dumps({"records": records, "public_inputs_sha256": sha256(args.public_inputs)}))
        return 0
    if not all((args.reviewer_a, args.reviewer_b, args.report, args.human_queue)):
        raise ValueError("compilation requires --reviewer-a, --reviewer-b, --report and --human-queue")
    for output in (args.report, args.human_queue, args.human_html):
        if output and output.exists():
            raise ValueError(f"refusing to overwrite issued review evidence: {output}")
    if not 0 <= args.min_agreement < 1:
        raise ValueError("--min-agreement must be in [0,1)")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be in [0,1]")
    report, queue = compile_streams(args)
    if args.human_html:
        if args.pack:
            html = render_human_html(args.pack, report, queue)
        elif args.public_inputs:
            html = render_public_human_html(args.public_inputs, report, queue)
        else:
            raise ValueError("--human-html requires --pack or --public-inputs")
        args.human_html.parent.mkdir(parents=True, exist_ok=True)
        args.human_html.write_text(html, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.human_queue.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.human_queue.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in queue), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
