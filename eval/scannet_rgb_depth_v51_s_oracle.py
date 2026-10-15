#!/usr/bin/env python3
"""Score the bound v5.1 formal40 S-oracle stage independently.

The paired G -> S protocol remains frozen.  This scorer provides a separate
held-out result for the already generated S-oracle bundle when the named-pair
G interface is not identifiable on most source frames.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
import random
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_PROTOCOL = "gsa-depth-v5.1-formal40-s-oracle-result-v1"
RESULT_STATUS = "FORMAL_S_ORACLE_SCORE_COMPLETE"
G_INTERFACE_STATUS = "INTERFACE_BLOCKED_NONUNIQUE_REFERENTS"
EXPECTED_RECORDS = 40
EXPECTED_UNIQUE_REFERENT_PAIRS = 7
EXPECTED_NONUNIQUE_REFERENT_PAIRS = 33
BOOTSTRAP_SEED = 51051
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95

FROZEN_CODE_SHA256 = {
    "scannet_rgb_depth_v51.py": (
        "eval/scannet_rgb_depth_v51.py",
        "6020f75d886a306b75445ae59753325c9bb5d34f215ae3fd2a9dea27c21f3b48",
    ),
    "scannet_rgb_depth_v51_dlc.py": (
        "eval/scannet_rgb_depth_v51_dlc.py",
        "107c050fffb98bb6911234000dde8c91bb10aed0f45749bd009539e8f9267048",
    ),
    "runner.py": (
        "eval/gsa_eval/runner.py",
        "0b25b84eeda1e7519aedb5a87263df84ffb4525fae7718e97e410835ba2186ac",
    ),
    "run_scannet_rgb_depth_v51_eval_l20_4gpu.sh": (
        "scripts/run_scannet_rgb_depth_v51_eval_l20_4gpu.sh",
        "df1aa1b3605dc50ee9bd48a0b35282db03a1b033afbd4e635ee0cd3d12416880",
    ),
}


class OracleScoreError(ValueError):
    """Raised when a standalone S-oracle result cannot be trusted."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise OracleScoreError(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise OracleScoreError(f"missing file: {path}")
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": file_sha256(path)}


def _atomic_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise OracleScoreError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OracleScoreError(f"invalid {label}: {path} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise OracleScoreError(f"{label} must be a JSON object: {path}")
    return value


def _frozen_code_bindings() -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for name, (relative, expected_sha256) in FROZEN_CODE_SHA256.items():
        binding = file_binding(ROOT / relative)
        if binding["sha256"] != expected_sha256:
            raise OracleScoreError(f"frozen evaluation code drift: {name}")
        bindings[name] = binding
    return bindings


def _verify_receipt_code_binding(
    receipt: dict[str, Any], live_bindings: dict[str, dict[str, Any]]
) -> None:
    evaluation_code = receipt.get("evaluation_code")
    if not isinstance(evaluation_code, dict):
        raise OracleScoreError("input receipt lacks evaluation code binding")
    if evaluation_code.get("files") != live_bindings:
        raise OracleScoreError("input receipt frozen evaluation code binding drift")
    if evaluation_code.get("identity_sha256") != canonical_hash(live_bindings):
        raise OracleScoreError("input receipt evaluation code identity drift")


def _load_frozen_modules() -> tuple[Any, Any, dict[str, dict[str, Any]]]:
    live_bindings = _frozen_code_bindings()
    pair_eval = importlib.import_module("eval.scannet_rgb_depth_v51")
    dlc = importlib.import_module("eval.scannet_rgb_depth_v51_dlc")
    return pair_eval, dlc, live_bindings


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise OracleScoreError("cannot compute a percentile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_binary_ci(
    outcomes: list[int],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    if not outcomes or any(value not in (0, 1) for value in outcomes):
        raise OracleScoreError("scene bootstrap requires nonempty binary outcomes")
    if samples <= 0 or not 0.0 < confidence < 1.0:
        raise OracleScoreError("invalid bootstrap configuration")
    rng = random.Random(seed)
    n = len(outcomes)
    means = [sum(outcomes[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    alpha = 1.0 - confidence
    return {
        "method": "deterministic_percentile_bootstrap",
        "unit": "scene",
        "scenes": n,
        "samples": samples,
        "seed": seed,
        "confidence": confidence,
        "ci": [
            round(_percentile(means, alpha / 2.0), 6),
            round(_percentile(means, 1.0 - alpha / 2.0), 6),
        ],
    }


def wilson_binomial_ci(
    successes: int, total: int, *, confidence: float = CONFIDENCE
) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total or not 0.0 < confidence < 1.0:
        raise OracleScoreError("invalid binomial interval inputs")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "method": "wilson_score_binomial",
        "successes": successes,
        "trials": total,
        "confidence": confidence,
        "ci": [round(max(0.0, centre - radius), 6), round(min(1.0, centre + radius), 6)],
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _key_prediction_rows(
    rows: Iterable[dict[str, Any]], expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        eval_id = row.get("eval_id")
        if not isinstance(eval_id, str) or eval_id in keyed:
            raise OracleScoreError("S-oracle predictions contain a missing or duplicate eval_id")
        if not isinstance(row.get("response"), str):
            raise OracleScoreError(f"S-oracle/{eval_id}: response is not a string")
        if not isinstance(row.get("hit_token_cap"), bool):
            raise OracleScoreError(f"S-oracle/{eval_id}: token-cap metadata is absent")
        keyed[eval_id] = row
    if set(keyed) != expected_ids:
        raise OracleScoreError("S-oracle predictions do not exactly cover the formal source")
    return keyed


def _summarise_stratum(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    correct = sum(int(item["correct"]) for item in items)
    parsed = sum(int(item["parsed"]) for item in items)
    uncapped_parsed = sum(int(item["parsed"] and not item["hit_token_cap"]) for item in items)
    capped = sum(int(item["hit_token_cap"]) for item in items)
    return {
        "n_scenes": total,
        "correct": correct,
        "accuracy": _ratio(correct, total),
        "parsed_responses": parsed,
        "parse_coverage": _ratio(parsed, total),
        "uncapped_parsed_responses": uncapped_parsed,
        "uncapped_parse_coverage": _ratio(uncapped_parsed, total),
        "capped_responses": capped,
        "cap_rate": _ratio(capped, total),
        "accuracy_binomial_ci": wilson_binomial_ci(correct, total),
    }


def score_s_oracle_predictions(
    pairs: Iterable[Any],
    rows: Iterable[dict[str, Any]],
    *,
    parse_choice: Callable[[Any], Any],
    provenance: dict[str, Any] | None = None,
    interface_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_list = list(pairs)
    if not pair_list:
        raise OracleScoreError("S-oracle score requires source scenes")
    pair_ids = [pair.eval_id for pair in pair_list]
    if len(set(pair_ids)) != len(pair_ids):
        raise OracleScoreError("formal source contains duplicate eval IDs")
    predictions = _key_prediction_rows(rows, set(pair_ids))

    parse_status: Counter[str] = Counter()
    per_item: list[dict[str, Any]] = []
    outcomes: list[int] = []
    for pair in pair_list:
        row = predictions[pair.eval_id]
        parsed = parse_choice(row["response"])
        is_parsed = parsed.choice is not None
        hit_cap = row["hit_token_cap"]
        correct = bool(not hit_cap and parsed.choice == pair.nearer_choice)
        parse_status[parsed.status] += 1
        outcomes.append(int(correct))
        per_item.append(
            {
                "eval_id": pair.eval_id,
                "target_stratum": pair.nearer_choice,
                "parse_status": parsed.status,
                "parsed": is_parsed,
                "hit_token_cap": hit_cap,
                "correct": correct,
            }
        )

    overall = _summarise_stratum(per_item)
    strata = {
        choice: _summarise_stratum(
            [item for item in per_item if item["target_stratum"] == choice]
        )
        for choice in ("A", "B")
    }
    if sum(item["n_scenes"] for item in strata.values()) != len(pair_list):
        raise OracleScoreError("S-oracle targets are not exclusively A/B")

    audit = interface_audit or {
        "status": "SYNTHETIC_INTERFACE_AUDIT_NOT_RUN",
        "unique_category_referent_pairs": None,
        "nonunique_category_referent_pairs": None,
    }
    report: dict[str, Any] = {
        "result_schema_version": 1,
        "protocol": RESULT_PROTOCOL,
        "evaluation_protocol": "gsa-depth-v5.1-formal40-paired-gs-eval-v1",
        "release": "formal40",
        "status": RESULT_STATUS if provenance is not None else "SYNTHETIC_S_ORACLE_SCORE_COMPLETE",
        "track": {
            "name": "S_oracle_grounded_pair",
            "status": "FORMAL_SCORE_COMPLETE" if provenance is not None else "SYNTHETIC_SCORE_COMPLETE",
            "context": "human_ground_truth_bounding_boxes",
        },
        "related_tracks": {
            "G_named_pair_bbox": {
                "status": G_INTERFACE_STATUS,
                "score_available": False,
                "diagnostic": audit,
                "interpretation": (
                    "The named-category interface is not identifiable on most frozen frames; "
                    "the existing G run is not a model-failure measurement."
                ),
            },
            "G_to_S_predicted_grounding": {
                "status": "NOT_SCORED_G_INTERFACE_BLOCKED",
                "score_available": False,
            },
        },
        "task_contract": {
            "scope": "held_out_s_oracle_only",
            "primary_unit": "scene",
            "all_item_denominator": len(pair_list),
            "capped_response_credit": 0,
            "oracle_context_source": "frozen_human_ground_truth_boxes",
            "training_materialization_eligible": False,
        },
        "metrics": overall,
        "ab_strata": strata,
        "confidence_intervals": {
            "accuracy_scene_bootstrap": bootstrap_binary_ci(outcomes),
            "accuracy_binomial": overall["accuracy_binomial_ci"],
        },
        "parse_status": dict(sorted(parse_status.items())),
        "per_item": per_item,
        "provenance": provenance or {"synthetic_fixture_only": True},
        "trust": {
            "score_available": provenance is not None,
            "complete_bound_s_oracle_bundle": provenance is not None,
            "formal_source_reverified": provenance is not None,
            "all_token_cap_metadata_complete": True,
            "zero_capped_responses": overall["capped_responses"] == 0,
            "held_out_evaluation_only": True,
            "training_materialization_eligible": False,
        },
    }
    report["identity_sha256"] = canonical_hash(report)
    return report


def _resolve_source_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise OracleScoreError("source diagnostic path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise OracleScoreError(f"source diagnostic file is missing: {path}")
    return path


def audit_named_referent_uniqueness(pair_eval: Any, release: Any) -> dict[str, Any]:
    wrappers_binding = release.source.binding.get("source_wrappers")
    wrappers_path = pair_eval.verify_file_binding(wrappers_binding, deep=True)
    wrappers = pair_eval.read_jsonl(wrappers_path)
    records: dict[str, dict[str, Any]] = {}
    for wrapper in wrappers:
        record = wrapper.get("source_record")
        identity = record.get("sample_id") if isinstance(record, dict) else None
        if not isinstance(identity, str) or identity in records:
            raise OracleScoreError("source wrapper identity drift during interface audit")
        records[identity] = record

    diagnostics: list[dict[str, Any]] = []
    for pair in release.source.pairs:
        record = records.get(pair.source_identity)
        if record is None:
            raise OracleScoreError(f"missing interface-audit source: {pair.eval_id}")
        measurements = record.get("sensor_measurements")
        if not isinstance(measurements, list) or len(measurements) != 2:
            raise OracleScoreError(f"invalid source measurements: {pair.eval_id}")
        categories = [item.get("category") for item in measurements if isinstance(item, dict)]
        if len(categories) != 2 or any(not isinstance(item, str) for item in categories):
            raise OracleScoreError(f"invalid source categories: {pair.eval_id}")

        sensor = record.get("sensor_provenance")
        if not isinstance(sensor, dict):
            raise OracleScoreError(f"missing sensor provenance: {pair.eval_id}")
        aggregation = sensor.get("aggregation")
        projection = sensor.get("instance_projection")
        if not isinstance(aggregation, dict) or not isinstance(projection, dict):
            raise OracleScoreError(f"missing referent provenance: {pair.eval_id}")

        aggregation_path = _resolve_source_path(aggregation.get("path"))
        if file_sha256(aggregation_path) != aggregation.get("sha256"):
            raise OracleScoreError(f"aggregation hash drift: {pair.eval_id}")
        try:
            aggregation_payload = json.loads(aggregation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OracleScoreError(f"invalid aggregation JSON: {pair.eval_id}") from exc
        groups = aggregation_payload.get("segGroups") if isinstance(aggregation_payload, dict) else None
        if not isinstance(groups, list):
            raise OracleScoreError(f"aggregation groups are absent: {pair.eval_id}")
        labels: dict[int, str] = {}
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("objectId"), int):
                raise OracleScoreError(f"invalid aggregation group: {pair.eval_id}")
            label = group.get("label")
            instance_id = group["objectId"] + 1
            if not isinstance(label, str) or instance_id in labels:
                raise OracleScoreError(f"ambiguous aggregation group: {pair.eval_id}")
            labels[instance_id] = label

        zip_path = _resolve_source_path(projection.get("zip_path"))
        member = projection.get("member")
        if not isinstance(member, str) or not member:
            raise OracleScoreError(f"instance projection member is absent: {pair.eval_id}")
        try:
            with zipfile.ZipFile(zip_path) as archive:
                member_bytes = archive.read(member)
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise OracleScoreError(f"cannot read instance projection: {pair.eval_id}") from exc
        if hashlib.sha256(member_bytes).hexdigest() != projection.get("sha256"):
            raise OracleScoreError(f"instance projection hash drift: {pair.eval_id}")
        try:
            with Image.open(io.BytesIO(member_bytes)) as image:
                visible_ids = set(image.convert("I").getdata())
        except OSError as exc:
            raise OracleScoreError(f"invalid instance projection image: {pair.eval_id}") from exc
        visible_ids.discard(0)
        visible_category_counts = Counter(labels.get(instance_id) for instance_id in visible_ids)
        target_counts = [visible_category_counts[category] for category in categories]
        diagnostics.append(
            {
                "eval_id": pair.eval_id,
                "target_category_visible_instance_counts": target_counts,
                "both_target_categories_unique": target_counts == [1, 1],
            }
        )

    unique = sum(int(item["both_target_categories_unique"]) for item in diagnostics)
    nonunique = len(diagnostics) - unique
    if (
        len(diagnostics) != EXPECTED_RECORDS
        or unique != EXPECTED_UNIQUE_REFERENT_PAIRS
        or nonunique != EXPECTED_NONUNIQUE_REFERENT_PAIRS
    ):
        raise OracleScoreError("frozen named-referent interface diagnostic drift")
    return {
        "status": G_INTERFACE_STATUS,
        "method": "visible_instance_projection_exact_aggregation_label_count_v1",
        "definition": "both named target categories have exactly one visible instance",
        "records": len(diagnostics),
        "unique_category_referent_pairs": unique,
        "nonunique_category_referent_pairs": nonunique,
        "unique_pair_rate": _ratio(unique, len(diagnostics)),
        "aggregation_files_rehashed": len(diagnostics),
        "instance_projection_members_rehashed": len(diagnostics),
        "diagnostic_records_sha256": canonical_hash(diagnostics),
        "source_wrappers": wrappers_binding,
    }


def score_command(args: argparse.Namespace) -> int:
    pair_eval, dlc, live_code = _load_frozen_modules()
    try:
        release = pair_eval.validate_request_release(args.release_manifest, verify_images=True)
        binding = dlc.verify_model_binding(args.model_binding, deep=True)
        bundle = dlc.verify_prediction_bundle(
            manifest_path=args.s_oracle_manifest,
            binding_path=args.model_binding,
            predictions_path=args.s_oracle_predictions,
            expected_track=pair_eval.TRACK_S_ORACLE,
            deep_model=False,
        )
    except (pair_eval.ContractError, dlc.RunnerError) as exc:
        raise OracleScoreError(f"frozen v5.1 provenance failed: {exc}") from exc

    if len(release.source.pairs) != EXPECTED_RECORDS or len(bundle["rows"]) != EXPECTED_RECORDS:
        raise OracleScoreError("standalone S-oracle score requires exactly 40 scenes")
    receipt = bundle["receipt"]
    _verify_receipt_code_binding(receipt, live_code)
    if receipt.get("training_materialization_eligible") is not False:
        raise OracleScoreError("input receipt lacks the held-out training prohibition")
    if receipt.get("formal_source") != release.source.binding:
        raise OracleScoreError("prediction receipt and live formal source differ")
    if (receipt.get("request_release") or {}).get("manifest") != file_binding(
        args.release_manifest
    ):
        raise OracleScoreError("prediction receipt and request release differ")
    if bundle["provenance"].get("model_binding_identity_sha256") != binding.get(
        "identity_sha256"
    ):
        raise OracleScoreError("S-oracle bundle and model binding identity differ")

    interface_audit = audit_named_referent_uniqueness(pair_eval, release)
    frozen_code = {"files": live_code, "identity_sha256": canonical_hash(live_code)}
    provenance: dict[str, Any] = {
        "verified": True,
        "scorer": file_binding(Path(__file__)),
        "frozen_evaluation_code": frozen_code,
        "request_release": file_binding(args.release_manifest),
        "request_release_identity_sha256": release.manifest["identity_sha256"],
        "formal_source": release.source.binding,
        "input_receipt": receipt,
        "model_binding": file_binding(args.model_binding),
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_inventory": binding["model"],
        "s_oracle_bundle": bundle["provenance"],
        "referent_interface_audit_identity_sha256": canonical_hash(interface_audit),
        "training_materialization_eligible": False,
    }
    provenance["identity_sha256"] = canonical_hash(provenance)
    report = score_s_oracle_predictions(
        release.source.pairs,
        bundle["rows"],
        parse_choice=pair_eval.parse_s_choice,
        provenance=provenance,
        interface_audit=interface_audit,
    )

    output = args.out.expanduser().resolve()
    if output.exists():
        existing = _read_json(output, "existing S-oracle result")
        if existing != report:
            raise OracleScoreError(f"refusing to overwrite a different result: {output}")
    else:
        _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score-s-oracle", help="score one complete bound formal40 S-oracle bundle")
    score.add_argument("--release-manifest", type=Path, required=True)
    score.add_argument("--model-binding", type=Path, required=True)
    score.add_argument("--s-oracle-predictions", type=Path, required=True)
    score.add_argument("--s-oracle-manifest", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.set_defaults(func=score_command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except OracleScoreError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
