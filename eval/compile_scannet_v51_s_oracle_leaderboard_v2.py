#!/usr/bin/env python3
"""Compile the seven-model formal40 v5.1 S-oracle leaderboard.

Version 2 extends the frozen five-model leaderboard with two cross-family
checkpoints.  It independently revalidates and live-rehashes all seven result
bundles, while pinning the historical five-model artifact and requiring its
five row projections to remain byte-for-value equivalent.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import compile_scannet_v51_s_oracle_leaderboard as five


EXPECTED_MODELS = 7
LEADERBOARD_PROTOCOL = (
    "gsa-depth-v5.1-formal40-s-oracle-seven-model-leaderboard-v2"
)
LEADERBOARD_STATUS = "FORMAL_SEVEN_MODEL_LEADERBOARD_COMPLETE"
DEFAULT_OUT = ROOT / "artifacts/results/scannet_v51_s_oracle_leaderboard_v2.json"
BASE_VALIDATOR = ROOT / "eval/compile_scannet_v51_s_oracle_leaderboard.py"
BASE_VALIDATOR_SHA256 = (
    "e87a87ad1958a442e977bc479e916d7884c3e2065411067d0fd1bffb7dcb459a"
)

LeaderboardError = five.LeaderboardError
ModelSpec = five.ModelSpec


@dataclass(frozen=True)
class HistoricalAnchorSpec:
    path: Path
    expected_file_sha256: str
    expected_identity_sha256: str


DEFAULT_HISTORICAL_ANCHOR = HistoricalAnchorSpec(
    path=ROOT / "artifacts/results/scannet_v51_s_oracle_leaderboard.json",
    expected_file_sha256=(
        "2f490523452e4eaaced1c70192b3dabea37d33047cb00009c167019269376608"
    ),
    expected_identity_sha256=(
        "de32bdcec5c1cc5d7c960bfa176c8eb215314b57315311a698d9e86ee542e6f8"
    ),
)


DEFAULT_MODEL_SPECS = five.DEFAULT_MODEL_SPECS + (
    ModelSpec(
        key="internvl3_5_8b",
        display_name="InternVL3.5-8B-HF",
        role="general_model_cross_family",
        result_path=(
            ROOT
            / "output/scannet_rgb_depth_v51_s_oracle_internvl35_8b_20260823_v2/s_oracle/result.json"
        ),
        expected_model_inventory_identity_sha256=(
            "6fb636e94996db31de7b504dc5884d019a8bbdd99a07e8da22bf3ddca951cc76"
        ),
    ),
    ModelSpec(
        key="llava_onevision_qwen2_7b",
        display_name="LLaVA-OneVision-Qwen2-7B-OV-HF",
        role="general_model_cross_family",
        result_path=(
            ROOT
            / "output/scannet_rgb_depth_v51_s_oracle_llavaov7b_20260823_v1/s_oracle/result.json"
        ),
        expected_model_inventory_identity_sha256=(
            "cd27c300bb776462bd741c2c25f0260d4b5f58bd610b9d48c59adf2e218661cc"
        ),
    ),
)

HISTORICAL_KEYS = tuple(spec.key for spec in five.DEFAULT_MODEL_SPECS)
EXPECTED_KEYS = frozenset(spec.key for spec in DEFAULT_MODEL_SPECS)


def _validator_binding() -> dict[str, Any]:
    ledger = five.RehashLedger()
    expected = {
        "path": str(BASE_VALIDATOR.resolve()),
        "bytes": BASE_VALIDATOR.stat().st_size,
        "sha256": BASE_VALIDATOR_SHA256,
    }
    ledger.verify_binding(expected, "five_model_validator")
    return expected


def _read_historical_anchor(
    anchor: HistoricalAnchorSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = five.RehashLedger()
    report, binding = ledger.read_result(anchor.path)
    if binding["sha256"] != anchor.expected_file_sha256:
        raise LeaderboardError("historical five-model artifact file hash drift")
    five._verify_identity(report, "historical_five_model_artifact")
    five._require_equal(
        report.get("identity_sha256"),
        anchor.expected_identity_sha256,
        "historical_five_model_artifact.identity",
    )
    five._require_equal(
        report.get("protocol"),
        five.LEADERBOARD_PROTOCOL,
        "historical_five_model_artifact.protocol",
    )
    five._require_equal(
        report.get("status"),
        five.LEADERBOARD_STATUS,
        "historical_five_model_artifact.status",
    )
    rows = report.get("leaderboard")
    if not isinstance(rows, list) or len(rows) != five.EXPECTED_MODELS:
        raise LeaderboardError("historical five-model artifact row count drift")
    keys = [row.get("model_key") for row in rows if isinstance(row, dict)]
    if set(keys) != set(HISTORICAL_KEYS) or len(keys) != len(set(keys)):
        raise LeaderboardError("historical five-model artifact key set drift")
    live_rehash = report.get("live_rehash")
    if not isinstance(live_rehash, dict):
        raise LeaderboardError("historical five-model live-rehash receipt is absent")
    five._require_equal(
        live_rehash.get("status"), "LIVE_REHASH_PASS", "historical.live_rehash.status"
    )
    five._require_equal(
        live_rehash.get("explicit_result_files_rehashed"),
        five.EXPECTED_MODELS,
        "historical.live_rehash.result_files",
    )
    five._require_equal(
        live_rehash.get("all_size_sha256_and_bound_weight_mtime_checks_match"),
        True,
        "historical.live_rehash.binding_checks",
    )
    return report, binding


def _project_without_rank(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "accuracy_rank"}


def _build_rows(
    specs: Sequence[ModelSpec],
    results: Mapping[str, dict[str, Any]],
    result_bindings: Mapping[str, dict[str, Any]],
    item_maps: Mapping[str, Mapping[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    reference_items = item_maps[five.REFERENCE_KEY]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        result = results[spec.key]
        items = item_maps[spec.key]
        correct = sum(int(item["correct"]) for item in items.values())
        strata = {
            target: {
                "n": five.EXPECTED_LABEL_BALANCE[target],
                "correct": sum(
                    int(item["correct"])
                    for item in items.values()
                    if item["target_stratum"] == target
                ),
            }
            for target in ("A", "B")
        }
        for value in strata.values():
            value["accuracy"] = five._ratio(value["correct"], value["n"])
        inventory = result["provenance"]["model_inventory"]
        rows.append(
            {
                "model_key": spec.key,
                "display_name": spec.display_name,
                "role": spec.role,
                "model": {
                    "path": inventory["path"],
                    "architecture": inventory["architecture"],
                    "model_type": inventory["model_type"],
                    "inventory_identity_sha256": inventory["identity_sha256"],
                    "binding_identity_sha256": result["provenance"][
                        "model_binding_identity_sha256"
                    ],
                },
                "result": {
                    "file": result_bindings[spec.key],
                    "identity_sha256": result["identity_sha256"],
                },
                "n": five.EXPECTED_RECORDS,
                "correct": correct,
                "accuracy": five._ratio(correct, five.EXPECTED_RECORDS),
                "A": strata["A"],
                "B": strata["B"],
                "prediction_distribution": five._prediction_distribution(items),
                "vs_qwen3_5_9b": five._paired_comparison(items, reference_items),
            }
        )

    rows.sort(key=lambda row: (-row["accuracy"], row["model_key"]))
    previous_accuracy: float | None = None
    previous_rank = 0
    for index, row in enumerate(rows, 1):
        if previous_accuracy is None or row["accuracy"] != previous_accuracy:
            previous_rank = index
            previous_accuracy = row["accuracy"]
        row["accuracy_rank"] = previous_rank
    return rows


def compile_leaderboard(
    specs: Sequence[ModelSpec] = DEFAULT_MODEL_SPECS,
    historical_anchor: HistoricalAnchorSpec = DEFAULT_HISTORICAL_ANCHOR,
) -> dict[str, Any]:
    if len(specs) != EXPECTED_MODELS:
        raise LeaderboardError("formal v2 leaderboard requires exactly seven model results")
    keys = [spec.key for spec in specs]
    if frozenset(keys) != EXPECTED_KEYS or len(keys) != len(set(keys)):
        raise LeaderboardError("model keys must be the pinned seven-model key set")
    paths = [spec.result_path.expanduser().resolve() for spec in specs]
    if len(paths) != len(set(paths)):
        raise LeaderboardError("the seven model results must be distinct files")

    validator_binding = _validator_binding()
    historical, historical_binding = _read_historical_anchor(historical_anchor)

    ledger = five.RehashLedger()
    results: dict[str, dict[str, Any]] = {}
    result_bindings: dict[str, dict[str, Any]] = {}
    item_maps: dict[str, dict[str, dict[str, Any]]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    inventory_ids: set[str] = set()
    for spec in specs:
        result, binding = ledger.read_result(spec.result_path)
        item_maps[spec.key] = five._validate_result(spec, result)
        inventory_id = result["provenance"]["model_inventory"]["identity_sha256"]
        if inventory_id in inventory_ids:
            raise LeaderboardError(
                "the seven rows do not represent seven distinct model inventories"
            )
        inventory_ids.add(inventory_id)
        results[spec.key] = result
        result_bindings[spec.key] = binding
        contexts[spec.key] = five._common_context(result)

    reference_context = contexts[five.REFERENCE_KEY]
    for spec in specs:
        five._require_equal(
            contexts[spec.key],
            reference_context,
            f"{spec.key}.formal_comparability_context",
        )
    reference_items = item_maps[five.REFERENCE_KEY]
    eval_ids = sorted(reference_items)
    target_map = {
        eval_id: reference_items[eval_id]["target_stratum"] for eval_id in eval_ids
    }
    for spec in specs:
        candidate = item_maps[spec.key]
        five._require_equal(sorted(candidate), eval_ids, f"{spec.key}.eval_ids")
        five._require_equal(
            {eval_id: candidate[eval_id]["target_stratum"] for eval_id in eval_ids},
            target_map,
            f"{spec.key}.target_map",
        )

    for spec in specs:
        ledger.verify_all_bindings(results[spec.key], spec.key)

    rows = _build_rows(specs, results, result_bindings, item_maps)
    common = {
        "result_protocol": five.RESULT_PROTOCOL,
        "evaluation_protocol": five.EVALUATION_PROTOCOL,
        "run_protocol": five.RUN_PROTOCOL,
        "release": five.RELEASE,
        "track": five.TRACK,
        "records": five.EXPECTED_RECORDS,
        "eval_ids_sha256": five.canonical_hash(eval_ids),
        "target_map_sha256": five.canonical_hash(target_map),
        "label_balance": five.EXPECTED_LABEL_BALANCE,
        "formal_source_identity_sha256": reference_context["formal_source"][
            "identity_sha256"
        ],
        "request_release_identity_sha256": reference_context[
            "request_release_identity_sha256"
        ],
        "input_receipt_identity_sha256": reference_context["input_receipt"][
            "identity_sha256"
        ],
        "request_records_sha256": reference_context["request_records_sha256"],
        "decoding": reference_context["decoding"],
        "context_identity_sha256": five.canonical_hash(reference_context),
    }
    statistics_contract = {
        "accuracy_unit": "scene",
        "paired_delta_direction": "candidate_minus_qwen3_5_9b",
        "paired_bootstrap": {
            "method": "deterministic_percentile_bootstrap",
            "unit": "scene",
            "samples": five.BOOTSTRAP_SAMPLES,
            "seed": five.BOOTSTRAP_SEED,
            "confidence": five.CONFIDENCE,
        },
        "prediction_A_rate_derivation": (
            "The formal binary outputs are fully parsed and uncapped; a wrong prediction "
            "is therefore the opposite of target_stratum."
        ),
    }

    five._require_equal(
        historical.get("comparability", {}).get("common"),
        common,
        "historical_five_model_artifact.common_context",
    )
    five._require_equal(
        historical.get("statistics_contract"),
        statistics_contract,
        "historical_five_model_artifact.statistics_contract",
    )
    historical_rows = {
        row["model_key"]: row for row in historical["leaderboard"]
    }
    current_rows = {row["model_key"]: row for row in rows}
    for key in HISTORICAL_KEYS:
        five._require_equal(
            _project_without_rank(current_rows[key]),
            _project_without_rank(historical_rows[key]),
            f"historical_row_projection.{key}",
        )

    report: dict[str, Any] = {
        "schema_version": 2,
        "protocol": LEADERBOARD_PROTOCOL,
        "status": LEADERBOARD_STATUS,
        "reference_model_key": five.REFERENCE_KEY,
        "comparability": {
            "status": "PASS",
            "same_release_source_protocol_and_eval_ids": True,
            "all_seven_results_have_complete_formal_trust": True,
            "common": common,
        },
        "statistics_contract": statistics_contract,
        "leaderboard": rows,
        "lineage": {
            "historical_five_model_artifact": {
                "file": historical_binding,
                "identity_sha256": historical["identity_sha256"],
                "protocol": historical["protocol"],
                "status": historical["status"],
            },
            "five_model_validator": validator_binding,
            "historical_five_row_projection_match": True,
            "historical_rank_field_excluded_from_projection": True,
        },
        "live_rehash": ledger.report(),
        "trust": {
            "score_available": True,
            "leaderboard_rows": EXPECTED_MODELS,
            "historical_five_rows_preserved": True,
            "paired_scenes_per_comparison": five.EXPECTED_RECORDS,
            "formal_held_out_evaluation_only": True,
            "training_materialization_eligible": False,
        },
    }
    report["identity_sha256"] = five.canonical_hash(report)
    return report


def _specs_with_overrides(values: Iterable[str]) -> tuple[ModelSpec, ...]:
    overrides: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path or key in overrides:
            raise LeaderboardError(
                "--result must be a unique MODEL_KEY=/absolute/result.json"
            )
        overrides[key] = Path(path)
    if overrides and set(overrides) != EXPECTED_KEYS:
        raise LeaderboardError(
            "--result overrides must cover exactly: "
            + ", ".join(sorted(EXPECTED_KEYS))
        )
    return tuple(
        ModelSpec(
            key=spec.key,
            display_name=spec.display_name,
            role=spec.role,
            result_path=overrides.get(spec.key, spec.result_path),
            expected_model_inventory_identity_sha256=(
                spec.expected_model_inventory_identity_sha256
            ),
        )
        for spec in DEFAULT_MODEL_SPECS
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="MODEL_KEY=PATH",
        help="override all seven default result paths while retaining pinned identities",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = _specs_with_overrides(args.result)
        report = compile_leaderboard(specs)
        output = args.out.expanduser().resolve()
        if output.exists():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise LeaderboardError(f"existing output is invalid: {output}") from exc
            if existing != report:
                raise LeaderboardError(
                    f"refusing to overwrite a different leaderboard: {output}"
                )
        else:
            five._atomic_json(output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "out": str(output),
                    "identity_sha256": report["identity_sha256"],
                }
            )
        )
        return 0
    except LeaderboardError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
