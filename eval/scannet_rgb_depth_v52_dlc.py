#!/usr/bin/env python3
"""Four-L20 DLC runner bound to the formal ScanNet v5.2 paired evaluator.

The mature v5.1 receipt, model-inventory, shard, merge, and resume machinery is
reused verbatim after rebinding its module globals to the v5.2 protocol.  The
small v5.2-specific S-pred release below treats capped G output as invalid and
therefore never fabricates a downstream request.
"""
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import scannet_rgb_depth_v52_eval as pair_eval  # noqa: E402
from eval import scannet_rgb_depth_v52_interface as public_interface  # noqa: E402


_BASE_SPEC = importlib.util.spec_from_file_location(
    "eval._scannet_rgb_depth_v52_runner_base",
    ROOT / "eval/scannet_rgb_depth_v51_dlc.py",
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("cannot load the frozen v5.1 DLC runner base")
_base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = _base
_BASE_SPEC.loader.exec_module(_base)


RUN_PROTOCOL = "gsa-depth-v5.2-unique-referent-formal40-gs-intervention-dlc-v4"
PREDICTED_REQUEST_PROTOCOL = (
    "gsa-depth-v5.2-unique-referent-formal40-s-pred-request-v4"
)
NUM_SHARDS = 4
BATCH_SIZE = 4
STAGE_MAX_TOKENS = {
    pair_eval.TRACK_G: 64,
    pair_eval.TRACK_S_ORACLE: 8,
    pair_eval.TRACK_S_ABSENT: 8,
    pair_eval.TRACK_S_CORRUPTED: 8,
    pair_eval.TRACK_S_PRED: 8,
}
TRACK_ALIASES = {
    "g": pair_eval.TRACK_G,
    pair_eval.TRACK_G: pair_eval.TRACK_G,
    "s-oracle": pair_eval.TRACK_S_ORACLE,
    "s_oracle": pair_eval.TRACK_S_ORACLE,
    pair_eval.TRACK_S_ORACLE: pair_eval.TRACK_S_ORACLE,
    "s-absent": pair_eval.TRACK_S_ABSENT,
    "s_absent": pair_eval.TRACK_S_ABSENT,
    pair_eval.TRACK_S_ABSENT: pair_eval.TRACK_S_ABSENT,
    "s-corrupted": pair_eval.TRACK_S_CORRUPTED,
    "s_corrupted": pair_eval.TRACK_S_CORRUPTED,
    pair_eval.TRACK_S_CORRUPTED: pair_eval.TRACK_S_CORRUPTED,
    "s-pred": pair_eval.TRACK_S_PRED,
    "s_pred": pair_eval.TRACK_S_PRED,
    pair_eval.TRACK_S_PRED: pair_eval.TRACK_S_PRED,
}


def _bind_base_protocol() -> None:
    _base.pair_eval = pair_eval
    _base.RUN_PROTOCOL = RUN_PROTOCOL
    _base.PREDICTED_REQUEST_PROTOCOL = PREDICTED_REQUEST_PROTOCOL
    _base.NUM_SHARDS = NUM_SHARDS
    _base.BATCH_SIZE = BATCH_SIZE
    _base.STAGE_MAX_TOKENS = STAGE_MAX_TOKENS
    _base.TRACK_ALIASES = TRACK_ALIASES


_bind_base_protocol()

RunnerError = _base.RunnerError
RequestContext = _base.RequestContext
canonical_json = _base.canonical_json
content_hash = _base.content_hash
file_hash = _base.file_hash
file_binding = _base.file_binding
load_json = _base.load_json
load_jsonl = _base.load_jsonl
atomic_json = _base.atomic_json
atomic_jsonl = _base.atomic_jsonl
runtime_binding = _base.runtime_binding
model_inventory = _base.model_inventory
bind_model = _base.bind_model
verify_model_binding = _base.verify_model_binding
build_input_receipt = _base.build_input_receipt
verify_input_receipt = _base.verify_input_receipt
decoding_contract = _base.decoding_contract
infer_shard = _base.infer_shard
verify_shard = _base.verify_shard
merge_shards = _base.merge_shards
verify_prediction_bundle = _base.verify_prediction_bundle


def evaluation_code_binding() -> dict[str, Any]:
    paths = (
        Path(_base.__file__).resolve(),
        Path(__file__).resolve(),
        Path(pair_eval.__file__).resolve(),
        Path(public_interface.__file__).resolve(),
        pair_eval.INTERVENTION_CONTRACT_PATH.resolve(),
        ROOT / "eval/gsa_eval/runner.py",
        ROOT / "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh",
    )
    files = {path.name: file_binding(path) for path in paths}
    return {"files": files, "identity_sha256": content_hash(files)}


_base.evaluation_code_binding = evaluation_code_binding


def _build_input_receipt(release_manifest: Path, *, verify_images: bool) -> dict[str, Any]:
    release_manifest = release_manifest.expanduser().resolve()
    try:
        release = pair_eval.validate_request_release(
            release_manifest, verify_images=verify_images
        )
    except pair_eval.ContractError as exc:
        raise RunnerError(f"formal request release failed: {exc}") from exc
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "request_release": {
            "manifest": file_binding(release_manifest),
            "identity_sha256": release.manifest["identity_sha256"],
            "static_requests": release.manifest["requests"],
            "records": pair_eval.EXPECTED_RECORDS,
            "unique_images": pair_eval.EXPECTED_RECORDS,
            "label_balance": pair_eval.EXPECTED_LABEL_BALANCE,
            "blinded_model_fields": ["messages", "images"],
            "private_fields_passed_to_model": [],
        },
        "formal_source": release.source.binding,
        "generation": {
            "hardware": "exactly_4_physical_L20",
            "logical_shards": NUM_SHARDS,
            "shard_rule": "stable_request_order_stride_4",
            "image_max_token_num": 4096,
            "stage_decoding": {
                track: decoding_contract(track) for track in pair_eval.ALL_TRACKS
            },
            "same_bound_model_all_stages": True,
        },
        "evaluation_code": evaluation_code_binding(),
        "runtime": runtime_binding(),
        "training_materialization_eligible": False,
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def build_input_receipt(release_manifest: Path) -> dict[str, Any]:
    return _build_input_receipt(release_manifest, verify_images=True)


def _static_request_context(
    receipt_path: Path, binding_path: Path, track: str
) -> RequestContext:
    if track not in set(pair_eval.STATIC_TRACKS):
        raise RunnerError(f"{track}: not a static request track")
    receipt = verify_input_receipt(receipt_path)
    binding = verify_model_binding(binding_path, deep=False)
    if binding["input_receipt"] != file_binding(receipt_path):
        raise RunnerError("model binding and input receipt differ")
    release_binding = (receipt.get("request_release") or {}).get("manifest")
    release_path = _base._verify_file_binding(release_binding, deep=True)
    release = pair_eval.validate_request_release(release_path, verify_images=False)
    item = release.manifest["requests"][track]
    return RequestContext(
        track=track,
        request_manifest=file_binding(release_path),
        requests=item["file"],
        request_records_sha256=item["records_sha256"],
        rows=tuple(release.requests[track]),
    )


_base._build_input_receipt = _build_input_receipt
_base.build_input_receipt = build_input_receipt
_base._static_request_context = _static_request_context


def build_predicted_request_release(
    *,
    binding_path: Path,
    g_manifest_path: Path,
    g_predictions_path: Path,
    out: Path,
    manifest_out: Path,
) -> dict[str, Any]:
    binding_path = binding_path.expanduser().resolve()
    g_manifest_path = g_manifest_path.expanduser().resolve()
    g_predictions_path = g_predictions_path.expanduser().resolve()
    out = out.expanduser().resolve()
    manifest_out = manifest_out.expanduser().resolve()
    if out.exists() or manifest_out.exists():
        if not out.is_file() or not manifest_out.is_file():
            raise RunnerError("partial S-pred request release already exists")
        verified = verify_predicted_request_release(
            manifest_path=manifest_out,
            binding_path=binding_path,
            g_manifest_path=g_manifest_path,
            g_predictions_path=g_predictions_path,
            verify_images=True,
        )
        return {**verified["manifest"], "build_status": "reused_verified"}
    g_bundle = verify_prediction_bundle(
        manifest_path=g_manifest_path,
        binding_path=binding_path,
        predictions_path=g_predictions_path,
        expected_track=pair_eval.TRACK_G,
        deep_model=True,
    )
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for request, prediction in zip(g_bundle["requests"], g_bundle["rows"]):
        parsed = (
            pair_eval.GroundingParse(None, None, "hit_token_cap")
            if prediction["hit_token_cap"]
            else pair_eval.parse_grounding_response(prediction["response"])
        )
        if not parsed.valid:
            invalid.append({"eval_id": request["eval_id"], "status": parsed.status})
            continue
        rows.append(pair_eval.format_predicted_s_request(request, parsed))
    atomic_jsonl(out, rows)
    binding = verify_model_binding(binding_path, deep=False)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PREDICTED_REQUEST_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "track": pair_eval.TRACK_S_PRED,
        "input_receipt": binding["input_receipt"],
        "model_binding": file_binding(binding_path),
        "model_binding_identity_sha256": binding["identity_sha256"],
        "g_prediction_manifest": file_binding(g_manifest_path),
        "g_predictions": file_binding(g_predictions_path),
        "requests": file_binding(out),
        "request_records_sha256": pair_eval.canonical_hash(rows),
        "source_records": pair_eval.EXPECTED_RECORDS,
        "valid_grounding_records": len(rows),
        "invalid_groundings": invalid,
        "complete_g_coverage": len(g_bundle["rows"]) == pair_eval.EXPECTED_RECORDS,
        "training_materialization_eligible": False,
    }
    payload["identity_sha256"] = content_hash(payload)
    atomic_json(manifest_out, payload)
    return {**payload, "build_status": "generated"}


def verify_predicted_request_release(
    *,
    manifest_path: Path,
    binding_path: Path,
    g_manifest_path: Path | None = None,
    g_predictions_path: Path | None = None,
    verify_images: bool,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    manifest = load_json(manifest_path, "S-pred request manifest")
    if set(manifest) != _base._PREDICTED_REQUEST_MANIFEST_KEYS:
        raise RunnerError("S-pred request manifest schema drift")
    identity = manifest.get("identity_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise RunnerError("S-pred request manifest identity drift")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != PREDICTED_REQUEST_PROTOCOL
        or manifest.get("evaluation_protocol") != pair_eval.EVALUATION_PROTOCOL
        or manifest.get("track") != pair_eval.TRACK_S_PRED
        or manifest.get("source_records") != pair_eval.EXPECTED_RECORDS
        or manifest.get("complete_g_coverage") is not True
        or manifest.get("training_materialization_eligible") is not False
    ):
        raise RunnerError("S-pred request release contract drift")
    if manifest.get("model_binding") != file_binding(binding_path):
        raise RunnerError("S-pred request model binding drift")
    binding = verify_model_binding(binding_path, deep=False)
    if (
        manifest.get("model_binding_identity_sha256") != binding["identity_sha256"]
        or manifest.get("input_receipt") != binding["input_receipt"]
    ):
        raise RunnerError("S-pred request model/receipt identity drift")
    bound_g_manifest = _base._verify_file_binding(
        manifest.get("g_prediction_manifest"), deep=True
    )
    bound_g_predictions = _base._verify_file_binding(manifest.get("g_predictions"), deep=True)
    if g_manifest_path is not None and bound_g_manifest != g_manifest_path.expanduser().resolve():
        raise RunnerError("S-pred request G manifest argument drift")
    if g_predictions_path is not None and bound_g_predictions != g_predictions_path.expanduser().resolve():
        raise RunnerError("S-pred request G predictions argument drift")
    g_bundle = verify_prediction_bundle(
        manifest_path=bound_g_manifest,
        binding_path=binding_path,
        predictions_path=bound_g_predictions,
        expected_track=pair_eval.TRACK_G,
        deep_model=False,
    )
    expected_rows: list[dict[str, Any]] = []
    expected_invalid: list[dict[str, str]] = []
    for request, prediction in zip(g_bundle["requests"], g_bundle["rows"]):
        parsed = (
            pair_eval.GroundingParse(None, None, "hit_token_cap")
            if prediction["hit_token_cap"]
            else pair_eval.parse_grounding_response(prediction["response"])
        )
        if parsed.valid:
            expected_rows.append(pair_eval.format_predicted_s_request(request, parsed))
        else:
            expected_invalid.append({"eval_id": request["eval_id"], "status": parsed.status})
    request_path = _base._verify_file_binding(manifest.get("requests"), deep=True)
    rows = load_jsonl(request_path, "S-pred requests", allow_empty=True)
    if rows != expected_rows:
        raise RunnerError("S-pred requests do not exactly derive from verified G outputs")
    if (
        manifest.get("valid_grounding_records") != len(rows)
        or manifest.get("invalid_groundings") != expected_invalid
        or manifest.get("request_records_sha256") != pair_eval.canonical_hash(rows)
    ):
        raise RunnerError("S-pred request coverage or canonical hash drift")
    for row in rows:
        try:
            pair_eval.validate_public_request(row, expected_track=pair_eval.TRACK_S_PRED)
        except pair_eval.ContractError as exc:
            raise RunnerError(f"invalid S-pred public request: {exc}") from exc
        if verify_images:
            image = Path(row["images"][0])
            if file_hash(image) != row["image_sha256"]:
                raise RunnerError(f"S-pred request image hash drift: {row['eval_id']}")
    return {
        "manifest": manifest,
        "rows": rows,
        "invalid_groundings": expected_invalid,
        "g_bundle": g_bundle,
    }


_base.build_predicted_request_release = build_predicted_request_release
_base.verify_predicted_request_release = verify_predicted_request_release


def dry_run() -> dict[str, Any]:
    """CPU-only contract check; no production pack, model, or CUDA import is used."""
    strict = pair_eval.parse_grounding_response(
        '{"A":[1,2,100,200],"B":[300,4,900,800]}'
    )
    invalid = pair_eval.parse_grounding_response(
        '{"A":[1,2,100,200],"B":[300,4,900,800],"C":[1,2,3,4]}'
    )
    counts = [len(list(range(pair_eval.EXPECTED_RECORDS))[shard::NUM_SHARDS]) for shard in range(NUM_SHARDS)]
    passed = (
        strict.valid
        and not invalid.valid
        and counts == [10, 10, 10, 10]
        and set(STAGE_MAX_TOKENS) == {
            pair_eval.TRACK_G,
            pair_eval.TRACK_S_ORACLE,
            pair_eval.TRACK_S_ABSENT,
            pair_eval.TRACK_S_CORRUPTED,
            pair_eval.TRACK_S_PRED,
        }
    )
    return {
        "status": "DLC_DRY_RUN_PASS" if passed else "DLC_DRY_RUN_FAIL",
        "synthetic_or_contract_fixture_only": True,
        "records": pair_eval.EXPECTED_RECORDS,
        "stride_4_shard_counts": counts,
        "strict_parser": public_interface.INTERFACE_PROTOCOL,
        "invalid_g_emits_s_pred_request": False,
        "cuda_queried": False,
        "training_materialization_eligible": False,
    }


_base.dry_run = dry_run


def main(argv: list[str] | None = None) -> int:
    _bind_base_protocol()
    _base.evaluation_code_binding = evaluation_code_binding
    _base._build_input_receipt = _build_input_receipt
    _base.build_input_receipt = build_input_receipt
    _base._static_request_context = _static_request_context
    _base.build_predicted_request_release = build_predicted_request_release
    _base.verify_predicted_request_release = verify_predicted_request_release
    _base.dry_run = dry_run
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
