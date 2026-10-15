#!/usr/bin/env python3
"""Fail-fast audit for a contract-locked GSA GRPO dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel.contracts import validate_task_columns  # noqa: E402
from trains.flywheel.validators import validate, validate_structural  # noqa: E402


@contextmanager
def native_export_rows(path):
    """Read all spatial rows without concatenating or relabelling the export.

    Verify every stream, including auxiliary QA, against the complete export.
    QA was used for initial SFT and is not a spatial teacher-routing family.
    This validates representation/provenance, not teacher quality.
    """
    from tools.train_native_specialist import selected_streams
    from trains.flywheel.native_training import require, strict_json, validate_training_record
    manifest_raw = (path / "manifest.json").read_bytes()
    manifest = strict_json(manifest_raw)
    streams, _ = selected_streams(manifest, "joint", "record")
    require(sum(info["records"] for _, info in streams) == manifest["records"], "Export total mismatch")
    report = {"source_export_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
              "dataset_files": [str((path / name).resolve()) for name, info in streams if info["family"] != "QA"],
              "qa_replay_records_reserved_for_sft": sum(info["records"] for _, info in streams if info["family"] == "QA")}
    def rows():
        for name, info in streams:
            digest = hashlib.sha256(); count = 0
            with (path / name).open("rb") as stream:
                for raw in stream:
                    digest.update(raw); count += 1
                    require(raw.endswith(b"\n"), "Incomplete exported training row")
                    row = strict_json(raw)
                    require(row.get("gsa_form") == info["form"] and row.get("gsa_family") == info["family"]
                            and row.get("gsa_meta", {}).get("binding", {}).get("selection_sha256") == manifest["selection_sha256"],
                            "Export form/family/selection binding mismatch")
                    if info["family"] == "QA": validate_training_record(row)
                    else: yield raw.decode("utf-8")
            require(count == info["records"] and digest.hexdigest() == info["sha256"], "Training export stream changed: " + name)
        require((path / "manifest.json").read_bytes() == manifest_raw, "Export manifest changed during validation")
    yield rows(), report


@contextmanager
def dataset_rows(path, source_native):
    if source_native:
        with native_export_rows(path) as result: yield result
    else:
        with path.open(encoding="utf-8") as stream: yield stream, {}


def reference_tasks(record, anvil_router=None):
    """Validate every declared reference stage, without treating it as a rollout."""
    if anvil_router is None:
        answer = record["messages"][-1]["content"].strip()
        # Native SFT references use one opening stage delimiter. Preserve that
        # training serialization while validating its complete JSON payload.
        prefix = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}.get(record.get("gsa_family"))
        if prefix and answer.startswith(prefix):
            answer = answer[len(prefix):]
        payload = json.loads(answer)
        if "gsa_payload" in record and payload != record["gsa_payload"]:
            raise ValueError("reference answer and declared payload differ")
        return [(record["gsa_form"], payload, record["gsa_context"])]
    contract = anvil_router._record(record)
    if contract.record_type is anvil_router.contract.RecordType.ATOMIC:
        return [(record["gsa_form"], json.loads(contract.serialized_text), record["gsa_context"])]
    from trains.mopd.anvil_runtime import NATIVE_FORMS, parse_stage_segments
    stages = record["gsa_stages"]
    contexts = record.get("gsa_stage_contexts")
    if not isinstance(contexts, list) or len(contexts) != len(stages):
        raise ValueError("linked gsa_stage_contexts must cover all reference stages")
    tasks = []
    for index, (stage, _, _, payload, error) in enumerate(parse_stage_segments(contract.serialized_text, stages)):
        if error:
            raise ValueError(error)
        form = contract.stage_forms[anvil_router.contract.Stage(stage)]
        native = NATIVE_FORMS.get(form, form)
        check = dict(record, gsa_form=native, gsa_context=contexts[index])
        ok, reason = validate_task_columns(check)
        if not ok:
            raise ValueError(f"{stage}:{reason}")
        tasks.append((native, payload, contexts[index]))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", default="")
    ap.add_argument("--anvil", action="store_true", help="also require Anvil train metadata and validate linked reference stages")
    ap.add_argument("--alignment", action="store_true", help="validate v3 family-routed MOPD records, not v2 first-fault routing")
    ap.add_argument("--source-native", action="store_true", help="validate a complete native training export directory; reserve QA for SFT")
    ap.add_argument(
        "--allow-human-audit", action="store_true",
        help=("allow rows explicitly marked human_semantic_audit/PASS to use the "
              "structural validator; all other rows remain strict"),
    )
    args = ap.parse_args()
    if sum((args.alignment, args.anvil, args.source_native)) > 1:
        ap.error("alignment v3, frozen v2 and source-native admission are different protocols")
    anvil_router = None
    if args.anvil:
        from trains.mopd.anvil_runtime import ContractV2BatchRouter
        anvil_router = ContractV2BatchRouter()

    counts = Counter()
    errors = Counter()
    examples = []
    total = 0
    path = Path(args.dataset)
    if not (path.is_dir() if args.source_native else path.is_file()):
        raise SystemExit(f"dataset not found: {path}")

    with dataset_rows(path, args.source_native) as (f, export_report):
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors["invalid_jsonl"] += 1
                if len(examples) < 20:
                    examples.append({"line": line_no, "reason": "invalid_jsonl"})
                continue

            if args.source_native:
                from trains.flywheel.native_training import VERSION, validate_training_record
                try:
                    if record.get("gsa_contract_version") != VERSION:
                        raise ValueError("Expected a native spatial training row")
                    validate_training_record(record)
                    counts[record["gsa_form"]] += 1
                except (ValueError, KeyError, TypeError):
                    errors["invalid_source_native_record"] += 1
                continue
            if args.alignment:
                from trains.flywheel.alignment_schema import validate_training_record
                try:
                    validate_training_record(record)
                    counts[record["gsa_form"]] += 1
                except (ValueError, KeyError, TypeError):
                    errors["invalid_alignment_record"] += 1
                continue
            ok, reason = validate_task_columns(record)
            if not ok:
                errors[reason] += 1
                if len(examples) < 20:
                    examples.append({"line": line_no, "reason": reason})
                continue

            form = record["gsa_form"]
            counts[form] += 1
            try:
                tasks = reference_tasks(record, anvil_router)
            except (IndexError, KeyError, TypeError, ValueError):
                errors["invalid_training_answer"] += 1
                if len(examples) < 20:
                    examples.append({"line": line_no, "reason": "invalid_training_answer"})
                continue

            meta = record.get("gsa_meta") or {}
            human_audit = (
                args.allow_human_audit
                and meta.get("label_provenance") == "human_semantic_audit"
                and str(meta.get("semantic_audit_verdict", "")).upper() == "PASS"
                and meta.get("audit_reviewed") is True
            )
            for task_form, payload, context in tasks:
                if isinstance(payload, dict):
                    payload = [payload]
                validator = validate_structural if human_audit else validate
                valid, contract_reason = validator(task_form, payload, context)
                if not valid:
                    key = f"contract:{task_form}:{contract_reason}" if args.anvil else f"contract:{contract_reason}"
                    errors[key] += 1
                    if len(examples) < 20:
                        examples.append({"line": line_no, "reason": key})

    report = {
        **export_report,
        "dataset": str(path),
        "records": total,
        "by_form": dict(sorted(counts.items())),
        "errors": dict(errors.most_common()),
        "examples": examples,
        "verdict": "PASS" if total > 0 and not errors else "FAIL",
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output + "\n", encoding="utf-8")
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
