#!/usr/bin/env python3
"""Audit exact letter-plus-option responses against immutable v3 predictions.

This is a post-hoc parser sensitivity analysis, never a replacement formal score.
The parser takes the response and presented choices, never the reference answer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import string
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval import general_retention_formal_v3 as formal

PROTOCOL = "general-retention-mc-compat-v1"
PARSER_CONTRACT = {
    "base_parser": "strict_letter_or_unique_exact_option_text_v1",
    "extension": "letter_followed_by_exact_text_of_that_presented_option",
    "accepted_separators": [".", ")", ":", "parenthesized_letter"],
    "normalization": "casefold_whitespace_collapse_trailing_period_strip_as_in_v3",
    "ground_truth_used_by_parser": False,
    "extra_explanation_or_conflicting_letter_and_text": "reject",
    "existing_valid_parses": "preserve",
    "textvqa_and_truncation_scoring": "preserve_v3",
    "health_thresholds": "preserve_v3_5_percent_malformed_1_percent_truncated",
}


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split()).rstrip(".")


def parse_multiple_choice(text: str, choices: list[str]) -> tuple[str | None, bool]:
    parsed, unclosed = formal.parse_multiple_choice(text, choices)
    if parsed is not None or unclosed:
        return parsed, unclosed
    span, unclosed = formal._answer_span(text)
    if span is None:
        return None, unclosed
    match = re.fullmatch(r"(?:\(([A-Z])\)\s*[.:]?|([A-Z])[.):])\s*(\S.*)",
                         span, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None, unclosed
    letter = (match.group(1) or match.group(2)).upper()
    index = string.ascii_uppercase.index(letter)
    if index >= len(choices) or _normalize(match.group(3)) != _normalize(str(choices[index])):
        return None, unclosed
    return letter, unclosed


def _rescore(result: dict[str, Any], runtime: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for binding in result["prediction_shards"]:
        path = formal.verify_file_binding(binding, deep=True)
        records.extend(formal.load_jsonl(path, "verified v3 predictions"))
    records.sort(key=lambda row: row["request_index"])
    if [r["global_uid"] for r in records] != [r["global_uid"] for r in runtime]:
        raise formal.FormalProtocolError("compat audit runtime/prediction UID mismatch")
    old_samples = {r["global_uid"]: r for r in result["per_sample"]}
    per_sample, changes = [], []
    for sample, prediction in zip(runtime, records):
        old = old_samples[sample["global_uid"]]
        row = dict(old)
        if sample["metric"] == "multiple_choice_accuracy":
            parsed, unclosed = parse_multiple_choice(prediction["response"], sample["choices"])
            row.update(credit=float(parsed == sample["answer"]), parsed=parsed is not None,
                       malformed=parsed is None)
            if row["truncated"] != (prediction["hit_token_cap"] or unclosed):
                raise formal.FormalProtocolError("compat parser changed truncation status")
            if row != old:
                changes.append({"global_uid": sample["global_uid"],
                                "response": prediction["response"],
                                "parsed_letter": parsed, "old_credit": old["credit"],
                                "new_credit": row["credit"]})
        per_sample.append(row)
    receipt = formal.load_json(Path(result["input_receipt"]["path"]), "input receipt")
    health = receipt["health_admission"]
    benchmarks = {}
    for dataset in formal.DATASET_ORDER:
        rows = [r for r in per_sample if r["dataset"] == dataset]
        malformed = sum(r["malformed"] for r in rows)
        capped = sum(r["truncated"] for r in rows)
        benchmarks[dataset] = {
            "n": len(rows), "score_percent": 100 * sum(r["credit"] for r in rows) / len(rows),
            "malformed_count": malformed, "truncated_count": capped,
            "within_original_health_bands": (
                100 * malformed / len(rows) <= health["max_malformed_percent"]
                and 100 * capped / len(rows) <= health["max_truncation_percent"]),
        }
    return {"model_role": result["model_role"],
            "checkpoint_content_identity_sha256": result["model_checkpoint_content_identity_sha256"],
            "original_status": result["status"],
            "original_raw_general_avg_percent": result["raw_general_avg_percent"],
            "compat_raw_general_avg_percent": sum(r["score_percent"] for r in benchmarks.values()) / 3,
            "benchmarks": benchmarks, "changed_count": len(changes), "changes": changes}


def audit_pair(base_path: Path, candidate_path: Path, *, allow_test_fixture: bool = False) -> dict[str, Any]:
    # Recompute BOTH original results under the original, unchanged scorer. Health
    # failure is retained; only invalid provenance/coverage or drift aborts the audit.
    base = formal._verify_result(base_path, allow_test_fixture=allow_test_fixture)
    candidate = formal._verify_result(candidate_path, allow_test_fixture=allow_test_fixture)
    if (base["model_role"], candidate["model_role"]) != ("base", "e1"):
        raise formal.FormalProtocolError("expected base and post-trained roles")
    if base["input_receipt"] != candidate["input_receipt"]:
        raise formal.FormalProtocolError("paired inputs differ")
    if base["model_checkpoint_content_identity_sha256"] == candidate["model_checkpoint_content_identity_sha256"]:
        raise formal.FormalProtocolError("paired models have identical content")
    receipt = formal.load_json(Path(base["input_receipt"]["path"]), "input receipt")
    runtime = formal._runtime_rows(receipt)
    a, b = _rescore(base, runtime), _rescore(candidate, runtime)
    denominator = a["compat_raw_general_avg_percent"]
    result = {
        "schema_version": 1, "protocol": PROTOCOL,
        "status": "POSTHOC_PARSER_SENSITIVITY_ONLY",
        "may_populate_formal_paper_metrics": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parser_contract": PARSER_CONTRACT,
        "parser_contract_sha256": formal.content_hash(PARSER_CONTRACT),
        "audit_code": formal.file_binding(Path(__file__)),
        "base_result": formal.file_binding(base_path),
        "candidate_result": formal.file_binding(candidate_path),
        "input_receipt": base["input_receipt"], "base": a, "candidate": b,
        "compat_raw_delta_points": b["compat_raw_general_avg_percent"] - denominator,
        "compat_raw_retention_percent": 100 * b["compat_raw_general_avg_percent"] / denominator if denominator else None,
        "limitations": ["Post-hoc grammar extension after observing v3 format failures.",
                        "Reuses predictions; this is not a new inference experiment.",
                        "Does not establish non-inferiority or teacher/distillation gains.",
                        "Original score files, health admission, and paper cells remain unchanged."],
    }
    result["identity_sha256"] = formal.content_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise formal.FormalProtocolError("refusing to overwrite existing compatibility audit")
    result = audit_pair(args.base_result, args.candidate_result)
    formal.atomic_json(args.out, result, idempotent=False)
    print(json.dumps({k: result[k] for k in ("status", "compat_raw_delta_points", "compat_raw_retention_percent")}, indent=2))


if __name__ == "__main__":
    main()
