#!/usr/bin/env python3
"""Build benchmark tables only from registered, protocol-compatible results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.gsa_eval.runner import evaluation_code_sha256  # noqa: E402


def nested(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def latex_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in value)


def audit_model_provenance(run: dict[str, Any]) -> list[str]:
    """Validate an optional pinned download receipt for a formal model."""
    audit_name = run.get("download_audit")
    if audit_name is None:
        return []
    if not isinstance(audit_name, str):
        return ["model_audit_path_invalid"]
    audit_path = Path(audit_name).expanduser()
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path
    try:
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes.decode("utf-8"))
    except Exception:
        return ["model_audit_unreadable"]

    reasons = []
    expected_audit_sha = run.get("download_audit_sha256")
    actual_audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    if not isinstance(expected_audit_sha, str) or expected_audit_sha != actual_audit_sha:
        reasons.append("model_audit_sha256_mismatch")
    if audit.get("schema_version") != 2:
        reasons.append("model_audit_schema_mismatch")
    expected_path = run.get("model_path")
    resolved_path = audit.get("resolved_path")
    if not isinstance(expected_path, str) or not isinstance(resolved_path, str):
        reasons.append("model_audit_resolved_path_missing")
    elif Path(expected_path).expanduser().resolve() != Path(resolved_path).expanduser().resolve():
        reasons.append("model_audit_resolved_path_mismatch")
    for run_key, audit_key, reason in (
        ("repo_id", "repo_id", "model_repo_mismatch"),
        ("model_revision", "requested_revision", "model_requested_revision_mismatch"),
        ("model_revision", "resolved_revision", "model_resolved_revision_mismatch"),
    ):
        if run.get(run_key) != audit.get(audit_key):
            reasons.append(reason)
    if audit.get("revision_is_pinned_commit") is not True:
        reasons.append("model_revision_not_pinned")
    inventory = audit.get("local_inventory")
    shards = audit.get("weight_shards")
    if not isinstance(inventory, list) or not inventory:
        reasons.append("model_audit_inventory_missing")
    else:
        paths = []
        for item in inventory:
            if not isinstance(item, dict):
                reasons.append("model_audit_inventory_invalid")
                break
            path = item.get("path")
            size = item.get("size_bytes")
            sha = item.get("sha256")
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or not isinstance(sha, str)
                or len(sha) != 64
                or any(char not in "0123456789abcdef" for char in sha)
            ):
                reasons.append("model_audit_inventory_invalid")
                break
            paths.append(path)
        if len(paths) != len(set(paths)):
            reasons.append("model_audit_inventory_duplicate_path")
        if (
            not isinstance(shards, list)
            or not shards
            or any(not isinstance(shard, str) or shard not in set(paths) for shard in shards)
        ):
            reasons.append("model_audit_weight_shards_invalid")
    return reasons


def audit_claim_evidence(
    run: dict[str, Any], benchmark_id: str
) -> tuple[list[str], dict[str, Any] | None]:
    """Bind an optional train/eval-separation audit to one reported cell."""
    config = run.get("claim_evidence_audit")
    if config is None:
        return [], None
    if not isinstance(config, dict):
        return ["claim_audit_config_invalid"], None
    path_value = config.get("path")
    cells = config.get("cells")
    cell_config = cells.get(benchmark_id) if isinstance(cells, dict) else None
    if not isinstance(path_value, str) or not isinstance(cell_config, dict):
        return ["claim_audit_cell_config_missing"], None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    try:
        raw = path.read_bytes()
        report = json.loads(raw.decode("utf-8"))
    except Exception:
        return ["claim_audit_unreadable"], None
    reasons: list[str] = []
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config.get("sha256"):
        reasons.append("claim_audit_sha256_mismatch")
    unsigned = dict(report)
    observed_identity = unsigned.pop("identity_sha256", None)
    if (
        observed_identity != config.get("identity_sha256")
        or canonical_hash(unsigned) != observed_identity
    ):
        reasons.append("claim_audit_identity_mismatch")
    if nested(report, "posttraining_corpus.sha256") != config.get("corpus_sha256"):
        reasons.append("claim_audit_corpus_mismatch")
    report_path = cell_config.get("report_path")
    required_state = cell_config.get("required_state")
    status = nested(report, report_path) if isinstance(report_path, str) else None
    if not isinstance(status, dict) or not isinstance(status.get("state"), str):
        reasons.append("claim_audit_status_missing")
        return reasons, None
    observed_state = status["state"]
    if observed_state != required_state:
        if observed_state == "LEAKAGE_DETECTED":
            reasons.append("verified_training_evaluation_overlap")
        elif (observed_state == "BLOCKED" and status.get("contamination_verdict") == "PASS"
              and status.get("metadata_completeness_verdict") == "INCOMPLETE_EVALUATION_RELEASE_METADATA"):
            reasons.append("incomplete_evaluation_release_metadata")
        elif observed_state == "BLOCKED":
            reasons.append("unresolved_perceptual_training_evaluation_overlap")
        else:
            reasons.append(f"claim_audit_state={observed_state}")
    return reasons, {
        "path": str(path.resolve()),
        "sha256": digest,
        "identity_sha256": observed_identity,
        "report_path": report_path,
        "state": observed_state,
        "contamination_verdict": status.get("contamination_verdict"),
        "metadata_completeness_verdict": status.get("metadata_completeness_verdict"),
    }


def audit_result(
    record: dict[str, Any], run: dict[str, Any], benchmark: dict[str, Any],
    protocol: dict[str, Any], current_hash: str,
) -> tuple[bool, list[str], float | None]:
    reasons = []
    is_formal = run.get("tier") == "formal"
    if not is_formal:
        reasons.append(f"run_tier={run.get('tier')}")
    else:
        recorded_model = record.get("model")
        registered_model = run.get("model_path")
        if not isinstance(recorded_model, str) or not isinstance(registered_model, str):
            reasons.append("model_path_missing")
        else:
            recorded_path = Path(recorded_model).expanduser().resolve()
            registered_path = Path(registered_model).expanduser().resolve()
            if recorded_path != registered_path:
                reasons.append("model_path_mismatch")
        reasons.extend(audit_model_provenance(run))
    if record.get("schema_version") != protocol["required_schema"]:
        reasons.append(f"schema={record.get('schema_version')}")
    if record.get("evaluation_protocol") != protocol["name"]:
        reasons.append(f"protocol={record.get('evaluation_protocol')}")
    if record.get("evaluation_code_sha256") != current_hash:
        reasons.append("evaluation_code_hash_mismatch")
    if record.get("thinking") is not protocol["thinking"]:
        reasons.append(f"thinking={record.get('thinking')}")
    if record.get("subset_seed") != protocol["subset_seed"]:
        reasons.append(f"subset_seed={record.get('subset_seed')}")
    if record.get("subset_limit") != protocol["subset_limit"]:
        reasons.append(f"subset_limit={record.get('subset_limit')}")
    if record.get("generation_batch_size") != protocol["generation_batch_size"]:
        reasons.append(f"generation_batch_size={record.get('generation_batch_size')}")
    if record.get("max_tokens") != benchmark["required_max_tokens"]:
        reasons.append(
            f"max_tokens={record.get('max_tokens')} "
            f"(required={benchmark['required_max_tokens']})")
    if record.get("n_evaluated", 0) < benchmark["required_n"]:
        reasons.append(f"n_evaluated={record.get('n_evaluated')}")
    if not record.get("parsed") or record.get("score") is None:
        reasons.append("trust_gate_withheld")
    value = nested(record, benchmark["value_path"])
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        reasons.append(f"missing_value={benchmark['value_path']}")
        value = None
    subgroup_n = benchmark.get("required_subgroup_n")
    if subgroup_n is not None:
        subgroup_path = benchmark["value_path"].rsplit(".", 1)[0] + ".n"
        if nested(record, subgroup_path) != subgroup_n:
            reasons.append(f"subgroup_n={nested(record, subgroup_path)}")
    return not reasons, reasons, None if value is None else float(value)


def write_latex(index: dict[str, Any], path: Path) -> None:
    benches = index["benchmarks"]
    main_runs = [row for row in index["runs"] if row["include_in_main_table"]]
    pretrained_runs = [row for row in main_runs if row.get("table_group") != "posttrained"]
    pretrained_best = {
        bench["id"]: max((
            float(run["results"][bench["id"]]["value"])
            for run in pretrained_runs
            if run["results"][bench["id"]].get("eligible")
        ), default=None)
        for bench in benches
    }
    header_style = {
        "cv_bench": ("taskS", "CV-Bench [S]", ""),
        "embspatial": ("taskS", "EmbSpatial [S]", ""),
        "refspatial_bench": ("taskG", "RefSpatial [G]", ""),
        "where2place": ("taskStress", "Where2Place", "stress; "),
        "gsa_bench_test": ("taskG", "GSA-G [G]", ""),
    }
    headers = []
    for bench in benches:
        macro, name, qualifier = header_style[bench["id"]]
        scored_n = bench.get("required_subgroup_n", bench["required_n"])
        latex_n = f"{scored_n:,}".replace(",", "{,}")
        headers.append(
            rf"\shortstack{{\{macro}{{{name}}}\\"
            rf"{{\scriptsize {qualifier}$n={latex_n}$}}}}"
        )
    lines = [
        "% Generated by eval/build_result_index.py; do not edit numeric cells by hand.",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xccccc@{}}",
        r"\toprule",
        "Model & " + " & ".join(headers) + r" \\",
        r"\midrule",
        r"\rowcolor{softgray}\multicolumn{6}{@{}l}{\(\circ\)\;\textbf{Pretrained VLMs}} \\",
    ]
    for run in main_runs:
        if run.get("table_group") == "posttrained":
            lines.extend([
                r"\addlinespace[2pt]",
                r"\rowcolor{losswash}\multicolumn{6}{@{}l}{\(\diamond\)\;\textbf{Post-trained diagnostic}} \\",
            ])
        cells = []
        for bench in benches:
            cell = run["results"].get(bench["id"], {})
            if cell.get("eligible"):
                rendered = f"{cell['value']:.2f}"
                if (
                    run.get("table_group") != "posttrained"
                    and pretrained_best[bench["id"]] is not None
                    and math.isclose(float(cell["value"]), pretrained_best[bench["id"]])
                ):
                    rendered = rf"\bestscore{{{rendered}}}"
                cells.append(rendered)
            elif cell.get("exists") and "trust_gate_withheld" in cell.get("reasons", []):
                diagnostic = cell.get("score_if_untrusted")
                if isinstance(diagnostic, (int, float)) and math.isfinite(float(diagnostic)):
                    cells.append(rf"\textit{{{float(diagnostic):.2f}}}$^{{\dagger}}$")
                else:
                    cells.append(r"\textit{not ranked}")
            elif cell.get("exists") and any(
                reason in cell.get("reasons", [])
                for reason in (
                    "verified_training_evaluation_overlap",
                    "unresolved_perceptual_training_evaluation_overlap",
                )
            ):
                diagnostic = cell.get("value")
                if isinstance(diagnostic, (int, float)) and math.isfinite(float(diagnostic)):
                    cells.append(rf"\textit{{{float(diagnostic):.2f}}}$^{{\mathsection}}$")
                else:
                    cells.append(r"\textit{not compared}$^{\mathsection}$")
            else:
                cells.append("--")
        lines.append(latex_escape(run["display_name"]) + " & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/result_registry.yaml")
    parser.add_argument("--out-dir", default="artifacts/results")
    parser.add_argument("--latex", default="paper/tables/tab_benchmark_v7.tex")
    parser.add_argument("--evaluation-code-snapshot", type=Path,
                        help="Explicit historical five-module scorer snapshot; result hashes must match its actual bytes")
    args = parser.parse_args()

    registry_path = ROOT / args.registry
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    protocol = registry["protocol"]
    current_hash = evaluation_code_sha256()
    accepted_hash = current_hash
    snapshot_binding = None
    if args.evaluation_code_snapshot is not None:
        root = args.evaluation_code_snapshot.resolve()
        digest, bindings = hashlib.sha256(), {}
        for name in ("datasets.py", "parsing.py", "registry.py", "runner.py", "scorers.py"):
            path = root / name
            raw = path.read_bytes()
            digest.update(name.encode("utf-8") + b"\0" + raw + b"\0")
            bindings[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        accepted_hash = digest.hexdigest()
        snapshot_binding = {"root": str(root), "evaluation_code_sha256": accepted_hash, "files": bindings}
    benches = [{"id": key, **value} for key, value in registry["benchmarks"].items()]
    runs = []
    flat = []
    for run_id, config in registry["runs"].items():
        row = {"id": run_id, **config, "results": {}}
        result_dir = ROOT / config["result_dir"]
        for bench in benches:
            path = result_dir / f"{bench['id']}.json"
            cell = {"path": str(path), "exists": path.is_file(), "eligible": False,
                    "value": None, "reasons": []}
            record = {}
            if path.is_file():
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    eligible, reasons, value = audit_result(
                        record, config, bench, protocol, accepted_hash)
                    claim_reasons, claim_audit = audit_claim_evidence(config, bench["id"])
                    reasons.extend(claim_reasons)
                    cell.update({
                        "eligible": eligible and not claim_reasons,
                        "reasons": reasons,
                        "value": value,
                        "claim_audit": claim_audit,
                    })
                except Exception as exc:
                    cell["reasons"] = [f"unreadable:{type(exc).__name__}:{exc}"]
            else:
                cell["reasons"] = ["missing_result"]
            for key in ("score", "score_if_untrusted", "n_evaluated", "parse_failure_rate",
                        "truncation_rate", "schema_version", "evaluation_code_sha256"):
                cell[key] = record.get(key)
            row["results"][bench["id"]] = cell
            flat.append({
                "run_id": run_id,
                "model": config["display_name"],
                "benchmark": bench["id"],
                "role": bench["role"],
                **{key: cell.get(key) for key in (
                    "exists", "eligible", "value", "score", "score_if_untrusted",
                    "n_evaluated", "parse_failure_rate", "truncation_rate", "schema_version")},
                "reasons": ";".join(cell["reasons"]),
                "path": cell["path"],
            })
        runs.append(row)

    index = {
        "schema_version": 1,
        "registry": str(registry_path),
        "current_evaluation_code_sha256": current_hash,
        "accepted_evaluation_code_sha256": accepted_hash,
        "explicit_historical_evaluation_snapshot": snapshot_binding,
        "protocol": protocol,
        "benchmarks": benches,
        "runs": runs,
    }
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "result_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    write_latex(index, ROOT / args.latex)

    eligible = sum(item["eligible"] for item in flat)
    withheld = sum(item["exists"] and not item["eligible"] for item in flat)
    print(json.dumps({"registered_cells": len(flat), "eligible": eligible,
                      "withheld_or_diagnostic": withheld,
                      "missing": sum(not item["exists"] for item in flat),
                      "current_code_sha256": current_hash,
                      "accepted_code_sha256": accepted_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
