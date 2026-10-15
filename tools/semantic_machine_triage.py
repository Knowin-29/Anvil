#!/usr/bin/env python3
"""Prepare or run non-gating machine triage for the fixed semantic-audit pack.

One multimodal request is constructed per image and asks for judgments on all eight
forms. Results are written only to ``artifacts/audits`` by default and are deliberately
incompatible with the human annotation records consumed by
``eval/compile_semantic_audit.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PACK = ROOT / "data/semantic_audit_v2"
DEFAULT_OUTPUT = ROOT / "artifacts/audits/semantic_machine_triage.jsonl"
DEFAULT_PROMPT_OUTPUT = ROOT / "artifacts/audits/semantic_machine_triage_prompts.jsonl"
DEFAULT_PROMPT_TEMPLATE = ROOT / "configs/prompts/semantic_machine_triage_v1.txt"
PROMPT_VERSION = "semantic-machine-triage-v1"
RESULT_FIELDS = (
    "sample_id",
    "form",
    "suggested_correct",
    "confidence",
    "reason",
    "model",
    "prompt_version",
)


def env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def assert_machine_only_path(path: Path, pack: Path) -> None:
    resolved = path.resolve()
    for annotator in ("annotator_a", "annotator_b"):
        human_root = (pack / annotator).resolve()
        if resolved == human_root or resolved.is_relative_to(human_root):
            raise ValueError(f"machine output must not be inside human root: {resolved}")


@dataclass(frozen=True)
class TriageCase:
    index: int
    image_path: Path
    sample_ids: dict[str, str]
    labels: list[dict[str, Any]]
    prompt: str


class SemanticTriagePack:
    """Read the fixed manifest and the read-only annotator-A label mirror."""

    def __init__(self, pack: Path, prompt_template: Path):
        self.pack = pack.resolve()
        manifest_path = self.pack / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.forms = list(self.manifest.get("forms") or [])
        self.selected = list(self.manifest.get("selected") or [])
        if len(self.forms) != 8:
            raise ValueError(f"expected exactly eight forms, found {self.forms}")
        if int(self.manifest.get("images", -1)) != len(self.selected):
            raise ValueError("manifest image count does not match selected records")

        self.label_root = self.pack / "annotator_a/labels"
        self.labels: dict[str, dict[str, Any]] = {}
        for label_path in sorted(self.label_root.rglob("*.jsonl")):
            for label in read_jsonl(label_path):
                sample_id = label.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"label without sample_id in {label_path}")
                if sample_id in self.labels:
                    raise ValueError(f"duplicate sample_id: {sample_id}")
                self.labels[sample_id] = label

        self.prompt_template_path = prompt_template.resolve()
        self.prompt_template = self.prompt_template_path.read_text(encoding="utf-8")
        for placeholder in ("{{IMAGE_METADATA_JSON}}", "{{CANDIDATE_LABELS_JSON}}"):
            if self.prompt_template.count(placeholder) != 1:
                raise ValueError(f"prompt template must contain {placeholder} exactly once")
        self.cases = [self._build_case(index, item) for index, item in enumerate(self.selected)]

    def _build_case(self, index: int, item: dict[str, Any]) -> TriageCase:
        sample_ids = item.get("sample_ids") or {}
        if set(sample_ids) != set(self.forms):
            raise ValueError(f"manifest item {index} does not contain all eight forms")
        image_path = Path(str(item.get("image_path", ""))).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for manifest item {index}: {image_path}")

        candidates: list[dict[str, Any]] = []
        for form in self.forms:
            sample_id = sample_ids[form]
            label = self.labels.get(sample_id)
            if label is None:
                raise ValueError(f"manifest sample absent from annotator-A labels: {sample_id}")
            if label.get("task_form") != form:
                raise ValueError(f"sample {sample_id} has form {label.get('task_form')!r}, expected {form!r}")
            if Path(str(label.get("image_path", ""))).resolve() != image_path:
                raise ValueError(f"sample {sample_id} points to a different image")
            candidates.append(
                {
                    "sample_id": sample_id,
                    "form": form,
                    "pipeline": label.get("pipeline"),
                    "object_class": label.get("object_class"),
                    "payload": label.get("payload"),
                    "gsa_context": label.get("gsa_context"),
                }
            )

        first = self.labels[sample_ids[self.forms[0]]]
        metadata = {
            "image_index": index,
            "image_name": image_path.name,
            "scene_type": first.get("scene_type"),
            "source_bucket": first.get("source_bucket"),
            "source_pack": "semantic_audit_v2",
            "coordinate_scale": "0-1000",
        }
        prompt = self.prompt_template.replace(
            "{{IMAGE_METADATA_JSON}}", json.dumps(metadata, ensure_ascii=False, indent=2)
        ).replace(
            "{{CANDIDATE_LABELS_JSON}}", json.dumps(candidates, ensure_ascii=False, indent=2)
        )
        return TriageCase(index, image_path, dict(sample_ids), candidates, prompt)

    def prompt_rows(self, model: str) -> list[dict[str, Any]]:
        return [
            {
                "image_index": case.index,
                "image_path": str(case.image_path),
                "image_sha256": file_sha256(case.image_path),
                "sample_ids": case.sample_ids,
                "source_manifest": str(self.pack / "manifest.json"),
                "source_label_root": str(self.label_root),
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "prompt": case.prompt,
            }
            for case in self.cases
        ]


def normalize_response(raw: Any, case: TriageCase, model: str) -> list[dict[str, Any]]:
    judgments = raw.get("judgments") if isinstance(raw, dict) else None
    if not isinstance(judgments, list) or len(judgments) != len(case.sample_ids):
        raise ValueError(f"image {case.index}: response must contain exactly eight judgments")
    expected = {(form, sample_id) for form, sample_id in case.sample_ids.items()}
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise ValueError(f"image {case.index}: each judgment must be an object")
        form = judgment.get("form")
        sample_id = judgment.get("sample_id")
        key = (form, sample_id)
        if key not in expected or key in seen:
            raise ValueError(f"image {case.index}: unexpected or duplicate judgment {key}")
        suggested = judgment.get("suggested_correct")
        confidence = judgment.get("confidence")
        reason = judgment.get("reason")
        if not isinstance(suggested, bool):
            raise ValueError(f"image {case.index} {form}: suggested_correct must be Boolean")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"image {case.index} {form}: confidence must be in [0,1]")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"image {case.index} {form}: reason must be non-empty")
        seen.add(key)
        rows.append(
            dict(
                zip(
                    RESULT_FIELDS,
                    (
                        sample_id,
                        form,
                        suggested,
                        float(confidence),
                        reason.strip(),
                        model,
                        PROMPT_VERSION,
                    ),
                )
            )
        )
    if seen != expected:
        raise ValueError(f"image {case.index}: response did not cover all expected labels")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("SEMANTIC_TRIAGE_OUT", DEFAULT_OUTPUT)))
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=Path(os.environ.get("SEMANTIC_TRIAGE_PROMPT_OUT", DEFAULT_PROMPT_OUTPUT)),
    )
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--model", default=os.environ.get("GSA_VLM_MODEL", "Qwen3.5-397B-A17B"))
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN"))
    parser.add_argument("--prompt-only", action="store_true", default=env_flag("PROMPT_ONLY"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack = SemanticTriagePack(args.pack, args.prompt_template)
    assert_machine_only_path(args.output, pack.pack)
    assert_machine_only_path(args.prompt_output, pack.pack)
    summary = {
        "images": len(pack.cases),
        "forms": len(pack.forms),
        "planned_judgments": len(pack.cases) * len(pack.forms),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "source_label_root": str(pack.label_root),
    }

    if args.dry_run and not args.prompt_only:
        print(json.dumps({**summary, "mode": "dry_run", "api_calls": 0}, indent=2))
        return 0

    prompt_rows = pack.prompt_rows(args.model)
    write_jsonl_atomic(args.prompt_output, prompt_rows)
    if args.prompt_only:
        print(
            json.dumps(
                {**summary, "mode": "prompt_only", "api_calls": 0, "prompt_output": str(args.prompt_output)},
                indent=2,
            )
        )
        return 0

    # Import only in live mode. PROMPT_ONLY and DRY_RUN cannot reach the HTTP client.
    from trains.flywheel.vlm_client import chat_json

    results: list[dict[str, Any]] = []
    for case in pack.cases:
        raw = chat_json(
            str(case.image_path),
            case.prompt,
            model=args.model,
            temperature=0.0,
            max_tokens=4096,
            retries=3,
            use_cache=True,
        )
        results.extend(normalize_response(raw, case, args.model))
        print(f"machine triage: image {case.index + 1}/{len(pack.cases)} validated", flush=True)
    write_jsonl_atomic(args.output, results)
    print(json.dumps({**summary, "mode": "live", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
