from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_depth_v43_generator import (  # noqa: E402
    FakePairClient,
    attach_fixture_run_audit,
    make_candidate,
    write_human_review,
)
from tools import generate_depth_v43_pairwise as generator  # noqa: E402
from eval.compile_semantic_audit import manifest_payload_hash  # noqa: E402
from trains.flywheel.build_depth_semantic_audit_pack import (  # noqa: E402
    generation_run_evidence_hash,
)

BUILDER = ROOT / "trains/flywheel/build_depth_semantic_audit_pack.py"
COMPILER = ROOT / "eval/compile_semantic_audit.py"
CANARY_NAMES = (
    "kit-033-2ae66a3c4b.jpg",
    "kit-066-cf951e8ea8.jpg",
    "kit-100-db2a723fc5.jpg",
    "kit-031-13f210425e.jpg",
    "kit-072-982558e28d.jpg",
    "kit-052-de367c6bda.jpg",
    "kit-102-f7ad5341ab.jpg",
    "emb-102-27c493360d.jpg",
)


def existing_canary_candidates(root: Path) -> list[dict]:
    root.mkdir()
    image_root = ROOT / "artifacts/upload_review_500_v2/images"
    candidates = []
    for position, name in enumerate(CANARY_NAMES):
        image = image_root / name
        candidate = make_candidate(root, position)
        candidate["image_path"] = str(image)
        candidate["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        candidate["detection"]["image_path"] = str(image)
        candidate["bbox"]["image_path"] = str(image)
        candidates.append(candidate)
    assert tuple(item["image_sha256"] for item in candidates) == (
        generator.FROZEN_CANARY_SHA256S
    )
    return candidates


def build_generation_fixture(root: Path) -> tuple[Path, Path]:
    canary_candidates = existing_canary_candidates(root / "canary-candidates")
    canary_provenance = {"fixture": "frozen-canary"}
    canary_out = root / "canary"
    canary_report, canary_rc = generator.run_generation(
        canary_candidates,
        canary_provenance,
        canary_out,
        workers=4,
        client=FakePairClient(),
        protocol_artifact=False,
        stage="canary",
    )
    assert canary_rc == 0
    canary_report_path = attach_fixture_run_audit(canary_report, canary_out)
    canary_labels = [
        json.loads(line)
        for line in (
            canary_out / "labels/S_spatial/depth.jsonl"
        ).read_text().splitlines()
    ]
    review_path = write_human_review(
        canary_report_path,
        canary_labels,
        correct_count=len(canary_labels),
    )
    canary_binding = generator.validate_canary_binding(
        canary_report_path,
        review_path,
        canary_candidates,
        canary_provenance,
    )

    full_candidate_root = root / "full-candidates"
    full_candidate_root.mkdir()
    full_candidates = [
        make_candidate(full_candidate_root, index) for index in range(200, 240)
    ]
    full_out = root / "full"
    full_report, full_rc = generator.run_generation(
        full_candidates,
        {"fixture": "full", "canary_binding": canary_binding},
        full_out,
        workers=8,
        client=FakePairClient(),
        protocol_artifact=True,
        stage="full",
    )
    assert full_rc == 0
    full_report_path = attach_fixture_run_audit(full_report, full_out)
    return full_out / "labels/S_spatial/depth.jsonl", full_report_path


def exclusion_args(root: Path) -> list[str]:
    args: list[str] = []
    for schema in (1, 3):
        path = root / f"exclude-{schema}.json"
        path.write_text(json.dumps({"schema_version": schema, "selected": []}))
        args.extend(["--exclude-manifest", str(path)])
    return args


def builder_command(
    source: Path,
    report: Path,
    out: Path,
    exclusions: list[str],
) -> list[str]:
    return [
        sys.executable,
        str(BUILDER),
        "--source",
        str(source),
        "--generation-report",
        str(report),
        "--out",
        str(out),
        *exclusions,
    ]


def test_v43_pack_compiler_bind_full_run_and_human_canary(tmp_path: Path):
    source, generation_report = build_generation_fixture(tmp_path)
    exclusions = exclusion_args(tmp_path)
    pack = tmp_path / "pack"
    built = subprocess.run(
        builder_command(source, generation_report, pack, exclusions),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    manifest = json.loads((pack / "manifest.json").read_text())
    assert manifest["protocol"] == "depth-v4.3-pairwise-semantic-audit-v1"
    assert manifest["round"] == "v4.3-depth-pairwise"
    assert len(manifest["generation_run"]["report_sha256"]) == 64
    assert len(manifest["generation_run"]["run_audit_sha256"]) == 64
    assert len(manifest["generation_run"]["canary_binding_sha256"]) == 64

    for root_name, annotator in (("annotator_a", "a"), ("annotator_b", "b")):
        labels = [
            json.loads(line)
            for line in (
                pack / root_name / "labels/S_spatial/depth.jsonl"
            ).read_text().splitlines()
        ]
        annotation_path = pack / root_name / "annotations/annotations.jsonl"
        annotation_path.write_text("".join(
            json.dumps({
                "correct": True,
                "annotator_key": annotator,
                "label_snapshot": label,
            }) + "\n"
            for label in labels
        ))
    compiled_report = tmp_path / "compiled.json"
    compiled = subprocess.run(
        [
            sys.executable,
            str(COMPILER),
            "--pack",
            str(pack),
            "--out",
            str(compiled_report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    assert json.loads(compiled_report.read_text())["verdict"] == "PASS"

    failed_report = json.loads(generation_report.read_text())
    failed_report["human_canary_gate"]["passed"] = False
    generation_report.write_text(json.dumps(failed_report, indent=2))
    rejected = subprocess.run(
        builder_command(
            source,
            generation_report,
            tmp_path / "rejected-pack",
            exclusions,
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "human_canary_gate did not pass" in rejected.stderr

    manifest["generation_run"]["report_sha256"] = hashlib.sha256(
        generation_report.read_bytes()
    ).hexdigest()
    manifest["generation_run"]["run_evidence_sha256"] = (
        generation_run_evidence_hash(failed_report)
    )
    manifest["integrity"]["manifest_payload_sha256"] = manifest_payload_hash(
        manifest
    )
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=2))
    recompiled_report = tmp_path / "recompiled.json"
    recompiled = subprocess.run(
        [
            sys.executable,
            str(COMPILER),
            "--pack",
            str(pack),
            "--out",
            str(recompiled_report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert recompiled.returncode == 2
    assert any(
        "human_canary_gate did not pass" in error
        for error in json.loads(recompiled_report.read_text())["errors"]
    )
