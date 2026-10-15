from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.compile_g_family_scoped_release import (  # noqa: E402
    G_FORMS,
    PROTOCOL,
    SCOPED_PASS,
    _reviewer_c_errors,
    compile_scoped,
)
from scripts.verify_g_family_scoped_qualification import verify  # noqa: E402
from trains.materialize_g_family_scoped_v3 import materialize  # noqa: E402

PACK = ROOT / "data/semantic_audit_v3"
ADJUDICATION = PACK / "adjudication.jsonl"
FULL_REPORT = ROOT / "artifacts/audits/pilot_semantic_audit_v3_adjudicated.json"


def compiler_args() -> argparse.Namespace:
    return argparse.Namespace(
        pack=PACK,
        adjudication=ADJUDICATION,
        full_report=FULL_REPORT,
        min_accuracy=0.80,
        min_agreement=0.80,
    )


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("g_family_scoped_release")
    report_path = root / "report.json"
    report, rc = compile_scoped(compiler_args())
    assert rc == 0
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    corpus = root / "corpus"
    materialize(
        argparse.Namespace(
            pack=PACK,
            adjudication=ADJUDICATION,
            full_report=FULL_REPORT,
            scoped_report=report_path,
            out=corpus,
            seed=20260820,
            val_fraction=0.20,
            min_accuracy=0.80,
            min_agreement=0.80,
        )
    )
    return report_path, corpus


def verifier_args(report: Path, corpus: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=corpus / "manifest.json",
        scoped_report=report,
        full_report=FULL_REPORT,
        pack=PACK,
        adjudication=ADJUDICATION,
        min_accuracy=0.80,
        min_agreement=0.80,
    )


def test_current_gate_is_explicitly_g_only_while_global_is_blocked() -> None:
    report, rc = compile_scoped(compiler_args())
    assert rc == 0
    assert report["protocol"] == PROTOCOL
    assert report["verdict"] == SCOPED_PASS
    assert report["global_audit_verdict"] == "BLOCKED"
    assert report["global_blocking_forms"] == ["depth"]
    assert set(report["by_form"]) == set(G_FORMS)
    assert all(report["by_form"][form]["eligible"] for form in G_FORMS)
    assert "joint_training" in report["forbidden_uses"]
    assert "mopd_training" in report["forbidden_uses"]


def test_reviewer_c_identity_is_fail_closed(tmp_path: Path) -> None:
    bad = tmp_path / "adjudication.jsonl"
    bad.write_text(
        json.dumps(
            {
                "sample_id": "v3_bbox_example",
                "correct": True,
                "reviewer_key": "not-reviewer-c",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    errors = _reviewer_c_errors(bad)
    assert errors
    assert "reviewer_key" in errors[0]


def test_materialized_release_is_exact_and_image_disjoint(
    release: tuple[Path, Path],
) -> None:
    report_path, corpus = release
    result = verify(verifier_args(report_path, corpus))
    assert result["verdict"] == SCOPED_PASS
    assert result["global_audit_verdict"] == "BLOCKED"
    assert result["records"] == 148
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"train": 116, "val": 32}
    assert manifest["unique_images"] == {"train": 32, "val": 8}
    assert not (corpus / "train.jsonl").exists()
    assert not (corpus / "val.jsonl").exists()
    train_images = set(manifest["image_split"]["train_image_sha256"])
    val_images = set(manifest["image_split"]["val_image_sha256"])
    assert not train_images & val_images
    assert set(manifest["by_form"]["train"]) == set(G_FORMS)
    assert set(manifest["by_form"]["val"]) == set(G_FORMS)


def test_preflight_rejects_corpus_byte_drift(
    release: tuple[Path, Path], tmp_path: Path
) -> None:
    report_path, corpus = release
    tampered = tmp_path / "corpus"
    shutil.copytree(corpus, tampered)
    train = tampered / "train_G_only.jsonl"
    train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus file hash drift"):
        verify(verifier_args(report_path, tampered))


def test_launcher_cpu_preflight_does_not_start_gpu(
    release: tuple[Path, Path], tmp_path: Path
) -> None:
    report_path, corpus = release
    env = os.environ.copy()
    env.update(
        {
            "PREFLIGHT_ONLY": "1",
            "G_SCOPED_REPORT": str(report_path),
            "G_SCOPED_CORPUS": str(corpus),
            "OUT": str(tmp_path / "must_not_be_created"),
            "PYTHON_BIN": sys.executable,
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_g_family_qualification_l20.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "G_ONLY_PREFLIGHT_PASS: no GPU process was started" in result.stdout
    assert "steps=73" in result.stdout
    assert not (tmp_path / "must_not_be_created").exists()
