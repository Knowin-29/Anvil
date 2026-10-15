from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path

from PIL import Image

from scripts import preflight_general_specialist as preflight


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "image", "question", "answer", "A", "B"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return preflight.file_md5(path)


def _dataset(path: Path, md5: str, n: int) -> dict:
    return {
        "id": "fixture",
        "current_state": "TBD",
        "local_path": str(path),
        "md5": md5,
        "expected_n": n,
        "id_column": "index",
        "required_columns": ["index", "image", "question", "answer"],
        "image_columns": ["image", "image_path"],
    }


def _image_b64() -> str:
    payload = io.BytesIO()
    Image.new("RGB", (2, 2), color=(20, 40, 60)).save(payload, format="PNG")
    return base64.b64encode(payload.getvalue()).decode()


def _make_checkpoint(path: Path) -> None:
    path.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (path / name).write_text("{}\n", encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
    (path / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    shard = "model-00001-of-00001.safetensors"
    (path / shard).write_bytes(b"weights")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard}}), encoding="utf-8"
    )


def test_tsv_requires_release_identity_unique_ids_and_decodable_images(tmp_path: Path) -> None:
    image = _image_b64()
    path = tmp_path / "eval.tsv"
    md5 = _write_tsv(
        path,
        [
            {"index": "0", "image": image, "question": "q0", "answer": "A", "A": "x", "B": "y"},
            {"index": "1", "image": image, "question": "q1", "answer": "B", "A": "x", "B": "y"},
        ],
    )
    report = preflight.inspect_tsv(_dataset(path, md5, 2), decode_images=True)
    assert report["state"] == "formal"
    assert report["n"] == 2
    assert report["unique_images"] == 1

    wrong = preflight.inspect_tsv(_dataset(path, "0" * 32, 2), decode_images=True)
    assert wrong["state"] == "diagnostic"
    assert "md5 mismatch" in wrong["reasons"][0]


def test_duplicate_ids_are_not_formal(tmp_path: Path) -> None:
    image = _image_b64()
    path = tmp_path / "duplicate.tsv"
    md5 = _write_tsv(
        path,
        [
            {"index": "0", "image": image, "question": "q0", "answer": "A", "A": "x", "B": "y"},
            {"index": "0", "image": image, "question": "q1", "answer": "B", "A": "x", "B": "y"},
        ],
    )
    report = preflight.inspect_tsv(_dataset(path, md5, 2), decode_images=True)
    assert report["state"] == "diagnostic"
    assert any("duplicate" in reason for reason in report["reasons"])


def test_checkpoint_requires_indexed_nonempty_shards(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _make_checkpoint(model)
    ok, reasons = preflight.inspect_checkpoint(model)
    assert ok and not reasons
    (model / "model-00001-of-00001.safetensors").write_bytes(b"")
    ok, reasons = preflight.inspect_checkpoint(model)
    assert not ok
    assert any("missing or empty shard" in reason for reason in reasons)


def test_general_suite_stays_tbd_without_runner_and_leakage_audit(tmp_path: Path) -> None:
    image = _image_b64()
    data = tmp_path / "eval.tsv"
    md5 = _write_tsv(
        data,
        [{"index": "0", "image": image, "question": "q", "answer": "A", "A": "x", "B": "y"}],
    )
    model = tmp_path / "model"
    _make_checkpoint(model)
    registry = {
        "suite": {"id": "fixture-suite", "current_state": "TBD"},
        "benchmarks": [_dataset(data, md5, 1)],
        "models": [{
            "id": "model",
            "current_state": "formal",
            "required_for_initial_release": True,
            "path": str(model),
        }],
        "posttraining_corpora": [{
            "id": "corpus",
            "path": str(data),
            "expected_rows": 2,
            "sha256": preflight.file_sha256(data),
        }],
        "formal_runner": {"required_path": str(tmp_path / "missing.sh"), "blocker": "runner missing"},
        "leakage_audit": {
            "required_path": str(tmp_path / "missing.json"),
            "required_fields": {"exact_image_collisions": 0},
        },
    }
    report = preflight.inspect_general(registry, decode_images=True)
    assert report["state"] == "TBD"
    assert report["ready_for_formal_gpu"] is False
    assert report["datasets"][0]["state"] == "formal"


def test_metadata_only_never_admits_formal_dataset(tmp_path: Path) -> None:
    path = tmp_path / "eval.tsv"
    md5 = _write_tsv(
        path,
        [{
            "index": "0",
            "image": "not-even-base64",
            "question": "q",
            "answer": "A",
            "A": "x",
            "B": "y",
        }],
    )
    report = preflight.inspect_tsv(_dataset(path, md5, 1), decode_images=False)
    assert report["state"] == "diagnostic"
    assert any("not decoded" in reason for reason in report["reasons"])


def test_invalid_image_payload_is_not_formal(tmp_path: Path) -> None:
    path = tmp_path / "eval.tsv"
    md5 = _write_tsv(
        path,
        [{
            "index": "0",
            "image": base64.b64encode(b"not-an-image").decode(),
            "question": "q",
            "answer": "A",
            "A": "x",
            "B": "y",
        }],
    )
    report = preflight.inspect_tsv(_dataset(path, md5, 1), decode_images=True)
    assert report["state"] == "diagnostic"
    assert any("invalid image" in reason for reason in report["reasons"])
