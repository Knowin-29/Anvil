from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import download_baselines as downloads


REVISION = "a" * 40


def _entry(path: Path, *, family: str = "qwen") -> dict:
    if family == "internvl":
        model_id = "internvl3_5_8b_hf"
        repo_id = "OpenGVLab/InternVL3_5-8B-HF"
    else:
        model_id = "qwen3_vl_8b_instruct"
        repo_id = "Qwen/Qwen3-VL-8B-Instruct"
    return {
        "id": model_id,
        "source": "huggingface",
        "repo_id": repo_id,
        "revision": REVISION,
        "path": str(path),
        "harness": "unified_gsa_eval",
    }


def _make_checkpoint(path: Path, *, family: str = "qwen") -> dict:
    path.mkdir(parents=True)
    files = {
        "config.json": b"{}\n",
        "preprocessor_config.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "model-00001-of-00002.safetensors": b"first-shard",
        "model-00002-of-00002.safetensors": b"second-shard",
    }
    if family == "internvl":
        files["processor_config.json"] = b"{}\n"
        files["chat_template.jinja"] = b"template\n"
    else:
        files["chat_template.json"] = b"{}\n"
    index = {
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        }
    }
    files["model.safetensors.index.json"] = (
        json.dumps(index).encode("utf-8") + b"\n"
    )
    for name, content in files.items():
        (path / name).write_bytes(content)
    return files


def _info(files: dict[str, bytes], *, sha: str = REVISION) -> SimpleNamespace:
    return SimpleNamespace(
        sha=sha,
        siblings=[
            SimpleNamespace(rfilename=name, size=len(content), lfs=None)
            for name, content in files.items()
        ],
    )


class _Api:
    def __init__(self, info: SimpleNamespace):
        self.info = info
        self.calls = []

    def model_info(self, repo_id, revision, files_metadata):
        self.calls.append((repo_id, revision, files_metadata))
        return self.info


@pytest.mark.parametrize("failure", ["missing", "empty", "incomplete"])
def test_local_integrity_rejects_bad_weight_payload(tmp_path: Path, failure: str) -> None:
    model = tmp_path / "model"
    files = _make_checkpoint(model)
    shard = model / "model-00002-of-00002.safetensors"
    if failure == "missing":
        shard.unlink()
    elif failure == "empty":
        shard.write_bytes(b"")
    else:
        (model / ".cache" / "download").mkdir(parents=True)
        (model / ".cache" / "download" / "model.safetensors.incomplete").write_bytes(b"x")

    with pytest.raises(downloads.IntegrityError):
        downloads.inspect_local(model, _entry(model), downloads.remote_inventory(_info(files)))


@pytest.mark.parametrize(
    ("family", "missing"),
    [
        ("qwen", "chat_template.json"),
        ("internvl", "processor_config.json"),
        ("internvl", "chat_template.jinja"),
    ],
)
def test_model_family_required_files_are_enforced(
    tmp_path: Path, family: str, missing: str
) -> None:
    model = tmp_path / family
    _make_checkpoint(model, family=family)
    (model / missing).unlink()
    with pytest.raises(downloads.IntegrityError, match="required files"):
        downloads.inspect_local(model, _entry(model, family=family))


def test_required_metadata_must_be_nonempty(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _make_checkpoint(model)
    (model / "tokenizer_config.json").write_bytes(b"")
    with pytest.raises(downloads.IntegrityError, match="empty required files"):
        downloads.inspect_local(model, _entry(model))


def test_remote_revision_must_match_pinned_commit(tmp_path: Path) -> None:
    model = tmp_path / "model"
    files = _make_checkpoint(model)
    api = _Api(_info(files, sha="b" * 40))
    with pytest.raises(downloads.IntegrityError, match="revision drift"):
        downloads.inspect_remote(_entry(model), api)


def test_audit_records_remote_sizes_and_local_hashes(tmp_path: Path) -> None:
    model = tmp_path / "model"
    files = _make_checkpoint(model)
    entry = _entry(model)
    remote = downloads.inspect_remote(entry, _Api(_info(files)))
    local = downloads.inspect_local(model, entry, remote["files"])
    record = downloads.build_audit_record(
        model_id=entry["id"],
        entry=entry,
        destination=model,
        resolved_path=str(model),
        remote=remote,
        local=local,
        download_performed=True,
    )
    local_files = {item["path"]: item for item in record["local_inventory"]}
    assert record["resolved_revision"] == REVISION
    assert record["estimated_repo_bytes"] == sum(map(len, files.values()))
    assert record["local_bytes"] == sum(map(len, files.values()))
    assert len(local_files["config.json"]["sha256"]) == 64
    assert record["weight_shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def _write_registry(path: Path, entry: dict) -> None:
    path.write_text(yaml.safe_dump({"models": [entry]}), encoding="utf-8")


def test_check_only_validates_metadata_without_download_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "missing-model"
    entry = _entry(model)
    files = _make_checkpoint(tmp_path / "remote-fixture")
    registry = tmp_path / "registry.yaml"
    audit = tmp_path / "audit"
    _write_registry(registry, entry)
    monkeypatch.setattr(downloads, "HfApi", lambda: _Api(_info(files)))
    monkeypatch.setattr(
        downloads,
        "snapshot_download",
        lambda **kwargs: pytest.fail("check-only attempted a download"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_baselines.py",
            entry["id"],
            "--registry",
            str(registry),
            "--audit-dir",
            str(audit),
            "--check-only",
        ],
    )
    assert downloads.main() == 0
    assert "local=missing" in capsys.readouterr().out
    assert not model.exists()
    assert not audit.exists()


def test_existing_complete_checkpoint_is_preserved_and_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    files = _make_checkpoint(model)
    entry = _entry(model)
    registry = tmp_path / "registry.yaml"
    audit = tmp_path / "audit"
    _write_registry(registry, entry)
    monkeypatch.setattr(downloads, "HfApi", lambda: _Api(_info(files)))
    monkeypatch.setattr(
        downloads,
        "snapshot_download",
        lambda **kwargs: pytest.fail("complete checkpoint was downloaded again"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_baselines.py",
            entry["id"],
            "--registry",
            str(registry),
            "--audit-dir",
            str(audit),
        ],
    )
    assert downloads.main() == 0
    record = json.loads((audit / f"{entry['id']}.json").read_text(encoding="utf-8"))
    assert record["download_performed"] is False
    assert record["resolved_revision"] == REVISION
    assert record["local_inventory"]
