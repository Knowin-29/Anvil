"""Offline release guards; synthetic bytes are not a model-load acceptance test."""
import json
import sys
import types

import pytest

from tools.publish_anvil_model import MANIFEST, digest, upload, validate
from tools.evaluate_release_candidate import BENCHMARKS, result_is_complete, validate_shard_uids


def fixture_release(root):
    (root / "adapter_model.safetensors").write_bytes(b"synthetic-test-weight-bytes")
    (root / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3.5-9B", "revision": "a" * 40,
    }))
    (root / "README.md").write_text("# Synthetic release test\nNot real model weights.\n")
    (root / "NOTICE.md").write_text("Synthetic test fixture; no model license claimed.\n")
    files = {p.name: digest(p) for p in root.iterdir()}
    manifest = {"schema": "anvil-model-release-v1", "artifact_role": "final_mopd",
                "status": "verified", "base_model": "Qwen/Qwen3.5-9B",
                "base_revision": "a" * 40, "training_manifest_sha256": "b" * 64,
                "files": files, "evaluation": {"weights_sha256": {
                    "adapter_model.safetensors": files["adapter_model.safetensors"]},
                    "evidence_manifest_sha256": "c" * 64, "protocol_id": "synthetic-fixture"}}
    save(root, manifest)
    return manifest


def save(root, manifest):
    (root / MANIFEST).write_text(json.dumps(manifest))


def test_offline_validation_has_no_hub_dependency(tmp_path):
    expected = fixture_release(tmp_path)
    assert validate(tmp_path) == expected


def test_private_staging_config_only_changes_portability_fields():
    from tools.stage_anvil_adapters import portable_config
    config={'peft_type':'LORA','task_type':'CAUSAL_LM','r':64,'target_modules':'fixture',
            'base_model_name_or_path':'/private/local/base','revision':None}
    result=portable_config(config,'Qwen/Qwen3.5-9B','a'*40)
    assert result['base_model_name_or_path']=='Qwen/Qwen3.5-9B' and result['revision']=='a'*40
    assert {k:v for k,v in result.items() if k not in {'base_model_name_or_path','revision'}} == {
        k:v for k,v in config.items() if k not in {'base_model_name_or_path','revision'}}
    assert config['revision'] is None
    with pytest.raises(ValueError): portable_config(config,'Qwen/Qwen3.5-9B','main')


def test_private_staging_cannot_use_final_model_publisher(tmp_path):
    manifest=fixture_release(tmp_path)
    manifest.update(artifact_role='source_native_sft_preparation',status='preparation')
    save(tmp_path,manifest)
    with pytest.raises(ValueError,match='final_mopd'): validate(tmp_path)


def test_evaluation_requires_every_benchmark_and_successful_scoring():
    results = [{'benchmark': name, 'state': 'scored', 'metrics': {'n_evaluated': 1}}
               for name in BENCHMARKS]
    scores = [0] * len(BENCHMARKS)
    assert result_is_complete(results, [0, 0], scores, True)
    assert not result_is_complete(results, [0, 1], scores, True)
    assert not result_is_complete(results, [], scores, True)
    assert not result_is_complete(results, [0], scores[:-1], True)
    assert not result_is_complete(results, [0], scores, False)
    assert not result_is_complete(results[:-1], [0], scores, True)
    assert not result_is_complete(results, [0], [1] + scores[1:], True)
    results[-1]['metrics']['skipped'] = 'source unavailable'
    assert not result_is_complete(results, [0], scores, True)
    results[-1] = {'benchmark': BENCHMARKS[-1], 'state': 'score_failed', 'score': None}
    assert not result_is_complete(results, [0], scores, True)


def test_evaluation_shard_ids_cannot_silently_overwrite(tmp_path):
    first, second = tmp_path / 'one.jsonl', tmp_path / 'two.jsonl'
    first.write_text(json.dumps({'uid': 1}) + '\n\n')
    second.write_text(json.dumps({'uid': '2'}) + '\n')
    assert validate_shard_uids([first, second]) == 2
    second.write_text(json.dumps({'uid': '1'}) + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        validate_shard_uids([first, second])


@pytest.mark.parametrize('uid', [None, True, [], {}, ''])
def test_evaluation_rejects_malformed_identity(tmp_path, uid):
    shard = tmp_path / 'shard.jsonl'
    shard.write_text(json.dumps({'uid': uid}) + '\n')
    with pytest.raises(ValueError):
        validate_shard_uids([shard])


@pytest.mark.parametrize("field,value", [("artifact_role", "historical_mechanism_control"),
                                         ("status", "unverified"), ("base_revision", "main")])
def test_unqualified_identity_rejected(tmp_path, field, value):
    manifest = fixture_release(tmp_path)
    manifest[field] = value
    save(tmp_path, manifest)
    with pytest.raises(ValueError):
        validate(tmp_path)


def test_evaluation_cannot_bind_another_checkpoint(tmp_path):
    manifest = fixture_release(tmp_path)
    manifest["evaluation"]["weights_sha256"]["adapter_model.safetensors"] = "f" * 64
    save(tmp_path, manifest)
    with pytest.raises(ValueError, match="exact uploaded weight set"):
        validate(tmp_path)


def test_changed_weights_rejected(tmp_path):
    fixture_release(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate(tmp_path)


def test_runtime_payload_rejected(tmp_path):
    fixture_release(tmp_path)
    (tmp_path / "train.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="unlisted"):
        validate(tmp_path)


def test_token_is_rejected_without_echo(tmp_path):
    manifest = fixture_release(tmp_path)
    secret = "hf" + "_" + "A" * 32
    (tmp_path / "README.md").write_text(secret)
    manifest["files"]["README.md"] = digest(tmp_path / "README.md")
    save(tmp_path, manifest)
    with pytest.raises(ValueError, match="credential-like") as exc:
        validate(tmp_path)
    assert secret not in str(exc.value)


def test_symlinked_weight_rejected(tmp_path):
    fixture_release(tmp_path)
    weight = tmp_path / "adapter_model.safetensors"
    weight.rename(tmp_path / "real-weight")
    weight.symlink_to("real-weight")
    with pytest.raises(ValueError, match="symlinked"):
        validate(tmp_path)


def test_adapter_base_must_match(tmp_path):
    manifest = fixture_release(tmp_path)
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Another/Model"}))
    manifest["files"]["adapter_config.json"] = digest(tmp_path / "adapter_config.json")
    save(tmp_path, manifest)
    with pytest.raises(ValueError, match="declared public base"):
        validate(tmp_path)


def test_traversal_filename_rejected(tmp_path):
    manifest = fixture_release(tmp_path)
    manifest["files"]["../README.md"] = "f" * 64
    save(tmp_path, manifest)
    with pytest.raises(ValueError, match="non-allowlisted"):
        validate(tmp_path)


@pytest.mark.parametrize("account,private,message", [
    ("different-user", True, "authenticated account"),
    ("owner", False, "visibility mismatch"),
])
def test_upload_account_and_visibility_guards_precede_writes(tmp_path, monkeypatch,
                                                            account, private, message):
    manifest = fixture_release(tmp_path)

    class FakeApi:
        def whoami(self):
            return {"name": account, "orgs": []}

        def repo_info(self, *args, **kwargs):
            return types.SimpleNamespace(private=private, siblings=[], sha="old")

        def create_repo(self, *args, **kwargs):
            pytest.fail("must not write a repository before identity/visibility checks")

        upload_folder = create_repo

    module = types.ModuleType("huggingface_hub")
    module.HfApi = FakeApi
    module.hf_hub_download = lambda *args, **kwargs: pytest.fail("must not download")
    errors = types.ModuleType("huggingface_hub.errors")
    errors.RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    with pytest.raises(ValueError, match=message):
        upload(tmp_path, manifest, "owner/Anvil-MOPD", "owner")
