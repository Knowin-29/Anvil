"""Release checks operate on isolated Git fixtures, never production artifacts."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


EXPORTER = Path(__file__).resolve().parents[1] / "tools/export_anvil_release.py"


def test_historical_registries_do_not_claim_a_final_model_release():
    root = EXPORTER.parents[1]
    for name in ("models.json", "model_registry.json"):
        registry = json.loads((root / "registry" / name).read_text())
        assert registry["artifact_role"] == "historical_mechanism_control"
        assert registry["eligible_for_final_model_release"] is False
        final = registry["final_release"]
        assert final["checkpoint"] is None
        assert final["weights_sha256"] is None
        assert final["evaluation_manifest"] is None
        assert final["uploaded"] is False
        assert registry["weights_sha256"] != final["weights_sha256"]


def test_missing_verifier_key_fails_before_reading_image_or_network():
    source = EXPORTER.parents[1] / "scripts/verifier_service.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_seed_vlm_predict")
    def unexpected_call(*args, **kwargs):
        pytest.fail("verification without credentials must not read images or send requests")
    namespace = {"SEED_VLM_KEY": "", "open": unexpected_call}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    with pytest.raises(RuntimeError, match="VERIFIER_SEED_KEY must be configured"):
        namespace["_seed_vlm_predict"]("unused.jpg", "verify")
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "SEED_VLM_KEY"
                              for t in node.targets))
    assert isinstance(assignment.value, ast.Call)
    assert ast.literal_eval(assignment.value.args[1]) == ""


@pytest.mark.parametrize("version", ["42", "43"])
@pytest.mark.parametrize("approved", ["", "http://localhost:1/different"])
def test_depth_cli_rejects_missing_or_mismatched_authorization(tmp_path, version, approved):
    command = [sys.executable, str(EXPORTER.parents[1] / f"tools/generate_depth_v{version}_pairwise.py"),
               "--index", str(tmp_path / "missing.index"), "--exclude-manifest",
               str(tmp_path / "missing.json"), "--out", str(tmp_path / "output"), "--canary", "8"]
    env = {**os.environ, "AUTHORIZED_ENDPOINT": approved, "GSA_VLM_URL": "http://localhost:1/v1"}
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "not the explicitly authorized endpoint" in result.stderr
    assert not (tmp_path / "output").exists()


def fixture_repo(tmp_path, name="README.md", content="Example research code\n"):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    return root


@pytest.mark.parametrize("prefix", ["gh" + "p_", "github" + "_pat_", "hf" + "_", "ar" + "k-"])
def test_credentials_fail_without_echo(tmp_path, prefix):
    secret = prefix + "A" * 40
    root = fixture_repo(tmp_path, content=secret)
    result = subprocess.run([sys.executable, str(EXPORTER), "--worktree", str(root),
                             "--check-only"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "credential-like" in result.stderr
    assert secret not in result.stderr + result.stdout


def test_commented_key_uuid_is_not_publishable(tmp_path):
    secret = "12345678" + "-1234" * 3 + "-123456789abc"
    root = fixture_repo(tmp_path, content='# api_key_env: "' + secret + '"')
    result = subprocess.run([sys.executable, str(EXPORTER), "--worktree", str(root),
                             "--check-only"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "credential-like" in result.stderr
    assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize("host", ["192.168.1.10", "[fd00::1]",
                                  "example." + "apigateway-region" + ".volceapi.com"])
def test_deployment_endpoint_is_not_publishable(tmp_path, host):
    endpoint = "http" + "://" + host + ":8000/v1"
    root = fixture_repo(tmp_path, content=endpoint)
    result = subprocess.run([sys.executable, str(EXPORTER), "--worktree", str(root),
                             "--check-only"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "deployed endpoint" in result.stderr
    assert endpoint not in result.stdout + result.stderr


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]", "0.0.0.0"])
def test_local_endpoint_and_package_versions_are_allowed(tmp_path, host):
    root = fixture_repo(tmp_path, content="http" + "://" + host + ":8000/v1\nopencv==4.10.0.84")
    result = subprocess.run([sys.executable, str(EXPORTER), "--worktree", str(root),
                             "--check-only"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["run.log", "private/notes.md", "engineering/OPERATIONS.md"])
def test_runtime_files_fail(tmp_path, name):
    root = fixture_repo(tmp_path, name=name)
    result = subprocess.run([sys.executable, str(EXPORTER), "--worktree", str(root),
                             "--check-only"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "private/runtime" in result.stderr


def test_private_path_fails(tmp_path):
    root = fixture_repo(tmp_path, content="/" + "root" + "/private-project")
    result = subprocess.run([sys.executable, str(EXPORTER), "--worktree", str(root),
                             "--check-only"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "private identity or server path" in result.stderr


def test_explicit_placeholder_is_not_a_server_identity(tmp_path):
    root = fixture_repo(tmp_path, content="/path/to/workspace/foundation/contributor/data")
    result = subprocess.run([sys.executable, str(EXPORTER), "--worktree", str(root),
                             "--check-only"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_clean_export_contains_checksums_not_history(tmp_path):
    root = fixture_repo(tmp_path)
    output = tmp_path / "release.tar.gz"
    result = subprocess.run([sys.executable, str(EXPORTER), "--worktree", str(root),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == {"Anvil/README.md", "Anvil/SOURCE_CHECKSUMS.json"}
    assert json.loads(result.stdout)["files"] == 1


def test_dirty_check_does_not_authorize_export(tmp_path):
    root = fixture_repo(tmp_path)
    (root / "README.md").write_text("Reviewed current file\n")
    command = [sys.executable, str(EXPORTER), "--worktree", str(root)]
    check = subprocess.run(command + ["--check-only"], capture_output=True, text=True)
    assert check.returncode == 0
    assert json.loads(check.stdout)["archive_written"] is False
    output = tmp_path / "release.tar.gz"
    export = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
    assert export.returncode != 0
    assert not output.exists()


def test_review_zip_redacts_identity_preserves_license_and_checksums(tmp_path):
    text = 'Copyright (c) 2026 FixtureAuthor\nPermission is hereby granted.\n'
    root = fixture_repo(tmp_path, name='LICENSE', content=text)
    output = tmp_path / 'review.zip'
    result = subprocess.run([sys.executable, str(EXPORTER), '--worktree', str(root),
        '--output', str(output), '--anonymous-review', '--redact-term', 'FixtureAuthor'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(output) as archive:
        license_text = archive.read('Anvil/LICENSE').decode()
        assert 'FixtureAuthor' not in license_text
        assert 'Copyright (c) 2026 [ANON]' in license_text
        assert 'Permission is hereby granted.' in license_text
        manifest = json.loads(archive.read('Anvil/SOURCE_CHECKSUMS.json'))
        assert 'commit' not in manifest
        import hashlib
        assert manifest['files']['LICENSE'] == hashlib.sha256(license_text.encode()).hexdigest()
        assert archive.comment == b''
        assert archive.getinfo('Anvil/LICENSE').date_time == (1980,1,1,0,0,0)
    assert (root / 'LICENSE').read_text() == text


def test_review_requires_explicit_identity_terms(tmp_path):
    root = fixture_repo(tmp_path)
    output = tmp_path / 'review.zip'
    result = subprocess.run([sys.executable, str(EXPORTER), '--worktree', str(root),
        '--output', str(output), '--anonymous-review'], capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()


def test_export_preserves_executable_mode(tmp_path):
    root = fixture_repo(tmp_path, name='run.sh', content='#!/bin/sh\nexit 0\n')
    (root / 'run.sh').chmod(0o755)
    subprocess.run(['git','-C',str(root),'add','run.sh'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Fixture',
                    '-c','user.email=fixture@example.invalid','commit','-qm','mode'],check=True)
    output = tmp_path / 'release.tar.gz'
    result = subprocess.run([sys.executable,str(EXPORTER),'--worktree',str(root),
        '--output',str(output)],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    with tarfile.open(output) as archive:
        assert archive.getmember('Anvil/run.sh').mode == 0o755
