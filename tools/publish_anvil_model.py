"""Validate a curated final checkpoint folder; upload only with --upload.

Run as ``python -m tools.publish_anvil_model --help`` from an installed checkout.
This does not select, merge, train or certify an unidentified checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from tools.export_anvil_release import (
    SECRET, KEY_UUID, PRIVATE_IDENTITY, has_deployed_endpoint,
)

MANIFEST = "MODEL_RELEASE.json"
WEIGHT = re.compile(r"(?:adapter_model|model(?:-\d{5}-of-\d{5})?)\.safetensors\Z")
METADATA = {
    "README.md", "LICENSE", "NOTICE.md", ".gitattributes", "config.json",
    "adapter_config.json", "model.safetensors.index.json", "generation_config.json",
    "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
    "chat_template.json", "chat_template.jinja",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate(folder: Path) -> dict:
    """Check upload bytes and declared identity; not a substitute for evaluation."""
    folder = folder.resolve(strict=True)
    manifest_path = folder / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("release manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "anvil-model-release-v1":
        raise ValueError("unsupported or missing release schema")
    if manifest.get("artifact_role") != "final_mopd" or manifest.get("status") != "verified":
        raise ValueError("only an explicitly verified final_mopd identity can be released")
    if not REPO.fullmatch(manifest.get("base_model", "")):
        raise ValueError("base_model must be a Hub repository, not a server path")
    if not REVISION.fullmatch(manifest.get("base_revision", "")):
        raise ValueError("an immutable base revision is required")
    if not SHA256.fullmatch(manifest.get("training_manifest_sha256", "")):
        raise ValueError("a verified training-manifest digest is required")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("empty upload file inventory")
    if "README.md" not in files or not ({"LICENSE", "NOTICE.md"} & files.keys()):
        raise ValueError("model card and weight license/notice are required")
    for name, checksum in files.items():
        if not (name in METADATA or WEIGHT.fullmatch(name)):
            raise ValueError("non-allowlisted file in model inventory")
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or symlinked upload file: {name}")
        if not isinstance(checksum, str) or not SHA256.fullmatch(checksum):
            raise ValueError(f"invalid checksum: {name}")
        if digest(path) != checksum:
            raise ValueError(f"checksum mismatch: {name}")
    weights = {k: v for k, v in files.items() if WEIGHT.fullmatch(k)}
    if not weights:
        raise ValueError("no weight files; a model card alone is not a weight release")
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("weights_sha256") != weights:
        raise ValueError("evaluation must bind the exact uploaded weight set")
    if not SHA256.fullmatch(evaluation.get("evidence_manifest_sha256", "")) or not evaluation.get("protocol_id"):
        raise ValueError("evaluation evidence digest and protocol identity are required")
    if "adapter_model.safetensors" in weights:
        if len(weights) != 1 or "adapter_config.json" not in files:
            raise ValueError("adapter release must contain one adapter and its config")
        config = json.loads((folder / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != manifest["base_model"]:
            raise ValueError("adapter config must reference the declared public base model")
        if config.get("revision") != manifest["base_revision"]:
            raise ValueError("adapter config must pin the declared base revision")
    else:
        if "config.json" not in files or "tokenizer_config.json" not in files:
            raise ValueError("merged release requires model and tokenizer configuration")
        if "model.safetensors" not in weights:
            if "model.safetensors.index.json" not in files:
                raise ValueError("sharded checkpoint requires its index")
            index = json.loads((folder / "model.safetensors.index.json").read_text())
            if set(index.get("weight_map", {}).values()) != set(weights):
                raise ValueError("shard index and uploaded weights differ")
        elif len(weights) != 1:
            raise ValueError("cannot mix single-file and sharded weights")
    # Reject unlisted root files before handing the folder to the Hub client.
    extras = {p.name for p in folder.iterdir()} - set(files) - {MANIFEST, ".cache"}
    if extras:
        raise ValueError("release folder contains unlisted files or directories")
    for name in [*files, MANIFEST]:
        if WEIGHT.fullmatch(name):
            continue
        raw = (folder / name).read_bytes()
        if SECRET.search(raw) or KEY_UUID.search(raw):
            raise ValueError(f"credential-like content in {name}; value not printed")
        if PRIVATE_IDENTITY.search(raw) or has_deployed_endpoint(raw):
            raise ValueError(f"private path or deployment endpoint in {name}; value not printed")
    return manifest


def upload(folder: Path, manifest: dict, repo_id: str, expected_account: str,
           public: bool = False) -> dict:
    if not REPO.fullmatch(repo_id) or not expected_account:
        raise ValueError("explicit repo-id and expected-account are required for upload")
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    # Revalidate immediately before any remote write. Credentials come from the
    # Hub's configured environment/login; never accept or print a CLI token.
    if validate(folder) != manifest:
        raise ValueError("release changed after preflight")
    api = HfApi()
    actor = api.whoami()
    if actor.get("name") != expected_account:
        raise ValueError("authenticated account differs from expected-account")
    namespace = repo_id.split("/")[0]
    if namespace != actor["name"] and namespace not in {o["name"] for o in actor.get("orgs", [])}:
        raise ValueError("target namespace is not the authenticated account or an organization membership")
    try:
        info = api.repo_info(repo_id, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="model", private=not public)
        info = api.repo_info(repo_id, repo_type="model")
    if info.private != (not public):
        raise ValueError("repository visibility mismatch; no visibility change performed")
    allowed = set(manifest["files"]) | {MANIFEST, ".gitattributes"}
    if {s.rfilename for s in info.siblings or []} - allowed:
        raise ValueError("remote has unrelated/stale files; no deletion performed")
    result = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(folder),
        allow_patterns=sorted(set(manifest["files"]) | {MANIFEST}),
        parent_commit=info.sha,
        commit_message="Release verified Anvil final-model artifacts",
    )
    verified = api.repo_info(repo_id, repo_type="model", revision=result.oid, files_metadata=True)
    remote = {s.rfilename: s for s in verified.siblings}
    expected = {**manifest["files"], MANIFEST: digest(folder / MANIFEST)}
    for name, checksum in expected.items():
        if name not in remote:
            raise ValueError("uploaded revision is missing an expected file")
        lfs = remote[name].lfs
        if lfs is not None:
            actual = lfs.sha256
        else:
            actual = digest(Path(hf_hub_download(repo_id, name, revision=result.oid)))
        if actual != checksum:
            raise ValueError(f"remote checksum mismatch: {name}")
    return {"repo_id": repo_id, "revision": result.oid,
            "private": verified.private, "verified_files": len(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--upload", action="store_true", help="Opt in to authenticated Hub writes")
    parser.add_argument("--repo-id")
    parser.add_argument("--expected-account")
    parser.add_argument("--public", action="store_true", help="Explicitly authorize a public model repository")
    args = parser.parse_args()
    try:
        manifest = validate(args.folder)
        if args.upload:
            result = upload(args.folder, manifest, args.repo_id or "", args.expected_account or "", args.public)
        else:
            result = {"validated": True, "uploaded": False, "files": len(manifest["files"]),
                      "scope": "byte inventory and declared binding; not a new model evaluation"}
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Release blocked: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
