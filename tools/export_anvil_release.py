#!/usr/bin/env python3
"""Export only committed Anvil source; never publish or include old Git history."""
from __future__ import annotations
import argparse
import hashlib
import ipaddress
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

SECRET = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ark-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b")
KEY_UUID = re.compile(rb"(?:api[_-]?key(?:_env)?|token|password|secret)[^\n]{0,80}[\"'][0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}[\"']", re.I)
URL_HOST = re.compile(rb"https?://(\[[0-9a-f:]+\]|[^/\s:\"'<>]+)", re.I)
DEPLOYMENT_HOST = re.compile(rb"[a-z0-9-]+\.apigateway-[a-z0-9-]+\.volceapi\.com", re.I)
PRIVATE_IDENTITY = re.compile(rb"(?<![A-Za-z0-9_/])(?:/(?:root|home)/|/[A-Za-z0-9_-]+/(?:demo|foundation|sim)/)", re.I)
PRIVATE_PARTS = {".git", ".cache", ".venv", "__pycache__", "logs", "output", "outputs", "checkpoints", "private", "artifacts", "wandb", "arxiv"}
PRIVATE_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".log", ".out", ".err", ".pem", ".key", ".tex", ".pdf", ".pptx", ".zip", ".tar", ".gz", ".pyc"}

def has_deployed_endpoint(raw):
    if DEPLOYMENT_HOST.search(raw):
        return True
    for host in URL_HOST.findall(raw):
        try:
            address = ipaddress.ip_address(host.decode().strip("[]"))
        except ValueError:
            continue
        if not (address.is_loopback or address.is_unspecified):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="New .tar.gz or .zip path outside the repository")
    parser.add_argument("--check-only", action="store_true", help="Scan tracked worktree files without writing an archive or requiring a clean tree")
    parser.add_argument("--anonymous-review", action="store_true", help="Redact supplied identity terms and omit source commit metadata")
    parser.add_argument("--redact-term", action="append", default=[], help="Literal identity term to mask in review text; repeat as needed, not a regex")
    args = parser.parse_args()
    if args.anonymous_review and (not args.redact_term or args.check_only):
        parser.error("anonymous review requires explicit redaction terms and an archive output")
    if args.redact_term and not args.anonymous_review:
        parser.error("redaction terms require --anonymous-review")
    if any(len(t.strip()) < 3 for t in args.redact_term):
        parser.error("redaction terms must have at least three non-whitespace characters")
    redact = re.compile("|".join(re.escape(t) for t in args.redact_term), re.I) if args.redact_term else None
    root = args.worktree.resolve()
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only is set")
    output = args.output.resolve() if args.output else None
    if output is not None and output.is_relative_to(root):
        parser.error("output must be outside the source repository")
    if output is not None and output.exists():
        parser.error("refusing to overwrite an existing release")
    def git(*command):
        return subprocess.check_output(["git", "-C", str(root), *command])
    if not args.check_only and git("status", "--porcelain").strip():
        parser.error("commit/review local changes before exporting a frozen source release")
    commit = git("rev-parse", "HEAD").decode().strip()
    names = [n.decode() for n in git("ls-files", "-z").split(b"\0") if n]
    manifest = {"project":"Anvil", "license":"MIT", "commit":commit, "files":{}}
    if args.anonymous_review:
        manifest.pop("commit")
        manifest["scope"] = "Anonymous review rendering; identity terms masked, license clauses retained"
    redacted_files = 0
    payloads = []
    for name in names:
        path = root/name
        if path.is_symlink() or not path.is_file():
            parser.error(f"unsupported tracked file: {name}")
        if ((path.name.startswith('.env') and path.name != '.env.example')
            or path.suffix.lower() in PRIVATE_SUFFIXES
            or PRIVATE_PARTS.intersection(Path(name).parts)
            or name == 'engineering/OPERATIONS.md'
            or path.name.startswith('pytorch_model') and path.suffix == '.bin'):
            parser.error(f"private/runtime asset tracked: {name}")
        raw = path.read_bytes()
        if SECRET.search(raw) or KEY_UUID.search(raw):
            parser.error(f"credential-like content detected in {name}; content not printed")
        if has_deployed_endpoint(raw):
            parser.error(f"deployed endpoint detected in {name}; content not printed")
        if PRIVATE_IDENTITY.search(raw):
            parser.error(f"private identity or server path detected in {name}; content not printed")
        if redact:
            if redact.search(name):
                parser.error("identity term in filename; review its references before exporting")
            try:
                original = raw.decode("utf-8")
            except UnicodeDecodeError:
                if any(t.lower().encode() in raw.lower() for t in args.redact_term):
                    parser.error(f"identity term in binary file {name}; inspect metadata first")
            else:
                revised = redact.sub("[ANON]", original)
                if revised != original:
                    redacted_files += 1
                    raw = revised.encode("utf-8")
        manifest['files'][name] = hashlib.sha256(raw).hexdigest()
        payloads.append((name,raw,0o755 if path.stat().st_mode & 0o111 else 0o644))
    if args.check_only:
        print(json.dumps({'tracked_files_checked':len(names),'archive_written':False,
                          'scope':'tracked current files only; untracked files and Git history not audited'}))
        return
    payloads.append(('SOURCE_CHECKSUMS.json',json.dumps(manifest,indent=2).encode(),0o644))
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.suffix == '.zip':
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw, mode in payloads:
                info = zipfile.ZipInfo('Anvil/'+name, (1980,1,1,0,0,0))
                info.create_system = 3
                info.external_attr = (0o100000 | mode) << 16
                archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED)
    else:
        with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, raw, mode in payloads:
                info = tarfile.TarInfo('Anvil/'+name)
                info.size=len(raw); info.mode=mode; info.mtime=0
                archive.addfile(info,io.BytesIO(raw))
    print(json.dumps({'files':len(names),'license':'MIT','anonymous_review':args.anonymous_review,
                      'redacted_files':redacted_files,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))

if __name__=='__main__':
    main()
