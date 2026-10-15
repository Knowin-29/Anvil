#!/usr/bin/env python3
"""Submit / poll GSA-Forge jobs on Volcengine ML Platform (custom task).

Reuses the verified toolchain pattern from ActionMap/genroute: `volc ml_task
submit --conf <yaml>`, single 8xA800 node, vePFS /path/to/workspace/foundation mounted RW,
image already contains ms-swift. Submission records are written atomically so a
job is never double-submitted by accident.

Prereq: `volc configure --region cn-beijing` must have populated
~/.volc/credentials. Run `python submit_volc.py check` to verify.

Examples:
  python scripts/submit_volc.py check
  python scripts/submit_volc.py submit --conf configs/volc/smoke.yaml --name gsa-smoke
  python scripts/submit_volc.py status --id t-20260808...-abcde
  python scripts/submit_volc.py logs   --id t-20260808...-abcde --lines 120
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
QUEUE_ID = os.environ.get("ANVIL_RESOURCE_QUEUE_ID", "")
IMAGE = os.environ.get("ANVIL_CONTAINER_IMAGE", "")
STATE_PATH = REPO / "configs/volc/submissions.json"
TASK_ID_RE = re.compile(r"t-\d{14}-[a-z0-9]+")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)


def cmd_check(_: argparse.Namespace) -> int:
    res = _run(["volc", "ml_task", "list", "--output", "json"])
    combined = (res.stdout or "") + (res.stderr or "")
    if "AccessKeyId or SecretAccessKey is empty" in combined or "InvalidCredential" in combined:
        print("NO CREDENTIALS: run `volc configure --region cn-beijing` first.", file=sys.stderr)
        print(combined[:400], file=sys.stderr)
        return 2
    if res.returncode != 0:
        print(combined[:800], file=sys.stderr)
        return res.returncode
    print("OK: volc credentials valid, ml_task reachable.")
    return 0


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"schema_version": 1, "queue_id": QUEUE_ID, "submissions": []}
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload.get("submissions"), list):
        raise ValueError(f"invalid submission state: {STATE_PATH}")
    return payload


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def cmd_submit(args: argparse.Namespace) -> int:
    if not QUEUE_ID or not IMAGE:
        raise SystemExit("Set ANVIL_RESOURCE_QUEUE_ID and ANVIL_CONTAINER_IMAGE before submission")
    conf = (REPO / args.conf).resolve() if not os.path.isabs(args.conf) else Path(args.conf)
    if not conf.is_file():
        print(f"conf not found: {conf}", file=sys.stderr)
        return 1
    # Volc packages UserCodePath (relative to the yaml's dir) and uploads it.
    command = [
        "volc", "ml_task", "submit",
        "--conf", str(conf),
        "--task_name", args.name,
        "--resource_queue_id", QUEUE_ID,
        "--image", IMAGE,
        "--output", "json",
    ]
    if args.entrypoint:
        command += ["--entrypoint", args.entrypoint]
    print("SUBMIT " + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    state = _load_state()
    if any(s["task_name"] == args.name for s in state["submissions"]) and not args.force:
        print(f"'{args.name}' already submitted; use --force to resubmit.", file=sys.stderr)
        return 1
    res = _run(command)
    combined = "\n".join(p for p in (res.stdout, res.stderr) if p)
    if res.returncode != 0:
        print(combined, file=sys.stderr)
        return res.returncode
    m = TASK_ID_RE.search(combined)
    if not m:
        print(combined, file=sys.stderr)
        print("no task id parsed", file=sys.stderr)
        return 1
    task_id = m.group(0)
    state["submissions"].append({
        "task_id": task_id,
        "task_name": args.name,
        "conf": str(conf.relative_to(REPO)) if str(conf).startswith(str(REPO)) else str(conf),
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
    })
    _save_state(state)
    print(f"SUBMITTED task_id={task_id} state={STATE_PATH}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    res = _run(["volc", "ml_task", "get", "--id", args.id, "--output", "json"])
    print((res.stdout or "") + (res.stderr or ""))
    return res.returncode


def cmd_logs(args: argparse.Namespace) -> int:
    res = _run(["volc", "ml_task", "logs", "--task", args.id, "--lines", str(args.lines)])
    print((res.stdout or "") + (res.stderr or ""))
    return res.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check").set_defaults(func=cmd_check)

    ps = sub.add_parser("submit")
    ps.add_argument("--conf", required=True)
    ps.add_argument("--name", required=True)
    ps.add_argument("--entrypoint", default="")
    ps.add_argument("--dry-run", action="store_true")
    ps.add_argument("--force", action="store_true")
    ps.set_defaults(func=cmd_submit)

    pst = sub.add_parser("status")
    pst.add_argument("--id", required=True)
    pst.set_defaults(func=cmd_status)

    pl = sub.add_parser("logs")
    pl.add_argument("--id", required=True)
    pl.add_argument("--lines", type=int, default=80)
    pl.set_defaults(func=cmd_logs)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
