"""Thin entry points into the consolidated implementations, not replacement pipelines."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "align": ["-m", "trains.flywheel.alignment"],
    "generate": ["trains/flywheel/generate_multiform.py"],
    "pipeline": ["-m", "apps.data_flywheel.pipeline"],
    "prepare": ["engineering/anvil.py"],
    "evaluate": ["eval/run_benchmark.py"],
    "dashboard": ["eval/dashboard.py"],
    "replay": ["scripts/replay_results.py"],
}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "gsa", "doctor"])
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    remaining = argv[1:]
    os.environ.setdefault("ANVIL_ROOT", str(ROOT))
    os.environ.setdefault("ANVIL_WORKTREE", str(ROOT))
    os.environ.setdefault("GSA_ROOT", str(ROOT))
    if args.command == "doctor":
        modules = ("numpy", "PIL", "yaml", "requests", "flask", "torch", "transformers", "peft", "swift")
        print(json.dumps({"python":sys.version.split()[0], "dependencies":{
            name: importlib.util.find_spec(name) is not None for name in modules},
            "datasets_index": (ROOT/"registry/datasets.json").is_file(),
            "models_index": (ROOT/"registry/models.json").is_file()}, indent=2))
        return 0
    if args.command == "gsa":
        sub = argparse.ArgumentParser(description="Run the inherited configurable G/S/A pipeline")
        sub.add_argument("--config", required=True)
        parsed = sub.parse_args(remaining)
        from apps.data_flywheel.gsa.runner import GSAv0Runner
        print(json.dumps(GSAv0Runner(parsed.config).run(), indent=2, default=str))
        return 0
    return subprocess.call([sys.executable, *COMMANDS[args.command], *remaining], cwd=ROOT)

if __name__ == "__main__":
    raise SystemExit(main())
