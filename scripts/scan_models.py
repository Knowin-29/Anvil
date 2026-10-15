#!/usr/bin/env python3
"""Inventory local checkpoints against the controlled baseline registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            pass
    return total


def inspect(entry: dict) -> dict:
    path = Path(entry["path"])
    config_path = path / "config.json"
    config = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {"_config_error": "invalid_json"}
    weights = sorted(
        str(p.relative_to(path))
        for pattern in ("*.safetensors", "*.bin", "*.pth")
        for p in path.glob(pattern)
        if p.is_file()
    ) if path.is_dir() else []
    architecture = config.get("architectures") or [config.get("type") or config.get("model_type")]
    ready = path.is_dir() and config_path.is_file() and bool(weights)
    result = dict(entry)
    size = directory_bytes(path) if path.is_dir() else 0
    result.update({
        "exists": path.is_dir(),
        "ready": ready,
        "bytes": size,
        "size_gib": round(size / (1024 ** 3), 2),
        "model_type": config.get("model_type") or config.get("type"),
        "architectures": [value for value in architecture if value],
        "weight_files": len(weights),
        "table_ready": bool(ready and entry.get("table_eligible")),
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="configs/models/baselines.yaml")
    ap.add_argument("--out", default="artifacts/model_inventory.json")
    ap.add_argument(
        "--roots", nargs="*",
        default=[
            "/path/to/environment/models",
            "/path/to/environment/hf_models",
        ],
    )
    args = ap.parse_args()

    registry = yaml.safe_load(Path(args.registry).read_text())
    inventory = [inspect(entry) for entry in registry["models"]]
    registered = {str(Path(entry["path"]).resolve()) for entry in registry["models"]}

    unregistered = []
    for root_value in args.roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for config in root.rglob("config.json"):
            model_path = str(config.parent.resolve())
            if model_path in registered:
                continue
            unregistered.append(inspect({
                "id": f"unregistered:{config.parent.name}",
                "display_name": config.parent.name,
                "source": "local_scan",
                "path": str(config.parent),
                "protocol": "unknown",
                "harness": "unassigned",
                "table_eligible": False,
                "exclusion_reason": "Not reviewed in configs/models/baselines.yaml.",
            }))

    report = {
        "schema_version": 1,
        "registry": args.registry,
        "models": inventory,
        "unregistered": unregistered,
        "summary": {
            "registered": len(inventory),
            "ready": sum(item["ready"] for item in inventory),
            "table_ready": sum(item["table_ready"] for item in inventory),
            "unregistered": len(unregistered),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("id                               ready table  GiB  harness")
    for item in inventory:
        print(f"{item['id']:<32} {str(item['ready']):<5} "
              f"{str(item['table_ready']):<5} {item['size_gib']:>5.2f}  {item['harness']}")
    if unregistered:
        print(f"unregistered checkpoints: {len(unregistered)}")
    print(f"inventory: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
