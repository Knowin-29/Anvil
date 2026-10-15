#!/usr/bin/env python3
"""Resolve a full model or LoRA adapter checkpoint for GSA eval runners."""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_9B_BASES = [
    "${MODEL_ROOT}/Qwen/Qwen3.5-9B",
    "${MODEL_ROOT}/Qwen/Qwen3.5-9B",
]


def first_existing(paths: Iterable[Optional[str]]) -> Optional[str]:
    for raw in paths:
        if raw and Path(raw).is_dir():
            return str(Path(raw))
    return None


def resolve_checkpoint(path: str, *, base_model_path: str = "", adapters: str = "") -> Dict[str, Any]:
    ckpt = Path(path)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint path not found: {ckpt}")

    info: Dict[str, Any] = {
        "input_path": str(ckpt),
        "model_path": str(ckpt),
        "adapters": adapters,
        "kind": "full",
        "base_source": "checkpoint",
        "lora_rank": "",
    }

    adapter_config = ckpt / "adapter_config.json"
    model_config = ckpt / "config.json"
    if adapter_config.is_file() and not model_config.is_file():
        cfg = json.loads(adapter_config.read_text(encoding="utf-8"))
        cfg_base = cfg.get("base_model_name_or_path") or ""
        base = first_existing([base_model_path, *DEFAULT_9B_BASES, cfg_base])
        if not base:
            raise FileNotFoundError(
                f"cannot resolve base model for adapter {ckpt}; "
                f"base_model_path={base_model_path!r}, adapter_config base={cfg_base!r}"
            )
        info.update(
            {
                "model_path": base,
                "adapters": adapters or str(ckpt),
                "kind": "adapter",
                "adapter_config_base": cfg_base,
                "lora_rank": cfg.get("r") or "",
                "base_source": (
                    "explicit" if base_model_path and Path(base_model_path).is_dir()
                    else "default" if first_existing(DEFAULT_9B_BASES) == base
                    else "adapter_config"
                ),
            }
        )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--adapters", default="")
    parser.add_argument("--format", choices=["json", "shell"], default="json")
    args = parser.parse_args()

    info = resolve_checkpoint(args.path, base_model_path=args.base_model_path, adapters=args.adapters)
    if args.format == "shell":
        for key in ["model_path", "adapters", "kind", "base_source", "input_path", "lora_rank"]:
            env_key = {
                "model_path": "MODEL_PATH",
                "adapters": "ADAPTERS",
                "kind": "CKPT_KIND",
                "base_source": "BASE_SOURCE",
                "input_path": "CKPT_INPUT_PATH",
                "lora_rank": "LORA_RANK",
            }[key]
            print(f"{env_key}={shlex.quote(str(info.get(key) or ''))}")
    else:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
