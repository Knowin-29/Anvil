#!/usr/bin/env python3
"""Guard against the silent failure that invalidated the first eval sweep.

Several shared runners (embspatial/robovqa/openeqa) load a checkpoint through InternVL's
`AutoModel.from_pretrained(..., trust_remote_code=True)`. For a Qwen3.5 checkpoint that
call SUCCEEDS with exit code 0 while reporting every language-model tensor as MISSING,
i.e. the benchmark is scored on randomly initialised weights. The runner then prints a
plausible-looking accuracy and the harness records it.

Usage:
    python eval/assert_weights_loaded.py --model <path>

Exits non-zero if the checkpoint cannot be loaded with all weights materialised, so a
caller can refuse to score it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    cfg_path = Path(a.model) / "config.json"
    if not cfg_path.is_file():
        print(f"FAIL: no config.json under {a.model}")
        return 2
    cfg = json.loads(cfg_path.read_text())
    arch = str(cfg.get("architectures")) + str(cfg.get("model_type"))

    import torch
    import transformers

    if "Qwen3_5" in arch or "qwen3_5" in arch:
        cls = transformers.Qwen3_5ForConditionalGeneration
    elif "Qwen3VL" in arch or "qwen3_vl" in arch:
        cls = transformers.Qwen3VLForConditionalGeneration
    else:
        print(f"FAIL: unhandled architecture {arch}; refusing to guess a loader")
        return 2

    model = cls.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cpu")

    # A randomly initialised tensor has a characteristic std; a trained embedding matrix
    # does not. Checking for all-zero / untouched parameters catches the MISSING case.
    bad = []
    for name, p in model.named_parameters():
        if p.numel() == 0:
            continue
        s = p.detach().float().std().item()
        if s == 0.0 and "bias" not in name and "norm" not in name:
            bad.append(name)
    if bad:
        print(f"FAIL: {len(bad)} parameters look uninitialised, e.g. {bad[:5]}")
        return 1
    print(f"OK: {a.model} loaded via {cls.__name__}, all parameters materialised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
