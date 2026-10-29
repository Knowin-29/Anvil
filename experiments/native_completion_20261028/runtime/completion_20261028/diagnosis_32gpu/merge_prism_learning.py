"""Create derived BF16 inference copies; original base and LoRA bytes stay immutable."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
RECOVERY = ROOT.parent
BASE = Path("/Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B")
RUNS = RECOVERY.parent / "arxiv"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    adapter = args.adapter.resolve()
    dest = args.output.resolve()
    dest.mkdir(exist_ok=False)
    status = json.loads((adapter.parent / "complete.json").read_text())
    protocol = json.loads((RUNS / "native_a_specialist_20261012_v4/protocol.json").read_text())
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert adapter.is_relative_to(ROOT / "diagnosis_32gpu/prism_learning_repairs")
    assert status['state'] == 'repair_complete_requires_evaluation' and status['step'] == 2048 and status['smoke'] is False
    import sys
    sys.path.insert(0, str(RECOVERY / 'cloud_a800'))
    from checkpoints import verify
    verify(adapter, 4)
    repair_protocol = json.loads((adapter.parent / 'protocol.json').read_text())
    assert repair_protocol['heldout_training'] is False and repair_protocol['source_pool'] == 5220000
    assert sha(adapter / "adapter_model.safetensors") == status["adapter_sha256"]
    assert config["r"] == 64 and config["lora_alpha"] == 128 and not config["use_dora"]
    assert not config["modules_to_save"] and not config["rank_pattern"] and not config["alpha_pattern"]
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    pairs = {}
    used = set()
    for key in weights:
        if not key.endswith(".lora_A.weight"):
            continue
        assert key.startswith("base_model.model.")
        target = key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
        bkey = key.replace(".lora_A.weight", ".lora_B.weight")
        assert target in index["weight_map"] and bkey in weights
        pairs[target] = (weights[key], weights[bkey])
        used.update((key, bkey))
    assert used == set(weights), "Unmapped adapter parameters"
    receipt = {"original_base": str(BASE), "original_adapter": str(adapter),
               "adapter_sha256": status["adapter_sha256"], "target_tensors": len(pairs),
               "algorithm": "float32 W + (alpha/r) * (B @ A), rounded to original BF16",
               "originals_modified": False, "numerically_identical_to_unmerged_claimed": False,
               "started_at": time.time(), "shards": {}}
    merged = set()
    for name in sorted(set(index["weight_map"].values())):
        src = BASE / name
        original_sha = sha(src)
        assert original_sha == protocol["base_files_sha256"][name], f"Base identity mismatch: {name}"
        tensors = load_file(str(src))
        for key in list(tensors):
            if key not in pairs:
                continue
            a, b = pairs[key]
            w = tensors[key]
            assert tuple(b.shape[:1] + a.shape[1:]) == tuple(w.shape)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            result = (w.float() + (b.float() @ a.float()) * (config["lora_alpha"] / config["r"])).to(w.dtype)
            assert torch.isfinite(result).all()
            tensors[key] = result.contiguous()
            merged.add(key)
        output = dest / name
        save_file(tensors, str(output), metadata={"format": "pt"})
        receipt["shards"][name] = {"original_sha256": original_sha, "derived_sha256": sha(output),
                                   "bytes": output.stat().st_size}
        (dest / "merge_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(adapter.parent.name, name, "saved", flush=True)
        del tensors
    assert merged == set(pairs)
    for p in BASE.iterdir():
        if p.is_file() and p.suffix != ".safetensors":
            shutil.copy2(p, dest / p.name)
    receipt.update(completed_at=time.time(), state="verified_derived_inference_copy")
    (dest / "merge_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
