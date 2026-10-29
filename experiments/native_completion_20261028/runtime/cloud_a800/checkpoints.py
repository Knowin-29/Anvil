"""Atomic completion receipts for model, optimizer, scheduler and rank RNGs."""
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seal(path, world):
    path = Path(path)
    required = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                "scheduler.pt", "trainer_state.json"]
    required += (["rng_state.pth"] if world == 1 else
                 [f"rng_state_{rank}.pth" for rank in range(world)])
    for name in required:
        assert (path / name).stat().st_size > 0, f"Incomplete checkpoint: {name}"
    state = json.loads((path / "trainer_state.json").read_text())
    assert path.name == f"checkpoint-{state['global_step']}"
    files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(path.iterdir())
             if p.is_file() and p.name != "checkpoint_complete.json"
             and not p.name.endswith(".pending")}
    receipt = dict(schema="anvil-complete-checkpoint-v1", world_size=world,
                   global_step=state["global_step"], files=files, sealed_at=time.time())
    save(path / "checkpoint_complete.json", receipt)
    return receipt


def verify(path, world):
    path = Path(path)
    receipt = json.loads((path / "checkpoint_complete.json").read_text())
    assert receipt["world_size"] == world, "Resume requires the same distributed world size"
    assert path.name == f"checkpoint-{receipt['global_step']}"
    for name, info in receipt["files"].items():
        assert Path(name).name == name
        p = path / name
        assert p.stat().st_size == info["bytes"] and sha(p) == info["sha256"], p
    return receipt


def latest(path, world):
    candidates = [p for p in Path(path).glob("checkpoint-*")
                  if p.name.removeprefix("checkpoint-").isdigit()
                  and (p / "checkpoint_complete.json").is_file()]
    for candidate in sorted(candidates, key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        # A corrupted sealed checkpoint is an integrity failure, not a reason
        # to silently rewind training and conceal the corruption.
        verify(candidate, world)
        return candidate
    return None
