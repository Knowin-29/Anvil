#!/usr/bin/env python3
"""Fixed-protocol source-native PRISM specialist, not final Anvil MOPD.

Requires explicit scene-partitioned exports from map_prism_native.py and a local
Qwen3.5-9B-compatible base. Runs a separate 2-step smoke, fresh-base 512-step LoRA
training, then fixed-checkpoint and base-model held-out evaluation. No held-out
checkpoint selection, hard wall-clock timeout, or implicit GPU eviction.
Use --cpu-preflight-only to encode all training rows without a GPU.

CPU preflight was exercised with ms-swift 4.2.0.dev0; GPU stages remain pending
validation. Run artifacts contain local paths and need a separate privacy review.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

if __package__:
    from .response_text import POLICY as RESPONSE_POLICY, generated_text, SWIFT_NON_THINKING_PREFIX
else:
    from response_text import POLICY as RESPONSE_POLICY, generated_text, SWIFT_NON_THINKING_PREFIX

CONTRACT = "anvil-prism-gripper-se3-v1"
ARGS = None
RUN = BASE = TRAIN = EVALUATION = DATA_MANIFEST = None
RUN_CREATED = False


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def status(stage, **values):
    result = {"stage": stage, "pid": os.getpid(), "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values}
    save(RUN / "status.json", result)
    print(json.dumps(result), flush=True)


def environment():
    env = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "NPROC_PER_NODE", "PYTHONPATH"):
        env.pop(key, None)
    ARGS.runtime_parent.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix="anvil-prism-", dir=ARGS.runtime_parent))
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1", USE_HF="1", DS_IGNORE_CUDA_DETECTION="1", WANDB_MODE="disabled",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", IMAGE_MAX_TOKEN_NUM="1536",
        HF_HOME=str(RUN / "cache/hf"), HF_DATASETS_CACHE=str(RUN / "cache/datasets"),
        XDG_CACHE_HOME=str(RUN / "cache/xdg"), TMPDIR=str(runtime),
        TRITON_CACHE_DIR=str(RUN / "cache/triton"), TORCHINDUCTOR_CACHE_DIR=str(RUN / "cache/torchinductor"),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CUDA_VISIBLE_DEVICES="")
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    save(RUN / "runtime.json", {k: env[k] for k in ("TMPDIR", "IMAGE_MAX_TOKEN_NUM", "HF_HOME", "HF_DATASETS_CACHE", "TRITON_CACHE_DIR")})
    return env


def command(name, args, env):
    save(RUN / (name + ".command.json"), args)
    status(name)
    start = time.monotonic()
    with (RUN / (name + ".log")).open("x") as log:
        child = subprocess.Popen(args, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT)
        save(RUN / (name + ".process.json"), {"pid": child.pid, "parent_pid": os.getpid(), "command": args})
        result = child.wait()  # Deliberately no wall-clock timeout.
    save(RUN / (name + ".result.json"), {"exit_code": result, "seconds": time.monotonic() - start})
    if result:
        raise RuntimeError(f"{name} exited {result}; inspect {name}.log")


def validate_dataset_split(train, evaluation, binding):
    """Recheck geometry, image bytes, policy bindings, and scene-disjoint admission."""
    if __package__:
        from .map_prism_native import validate_native_gripper
    else:
        from map_prism_native import validate_native_gripper
    if binding.get("contract") != CONTRACT or not binding.get("policy_bindings"):
        raise ValueError("expected a bound PRISM source-native SFT manifest")
    if not train or not evaluation:
        raise ValueError("both training and held-out records are required")
    cache, seen_ids, scenes, hashes = {}, set(), [set(), set()], [set(), set()]
    for index, rows in enumerate((train, evaluation)):
        expected_train = index == 0
        for row in rows:
            meta = row.get("anvil_meta", {})
            provenance = meta.get("source_provenance", {})
            scene = provenance.get("scene_id")
            if (not scene or meta.get("group_id") != "PRISM:" + scene
                    or meta.get("native_contract") != CONTRACT
                    or meta.get("schema") != "anvil-source-native-v1"
                    or meta.get("native_form") != "dof"
                    or meta.get("legacy_v3_admitted") is not False
                    or meta.get("training_approved") is not expected_train
                    or meta.get("policy_bindings") != binding["policy_bindings"]
                    or (meta.get("split") != "train" if expected_train
                        else meta.get("split") not in {"validation", "test"})):
                raise ValueError("dataset scene/contract/approval binding mismatch")
            identity = row.get("id")
            if not identity or identity in seen_ids:
                raise ValueError("missing or duplicate training/evaluation record ID")
            seen_ids.add(identity)
            images = row.get("images", [])
            if len(images) != 1:
                raise ValueError("exactly one source image is required")
            image = Path(images[0]).resolve(strict=True)
            if image not in cache:
                cache[image] = sha(image)
            if cache[image] != provenance.get("image_sha256"):
                raise ValueError("image bytes differ from source provenance")
            messages = row.get("messages", [])
            if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
                raise ValueError("expected one image prompt and one source target")
            if not messages[0]["content"].startswith("<image>\n"):
                raise ValueError("missing image placeholder")
            prompt = json.loads(messages[0]["content"].split("\n", 1)[1])
            answer = json.loads(messages[1]["content"])
            inp = prompt["input"]
            entity = inp["entity"]
            if (prompt.get("contract") != CONTRACT or answer["entity_id"] != entity["entity_id"]
                    or not answer["entity_id"].startswith("prism:" + scene + ":")):
                raise ValueError("prompt/target scene or contract mismatch")
            validate_native_gripper(answer, inp["intrinsics"], *inp["image_size_wh"])
            scenes[index].add(meta["group_id"])
            hashes[index].add(cache[image])
    if scenes[0] & scenes[1] or hashes[0] & hashes[1]:
        raise ValueError("training and held-out scenes/images must be disjoint")
    return {"train_records": len(train), "heldout_records": len(evaluation),
            "train_scenes": len(scenes[0]), "heldout_scenes": len(scenes[1]),
            "images_checked": len(cache)}


def prepare():
    status("prepare_cpu")
    binding = json.loads(DATA_MANIFEST.read_text())
    dataset = RUN / "data"
    dataset.mkdir()
    for name, source in (("train.jsonl", TRAIN), ("evaluation.jsonl", EVALUATION)):
        if sha(source) != binding["files"][name]:
            raise ValueError("input dataset hash mismatch")
        shutil.copyfile(source, dataset / name)
        assert sha(dataset / name) == binding["files"][name]
    train = [json.loads(line) for line in (dataset / "train.jsonl").open()]
    evaluation = [json.loads(line) for line in (dataset / "evaluation.jsonl").open()]
    validation = validate_dataset_split(train, evaluation, binding)
    save(RUN / "dataset_preflight.json", validation)
    # The smoke dataset contains unchanged lines selected only from training.
    with (dataset / "smoke_train.jsonl").open("x") as out, (dataset / "train.jsonl").open() as inp:
        for _, line in zip(range(8), inp):
            out.write(line)
    filenames = ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                 "chat_template.jinja", "model.safetensors.index.json"]
    shards = sorted(set(json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"].values()))
    model_hashes = {}
    for name in filenames + shards:
        model_hashes[name] = sha(BASE / name)
        print(json.dumps({"base_file_hashed": name}), flush=True)
    save(RUN / "input_binding.json", {"dataset_files": binding["files"], "source_manifest_sha256": sha(DATA_MANIFEST),
        "base_model": str(BASE), "base_files_sha256": model_hashes,
        "train_count": len(train), "heldout_count": len(evaluation), "training_contract": CONTRACT,
        "source_policy_bindings": binding["policy_bindings"],
        "training_role": "fresh-base source-native DoF SE3 specialist; not final Anvil MOPD or legacy categorical DoF",
        "code_sha256": sha(RUN / "code/train_prism_specialist.py"),
        "mapper_sha256": sha(RUN / "code/map_prism_native.py"),
        "response_text_sha256": sha(RUN / "code/response_text.py"),
        "packages": {name: importlib.metadata.version(name) for name in ("ms-swift", "torch", "transformers", "peft", "datasets", "accelerate")}})
    save(RUN / "protocol.json", {"frozen_before_gpu_training": True, "seed": 17, "steps": 512,
        "effective_batch": 4, "learning_rate": 1e-4, "lora_rank": 64, "lora_alpha": 128,
        "lora_dropout": .05, "freeze_vit": True, "freeze_aligner": True, "max_length": 3072,
        "image_max_tokens": 1536, "schedule": "cosine", "warmup_ratio": .03,
        "checkpoints": [128, 256, 384, 512], "selected_checkpoint": 512,
        "heldout_used_for_tuning": False, "smoke": "separate 2-step adapter discarded; main starts again from base",
        "evaluation": {"decoding": "greedy", "max_new_tokens": 2048, "batch_size": 4,
            "enable_thinking": False, "max_context_tokens": 32768,
            "response_text_policy": RESPONSE_POLICY,
            "template_policy": "explicit constructor; identical base and specialist inference template",
            "json_valid_rate": "whole-response JSON object; no repaired extraction; denominator all held-out records",
            "proper_rotation_rate": "finite 3x3 R, max|R^T R-I|<=0.01 and |det(R)-1|<=0.01; denominator all held-out records",
            "projected_contact_error": "sqrt((du/width)^2+(dv/height)^2), using K@predicted_contact_xyz vs source contact_px; positive predicted Z required",
            "rotation_geodesic_degrees": "acos(clamp((trace(Rpred^T Rref)-1)/2))*180/pi; near-proper Rpred projected to SO3 by SVD",
            "rotation_symmetry_diagnostic": "minimum geodesic over reference and reference@diag(-1,-1,1), parallel-jaw pi symmetry",
            "invalid_outputs": "report separate counts/rates; no fabricated finite distance for invalid geometry",
            "scope": "distance to one source demonstration, NOT semantic action success or failure; no metric-meter or physics claim"}})


def cpu_encode():
    import torch
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(BASE), load_model=False, model_type="qwen3_5", torch_dtype=torch.bfloat16, use_hf=True)
    template = get_template(processor, template_type="qwen3_5", max_length=3072, truncation_strategy="raise", enable_thinking=False)
    template.set_mode("train")
    lengths, labels = [], []
    for index, line in enumerate((RUN / "data/train.jsonl").open(), 1):
        row = json.loads(line)
        encoded = template.encode({"messages": copy.deepcopy(row["messages"]), "images": row["images"]})
        lengths.append(len(encoded["input_ids"]))
        labels.append(sum(value != -100 for value in encoded["labels"]))
        if labels[-1] <= 0:
            raise ValueError("reference tokens absent after encoding")
        if index % 200 == 0:
            print(json.dumps({"encoded": index, "max_length": max(lengths)}), flush=True)
    save(RUN / "encoding_preflight.json", {"records": len(lengths), "min_length": min(lengths),
        "max_length": max(lengths), "min_supervised_tokens": min(labels), "max_supervised_tokens": max(labels),
        "dropped_or_truncated_records": 0, "model_weights_loaded": False})


def training_command(smoke=False):
    return [sys.executable, "-B", "-m", "swift.cli.sft", "--model", str(BASE),
        "--model_type", "qwen3_5", "--template", "qwen3_5", "--dataset", str(RUN / "data" / ("smoke_train.jsonl" if smoke else "train.jsonl")),
        "--split_dataset_ratio", "0", "--tuner_type", "lora", "--torch_dtype", "bfloat16",
        "--enable_thinking", "false", "--attn_impl", "sdpa", "--freeze_vit", "true", "--freeze_aligner", "true",
        "--lora_rank", "64", "--lora_alpha", "128", "--lora_dropout", ".05", "--target_modules", "all-linear",
        "--max_steps", "2" if smoke else "512", "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "4",
        "--seed", "17", "--data_seed", "17", "--learning_rate", "1e-4", "--warmup_ratio", ".03", "--lr_scheduler_type", "cosine",
        "--max_length", "3072", "--gradient_checkpointing", "true", "--logging_steps", "1" if smoke else "8",
        "--save_steps", "2" if smoke else "128", "--save_total_limit", "1" if smoke else "4",
        "--output_dir", str(RUN / ("smoke_training" if smoke else "training")), "--add_version", "false",
        "--eval_strategy", "no", "--dataloader_num_workers", "2", "--dataset_num_proc", "2",
        "--strict", "true", "--report_to", "none"]


def gpu_idle_snapshot(gpu_uuid):
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader"], text=True)
    busy = {row[0].strip() for row in csv.reader(active.splitlines()) if row}
    raw = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid,
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    uuid, memory, utilization = map(str.strip, raw.split(","))
    if uuid != gpu_uuid:
        raise ValueError("physical GPU identity changed")
    return {"gpu_uuid": uuid, "memory_mib": int(memory), "utilization_percent": int(utilization),
            "compute_pid_present": uuid in busy, "sampled_at": time.time()}


def is_gpu_idle(snapshot):
    # Persistence-mode driver allocations can remain after the final process exits.
    return (not snapshot["compute_pid_present"] and snapshot["memory_mib"] <= 1024
            and snapshot["utilization_percent"] == 0)


def acquire_gpu():
    ARGS.lock_root.mkdir(parents=True, exist_ok=True)
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", ARGS.gpu,
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
        raise ValueError("--gpu must identify exactly one physical GPU")
    lock = (ARGS.lock_root / (gpu_uuid + ".lock")).open("a")
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            status("waiting_for_gpu_lock", gpu_uuid=gpu_uuid)
            time.sleep(20)
            continue
        snapshots = [gpu_idle_snapshot(gpu_uuid)]
        if is_gpu_idle(snapshots[0]):
            time.sleep(2)
            snapshots.append(gpu_idle_snapshot(gpu_uuid))
            if is_gpu_idle(snapshots[1]):
                save(RUN / "gpu_allocation.json", {"requested": ARGS.gpu, "uuid": gpu_uuid,
                    "at": time.time(), "idle_checks_while_locked": snapshots})
                return lock, gpu_uuid
        fcntl.flock(lock, fcntl.LOCK_UN)
        status("waiting_for_gpu", **snapshots[-1])
        time.sleep(20)

def metrics(text, source):
    import numpy as np
    result = {"json_valid": False, "proper_rotation": False, "projected_contact_error": None,
              "rotation_geodesic_degrees": None, "rotation_pi_symmetric_degrees": None}
    try:
        def reject_nonfinite(token):
            raise ValueError("non-JSON numeric token: " + token)
        answer = json.loads(text, parse_constant=reject_nonfinite)
        if not isinstance(answer, dict):
            return result
        result["json_valid"] = True
        target = json.loads(source["messages"][1]["content"])
        inp = json.loads(source["messages"][0]["content"].split("\n", 1)[1])["input"]
        point = np.asarray(answer.get("contact_point_camera"), dtype=float)
        if point.shape == (3,) and np.isfinite(point).all() and point[2] > 0:
            p = np.asarray(inp["intrinsics"]) @ point
            delta = (p[:2] / p[2] - np.asarray(target["contact_point_px"])) / np.asarray(inp["image_size_wh"])
            result["projected_contact_error"] = float(np.linalg.norm(delta))
        pose = np.asarray(answer.get("camera_from_gripper"), dtype=float)
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            rotation = pose[:3, :3]
            proper = np.max(np.abs(rotation.T @ rotation - np.eye(3))) <= .01 and abs(np.linalg.det(rotation) - 1) <= .01
            result["proper_rotation"] = bool(proper)
            if proper:
                u, _, vt = np.linalg.svd(rotation)
                rotation = u @ vt
                reference = np.asarray(target["camera_from_gripper"])[:3, :3]
                distance = lambda other: float(np.degrees(np.arccos(np.clip((np.trace(rotation.T @ other) - 1) / 2, -1, 1))))
                canonical = distance(reference)
                result["rotation_geodesic_degrees"] = canonical
                result["rotation_pi_symmetric_degrees"] = min(canonical, distance(reference @ np.diag([-1, -1, 1])))
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        pass
    return result


def metric_selftest():
    import numpy as np
    target = {"camera_from_gripper": np.eye(4).tolist(), "contact_point_camera": [.1, .2, 1.], "contact_point_px": [10., 20.]}
    source = {"messages": [{"content": '<image>\n' + json.dumps({"input": {"intrinsics": [[100., 0., 0.], [0., 100., 0.], [0., 0., 1.]], "image_size_wh": [100, 100]}})}, {"content": json.dumps(target)}]}
    assert metrics(json.dumps(target), source)["rotation_geodesic_degrees"] == 0
    altered = copy.deepcopy(target)
    altered["camera_from_gripper"] = [[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    altered["contact_point_camera"][0] += .1
    score = metrics(json.dumps(altered), source)
    assert abs(score["rotation_geodesic_degrees"] - 90.) < 1e-6 and abs(score["projected_contact_error"] - .1) < 1e-6
    assert metrics("not JSON", source)["json_valid"] is False
    altered["camera_from_gripper"][2][2] = -1
    assert metrics(json.dumps(altered), source)["proper_rotation"] is False


def inference_template(processor):
    from swift.template import get_template
    # Changing enable_thinking after construction does not update response_prefix.
    return get_template(processor, template_type="qwen3_5", max_length=32768,
                        truncation_strategy="raise", enable_thinking=False)


def evaluate(arm):
    import numpy as np
    import torch
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    checkpoint = RUN / "training/checkpoint-512"
    adapters = [] if arm == "base" else [str(checkpoint)]
    engine = TransformersEngine(str(BASE), adapters=adapters, model_type="qwen3_5", template_type="qwen3_5",
        torch_dtype=torch.bfloat16, device_map="cuda:0", attn_impl="sdpa", max_batch_size=4, use_hf=True)
    engine.template = inference_template(engine.template.processor)
    if engine.template.response_prefix != SWIFT_NON_THINKING_PREFIX:
        raise ValueError('SDK template scaffold differs from protocol')
    engine.strict = True
    engine.model.eval()
    config = RequestConfig(max_tokens=2048, temperature=0., seed=17)
    records = [json.loads(line) for line in (RUN / "data/evaluation.jsonl").open()]
    scored = []
    with (RUN / (arm + "_predictions.jsonl")).open("x") as out:
        for start in range(0, len(records), 4):
            batch = records[start:start + 4]
            requests = [InferRequest(messages=copy.deepcopy(row["messages"][:1]), images=row["images"]) for row in batch]
            responses = engine.infer(requests, config, use_tqdm=False)
            if len(responses) != len(batch):
                raise RuntimeError("inference returned fewer predictions than held-out requests")
            for row, response in zip(batch, responses):
                choice = response.choices[0]
                sdk_response = choice.message.content or ""
                text = generated_text(sdk_response, engine.template.response_prefix)
                score = metrics(text, row)
                item = {"id": row["id"], "group_id": row["anvil_meta"]["group_id"], "response": text,
                    "sdk_response": sdk_response, "sdk_response_prefix": engine.template.response_prefix,
                    "finish_reason": choice.finish_reason, "usage": vars(response.usage) if response.usage else None, **score}
                out.write(json.dumps(item, allow_nan=False) + "\n")
                scored.append(item)
            out.flush()
            if len(scored) % 20 == 0:
                print(json.dumps({"arm": arm, "completed": len(scored), "total": len(records)}), flush=True)
    result = {"arm": arm, "n": len(scored), "contract": CONTRACT, "model_role": "source-native DoF specialist" if adapters else "fresh base",
              "json_valid_rate": sum(r["json_valid"] for r in scored) / len(scored),
              "proper_rotation_rate": sum(r["proper_rotation"] for r in scored) / len(scored),
              "truncated": sum(r["finish_reason"] == "length" for r in scored),
              "dataset_sha256": sha(RUN / "data/evaluation.jsonl"), "protocol_sha256": sha(RUN / "protocol.json"),
              "adapter_sha256": sha(checkpoint / "adapter_model.safetensors") if adapters else None}
    for key in ("projected_contact_error", "rotation_geodesic_degrees", "rotation_pi_symmetric_degrees"):
        values = [r[key] for r in scored if r[key] is not None]
        result[key] = {"valid_n": len(values), "mean": float(np.mean(values)) if values else None,
                       "median": float(np.median(values)) if values else None}
    save(RUN / (arm + "_metrics.json"), result)


def child_command(stage):
    return [sys.executable, "-B", str(RUN / "code/train_prism_specialist.py"),
        "--model", str(BASE), "--train", str(TRAIN), "--evaluation", str(EVALUATION),
        "--data-manifest", str(DATA_MANIFEST), "--output", str(RUN), "--gpu", ARGS.gpu,
        "--lock-root", str(ARGS.lock_root), "--runtime-parent", str(ARGS.runtime_parent),
        "--stage", stage]


def run():
    global RUN_CREATED
    if RUN.exists():
        raise FileExistsError("output already exists; inspect live handles/checkpoints before any resume")
    RUN.mkdir(parents=True)
    RUN_CREATED = True
    code_directory = RUN / "code"
    code_directory.mkdir()
    for source in (Path(__file__).resolve(), Path(__file__).with_name("map_prism_native.py").resolve(),
                   Path(__file__).with_name("response_text.py").resolve()):
        destination = code_directory / source.name
        shutil.copy2(source, destination)
        if sha(source) != sha(destination):
            raise RuntimeError("experiment code snapshot differs from entry point")
    save(RUN / "code_snapshot.json", {path.name: sha(path) for path in sorted(code_directory.glob("*.py"))})
    sys.path.insert(0, str(code_directory))
    save(RUN / "invocation.json", {"argv": sys.argv, "cpu_preflight_only": ARGS.cpu_preflight_only})
    env = environment()
    os.environ.update(env)
    metric_selftest()
    prepare()
    command("cpu_encode", child_command("cpu-encode"), env)
    if ARGS.cpu_preflight_only:
        status("cpu_preflight_complete", gpu_training_executed=False)
        return
    lock, gpu_uuid = acquire_gpu()
    try:
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        command("gpu_smoke", training_command(smoke=True), env)
        command("training", training_command(), env)
        checkpoint = RUN / "training/checkpoint-512"
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        assert state["global_step"] == 512
        save(RUN / "trained_adapter.json", {"checkpoint": str(checkpoint), "step": 512,
            "sha256": sha(checkpoint / "adapter_model.safetensors"), "selected_without_heldout": True})
        # Evaluate the requested specialist first; then its fixed fresh-base comparator.
        command("specialist_eval", child_command("evaluate-specialist"), env)
        command("base_eval", child_command("evaluate-base"), env)
        status("complete", adapter=str(checkpoint))
    finally:
        lock.close()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local Qwen3.5-9B-compatible base directory")
    parser.add_argument("--train", required=True, type=Path, help="Byte-bound PRISM train.jsonl export")
    parser.add_argument("--evaluation", required=True, type=Path, help="Scene-disjoint evaluation.jsonl export")
    parser.add_argument("--data-manifest", type=Path, help="Defaults to manifest.json beside --train")
    parser.add_argument("--output", required=True, type=Path, help="New run directory; never overwritten")
    parser.add_argument("--gpu", required=True, help="Physical GPU index or UUID; wait until idle, never evict")
    parser.add_argument("--lock-root", required=True, type=Path, help="Shared cooperative GPU lock directory")
    parser.add_argument("--runtime-parent", required=True, type=Path, help="Filesystem for temporary sockets/caches")
    parser.add_argument("--cpu-preflight-only", action="store_true", help="Hash inputs and encode every train row; no GPU stages")
    parser.add_argument("--stage", choices=["run", "cpu-encode", "evaluate-specialist", "evaluate-base"],
                        default="run", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def configure(args):
    global ARGS, RUN, BASE, TRAIN, EVALUATION, DATA_MANIFEST
    ARGS = args
    BASE, TRAIN, EVALUATION = (p.resolve(strict=True) for p in (args.model, args.train, args.evaluation))
    DATA_MANIFEST = (args.data_manifest or TRAIN.parent / "manifest.json").resolve(strict=True)
    RUN = args.output.resolve()
    args.lock_root, args.runtime_parent = args.lock_root.resolve(), args.runtime_parent.resolve()


if __name__ == "__main__":
    try:
        configure(parse_arguments())
        if ARGS.stage == "cpu-encode":
            cpu_encode()
        elif ARGS.stage.startswith("evaluate-"):
            evaluate(ARGS.stage.split("-", 1)[1])
        else:
            run()
    except BaseException as exc:
        if ARGS is not None and ARGS.stage == "run" and RUN_CREATED:
            status("failed", error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, SystemExit):
            traceback.print_exc()
        raise
