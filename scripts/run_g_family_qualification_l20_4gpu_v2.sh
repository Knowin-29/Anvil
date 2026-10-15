#!/usr/bin/env bash
# Matched 4xL20 variant of the schema-v3 G-only v2 qualification pilot.
# It preserves the 8-GPU run's effective batch, optimizer steps, and data exposure.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || { echo "FATAL: required environment file is missing: $W/.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$W/.env"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_ENV="/path/to/environment/env/gsa_train"
PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
[[ "$PYTHON_PREFIX" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: .env did not activate the required training environment" >&2
  echo "  expected sys.prefix: $EXPECTED_ENV" >&2
  echo "  actual sys.prefix  : $PYTHON_PREFIX" >&2
  exit 2
}
SWIFT_BIN=$(command -v swift || true)
[[ "$SWIFT_BIN" == "$EXPECTED_ENV/bin/swift" ]] || {
  echo "FATAL: swift does not resolve inside the required environment: ${SWIFT_BIN:-MISSING}" >&2
  exit 2
}

SOURCE_CORPUS="${G_V1_CORPUS:-$W/trains/gsa_bench_v3_g_family_scoped_v1}"
CORPUS="${G_V2_CORPUS:-$W/trains/gsa_bench_v3_g_family_scoped_v2}"
MANIFEST="$CORPUS/manifest.json"
TRAIN="$CORPUS/train_G_only.jsonl"
DEV="$CORPUS/dev_G_only.jsonl"
QUALIFICATION="$CORPUS/qualification_G_only.jsonl"
SCOPED_REPORT="${G_SCOPED_REPORT:-$W/artifacts/audits/pilot_semantic_audit_v3_g_family_scoped_v1.json}"
FULL_REPORT="${V3_FULL_REPORT:-$W/artifacts/audits/pilot_semantic_audit_v3_adjudicated.json}"
MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"

SEED="${SEED:-42}"
case "$SEED" in
  42|43|44) ;;
  *) echo "FATAL: frozen 4xL20 qualification seed must be 42, 43, or 44" >&2; exit 2 ;;
esac
OUT="${OUT:-$W/output/g_family_v3_scoped_v2_qualification_lora_4gpu_seed${SEED}}"

"$PYTHON_BIN" "$W/scripts/verify_g_family_scoped_v2.py" \
  --manifest "$MANIFEST" --source "$SOURCE_CORPUS" \
  --scoped-report "$SCOPED_REPORT" --full-report "$FULL_REPORT"
[[ -s "$TRAIN" && -s "$DEV" && -s "$QUALIFICATION" ]] || {
  echo "FATAL: v2 train/dev/qualification files are missing or empty" >&2; exit 2;
}
[[ -f "$MODEL/config.json" ]] || { echo "FATAL: model config is missing: $MODEL/config.json" >&2; exit 2; }

TRAIN_ROWS=$(wc -l < "$TRAIN")
DEV_ROWS=$(wc -l < "$DEV")
QUALIFICATION_ROWS=$(wc -l < "$QUALIFICATION")
[[ "$TRAIN_ROWS" == 84 && "$DEV_ROWS" == 32 && "$QUALIFICATION_ROWS" == 32 ]] || {
  echo "FATAL: expected v2 rows train/dev/qualification=84/32/32" >&2; exit 2;
}

WORLD_SIZE=4
BS="${BS:-1}"
GA="${GA:-2}"
EPOCHS="${EPOCHS:-5}"
HOURS="${HOURS:-6}"
LR="${LR:-1e-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
[[ "$BS" == 1 && "$GA" == 2 && "$EPOCHS" == 5 ]] || {
  echo "FATAL: matched 4xL20 protocol requires BS=1, GA=2, EPOCHS=5" >&2; exit 2;
}
[[ "$DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ && "$DATASET_NUM_PROC" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: worker counts must be valid non-negative/positive integers" >&2; exit 2;
}
[[ "$LR" == "1e-5" ]] || { echo "FATAL: matched qualification requires LR=1e-5" >&2; exit 2; }
GLOBAL_BATCH=$(( WORLD_SIZE * BS * GA ))
[[ "$GLOBAL_BATCH" == 8 ]] || { echo "FATAL: effective batch must remain 8" >&2; exit 2; }
COMPUTED_STEPS=$(( (TRAIN_ROWS * EPOCHS + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
STEPS="${STEPS:-$COMPUTED_STEPS}"
[[ "$STEPS" == "$COMPUTED_STEPS" && "$STEPS" == 53 ]] || {
  echo "FATAL: matched 4xL20 protocol requires exactly 53 optimizer steps" >&2; exit 2;
}
EVAL_STEPS="${EVAL_STEPS:-$STEPS}"
SAVE_STEPS="${SAVE_STEPS:-$STEPS}"
[[ "$EVAL_STEPS" == "$STEPS" && "$SAVE_STEPS" == "$STEPS" ]] || {
  echo "FATAL: matched qualification evaluates and saves only at step 53" >&2; exit 2;
}

echo "=================================================="
echo " protocol      : semantic-audit-v3-g-family-scoped-v2-independent-qualification"
echo " scope         : G specialist qualification pilot only"
echo " train/dev     : 84/32 rows on 24/8 images"
echo " qualification : 32 rows on 8 images (never exposed to trainer)"
echo " plan          : 4xL20 BS=$BS GA=$GA epochs=$EPOCHS steps=$STEPS"
echo " effective     : $GLOBAL_BATCH (matched to the frozen 8-GPU pilot)"
echo " seed          : $SEED (also used as data_seed)"
echo " workers       : dataloader=$DATALOADER_NUM_WORKERS dataset_proc=$DATASET_NUM_PROC"
echo " tracking      : WANDB_MODE=offline"
echo " global        : BLOCKED; joint/MOPD are not authorized"
echo " output        : $OUT"
echo "=================================================="

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "G_V2_4GPU_TRAIN_PREFLIGHT_READY: no GPU process was started"
  exit 0
fi

mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF' | sort -u
)
if [[ "${#GPU_PIDS[@]}" -gt 0 && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "FATAL: visible GPUs already have compute processes: ${GPU_PIDS[*]}" >&2; exit 3;
fi
"$PYTHON_BIN" - <<'PY'
import torch
count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"FATAL: local G qualification requires exactly 4 GPUs, found {count}")
for index in range(count):
    name = torch.cuda.get_device_name(index)
    gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    if "L20" not in name.upper() or gib < 44:
        raise SystemExit(f"FATAL: GPU {index} is not a >=44 GiB L20: {name} ({gib:.1f} GiB)")
print("4xL20 hardware gate: READY")
PY

mkdir -p "$OUT"
RECEIPT="$OUT/g_family_qualification_inputs_v2.json"
"$PYTHON_BIN" - "$MANIFEST" "$SCOPED_REPORT" "$FULL_REPORT" "$MODEL/config.json" \
  "$TRAIN" "$DEV" "$QUALIFICATION" "$RECEIPT" "$STEPS" "$BS" "$GA" "$EPOCHS" \
  "$SEED" "$DATALOADER_NUM_WORKERS" "$DATASET_NUM_PROC" "$LR" <<'PY'
import hashlib, json, os, pathlib, sys
manifest, scoped, full, model, train, dev, qualification, receipt = map(pathlib.Path, sys.argv[1:9])
steps, bs, ga, epochs, seed, dataloader_workers, dataset_proc = map(int, sys.argv[9:16])
learning_rate = sys.argv[16]
def bind(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
payload = {
    "protocol": "semantic-audit-v3-g-family-scoped-v2-independent-qualification",
    "source_audit_protocol": "semantic-audit-v3-g-family-scoped-v1",
    "authorization": "G_specialist_qualification_only",
    "global_audit_verdict": "BLOCKED",
    "manifest": bind(manifest), "scoped_report": bind(scoped), "full_report": bind(full),
    "model_config": bind(model), "train": bind(train), "dev": bind(dev),
    "qualification": bind(qualification),
    "train_rows": 84, "dev_rows": 32, "qualification_rows": 32,
    "train_images": 24, "dev_images": 8, "qualification_images": 8,
    "qualification_exposed_to_trainer": False,
    "trainer_dataset_paths": [str(train.resolve()), str(dev.resolve())],
    "hardware": "4xL20", "world_size": 4, "per_device_batch_size": bs,
    "gradient_accumulation_steps": ga, "effective_batch_size": 4 * bs * ga,
    "epochs_for_step_calculation": epochs, "max_steps": steps,
    "seed": seed, "data_seed": seed, "deepspeed": "zero2",
    "learning_rate": learning_rate, "max_length": 3072,
    "lora": {"rank": 64, "alpha": 128, "dropout": 0.05},
    "torch_dtype": "bfloat16", "gradient_checkpointing": True,
    "dataloader_num_workers": dataloader_workers, "dataset_num_proc": dataset_proc,
    "wandb_mode": "offline",
    "teacher_authorization": "NONE", "joint_training_authorized": False,
    "mopd_training_authorized": False,
}
if receipt.exists():
    if json.loads(receipt.read_text()) != payload:
        raise SystemExit("FATAL: output directory is bound to different v2 inputs/topology/seed")
else:
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, receipt)
print(json.dumps(payload, indent=2))
PY

# Qualification is deliberately absent: only train/dev can affect gradients or selection.
export WANDB_MODE=offline
TRAIN="$TRAIN" VAL="$DEV" MODEL="$MODEL" OUT="$OUT" \
  LR="$LR" STEPS="$STEPS" BS="$BS" GA="$GA" SEED="$SEED" \
  EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" HOURS="$HOURS" \
  DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" DATASET_NUM_PROC="$DATASET_NUM_PROC" \
  bash "$W/scripts/run_sft_9b_lora_l20_dlc.sh"
