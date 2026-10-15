#!/usr/bin/env bash
# run_sweep_train_l20_3gpu.sh — one configuration of the corpus-P4 training sweep.
#
#   CFG=<name> TRAIN=<jsonl> VAL=<jsonl> LR=<lr> RANK=<r> ALPHA=<a> EPOCHS=<n> \
#     bash scripts/run_sweep_train_l20_3gpu.sh
#
# 3xL20 (GPU 5/6/7). Effective batch is pinned to 24 across every configuration so a
# score difference cannot come from a different optimisation trajectory length. Step
# count is derived from rows x epochs, never hand-set, so "1 epoch" means one epoch.
#
# LoRA only. E2's full-parameter SFT OOMs on an L20 at the first batch; there is no
# variant of this script that tries it.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing"; exit 1; }
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DIR="$W/logs"
# PCIe-only node, no InfiniBand.
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=PIX
# This container no longer ships the CUDA toolkit, so deepspeed's `installed_cuda_version`
# raises MissingCUDAException while merely *probing* op compatibility at import time and
# every rank dies before training starts. This is upstream's own escape hatch for the
# case where torch already carries its CUDA runtime; ZeRO-2 needs no JIT-compiled op
# here, and fused_adam/cpu_adam report prebuilt. The August runs predate the toolkit
# being dropped from the image, which is why their identical command line worked.
export DS_IGNORE_CUDA_DETECTION=1
# The node runs at loadavg 130-240 from unrelated inference. Shells died at 126-158
# last time, so every worker count here is deliberately small.
export OMP_NUM_THREADS=4

CFG="${CFG:?set CFG}"
TRAIN="${TRAIN:?set TRAIN}"
VAL="${VAL:?set VAL}"
LR="${LR:-1e-4}"
RANK="${RANK:-64}"
ALPHA="${ALPHA:-128}"
EPOCHS="${EPOCHS:-1}"
BS="${BS:-1}"
MAXLEN="${MAXLEN:-2048}"
SEED="${SEED:-42}"
HOURS="${HOURS:-6}"
# Picked from measured free memory at launch, not hard-coded: the first smoke run
# reserved 5,6,7 up front, lost 6 and 7 to unrelated jobs during the ~2 min model load
# and OOMed every rank. GPUs 0-3 stay excluded (General Retention evaluation).
NEED_MIB="${NEED_MIB:-34000}"
MAX_GPUS="${MAX_GPUS:-3}"
GPUS="${GPUS:-$(python "$W/tools/pick_free_gpus.py" --need-mib "$NEED_MIB" \
  --max-gpus "$MAX_GPUS" --min-gpus 1 --exclude 0,1,2,3 --verbose)}"
[[ -n "$GPUS" ]] || { echo "FATAL: no free GPU with >=${NEED_MIB} MiB" >&2; exit 4; }
MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
MODEL_ARGS=()
[[ -z "${MODEL_TYPE:-}" ]] || MODEL_ARGS=(--model_type "$MODEL_TYPE")
OUT="${OUT:-/path/to/Anvil/output/sweep_v1/$CFG}"

[[ -s "$TRAIN" ]] || { echo "FATAL: TRAIN missing/empty: $TRAIN" >&2; exit 2; }
[[ -s "$VAL" ]]   || { echo "FATAL: VAL missing/empty: $VAL" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "FATAL: model missing: $MODEL" >&2; exit 2; }

REQUESTED_GPUS="$GPUS"
# This DeepSpeed release parses CUDA_VISIBLE_DEVICES as integer NVML indices.
# Resolve only the explicitly requested UUIDs; never substitute a different GPU.
if [[ "$GPUS" == GPU-* ]]; then
 GPUS=$(env -u CUDA_VISIBLE_DEVICES python - "$GPUS" <<'PY'
import subprocess,sys,torch
requested=sys.argv[1].split(',')
visible={'GPU-'+str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-'):i
         for i in range(torch.cuda.device_count())}
nvml=dict(line.strip().split(', ') for line in subprocess.check_output(
    ['nvidia-smi','--query-gpu=uuid,index','--format=csv,noheader'],text=True).splitlines())
indices=[visible[u] for u in requested]
assert all(int(nvml[u])==i for u,i in zip(requested,indices)), 'CUDA/NVML ordering mismatch'
print(','.join(map(str,indices)))
PY
 ) || exit 4
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs for CUDA_VISIBLE_DEVICES=$GPUS" >&2; exit 1; }

EFF=24
GA=$(( EFF / (NG * BS) ))
[[ "$GA" -ge 1 ]] || GA=1
ROWS=$(wc -l < "$TRAIN")
STEPS=$(python -c "import math;print(max(1,math.ceil($ROWS*$EPOCHS/($NG*$BS*$GA))))")
# Save/eval only at the end: a mid-run checkpoint of a 250-step run is not a
# configuration anyone would ship, and each extra save costs ~1 GiB and stalls the
# vePFS mount that the concurrent evaluation jobs also read from.
EVAL_STEPS="${EVAL_STEPS:-$STEPS}"
SAVE_STEPS="${SAVE_STEPS:-$STEPS}"

mkdir -p "$OUT" "$W/logs/sweep"
LOG="$W/logs/sweep/train_${CFG}_$(date -u +%Y%m%d_%H%M%S).log"

# Written before the trainer starts so a dead shell still leaves the configuration on
# disk; the progress table is rebuilt from these files, not from memory.
python - "$OUT/sweep_config.json" <<PY
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
train = pathlib.Path("$TRAIN")
payload = {
    "cfg": "$CFG", "model": "$MODEL",
    "train": {"path": str(train), "rows": $ROWS,
              "sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
    "val": "$VAL",
    "lr": "$LR", "lora_rank": $RANK, "lora_alpha": $ALPHA, "lora_dropout": 0.05,
    "epochs": $EPOCHS, "max_steps": $STEPS, "world_size": $NG,
    "per_device_batch_size": $BS, "gradient_accumulation_steps": $GA,
    "effective_batch_size": $NG * $BS * $GA,
    "max_length": $MAXLEN, "seed": $SEED, "data_seed": $SEED,
    "deepspeed": "zero2", "torch_dtype": "bfloat16",
    "gradient_checkpointing": True, "tuner_type": "lora",
    "gpus": "$GPUS", "requested_gpu_uuids": "$REQUESTED_GPUS", "output_dir": "$OUT", "log": "$LOG",
}
p.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

RESUME=()
LAST=$(ls -d "$OUT"/checkpoint-* "$OUT"/v*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[[ -n "$LAST" ]] && RESUME=(--resume_from_checkpoint "$LAST")

echo "=================================================="
echo " sweep cfg : $CFG"
echo " train     : $(basename "$TRAIN") ($ROWS rows)  epochs=$EPOCHS"
echo " gpus      : $GPUS (n=$NG)  bs=$BS ga=$GA eff=$(( NG * BS * GA ))  steps=$STEPS"
echo " lora      : r=$RANK a=$ALPHA   lr=$LR   max_len=$MAXLEN"
echo " out       : $OUT"
echo " log       : $LOG"
echo "=================================================="

NPROC_PER_NODE="$NG" timeout "${HOURS}h" swift sft \
  --model "$MODEL" \
  "${MODEL_ARGS[@]}" \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  --tuner_type lora --lora_rank "$RANK" --lora_alpha "$ALPHA" --lora_dropout 0.05 \
  --torch_dtype bfloat16 --deepspeed zero2 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --seed "$SEED" --data_seed "$SEED" \
  --learning_rate "$LR" --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_length "$MAXLEN" --gradient_checkpointing true \
  --logging_steps 10 --eval_steps "$EVAL_STEPS" --save_steps "$SAVE_STEPS" \
  --save_total_limit 1 \
  --output_dir "$OUT" --dataloader_num_workers 2 --dataset_num_proc 4 \
  --report_to none "${RESUME[@]}" \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
python - "$OUT/train_status.json" "$RC" "$LOG" <<'PY'
import json, pathlib, re, sys
out, rc, log = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
text = log.read_text(errors="replace") if log.is_file() else ""
losses = re.findall(r"'loss': ([0-9.]+)", text)
evals = re.findall(r"'eval_loss': ([0-9.]+)", text)
tacc = re.findall(r"'token_acc': ([0-9.]+)", text)
out.write_text(json.dumps({
    "rc": rc,
    "state": "done" if rc == 0 else ("timeout" if rc == 124 else "failed"),
    "last_loss": float(losses[-1]) if losses else None,
    "eval_losses": [float(v) for v in evals],
    "last_token_acc": float(tacc[-1]) if tacc else None,
    "oom": bool(re.search(r"out of memory", text, re.I)),
    "log": str(log),
}, indent=2) + "\n")
PY
cat "$OUT/train_status.json"
case "$RC" in
  0)   echo " done: $OUT" ;;
  124) echo " hit ${HOURS}h cap — re-run the same command to resume" ;;
  *)   echo " FAILED rc=$RC"
       grep -iE 'error|out of memory|Signal 9|Traceback' "$LOG" | tail -8 ;;
esac
exit "$RC"
