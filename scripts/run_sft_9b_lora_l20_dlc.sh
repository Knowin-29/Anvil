#!/usr/bin/env bash
# run_sft_9b_lora_l20_dlc.sh — 9B LoRA SFT tuned for 8x L20 (48 GiB, no NVLink)
#
# Platform entrypoint (one line):
#   bash scripts/run_sft_9b_lora_l20_dlc.sh
#
# Why this differs from the A800 script:
#   * 48 GiB, not 80 -> per-device batch 1. The current Qwen3.5-9B/3,072-token
#     stack reached 41 GiB at batch 2 before a 5.45-GiB allocation and OOMed.
#   * No NVLink, PCIe only -> ZeRO-2, never ZeRO-3. ZeRO-3 reshards parameters every
#     forward/backward, which is bandwidth-bound and would dominate runtime here.
#   * L20 is ~0.40x an A800 in both BF16 throughput and memory bandwidth, so the step
#     count is sized to that, not to the A800 numbers.
#
# Effective batch is pinned to 32 across all allocations so results stay comparable
# with the earlier A800 runs.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing (check the vePFS mount)"; exit 1; }
# Must precede any python call: torch lives in this venv, not in the container python.
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

# Caches must stay under the writable project root, never under a read-only model mount.
export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$W/logs" WANDB_PROJECT="${WANDB_PROJECT:-Anvil}"
# PCIe-only node: keep NCCL off any InfiniBand path and allow P2P over PCIe.
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=PIX
export DS_IGNORE_CUDA_DETECTION=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/output"

HOURS="${HOURS:-46}"
BS="${BS:-1}"                 # measured-safe default for 3,072-token multimodal batches
[[ "$BS" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: BS must be a positive integer" >&2
  exit 2
}
MODEL="${MODEL:-${MODEL_9B:-}}"
[[ -n "$MODEL" ]] || {
  echo "FATAL: set MODEL or MODEL_9B to a local Qwen3.5-9B directory" >&2
  exit 2
}
[[ -f "$MODEL/config.json" ]] || {
  echo "FATAL: MODEL is not a local model directory with config.json: $MODEL" >&2
  exit 2
}
SEED="${SEED:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || {
  echo "FATAL: SEED must be a non-negative integer" >&2
  exit 2
}
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-24}"
[[ "$DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ && "$DATASET_NUM_PROC" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: DATALOADER_NUM_WORKERS must be non-negative and DATASET_NUM_PROC positive" >&2
  exit 2
}

# torch, not nvidia-smi: nvidia-smi -L ignores CUDA_VISIBLE_DEVICES and over-reports on
# a sliced node, which previously made torchrun spawn 8 ranks for 2 visible devices.
NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
if [[ "$NG" -lt 1 ]]; then
  echo "FATAL: torch sees 0 GPUs. Diagnostics:"
  echo "  python: $(command -v python || echo MISSING)"
  python -c 'import torch;print("  torch:",torch.__version__,"cuda",torch.cuda.is_available())' 2>&1 | head -2
  nvidia-smi -L 2>&1 | head -3
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  exit 1
fi

GA="${GA:-$(( 32 / (NG * BS) ))}"
[[ "$GA" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: GA must be a positive integer" >&2
  exit 2
}
# The per-form files are concatenated in order, so the unshuffled corpus is locally
# clustered (a previous run gave dof 0 samples and mask only 562). Prefer the shuffled copy.
if [[ -n "${TRAIN+x}" ]]; then
  [[ -s "$TRAIN" ]] || { echo "FATAL: explicit TRAIN is missing or empty: $TRAIN" >&2; exit 2; }
else
  TRAIN="$W/trains/gsa_bench/train_shuf.jsonl"
  [[ -s "$TRAIN" ]] || TRAIN="$W/trains/gsa_bench/train.jsonl"
fi
# 1,200-row stratified val: the full 13,978-row split cost 47 min per eval.
if [[ -n "${VAL+x}" ]]; then
  [[ -s "$VAL" ]] || { echo "FATAL: explicit VAL is missing or empty: $VAL" >&2; exit 2; }
else
  VAL="$W/trains/gsa_bench/val_small.jsonl"
  [[ -s "$VAL" ]] || VAL="$W/trains/gsa_bench/val.jsonl"
fi
[[ -s "$TRAIN" && -s "$VAL" ]] || { echo "FATAL: no nonempty training/development corpus" >&2; exit 2; }
if [[ "$TRAIN" == *"/corpus_source_clean_v1/p4/train.jsonl" && \
      "$VAL" == *"/gsa_bench/val"* && \
      "${ALLOW_KNOWN_VAL_OVERLAP:-0}" != "1" ]]; then
  echo "FATAL: the historical GSA validation split overlaps corpus P4 (593 exact image paths)." >&2
  echo "Build and use the image-disjoint Anvil split; bypass is debug-only via ALLOW_KNOWN_VAL_OVERLAP=1." >&2
  exit 3
fi

CFG="${CFG:-}"
if [[ -n "${OUT:-}" ]]; then
  :
elif [[ -n "$CFG" ]]; then
  [[ "$CFG" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "FATAL: CFG may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
  }
  OUT="$W/output/${CFG}_seed${SEED}"
else
  OUT="$W/output/sft_9b_lora_l20"
fi

RANK="${RANK:-64}"
ALPHA="${ALPHA:-128}"
DROPOUT="${DROPOUT:-0.05}"
for value in "$RANK" "$ALPHA"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: RANK and ALPHA must be positive integers" >&2
    exit 2
  }
done
[[ "$DROPOUT" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || {
  echo "FATAL: DROPOUT must be between 0 and 1" >&2
  exit 2
}

TRAIN_ROWS=$(wc -l < "$TRAIN")
EFFECTIVE_BATCH=$(( NG * BS * GA ))
if [[ -z "${STEPS:-}" ]]; then
  if [[ -n "${EPOCHS:-}" ]]; then
    [[ "$EPOCHS" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
      echo "FATAL: EPOCHS must be a positive number" >&2
      exit 2
    }
    STEPS=$(awk -v rows="$TRAIN_ROWS" -v epochs="$EPOCHS" -v batch="$EFFECTIVE_BATCH" \
      'BEGIN { steps = int((rows * epochs + batch - 1) / batch); print steps }')
    [[ "$STEPS" -ge 1 ]] || { echo "FATAL: EPOCHS produced zero steps" >&2; exit 2; }
  else
    # Historical default: 8 GPUs -> 8,960 steps, about 27 h on this node.
    STEPS=$(( NG * 1120 ))
  fi
fi
EVAL_STEPS="${EVAL_STEPS:-750}"
SAVE_STEPS="${SAVE_STEPS:-750}"
for value in "$STEPS" "$EVAL_STEPS" "$SAVE_STEPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: STEPS, EVAL_STEPS and SAVE_STEPS must be positive integers" >&2
    exit 2
  }
done
(( SAVE_STEPS % EVAL_STEPS == 0 )) || {
  echo "FATAL: SAVE_STEPS must be a multiple of EVAL_STEPS with load_best_model_at_end" >&2
  exit 2
}
REPORT_TO="${REPORT_TO:-wandb}"

RESUME=()
LAST=$(ls -d "$OUT"/checkpoint-* "$OUT"/v*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[[ -n "$LAST" ]] && RESUME=(--resume_from_checkpoint "$LAST")
INITIAL_ADAPTER="${INITIAL_ADAPTER:-}"
if [[ -n "$INITIAL_ADAPTER" ]]; then
  [[ -z "$LAST" ]] || {
    echo "FATAL: INITIAL_ADAPTER cannot be combined with implicit checkpoint resume; use a fresh OUT" >&2
    exit 2
  }
  [[ -s "$INITIAL_ADAPTER/adapter_config.json" && -s "$INITIAL_ADAPTER/adapter_model.safetensors" ]] || {
    echo "FATAL: INITIAL_ADAPTER requires saved LoRA configuration and weights" >&2
    exit 2
  }
  RESUME=(--adapters "$INITIAL_ADAPTER")
fi
TOKEN_ACCOUNTING=()
[[ "${COUNT_INPUT_TOKENS:-0}" == "1" ]] && TOKEN_ACCOUNTING=(--include_num_input_tokens_seen true)

LOG="$W/logs/sft_9b_lora_l20_$(date -u +%Y%m%d_%H%M%S).log"
echo "=================================================="
echo " run     : sft_9b_lora_l20  (8x L20 tuned)"
echo " GPUs    : $NG   per_device_batch=$BS   grad_accum=$GA   eff_batch=$(( NG * BS * GA ))"
echo " steps   : $STEPS   cap=${HOURS}h"
echo " cadence : eval=$EVAL_STEPS   save=$SAVE_STEPS"
echo " seed    : $SEED"
echo " lora    : rank=$RANK alpha=$ALPHA dropout=$DROPOUT"
echo " workers : dataloader=$DATALOADER_NUM_WORKERS   dataset_proc=$DATASET_NUM_PROC"
echo " zero    : zero2 (PCIe-only node; zero3 would be bandwidth-bound)"
echo " train   : $(basename "$TRAIN") ($(wc -l < "$TRAIN") rows)"
echo " val     : $(basename "$VAL") ($(wc -l < "$VAL") rows)"
echo " resume  : ${LAST:-none}"
echo " initial : ${INITIAL_ADAPTER:-fresh base}"
python -c "import torch;print(' GPU0    :',torch.cuda.get_device_name(0),
      f'{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB')" 2>/dev/null
free -g 2>/dev/null | awk 'NR==2{print " host RAM: "$2"G"}'
df -h "$W" 2>/dev/null | awk 'NR==2{print " disk free: "$4}'
echo " log     : $LOG"
echo "=================================================="

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1: configuration validated; not launching training"
  exit 0
fi

NPROC_PER_NODE="$NG" timeout "${HOURS}h" swift sft \
  --model "$MODEL" \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  --tuner_type lora --lora_rank "$RANK" --lora_alpha "$ALPHA" --lora_dropout "$DROPOUT" \
  --torch_dtype bfloat16 --deepspeed zero2 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --seed "$SEED" --data_seed "$SEED" \
  --learning_rate "${LR:-1e-4}" --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_length 3072 --gradient_checkpointing true \
  --logging_steps 20 --eval_steps "$EVAL_STEPS" --save_steps "$SAVE_STEPS" --save_total_limit 3 \
  --load_best_model_at_end true --metric_for_best_model loss \
  --greater_is_better false \
  --output_dir "$OUT" --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --report_to "$REPORT_TO" "${RESUME[@]}" "${TOKEN_ACCOUNTING[@]}" \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "=================================================="
case "$RC" in
  0)   echo " done. checkpoints in $OUT"
       grep -oE "'loss': '[0-9.]+'.*'token_acc': '[0-9.]+'" "$LOG" | tail -2 ;;
  124) echo " hit ${HOURS}h cap — re-run the SAME command to resume from the last checkpoint" ;;
  *)   echo " FAILED rc=$RC"
       if grep -qiE 'out of memory' "$LOG"; then
         echo " -> OOM. Re-submit with:  BS=1 bash $0"
       fi
       grep -iE 'error|assert|out of memory|Signal 9' "$LOG" | tail -5 ;;
esac
exit "$RC"
