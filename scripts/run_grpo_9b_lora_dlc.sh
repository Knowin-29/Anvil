#!/usr/bin/env bash
# run_grpo_9b_lora_dlc.sh — E3: verifier-as-REWARD (not just filter)
#
# This is the paper's methodological claim. OpenThoughts (ICLR 2026 Oral) shows that using
# a verifier only to FILTER data often fails to beat no filtering; the escape is to make the
# verification signal a process reward. gsa_reward_plugin.py computes exactly that, from the
# same per-form contracts that filtered the corpus.
#
# Requires the E1 LoRA checkpoint as the starting policy.
#   bash /path/to/Anvil/scripts/run_grpo_9b_lora_dlc.sh
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
export WANDB_DIR="$W/logs" WANDB_PROJECT=GSA_Forge
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/output"

HOURS="${HOURS:-12}"
MODEL=/path/to/environment/hf_models/Qwen/Qwen3.5-9B
OUT="$W/output/grpo_9b_lora"

NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs"; exit 1; }

# Start from the SFT policy produced by E1. Resolution (including the `_l20` variant
# that run_all_dlc.sh actually writes to on L20 nodes) lives in one helper so the GRPO
# and eval call sites cannot drift apart again.
ADAPTER=$(bash "$W/scripts/_find_sft_adapter.sh") || exit 1

TRAIN="$W/trains/gsa_bench/train_shuf.jsonl"
[[ -s "$TRAIN" ]] || TRAIN="$W/trains/gsa_bench/train.jsonl"

PREFLIGHT="$W/artifacts/audits/grpo_dataset_contract.json"
mkdir -p "$(dirname "$PREFLIGHT")"
python "$W/scripts/validate_grpo_dataset.py" "$TRAIN" --out "$PREFLIGHT" || {
  echo "FATAL: GRPO dataset is not contract-locked; see $PREFLIGHT"; exit 1; }

# GA was referenced but never assigned; under `set -u` that aborts the swift invocation
# before a single step runs. Effective batch is pinned to 32 like the SFT scripts.
BS="${BS:-4}"
GA="${GA:-$(( 32 / (NG * BS) ))}"; [[ "$GA" -lt 1 ]] && GA=1
# GRPO requires (num_processes * BS * steps_per_generation) % num_generations == 0
# (swift/rlhf_trainers/arguments.py:check_num_generations). generation_batch_size here is
# NG*BS*GA; with NG*BS*GA == 32 any GEN in {2,4,8} divides it.
GEN="${GEN:-4}"
if (( (NG * BS * GA) % GEN != 0 )); then
  echo "FATAL: generation batch $((NG*BS*GA)) not divisible by num_generations $GEN"; exit 1
fi

RESUME=()
# Rank by numeric step. `sort -t- -k2 -n` compares the version-dir field of
# v0-<date>-<time>/checkpoint-N and silently resumes from the OLDEST checkpoint.
LAST=$(ls -d "$OUT"/checkpoint-* "$OUT"/v*/checkpoint-* 2>/dev/null \
       | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 \
       | awk 'NF==2' | tail -1 | cut -d' ' -f2)
[[ -n "$LAST" ]] && RESUME=(--resume_from_checkpoint "$LAST")

LOG="$W/logs/grpo_9b_lora_dlc_$(date -u +%Y%m%d_%H%M%S).log"
echo "=================================================="
echo " run     : grpo_9b_lora_dlc  (E3 verifier-as-reward)"
echo " GPUs    : $NG"
echo " policy  : $ADAPTER"
echo " reward  : gsa_reward (0.3*contract + 0.7*quality)"
echo " batch   : NG=$NG BS=$BS GA=$GA gen_batch=$((NG*BS*GA)) num_generations=$GEN"
echo " log     : $LOG"
echo "=================================================="

NPROC_PER_NODE="$NG" timeout "${HOURS}h" swift rlhf --rlhf_type grpo \
  --model "$MODEL" \
  --adapters "$ADAPTER" \
  --dataset "$TRAIN" \
  --external_plugins "$W/scripts/gsa_reward_plugin.py" \
  --reward_funcs gsa_reward \
  --tuner_type lora \
  `# rank/alpha come from $ADAPTER/adapter_config.json via tuner.from_pretrained` \
  --torch_dtype bfloat16 --deepspeed zero2 \
  --split_dataset_ratio 0 \
  --max_steps 600 \
  --num_generations "$GEN" \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --learning_rate 2e-6 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_length 3072 --max_completion_length 512 \
  --temperature 1.0 \
  --gradient_checkpointing true \
  --logging_steps 5 --save_steps 100 --save_total_limit 3 \
  --output_dir "$OUT" --dataloader_num_workers 16 --dataset_num_proc 32 \
  --report_to wandb "${RESUME[@]}" \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "=================================================="
case "$RC" in
  0)   echo " done. checkpoints in $OUT"
       grep -oE "'reward': '[-0-9.]+'" "$LOG" | tail -3 ;;
  124) echo " hit ${HOURS}h cap — re-run to resume" ;;
  *)   echo " FAILED rc=$RC"; grep -iE 'error|assert|out of memory' "$LOG" | tail -5 ;;
esac
exit "$RC"
