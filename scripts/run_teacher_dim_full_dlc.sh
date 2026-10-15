#!/usr/bin/env bash
# run_teacher_dim_full_dlc.sh — S1: one per-dimension teacher (G | S | A).
#
# Trains one specialist on a declared G, S, or A slice. Specialist value is an
# empirical question: a checkpoint is teacher-eligible only after it beats the
# matched joint model on its assigned held-out family.
#
#   DIM=G bash scripts/run_teacher_dim_full_dlc.sh
#   DIM=S bash scripts/run_teacher_dim_full_dlc.sh
#   DIM=A bash scripts/run_teacher_dim_full_dlc.sh
#
# SIZE=9b (default) or SIZE=27b. See docs/DLC_NEXT.md for why 9b is the
# recommendation on 48GiB L20 nodes -- 27b full-parameter needs ~54GiB/GPU of
# persistent state under zero3 across 8 ranks, which does not fit, and the
# vocab/tokenizer check below is what makes a 27b teacher legal at all.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing (check the vePFS mount)"; exit 1; }
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

# caches must not touch read-only /path/to/workspace/foundation
export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$W/logs" WANDB_PROJECT=GSA_Forge
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/output"

DIM="${DIM:?set DIM=G|S|A}"
case "$DIM" in
  G|S|A) ;;
  *) echo "FATAL: DIM must be G, S or A (got '$DIM')"; exit 1 ;;
esac

SIZE="${SIZE:-9b}"
case "$SIZE" in
  9b)  MODEL=/path/to/environment/hf_models/Qwen/Qwen3.5-9B ;;
  27b) MODEL=/path/to/environment/hf_models/Qwen/Qwen3.6-27B ;;
  *)   echo "FATAL: SIZE must be 9b or 27b (got '$SIZE')"; exit 1 ;;
esac
[[ -f "$MODEL/config.json" ]] || { echo "FATAL: no model at $MODEL"; exit 1; }

HOURS="${HOURS:-46}"
BENCH_DIR="${BENCH_DIR:-$W/trains/gsa_bench_v4_composite_pilot_clean}"
TRAIN="$BENCH_DIR/train_$DIM.jsonl"
[[ -s "$TRAIN" ]] || { echo "FATAL: missing slice $TRAIN"; exit 1; }

# GPU count from torch, NOT nvidia-smi: `nvidia-smi -L | wc -l` ignores
# CUDA_VISIBLE_DEVICES and over-reports on a sliced node, which once pinned an
# 8-card job to 2 ranks and invalidated every time estimate.
NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs"; nvidia-smi -L | head -3; exit 1; }

TUNER="${TUNER:-lora}"
case "$TUNER" in
  full) OUT="${OUT:-$W/output/teacher_$DIM}" ;;
  lora) OUT="${OUT:-$W/output/teacher_lora_$DIM}" ;;
  *) echo "FATAL: TUNER must be full or lora (got '$TUNER')"; exit 1 ;;
esac
VAL="$BENCH_DIR/val_$DIM.jsonl"
[[ -s "$VAL" ]] || VAL="$BENCH_DIR/val.jsonl"
BS="${BS:-1}"
# Effective batch pinned to 32 so these runs stay comparable with E1/E2.
# GA must be assigned even when the arithmetic degenerates: this script runs
# under `set -u`, and an unset $GA is exactly what made E3 exit in 2 seconds
# with zero steps.
GA="${GA:-$(( 32 / (NG * BS) ))}"; [[ "$GA" -lt 1 ]] && GA=1
DS="${DS:-zero3_offload}"
MAXLEN="${MAXLEN:-3072}"

# Steps scale with slice size: S is only 43,873 rows (27% of G), so a fixed step
# count would give S ~4 epochs while G gets ~0.8.
ROWS=$(wc -l < "$TRAIN")
EPOCHS="${EPOCHS:-1}"
STEPS="${STEPS:-$(( (ROWS * EPOCHS + 31) / 32 ))}"
MIN_STEPS="${MIN_STEPS:-50}"
[[ "$MIN_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: MIN_STEPS must be a positive integer" >&2; exit 2;
}
[[ "$STEPS" -lt "$MIN_STEPS" ]] && STEPS="$MIN_STEPS"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
for value in "$STEPS" "$EVAL_STEPS" "$SAVE_STEPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: STEPS, EVAL_STEPS and SAVE_STEPS must be positive integers" >&2
    exit 2
  }
done

# Rank checkpoints by NUMERIC step. A plain `sort -t- -k2 -n` splits
# v0-20260816-044214/checkpoint-N on the first '-' and compares the date field,
# so it resumes from the OLDEST checkpoint.
RESUME=()
LAST=$(ls -d "$OUT"/checkpoint-* "$OUT"/v*/checkpoint-* 2>/dev/null \
       | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 \
       | awk 'NF==2' | tail -1 | cut -d' ' -f2)
[[ -n "$LAST" ]] && RESUME=(--resume_from_checkpoint "$LAST")

TUNER_ARGS=(--tuner_type "$TUNER")
if [[ "$TUNER" == "full" ]]; then
  TUNER_ARGS+=(--freeze_vit false --freeze_aligner false)
else
  TUNER_ARGS+=(--lora_rank "${LORA_RANK:-64}" --lora_alpha "${LORA_ALPHA:-128}" --lora_dropout 0.05)
fi

LOG="$W/logs/teacher_${DIM}_${SIZE}_dlc_$(date -u +%Y%m%d_%H%M%S).log"
echo "=================================================="
echo " run   : teacher_$DIM  ($SIZE, $TUNER)"
echo " GPUs  : $NG   per_device=$BS grad_accum=$GA  eff_batch=$(( NG * BS * GA ))"
echo " ds    : $DS   max_length=$MAXLEN"
echo " slice : $(basename "$TRAIN")  ($ROWS rows)   steps=$STEPS   cap=${HOURS}h"
echo " cadence: eval=$EVAL_STEPS   save=$SAVE_STEPS"
echo " val   : $(basename "$VAL")  ($(wc -l < "$VAL") rows)"
echo " resume: ${LAST:-none}"
free -g 2>/dev/null | awk 'NR==2{print " host RAM: "$2"G"}'
df -h "$W" 2>/dev/null | awk 'NR==2{print " disk free: "$4}'
echo " log   : $LOG"
echo "=================================================="

# A 27b teacher is only usable for GKD against a 9b student if both share the
# tokenizer and vocab; generalized_jsd_loss would otherwise pad the smaller vocab
# with the larger model's logits and silently distill garbage on those columns.
if [[ "$SIZE" == "27b" ]]; then
  python - "$MODEL" <<'PY' || exit 1
import json, sys, hashlib, pathlib
t = pathlib.Path(sys.argv[1]); s = pathlib.Path('/path/to/environment/hf_models/Qwen/Qwen3.5-9B')
def vocab(p):
    c = json.load(open(p / 'config.json'))
    return c.get('vocab_size') or (c.get('text_config') or {}).get('vocab_size')
def tok(p):
    f = p / 'tokenizer.json'
    return hashlib.md5(f.read_bytes()).hexdigest() if f.exists() else None
ok = vocab(t) == vocab(s) and tok(t) == tok(s)
print(f'  vocab teacher={vocab(t)} student={vocab(s)}  tokenizer_match={tok(t)==tok(s)}')
print('  OK: teacher/student vocab aligned' if ok else '  FATAL: vocab/tokenizer mismatch -> GKD would distill padded columns')
sys.exit(0 if ok else 1)
PY
fi

NPROC_PER_NODE="$NG" timeout "${HOURS}h" swift sft \
  --model "$MODEL" \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  "${TUNER_ARGS[@]}" \
  --torch_dtype bfloat16 --deepspeed "$DS" \
  --max_steps "$STEPS" \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --learning_rate 1e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_length "$MAXLEN" --gradient_checkpointing true \
  --logging_steps 20 --eval_steps "$EVAL_STEPS" --save_steps "$SAVE_STEPS" --save_total_limit 3 \
  --output_dir "$OUT" --dataloader_num_workers 16 --dataset_num_proc 32 \
  --report_to wandb "${RESUME[@]}" \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "=================================================="
case "$RC" in
  0)   echo " done. teacher_$DIM checkpoints in $OUT" ;;
  124) echo " hit ${HOURS}h cap — re-run the SAME command to resume" ;;
  *)   echo " FAILED rc=$RC"; grep -iE 'error|assert|out of memory|Signal 9' "$LOG" | tail -5 ;;
esac
exit "$RC"
