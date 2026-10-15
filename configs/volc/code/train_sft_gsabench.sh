#!/usr/bin/env bash
# Stage 2 / run B0: Qwen3.6-27B teacher SFT on GSA-Bench train (pure flywheel data,
# 8 task forms). This run is what substantiates the paper's core claim that the
# flywheel's generated supervision is useful.
#
# Resumable (72h task deadline), ZeRO-3, 8xA800. Validation uses the image-disjoint
# GSA-Bench val split so we can watch for overfitting without touching test.
set -uo pipefail
REPO=/path/to/Anvil
source "${REPO}/.env" 2>/dev/null || true

[[ -d /path/to/workspace/foundation ]] || { echo "FATAL: vePFS not mounted"; exit 1; }
NG=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null); echo "GPUs=${NG}"

MODEL="${BASE_MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.6-27B}"
TRAIN="${REPO}/trains/gsa_bench/train.jsonl"
VAL="${REPO}/trains/gsa_bench/val.jsonl"
OUT="${REPO}/output/sft_27b_gsabench"

for f in "$TRAIN" "$VAL"; do
  [[ -s "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done
mkdir -p "$OUT" "${REPO}/logs"

RES=""
LAST=$(ls -d "${OUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[[ -n "$LAST" ]] && RES="--resume_from_checkpoint ${LAST}" && echo "RESUME from ${LAST}"

cd "$REPO"
NPROC_PER_NODE=${NG} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
  --model "$MODEL" \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --deepspeed zero3 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --max_length 4096 \
  --gradient_checkpointing true \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 2 \
  --output_dir "$OUT" \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --report_to wandb \
  ${RES} \
  2>&1 | tee -a "${REPO}/logs/volc_sft_27b_gsabench_$(date -u +%Y%m%d_%H%M%S).log"
echo "=== B0 (27B SFT on GSA-Bench) done ==="
