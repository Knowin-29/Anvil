#!/usr/bin/env bash
# Volc entrypoint: per-dimension teachers (G / S / A) from Qwen3.6-27B, one after
# another on the 8xA800 node. Each consumes a dimension-sliced clean-v1 jsonl.
# Resumable per teacher.
set -uo pipefail
REPO=/path/to/Anvil
source "${REPO}/.env" 2>/dev/null || true
[[ -d /path/to/workspace/foundation ]] || { echo "FATAL: mount missing"; exit 1; }
NG=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null)
MODEL_27B="/path/to/environment/hf_models/Qwen/Qwen3.6-27B"
cd "${REPO}"

for DIM in G S A; do
  DATA="${REPO}/trains/gsa_sft_clean_v1_${DIM}.jsonl"
  OUT="${REPO}/output/teacher_${DIM}_clean_v1"
  [[ -f "${DATA}" ]] || { echo "SKIP ${DIM}: ${DATA} missing"; continue; }
  mkdir -p "${OUT}"
  LAST=$(ls -d "${OUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  RES=""; [[ -n "${LAST}" ]] && RES="--resume_from_checkpoint ${LAST}"
  echo "=== teacher ${DIM} on ${DATA} ${RES} ==="
  NPROC_PER_NODE=${NG} PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
  swift sft --model "${MODEL_27B}" --dataset "${DATA}" \
    --tuner_type full --torch_dtype bfloat16 --deepspeed zero3 \
    --num_train_epochs 2 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 --learning_rate 1e-5 --warmup_ratio 0.05 \
    --lr_scheduler_type cosine --max_length 4096 --gradient_checkpointing true \
    --logging_steps 5 --save_steps 200 --save_total_limit 2 \
    --output_dir "${OUT}" --dataloader_num_workers 8 --dataset_num_proc 8 ${RES} \
    2>&1 | tee -a "${REPO}/logs/volc_teacher_${DIM}_$(date -u +%Y%m%d_%H%M%S).log"
done
echo "=== all teachers done ==="
