#!/usr/bin/env bash
# Volcengine entrypoint: 27B teacher SFT on clean-v1 data, 8xA800, ZeRO-3.
# Resumable: if a checkpoint exists in OUTPUT_DIR, swift resumes from it.
set -uo pipefail

REPO=/path/to/Anvil
source "${REPO}/.env" 2>/dev/null || true

# Preflight (fail fast, per ActionMap runbook lessons)
[[ -d /path/to/workspace/foundation ]] || { echo "FATAL: /path/to/workspace/foundation not mounted"; exit 1; }
NG=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null); echo "GPUs=${NG}"
[[ "${NG}" -ge 8 ]] || echo "WARN: expected 8 GPUs, got ${NG}"

MODEL_27B="/path/to/environment/hf_models/Qwen/Qwen3.6-27B"
DATA="${TRAIN_DATA:-${REPO}/trains/gsa_sft_clean_v1.jsonl}"
OUTPUT_DIR="${REPO}/output/sft_27b_clean_v1"
mkdir -p "${OUTPUT_DIR}" "${REPO}/logs"

RESUME_ARG=""
LAST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[[ -n "${LAST_CKPT}" ]] && RESUME_ARG="--resume_from_checkpoint ${LAST_CKPT}" && echo "RESUME ${LAST_CKPT}"

cd "${REPO}"
NPROC_PER_NODE=${NG} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
  --model "${MODEL_27B}" \
  --dataset "${DATA}" \
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
  --logging_steps 5 \
  --save_steps 200 \
  --save_total_limit 3 \
  --output_dir "${OUTPUT_DIR}" \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --save_only_model false \
  ${RESUME_ARG} \
  2>&1 | tee -a "${REPO}/logs/volc_sft_27b_$(date -u +%Y%m%d_%H%M%S).log"
echo "=== 27B clean-v1 SFT done ==="
