#!/usr/bin/env bash
# Volc entrypoint: OPD/GKD distillation from the 27B teacher into Qwen3.5-9B student.
# ms-swift rlhf --rlhf_type gkd. Teacher served in-process. Resumable.
set -uo pipefail
REPO=/path/to/Anvil
source "${REPO}/.env" 2>/dev/null || true
[[ -d /path/to/workspace/foundation ]] || { echo "FATAL: mount missing"; exit 1; }
NG=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null)
STUDENT="/path/to/environment/hf_models/Qwen/Qwen3.5-9B"
TEACHER="${TEACHER_CKPT:-${REPO}/output/sft_27b_clean_v1}"   # or a GRPO'd teacher
DATA="${DISTILL_DATA:-${REPO}/trains/gsa_sft_clean_v1.jsonl}"
OUT="${REPO}/output/opd_9b_clean_v1"
mkdir -p "${OUT}"
LAST=$(ls -d "${OUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
RES=""; [[ -n "${LAST}" ]] && RES="--resume_from_checkpoint ${LAST}"
cd "${REPO}"
echo "=== OPD distill teacher=${TEACHER} -> student=${STUDENT} ${RES} ==="
NPROC_PER_NODE=${NG} PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf --rlhf_type gkd \
  --model "${STUDENT}" --teacher_model "${TEACHER}" \
  --dataset "${DATA}" --torch_dtype bfloat16 --deepspeed zero3 \
  --num_train_epochs 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 --learning_rate 5e-6 --warmup_ratio 0.05 \
  --lr_scheduler_type cosine --max_length 4096 --gradient_checkpointing true \
  --logging_steps 5 --save_steps 200 --save_total_limit 2 \
  --output_dir "${OUT}" --dataloader_num_workers 8 --dataset_num_proc 8 ${RES} \
  2>&1 | tee -a "${REPO}/logs/volc_opd_9b_$(date -u +%Y%m%d_%H%M%S).log"
echo "=== OPD distill done ==="
