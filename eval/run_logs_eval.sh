#!/bin/bash
# Run 10k database-log evaluation with MS-Swift/vLLM and the hybrid scorer.
set -euo pipefail

ROOT="${GSA_FORGE_ROOT}"
cd "${ROOT}"

DEFAULT_CKPT="${ROOT}/output/sft_9b_lora_v1/Qwen3.5-9B_sft_lorar128_ep1_lr1e-4_9b_GSA_v1/v9-20260620-030752/checkpoint-12000"
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
CKPT_TAG="${CKPT_TAG:-$(basename "$(dirname "${CKPT_PATH}")")_$(basename "${CKPT_PATH}")}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="${REPO_ROOT:-${GSA_FORGE_ROOT}/../ms-swift}"
RUNNER_PY="${RUNNER_PY:-${ROOT}/eval/run_swift_vllm_resumable.py}"

VAL_DATASET="${VAL_DATASET:-${DATA_ROOT}/datasets/correct-balanced-train-test-v1/swift_infer_16k/test.swift_infer.labeled_eval.sample_10k.jsonl}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/eval/log_runs/${CKPT_TAG}_${STAMP}}"
RESULT_DIR="${RUN_ROOT}/infer"
TAG="${TAG:-${CKPT_TAG}_logs10k}"
RESULT_PATH="${RESULT_PATH:-${RESULT_DIR}/${TAG}.jsonl}"
FAILED_PATH="${FAILED_PATH:-${RESULT_DIR}/${TAG}.failed.jsonl}"
LOG_PATH="${LOG_PATH:-${RESULT_DIR}/${TAG}.log}"
SCORED_DIR="${RUN_ROOT}/scored"
mkdir -p "${RESULT_DIR}" "${SCORED_DIR}"

if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "ERROR: CKPT_PATH not found: ${CKPT_PATH}" >&2
    exit 2
fi

eval "$("${PYTHON_BIN}" eval/checkpoint_resolver.py "${CKPT_PATH}" \
    --base-model-path "${BASE_MODEL_PATH:-}" \
    --adapters "${ADAPTERS:-}" \
    --format shell)"

cat <<EOF
============================================================
 Logs hybrid eval
 ckpt_path=${CKPT_PATH}
 ckpt_kind=${CKPT_KIND}
 model=${MODEL_PATH}
 adapters=${ADAPTERS:-<none>}
 lora_rank=${LORA_RANK:-<none>}
 dataset=${VAL_DATASET}
 result=${RESULT_PATH}
 run_root=${RUN_ROOT}
============================================================
EOF

if [[ "${SKIP_INFERENCE:-false}" != "true" ]]; then
    export REPO_ROOT MODEL_PATH ADAPTERS
    export VAL_DATASET RESULT_PATH FAILED_PATH LOG_PATH RESULT_DIR TAG
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
    export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
    export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
    export SWIFT_VLLM_ATTENTION_BACKEND="${SWIFT_VLLM_ATTENTION_BACKEND:-auto_flash}"
    export SWIFT_VLLM_TENSOR_PARALLEL_SIZE="${SWIFT_VLLM_TENSOR_PARALLEL_SIZE:-1}"
    export SWIFT_VLLM_GPU_MEMORY_UTILIZATION="${SWIFT_VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
    export SWIFT_VLLM_MAX_MODEL_LEN="${SWIFT_VLLM_MAX_MODEL_LEN:-16384}"
    export SWIFT_VLLM_MAX_NUM_SEQS="${SWIFT_VLLM_MAX_NUM_SEQS:-16}"
    export SWIFT_VLLM_ENFORCE_EAGER="${SWIFT_VLLM_ENFORCE_EAGER:-false}"
    if [[ -z "${SWIFT_VLLM_LIMIT_MM_PER_PROMPT:-}" ]]; then
        export SWIFT_VLLM_LIMIT_MM_PER_PROMPT='{"image": 10, "video": 0}'
    fi
    if [[ -n "${ADAPTERS:-}" ]]; then
        export SWIFT_VLLM_MAX_LORA_RANK="${SWIFT_VLLM_MAX_LORA_RANK:-${LORA_RANK:-128}}"
    fi
    if [[ "${ENABLE_QWEN35_VLLM_TIPS:-true}" == "true" && -z "${SWIFT_VLLM_ENGINE_KWARGS:-}" ]]; then
        export SWIFT_VLLM_ENGINE_KWARGS='{"mm_encoder_tp_mode":"data","mm_processor_cache_type":"shm"}'
    fi
    export INFER_BACKEND="${INFER_BACKEND:-vllm}"
    export AUTO_FALLBACK_TO_TRANSFORMERS="${AUTO_FALLBACK_TO_TRANSFORMERS:-true}"
    export TRANSFORMERS_BATCH_SIZE="${TRANSFORMERS_BATCH_SIZE:-1}"
    export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
    export MAX_INPUT_MARGIN="${MAX_INPUT_MARGIN:-32}"
    export MAX_LENGTH="${MAX_LENGTH:-15328}"
    export OVERLONG_POLICY="${OVERLONG_POLICY:-skip}"
    export ENABLE_THINKING="${ENABLE_THINKING:-false}"
    export LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-false}"
    export BATCH_SIZE="${BATCH_SIZE:-32}"
    export MAX_RETRIES="${MAX_RETRIES:-2}"
    export RETRY_SLEEP="${RETRY_SLEEP:-5}"
    export SPLIT_ON_ERROR="${SPLIT_ON_ERROR:-true}"
    export RESUME="${RESUME:-true}"
    export RESET_OUTPUT="${RESET_OUTPUT:-false}"
    export ALLOW_FAILED="${ALLOW_FAILED:-true}"

    set +e
    "${PYTHON_BIN}" "${RUNNER_PY}" 2>&1 | tee "${LOG_PATH}"
    INFER_RC="${PIPESTATUS[0]}"
    set -e
    if [[ "${INFER_RC}" != "0" ]]; then
        echo "WARNING: inference runner exited with code ${INFER_RC}; marking missing log predictions as failed before scoring." >&2
        "${PYTHON_BIN}" eval/fill_failed_predictions.py \
            --source "${VAL_DATASET}" \
            --pred "${RESULT_PATH}" \
            --failed "${FAILED_PATH}" \
            --error-type "InferenceRunnerFailed" \
            --error-message "runner exited with code ${INFER_RC}" \
            --log "${LOG_PATH}"
    fi
else
    echo "SKIP_INFERENCE=true: scoring existing RESULT_PATH=${RESULT_PATH}"
    INFER_RC=0
fi

"${PYTHON_BIN}" eval/score_logs_hybrid.py \
    --pred "${RESULT_PATH}" \
    --output-dir "${SCORED_DIR}" \
    --tag "${TAG}"

echo "Done."
echo "Summary: ${SCORED_DIR}/summary.md"
if [[ "${INFER_RC:-0}" != "0" ]]; then
    exit "${INFER_RC}"
fi
