#!/bin/bash
# Offline embodied-arena benchmark runner.
#
# This wrapper is owned by GSA_Forge. It calls the benchmark repo read-only:
#   1) export embodied task prompts to MS-Swift JSONL
#   2) run Swift/vLLM resumable inference in the instance default environment
#   3) rescore predictions with benchmark-native process_results
#
# Usage:
#   CKPT_TAG=base_27b LIMIT=2 bash eval/run_embodied_benchmark_offline.sh
#   CKPT_TAG=v7_3000 LIMIT=0 TASK_SET=promoted9 bash eval/run_embodied_benchmark_offline.sh
set -euo pipefail

ROOT="${GSA_FORGE_ROOT}"
cd "${ROOT}"

CKPT_REGISTRY="${CKPT_REGISTRY:-${ROOT}/eval/checkpoints_v0616.json}"
CKPT_TAG="${CKPT_TAG:-base_27b}"
TASK_SET="${TASK_SET:-promoted9}"
TASKS="${TASKS:-}"
LIMIT="${LIMIT:-2}"  # 0 means full task set.

RUNNER_DIR="${DATA_ROOT}/datasets/correct-balanced-train-test-v1/swift_infer_16k"
VLLM_RUNNER_PY="${VLLM_RUNNER_PY:-${ROOT}/eval/run_swift_vllm_resumable.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="${REPO_ROOT:-${GSA_FORGE_ROOT}/../ms-swift}"
VENV_DIR="${VENV_DIR:-}"
SKIP_INFERENCE="${SKIP_INFERENCE:-false}"

if [[ -z "${MODEL_PATH:-}" ]]; then
MODEL_PATH="$("${PYTHON_BIN}" - "${CKPT_REGISTRY}" "${CKPT_TAG}" <<'PY'
import json, sys
registry = json.load(open(sys.argv[1], encoding="utf-8"))
tag = sys.argv[2]
if tag not in registry:
    raise SystemExit(f"unknown CKPT_TAG={tag}; available={','.join(sorted(registry))}")
print(registry[tag]["model_path"])
PY
)"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model path not found for ${CKPT_TAG}: ${MODEL_PATH}" >&2
    exit 2
fi

eval "$("${PYTHON_BIN}" eval/checkpoint_resolver.py "${MODEL_PATH}" \
    --base-model-path "${BASE_MODEL_PATH:-}" \
    --adapters "${ADAPTERS:-}" \
    --format shell)"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: resolved base model path not found for ${CKPT_TAG}: ${MODEL_PATH}" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/eval/benchmark_runs/${CKPT_TAG}_${TASK_SET}_${STAMP}}"
DATASET="${RUN_ROOT}/swift_infer.jsonl"
RESULT_DIR="${RUN_ROOT}/infer"
TAG="${TAG:-${CKPT_TAG}_${TASK_SET}}"
PRED="${RESULT_DIR}/${TAG}.jsonl"
LOG_PATH="${RESULT_DIR}/${TAG}.log"
SCORED_DIR="${RUN_ROOT}/scored"
mkdir -p "${RUN_ROOT}" "${RESULT_DIR}" "${SCORED_DIR}"

EXPORT_ARGS=(--task-set "${TASK_SET}" --output "${DATASET}")
SCORE_ARGS=(--task-set "${TASK_SET}" --pred "${PRED}" --output-dir "${SCORED_DIR}")
if [[ -n "${TASKS}" ]]; then
    EXPORT_ARGS+=(--tasks "${TASKS}")
    SCORE_ARGS+=(--tasks "${TASKS}")
fi
if [[ "${LIMIT}" != "0" ]]; then
    EXPORT_ARGS+=(--limit "${LIMIT}")
    SCORE_ARGS+=(--limit "${LIMIT}")
fi

echo "============================================================"
echo " Embodied benchmark offline eval"
echo " ckpt_tag=${CKPT_TAG}"
echo " model=${MODEL_PATH}"
echo " adapters=${ADAPTERS:-<none>}"
echo " ckpt_kind=${CKPT_KIND}"
echo " task_set=${TASK_SET} limit=${LIMIT}"
echo " run_root=${RUN_ROOT}"
echo "============================================================"

"${PYTHON_BIN}" eval/embodied_benchmark_offline.py export "${EXPORT_ARGS[@]}"

if [[ "${SKIP_INFERENCE}" == "true" ]]; then
    echo "SKIP_INFERENCE=true: exported dataset only."
    echo "Dataset: ${DATASET}"
    exit 0
fi

export REPO_ROOT VENV_DIR MODEL_PATH ADAPTERS
export VAL_DATASET="${DATASET}"
export RESULT_DIR TAG
export RESULT_PATH="${PRED}"
export FAILED_PATH="${RESULT_DIR}/${TAG}.failed.jsonl"
export LOG_PATH
export RESUME="${RESUME:-true}"
export RESET_OUTPUT="${RESET_OUTPUT:-false}"
export ALLOW_FAILED="${ALLOW_FAILED:-true}"
export OVERLONG_POLICY="${OVERLONG_POLICY:-skip}"
export GLOBAL_TQDM="${GLOBAL_TQDM:-true}"
export MAX_RETRIES="${MAX_RETRIES:-2}"
export RETRY_SLEEP="${RETRY_SLEEP:-5}"
export SPLIT_ON_ERROR="${SPLIT_ON_ERROR:-true}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export MAX_LENGTH="${MAX_LENGTH:-15328}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-false}"
export SWIFT_VLLM_ATTENTION_BACKEND="${SWIFT_VLLM_ATTENTION_BACKEND:-auto_flash}"

MODEL_SIZE_CLASS="${MODEL_SIZE_CLASS:-auto}"
if [[ "${MODEL_SIZE_CLASS}" == "auto" ]]; then
    MODEL_PATH_LC="$(printf '%s %s' "${MODEL_PATH}" "${ADAPTERS:-}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${CKPT_TAG}" == *"9b"* || "${CKPT_TAG}" == "base_9b" || "${MODEL_PATH_LC}" == *"9b"* ]]; then
        MODEL_SIZE_CLASS="9b"
    else
        MODEL_SIZE_CLASS="27b"
    fi
fi

if [[ "${MODEL_SIZE_CLASS}" == "9b" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export SWIFT_VLLM_TENSOR_PARALLEL_SIZE="${SWIFT_VLLM_TENSOR_PARALLEL_SIZE:-1}"
    export BATCH_SIZE="${BATCH_SIZE:-32}"
    export SWIFT_VLLM_MAX_NUM_SEQS="${SWIFT_VLLM_MAX_NUM_SEQS:-16}"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export SWIFT_VLLM_TENSOR_PARALLEL_SIZE="${SWIFT_VLLM_TENSOR_PARALLEL_SIZE:-1}"
    export BATCH_SIZE="${BATCH_SIZE:-8}"
    export SWIFT_VLLM_MAX_NUM_SEQS="${SWIFT_VLLM_MAX_NUM_SEQS:-8}"
fi
export SWIFT_VLLM_MAX_MODEL_LEN="${SWIFT_VLLM_MAX_MODEL_LEN:-16384}"
export SWIFT_VLLM_GPU_MEMORY_UTILIZATION="${SWIFT_VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
if [[ -n "${ADAPTERS:-}" ]]; then
    export SWIFT_VLLM_MAX_LORA_RANK="${SWIFT_VLLM_MAX_LORA_RANK:-${LORA_RANK:-128}}"
fi
if [[ "${ENABLE_QWEN35_VLLM_TIPS:-true}" == "true" && -z "${SWIFT_VLLM_ENGINE_KWARGS:-}" ]]; then
    # vLLM 0.19+ natively supports Qwen3.5; only need mm optimizations.
    export SWIFT_VLLM_ENGINE_KWARGS='{"mm_encoder_tp_mode":"data","mm_processor_cache_type":"shm"}'
fi
export INFER_BACKEND="${INFER_BACKEND:-vllm}"
export AUTO_FALLBACK_TO_TRANSFORMERS="${AUTO_FALLBACK_TO_TRANSFORMERS:-true}"
export TRANSFORMERS_BATCH_SIZE="${TRANSFORMERS_BATCH_SIZE:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Running Swift inference with default environment: ${VLLM_RUNNER_PY}"
echo "python=${PYTHON_BIN}"
echo "repo_root=${REPO_ROOT}"
echo "infer_backend=${INFER_BACKEND}"
echo "log=${LOG_PATH}"
if [[ -n "${VENV_DIR}" ]]; then
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        echo "ERROR: VENV_DIR was set but activate was not found: ${VENV_DIR}/bin/activate" >&2
        exit 2
    fi
    # Optional compatibility path. By default this runner uses the instance Python.
    source "${VENV_DIR}/bin/activate"
fi
set +e
"${PYTHON_BIN}" "${VLLM_RUNNER_PY}" 2>&1 | tee "${LOG_PATH}"
INFER_RC="${PIPESTATUS[0]}"
set -e

if [[ "${INFER_RC}" != "0" ]]; then
    echo "WARNING: inference runner exited with code ${INFER_RC}; marking missing predictions as failed before scoring." >&2
    "${PYTHON_BIN}" eval/fill_failed_predictions.py \
        --source "${DATASET}" \
        --pred "${PRED}" \
        --failed "${FAILED_PATH}" \
        --error-type "InferenceRunnerFailed" \
        --error-message "runner exited with code ${INFER_RC}" \
        --log "${LOG_PATH}"
fi

"${PYTHON_BIN}" eval/embodied_benchmark_offline.py score "${SCORE_ARGS[@]}"

echo "Done."
echo "Summary: ${SCORED_DIR}/summary.md"
if [[ "${INFER_RC:-0}" != "0" ]]; then
    exit "${INFER_RC}"
fi
