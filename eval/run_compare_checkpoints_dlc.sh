#!/bin/bash
# Multi-checkpoint benchmark comparison (DLC multi-GPU)
# Runs: 27b_v7_3000 on DLC (needs more VRAM for 27B model)
# DLC config: 1 node, 2+ GPUs (H200 or L20X)
#
# DLC launch command:
#   cd ${GSA_FORGE_ROOT} && bash eval/run_compare_checkpoints_dlc.sh
set -euo pipefail

ROOT="${GSA_FORGE_ROOT}"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN}"
export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"

# 27B checkpoint - needs TP=2 for comfortable inference
CKPT_27B="${GSA_FORGE_ROOT}/output/sft_27b_v7_fresh/v0-20260612-145559/checkpoint-3000"

# Override for 27B: TP=2, lower batch, 2 GPUs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SWIFT_VLLM_TENSOR_PARALLEL_SIZE="${SWIFT_VLLM_TENSOR_PARALLEL_SIZE:-2}"
export SWIFT_VLLM_GPU_MEMORY_UTILIZATION="${SWIFT_VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export SWIFT_VLLM_MAX_NUM_SEQS="${SWIFT_VLLM_MAX_NUM_SEQS:-8}"
export SWIFT_VLLM_MAX_MODEL_LEN="${SWIFT_VLLM_MAX_MODEL_LEN:-16384}"

echo "============================================================"
echo " 27B Checkpoint Benchmark (DLC Multi-GPU)"
echo " Checkpoint: ${CKPT_27B}"
echo " GPU: ${CUDA_VISIBLE_DEVICES}, TP=${SWIFT_VLLM_TENSOR_PARALLEL_SIZE}"
echo " Task set: promoted9 (full)"
echo "============================================================"

"${PYTHON_BIN}" eval/run_output_benchmark_suite.py \
    --paths "${CKPT_27B}" \
    --task-set promoted9 \
    --limit 0 \
    --reset-output \
    --python-bin "${PYTHON_BIN}" \
    --ms-swift-root ${GSA_FORGE_ROOT}/../ms-swift

echo "Done. Results in eval/benchmark_runs/"
