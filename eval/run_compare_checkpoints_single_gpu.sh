#!/bin/bash
# Multi-checkpoint benchmark comparison (Single GPU)
# Runs: base_9b, 9b_full_3000, 9b_lora_9000 on this machine (1x L20X)
# 9b_lora_12000 already done, results reused from existing suite
set -euo pipefail

ROOT="${GSA_FORGE_ROOT}"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN}"
export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"

CHECKPOINTS=(
    "${MODEL_ROOT}/Qwen/Qwen3.5-9B"
    "${GSA_FORGE_ROOT}/output/sft_9b_v1/v0-20260616-223759/checkpoint-3000"
    "${GSA_FORGE_ROOT}/output/sft_9b_lora_v1/Qwen3.5-9B_sft_lorar128_ep1_lr1e-4_9b_GSA_v1/v9-20260620-030752/checkpoint-9000"
)

PATHS_ARG=""
for ckpt in "${CHECKPOINTS[@]}"; do
    PATHS_ARG="${PATHS_ARG} --paths ${ckpt}"
done

echo "============================================================"
echo " Multi-Checkpoint Benchmark (Single GPU)"
echo " Checkpoints: ${#CHECKPOINTS[@]}"
echo " Task set: promoted9 (full)"
echo "============================================================"

"${PYTHON_BIN}" eval/run_output_benchmark_suite.py \
    ${PATHS_ARG} \
    --task-set promoted9 \
    --limit 0 \
    --reset-output \
    --python-bin "${PYTHON_BIN}" \
    --ms-swift-root ${GSA_FORGE_ROOT}/../ms-swift

echo "Done. Results in eval/benchmark_runs/"
