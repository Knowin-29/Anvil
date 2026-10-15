#!/bin/bash
# Run a model over all GSA-Bench dimension eval sets + score. EVAL_MODEL/EVAL_ADAPTER overridable.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env 2>/dev/null || true

MODEL="${EVAL_MODEL:-${MODEL_9B}}"
ADAPTER="${EVAL_ADAPTER:-}"
TAG="${EVAL_TAG:-base9b}"
GPU="${EVAL_GPUS:-0}"
PORT="${MASTER_PORT:-29601}"
DIMS=(G_grounding S_spatial A_identity_attr A_obj_state A_action_precondition A_action_interface)
SWIFT=/path/to/environment/env/gsa_train/bin/swift
PY=/path/to/environment/env/gsa_train/bin/python
mkdir -p eval/gsa_bench/out logs

SUF="${EVAL_SUFFIX:-}"   # e.g. "60" to use the _eval60 subsets
MBS="${EVAL_BATCH:-8}"

MODEL_ARG=(--model "${MODEL}")
[ -n "${ADAPTER}" ] && MODEL_ARG=(--adapters "${ADAPTER}")

echo "=== [$(date)] GSA-Bench eval: tag=${TAG} model=${MODEL} adapter=${ADAPTER:-none} gpu=${GPU} ==="
RESULTS="eval/gsa_bench/results_${TAG}.jsonl"; : > "${RESULTS}"
for dim in "${DIMS[@]}"; do
    OUT="eval/gsa_bench/out/${dim}_${TAG}_out.jsonl"
    DIMLOG="logs/gsa_bench_${dim}_${TAG}.log"
    echo "--- ${dim} ---"
    CUDA_VISIBLE_DEVICES="${GPU}" NPROC_PER_NODE=1 MASTER_PORT="${PORT}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${SWIFT}" infer "${MODEL_ARG[@]}" \
        --val_dataset "eval/gsa_bench/${dim}_eval${SUF}.jsonl" \
        --infer_backend pt --max_new_tokens 512 --max_batch_size "${MBS}" \
        --result_path "${OUT}" > "${DIMLOG}" 2>&1
    if [ -s "${OUT}" ]; then
        "${PY}" eval/score_gsa_eval.py --dim "${dim}" --pred "${OUT}" \
            --gt "eval/gsa_bench/${dim}_gt${SUF}.jsonl" --name "${TAG}" | tee -a "${RESULTS}"
    else
        echo "{\"dim\":\"${dim}\",\"status\":\"infer_failed\",\"see\":\"${DIMLOG}\"}" | tee -a "${RESULTS}"
    fi
done
echo "=== [$(date)] done -> ${RESULTS} ==="
