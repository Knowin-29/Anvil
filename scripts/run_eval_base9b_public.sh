#!/usr/bin/env bash
# Evaluate a served model on the public benchmarks (E1). Requires the model to be
# served OpenAI-compatibly (vLLM/lmdeploy) at $API_HOST:$API_PORT. Since the local
# box has no vLLM, this is normally run either (a) on a Volc eval task, or (b) against
# a remote served checkpoint. Backend=api reuses the validated runners.
set -uo pipefail
cd "$(dirname "$0")/.."
source .env
API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-5031}"
API_MODEL="${API_MODEL:-gsa-9b}"
RUNROOT=/path/to/datasets/datasets/benchmark_eval_all/src
DATA=/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark
OUT="eval/public/${API_MODEL}"
mkdir -p "$OUT"

echo "=== EmbSpatial ==="
python "$RUNROOT/embspatial/inference.py" --backend api --api-host "$API_HOST" \
  --api-port "$API_PORT" --api-model "$API_MODEL" \
  --data-path "$DATA/EmbSpatial-Bench/embspatial_bench.json" \
  --output-path "$OUT/embspatial.json" 2>&1 | tail -5

echo "NOTE: add cv_bench / erqa / where2place / refspatial / vsibench / blink / robovqa / openeqa"
echo "using configs/eval/benchmarks.yaml runner list once a checkpoint is served."
