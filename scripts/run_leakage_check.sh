#!/usr/bin/env bash
set -euo pipefail
W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
source .env 2>/dev/null || true

: "${BENCH_FILES:?set BENCH_FILES to the space-separated JSONL inputs for every evaluated benchmark}"
TRAIN_FILES="${TRAIN_FILES:-$W/trains/gsa_bench/train.jsonl}"
read -r -a train_paths <<< "$TRAIN_FILES"
read -r -a bench_paths <<< "$BENCH_FILES"

python trains/check_benchmark_leakage.py \
  --train "${train_paths[@]}" --bench "${bench_paths[@]}" \
  --image-roots "${IMAGE_ROOTS:-/path/to/workspace/foundation:/path/to/workspace/demo}" \
