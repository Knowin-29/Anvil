#!/bin/bash
# Stage 0: Data Preparation - Build ms-swift data from registry + extract log
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env

echo "=== [$(date)] Stage 0: Data Preparation ==="
echo "Registry: ${GSA_FORGE_ROOT}/data/sources_registry.yaml"
echo "Output:   ${GSA_FORGE_ROOT}/trains/"
echo ""

# Step 1: Build SFT/GRPO/OPD from registry (msswift_ready sources)
echo "--- Step 1: build_msswift_from_registry.py ---"
python trains/build_msswift_from_registry.py \
    --registry data/sources_registry.yaml \
    --output_dir trains \
    --grpo_ratio 0.3 \
    --opd_ratio 0.2 \
    --skip_raw \
    --seed 42

echo ""

# Step 2: Extract log data (sqlite -> ms-swift)
echo "--- Step 2: extract_log_to_swift.py ---"
python trains/extract_log_to_swift.py \
    --output_dir trains \
    --min_completion_tokens 30 \
    --test_days_ratio 0.1

echo ""
echo "=== Data Stats ==="
wc -l trains/gsa_sft_train.jsonl trains/gsa_grpo_train.jsonl trains/gsa_opd_train.jsonl trains/gsa_log_sft.jsonl 2>/dev/null || true

echo ""
echo "--- Step 3: benchmark leakage gate (train <-> eval) ---"
# Non-fatal gate: reports overlap between the training pool and eval benchmarks.
python trains/check_benchmark_leakage.py \
    --train trains/gsa_sft_train.jsonl \
    --bench eval/replay_100.jsonl eval/replay_strat.jsonl \
    --out trains/leakage_report.json \
    --hash both || echo "WARNING: leakage gate reported overlap (see trains/leakage_report.json)"

echo ""
echo "=== [$(date)] Stage 0 Complete ==="
