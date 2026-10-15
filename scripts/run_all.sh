#!/usr/bin/env bash
# Rex-bench-v1: run the full experiment pipeline end-to-end.
# Usage: bash scripts/run_all.sh [BENCH_PATH] [ROUND_SPEC...]
#
# If BENCH_PATH is not given, defaults to data/bench/bench_v1_from_round_0.json.
# All outputs go under data/bench/.

set -euo pipefail
cd "$(dirname "$0")/.."

BENCH="${1:-data/bench/bench_v1_from_round_0.json}"
PYTHON="${PYTHON:-python}"
DEFAULT_ROUND_SPEC="round_0:/path/to/storage/anvil/rex-embodied-dev/data/output/flywheel/round_0"

if [[ "$#" -ge 2 ]]; then
  ROUND_SPECS=("${@:2}")
elif [[ -n "${ROUND_SPECS:-}" ]]; then
  read -r -a ROUND_SPECS <<<"${ROUND_SPECS}"
else
  ROUND_SPECS=("${DEFAULT_ROUND_SPEC}")
fi

echo "=== Rex-bench-v1 full pipeline ==="
echo "bench: $BENCH"
echo "python: $PYTHON"
echo "round specs: ${ROUND_SPECS[*]}"
echo ""

# --- Stage 1: Verifier scoring ---
echo "[1/8] Running verifier on all samples..."
$PYTHON -m benchmark.scripts.run_verifier \
  --bench "$BENCH" \
  --out data/bench/verifier_scores_round_0.jsonl

# --- Stage 2: Scaling ---
echo "[2/8] Scaling experiment..."
$PYTHON -m benchmark.scripts.exp_scaling \
  --rounds "${ROUND_SPECS[@]}" \
  --out data/bench/scaling_round_0.json

# --- Stage 3: RFT smoke ---
echo "[3/8] RFT reward smoke test (n=120, 4 rollouts)..."
$PYTHON -m benchmark.scripts.exp_rft_smoke \
  --bench "$BENCH" \
  --rollouts-per-sample 4 --max-samples 120 \
  --out data/bench/rft_smoke_round_0_n120.json

# --- Stage 4: Case gallery ---
echo "[4/8] Case gallery (perturb mode)..."
$PYTHON -m benchmark.scripts.exp_cases \
  --bench "$BENCH" \
  --per-task 10 --mode perturb \
  --out data/bench/cases_round_0.jsonl

# --- Stage 5: PRM data + training ---
echo "[5/8] PRM dataset generation + training..."
$PYTHON -m benchmark.scripts.exp_prm_data \
  --bench "$BENCH" \
  --per-task 50 \
  --out data/bench/prm_dataset_round_0.jsonl \
  --feature-config data/bench/prm_feature_config.json

$PYTHON -m benchmark.scripts.exp_prm_train \
  --data data/bench/prm_dataset_round_0.jsonl \
  --feature-config data/bench/prm_feature_config.json \
  --out-model data/bench/prm_model_round_0.json \
  --out-history data/bench/prm_history_round_0.json \
  --kind mlp --hidden 32 --epochs 300

# --- Stage 6: GRPO dry-run (base + PRM) ---
echo "[6/8] GRPO dry-run (priority-weighted)..."
$PYTHON -m benchmark.scripts.train_grpo \
  --bench "$BENCH" \
  --rollouts 4 \
  --out-summary data/bench/grpo_dryrun_summary.json \
  --out-report data/bench/grpo_dryrun_report.jsonl

echo "[6b/8] GRPO dry-run (PRM-weighted)..."
$PYTHON -m benchmark.scripts.train_grpo \
  --bench "$BENCH" \
  --rollouts 4 \
  --prm-model data/bench/prm_model_round_0.json \
  --prm-feature-config data/bench/prm_feature_config.json \
  --prm-blend 1.0 \
  --out-summary data/bench/grpo_dryrun_prm_summary.json \
  --out-report data/bench/grpo_dryrun_prm_report.jsonl

# --- Stage 7: Individual visualizations ---
echo "[7/8] Rendering individual dashboards..."
$PYTHON -m benchmark.scripts.viz_bench \
  --bench "$BENCH" --out data/bench/viz/bench_round_0.html
$PYTHON -m benchmark.scripts.viz_scaling \
  --report data/bench/scaling_round_0.json --out data/bench/viz/scaling_round_0.html
$PYTHON -m benchmark.scripts.viz_verifier \
  --scores data/bench/verifier_scores_round_0.jsonl --out data/bench/viz/verifier_round_0.html
$PYTHON -m benchmark.scripts.viz_prm \
  --history data/bench/prm_history_round_0.json \
  --model data/bench/prm_model_round_0.json \
  --data data/bench/prm_dataset_round_0.jsonl \
  --out data/bench/viz/prm_round_0.html
$PYTHON -m benchmark.scripts.viz_rft \
  --report data/bench/rft_smoke_round_0_n120.json --out data/bench/viz/rft_smoke_round_0_n120.html
$PYTHON -m benchmark.scripts.viz_grpo \
  --summary data/bench/grpo_dryrun_summary.json \
  --report data/bench/grpo_dryrun_report.jsonl \
  --compare-summary data/bench/grpo_dryrun_prm_summary.json \
  --compare-report data/bench/grpo_dryrun_prm_report.jsonl \
  --out data/bench/viz/grpo_round_0.html
$PYTHON -m benchmark.scripts.viz_cases \
  --cases data/bench/cases_round_0.jsonl --out data/bench/viz/cases_round_0.html

# --- Stage 8: Master dashboard ---
echo "[8/8] Rendering master dashboard (7 tabs)..."
$PYTHON -m benchmark.scripts.viz_index \
  --bench "$BENCH" \
  --scaling-report data/bench/scaling_round_0.json \
  --verifier-scores data/bench/verifier_scores_round_0.jsonl \
  --prm-history data/bench/prm_history_round_0.json \
  --prm-model data/bench/prm_model_round_0.json \
  --prm-data data/bench/prm_dataset_round_0.jsonl \
  --rft-report data/bench/rft_smoke_round_0_n120.json \
  --grpo-summary data/bench/grpo_dryrun_summary.json \
  --grpo-report data/bench/grpo_dryrun_report.jsonl \
  --grpo-compare-summary data/bench/grpo_dryrun_prm_summary.json \
  --grpo-compare-report data/bench/grpo_dryrun_prm_report.jsonl \
  --cases data/bench/cases_round_0.jsonl \
  --out data/bench/viz/index.html

echo ""
echo "=== Done! ==="
echo "Master dashboard: data/bench/viz/index.html"
echo "Tests: python -m pytest benchmark/tests -q"
