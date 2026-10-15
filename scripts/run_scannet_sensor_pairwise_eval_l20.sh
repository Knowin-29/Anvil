#!/usr/bin/env bash
# Formal full_v3 ScanNet sensor-pair depth evaluation on exactly 8xL20.
# Runs the frozen base model and, when MERGED_MODEL is set, one already-merged model.
# Adapter checkpoints are rejected; merge them before invoking this launcher.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || {
  echo "FATAL: required environment file is missing: $W/.env" >&2
  exit 2
}
# shellcheck disable=SC1091
source "$W/.env"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_ENV="/path/to/environment/env/gsa_train"
PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
[[ "$PYTHON_PREFIX" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: .env did not activate the required evaluation environment" >&2
  echo "  expected sys.prefix: $EXPECTED_ENV" >&2
  echo "  actual sys.prefix  : $PYTHON_PREFIX" >&2
  exit 2
}
SWIFT_BIN=$(command -v swift || true)
[[ "$SWIFT_BIN" == "$EXPECTED_ENV/bin/swift" ]] || {
  echo "FATAL: swift does not resolve inside the required environment: ${SWIFT_BIN:-MISSING}" >&2
  exit 2
}

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs"

RUNNER="$W/eval/scannet_sensor_pairwise_dlc.py"
SCORER="$W/eval/scannet_sensor_pairwise.py"
REQUEST_MANIFEST="${REQUEST_MANIFEST:-$W/artifacts/depth_v50_eval_interface/full_v3/s_oracle_requests.manifest.json}"
REQUESTS="${REQUESTS:-$W/artifacts/depth_v50_eval_interface/full_v3/s_oracle_requests.jsonl}"
SOURCE_LABELS="$W/data/depth_v50_scannet_sensor/full_v3/labels/S_spatial/depth.jsonl"
SOURCE_REPORT="$W/data/depth_v50_scannet_sensor/full_v3/generation_report.json"
BASE_MODEL="${BASE_MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
MERGED_MODEL="${MERGED_MODEL:-}"
OUT="${OUT:-$W/output/scannet_sensor_pairwise_full_v3_eval_v3}"
CELL_TIMEOUT="${CELL_TIMEOUT:-2h}"
BATCH="${BATCH:-4}"
MAX_TOKENS="${MAX_TOKENS:-16}"

[[ "$BATCH" == 4 ]] || { echo "FATAL: frozen protocol requires BATCH=4" >&2; exit 2; }
[[ "$MAX_TOKENS" == 16 ]] || {
  echo "FATAL: frozen protocol requires MAX_TOKENS=16" >&2
  exit 2
}
case "${THINKING:-off}" in
  off|0|false) ;;
  *) echo "FATAL: frozen protocol requires THINKING=off" >&2; exit 2 ;;
esac
if [[ -n "$MERGED_MODEL" ]]; then
  [[ "$(realpath -m "$MERGED_MODEL")" != "$(realpath -m "$BASE_MODEL")" ]] || {
    echo "FATAL: MERGED_MODEL must differ from BASE_MODEL" >&2
    exit 2
  }
fi

mkdir -p "$OUT"/{model_bindings,predictions,results,logs}
INPUT_RECEIPT="$OUT/scannet_sensor_pairwise_full_v3_inputs.json"
"$PYTHON_BIN" "$RUNNER" preflight \
  --request-manifest "$REQUEST_MANIFEST" --requests "$REQUESTS" \
  --out "$INPUT_RECEIPT"

BASE_BINDING="$OUT/model_bindings/base.json"
"$PYTHON_BIN" "$RUNNER" bind-model --role base --model "$BASE_MODEL" \
  --input-receipt "$INPUT_RECEIPT" --out "$BASE_BINDING"
MERGED_BINDING=""
if [[ -n "$MERGED_MODEL" ]]; then
  MERGED_BINDING="$OUT/model_bindings/merged.json"
  "$PYTHON_BIN" "$RUNNER" bind-model --role merged --model "$MERGED_MODEL" \
    --input-receipt "$INPUT_RECEIPT" --out "$MERGED_BINDING"
fi

echo "=================================================="
echo " protocol        : gsa-depth-v5.0-oracle-pairwise-dlc-v1"
echo " request release : full_v3 / 120 pairs / A=60 B=60"
echo " base model      : $BASE_MODEL"
echo " merged model    : ${MERGED_MODEL:-not supplied}"
echo " generation      : 8xL20, stride shards, thinking off, temp=0, max_tokens=16"
echo " output          : $OUT"
echo "=================================================="
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "SCANNET_PAIRWISE_EVAL_PREFLIGHT_READY: no GPU process was started"
  exit 0
fi

mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF' | sort -u
)
if [[ "${#GPU_PIDS[@]}" -gt 0 && "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
  echo "FATAL: visible GPUs already have compute processes: ${GPU_PIDS[*]}" >&2
  exit 3
fi
"$PYTHON_BIN" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"FATAL: formal pairwise evaluation requires exactly 8 GPUs, found {count}")
for index in range(count):
    name = torch.cuda.get_device_name(index)
    gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    if "L20" not in name.upper() or gib < 44:
        raise SystemExit(f"FATAL: GPU {index} is not a >=44 GiB L20: {name} ({gib:.1f} GiB)")
print("8xL20 hardware gate: READY")
PY

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_IDS=(0 1 2 3 4 5 6 7)
fi
[[ "${#GPU_IDS[@]}" == 8 ]] || {
  echo "FATAL: expected exactly 8 visible GPU ids" >&2
  exit 2
}

CURRENT_PIDS=()
cleanup_workers() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${CURRENT_PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${CURRENT_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup_workers EXIT INT TERM

run_model() {
  local role="$1" binding="$2"
  local pred_dir="$OUT/predictions/$role" result="$OUT/results/$role.json"
  local merged="$pred_dir/predictions.jsonl"
  local merged_manifest="$pred_dir/predictions.manifest.json"
  local rc=0
  mkdir -p "$pred_dir"
  CURRENT_PIDS=()
  for ((shard = 0; shard < 8; shard++)); do
    local shard_out="$pred_dir/shard_${shard}.jsonl"
    if [[ -e "$shard_out" ]]; then
      "$PYTHON_BIN" "$RUNNER" verify-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
        --shard "$shard_out" --expected-shard "$shard" >/dev/null
      echo "[$role] reusing verified shard $shard"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" timeout "$CELL_TIMEOUT" \
      "$PYTHON_BIN" "$RUNNER" infer-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
        --shard "$shard" --num-shards 8 --batch-size "$BATCH" \
        --max-tokens "$MAX_TOKENS" --out "$shard_out" \
        >"$OUT/logs/${role}_shard_${shard}.log" 2>&1 &
    CURRENT_PIDS+=("$!")
  done
  for pid in "${CURRENT_PIDS[@]:-}"; do wait "$pid" || rc=1; done
  CURRENT_PIDS=()
  [[ "$rc" == 0 ]] || {
    echo "FATAL: $role inference failed; inspect $OUT/logs/${role}_shard_*.log" >&2
    return 1
  }
  local shards=("$pred_dir"/shard_*.jsonl)
  [[ "${#shards[@]}" == 8 ]] || {
    echo "FATAL: $role did not produce eight shards" >&2
    return 2
  }
  "$PYTHON_BIN" "$RUNNER" merge-shards \
    --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
    --predictions "${shards[@]}" --out "$merged" \
    --manifest-out "$merged_manifest"
  "$PYTHON_BIN" "$SCORER" score \
    --source "$SOURCE_LABELS" --source-report "$SOURCE_REPORT" \
    --track s-oracle --predictions "$merged" \
    --prediction-manifest "$merged_manifest" --model-binding "$binding" \
    --out "$result"
  echo "[$role] trusted result: $result"
}

run_model base "$BASE_BINDING"
if [[ -n "$MERGED_BINDING" ]]; then
  run_model merged "$MERGED_BINDING"
fi
trap - EXIT INT TERM

echo "=================================================="
echo " completed: $OUT/results"
echo " each result is bound to the full_v3 requests and a fully hashed model inventory"
echo "=================================================="
