#!/usr/bin/env bash
# Held-out formal40 G -> S evaluation on exactly four physical L20 GPUs.
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

EVALUATOR="$W/eval/scannet_rgb_depth_v51.py"
RUNNER="$W/eval/scannet_rgb_depth_v51_dlc.py"
RELEASE_MANIFEST="${RELEASE_MANIFEST:-$W/artifacts/depth_v51_eval_interface/formal40/release.manifest.json}"
MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
MODEL_ROLE="${MODEL_ROLE:-evaluated}"
OUT="${OUT:-$W/output/scannet_rgb_depth_v51_formal40_eval}"
CELL_TIMEOUT="${CELL_TIMEOUT:-2h}"
BATCH="${BATCH:-4}"
G_MAX_TOKENS="${G_MAX_TOKENS:-96}"
S_MAX_TOKENS="${S_MAX_TOKENS:-16}"

[[ "$BATCH" == 4 ]] || { echo "FATAL: frozen protocol requires BATCH=4" >&2; exit 2; }
[[ "$G_MAX_TOKENS" == 96 ]] || {
  echo "FATAL: frozen G protocol requires G_MAX_TOKENS=96" >&2
  exit 2
}
[[ "$S_MAX_TOKENS" == 16 ]] || {
  echo "FATAL: frozen S protocol requires S_MAX_TOKENS=16" >&2
  exit 2
}
case "${THINKING:-off}" in
  off|0|false) ;;
  *) echo "FATAL: frozen protocol requires THINKING=off" >&2; exit 2 ;;
esac

mkdir -p "$OUT"/{model_binding,g,s_oracle,s_pred,logs}
INPUT_RECEIPT="$OUT/formal40_inputs.json"
MODEL_BINDING="$OUT/model_binding/model.json"

"$PYTHON_BIN" "$RUNNER" preflight \
  --release-manifest "$RELEASE_MANIFEST" --out "$INPUT_RECEIPT"
"$PYTHON_BIN" "$RUNNER" bind-model \
  --role "$MODEL_ROLE" --model "$MODEL" \
  --input-receipt "$INPUT_RECEIPT" --out "$MODEL_BINDING"

echo "=================================================="
echo " protocol        : gsa-depth-v5.1-formal40-paired-gs-dlc-v1"
echo " release         : formal40 / 40 scenes / held-out only"
echo " model           : $MODEL"
echo " stages          : G -> S-oracle -> S-pred"
echo " generation      : 4x physical L20, stride-4, thinking off, temp=0"
echo " token budgets   : G=96, S=16"
echo " output          : $OUT"
echo "=================================================="

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "SCANNET_RGB_DEPTH_V51_PREFLIGHT_READY: no GPU query or process was started"
  exit 0
fi

mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF' | sort -u
)
if [[ "${#GPU_PIDS[@]}" -gt 0 ]]; then
  echo "FATAL: visible node already has GPU compute processes: ${GPU_PIDS[*]}" >&2
  exit 3
fi

"$PYTHON_BIN" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"FATAL: formal40 evaluation requires exactly 4 visible GPUs, found {count}")
uuids = []
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    name = props.name
    gib = props.total_memory / 2**30
    uuid = str(getattr(props, "uuid", ""))
    if "L20" not in name.upper() or gib < 44:
        raise SystemExit(f"FATAL: GPU {index} is not a >=44 GiB L20: {name} ({gib:.1f} GiB)")
    if "MIG" in name.upper() or uuid.upper().startswith("MIG"):
        raise SystemExit(f"FATAL: GPU {index} is a MIG slice, not a physical L20")
    if not uuid:
        raise SystemExit(f"FATAL: GPU {index} exposes no physical UUID")
    uuids.append(uuid)
if len(set(uuids)) != 4:
    raise SystemExit("FATAL: visible CUDA devices do not resolve to four unique physical GPUs")
print("4x physical L20 hardware gate: READY")
PY

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_IDS=(0 1 2 3)
fi
[[ "${#GPU_IDS[@]}" == 4 ]] || {
  echo "FATAL: expected exactly four visible GPU identifiers" >&2
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

run_stage() {
  local alias="$1" track="$2" max_tokens="$3" dynamic_manifest="${4:-}"
  local stage_dir="$OUT/$alias"
  local merged="$stage_dir/predictions.jsonl"
  local merged_manifest="$stage_dir/predictions.manifest.json"
  local rc=0
  local extra=()
  mkdir -p "$stage_dir"
  if [[ -n "$dynamic_manifest" ]]; then
    extra=(--dynamic-request-manifest "$dynamic_manifest")
  fi
  CURRENT_PIDS=()
  for shard in 0 1 2 3; do
    local shard_out="$stage_dir/shard_${shard}.jsonl"
    if [[ -e "$shard_out" ]]; then
      "$PYTHON_BIN" "$RUNNER" verify-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
        --track "$track" "${extra[@]}" \
        --shard "$shard_out" --expected-shard "$shard" >/dev/null
      echo "[$alias] reusing verified shard $shard"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" timeout "$CELL_TIMEOUT" \
      "$PYTHON_BIN" "$RUNNER" infer-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
        --track "$track" "${extra[@]}" \
        --shard "$shard" --num-shards 4 --batch-size "$BATCH" \
        --max-tokens "$max_tokens" --out "$shard_out" \
        >"$OUT/logs/${alias}_shard_${shard}.log" 2>&1 &
    CURRENT_PIDS+=("$!")
  done
  for pid in "${CURRENT_PIDS[@]:-}"; do wait "$pid" || rc=1; done
  CURRENT_PIDS=()
  [[ "$rc" == 0 ]] || {
    echo "FATAL: $alias inference failed; inspect $OUT/logs/${alias}_shard_*.log" >&2
    return 1
  }
  local shards=("$stage_dir"/shard_0.jsonl "$stage_dir"/shard_1.jsonl \
                "$stage_dir"/shard_2.jsonl "$stage_dir"/shard_3.jsonl)
  "$PYTHON_BIN" "$RUNNER" merge-shards \
    --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
    --track "$track" "${extra[@]}" \
    --predictions "${shards[@]}" --out "$merged" --manifest-out "$merged_manifest"
}

run_stage g g "$G_MAX_TOKENS"
run_stage s_oracle s-oracle "$S_MAX_TOKENS"

S_PRED_REQUESTS="$OUT/s_pred/requests.jsonl"
S_PRED_REQUEST_MANIFEST="$OUT/s_pred/requests.manifest.json"
"$PYTHON_BIN" "$RUNNER" build-s-pred-requests \
  --model-binding "$MODEL_BINDING" \
  --g-manifest "$OUT/g/predictions.manifest.json" \
  --g-predictions "$OUT/g/predictions.jsonl" \
  --out "$S_PRED_REQUESTS" --manifest-out "$S_PRED_REQUEST_MANIFEST"

run_stage s_pred s-pred "$S_MAX_TOKENS" "$S_PRED_REQUEST_MANIFEST"

RESULT="$OUT/result.json"
"$PYTHON_BIN" "$EVALUATOR" score \
  --release-manifest "$RELEASE_MANIFEST" \
  --model-binding "$MODEL_BINDING" \
  --g-predictions "$OUT/g/predictions.jsonl" \
  --g-manifest "$OUT/g/predictions.manifest.json" \
  --s-oracle-predictions "$OUT/s_oracle/predictions.jsonl" \
  --s-oracle-manifest "$OUT/s_oracle/predictions.manifest.json" \
  --s-pred-request-manifest "$S_PRED_REQUEST_MANIFEST" \
  --s-pred-predictions "$OUT/s_pred/predictions.jsonl" \
  --s-pred-manifest "$OUT/s_pred/predictions.manifest.json" \
  --out "$RESULT"

trap - EXIT INT TERM
echo "=================================================="
echo " completed       : $RESULT"
echo " logs            : $OUT/logs"
echo " training use    : prohibited"
echo "=================================================="
