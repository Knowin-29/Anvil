#!/usr/bin/env bash
# Formal five-track unique-referent G -> S intervention on four physical L20 GPUs.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
if [[ -r "$W/.env" ]]; then
  # shellcheck disable=SC1091
  source "$W/.env"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PACK="${V52_PACK:-$W/data/semantic_audit_v52_scannet_unique_referent_gs_formal}"
FORMAL_REPORT="${V52_FORMAL_REPORT:-$W/artifacts/audits/scannet_rgb_depth_v52_formal_live.json}"
MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
MODEL_ROLE="${MODEL_ROLE:-evaluated}"
OUT="${OUT:-$W/output/scannet_rgb_depth_v52_formal40_eval}"
RELEASE="${V52_RELEASE:-$OUT/release}"
CELL_TIMEOUT="${CELL_TIMEOUT:-45m}"
BATCH="${BATCH:-4}"
G_MAX_TOKENS="${G_MAX_TOKENS:-64}"
S_MAX_TOKENS="${S_MAX_TOKENS:-8}"
GPU_SET="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_SET"
[[ "${#GPU_IDS[@]}" == 4 ]] || {
  echo "FATAL: expected exactly four visible GPU identifiers"
  exit 3
}
[[ "$(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l)" == 4 ]] || {
  echo "FATAL: duplicate visible GPU identifiers"
  exit 3
}
export CUDA_VISIBLE_DEVICES="$GPU_SET"
EVALUATOR="$W/eval/scannet_rgb_depth_v52_eval.py"
RUNNER="$W/eval/scannet_rgb_depth_v52_dlc.py"

mkdir -p "$OUT/logs"
PIPELINE_LOG="$OUT/logs/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
echo "$(timestamp) V52_PAIRED PREFLIGHT_START"

[[ -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]] || {
  echo "FATAL: PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 2
}
[[ -d "$PACK" ]] || { echo "FATAL: v5.2 pack is missing: $PACK"; exit 2; }
[[ -f "$EVALUATOR" && -f "$RUNNER" ]] || {
  echo "FATAL: v5.2 evaluator implementation is incomplete"
  exit 2
}
[[ "$BATCH" == 4 ]] || { echo "FATAL: frozen protocol requires BATCH=4"; exit 2; }
[[ "$G_MAX_TOKENS" == 64 ]] || {
  echo "FATAL: frozen protocol requires G_MAX_TOKENS=64"
  exit 2
}
[[ "$S_MAX_TOKENS" == 8 ]] || {
  echo "FATAL: frozen protocol requires S_MAX_TOKENS=8"
  exit 2
}
case "${THINKING:-off}" in
  off|0|false) ;;
  *) echo "FATAL: frozen protocol requires THINKING=off"; exit 2 ;;
esac

export HF_HOME="${HF_HOME:-$W/.cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$W/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$W/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$W/.cache}"
export TMPDIR="${TMPDIR:-$W/.cache/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$W/.cache/mpl}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM=4096
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$TMPDIR" "$MPLCONFIGDIR"

GATE_ARGS=(gate --pack "$PACK" --report "$FORMAL_REPORT")
if [[ -n "${V52_ADJUDICATION:-}" ]]; then
  GATE_ARGS+=(--adjudication "$V52_ADJUDICATION")
fi
set +e
"$PYTHON_BIN" "$EVALUATOR" "${GATE_ARGS[@]}" \
  >"$OUT/logs/formal_gate.log" 2>&1
gate_rc=$?
set -e
cat "$OUT/logs/formal_gate.log"
if [[ "$gate_rc" == 20 ]]; then
  echo "$(timestamp) BLOCKED_PENDING_HUMAN_REVIEW gpu_started=false"
  echo "formal gate log: $OUT/logs/formal_gate.log"
  exit 20
fi
[[ "$gate_rc" == 0 ]] || {
  echo "FATAL: v5.2 formal gate failed with rc=$gate_rc"
  exit "$gate_rc"
}

"$PYTHON_BIN" "$EVALUATOR" build-release \
  --pack "$PACK" --report "$FORMAL_REPORT" --out "$RELEASE" \
  >"$OUT/logs/release.log" 2>&1
RELEASE_MANIFEST="$RELEASE/release.manifest.json"
"$PYTHON_BIN" "$RUNNER" dry-run >"$OUT/logs/dlc_contract_dry_run.log" 2>&1

mkdir -p "$OUT"/{model_binding,g,s_oracle,s_absent,s_corrupted,s_pred}
INPUT_RECEIPT="$OUT/formal40_inputs.json"
MODEL_BINDING="$OUT/model_binding/model.json"
"$PYTHON_BIN" "$RUNNER" preflight \
  --release-manifest "$RELEASE_MANIFEST" --out "$INPUT_RECEIPT"
"$PYTHON_BIN" "$RUNNER" bind-model \
  --role "$MODEL_ROLE" --model "$MODEL" \
  --input-receipt "$INPUT_RECEIPT" --out "$MODEL_BINDING"

echo "=================================================="
echo " protocol        : gsa-depth-v5.2-unique-referent-formal40-gs-intervention-dlc-v4"
echo " release         : unique-referent formal40 / held-out ScanNet val"
echo " model           : $MODEL"
echo " stages          : G -> S-oracle/S-absent/S-corrupted -> S-pred"
echo " generation      : 4 physical L20, stride-4, thinking off, temperature 0"
echo " output          : $OUT"
echo " positive log    : $PIPELINE_LOG"
echo " result          : $OUT/result.json"
echo " estimated GPU   : 25-50 minutes for five sequential 4-replica loads"
echo "=================================================="

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "$(timestamp) SCANNET_RGB_DEPTH_V52_PREFLIGHT_READY no_cuda_query=true gpu_started=false"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi is unavailable"; exit 3; }
command -v setsid >/dev/null || { echo "FATAL: setsid is unavailable"; exit 3; }
GPU_QUERY_ERROR="$OUT/logs/nvidia_smi_compute_query.err"
if ! GPU_QUERY_RAW=$(nvidia-smi --id="$GPU_SET" --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>"$GPU_QUERY_ERROR"); then
  echo "FATAL: nvidia-smi compute-process query failed; refusing to assume GPUs are idle"
  [[ ! -s "$GPU_QUERY_ERROR" ]] || cat "$GPU_QUERY_ERROR"
  exit 3
fi
mapfile -t GPU_PIDS < <(printf '%s\n' "$GPU_QUERY_RAW" | awk 'NF' | sort -u)
if [[ "${#GPU_PIDS[@]}" -gt 0 ]]; then
  echo "FATAL: visible node already has GPU compute processes: ${GPU_PIDS[*]}"
  exit 3
fi

"$PYTHON_BIN" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"FATAL: v5.2 formal evaluation requires exactly 4 visible GPUs, found {count}")
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

WORKER_GROUPS=()
cleanup_workers() {
  local status=$?
  trap - EXIT INT TERM
  for pgid in "${WORKER_GROUPS[@]:-}"; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done
  for pgid in "${WORKER_GROUPS[@]:-}"; do
    wait "$pgid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup_workers EXIT INT TERM

run_stage() {
  local alias="$1" track="$2" max_tokens="$3" dynamic_manifest="${4:-}"
  local stage_dir="$OUT/$alias"
  local merged="$stage_dir/predictions.jsonl"
  local merged_manifest="$stage_dir/predictions.manifest.json"
  local extra=() rc=0
  mkdir -p "$stage_dir"
  if [[ -n "$dynamic_manifest" ]]; then
    extra=(--dynamic-request-manifest "$dynamic_manifest")
  fi
  WORKER_GROUPS=()
  echo "$(timestamp) $alias STARTING"
  for shard in 0 1 2 3; do
    local shard_out="$stage_dir/shard_${shard}.jsonl"
    if [[ -e "$shard_out" ]]; then
      "$PYTHON_BIN" "$RUNNER" verify-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
        --track "$track" "${extra[@]}" \
        --shard "$shard_out" --expected-shard "$shard" >/dev/null
      echo "$(timestamp) $alias shard=$shard REUSED_VERIFIED"
      continue
    fi
    setsid env CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" \
      timeout --signal=TERM --kill-after=2m "$CELL_TIMEOUT" \
      "$PYTHON_BIN" "$RUNNER" infer-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
        --track "$track" "${extra[@]}" \
        --shard "$shard" --num-shards 4 --batch-size "$BATCH" \
        --max-tokens "$max_tokens" --out "$shard_out" \
        >"$OUT/logs/${alias}_shard_${shard}.log" 2>&1 &
    WORKER_GROUPS+=("$!")
  done
  for pgid in "${WORKER_GROUPS[@]:-}"; do
    wait "$pgid" || rc=1
  done
  WORKER_GROUPS=()
  if [[ "$rc" != 0 ]]; then
    echo "FATAL: $alias failed; inspect $OUT/logs/${alias}_shard_*.log"
    for log in "$OUT/logs/${alias}_shard_"*.log; do
      [[ -f "$log" ]] && { echo "--- $log (tail)"; tail -n 30 "$log"; }
    done
    return 1
  fi
  local shards=(
    "$stage_dir/shard_0.jsonl" "$stage_dir/shard_1.jsonl"
    "$stage_dir/shard_2.jsonl" "$stage_dir/shard_3.jsonl"
  )
  "$PYTHON_BIN" "$RUNNER" merge-shards \
    --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
    --track "$track" "${extra[@]}" \
    --predictions "${shards[@]}" --out "$merged" --manifest-out "$merged_manifest"
  echo "$(timestamp) $alias FORMAL_COMPLETE predictions=$merged"
}

run_stage g g "$G_MAX_TOKENS"
run_stage s_oracle s-oracle "$S_MAX_TOKENS"
run_stage s_absent s-absent "$S_MAX_TOKENS"
run_stage s_corrupted s-corrupted "$S_MAX_TOKENS"

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
  --s-absent-predictions "$OUT/s_absent/predictions.jsonl" \
  --s-absent-manifest "$OUT/s_absent/predictions.manifest.json" \
  --s-corrupted-predictions "$OUT/s_corrupted/predictions.jsonl" \
  --s-corrupted-manifest "$OUT/s_corrupted/predictions.manifest.json" \
  --s-pred-request-manifest "$S_PRED_REQUEST_MANIFEST" \
  --s-pred-predictions "$OUT/s_pred/predictions.jsonl" \
  --s-pred-manifest "$OUT/s_pred/predictions.manifest.json" \
  --out "$RESULT"

trap - EXIT INT TERM
echo "$(timestamp) V52_PAIRED FORMAL_COMPLETE result=$RESULT"
echo "positive log: $PIPELINE_LOG"
