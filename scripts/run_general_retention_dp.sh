#!/usr/bin/env bash
# Paired base/E1 general-retention-v3 evaluation on exactly four physical L20 GPUs.
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
EXPECTED_ENV="${EXPECTED_ENV:-${GSA_PYTHON_PREFIX:-}}"
PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
if [[ -n "$EXPECTED_ENV" && "$PYTHON_PREFIX" != "$EXPECTED_ENV" ]]; then
  echo "FATAL: .env did not activate the required evaluation environment" >&2
  echo "  expected sys.prefix: $EXPECTED_ENV" >&2
  echo "  actual sys.prefix  : $PYTHON_PREFIX" >&2
  exit 2
fi
SWIFT_BIN=$(command -v swift || true)
[[ -n "$SWIFT_BIN" ]] || {
  echo "FATAL: swift is not available in the activated environment" >&2
  exit 2
}
if [[ -n "$EXPECTED_ENV" && "$SWIFT_BIN" != "$EXPECTED_ENV/bin/swift" ]]; then
  echo "FATAL: swift does not resolve inside the required environment: ${SWIFT_BIN:-MISSING}" >&2
  exit 2
fi

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=4096
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs"

RUNNER="$W/eval/general_retention_formal_v3.py"
CONFIG="$W/configs/eval/general_retention_formal_v3.yaml"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
BASE_MODEL="${BASE_MODEL:-${MODEL_9B:-}}"
E1_MODEL="${E1_MODEL:-${CANDIDATE_MODEL:-}}"
OUT="${OUT:-$W/output/general_retention_formal_v3_base_e1}"
CELL_TIMEOUT="${CELL_TIMEOUT:-8h}"
BATCH="${BATCH:-16}"
RUN_STAGE="${RUN_STAGE:-paired}"
RESUME="${RESUME:-0}"
NUM_SHARDS=4

[[ "$BATCH" == 16 ]] || {
  echo "FATAL: frozen protocol requires max engine BATCH=16 per GPU" >&2
  exit 2
}
case "${THINKING:-off}" in
  off|0|false) ;;
  *) echo "FATAL: frozen protocol requires THINKING=off" >&2; exit 2 ;;
esac
case "$RUN_STAGE" in
  base|e1|paired) ;;
  *) echo "FATAL: RUN_STAGE must be base, e1, or paired" >&2; exit 2 ;;
esac
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || {
  echo "FATAL: RESUME must be 0 or 1" >&2
  exit 2
}
[[ -n "$BASE_MODEL" && -n "$E1_MODEL" && -d "$BASE_MODEL" && -d "$E1_MODEL" ]] || {
  echo "FATAL: BASE_MODEL and E1_MODEL must both be local model directories" >&2
  exit 2
}
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "FATAL: CUDA_VISIBLE_DEVICES must explicitly select exactly four GPUs" >&2
  exit 2
}
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#GPU_IDS[@]}" == 4 ]] || {
  echo "FATAL: expected exactly four visible GPU identifiers" >&2
  exit 2
}
UNIQUE_GPU_IDS=$(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l)
[[ "$UNIQUE_GPU_IDS" == 4 ]] || {
  echo "FATAL: CUDA_VISIBLE_DEVICES contains duplicate selectors" >&2
  exit 2
}

if [[ -e "$OUT" && "$RESUME" != 1 ]]; then
  echo "FATAL: fresh run refuses existing OUT (set RESUME=1 only for this exact v3 run): $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"/{inputs,model_bindings,predictions,results,logs,state}
STATE_FILE="$OUT/state/pipeline.json"
write_state() {
  "$PYTHON_BIN" - "$STATE_FILE" "$1" "$2" "${3:-}" <<'PY'
import datetime, json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
event = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "stage": sys.argv[2], "state": sys.argv[3], "detail": sys.argv[4],
}
payload = {"schema_version": 1, "protocol": "general-retention-formal-v3", "events": []}
if path.is_file():
    payload = json.loads(path.read_text(encoding="utf-8"))
payload["events"].append(event)
payload["latest"] = event
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}
early_exit_state() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$status" != 0 ]]; then
    write_state pipeline FAILED "pre-worker exit_code=$status" || true
  fi
  exit "$status"
}
trap early_exit_state EXIT INT TERM
write_state pipeline PREFLIGHT_START "resume=$RESUME run_stage=$RUN_STAGE"
INPUT_RECEIPT="$OUT/inputs/input_receipt.json"
RUNTIME_INDEX="$OUT/inputs/runtime_index.jsonl"
BASE_BINDING="$OUT/model_bindings/base.json"
E1_BINDING="$OUT/model_bindings/e1.json"

if [[ -e "$INPUT_RECEIPT" || -e "$RUNTIME_INDEX" ]]; then
  [[ -f "$INPUT_RECEIPT" && -f "$RUNTIME_INDEX" ]] || {
    echo "FATAL: incomplete preflight artifacts under $OUT/inputs" >&2
    exit 2
  }
  "$PYTHON_BIN" "$RUNNER" verify-receipt \
    --input-receipt "$INPUT_RECEIPT" --deep-sources \
    --live-hardware --require-idle-hardware \
    | tee "$OUT/logs/input_preflight.log"
else
  PREFLIGHT_ARGS=(
    preflight --config "$CONFIG" --runtime-index "$RUNTIME_INDEX"
    --nvidia-smi-bin "$NVIDIA_SMI_BIN" --out "$INPUT_RECEIPT"
  )
  for selector in "${GPU_IDS[@]}"; do
    PREFLIGHT_ARGS+=(--gpu-selector "$selector")
  done
  "$PYTHON_BIN" "$RUNNER" "${PREFLIGHT_ARGS[@]}" | tee "$OUT/logs/input_preflight.log"
fi

"$PYTHON_BIN" "$RUNNER" bind-model \
  --role base --model "$BASE_MODEL" --input-receipt "$INPUT_RECEIPT" --out "$BASE_BINDING" \
  | tee "$OUT/logs/base_binding.log"
"$PYTHON_BIN" "$RUNNER" bind-model \
  --role e1 --model "$E1_MODEL" --input-receipt "$INPUT_RECEIPT" --out "$E1_BINDING" \
  | tee "$OUT/logs/e1_binding.log"

BASE_ID=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["checkpoint_content_identity_sha256"])' "$BASE_BINDING")
E1_ID=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["checkpoint_content_identity_sha256"])' "$E1_BINDING")
[[ "$BASE_ID" != "$E1_ID" ]] || {
  echo "FATAL: base and E1 bindings resolve to byte-identical checkpoint content" >&2
  exit 2
}
"$PYTHON_BIN" - "$BASE_BINDING" "$E1_BINDING" <<'PY'
import json
import sys

base = json.load(open(sys.argv[1], encoding="utf-8"))["model"]
e1 = json.load(open(sys.argv[2], encoding="utf-8"))["model"]
for field in ("architecture", "model_type"):
    if base[field] != e1[field]:
        raise SystemExit(f"FATAL: base/E1 {field} differs: {base[field]} != {e1[field]}")
print(f"same-family model gate: READY ({base['architecture']}, {base['model_type']})")
PY
"$PYTHON_BIN" "$RUNNER" verify-receipt \
  --input-receipt "$INPUT_RECEIPT" --live-hardware --require-idle-hardware \
  | tee "$OUT/logs/hardware_final_gate.log"

echo "=================================================="
echo " protocol        : general-retention-formal-v3"
echo " official splits : AI2D_TEST=3088, ScienceQA_TEST-image=2017, TextVQA_VAL=5000"
echo " base model      : $BASE_MODEL"
echo " E1 model        : $E1_MODEL"
echo " run stage       : $RUN_STAGE"
echo " generation      : four concurrent single-GPU workers; AI2D/SQA batch16; TextVQA batch1"
echo " visual budget   : IMAGE_MAX_TOKEN_NUM=4096; CUDA OOM bisects to singleton then fails"
echo " output          : $OUT"
echo " score logs      : $OUT/logs/{base_score,e1_score,paired_compare}.log"
echo "=================================================="

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  write_state pipeline PREFLIGHT_READY "no CUDA context or model process started"
  echo "GENERAL_RETENTION_FORMAL_V3_PREFLIGHT_READY: sources, leakage, runner, scorers, both model inventories, and four idle physical L20s are bound/verified; no CUDA context or model process was started"
  exit 0
fi

"$PYTHON_BIN" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"FATAL: formal general-retention requires exactly 4 visible GPUs, found {count}")
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
    raise SystemExit("FATAL: visible devices do not resolve to four unique physical GPUs")
print("4x physical L20 hardware gate: READY")
PY

CURRENT_PIDS=()
CURRENT_PGIDS=()
stop_workers() {
  local pgid pid
  for pgid in "${CURRENT_PGIDS[@]:-}"; do
    [[ -n "$pgid" ]] && kill -TERM -- "-$pgid" 2>/dev/null || true
  done
  for _ in $(seq 1 20); do
    local alive=0
    for pgid in "${CURRENT_PGIDS[@]:-}"; do
      [[ -n "$pgid" ]] && kill -0 -- "-$pgid" 2>/dev/null && alive=1
    done
    [[ "$alive" == 0 ]] && break
    sleep 1
  done
  for pgid in "${CURRENT_PGIDS[@]:-}"; do
    [[ -n "$pgid" ]] && kill -KILL -- "-$pgid" 2>/dev/null || true
  done
  for pid in "${CURRENT_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  CURRENT_PIDS=()
  CURRENT_PGIDS=()
}
cleanup_workers() {
  local status=$?
  trap - EXIT INT TERM
  stop_workers
  write_state pipeline FAILED "exit_code=$status"
  exit "$status"
}
trap cleanup_workers EXIT INT TERM

run_role() {
  local role="$1" binding="$2"
  local pred_dir="$OUT/predictions/$role"
  local result="$OUT/results/$role.json"
  local rc=0
  mkdir -p "$pred_dir"
  if [[ -e "$result" ]]; then
    [[ "$RESUME" == 1 ]] || {
      echo "FATAL: refusing to overwrite completed $role result: $result" >&2
      return 2
    }
    "$PYTHON_BIN" "$RUNNER" verify-result --result "$result" --require-admitted \
      | tee "$OUT/logs/${role}_resume_verify.log"
    for shard in 0 1 2 3; do
      "$PYTHON_BIN" "$RUNNER" verify-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
        --shard "$pred_dir/shard_${shard}.jsonl" --expected-shard "$shard" >/dev/null
    done
    write_state "$role" REUSED_VERIFIED "result and all four shards recomputed"
    echo "[$role] reusing verified admitted result and four bound v3 shards"
    return 0
  fi
  write_state "$role" INFERENCE_START "four concurrent workers"
  CURRENT_PIDS=()
  CURRENT_PGIDS=()
  for shard in 0 1 2 3; do
    local shard_out="$pred_dir/shard_${shard}.jsonl"
    if [[ -e "$shard_out" ]]; then
      "$PYTHON_BIN" "$RUNNER" verify-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
        --shard "$shard_out" --expected-shard "$shard" >/dev/null
      echo "[$role] reusing verified shard $shard"
      continue
    fi
    setsid timeout --signal=TERM --kill-after=30s "$CELL_TIMEOUT" env \
      CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" IMAGE_MAX_TOKEN_NUM=4096 \
      "$PYTHON_BIN" "$RUNNER" infer-shard \
        --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
        --shard "$shard" --num-shards "$NUM_SHARDS" --batch-size "$BATCH" \
        --out "$shard_out" \
      >"$OUT/logs/${role}_shard_${shard}.log" 2>&1 &
    CURRENT_PIDS+=("$!")
    CURRENT_PGIDS+=("$!")
  done
  for pid in "${CURRENT_PIDS[@]:-}"; do
    if ! wait "$pid"; then rc=1; break; fi
  done
  [[ "$rc" == 0 ]] || stop_workers
  CURRENT_PIDS=()
  CURRENT_PGIDS=()
  [[ "$rc" == 0 ]] || {
    write_state "$role" FAILED "one or more shard workers exited nonzero"
    echo "FATAL: $role inference failed; inspect $OUT/logs/${role}_shard_*.log" >&2
    return 1
  }
  local shards=(
    "$pred_dir/shard_0.jsonl" "$pred_dir/shard_1.jsonl"
    "$pred_dir/shard_2.jsonl" "$pred_dir/shard_3.jsonl"
  )
  for shard in 0 1 2 3; do
    "$PYTHON_BIN" "$RUNNER" verify-shard \
      --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
      --shard "${shards[$shard]}" --expected-shard "$shard" \
      | tee "$OUT/logs/${role}_shard_${shard}_verify.log"
  done
  "$PYTHON_BIN" "$RUNNER" score \
    --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
    --predictions "${shards[@]}" --out "$result" \
    | tee "$OUT/logs/${role}_score.log"
  "$PYTHON_BIN" "$RUNNER" verify-result \
    --result "$result" --require-admitted \
    | tee -a "$OUT/logs/${role}_score.log"
  write_state "$role" COMPLETE "admitted result and four shards verified"
}

if [[ "$RUN_STAGE" == base || "$RUN_STAGE" == paired ]]; then
  run_role base "$BASE_BINDING"
fi

if [[ "$RUN_STAGE" == base ]]; then
  trap - EXIT INT TERM
  echo "BASE_ONLY_COMPLETE: $OUT/results/base.json is admitted; E1 was not queued or started"
  echo " positive log     : $OUT/logs/base_score.log"
  exit 0
fi

if [[ "$RUN_STAGE" == e1 ]]; then
  [[ -f "$OUT/results/base.json" ]] || {
    echo "FATAL: RUN_STAGE=e1 requires an admitted base result in the same OUT" >&2
    exit 2
  }
  "$PYTHON_BIN" "$RUNNER" verify-result \
    --result "$OUT/results/base.json" --require-admitted \
    | tee "$OUT/logs/base_gate_before_e1.log"
fi

run_role e1 "$E1_BINDING"

PAIRED_RESULT="$OUT/results/base_vs_e1.json"
if [[ -e "$PAIRED_RESULT" ]]; then
  [[ "$RESUME" == 1 ]] || { echo "FATAL: refusing existing comparison: $PAIRED_RESULT" >&2; exit 2; }
  "$PYTHON_BIN" "$RUNNER" verify-comparison \
    --comparison "$PAIRED_RESULT" --require-admitted \
    | tee "$OUT/logs/paired_compare_resume_verify.log"
else
  "$PYTHON_BIN" "$RUNNER" compare \
    --base "$OUT/results/base.json" --e1 "$OUT/results/e1.json" --out "$PAIRED_RESULT" \
    | tee "$OUT/logs/paired_compare.log"
fi
write_state pipeline COMPLETE "paired base/E1 comparison admitted and verified"

trap - EXIT INT TERM
echo "=================================================="
echo " paired result   : $PAIRED_RESULT"
echo " base result     : $OUT/results/base.json"
echo " E1 result       : $OUT/results/e1.json"
echo " positive logs   : $OUT/logs/base_score.log"
echo "                   $OUT/logs/e1_score.log"
echo "                   $OUT/logs/paired_compare.log"
echo " state receipt    : $STATE_FILE"
echo "=================================================="
