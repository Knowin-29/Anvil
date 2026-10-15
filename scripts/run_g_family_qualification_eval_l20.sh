#!/usr/bin/env bash
# Foreground 8xL20 evaluation of frozen base vs the v2 G specialist qualification
# checkpoint. An optional joint adapter is diagnostic only. No output from this script
# authorizes joint/MOPD or writes a global teacher-eligibility PASS.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || { echo "FATAL: required environment file is missing: $W/.env" >&2; exit 2; }
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

EVALUATOR="$W/eval/g_family_qualification.py"
MANIFEST="${G_V2_MANIFEST:-$W/trains/gsa_bench_v3_g_family_scoped_v2/manifest.json}"
SCOPED_REPORT="${G_SCOPED_REPORT:-$W/artifacts/audits/pilot_semantic_audit_v3_g_family_scoped_v1.json}"
FULL_REPORT="${V3_FULL_REPORT:-$W/artifacts/audits/pilot_semantic_audit_v3_adjudicated.json}"
BASE_MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
G_SOURCE="${G_SOURCE:-$W/output/g_family_v3_scoped_v2_qualification_lora}"
OUT="${OUT:-$W/output/g_family_v3_scoped_v2_qualification_eval}"
CELL_TIMEOUT="${CELL_TIMEOUT:-2h}"
BATCH="${BATCH:-4}"
MAX_TOKENS="${MAX_TOKENS:-512}"
EVAL_WORLD_SIZE="${GSA_EVAL_WORLD_SIZE:-8}"
NUM_SHARDS=8
[[ "$BATCH" =~ ^[1-9][0-9]*$ && "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: BATCH and MAX_TOKENS must be positive integers" >&2; exit 2;
}
[[ "$EVAL_WORLD_SIZE" == 4 || "$EVAL_WORLD_SIZE" == 8 ]] || {
  echo "FATAL: GSA_EVAL_WORLD_SIZE must be exactly 4 or 8" >&2; exit 2;
}
(( NUM_SHARDS % EVAL_WORLD_SIZE == 0 )) || {
  echo "FATAL: fixed logical shard count must divide by evaluation world size" >&2; exit 2;
}
[[ "$MAX_TOKENS" == 512 ]] || { echo "FATAL: frozen protocol requires MAX_TOKENS=512" >&2; exit 2; }
[[ "$BATCH" == 4 ]] || { echo "FATAL: frozen protocol requires BATCH=4" >&2; exit 2; }
case "${THINKING:-off}" in
  off|0|false) ;;
  *) echo "FATAL: this frozen protocol requires THINKING=off" >&2; exit 2 ;;
esac

G_CHECKPOINT=$("$PYTHON_BIN" "$EVALUATOR" resolve-checkpoint --source "$G_SOURCE")
G_ID=$("$PYTHON_BIN" "$EVALUATOR" checkpoint-id --checkpoint "$G_CHECKPOINT")
JOINT_CHECKPOINT=""
if [[ -n "${JOINT_SOURCE:-}" ]]; then
  JOINT_CHECKPOINT=$("$PYTHON_BIN" "$EVALUATOR" resolve-checkpoint --source "$JOINT_SOURCE")
fi

mkdir -p "$OUT" "$OUT/model_bindings" "$OUT/predictions" "$OUT/results" "$OUT/logs" "$OUT/models"
INPUT_RECEIPT="$OUT/g_family_qualification_eval_inputs_v2.json"
PREFLIGHT_ARGS=(
  preflight --manifest "$MANIFEST" --scoped-report "$SCOPED_REPORT"
  --full-report "$FULL_REPORT" --base-model "$BASE_MODEL"
  --source "$G_SOURCE" --checkpoint "$G_CHECKPOINT" --out "$INPUT_RECEIPT"
  --eval-world-size "$EVAL_WORLD_SIZE"
  --batch-size "$BATCH"
)
if [[ -n "$JOINT_CHECKPOINT" ]]; then
  PREFLIGHT_ARGS+=(--joint-checkpoint "$JOINT_CHECKPOINT")
fi
"$PYTHON_BIN" "$EVALUATOR" "${PREFLIGHT_ARGS[@]}"

echo "=================================================="
echo " protocol       : g-family-specialist-qualification-pilot-v2"
echo " qualification  : 32 rows / 8 images; never used for train/dev"
echo " frozen base    : $BASE_MODEL"
echo " G checkpoint   : $G_CHECKPOINT"
echo " G identity     : $G_ID"
echo " hardware       : ${EVAL_WORLD_SIZE}xL20"
echo " schedule       : $NUM_SHARDS fixed logical shards / $EVAL_WORLD_SIZE concurrent workers"
echo " joint reference: ${JOINT_CHECKPOINT:-not supplied}"
echo " output         : $OUT"
echo " claim scope    : directional pilot only; global audit remains BLOCKED"
echo "=================================================="
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "G_V2_EVAL_PREFLIGHT_READY: no model merge or GPU process was started"
  exit 0
fi

mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF' | sort -u
)
if [[ "${#GPU_PIDS[@]}" -gt 0 && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "FATAL: visible GPUs already have compute processes: ${GPU_PIDS[*]}" >&2; exit 3;
fi
"$PYTHON_BIN" - "$EVAL_WORLD_SIZE" <<'PY'
import sys, torch
count = torch.cuda.device_count()
expected = int(sys.argv[1])
if count != expected:
    raise SystemExit(f"FATAL: qualification evaluation requires exactly {expected} GPUs, found {count}")
for index in range(count):
    name = torch.cuda.get_device_name(index)
    gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    if "L20" not in name.upper() or gib < 44:
        raise SystemExit(f"FATAL: GPU {index} is not a >=44 GiB L20: {name} ({gib:.1f} GiB)")
print(f"{expected}xL20 hardware gate: READY")
PY

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  for ((index = 0; index < EVAL_WORLD_SIZE; index++)); do GPU_IDS+=("$index"); done
fi
[[ "${#GPU_IDS[@]}" == "$EVAL_WORLD_SIZE" ]] || {
  echo "FATAL: expected exactly $EVAL_WORLD_SIZE visible GPU ids" >&2; exit 2;
}

BASE_BINDING="$OUT/model_bindings/base.json"
"$PYTHON_BIN" "$EVALUATOR" bind-model --role base --model "$BASE_MODEL" \
  --input-receipt "$INPUT_RECEIPT" --out "$BASE_BINDING"

prepare_merged_binding() {
  local role="$1" checkpoint="$2" checkpoint_id merged binding
  checkpoint_id=$("$PYTHON_BIN" "$EVALUATOR" checkpoint-id --checkpoint "$checkpoint")
  merged="$OUT/models/${role}_${checkpoint_id:0:16}"
  binding="$OUT/model_bindings/${role}.json"
  if [[ -f "$binding" ]]; then
    "$PYTHON_BIN" "$EVALUATOR" bind-model --role "$role" --model "$merged" \
      --adapter "$checkpoint" --input-receipt "$INPUT_RECEIPT" --out "$binding" >&2
  else
    if [[ -e "$merged" ]]; then
      echo "FATAL: unbound merged directory already exists: $merged" >&2
      return 2
    fi
    echo "[merge] $role: $checkpoint -> $merged" >&2
    swift export --adapters "$checkpoint" --merge_lora true --output_dir "$merged" >&2
    "$PYTHON_BIN" "$EVALUATOR" bind-model --role "$role" --model "$merged" \
      --adapter "$checkpoint" --input-receipt "$INPUT_RECEIPT" --out "$binding" >&2
  fi
  printf '%s\n%s\n' "$merged" "$binding"
}

mapfile -t G_MERGED_INFO < <(prepare_merged_binding specialist "$G_CHECKPOINT")
[[ "${#G_MERGED_INFO[@]}" -eq 2 ]] || { echo "FATAL: specialist merge did not resolve" >&2; exit 2; }
G_MODEL="${G_MERGED_INFO[0]}"
G_BINDING="${G_MERGED_INFO[1]}"

JOINT_MODEL=""
JOINT_BINDING=""
if [[ -n "$JOINT_CHECKPOINT" ]]; then
  mapfile -t JOINT_MERGED_INFO < <(prepare_merged_binding joint_reference "$JOINT_CHECKPOINT")
  [[ "${#JOINT_MERGED_INFO[@]}" -eq 2 ]] || { echo "FATAL: joint merge did not resolve" >&2; exit 2; }
  JOINT_MODEL="${JOINT_MERGED_INFO[0]}"
  JOINT_BINDING="${JOINT_MERGED_INFO[1]}"
fi

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
  local role="$1" model="$2" binding="$3"
  local pred_dir="$OUT/predictions/$role" result="$OUT/results/$role.json" rc=0
  mkdir -p "$pred_dir"
  CURRENT_PIDS=()
  for ((wave_start = 0; wave_start < NUM_SHARDS; wave_start += EVAL_WORLD_SIZE)); do
    CURRENT_PIDS=()
    for ((slot = 0; slot < EVAL_WORLD_SIZE; slot++)); do
      local shard=$(( wave_start + slot ))
      local shard_out="$pred_dir/shard_${shard}.jsonl"
      if [[ -s "$shard_out" ]]; then
        echo "[$role] reusing bound shard $shard"
        continue
      fi
      CUDA_VISIBLE_DEVICES="${GPU_IDS[$slot]}" timeout "$CELL_TIMEOUT" \
        "$PYTHON_BIN" "$EVALUATOR" infer-shard \
          --input-receipt "$INPUT_RECEIPT" --model-binding "$binding" \
          --shard "$shard" --num-shards "$NUM_SHARDS" --batch-size "$BATCH" \
          --max-tokens "$MAX_TOKENS" --out "$shard_out" \
          >"$OUT/logs/${role}_shard_${shard}.log" 2>&1 &
      CURRENT_PIDS+=("$!")
    done
    for pid in "${CURRENT_PIDS[@]:-}"; do wait "$pid" || rc=1; done
    CURRENT_PIDS=()
    [[ "$rc" == 0 ]] || break
  done
  [[ "$rc" == 0 ]] || {
    echo "FATAL: $role inference failed; inspect $OUT/logs/${role}_shard_*.log" >&2
    return 1
  }
  local shards=("$pred_dir"/shard_*.jsonl)
  [[ "${#shards[@]}" == "$NUM_SHARDS" ]] || {
    echo "FATAL: $role did not produce $NUM_SHARDS logical shards" >&2; return 2;
  }
  "$PYTHON_BIN" "$EVALUATOR" score --input-receipt "$INPUT_RECEIPT" \
    --model-binding "$binding" --predictions "${shards[@]}" --out "$result"
}

run_model base "$BASE_MODEL" "$BASE_BINDING"
run_model specialist "$G_MODEL" "$G_BINDING"
if [[ -n "$JOINT_MODEL" ]]; then
  run_model joint_reference "$JOINT_MODEL" "$JOINT_BINDING"
fi

COMPARE_ARGS=(
  compare --base "$OUT/results/base.json" --specialist "$OUT/results/specialist.json"
  --out "$OUT/g_family_qualification_pilot_v2.json"
)
if [[ -n "$JOINT_MODEL" ]]; then
  COMPARE_ARGS+=(--joint "$OUT/results/joint_reference.json")
fi
"$PYTHON_BIN" "$EVALUATOR" "${COMPARE_ARGS[@]}"
trap - EXIT INT TERM

echo "=================================================="
echo " qualification report: $OUT/g_family_qualification_pilot_v2.json"
echo " This report is a pilot engineering decision only."
echo " It does not authorize joint training, MOPD, or a global teacher PASS."
echo "=================================================="
