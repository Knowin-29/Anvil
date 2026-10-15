#!/usr/bin/env bash
# Run the two cross-family formal40 S-oracle controls sequentially on exact 4xL20.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"

RUN_ID="${RUN_ID:-scannet_rgb_depth_v51_s_oracle_crossfamily_20260823_v1}"
QUEUE_DIR="${QUEUE_DIR:-$W/artifacts/local4_queue/$RUN_ID}"
STATUS="$QUEUE_DIR/status.tsv"
QUEUE_LOG="$QUEUE_DIR/queue.log"
LOCK="$W/artifacts/local4_queue/queue.lock"
LAUNCHER="$W/scripts/run_scannet_rgb_depth_v51_s_oracle_l20_4gpu.sh"
PYTHON_BIN="${PYTHON_BIN:-/path/to/environment/env/gsa_train/bin/python}"
SCORER="$W/eval/scannet_rgb_depth_v51_s_oracle.py"
RELEASE_MANIFEST="$W/artifacts/depth_v51_eval_interface/formal40/release.manifest.json"

mkdir -p "$QUEUE_DIR/jobs"
command -v setsid >/dev/null || { echo "FATAL: setsid is required" >&2; exit 2; }
command -v flock >/dev/null || { echo "FATAL: flock is required" >&2; exit 2; }
[[ ! -e "$QUEUE_DIR/queue.completed" ]] || {
  echo "FATAL: queue was already completed: $QUEUE_DIR" >&2
  exit 2
}
exec 9>"$LOCK"
flock -n 9 || {
  echo "FATAL: another project 4-GPU queue holds $LOCK" >&2
  exit 3
}
printf '%s\n' "$$" > "$QUEUE_DIR/queue.pid"

record() {
  local state="$1" job="$2" detail="$3"
  printf '%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$job" "$detail" \
    | tee -a "$STATUS" "$QUEUE_LOG"
}

four_l20s_present() {
  local output
  output=$(nvidia-smi --query-gpu=name,memory.total \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  mapfile -t rows <<< "$output"
  [[ "${#rows[@]}" == 4 ]] || return 1
  local row name memory
  for row in "${rows[@]}"; do
    name=${row%,*}
    memory=${row##*,}
    memory=${memory// /}
    [[ "${name^^}" == *L20* && "$memory" -ge 45056 ]] || return 1
  done
}

gpus_idle() {
  local output
  output=$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  [[ -z "$(printf '%s\n' "$output" | awk 'NF')" ]]
}

wait_for_idle_node() {
  local job="$1"
  record WAITING "$job" "waiting for a stable idle exact-4xL20 allocation"
  while true; do
    if four_l20s_present && gpus_idle; then
      sleep 10
      if four_l20s_present && gpus_idle; then
        return 0
      fi
    fi
    sleep 30
  done
}

verify_existing_result() {
  local out="$1" job="$2"
  local result="$out/s_oracle/result.json"
  local temporary="$QUEUE_DIR/.${job}.verified-result.json"
  local verify_log="$QUEUE_DIR/jobs/${job}.verify.log"
  [[ -s "$out/formal40_inputs.json" \
     && -s "$out/model_binding/model.json" \
     && -s "$out/s_oracle/predictions.jsonl" \
     && -s "$out/s_oracle/predictions.manifest.json" \
     && -s "$result" ]] || return 1
  rm -f "$temporary"
  "$PYTHON_BIN" "$SCORER" score-s-oracle \
    --release-manifest "$RELEASE_MANIFEST" \
    --model-binding "$out/model_binding/model.json" \
    --s-oracle-predictions "$out/s_oracle/predictions.jsonl" \
    --s-oracle-manifest "$out/s_oracle/predictions.manifest.json" \
    --out "$temporary" >"$verify_log" 2>&1 || {
      rm -f "$temporary"
      return 1
    }
  local rc=0
  cmp -s "$temporary" "$result" || rc=$?
  rm -f "$temporary"
  return "$rc"
}

ACTIVE_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || true
    wait "$ACTIVE_PID" 2>/dev/null || true
  fi
  rm -f "$QUEUE_DIR/queue.pid"
  if [[ "$rc" -ne 0 ]]; then
    record INTERRUPTED queue "rc=$rc"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

MODEL_IDS=(internvl35_8b llava_onevision_7b)
MODEL_PATHS=(
  /path/to/environment/models/InternVL3_5-8B-HF
  /path/to/environment/models/llava-onevision-qwen2-7b-ov-hf
)
OUTPUT_PATHS=(
  "$W/output/scannet_rgb_depth_v51_s_oracle_internvl35_8b_20260823_v2"
  "$W/output/scannet_rgb_depth_v51_s_oracle_llavaov7b_20260823_v1"
)

record QUEUED queue "jobs=${MODEL_IDS[*]}"
failures=0
for index in "${!MODEL_IDS[@]}"; do
  job="${MODEL_IDS[$index]}"
  model="${MODEL_PATHS[$index]}"
  out="${OUTPUT_PATHS[$index]}"
  job_log="$QUEUE_DIR/jobs/${job}.log"
  result="$out/s_oracle/result.json"

  if [[ -e "$result" ]]; then
    if verify_existing_result "$out" "$job"; then
      record REUSED "$job" "fully reverified result=$result"
      continue
    fi
    record FAILED "$job" "pre-existing result failed frozen revalidation: $result"
    exit 2
  fi

  wait_for_idle_node "$job"
  record STARTED "$job" "model=$model out=$out log=$job_log"
  set +e
  setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTHON_BIN="$PYTHON_BIN" MODEL="$model" MODEL_ROLE=evaluated OUT="$out" \
    PREFLIGHT_ONLY=0 THINKING=off BATCH=4 MAX_TOKENS=16 CELL_TIMEOUT=2h \
    bash "$LAUNCHER" >>"$job_log" 2>&1 &
  ACTIVE_PID=$!
  wait "$ACTIVE_PID"
  rc=$?
  ACTIVE_PID=""
  set -e

  if [[ "$rc" -eq 0 && -s "$result" ]] && verify_existing_result "$out" "$job"; then
    record SUCCEEDED "$job" "result=$result"
  else
    failures=$((failures + 1))
    record FAILED "$job" "rc=$rc log=$job_log"
  fi
done

rm -f "$QUEUE_DIR/queue.pid"
if [[ "$failures" -eq 0 ]]; then
  record COMPLETE queue "all ${#MODEL_IDS[@]} jobs succeeded"
  rm -f "$QUEUE_DIR/queue.completed_with_failures"
  touch "$QUEUE_DIR/queue.completed"
  trap - EXIT INT TERM
  exit 0
fi

record COMPLETE_WITH_FAILURES queue "failed_jobs=$failures"
touch "$QUEUE_DIR/queue.completed_with_failures"
trap - EXIT INT TERM
exit 2
