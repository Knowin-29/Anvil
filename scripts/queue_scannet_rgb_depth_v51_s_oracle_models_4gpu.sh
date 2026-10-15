#!/usr/bin/env bash
# Run the formal40 S-oracle leaderboard sequentially on one exact 4xL20 node.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"

ONLY_JOB="${ONLY_JOB:-}"
case "$ONLY_JOB" in
  "") RUN_ID="scannet_rgb_depth_v51_s_oracle_leaderboard_20260823_v2" ;;
  g_seed42) RUN_ID="scannet_rgb_depth_v51_s_oracle_g_seed42_20260823_v1" ;;
  *) echo "FATAL: unsupported ONLY_JOB: $ONLY_JOB" >&2; exit 2 ;;
esac
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

verify_existing_result() {
  local out="$1" job="$2"
  local result="$out/s_oracle/result.json"
  local temporary="$QUEUE_DIR/.${job}.verified-result.json"
  local verify_log="$QUEUE_DIR/jobs/${job}.verify.log"
  [[ -s "$out/model_binding/model.json" \
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

wait_for_idle_node() {
  local job="$1"
  record WAITING "$job" "waiting for a stable idle 4xL20 allocation"
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

MODEL_IDS=(
  qwen3vl8b
  qwen25vl7b
  qwen25vl3b
  e1
  g_seed42
)
MODEL_PATHS=(
  /path/to/environment/models/Qwen3-VL-8B-Instruct
  /path/to/environment/models/Qwen2.5-VL-7B-Instruct
  /path/to/environment/models/Qwen2.5-VL-3B-Instruct
  /path/to/Anvil/output/merged_e1
  /path/to/Anvil/output/g_family_v3_scoped_v2_qualification_eval_4gpu_seed42_local4_20260821_125929_tmux/models/specialist_79a727945e66a703
)
OUTPUT_PATHS=(
  "$W/output/scannet_rgb_depth_v51_s_oracle_qwen3vl8b_20260823_v1"
  "$W/output/scannet_rgb_depth_v51_s_oracle_qwen25vl7b_20260823_v1"
  "$W/output/scannet_rgb_depth_v51_s_oracle_qwen25vl3b_20260823_v1"
  "$W/output/scannet_rgb_depth_v51_s_oracle_e1_20260823_v1"
  "$W/output/scannet_rgb_depth_v51_s_oracle_g_seed42_20260823_v1"
)

record QUEUED queue "jobs=${ONLY_JOB:-${MODEL_IDS[*]}}"
failures=0
jobs_run=0
for index in "${!MODEL_IDS[@]}"; do
  job="${MODEL_IDS[$index]}"
  if [[ -n "$ONLY_JOB" && "$job" != "$ONLY_JOB" ]]; then
    continue
  fi
  jobs_run=$((jobs_run + 1))
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
    PYTHON_BIN="$PYTHON_BIN" MODEL="$model" OUT="$out" \
    PREFLIGHT_ONLY=0 THINKING=off BATCH=4 MAX_TOKENS=16 CELL_TIMEOUT=2h \
    bash "$LAUNCHER" >>"$job_log" 2>&1 &
  ACTIVE_PID=$!
  wait "$ACTIVE_PID"
  rc=$?
  ACTIVE_PID=""
  set -e
  if [[ "$rc" -eq 0 && -s "$result" ]]; then
    record SUCCEEDED "$job" "result=$result"
  else
    failures=$((failures + 1))
    record FAILED "$job" "rc=$rc log=$job_log"
  fi
done

rm -f "$QUEUE_DIR/queue.pid"
if [[ "$failures" -eq 0 ]]; then
  record COMPLETE queue "all $jobs_run selected jobs succeeded"
  rm -f "$QUEUE_DIR/queue.completed_with_failures"
  touch "$QUEUE_DIR/queue.completed"
  trap - EXIT INT TERM
  exit 0
fi

record COMPLETE_WITH_FAILURES queue "failed_jobs=$failures"
touch "$QUEUE_DIR/queue.completed_with_failures"
trap - EXIT INT TERM
exit 2
