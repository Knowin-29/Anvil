#!/usr/bin/env bash
# Wait for four idle local L20s, then run only the formal general-retention base stage.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || { echo "FATAL: missing $W/.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$W/.env"

PYTHON_BIN="${PYTHON_BIN:-python}"
MONITOR_PYTHON_BIN="${MONITOR_PYTHON_BIN:-$PYTHON_BIN}"
RUNNER="$W/eval/general_retention_formal_v1.py"
LAUNCHER="$W/scripts/run_general_retention_dp.sh"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)_$$}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]] || {
  echo "FATAL: RUN_TAG must contain only 1-81 safe filename characters" >&2
  exit 2
}

OUT_BASE="${OUT:-$W/output/general_retention_formal_v1_base_e1_${RUN_TAG}}"
OUT="$OUT_BASE"
QUEUE_ROOT="${QUEUE_ROOT:-$W/artifacts/general_retention_base_queue}"
QUEUE_DIR="$QUEUE_ROOT/$RUN_TAG"
STATUS="$QUEUE_DIR/status.tsv"
HARDWARE_LOG="$QUEUE_DIR/hardware_last.json"
BASE_LOG=""
CURRENT_JOB="$QUEUE_DIR/current_job.tsv"
LOCK="${QUEUE_LOCK:-$W/artifacts/local4_queue/queue.lock}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
MAX_HARDWARE_RACE_RETRIES="${MAX_HARDWARE_RACE_RETRIES:-8}"
RUNTIME_MONITOR_SECONDS="${RUNTIME_MONITOR_SECONDS:-1}"

[[ "$POLL_SECONDS" =~ ^[0-9]+$ \
    && "$STABILITY_SECONDS" =~ ^[0-9]+$ \
    && "$MAX_HARDWARE_RACE_RETRIES" =~ ^[0-9]+$ \
    && "$RUNTIME_MONITOR_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: poll, stability, and retry settings must be non-negative integers" >&2
  exit 2
}
[[ -f "$RUNNER" && -x "$LAUNCHER" ]] || {
  echo "FATAL: formal runner or launcher is unavailable" >&2
  exit 2
}
[[ ! -e "$OUT_BASE" ]] || { echo "FATAL: refusing stale OUT: $OUT_BASE" >&2; exit 2; }
[[ ! -e "$QUEUE_DIR" ]] || {
  echo "FATAL: refusing stale queue directory: $QUEUE_DIR" >&2
  exit 2
}

mkdir -p "$QUEUE_DIR" "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "FATAL: another project 4-GPU queue holds $LOCK" >&2; exit 3; }
printf '%s\n' "$$" > "$QUEUE_DIR/queue.pid"
printf 'timestamp_utc\tscope\tstate\trc\tdetail\n' > "$STATUS"

CURRENT_PID=""
CURRENT_PGID=""
MONITOR_PID=""
TERMINAL_RECORDED=0
HW_STATE=""

record() {
  local scope="$1" state="$2" rc="$3" detail="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$scope" "$state" "$rc" "$detail" | tee -a "$STATUS"
}

process_group_alive() {
  [[ -n "$CURRENT_PGID" ]] && kill -0 -- "-$CURRENT_PGID" 2>/dev/null
}

terminate_current_group() {
  process_group_alive || return 0
  kill -TERM -- "-$CURRENT_PGID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    process_group_alive || return 0
    sleep 1
  done
  kill -KILL -- "-$CURRENT_PGID" 2>/dev/null || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "$MONITOR_PID" ]]; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if process_group_alive; then
    record base STOPPING "$rc" "pid=$CURRENT_PID pgid=$CURRENT_PGID"
    terminate_current_group
  fi
  if [[ -n "$CURRENT_PID" ]]; then
    wait "$CURRENT_PID" 2>/dev/null || true
  fi
  rm -f "$CURRENT_JOB"
  if [[ "$TERMINAL_RECORDED" == 0 ]]; then
    record queue INTERRUPTED "$rc" "out=$OUT"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

poll_hardware() {
  local output rc parse_rc
  set +e
  output=$("$PYTHON_BIN" "$RUNNER" hardware-status \
    --nvidia-smi-bin "$NVIDIA_SMI_BIN" \
    --gpu-selector 0 --gpu-selector 1 --gpu-selector 2 --gpu-selector 3 2>&1)
  rc=$?
  set -e
  printf '%s\n' "$output" > "$HARDWARE_LOG"
  if [[ "$rc" != 0 ]]; then
    HW_STATE="ERROR"
    return 0
  fi
  set +e
  HW_STATE=$(printf '%s\n' "$output" | "$PYTHON_BIN" -c \
    'import json,sys; value=json.load(sys.stdin); status=value.get("status"); assert status in {"HARDWARE_IDLE","HARDWARE_BUSY"}; print(status)')
  parse_rc=$?
  set -e
  if [[ "$parse_rc" != 0 ]]; then
    HW_STATE="ERROR"
  fi
}

monitor_runtime_hardware() {
  local reason_file="$1" output rc process_counts parse_rc total foreign
  while process_group_alive; do
    set +e
    output=$("$NVIDIA_SMI_BIN" \
      --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits 2>&1)
    rc=$?
    set -e
    if [[ "$rc" != 0 ]]; then
      printf 'QUERY_FAILED\t%s\n' "$rc" > "$reason_file"
      kill -TERM -- "-$CURRENT_PGID" 2>/dev/null || true
      return 0
    fi
    set +e
    process_counts=$(printf '%s\n' "$output" | "$MONITOR_PYTHON_BIN" -c '
import csv, io, pathlib, sys

expected_pgid = int(sys.argv[1])
text = sys.stdin.read()
rows = []
for row in csv.reader(io.StringIO(text)):
    if not row or all(not value.strip() for value in row):
        continue
    if len(row) != 4:
        raise SystemExit(2)
    try:
        rows.append(int(row[1].strip()))
    except ValueError as exc:
        raise SystemExit(2) from exc

foreign = 0
for pid in rows:
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2 :].split()
        process_pgid = int(tail[2])
    except (FileNotFoundError, IndexError, ValueError):
        foreign += 1
        continue
    if process_pgid != expected_pgid:
        foreign += 1
print(len(rows), foreign)
' "$CURRENT_PGID")
    parse_rc=$?
    set -e
    if [[ "$parse_rc" != 0 ]]; then
      printf 'QUERY_PARSE_FAILED\t%s\n' "$parse_rc" > "$reason_file"
      kill -TERM -- "-$CURRENT_PGID" 2>/dev/null || true
      return 0
    fi
    read -r total foreign <<< "$process_counts"
    # Every compute PID must be visible in this namespace and belong to the
    # launcher's process group. This also catches an intruder before all four
    # formal workers have created their CUDA contexts.
    if (( foreign > 0 || total > 4 )); then
      printf 'FOREIGN_COMPUTE_CONTEXTS\ttotal=%s\tforeign=%s\n' \
        "$total" "$foreign" > "$reason_file"
      kill -TERM -- "-$CURRENT_PGID" 2>/dev/null || true
      return 0
    fi
    sleep "$RUNTIME_MONITOR_SECONDS"
  done
}

record queue QUEUED - "tag=$RUN_TAG out_base=$OUT_BASE"
attempt=0
while true; do
  OUT="$OUT_BASE"
  BASE_LOG="$QUEUE_DIR/base_launcher.log"
  if [[ "$attempt" != 0 ]]; then
    OUT="${OUT_BASE}.retry${attempt}"
    BASE_LOG="$QUEUE_DIR/base_launcher_retry_${attempt}.log"
  fi
  [[ ! -e "$OUT" ]] || {
    record base FRESH_OUT_REJECTED 5 "attempt=$attempt out=$OUT"
    TERMINAL_RECORDED=1
    exit 5
  }

  record queue WAITING - "attempt=$attempt exact4 L20; hardware=$HARDWARE_LOG"
  while true; do
    poll_hardware
    if [[ "$HW_STATE" == ERROR ]]; then
      record queue HARDWARE_QUERY_FAILED 4 "hardware=$HARDWARE_LOG"
      TERMINAL_RECORDED=1
      exit 4
    fi
    if [[ "$HW_STATE" == HARDWARE_IDLE ]]; then
      record queue CANDIDATE_IDLE - "attempt=$attempt stability_wait=${STABILITY_SECONDS}s"
      sleep "$STABILITY_SECONDS"
      poll_hardware
      if [[ "$HW_STATE" == ERROR ]]; then
        record queue HARDWARE_QUERY_FAILED 4 "hardware=$HARDWARE_LOG"
        TERMINAL_RECORDED=1
        exit 4
      fi
      if [[ "$HW_STATE" == HARDWARE_IDLE ]]; then
        break
      fi
      record queue IDLE_REVOKED - "attempt=$attempt hardware=$HARDWARE_LOG"
    fi
    sleep "$POLL_SECONDS"
  done

  record queue IDLE_STABLE - "attempt=$attempt two checks; exact4 unique physical L20"
  if [[ "${QUEUE_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    record queue QUEUE_PREFLIGHT_READY 0 "no OUT or GPU process started"
    TERMINAL_RECORDED=1
    exit 0
  fi

  # Atomic reservation makes stale-result reuse impossible even under concurrent submitters.
  mkdir "$OUT" || {
    record base FRESH_OUT_REJECTED 5 "attempt=$attempt out=$OUT"
    TERMINAL_RECORDED=1
    exit 5
  }
  record base STARTING - "attempt=$attempt out=$OUT log=$BASE_LOG"
  printf 'base\t%s\t%s\n' "$OUT" "$BASE_LOG" > "$CURRENT_JOB"
  setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_STAGE=base THINKING=off BATCH=16 \
    PREFLIGHT_ONLY=0 OUT="$OUT" \
    bash "$LAUNCHER" > "$BASE_LOG" 2>&1 &
  CURRENT_PID=$!
  CURRENT_PGID=$CURRENT_PID
  record base RUNNING - "attempt=$attempt pid=$CURRENT_PID pgid=$CURRENT_PGID log=$BASE_LOG"

  monitor_reason="$QUEUE_DIR/runtime_monitor_attempt_${attempt}.tsv"
  rm -f "$monitor_reason"
  monitor_runtime_hardware "$monitor_reason" &
  MONITOR_PID=$!

  set +e
  wait "$CURRENT_PID"
  rc=$?
  set -e
  kill -TERM "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  MONITOR_PID=""
  CURRENT_PID=""
  rm -f "$CURRENT_JOB"
  group_leaked=0
  if process_group_alive; then
    group_leaked=1
  fi

  if [[ "$rc" == 0 \
        && "$group_leaked" == 0 \
        && -f "$OUT/results/base.json" \
        && ! -e "$OUT/results/e1.json" \
        && ! -e "$OUT/results/base_vs_e1.json" \
        && ! -e "$OUT/predictions/e1" ]] \
      && grep -Fq "BASE_ONLY_COMPLETE:" "$BASE_LOG" \
      && "$PYTHON_BIN" "$RUNNER" verify-result \
        --result "$OUT/results/base.json" --require-admitted >> "$BASE_LOG" 2>&1
  then
    CURRENT_PGID=""
    record base SUCCEEDED 0 "attempt=$attempt result=$OUT/results/base.json"
    record queue COMPLETE 0 "base admitted; E1 not queued"
    TERMINAL_RECORDED=1
    exit 0
  fi

  if [[ "$rc" == 2 \
        && "$group_leaked" == 0 \
        && ! -e "$OUT/predictions/base" \
        && ! -e "$OUT/results/base.json" ]] \
      && grep -Fq "FATAL: selected GPUs have compute processes:" "$BASE_LOG"
  then
    CURRENT_PGID=""
    record base HARDWARE_RACE_REQUEUE 2 "attempt=$attempt partial_out=$OUT"
    if [[ "$attempt" -ge "$MAX_HARDWARE_RACE_RETRIES" ]]; then
      record queue HARDWARE_RACE_RETRY_LIMIT 7 "attempts=$((attempt + 1))"
      TERMINAL_RECORDED=1
      exit 7
    fi
    attempt=$((attempt + 1))
    continue
  fi

  if [[ -s "$monitor_reason" ]] && grep -Fq $'FOREIGN_COMPUTE_CONTEXTS\t' "$monitor_reason"; then
    terminate_current_group
    CURRENT_PGID=""
    record base RUNTIME_HARDWARE_INTRUSION_REQUEUE 2 \
      "attempt=$attempt partial_out=$OUT reason=$monitor_reason"
    if [[ "$attempt" -ge "$MAX_HARDWARE_RACE_RETRIES" ]]; then
      record queue HARDWARE_RACE_RETRY_LIMIT 7 "attempts=$((attempt + 1))"
      TERMINAL_RECORDED=1
      exit 7
    fi
    attempt=$((attempt + 1))
    continue
  fi

  if [[ -s "$monitor_reason" ]]; then
    terminate_current_group
    CURRENT_PGID=""
    record queue RUNTIME_HARDWARE_QUERY_FAILED 4 "attempt=$attempt reason=$monitor_reason"
    TERMINAL_RECORDED=1
    exit 4
  fi

  failure_rc="$rc"
  [[ "$failure_rc" != 0 ]] || failure_rc=6
  record base FAILED "$failure_rc" "attempt=$attempt log=$BASE_LOG"
  record queue COMPLETE_WITH_FAILURE 6 "base not admitted; E1 not queued"
  TERMINAL_RECORDED=1
  exit 6
done
