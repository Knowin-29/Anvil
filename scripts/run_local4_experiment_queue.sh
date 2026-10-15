#!/usr/bin/env bash
# Evidence-gated local queue for one node with exactly four L20 GPUs.
# Independent full-split baselines continue after a failed G pilot; G replication does not.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: project root is unavailable: $W" >&2; exit 2; }
[[ -r "$W/.env" ]] || { echo "FATAL: missing $W/.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$W/.env"

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$W/logs"
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/artifacts/local4_queue"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_TOKEN=$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_' '_')
[[ -n "$RUN_TOKEN" ]] || { echo "FATAL: RUN_ID has no usable characters" >&2; exit 2; }
RUN_DIR="$W/artifacts/local4_queue/$RUN_ID"
STATE="$RUN_DIR/status.tsv"
QUEUE_LOG="$RUN_DIR/queue.log"
LOCK="$W/artifacts/local4_queue/queue.lock"
CURRENT_JOB_FILE="$RUN_DIR/current_job.tsv"
CURRENT_PID=""
CURRENT_PGID=""
INFRA_FAILURES=0

mkdir -p "$RUN_DIR/jobs"
exec 9>"$LOCK"
flock -n 9 || { echo "FATAL: another local4 queue holds $LOCK" >&2; exit 3; }
printf '%s\n' "$RUN_ID" > "$W/artifacts/local4_queue/current_run.txt"
printf '%s\n' "$$" > "$RUN_DIR/queue.pid"
[[ -s "$STATE" ]] || printf 'timestamp_utc\tjob\tstate\trc\tlog\n' > "$STATE"

record() {
  local job="$1" state="$2" rc="$3" log="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$job" "$state" "$rc" "$log" | tee -a "$STATE"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill -TERM -- "-${CURRENT_PGID:-$CURRENT_PID}" 2>/dev/null \
      || kill -TERM "$CURRENT_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$CURRENT_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$CURRENT_PID" 2>/dev/null; then
      kill -KILL -- "-${CURRENT_PGID:-$CURRENT_PID}" 2>/dev/null \
        || kill -KILL "$CURRENT_PID" 2>/dev/null || true
    fi
    wait "$CURRENT_PID" 2>/dev/null || true
  fi
  rm -f "$CURRENT_JOB_FILE"
  record queue "$([[ "$rc" == 0 ]] && echo COMPLETE || echo INTERRUPTED)" "$rc" "$QUEUE_LOG"
  exit "$rc"
}
trap cleanup EXIT INT TERM

check_storage() {
  local workspace_kb root_kb
  workspace_kb=$(df -Pk "$W" | awk 'NR==2 {print $4}')
  root_kb=$(df -Pk / | awk 'NR==2 {print $4}')
  [[ "$workspace_kb" -ge 209715200 ]] || {
    echo "FATAL: project filesystem has less than 200 GiB free" >&2; return 2;
  }
  [[ "$root_kb" -ge 2097152 ]] || {
    echo "FATAL: root filesystem has less than 2 GiB free" >&2; return 2;
  }
}

check_static_inputs() {
  local path command
  for command in flock jq nvidia-smi python setsid; do
    command -v "$command" >/dev/null || {
      echo "FATAL: required command is absent: $command" >&2; return 2;
    }
  done
  for path in \
    "$W/scripts/run_g_family_qualification_4gpu_queue.sh" \
    "$W/scripts/run_eval_adapter_dp.sh" \
    "$W/scripts/run_eval_dp.sh" \
    "/path/to/Anvil/output/merged_e1/config.json" \
    "/path/to/environment/models/Qwen2.5-VL-7B-Instruct/config.json" \
    "/path/to/environment/models/Qwen2.5-VL-3B-Instruct/config.json"
  do
    [[ -e "$path" ]] || { echo "FATAL: required queue input is absent: $path" >&2; return 2; }
  done
  check_storage
}

wait_for_four_idle_l20s() {
  local warned=0
  while true; do
    if python - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.device_count() == 4
for index in range(4):
    props = torch.cuda.get_device_properties(index)
    assert "L20" in torch.cuda.get_device_name(index).upper()
    assert props.total_memory / 2**30 >= 44
PY
    then
      mapfile -t gpu_pids < <(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
          | awk 'NF' | sort -u
      )
      if [[ "${#gpu_pids[@]}" == 0 ]]; then
        return 0
      fi
    fi
    if [[ "$warned" == 0 ]]; then
      echo "[queue] waiting for exactly four idle >=44 GiB L20 GPUs"
      warned=1
    fi
    sleep 30
  done
}

run_job() {
  local job="$1"; shift
  local log="$RUN_DIR/jobs/${job}.log" rc
  wait_for_four_idle_l20s
  check_storage || {
    record "$job" FAILED 2 "$log"
    return 2
  }
  record "$job" STARTED - "$log"
  printf '%s\t%s\n' "$job" "$log" > "$CURRENT_JOB_FILE"
  printf '[queue] command:' > "$log"
  printf ' %q' "$@" >> "$log"
  printf '\n' >> "$log"
  setsid "$@" >> "$log" 2>&1 &
  CURRENT_PID=$!
  CURRENT_PGID=$CURRENT_PID
  wait "$CURRENT_PID"
  rc=$?
  CURRENT_PID=""
  CURRENT_PGID=""
  rm -f "$CURRENT_JOB_FILE"
  if [[ "$rc" == 0 ]]; then
    record "$job" SUCCEEDED 0 "$log"
  else
    record "$job" FAILED "$rc" "$log"
  fi
  return "$rc"
}

g_train_out() {
  printf '%s/output/g_family_v3_scoped_v2_qualification_lora_4gpu_seed%s_%s\n' \
    "$W" "$1" "$RUN_TOKEN"
}

g_eval_out() {
  printf '%s/output/g_family_v3_scoped_v2_qualification_eval_4gpu_seed%s_%s\n' \
    "$W" "$1" "$RUN_TOKEN"
}

g_decision() {
  local seed="$1"
  local train_out eval_out report
  train_out=$(g_train_out "$seed")
  eval_out=$(g_eval_out "$seed")
  report="$eval_out/g_family_qualification_pilot_v2.json"
  python - "$report" "$train_out" "$eval_out" "$seed" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, train_out, eval_out = map(pathlib.Path, sys.argv[1:4])
seed = int(sys.argv[4])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value

def bound(binding, root=None):
    path = pathlib.Path(binding["path"]).resolve()
    if not path.is_file() or sha(path) != binding["sha256"]:
        raise ValueError(f"file binding drift: {path}")
    if binding.get("bytes") != path.stat().st_size:
        raise ValueError(f"file size binding drift: {path}")
    if root is not None and not path.is_relative_to(root.resolve()):
        raise ValueError(f"bound file escapes expected run directory: {path}")
    return path

report = load(report_path)
if report.get("schema_version") != 1 or report.get("protocol") != "g-family-specialist-qualification-pilot-v2":
    raise ValueError("qualification report identity drift")
if (report.get("publication_claim_eligible") is not False
        or report.get("teacher_authorization") != "NONE"
        or report.get("global_semantic_audit_verdict") != "BLOCKED"):
    raise ValueError("qualification report exceeded its evidence scope")
decision = report.get("engineering_decision")
allowed = {
    "G_PILOT_DIRECTIONAL_CONTINUE",
    "G_PILOT_NOT_SUPPORTED_REVISE_OR_STOP",
    "G_PILOT_INCONCLUSIVE",
    "UNTRUSTED_NO_DIRECTIONAL_DECISION",
}
if decision not in allowed:
    raise ValueError("invalid qualification decision")
base_result = load(bound(report["base_result"], eval_out))
specialist_result = load(bound(report["specialist_result"], eval_out))
if base_result.get("model_role") != "base" or specialist_result.get("model_role") != "specialist":
    raise ValueError("qualification result role drift")
if base_result.get("input_receipt") != specialist_result.get("input_receipt"):
    raise ValueError("base and specialist used different receipts")
receipt_path = bound(specialist_result["input_receipt"], eval_out)
receipt = load(receipt_path)
training = receipt.get("training") or {}
generation = receipt.get("generation") or {}
adapter = ((receipt.get("models") or {}).get("specialist_adapter") or {}).get("path")
if (training.get("seed") != seed or training.get("world_size") != 4
        or training.get("effective_batch_size") != 8 or training.get("max_steps") != 53
        or generation.get("eval_world_size") != 4 or generation.get("num_shards") != 8):
    raise ValueError("qualification receipt topology/seed drift")
if not adapter or not pathlib.Path(adapter).resolve().is_relative_to(train_out.resolve()):
    raise ValueError("qualification report is bound to a different training run")
print(decision)
PY
}

run_g_full_eval() {
  local seed="$1" tag="g_family_v2_seed${1}_full_local4_v8_${RUN_TOKEN}"
  local source merged results
  source=$(g_train_out "$seed")
  merged="$W/output/merged_$tag"
  results="$W/eval_results/$tag"
  [[ ! -e "$merged" && ! -e "$results" ]] || {
    echo "FATAL: refusing to reuse G full-eval output for $tag" >&2
    return 2
  }
  run_job "g_seed${seed}_full_eval" env CELL_TIMEOUT=12h THINKING=off NUM_GPUS=4 \
    MERGED_DIR="$merged" \
    bash "$W/scripts/run_eval_adapter_dp.sh" "$tag" \
    "$source" \
    "cv_bench embspatial refspatial_bench where2place gsa_bench_test" 0 16
}

run_fresh_full_eval() {
  local job="$1" tag="$2" model="$3"
  [[ ! -e "$W/eval_results/$tag" ]] || {
    echo "FATAL: refusing to reuse baseline output for $tag" >&2
    return 2
  }
  run_job "$job" env CELL_TIMEOUT=12h THINKING=off NUM_GPUS=4 \
    bash "$W/scripts/run_eval_dp.sh" "$tag" "$model" \
    "cv_bench embspatial refspatial_bench where2place gsa_bench_test" 0 16
}

check_static_inputs || exit $?
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  SEED=42 PREFLIGHT_ONLY=1 bash "$W/scripts/run_g_family_qualification_4gpu_queue.sh"
  echo "LOCAL4_EXPERIMENT_QUEUE_PREFLIGHT_READY: no GPU process was started"
  trap - EXIT INT TERM
  exit 0
fi

record queue STARTED - "$QUEUE_LOG"

# P0: the only currently authorized training experiment.
g_continue=0
seed42_train=$(g_train_out 42)
seed42_eval=$(g_eval_out 42)
if run_job g_seed42_train_qualification env SEED=42 HOURS=6 CELL_TIMEOUT=2h THINKING=off \
  TRAIN_OUT="$seed42_train" EVAL_OUT="$seed42_eval" \
  bash "$W/scripts/run_g_family_qualification_4gpu_queue.sh"
then
  if decision=$(g_decision 42); then
    if [[ "$decision" == G_PILOT_DIRECTIONAL_CONTINUE ]]; then
      g_continue=1
      record g_seed42_gate CONTINUE 0 "$seed42_eval/g_family_qualification_pilot_v2.json"
      run_g_full_eval 42 || INFRA_FAILURES=$((INFRA_FAILURES + 1))
    else
      record g_seed42_gate "$decision" 0 "$seed42_eval/g_family_qualification_pilot_v2.json"
    fi
  else
    INFRA_FAILURES=$((INFRA_FAILURES + 1))
    record g_seed42_gate INVALID_REPORT 5 "$seed42_eval/g_family_qualification_pilot_v2.json"
  fi
else
  INFRA_FAILURES=$((INFRA_FAILURES + 1))
  record g_seed42_gate STOP 4 "$RUN_DIR/jobs/g_seed42_train_qualification.log"
fi

# Independent full-split evidence. These jobs remain useful if the G pilot stops.
run_fresh_full_eval e1_full_eval "gsa_joint_sft_e1_full_local4_v8_${RUN_TOKEN}" \
  /path/to/Anvil/output/merged_e1 \
  || INFRA_FAILURES=$((INFRA_FAILURES + 1))

run_fresh_full_eval qwen2_5_vl_7b_full_eval "qwen2_5_vl_7b_full_local4_v8_${RUN_TOKEN}" \
  /path/to/environment/models/Qwen2.5-VL-7B-Instruct \
  || INFRA_FAILURES=$((INFRA_FAILURES + 1))

run_fresh_full_eval qwen2_5_vl_3b_full_eval "qwen2_5_vl_3b_full_local4_v8_${RUN_TOKEN}" \
  /path/to/environment/models/Qwen2.5-VL-3B-Instruct \
  || INFRA_FAILURES=$((INFRA_FAILURES + 1))

# Replication is preregistered but conditional; seed42 is always the primary run.
if [[ "$g_continue" == 1 ]]; then
  for seed in 43 44; do
    train_out=$(g_train_out "$seed")
    eval_out=$(g_eval_out "$seed")
    if run_job "g_seed${seed}_train_qualification" env SEED="$seed" HOURS=6 \
      CELL_TIMEOUT=2h THINKING=off TRAIN_OUT="$train_out" EVAL_OUT="$eval_out" \
      bash "$W/scripts/run_g_family_qualification_4gpu_queue.sh"
    then
      if decision=$(g_decision "$seed"); then
        if [[ "$decision" == G_PILOT_DIRECTIONAL_CONTINUE ]]; then
          record "g_seed${seed}_gate" CONTINUE 0 "$eval_out/g_family_qualification_pilot_v2.json"
          run_g_full_eval "$seed" || INFRA_FAILURES=$((INFRA_FAILURES + 1))
        else
          record "g_seed${seed}_gate" "$decision" 0 "$eval_out/g_family_qualification_pilot_v2.json"
        fi
      else
        INFRA_FAILURES=$((INFRA_FAILURES + 1))
        record "g_seed${seed}_gate" INVALID_REPORT 5 "$eval_out/g_family_qualification_pilot_v2.json"
      fi
    else
      INFRA_FAILURES=$((INFRA_FAILURES + 1))
      record "g_seed${seed}_gate" REPLICATION_FAILED 4 \
        "$RUN_DIR/jobs/g_seed${seed}_train_qualification.log"
    fi
  done
else
  record g_replication SKIPPED 0 "seed42 did not satisfy the frozen continue rule"
fi

trap - EXIT INT TERM
if [[ "$INFRA_FAILURES" -gt 0 ]]; then
  record queue COMPLETE_WITH_FAILURES "$INFRA_FAILURES" "$QUEUE_LOG"
  echo "LOCAL4_EXPERIMENT_QUEUE_COMPLETE_WITH_FAILURES: $RUN_DIR ($INFRA_FAILURES failures)" >&2
  exit 1
fi
record queue COMPLETE 0 "$QUEUE_LOG"
echo "LOCAL4_EXPERIMENT_QUEUE_COMPLETE: $RUN_DIR"
