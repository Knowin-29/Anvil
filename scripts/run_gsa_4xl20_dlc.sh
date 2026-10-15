#!/usr/bin/env bash
# Single-entry 4xL20 DLC orchestration. Defaults to preflight-only.
set -Eeuo pipefail
W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_4xl20}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]] || { echo "FATAL: unsafe RUN_ID: $RUN_ID" >&2; exit 2; }
RUN_ROOT="${RUN_ROOT:-$W/artifacts/dlc_runs/$RUN_ID}"
LOG_ROOT="$RUN_ROOT/logs"; STATUS="$RUN_ROOT/status.tsv"; MANIFEST="$RUN_ROOT/run_manifest.json"
LOCK="${DLC_RUN_LOCK:-$W/artifacts/dlc_runs/.4xl20.lock}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-1}"
RESUME_RUN="${RESUME_RUN:-0}"
RUN_GENERAL_RETENTION="${RUN_GENERAL_RETENTION:-1}"
RUN_V52_PAIRED="${RUN_V52_PAIRED:-auto}"
MIN_DISK_GB="${MIN_DISK_GB:-50}"
V52_PACK="${V52_PACK:-$W/data/semantic_audit_v52_scannet_unique_referent_gs_formal}"
V52_EVALUATOR="$W/eval/scannet_rgb_depth_v52_eval.py"
V52_ADJUDICATION="${V52_ADJUDICATION:-$V52_PACK/reviewer_c/annotations/annotations.jsonl}"
V52_PAIRED_LAUNCHER="${V52_PAIRED_LAUNCHER:-$W/scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh}"
GPU_SET="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_SET"
[[ "${#GPU_IDS[@]}" == 4 ]] || { echo "FATAL: DLC task must expose exactly four GPU selectors: $GPU_SET" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l)" == 4 ]] || { echo "FATAL: duplicate GPU selectors: $GPU_SET" >&2; exit 2; }
for selector in "${GPU_IDS[@]}"; do
  [[ -n "$selector" && "$selector" != *[[:space:]]* ]] || { echo "FATAL: invalid GPU selector: '$selector'" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="$GPU_SET"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || { echo "FATAL: PREFLIGHT_ONLY must be 0 or 1" >&2; exit 2; }
[[ "$RESUME_RUN" == 0 || "$RESUME_RUN" == 1 ]] || { echo "FATAL: RESUME_RUN must be 0 or 1" >&2; exit 2; }
[[ "$MIN_DISK_GB" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: MIN_DISK_GB must be positive" >&2; exit 2; }
for flag in RUN_GENERAL_RETENTION; do
  value="${!flag}"; [[ "$value" == 0 || "$value" == 1 ]] || { echo "FATAL: $flag must be 0 or 1" >&2; exit 2; }
done
case "$RUN_V52_PAIRED" in
  auto|0|1) ;;
  *) echo "FATAL: RUN_V52_PAIRED must be auto, 0, or 1" >&2; exit 2 ;;
esac
[[ -r "$W/.env" ]] || { echo "FATAL: missing $W/.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$W/.env"
PYTHON_BIN="${PYTHON_BIN:-/path/to/environment/env/gsa_train/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "FATAL: Python unavailable: $PYTHON_BIN" >&2; exit 2; }
EXPECTED_ENV="${EXPECTED_GSA_ENV:-/path/to/environment/env/gsa_train}"
PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
[[ "$PYTHON_PREFIX" == "$EXPECTED_ENV" ]] || { echo "FATAL: wrong Python env: $PYTHON_PREFIX (expected $EXPECTED_ENV)" >&2; exit 2; }
export PYTHON_BIN HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$W/.cache"/{hf,torch,triton,tmp,mpl}
if [[ "$RUN_GENERAL_RETENTION" == 1 ]]; then
  BASE_MODEL_CHECK="${BASE_MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
  E1_MODEL_CHECK="${E1_MODEL:-/path/to/Anvil/output/merged_e1}"
  [[ -d "$BASE_MODEL_CHECK" ]] || { echo "FATAL: base model directory is missing: $BASE_MODEL_CHECK" >&2; exit 2; }
  [[ -d "$E1_MODEL_CHECK" ]] || { echo "FATAL: E1 model directory is missing: $E1_MODEL_CHECK (mount GSA_Forge read-only)" >&2; exit 2; }
fi
if [[ -e "$RUN_ROOT" ]]; then
  [[ "$RESUME_RUN" == 1 && -f "$MANIFEST" && -f "$STATUS" ]] || {
    echo "FATAL: stale RUN_ROOT; use a fresh RUN_ID or RESUME_RUN=1 for this exact run: $RUN_ROOT" >&2
    exit 2
  }
else
  [[ "$RESUME_RUN" == 0 ]] || { echo "FATAL: cannot resume missing RUN_ROOT: $RUN_ROOT" >&2; exit 2; }
  mkdir -p "$LOG_ROOT"
fi
mkdir -p "$(dirname "$LOCK")" "$LOG_ROOT"
exec 9>"$LOCK"; flock -n 9 || { echo "FATAL: another 4xL20 run holds $LOCK" >&2; exit 3; }
MASTER_LOG="$LOG_ROOT/master.log"
if [[ "$RESUME_RUN" == 0 ]]; then
  printf 'timestamp_utc\tstage\tstate\trc\tdetail\n' > "$STATUS"
  : > "$MASTER_LOG"
fi
say() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$MASTER_LOG"; }
record() { printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$4" | tee -a "$STATUS" "$MASTER_LOG"; }

NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"; ACTIVE_PID=""; ACTIVE_PGID=""; MONITOR_PID=""; PIPELINE_TERMINAL=0
cleanup() {
  local rc=$?; trap - EXIT INT TERM
  [[ -z "$MONITOR_PID" ]] || kill "$MONITOR_PID" 2>/dev/null || true
  if [[ -n "$ACTIVE_PGID" ]] && kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; then
    kill -TERM -- "-$ACTIVE_PGID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 -- "-$ACTIVE_PGID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$ACTIVE_PGID" 2>/dev/null || true
  fi
  [[ -z "$ACTIVE_PID" ]] || wait "$ACTIVE_PID" 2>/dev/null || true
  if [[ "$rc" != 0 && "$PIPELINE_TERMINAL" == 0 ]]; then
    record pipeline FAILED "$rc" "orchestrator exited before all requested stages completed"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

HW_JSON="$RUN_ROOT/hardware_pre.json"
query_hardware() {
  local g p
  g=$("$NVIDIA_SMI_BIN" --id="$GPU_SET" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits) || return 1
  p=$("$NVIDIA_SMI_BIN" --id="$GPU_SET" --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits) || return 1
  "$PYTHON_BIN" - "$g" "$p" "$HW_JSON" <<'PY'
import csv, json, os, pathlib, sys, tempfile
g_text, p_text, out = sys.argv[1:]
gpus = []
for row in csv.reader(g_text.splitlines()):
    if not row: continue
    if len(row) != 6: raise SystemExit(f"invalid GPU row: {row!r}")
    index, uuid, name, total, used, util = (x.strip() for x in row)
    try: total, used, util = int(total), int(used), int(util)
    except ValueError as exc: raise SystemExit(f"invalid GPU numbers: {row!r}") from exc
    gpus.append({"index": index, "uuid": uuid, "name": name, "total_mib": total, "used_mib": used, "utilization": util})
if len(gpus) != 4 or len({x["uuid"] for x in gpus}) != 4:
    raise SystemExit("DLC requires exactly four unique physical GPUs")
for gpu in gpus:
    if "L20" not in gpu["name"].upper() or gpu["total_mib"] < 45056:
        raise SystemExit(f"GPU is not a >=44GiB L20: {gpu}")
procs = []
for row in csv.reader(p_text.splitlines()):
    if not row or all(not x.strip() for x in row): continue
    if len(row) != 4: raise SystemExit(f"invalid compute row: {row!r}")
    procs.append(dict(zip(("gpu_uuid","pid","process_name","used_gpu_memory"), (x.strip() for x in row))))
payload = {"contract":"exactly_4_unique_physical_L20_at_least_44GiB", "gpus":gpus, "compute_processes":procs, "idle":not procs}
path = pathlib.Path(out)
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
if procs: raise SystemExit(f"GPU node is busy: {procs}")
PY
}
if ! query_hardware; then record preflight HARDWARE_FAILED 4 "nvidia-smi query failed or node busy; see $HW_JSON"; exit 4; fi
DISK_KB=$(df -Pk "$W" | awk 'NR==2{print $4}')
[[ "$DISK_KB" =~ ^[0-9]+$ && "$DISK_KB" -ge $((MIN_DISK_GB*1024*1024)) ]] || { record preflight DISK_FAILED 4 "free_kb=$DISK_KB required_gb=$MIN_DISK_GB"; exit 4; }

"$PYTHON_BIN" - "$W" "$RUN_ROOT" "$RUN_ID" "$RESUME_RUN" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys, tempfile
w, root = map(pathlib.Path, sys.argv[1:3])
run_id, resume = sys.argv[3], bool(int(sys.argv[4]))
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()
files = ["scripts/run_gsa_4xl20_dlc.sh", "scripts/run_general_retention_dp.sh",
         "scripts/run_general_retention_v3_dlc.sh", "eval/general_retention_formal_v3.py",
         "configs/eval/general_retention_formal_v3.yaml",
         "tools/check_scannet_rgb_depth_v52_unique_referent_formal.py",
         "eval/scannet_rgb_depth_v52_eval.py",
         "eval/scannet_rgb_depth_v52_dlc.py",
         "eval/scannet_rgb_depth_v51_dlc.py",
         "eval/scannet_rgb_depth_v52_interface.py",
         "eval/gsa_eval/runner.py",
         "experiments/scannet_rgb_depth_v52_gs_intervention_contract.json",
         "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh"]
payload={"schema_version":2,"run_id":run_id,"root":str(w.resolve()),
         "git_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=w,text=True,capture_output=True,check=False).stdout.strip(),
         "tracked_code":{name:sha(w/name) for name in files if (w/name).is_file()},
         "environment":{key:__import__("os").environ.get(key) for key in ("CUDA_VISIBLE_DEVICES","PYTHON_BIN","BASE_MODEL","E1_MODEL","HF_HUB_OFFLINE","WANDB_MODE")}}
path = root / "run_manifest.json"
if resume:
    old = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema_version", "run_id", "root", "git_commit", "tracked_code", "environment"):
        if old.get(key) != payload.get(key):
            raise SystemExit(f"resume manifest drift: {key}")
    payload["resume_count"] = int(old.get("resume_count", 0)) + 1
else:
    payload["resume_count"] = 0
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

compile_v52() {
  V52_REPORT="$RUN_ROOT/v52_formal_compile.json"
  if [[ ! -d "$V52_PACK" ]]; then
    V52_STATUS="PACK_MISSING"
    record v52_formal BLOCKED 0 "pack missing: $V52_PACK"
    return 0
  fi
  [[ -f "$V52_EVALUATOR" ]] || {
    V52_STATUS="EVALUATOR_MISSING"
    record v52_formal BLOCKED 0 "formal evaluator missing: $V52_EVALUATOR"
    return 0
  }
  local args=(gate --pack "$V52_PACK" --report "$V52_REPORT")
  local rc
  [[ ! -s "$V52_ADJUDICATION" ]] || args+=(--adjudication "$V52_ADJUDICATION")
  set +e
  "$PYTHON_BIN" "$V52_EVALUATOR" "${args[@]}" >"$LOG_ROOT/v52_formal_compile.log" 2>&1
  rc=$?
  set -e
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then
    record v52_formal FAILED "$rc" "invalid pack/review; log=$LOG_ROOT/v52_formal_compile.log"
    return "$rc"
  fi
  [[ -s "$V52_REPORT" ]] || {
    record v52_formal FAILED 2 "gate returned rc=$rc without report: $V52_REPORT"
    return 2
  }
  V52_STATUS=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$V52_REPORT")
  if [[ "$rc" == 0 && "$V52_STATUS" == FORMAL_PASS ]]; then
    record v52_formal PASS 0 "report=$V52_REPORT"
  else
    record v52_formal BLOCKED 0 "status=$V52_STATUS report=$V52_REPORT"
  fi
}
V52_STATUS="NOT_REQUESTED"
V52_LAUNCH_READY=0
V52_ADJUDICATION_ENV=""
[[ ! -s "$V52_ADJUDICATION" ]] || V52_ADJUDICATION_ENV="$V52_ADJUDICATION"
if [[ "$RUN_V52_PAIRED" != 0 ]]; then compile_v52 || exit 5; fi
if [[ "$V52_STATUS" == FORMAL_PASS ]]; then
  if [[ ! -x "$V52_PAIRED_LAUNCHER" ]]; then
    record v52_paired FAILED 2 \
      "FORMAL_PASS but the pinned protocol-matched v5.2 launcher is not executable"
    exit 5
  fi
  if env GSA_ROOT="$W" PYTHON_BIN="$PYTHON_BIN" CUDA_VISIBLE_DEVICES="$GPU_SET" \
    V52_FORMAL_REPORT="$V52_REPORT" PREFLIGHT_ONLY=1 THINKING=off \
    V52_ADJUDICATION="$V52_ADJUDICATION_ENV" \
    OUT="$RUN_ROOT/scannet_rgb_depth_v52_paired_preflight" \
    bash "$V52_PAIRED_LAUNCHER" >"$LOG_ROOT/v52_paired_preflight.log" 2>&1; then
    V52_LAUNCH_READY=1
    record v52_paired PREFLIGHT_READY 0 "log=$LOG_ROOT/v52_paired_preflight.log"
  else
    rc=$?
    record v52_paired FAILED "$rc" "FORMAL_PASS preflight failed log=$LOG_ROOT/v52_paired_preflight.log"
    exit 5
  fi
fi

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  if [[ "$RUN_GENERAL_RETENTION" == 1 ]]; then
    env GSA_ROOT="$W" PYTHON_BIN="$PYTHON_BIN" CUDA_VISIBLE_DEVICES="$GPU_SET" \
      RUN_STAGE=paired THINKING=off BATCH=16 PREFLIGHT_ONLY=1 \
      RESUME="$RESUME_RUN" OUT="$RUN_ROOT/general_retention_base_e1" \
      bash "$W/scripts/run_general_retention_dp.sh" \
      >>"$LOG_ROOT/general_retention_preflight.log" 2>&1 || {
        record preflight FAILED 5 "general-retention preflight failed; see $LOG_ROOT/general_retention_preflight.log"
        exit 5
      }
  fi
  record preflight READY 0 "exact4xL20 idle; source/model/code receipts verified; no CUDA context started"
  say "DLC_PREFLIGHT_READY: CPU/source/model checks complete; no model inference started"
  trap - EXIT INT TERM
  exit 0
fi

"$PYTHON_BIN" - <<'PY'
import torch
count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"DLC requires exactly four CUDA-visible GPUs, found {count}")
uuids = []
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    name = props.name
    gib = props.total_memory / 2**30
    uuid = str(getattr(props, "uuid", ""))
    if "L20" not in name.upper() or gib < 44:
        raise SystemExit(f"GPU {index} is not a >=44 GiB L20: {name} ({gib:.1f} GiB)")
    if "MIG" in name.upper() or uuid.upper().startswith("MIG") or not uuid:
        raise SystemExit(f"GPU {index} is not a physical L20 with a UUID")
    uuids.append(uuid)
if len(set(uuids)) != 4:
    raise SystemExit("CUDA-visible devices do not resolve to four unique physical GPUs")
print("torch hardware gate: exactly 4 physical L20 GPUs")
PY

record firstfault BLOCKED 0 "implementation and aligned training-eligible S/A release incomplete"
record mopd BLOCKED 0 "specialist eligibility, aligned S/A supervision, and matched controls incomplete"

run_stage() {
  local stage="$1" log="$2"; shift 2
  local trace="$LOG_ROOT/${stage}_gpu_trace.csv" reason="$LOG_ROOT/${stage}_monitor_reason.txt" warning="$LOG_ROOT/${stage}_monitor_warnings.log"
  : > "$reason"
  : > "$warning"
  printf 'timestamp_utc,gpu_index,uuid,memory_used_mib,memory_free_mib,utilization_gpu_pct\n' > "$trace"
  record "$stage" STARTING - "log=$log"
  set +e; setsid "$@" >> "$log" 2>&1 & ACTIVE_PID=$!; ACTIVE_PGID=$ACTIVE_PID
  record "$stage" RUNNING - "pid=$ACTIVE_PID pgid=$ACTIVE_PGID"
  (
    set +e
    while kill -0 "$ACTIVE_PID" 2>/dev/null; do
      now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      gpu_rows="$($NVIDIA_SMI_BIN --id="$GPU_SET" --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"
      if [[ $? -ne 0 || -z "$gpu_rows" ]]; then
        printf '%s\n' "GPU_MONITOR_QUERY_FAILED at $now" > "$reason"
        kill -TERM -- "-$ACTIVE_PGID" 2>/dev/null || true
        break
      fi
      while IFS= read -r row; do
        [[ -z "$row" ]] || printf '%s,%s\n' "$now" "$row" >> "$trace"
      done <<< "$gpu_rows"
      proc_rows="$($NVIDIA_SMI_BIN --id="$GPU_SET" --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)"
      if [[ $? -ne 0 ]]; then
        printf '%s\n' "GPU_PROCESS_QUERY_FAILED at $now" > "$reason"
        kill -TERM -- "-$ACTIVE_PGID" 2>/dev/null || true
        break
      fi
      foreign=""; unknown=""; total_proc=0; declare -A gpu_proc_counts=()
      while IFS=',' read -r uuid pid pname mem; do
        pid="${pid//[[:space:]]/}"
        [[ -z "$pid" ]] && continue
        total_proc=$((total_proc + 1))
        uuid="${uuid//[[:space:]]/}"
        gpu_proc_counts["$uuid"]=$(( ${gpu_proc_counts["$uuid"]:-0} + 1 ))
        if [[ ! -r "/proc/$pid/stat" ]]; then
          unknown+="pid=$pid(uuid=$uuid);"
          continue
        fi
        pgrp="$(awk '{print $5}' "/proc/$pid/stat" 2>/dev/null)"
        owned=0; cur="$pid"
        for _ in $(seq 1 32); do
          [[ "$cur" == "$ACTIVE_PID" ]] && { owned=1; break; }
          [[ -r "/proc/$cur/stat" ]] || break
          parent="$(awk '{print $4}' "/proc/$cur/stat" 2>/dev/null)"
          [[ -n "$parent" && "$parent" != 1 && "$parent" != "$cur" ]] || break
          cur="$parent"
        done
        [[ "$pgrp" == "$ACTIVE_PGID" || "$owned" == 1 ]] || foreign+="pid=$pid(pgid=${pgrp:-unknown});"
      done <<< "$proc_rows"
      # In DLC containers nvidia-smi may expose host PIDs that are intentionally
      # absent from the container's /proc namespace. Record them, but do not
      # terminate a healthy stage solely for that namespace boundary. A second
      # compute process on one GPU is still a hard interference signal.
      if [[ -n "$unknown" ]]; then
        sig="$unknown"
        [[ "${last_unknown:-}" == "$sig" ]] || { printf '%s UNATTRIBUTABLE_GPU_PID %s\n' "$now" "$sig" >> "$warning"; last_unknown="$sig"; }
      fi
      overfull=""
      for uuid in "${!gpu_proc_counts[@]}"; do
        [[ "${gpu_proc_counts[$uuid]}" -le 1 ]] || overfull+="$uuid(count=${gpu_proc_counts[$uuid]});"
      done
      if [[ -n "$foreign" || "$total_proc" -gt 4 || -n "$overfull" ]]; then
        printf '%s\n' "EXTERNAL_GPU_PROCESS at $now: $foreign" > "$reason"
        [[ -z "$overfull" ]] || printf '%s\n' "GPU_PROCESS_COUNT_OVERCOMMITTED at $now: $overfull" >> "$reason"
        kill -TERM -- "-$ACTIVE_PGID" 2>/dev/null || true
        break
      fi
      sleep 5
    done
  ) & MONITOR_PID=$!
  wait "$ACTIVE_PID"; local rc=$?; set -e
  if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; MONITOR_PID=""; fi
  ACTIVE_PID=""; ACTIVE_PGID=""
  if [[ -s "$reason" && "$rc" == 0 ]]; then rc=75; fi
  if [[ "$rc" == 0 ]]; then record "$stage" SUCCEEDED 0 "log=$log trace=$trace"; else record "$stage" FAILED "$rc" "log=$log trace=$trace reason=$(tr '\n' ' ' < "$reason")"; return "$rc"; fi
}

if [[ "$RUN_GENERAL_RETENTION" == 1 ]]; then
  run_stage general_retention "$LOG_ROOT/general_retention.log" env \
    GSA_ROOT="$W" PYTHON_BIN="$PYTHON_BIN" CUDA_VISIBLE_DEVICES="$GPU_SET" \
    RUN_STAGE=paired THINKING=off BATCH=16 PREFLIGHT_ONLY=0 \
    IMAGE_MAX_TOKEN_NUM=4096 RESUME="$RESUME_RUN" \
    OUT="$RUN_ROOT/general_retention_base_e1" \
    bash "$W/scripts/run_general_retention_dp.sh" || exit 6
else
  record general_retention SKIPPED 0 "RUN_GENERAL_RETENTION=0"
fi

if [[ "$RUN_V52_PAIRED" == 0 ]]; then
  record v52_paired SKIPPED 0 "RUN_V52_PAIRED=0"
elif [[ "$V52_STATUS" != FORMAL_PASS ]]; then
  if [[ "$RUN_V52_PAIRED" == auto ]]; then
    record v52_paired SKIPPED_PENDING_ANNOTATION 0 "compile_status=$V52_STATUS"
  else
    record v52_paired BLOCKED_ANNOTATION_GATE 0 "compile_status=$V52_STATUS"
  fi
elif [[ ! -x "$V52_PAIRED_LAUNCHER" ]]; then
  record v52_paired FAILED 2 "FORMAL_PASS but v5.2 launcher disappeared after preflight"
  exit 6
elif [[ "$V52_LAUNCH_READY" != 1 ]]; then
  record v52_paired FAILED 2 "FORMAL_PASS but v5.2 preflight readiness was lost"
  exit 6
else
  run_stage v52_paired "$LOG_ROOT/v52_paired.log" env \
    GSA_ROOT="$W" PYTHON_BIN="$PYTHON_BIN" CUDA_VISIBLE_DEVICES="$GPU_SET" \
    V52_FORMAL_REPORT="$V52_REPORT" V52_ADJUDICATION="$V52_ADJUDICATION_ENV" \
    PREFLIGHT_ONLY=0 THINKING=off \
    OUT="$RUN_ROOT/scannet_rgb_depth_v52_paired" \
    bash "$V52_PAIRED_LAUNCHER" || exit 6
fi
record pipeline COMPLETE_WITH_BLOCKED 0 \
  "claim-bearing runnable stages completed; FirstFault/MOPD and any pending v5.2 review stay explicit"
say "DLC_RUN_COMPLETE_WITH_BLOCKED: status=$STATUS manifest=$MANIFEST"
PIPELINE_TERMINAL=1
trap - EXIT INT TERM
exit 0
