#!/usr/bin/env bash
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:-$(cat "$W/artifacts/local4_queue/current_run.txt" 2>/dev/null || true)}"
[[ -n "$RUN_ID" ]] || { echo "no local4 queue has been registered"; exit 1; }
RUN_DIR="$W/artifacts/local4_queue/$RUN_ID"

echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
if [[ -s "$RUN_DIR/queue.pid" ]]; then
  pid=$(cat "$RUN_DIR/queue.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "queue_pid=$pid RUNNING"; else echo "queue_pid=$pid EXITED"; fi
else
  echo "queue_pid=MISSING"
fi
echo "--- status ---"
tail -n 20 "$RUN_DIR/status.tsv" 2>/dev/null || true
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
echo "--- active compute ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null || true
echo "--- current job tail ---"
if [[ -s "$RUN_DIR/current_job.tsv" ]]; then
  IFS=$'\t' read -r current_job current_log < "$RUN_DIR/current_job.tsv"
  echo "current_job=$current_job"
  [[ -n "$current_log" && -f "$current_log" ]] && tail -n 40 "$current_log" || true
else
  echo "current_job=NONE"
fi
