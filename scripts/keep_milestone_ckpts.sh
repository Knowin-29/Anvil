#!/bin/bash
# Watch a swift training output dir and permanently keep milestone checkpoints
# (doubling schedule: 1000/2000/4000/8000/16000/...) via hardlinks, so that
# save_total_limit rolling-deletion does not lose them.
#
# Usage:
#   bash scripts/keep_milestone_ckpts.sh <run_dir> [poll_seconds]
# Example:
#   nohup bash scripts/keep_milestone_ckpts.sh \
#       output/sft_27b_teacher/v4-20260609-180043 120 \
#       > logs/keep_milestone.log 2>&1 &
set -u

RUN_DIR="${1:?usage: keep_milestone_ckpts.sh <run_dir> [poll_seconds]}"
POLL="${2:-120}"
KEEP_DIR="${RUN_DIR%/}/keep_milestones"
mkdir -p "${KEEP_DIR}"

# Milestone steps: doubling schedule. Extend upper bound as needed.
MILESTONES="1000 2000 4000 8000 16000 32000 64000 128000 256000"

is_complete() {
    # A checkpoint is complete when config.json + model.safetensors.index
    # (or at least one shard) exist and dir is not being written.
    local d="$1"
    [ -f "${d}/config.json" ] || return 1
    ls "${d}"/*.safetensors >/dev/null 2>&1 || return 1
    return 0
}

keep_one() {
    local step="$1"
    local src="${RUN_DIR%/}/checkpoint-${step}"
    local dst="${KEEP_DIR}/checkpoint-${step}"
    [ -e "${dst}" ] && return 0          # already kept
    [ -d "${src}" ] || return 0          # not produced yet (or already deleted)
    is_complete "${src}" || return 0     # still being written
    # Hardlink: instant, shares inodes, survives source deletion.
    if cp -al "${src}" "${dst}" 2>/dev/null; then
        echo "[$(date '+%F %T')] KEPT milestone checkpoint-${step} -> ${dst}"
    else
        echo "[$(date '+%F %T')] WARN hardlink failed for ${step}, fallback copy"
        cp -r "${src}" "${dst}" && echo "[$(date '+%F %T')] KEPT (copy) checkpoint-${step}"
    fi
}

echo "[$(date '+%F %T')] milestone watcher started: run=${RUN_DIR} poll=${POLL}s"
echo "[$(date '+%F %T')] milestones: ${MILESTONES}"

while true; do
    for step in ${MILESTONES}; do
        keep_one "${step}"
    done
    # Stop watching once training process is gone AND highest milestone passed.
    if ! pgrep -f "swift/cli/sft.py" >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] no training process detected; final sweep then exit"
        for step in ${MILESTONES}; do keep_one "${step}"; done
        break
    fi
    sleep "${POLL}"
done
echo "[$(date '+%F %T')] milestone watcher exiting"
