#!/usr/bin/env bash
# Launch a long-running GSA-Forge job fully detached from the caller's process
# group, so that killing/exiting the invoking shell never takes the job with it.
#   bash scripts/launch_bg.sh <logfile> <cmd...>
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="$1"; shift
mkdir -p "$(dirname "$LOG")"
setsid nohup "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "launched pid=$PID log=$LOG"
sleep 3
if kill -0 "$PID" 2>/dev/null; then echo "alive"; else echo "DIED EARLY:"; tail -5 "$LOG"; fi
