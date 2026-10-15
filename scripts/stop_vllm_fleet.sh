#!/usr/bin/env bash
# Stop all vLLM replicas started by serve_vllm_fleet.sh / serve_vllm_qwen25vl.sh.
set -uo pipefail
ANVIL_REPO="${GSA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PIDFILE="${PIDFILE:-$ANVIL_REPO/logs/vlm_serve/fleet.pids}"

if [[ -f "$PIDFILE" ]]; then
  while read -r pid rest; do
    [[ "${pid:-}" =~ ^[1-9][0-9]*$ ]] || continue
    [[ -r "/proc/$pid/cmdline" ]] || continue
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'vllm.entrypoints.openai.api_server' || continue
    [[ "$(readlink "/proc/$pid/cwd")" == "$ANVIL_REPO" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop] killing pid $pid ($rest)"
      kill "$pid" 2>/dev/null || true
    fi
  done < "$PIDFILE"
  sleep 3
fi

# Do not terminate servers belonging to another checkout or user.
echo "[stop] remaining:"
pgrep -af "vllm.entrypoints.openai.api_server" | grep -v pgrep || echo "  none"
