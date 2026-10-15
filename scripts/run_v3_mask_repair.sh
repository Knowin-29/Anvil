#!/usr/bin/env bash
set -euo pipefail
W="$(cd "$(dirname "$0")/.." && pwd)"
cd "$W"
source "$W/.env"

[[ "${ALLOW_EXTERNAL_UPLOAD:-0}" == 1 ]] || {
  echo "FATAL: set ALLOW_EXTERNAL_UPLOAD=1 after explicit approval" >&2
  exit 2
}
DATA="${DATA:-$W/data/flywheel_v3/unseen_clean_audit_v34}"
OUT="${OUT:-$W/data/flywheel_v3/mask_repair_v34}"
CACHE="${CACHE:-$W/logs/vlm_cache_v3_mask_repair_v34}"
REQUEST_LOG="${REQUEST_LOG:-$W/logs/vlm_requests_v3_mask_repair_v34.jsonl}"
[[ -d "$DATA" ]] || { echo "FATAL: missing data root: $DATA" >&2; exit 2; }
[[ ! -e "$OUT" && ! -e "$CACHE" && ! -e "$REQUEST_LOG" ]] || {
  echo "FATAL: repair output/cache/log must be fresh" >&2
  exit 2
}
export GSA_VLM_CACHE="$CACHE"
export GSA_VLM_REQUEST_LOG="$REQUEST_LOG"
exec python tools/repair_v3_masks.py --data "$DATA" --out "$OUT"
