#!/usr/bin/env bash
# Generate repaired v3 labels from the frozen, privacy-screened candidate index.
set -euo pipefail

W="$(cd "$(dirname "$0")/.." && pwd)"
cd "$W"
source "$W/.env"

INDEX="${INDEX:-$W/artifacts/manifests/v3_generation.index}"
OUT="${OUT:-$W/data/flywheel_v3/unseen_clean}"
V3_CACHE="${V3_CACHE:-$W/logs/vlm_cache_v3}"
V3_REQUEST_LOG="${V3_REQUEST_LOG:-$W/logs/vlm_requests_v3.jsonl}"
WORKERS="${WORKERS:-24}"
TARGET="${TARGET:-100000000}"
LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
RESUME="${RESUME:-0}"
ALLOW_EXTERNAL_UPLOAD="${ALLOW_EXTERNAL_UPLOAD:-0}"
AUDIT_CANDIDATE_MODE="${AUDIT_CANDIDATE_MODE:-0}"

[[ -s "$INDEX" ]] || { echo "FATAL: missing v3 generation index: $INDEX" >&2; exit 2; }
[[ -n "${GSA_VLM_URL:-}" ]] || { echo "FATAL: GSA_VLM_URL is unset" >&2; exit 2; }
[[ -n "${GSA_VLM_KEY:-}" ]] || { echo "FATAL: GSA_VLM_KEY is unset" >&2; exit 2; }
[[ -n "${GSA_VLM_MODEL:-}" ]] || { echo "FATAL: GSA_VLM_MODEL is unset" >&2; exit 2; }
[[ "$WORKERS" =~ ^[0-9]+$ ]] && (( WORKERS >= 1 && WORKERS <= 32 )) \
  || { echo "FATAL: WORKERS must be an integer in [1,32]" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] \
  || { echo "FATAL: LIMIT must be a non-negative integer" >&2; exit 2; }
[[ "$OFFSET" =~ ^[0-9]+$ ]] \
  || { echo "FATAL: OFFSET must be a non-negative integer" >&2; exit 2; }
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] \
  || { echo "FATAL: RESUME must be 0 or 1" >&2; exit 2; }
[[ "$ALLOW_EXTERNAL_UPLOAD" == 1 ]] || {
  echo "FATAL: external image upload is disabled; set ALLOW_EXTERNAL_UPLOAD=1 only after explicit approval" >&2
  exit 2
}
[[ "$AUDIT_CANDIDATE_MODE" == 0 || "$AUDIT_CANDIDATE_MODE" == 1 ]] || {
  echo "FATAL: AUDIT_CANDIDATE_MODE must be 0 or 1" >&2
  exit 2
}

index_images="$(wc -l < "$INDEX")"
available_images=$(( index_images > OFFSET ? index_images - OFFSET : 0 ))
if (( LIMIT > 0 && LIMIT < available_images )); then
  upload_images="$LIMIT"
else
  upload_images="$available_images"
fi
printf 'external_upload endpoint=%s model=%s images=%s offset=%s index=%s\n' "$GSA_VLM_URL" "$GSA_VLM_MODEL" "$upload_images" "$OFFSET" "$INDEX"
printf 'candidate_mode=%s (1 means human-audit only; never train directly)\n' \
  "$AUDIT_CANDIDATE_MODE"

resume_args=()
if [[ "$RESUME" == 1 ]]; then
  [[ -d "$OUT" ]] || { echo "FATAL: resume output does not exist: $OUT" >&2; exit 2; }
  resume_args+=(--resume)
else
  [[ ! -e "$OUT" ]] || { echo "FATAL: refusing non-fresh output: $OUT" >&2; exit 2; }
  [[ ! -e "$V3_CACHE" ]] || { echo "FATAL: refusing non-fresh v3 cache: $V3_CACHE" >&2; exit 2; }
  [[ ! -e "$V3_REQUEST_LOG" ]] \
    || { echo "FATAL: refusing existing v3 request log: $V3_REQUEST_LOG" >&2; exit 2; }
fi

export GSA_VLM_CACHE="$V3_CACHE"
export GSA_VLM_REQUEST_LOG="$V3_REQUEST_LOG"
# This is the configured self-hosted teacher; bypass the container's generic
# egress proxy for its exact host without embedding deployment addresses in code.
VLM_HOST="$(python -c 'import os, urllib.parse; host=urllib.parse.urlparse(os.environ["GSA_VLM_URL"]).hostname; assert host; print(host)')"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost},$VLM_HOST"
export no_proxy="$NO_PROXY"

exec python trains/flywheel/generate_multiform.py \
  --index "$INDEX" \
  --out "$OUT" \
  --workers "$WORKERS" \
  --target "$TARGET" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  "${resume_args[@]}"
