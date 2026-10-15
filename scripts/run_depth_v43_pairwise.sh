#!/usr/bin/env bash
# Guarded launcher for the isolated v4.3 pairwise depth protocol.
set -euo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$W"
source "$W/.env"

INDEX="${INDEX:-$W/artifacts/manifests/v3_generation.index}"
DETECTION="${DETECTION:-$W/data/viewer_v3_audit_ready_v34/labels/G_grounding/detection.jsonl}"
BBOX="${BBOX:-$W/data/viewer_v3_audit_ready_v34/labels/G_grounding/bbox.jsonl}"
V2_MANIFEST="${V2_MANIFEST:-$W/data/semantic_audit_v2/manifest.json}"
V3_MANIFEST="${V3_MANIFEST:-$W/data/semantic_audit_v3/manifest.json}"
CANARY="${CANARY:-0}"
if [[ "$CANARY" == 8 ]]; then
  DEFAULT_OUT="$W/data/depth_v43_pairwise/canary"
  STAGE="canary"
else
  DEFAULT_OUT="$W/data/depth_v43_pairwise/generated"
  STAGE="full"
fi
OUT="${OUT:-$DEFAULT_OUT}"
V43_CACHE="${V43_CACHE:-$W/logs/vlm_cache_depth_v43_pairwise}"
V43_REQUEST_LOG="${V43_REQUEST_LOG:-$W/logs/vlm_requests_depth_v43_pairwise.jsonl}"
WORKERS="${WORKERS:-8}"
TARGET="${TARGET:-40}"
SEED="${SEED:-20260821}"
CHECK_ONLY="${CHECK_ONLY:-0}"
ALLOW_EXTERNAL_UPLOAD="${ALLOW_EXTERNAL_UPLOAD:-0}"
CANARY_APPROVED="${CANARY_APPROVED:-0}"
CANARY_REPORT="${CANARY_REPORT:-$W/data/depth_v43_pairwise/canary/generation_report.json}"
HUMAN_REVIEW="${HUMAN_REVIEW:-$W/data/depth_v43_pairwise/canary/human_review.json}"
export AUTHORIZED_ENDPOINT="${AUTHORIZED_ENDPOINT:-}"

for path in "$INDEX" "$DETECTION" "$BBOX" "$V2_MANIFEST" "$V3_MANIFEST"; do
  [[ -s "$path" ]] || { echo "FATAL: missing required input: $path" >&2; exit 2; }
done
[[ "$WORKERS" =~ ^[0-9]+$ ]] && (( WORKERS >= 1 && WORKERS <= 16 )) \
  || { echo "FATAL: WORKERS must be an integer in [1,16]" >&2; exit 2; }
[[ "$TARGET" == 40 ]] \
  || { echo "FATAL: depth-v4.3 pairwise protocol requires TARGET=40" >&2; exit 2; }
[[ "$SEED" == 20260821 ]] \
  || { echo "FATAL: depth-v4.3 frozen protocol requires SEED=20260821" >&2; exit 2; }
[[ "$CHECK_ONLY" == 0 || "$CHECK_ONLY" == 1 ]] \
  || { echo "FATAL: CHECK_ONLY must be 0 or 1" >&2; exit 2; }
[[ "$CANARY" == 0 || "$CANARY" == 8 ]] \
  || { echo "FATAL: CANARY must be 0 or 8" >&2; exit 2; }

ARGS=(
  --index "$INDEX"
  --detection "$DETECTION"
  --bbox "$BBOX"
  --exclude-manifest "$V2_MANIFEST"
  --exclude-manifest "$V3_MANIFEST"
  --out "$OUT"
  --workers "$WORKERS"
  --target "$TARGET"
  --seed "$SEED"
)
if [[ "$CANARY" == 8 ]]; then
  ARGS+=(--canary 8)
else
  ARGS+=(--canary-report "$CANARY_REPORT" --human-review "$HUMAN_REVIEW")
fi
if [[ "$CHECK_ONLY" == 1 ]]; then
  exec python tools/generate_depth_v43_pairwise.py "${ARGS[@]}" --check-only
fi

[[ "$ALLOW_EXTERNAL_UPLOAD" == 1 ]] || {
  echo "FATAL: external image upload is disabled; set ALLOW_EXTERNAL_UPLOAD=1 only after explicit approval" >&2
  exit 2
}
if [[ "$STAGE" == "full" && "$CANARY_APPROVED" != 1 ]]; then
  echo "FATAL: full upload requires CANARY_APPROVED=1 after reviewing the v4.3 canary" >&2
  exit 2
fi
if [[ "$STAGE" == "full" ]]; then
  [[ -s "$CANARY_REPORT" ]] || {
    echo "FATAL: full upload requires a completed canary report: $CANARY_REPORT" >&2
    exit 2
  }
  [[ -s "$HUMAN_REVIEW" ]] || {
    echo "FATAL: full upload requires the hash-bound human canary review: $HUMAN_REVIEW" >&2
    exit 2
  }
  python tools/generate_depth_v43_pairwise.py "${ARGS[@]}" \
    --validate-canary-binding-only >/dev/null || {
    echo "FATAL: v4.3 machine/human canary binding preflight failed" >&2
    exit 2
  }
fi
[[ -n "${GSA_VLM_URL:-}" ]] || { echo "FATAL: GSA_VLM_URL is unset" >&2; exit 2; }
[[ -n "${GSA_VLM_KEY:-}" ]] || { echo "FATAL: GSA_VLM_KEY is unset" >&2; exit 2; }
[[ -n "${GSA_VLM_MODEL:-}" ]] || { echo "FATAL: GSA_VLM_MODEL is unset" >&2; exit 2; }
[[ "${GSA_VLM_URL%/}" == "${AUTHORIZED_ENDPOINT%/}" ]] || {
  echo "FATAL: endpoint differs from the explicitly authorized endpoint" >&2
  exit 2
}
[[ ! -e "$OUT" ]] || { echo "FATAL: refusing non-fresh output: $OUT" >&2; exit 2; }
if [[ -e "$V43_CACHE" && ! -d "$V43_CACHE" ]]; then
  echo "FATAL: v4.3 cache path exists but is not a directory: $V43_CACHE" >&2
  exit 2
fi
if [[ -e "$V43_REQUEST_LOG" && ! -f "$V43_REQUEST_LOG" ]]; then
  echo "FATAL: v4.3 request log path exists but is not a file: $V43_REQUEST_LOG" >&2
  exit 2
fi

export GSA_VLM_CACHE="$V43_CACHE"
export GSA_VLM_REQUEST_LOG="$V43_REQUEST_LOG"
export GSA_V43_STAGE="$STAGE"
export GSA_DEPTH_V43_EXTERNAL_UPLOAD_AUTHORIZED=1
export GSA_DEPTH_V43_CANARY_APPROVED="$CANARY_APPROVED"
export GSA_DEPTH_V43_CANARY_REPORT="$CANARY_REPORT"
export GSA_DEPTH_V43_HUMAN_REVIEW="$HUMAN_REVIEW"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
printf 'external_upload protocol=v4.3-pairwise stage=%s endpoint=%s model=%s target=%s index=%s cache=%s log_mode=append\n' \
  "$STAGE" "$GSA_VLM_URL" "$GSA_VLM_MODEL" "$TARGET" "$INDEX" "$V43_CACHE"
exec python tools/generate_depth_v43_pairwise.py "${ARGS[@]}"
