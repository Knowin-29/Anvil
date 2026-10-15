#!/usr/bin/env bash
# Anvil-Gen v3: bounded, resumable production. No model calls without --execute.
set -Eeuo pipefail
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${ANVIL_PYTHON:-python}"
CONFIG="$SCRIPT_DIR/configs/alignment.yaml"
OUT="$SCRIPT_DIR/outputs/alignment-production"
TARGET_RECORDS=5220000
TARGET_SOURCE_IMAGES=160000
TARGET_NEW_IMAGES=0
WORKERS=4
LIMIT=0
BASELINE=""
EXECUTE=0
BACKGROUND=0
LOCKED=0
MANIFESTS=()
ORIGINAL_ARGS=("$@")

usage() {
  echo 'Usage: bash generate_to_scale.sh --manifest scenes.jsonl [options]'
  echo '  --manifest PATH          Repeat for multiple approved v3 scene manifests.'
  echo '  --baseline-images PATH   Original image inventory: JSONL {"images":[...]} or {"image":"..."}.'
  echo '  --target-records N       Accepted v3 TRAIN records (default 5220000; benchmark counted separately).'
  echo '  --target-source-images N  Total distinct source images required before calls (default 160000).'
  echo '  --target-new-images N    Optional additional accepted TRAIN image goal (default 0).'
  echo '  --workers N              Concurrent scenes (default 4); endpoints manage GPU allocation.'
  echo '  --limit N                At most N scenes this invocation; 0 = all remaining input.'
  echo '  --out DIR --config FILE  Persistent output/ledger directory and role configuration.'
  echo '  --execute                Enable model requests; otherwise preflight only.'
  echo '  --background             Detach with nohup; logs and final exit code stay in --out.'
  echo 'Environment: ANVIL_PYTHON and ANVIL_{G,S,A,STRONG}_{URL,MODEL,KEY}.'
  echo 'Exit: 0 success/preflight; 2 configuration/runtime errors; 3 production target unmet; 75 locked.'
}
die() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --execute) EXECUTE=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    --_locked) LOCKED=1; BACKGROUND=0; shift ;;
    --manifest|--baseline-images|--target-records|--target-source-images|--target-new-images|--workers|--limit|--out|--config)
      (($# >= 2)) || die "Missing value for $1"
      case "$1" in
        --manifest) MANIFESTS+=("$2") ;;
        --baseline-images) BASELINE="$2" ;;
        --target-records) TARGET_RECORDS="$2" ;;
        --target-source-images) TARGET_SOURCE_IMAGES="$2" ;;
        --target-new-images) TARGET_NEW_IMAGES="$2" ;;
        --workers) WORKERS="$2" ;;
        --limit) LIMIT="$2" ;;
        --out) OUT="$2" ;;
        --config) CONFIG="$2" ;;
      esac
      shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
for n in "$TARGET_RECORDS" "$TARGET_SOURCE_IMAGES" "$TARGET_NEW_IMAGES" "$WORKERS" "$LIMIT"; do
  [[ "$n" =~ ^(0|[1-9][0-9]{0,11})$ ]] || die 'Counts must be nonnegative decimal integers.'
done
((WORKERS > 0 && TARGET_RECORDS > 0)) || die 'workers and target-records must be positive.'
((${#MANIFESTS[@]})) || die 'At least one --manifest is required.'
((TARGET_NEW_IMAGES == 0)) || [[ -n "$BASELINE" ]] || die 'Additional-image target requires --baseline-images.'
for input in "${MANIFESTS[@]}" "$CONFIG"; do [[ -r "$input" ]] || die "Unreadable input: $input"; done
[[ -z "$BASELINE" || -r "$BASELINE" ]] || die "Unreadable baseline: $BASELINE"
command -v "$PYTHON_BIN" >/dev/null || die "Python executable unavailable: $PYTHON_BIN"
command -v flock >/dev/null || die 'flock is required.'
mkdir -p -- "$OUT"
OUT="$(realpath -- "$OUT")"

if ((LOCKED == 0)); then
  if ((BACKGROUND)); then
    ((EXECUTE)) || die '--background requires --execute; inspect preflight in the foreground.'
    LOG="$(mktemp "$OUT/production.XXXXXXXX.log")"
    nohup flock -n -E 75 "$OUT/production.lock" bash "$SCRIPT_PATH" \
      "${ORIGINAL_ARGS[@]}" --_locked >"$LOG" 2>&1 </dev/null &
    printf 'Supervisor PID: %s\nLog: %s\nFinal status: %s/exit_code\n' "$!" "$LOG" "$OUT"
    exit 0
  fi
  exec flock -n -E 75 "$OUT/production.lock" bash "$SCRIPT_PATH" "${ORIGINAL_ARGS[@]}" --_locked
fi
# Resolve paths before changing directory; credentials are never echoed or copied.
ARGS=()
for manifest in "${MANIFESTS[@]}"; do ARGS+=(--manifest "$(realpath -- "$manifest")"); done
CONFIG="$(realpath -- "$CONFIG")"
[[ -z "$BASELINE" ]] || ARGS+=(--baseline-images "$(realpath -- "$BASELINE")")
if [[ "$PYTHON_BIN" == */* ]]; then PYTHON_BIN="$(realpath -s -- "$PYTHON_BIN")"; fi
cd -- "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$OUT/exit_code"
  echo "Finished with exit code $rc (0=success/preflight, 3=target not reached)."
  exit "$rc"
}
trap finish EXIT
printf 'running\n' >"$OUT/exit_code"
((EXECUTE == 0)) || ARGS+=(--execute)
"$PYTHON_BIN" -m anvil align run "${ARGS[@]}" --config "$CONFIG" \
  --ledger "$OUT/ledger.sqlite" --workers "$WORKERS" --limit "$LIMIT" \
  --min-source-images "$TARGET_SOURCE_IMAGES" \
  --target-records "$TARGET_RECORDS" --target-new-images "$TARGET_NEW_IMAGES"
if ((EXECUTE == 0)); then
  echo 'Preflight complete. No model calls made; target attainment is not implied.'
  exit 0
fi
# Export only after both targets are reached. Each release is immutable; a failed
# export remains explicitly partial and cannot be mistaken for a finished set.
STAGING="$(mktemp -d "$OUT/export.XXXXXXXX.partial")"
for split in train dev benchmark; do
  "$PYTHON_BIN" -m anvil align export --ledger "$OUT/ledger.sqlite" \
    --split "$split" --out "$STAGING/$split.jsonl"
done
"$PYTHON_BIN" scripts/validate_grpo_dataset.py "$STAGING/train.jsonl" --alignment
RELEASE="${STAGING%.partial}"
mv -T -- "$STAGING" "$RELEASE"
printf 'Validated release: %s\n' "$RELEASE"
