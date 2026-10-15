#!/usr/bin/env bash
# 100h+ supervised flywheel generation.
#
#   SHARD=0 NSHARD=4 bash scripts/run_flywheel_longrun.sh
#
# Per shard: takes a disjoint slice of the embodied index, generates all 8 task
# forms with validation, and RESUMES from done_images.txt after any crash/restart.
# The loop restarts the generator if it exits non-zero, audits quality every cycle,
# and stops cleanly when the shard's image slice is exhausted.
set -uo pipefail
cd "$(dirname "$0")/.."
source .env

INDEX="${INDEX:-configs/data/embodied_images.index}"
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-4}"
# Measured: the shared endpoint knees at ~30-40 concurrent requests in total. Past
# that, p50 latency climbs from <1s to 12-15s with no throughput gain (pure queueing).
# With NSHARD shards running, keep each shard's workers small.
WORKERS="${WORKERS:-10}"
CHUNK="${CHUNK:-4000}"          # images per cycle (bounded so audits happen often)
TARGET="${TARGET:-100000000}"   # effectively unlimited; the index bounds the work
OUT="${OUT:-data/flywheel_long/shard${SHARD}}"

TOTAL=$(wc -l < "$INDEX")
BASE=$(( TOTAL * SHARD / NSHARD ))
END=$(( TOTAL * (SHARD + 1) / NSHARD ))
PER=$(( END - BASE ))
mkdir -p "$OUT" logs

echo "=== longrun shard ${SHARD}/${NSHARD}: images [${BASE}, ${END}) of ${TOTAL} ==="

CYCLE=0
while :; do
  DONE_N=0
  [[ -f "$OUT/done_images.txt" ]] && DONE_N=$(wc -l < "$OUT/done_images.txt")
  if (( DONE_N >= PER )); then
    echo "=== shard ${SHARD} complete: ${DONE_N}/${PER} images ==="; break
  fi
  CYCLE=$(( CYCLE + 1 ))
  echo "--- cycle ${CYCLE} (done ${DONE_N}/${PER}) $(date -u +%H:%M:%S) ---"

  python trains/flywheel/generate_multiform.py \
      --index "$INDEX" --out "$OUT" \
      --offset "$BASE" --limit "$PER" \
      --workers "$WORKERS" --target "$TARGET" --resume \
      --batch-limit "$CHUNK" 2>&1 | tail -25
  rc=$?

  # periodic quality gate: audit the newest records of every form
  if (( CYCLE % 2 == 0 )); then
    python eval/audit_flywheel.py --data "$OUT" \
        --out "eval/audit_shard${SHARD}.json" 2>&1 | tail -11 || true
  fi

  if (( rc != 0 )); then
    echo "generator exited rc=${rc}; restarting in 30s"; sleep 30
  else
    # An empty filtered batch means the slice is exhausted. A non-empty batch
    # with no durable progress means every request failed transiently; retry it
    # instead of silently abandoning the rest of the shard.
    NEW_N=$(wc -l < "$OUT/done_images.txt" 2>/dev/null || echo 0)
    if (( NEW_N <= DONE_N )); then
      PROCESSED=$(python - "$OUT/stats.json" <<'PY'
import json
import sys

try:
    print(int(json.load(open(sys.argv[1], encoding="utf-8"))["images_processed"]))
except Exception:
    print(-1)
PY
)
      if (( PROCESSED == 0 )); then
        echo "=== no remaining images; shard ${SHARD} finished ==="; break
      fi
      echo "no durable progress across ${PROCESSED} attempted images; retrying in 30s"
      sleep 30
    fi
  fi
done
echo "=== longrun shard ${SHARD} exit $(date -u) ==="
