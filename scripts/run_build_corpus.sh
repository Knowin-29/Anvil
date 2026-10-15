#!/usr/bin/env bash
# Build the GSA training corpus. Visual Genome is explicitly down-weighted so no
# single source dominates; the flywheel's action-layer data is merged separately.
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then source .env; fi
CORPUS_OUT="${CORPUS_OUT:-trains/corpus_v2}"
[[ ! -e "$CORPUS_OUT" ]] || { echo "Output exists; choose a new CORPUS_OUT." >&2; exit 2; }
mkdir -p "$CORPUS_OUT"
exec python trains/build_corpus_5m.py --target "${TARGET:-5300000}" \
  --cap-frac "${CAP:-0.55}" --general-cap "${GEN:-0.05}" \
  --source-weight "visual_genome=${VGW:-0.45}" \
  --source-weight "visual_genome_cleaned=${VGW2:-0.6}" \
  --out-dir "$CORPUS_OUT" --shard-size 500000 --check-images 200
