#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/path/to/workspace/foundation/contributor/Project/Rex_Embodied_gsa_wbj-volc_jun12_v1"
INPUT_FILE="data/input/grounding_verifier_lvis_coco_rem_1k/lvis_coco_rem_grounding_1k.jsonl"
CONFIG_FILE="configs/grounding_verifier_rules.yaml"
OUTPUT_DIR="data/input/grounding_verifier_lvis_coco_rem_1k"
TIMESTAMP="${1:-$(date +%Y%m%d_%H%M)}"

OUT_TMP="${OUTPUT_DIR}/specialist_voting_paired_sweep_${TIMESTAMP}.json.tmp"
OUT_FILE="${OUTPUT_DIR}/specialist_voting_paired_sweep_${TIMESTAMP}.json"
LOG_FILE="${OUTPUT_DIR}/specialist_voting_paired_sweep_${TIMESTAMP}.log"
PID_FILE="${OUTPUT_DIR}/specialist_voting_paired_sweep_${TIMESTAMP}.pid"
RUN_FILE="${OUTPUT_DIR}/specialist_voting_paired_sweep_${TIMESTAMP}.run.sh"

if [[ -z "${VOLC_API_KEY:-}" ]]; then
  echo "ERROR: VOLC_API_KEY is not set in this shell. Export it first, then rerun this script." >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"

cat > "$RUN_FILE" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] specialist voting paired sweep started"
echo "input: $INPUT_FILE"
echo "config: $CONFIG_FILE"
echo "output_tmp: $OUT_TMP"
echo "output: $OUT_FILE"
PYTHONUNBUFFERED=1 python scripts/evaluate_specialist_voting.py \\
  --input "$INPUT_FILE" \\
  --config "$CONFIG_FILE" \\
  --sweep-prompts paired \\
  --output "$OUT_TMP"
python -m json.tool --indent 2 "$OUT_TMP" > "$OUT_FILE"
rm -f "$OUT_TMP"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] specialist voting paired sweep finished"
RUNNER
chmod +x "$RUN_FILE"

setsid nohup bash "$RUN_FILE" > "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started detached specialist voting sweep."
echo "PID: $PID"
echo "PID file: $ROOT_DIR/$PID_FILE"
echo "Log file: $ROOT_DIR/$LOG_FILE"
echo "Report file: $ROOT_DIR/$OUT_FILE"
