#!/usr/bin/env bash
# Run only the schema-v3 G-family specialist qualification on exactly 8x L20.
# This entrypoint does not authorize a joint, S/A, or MOPD run.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || {
  echo "FATAL: required environment file is missing or unreadable: $W/.env" >&2
  exit 2
}
# shellcheck disable=SC1091
source "$W/.env"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_ENV="/path/to/environment/env/gsa_train"
PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
[[ "$PYTHON_PREFIX" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: .env did not activate the required training environment" >&2
  echo "  expected sys.prefix: $EXPECTED_ENV" >&2
  echo "  actual sys.prefix  : $PYTHON_PREFIX" >&2
  exit 2
}
SWIFT_BIN=$(command -v swift || true)
[[ "$SWIFT_BIN" == "$EXPECTED_ENV/bin/swift" ]] || {
  echo "FATAL: swift does not resolve inside the required training environment: ${SWIFT_BIN:-MISSING}" >&2
  exit 2
}
PACK="${V3_AUDIT_PACK:-$W/data/semantic_audit_v3}"
ADJUDICATION="${V3_ADJUDICATION:-$PACK/adjudication.jsonl}"
FULL_REPORT="${V3_FULL_REPORT:-$W/artifacts/audits/pilot_semantic_audit_v3_adjudicated.json}"
SCOPED_REPORT="${G_SCOPED_REPORT:-$W/artifacts/audits/pilot_semantic_audit_v3_g_family_scoped_v1.json}"
CORPUS="${G_SCOPED_CORPUS:-$W/trains/gsa_bench_v3_g_family_scoped_v1}"
MANIFEST="$CORPUS/manifest.json"
TRAIN="$CORPUS/train_G_only.jsonl"
VAL="$CORPUS/val_G_only.jsonl"
MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
OUT="${OUT:-$W/output/g_family_v3_scoped_qualification_lora}"

"$PYTHON_BIN" "$W/scripts/verify_g_family_scoped_qualification.py" \
  --manifest "$MANIFEST" \
  --scoped-report "$SCOPED_REPORT" \
  --full-report "$FULL_REPORT" \
  --pack "$PACK" \
  --adjudication "$ADJUDICATION"

[[ -s "$TRAIN" && -s "$VAL" ]] || {
  echo "FATAL: G-only train/val files are missing or empty" >&2
  exit 2
}
[[ -f "$MODEL/config.json" ]] || {
  echo "FATAL: model config is missing: $MODEL/config.json" >&2
  exit 2
}

BS="${BS:-1}"
GA="${GA:-1}"
EPOCHS="${EPOCHS:-5}"
MIN_STEPS="${MIN_STEPS:-32}"
HOURS="${HOURS:-6}"
for value in "$BS" "$GA" "$EPOCHS" "$MIN_STEPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: BS, GA, EPOCHS and MIN_STEPS must be positive integers" >&2
    exit 2
  }
done

# Qualification step count is corpus-driven.  With the frozen 116-row train
# split and defaults this is ceil(116*5/(8*1*1)) = 73 optimizer steps.
ROWS=$(wc -l < "$TRAIN")
VAL_ROWS=$(wc -l < "$VAL")
GLOBAL_BATCH=$(( 8 * BS * GA ))
COMPUTED_STEPS=$(( (ROWS * EPOCHS + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
STEPS="${STEPS:-$COMPUTED_STEPS}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: STEPS must be a positive integer" >&2
  exit 2
}
[[ "$STEPS" -lt "$MIN_STEPS" ]] && STEPS="$MIN_STEPS"
EVAL_STEPS="${EVAL_STEPS:-$STEPS}"
SAVE_STEPS="${SAVE_STEPS:-$STEPS}"
for value in "$EVAL_STEPS" "$SAVE_STEPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: EVAL_STEPS and SAVE_STEPS must be positive integers" >&2
    exit 2
  }
done
(( SAVE_STEPS % EVAL_STEPS == 0 )) || {
  echo "FATAL: SAVE_STEPS must be a multiple of EVAL_STEPS" >&2
  exit 2
}

echo "=================================================="
echo " protocol : semantic-audit-v3-g-family-scoped-v1"
echo " scope    : G specialist qualification only"
echo " global   : BLOCKED (depth); joint/MOPD are not authorized"
echo " corpus   : $CORPUS"
echo " rows     : train=$ROWS val=$VAL_ROWS"
echo " plan     : 8xL20 BS=$BS GA=$GA global_batch=$GLOBAL_BATCH epochs=$EPOCHS steps=$STEPS"
echo " cadence  : eval=$EVAL_STEPS save=$SAVE_STEPS cap=${HOURS}h"
echo " output   : $OUT"
echo "=================================================="

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "G_ONLY_PREFLIGHT_PASS: no GPU process was started"
  exit 0
fi

mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF' | sort -u
)
if [[ "${#GPU_PIDS[@]}" -gt 0 && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "FATAL: visible GPUs already have compute processes: ${GPU_PIDS[*]}" >&2
  exit 3
fi

"$PYTHON_BIN" - <<'PY'
import sys
import torch

count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"FATAL: G-only qualification requires exactly 8 visible GPUs, found {count}")
for index in range(count):
    name = torch.cuda.get_device_name(index)
    gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    if "L20" not in name.upper():
        raise SystemExit(f"FATAL: GPU {index} is not an L20: {name}")
    if gib < 44:
        raise SystemExit(f"FATAL: GPU {index} exposes only {gib:.1f} GiB")
print("8xL20 hardware gate: PASS")
PY

mkdir -p "$OUT"
RECEIPT="$OUT/g_family_qualification_inputs.json"
"$PYTHON_BIN" - \
  "$MANIFEST" "$SCOPED_REPORT" "$MODEL/config.json" "$RECEIPT" \
  "$ROWS" "$VAL_ROWS" "$BS" "$GA" "$EPOCHS" "$STEPS" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

manifest, report, model_config, receipt = map(pathlib.Path, sys.argv[1:5])
rows, val_rows, bs, ga, epochs, steps = map(int, sys.argv[5:])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "protocol": "semantic-audit-v3-g-family-scoped-v1",
    "authorization": "G_specialist_qualification_only",
    "global_audit_verdict": "BLOCKED",
    "manifest": {"path": str(manifest.resolve()), "sha256": sha(manifest)},
    "scoped_report": {"path": str(report.resolve()), "sha256": sha(report)},
    "model_config": {"path": str(model_config.resolve()), "sha256": sha(model_config)},
    "hardware": "8xL20",
    "train_rows": rows,
    "val_rows": val_rows,
    "per_device_batch_size": bs,
    "gradient_accumulation_steps": ga,
    "effective_batch_size": 8 * bs * ga,
    "epochs_for_step_calculation": epochs,
    "max_steps": steps,
}
if receipt.exists():
    current = json.loads(receipt.read_text())
    if current != payload:
        raise SystemExit("FATAL: output directory is bound to different qualification inputs")
else:
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, receipt)
print(json.dumps(payload, indent=2))
PY

# The generic SFT engine receives only the two explicitly named G-only files.
# A checkpoint remains a qualification candidate until matched G evaluation.
TRAIN="$TRAIN" VAL="$VAL" MODEL="$MODEL" OUT="$OUT" \
  LR="${LR:-1e-5}" STEPS="$STEPS" BS="$BS" GA="$GA" \
  EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" HOURS="$HOURS" \
  bash "$W/scripts/run_sft_9b_lora_l20_dlc.sh"
