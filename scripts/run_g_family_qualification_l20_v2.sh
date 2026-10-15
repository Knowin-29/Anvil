#!/usr/bin/env bash
# Train the schema-v3 G specialist on v2 train/dev only; qualification is never passed
# to ms-swift. This does not authorize joint, S/A, or MOPD training.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || { echo "FATAL: required environment file is missing: $W/.env" >&2; exit 2; }
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
  echo "FATAL: swift does not resolve inside the required environment: ${SWIFT_BIN:-MISSING}" >&2
  exit 2
}

SOURCE_CORPUS="${G_V1_CORPUS:-$W/trains/gsa_bench_v3_g_family_scoped_v1}"
CORPUS="${G_V2_CORPUS:-$W/trains/gsa_bench_v3_g_family_scoped_v2}"
MANIFEST="$CORPUS/manifest.json"
TRAIN="$CORPUS/train_G_only.jsonl"
DEV="$CORPUS/dev_G_only.jsonl"
QUALIFICATION="$CORPUS/qualification_G_only.jsonl"
SCOPED_REPORT="${G_SCOPED_REPORT:-$W/artifacts/audits/pilot_semantic_audit_v3_g_family_scoped_v1.json}"
FULL_REPORT="${V3_FULL_REPORT:-$W/artifacts/audits/pilot_semantic_audit_v3_adjudicated.json}"
MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
OUT="${OUT:-$W/output/g_family_v3_scoped_v2_qualification_lora}"

"$PYTHON_BIN" "$W/scripts/verify_g_family_scoped_v2.py" \
  --manifest "$MANIFEST" --source "$SOURCE_CORPUS" \
  --scoped-report "$SCOPED_REPORT" --full-report "$FULL_REPORT"
[[ -s "$TRAIN" && -s "$DEV" && -s "$QUALIFICATION" ]] || {
  echo "FATAL: v2 train/dev/qualification files are missing or empty" >&2; exit 2;
}
[[ -f "$MODEL/config.json" ]] || { echo "FATAL: model config is missing: $MODEL/config.json" >&2; exit 2; }

TRAIN_ROWS=$(wc -l < "$TRAIN")
DEV_ROWS=$(wc -l < "$DEV")
QUALIFICATION_ROWS=$(wc -l < "$QUALIFICATION")
[[ "$TRAIN_ROWS" == 84 && "$DEV_ROWS" == 32 && "$QUALIFICATION_ROWS" == 32 ]] || {
  echo "FATAL: expected v2 rows train/dev/qualification=84/32/32" >&2; exit 2;
}

BS="${BS:-1}"
GA="${GA:-1}"
EPOCHS="${EPOCHS:-5}"
MIN_STEPS="${MIN_STEPS:-32}"
HOURS="${HOURS:-6}"
for value in "$BS" "$GA" "$EPOCHS" "$MIN_STEPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: batch/epoch values must be positive integers" >&2; exit 2; }
done
GLOBAL_BATCH=$(( 8 * BS * GA ))
COMPUTED_STEPS=$(( (TRAIN_ROWS * EPOCHS + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
STEPS="${STEPS:-$COMPUTED_STEPS}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: STEPS must be a positive integer" >&2; exit 2; }
[[ "$STEPS" -lt "$MIN_STEPS" ]] && STEPS="$MIN_STEPS"
EVAL_STEPS="${EVAL_STEPS:-$STEPS}"
SAVE_STEPS="${SAVE_STEPS:-$STEPS}"
for value in "$EVAL_STEPS" "$SAVE_STEPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: cadence must be positive integers" >&2; exit 2; }
done
(( SAVE_STEPS % EVAL_STEPS == 0 )) || { echo "FATAL: SAVE_STEPS must divide cleanly by EVAL_STEPS" >&2; exit 2; }

echo "=================================================="
echo " protocol      : semantic-audit-v3-g-family-scoped-v2-independent-qualification"
echo " scope         : G specialist qualification pilot only"
echo " train/dev     : 84/32 rows on 24/8 images"
echo " qualification : 32 rows on 8 images (never exposed to trainer)"
echo " plan          : 8xL20 BS=$BS GA=$GA epochs=$EPOCHS steps=$STEPS"
echo " global        : BLOCKED; joint/MOPD are not authorized"
echo " output        : $OUT"
echo "=================================================="

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "G_V2_TRAIN_PREFLIGHT_READY: no GPU process was started"
  exit 0
fi

mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF' | sort -u
)
if [[ "${#GPU_PIDS[@]}" -gt 0 && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "FATAL: visible GPUs already have compute processes: ${GPU_PIDS[*]}" >&2; exit 3;
fi
"$PYTHON_BIN" - <<'PY'
import torch
count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"FATAL: v2 G qualification training requires exactly 8 GPUs, found {count}")
for index in range(count):
    name = torch.cuda.get_device_name(index)
    gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    if "L20" not in name.upper() or gib < 44:
        raise SystemExit(f"FATAL: GPU {index} is not a >=44 GiB L20: {name} ({gib:.1f} GiB)")
print("8xL20 hardware gate: READY")
PY

mkdir -p "$OUT"
RECEIPT="$OUT/g_family_qualification_inputs_v2.json"
"$PYTHON_BIN" - "$MANIFEST" "$SCOPED_REPORT" "$FULL_REPORT" "$MODEL/config.json" \
  "$TRAIN" "$DEV" "$QUALIFICATION" "$RECEIPT" "$STEPS" "$BS" "$GA" "$EPOCHS" <<'PY'
import hashlib, json, os, pathlib, sys
manifest, scoped, full, model, train, dev, qualification, receipt = map(pathlib.Path, sys.argv[1:9])
steps, bs, ga, epochs = map(int, sys.argv[9:13])
def bind(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
payload = {
    "protocol": "semantic-audit-v3-g-family-scoped-v2-independent-qualification",
    "source_audit_protocol": "semantic-audit-v3-g-family-scoped-v1",
    "authorization": "G_specialist_qualification_only",
    "global_audit_verdict": "BLOCKED",
    "manifest": bind(manifest), "scoped_report": bind(scoped), "full_report": bind(full),
    "model_config": bind(model), "train": bind(train), "dev": bind(dev),
    "qualification": bind(qualification),
    "train_rows": 84, "dev_rows": 32, "qualification_rows": 32,
    "train_images": 24, "dev_images": 8, "qualification_images": 8,
    "qualification_exposed_to_trainer": False,
    "trainer_dataset_paths": [str(train.resolve()), str(dev.resolve())],
    "hardware": "8xL20", "per_device_batch_size": bs,
    "gradient_accumulation_steps": ga, "effective_batch_size": 8 * bs * ga,
    "epochs_for_step_calculation": epochs, "max_steps": steps,
    "teacher_authorization": "NONE", "joint_training_authorized": False,
    "mopd_training_authorized": False,
}
if receipt.exists():
    if json.loads(receipt.read_text()) != payload:
        raise SystemExit("FATAL: output directory is bound to different v2 inputs")
else:
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, receipt)
print(json.dumps(payload, indent=2))
PY

# Only train and dev are passed to the generic trainer. QUALIFICATION has no path into
# this command, so it cannot affect gradients, eval loss, or best-checkpoint selection.
TRAIN="$TRAIN" VAL="$DEV" MODEL="$MODEL" OUT="$OUT" \
  LR="${LR:-1e-5}" STEPS="$STEPS" BS="$BS" GA="$GA" \
  EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" HOURS="$HOURS" \
  bash "$W/scripts/run_sft_9b_lora_l20_dlc.sh"
