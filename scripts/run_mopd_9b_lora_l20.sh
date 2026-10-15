#!/usr/bin/env bash
# True routed MOPD: student on-policy rollouts, one shared 9B teacher backbone,
# and frozen G/S/A specialist LoRA adapters selected from declared metadata.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing"; exit 1; }
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$W/logs" WANDB_PROJECT="${WANDB_PROJECT:-Anvil}"
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/output" "$W/artifacts/audits"

MODEL="${MODEL:-${MODEL_9B:-}}"
[[ -n "$MODEL" ]] || { echo "FATAL: set MODEL or MODEL_9B to the local base model"; exit 2; }
TRAIN="${TRAIN:-$W/trains/gsa_bench_v4_composite_pilot_clean/train.jsonl}"
DEFAULT_OUT="$W/output/mopd_9b_lora"
[[ "${ANVIL_MTD:-0}" == "1" ]] && DEFAULT_OUT="${ANVIL_ROOT:-$(dirname "$W")/Anvil_ICLR27}/engineering/data/anvil_mtd_9b"
OUT="${OUT:-$DEFAULT_OUT}"
# The legacy budgets describe earlier linked-task pilots, not the native corpus.
# Require a deliberate native rollout budget before even querying CUDA. A rollout
# budget need not equal reference length; reference encoding must remain intact.
NATIVE_DATA_ARGS=()
if [[ "${ANVIL_DATA_CONTRACT:-}" == "anvil-source-native-train-v1" ]]; then
  for budget in MAX_LENGTH MAX_COMPLETION_LENGTH; do
    [[ "${!budget:-}" =~ ^[1-9][0-9]*$ ]] || {
      echo "FATAL: native MTD requires explicit positive-integer $budget" >&2; exit 2;
    }
  done
  NATIVE_DATA_ARGS=(--strict true --truncation_strategy delete)
fi
if [[ "${ANVIL_MTD:-0}" == "1" ]]; then
  export ANVIL_ROUTING_MODE="${ANVIL_ROUTING_MODE:-semantic}"
  case "$ANVIL_ROUTING_MODE" in
    semantic|structural|declared) export ANVIL_STRUCTURAL_ALPHA="${ANVIL_STRUCTURAL_ALPHA:-2}" ;;
    flat) export ANVIL_STRUCTURAL_ALPHA="${ANVIL_STRUCTURAL_ALPHA:-0}" ;;
    *) echo "FATAL: unsupported ANVIL_ROUTING_MODE=$ANVIL_ROUTING_MODE" >&2; exit 2 ;;
  esac
fi
# Zero leaves the step-bounded training run uninterrupted. A positive value is
# an explicit wall-clock limit, never the default for background experiments.
HOURS="${HOURS:-0}"
TIME_LIMIT=()
if [[ ! "$HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "FATAL: HOURS must be a nonnegative number (0 disables the wall-clock limit)" >&2
  exit 2
fi
if [[ ! "$HOURS" =~ ^0+([.]0+)?$ ]]; then
  TIME_LIMIT=(timeout --foreground "${HOURS}h")
fi

NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs"; exit 1; }
if [[ "${ANVIL_DATA_CONTRACT:-}" == "anvil-source-native-train-v1" ]]; then
  [[ -d "$TRAIN" && -s "$TRAIN/manifest.json" && "${ANVIL_MTD:-0}" == "1" && "${ANVIL_EXPERIMENT_SCOPE:-}" != GS_MECHANISM ]] || {
    echo "FATAL: native MTD requires a complete export directory, ANVIL_MTD=1 and full G/S/A scope" >&2; exit 2;
  }
else
  [[ -s "$TRAIN" ]] || { echo "FATAL: missing dataset $TRAIN"; exit 1; }
fi
[[ -f "$MODEL/config.json" ]] || { echo "FATAL: missing model $MODEL"; exit 1; }

latest_adapter() {
  local family="$1" root latest
  root="$W/output/teacher_lora_$family"
  if [[ -f "$root/adapter_config.json" ]]; then echo "$root"; return; fi
  latest=$(find "$root" -type f -name adapter_config.json -printf '%h\n' 2>/dev/null \
    | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 \
    | awk 'NF==2' | tail -1 | cut -d' ' -f2)
  echo "$latest"
}

export GSA_TEACHER_G_ADAPTER="${GSA_TEACHER_G_ADAPTER:-$(latest_adapter G)}"
export GSA_TEACHER_S_ADAPTER="${GSA_TEACHER_S_ADAPTER:-$(latest_adapter S)}"
export GSA_TEACHER_A_ADAPTER="${GSA_TEACHER_A_ADAPTER:-$(latest_adapter A)}"
ANVIL_REQUIRED_FAMILIES=(G S A)
[[ "${ANVIL_EXPERIMENT_SCOPE:-}" == GS_MECHANISM ]] && ANVIL_REQUIRED_FAMILIES=(G S)
for family in "${ANVIL_REQUIRED_FAMILIES[@]}"; do
  var="GSA_TEACHER_${family}_ADAPTER"; path="${!var}"
  [[ -f "$path/adapter_config.json" ]] || {
    echo "FATAL: missing $family specialist adapter. Run:" >&2
    echo "  DIM=$family TUNER=lora bash $W/scripts/run_teacher_dim_full_dlc.sh" >&2
    exit 1
  }
done

PREFLIGHT="${PREFLIGHT_REPORT:-$OUT/dataset_preflight.json}"
VALIDATE_ARGS=("$TRAIN" --out "$PREFLIGHT")
if [[ "${ANVIL_DATA_CONTRACT:-}" == "anvil-gen-v3" ]]; then
  [[ "${ANVIL_MTD:-0}" != "1" ]] || {
    echo "FATAL: v3 uses family-routed MOPD; frozen v2 first-fault routing is not a v3 protocol" >&2; exit 2;
  }
  VALIDATE_ARGS+=(--alignment)
fi
if [[ "${ANVIL_DATA_CONTRACT:-}" == "anvil-source-native-train-v1" ]]; then
  VALIDATE_ARGS+=(--source-native)
elif [[ "${ANVIL_MTD:-0}" == "1" ]]; then
  VALIDATE_ARGS+=(--anvil)
fi
if [[ "${ALLOW_HUMAN_AUDIT:-0}" == "1" ]]; then
  VALIDATE_ARGS+=(--allow-human-audit)
fi
python "$W/scripts/validate_grpo_dataset.py" "${VALIDATE_ARGS[@]}" || {
  echo "FATAL: MOPD dataset is not contract-locked; see $PREFLIGHT"; exit 1; }
TRAIN_DATASETS=("$TRAIN")
if [[ "${ANVIL_DATA_CONTRACT:-}" == "anvil-source-native-train-v1" ]]; then
  mapfile -t TRAIN_DATASETS < <(python - "$PREFLIGHT" <<'PY'
import json, sys
for path in json.load(open(sys.argv[1]))['dataset_files']:
    if '\n' in path: raise ValueError('Newline in dataset path')
    print(path)
PY
  )
  [[ "${#TRAIN_DATASETS[@]}" == 11 ]] || { echo "FATAL: native MTD requires all eleven spatial streams" >&2; exit 2; }
fi
python - "$PREFLIGHT" "$MODEL" "$GSA_TEACHER_G_ADAPTER" "$GSA_TEACHER_S_ADAPTER" "$GSA_TEACHER_A_ADAPTER" <<'PY' || exit 1
import json, pathlib, sys, os
report = json.load(open(sys.argv[1], encoding="utf-8"))
forms = report.get("by_form") or {}
family = {
    "G": sum(forms.get(k, 0) for k in ("bbox", "detection", "mask", "point")),
    "S": sum(forms.get(k, 0) for k in ("depth", "spatial", "depth_order", "direction", "metric_comparison", "relation_graph")),
    "A": sum(forms.get(k, 0) for k in ("affordance", "dof", "action_cot")),
}
scoped = os.environ.get('ANVIL_EXPERIMENT_SCOPE')=='GS_MECHANISM'
if scoped and (family['S']!=report['records'] or family['G'] or family['A']):
    raise SystemExit('FATAL: GS mechanism study requires linked G-to-S rows only')
if not scoped and min(family.values()) == 0:
    raise SystemExit(f"FATAL: all G/S/A families are required, got {family}")
expected = pathlib.Path(sys.argv[2]).resolve()
for path in map(pathlib.Path, sys.argv[3:5] if scoped else sys.argv[3:]):
    cfg = json.load(open(path / "adapter_config.json", encoding="utf-8"))
    base = pathlib.Path(cfg.get("base_model_name_or_path", "")).resolve()
    if base != expected:
        raise SystemExit(f"FATAL: adapter {path} base {base} != teacher backbone {expected}")
print(f"MOPD preflight PASS: family counts={family}")
PY

STUDENT_ADAPTER="${STUDENT_ADAPTER:-$(bash "$W/scripts/_find_sft_adapter.sh")}" || exit 1
export STUDENT_ADAPTER
if [[ "${ANVIL_MTD:-0}" == "1" ]]; then
  python - <<'PY' || exit 2
import os
from trains.mopd.anvil_runtime import validate_teacher_eligibility
validate_teacher_eligibility(os.environ.get("TEACHER_ELIGIBILITY_JSON"),
    {family: os.environ[f"GSA_TEACHER_{family}_ADAPTER"] for family in
     (("G", "S") if os.environ.get('ANVIL_EXPERIMENT_SCOPE')=='GS_MECHANISM' else ("G", "S", "A"))},
    os.environ.get("STUDENT_ADAPTER"))
print("Anvil path-bound teacher preflight PASS before Swift model loading")
PY
fi
# Swift optimizes equal student/teacher path strings into disable_adapter()
# self-distillation. Routed teachers need a separate backbone with three frozen
# adapters. A role-specific symlink keeps identical weight bytes without copies.
TEACHER_BACKBONE="${TEACHER_BACKBONE:-$OUT/teacher_backbone}"
mkdir -p "$OUT"
if [[ -e "$TEACHER_BACKBONE" || -L "$TEACHER_BACKBONE" ]]; then
  [[ "$(realpath "$TEACHER_BACKBONE")" == "$(realpath "$MODEL")" ]] || {
    echo "FATAL: teacher backbone alias points to a different checkpoint" >&2; exit 2;
  }
else
  ln -s -- "$(realpath "$MODEL")" "$TEACHER_BACKBONE" || exit 2
fi
BS="${BS:-1}"
GA="${GA:-$(( 32 / (NG * BS) ))}"; [[ "$GA" -lt 1 ]] && GA=1
GLOBAL_BATCH=$(( NG * BS ))
python - "$PREFLIGHT" "$GLOBAL_BATCH" <<'PY' || exit 1
import json, sys, os
d = json.load(open(sys.argv[1], encoding="utf-8")); b = int(sys.argv[2]); f = d["by_form"]
counts = {"G": sum(f.get(k, 0) for k in ("bbox", "detection", "mask", "point")),
          "S": sum(f.get(k, 0) for k in ("depth", "spatial", "depth_order", "direction", "metric_comparison", "relation_graph")),
          "A": sum(f.get(k, 0) for k in ("affordance", "dof", "action_cot"))}
required = [counts['S']] if os.environ.get('ANVIL_EXPERIMENT_SCOPE')=='GS_MECHANISM' else list(counts.values())
if min(required) < b: raise SystemExit(f"FATAL: family smaller than global batch {b}: {counts}")
PY

RESUME=()
LAST=$(find "$OUT" -type f -name trainer_state.json -printf '%h\n' 2>/dev/null \
  | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 \
  | awk 'NF==2' | tail -1 | cut -d' ' -f2)
[[ -n "$LAST" ]] && RESUME=(--resume_from_checkpoint "$LAST")

STEPS="${STEPS:-600}"
KD_ARGS=(--gkd_logits_topk "${GKD_TOPK:-64}")
METHOD_ARGS=(--enable_thinking "${ENABLE_THINKING:-false}" --eval_strategy no)
METHOD="vanilla routed MOPD (balanced G/S/A batches)"
if [[ "${ANVIL_MTD:-0}" == "1" ]]; then
  KD_ARGS=()
  METHOD_ARGS=(--enable_thinking false --eval_strategy no)
  METHOD="Anvil ${ANVIL_ROUTING_MODE} control plus reference SFT (alpha=${ANVIL_STRUCTURAL_ALPHA})"
  [[ -s "${TEACHER_ELIGIBILITY_JSON:-}" ]] || {
    echo "FATAL: ANVIL_MTD=1 requires TEACHER_ELIGIBILITY_JSON" >&2; exit 2;
  }
fi
LOG="$W/logs/mopd_9b_lora_$(date -u +%Y%m%d_%H%M%S)_$$.log"
echo "=================================================="
echo " method : $METHOD"
echo " GPUs   : $NG  BS=$BS GA=$GA eff_batch=$((NG * BS * GA))"
echo " student: $STUDENT_ADAPTER"
echo " teacher: shared $MODEL + ${ANVIL_REQUIRED_FAMILIES[*]} LoRA specialists"
echo " data   : $TRAIN"
echo " steps  : $STEPS  resume=${LAST:-none}"
echo " tokens : max_length=${MAX_LENGTH:-3072} max_completion_length=${MAX_COMPLETION_LENGTH:-512}"
echo " time cap: ${HOURS} hours (0 = disabled)"
echo " log    : $LOG"
echo "=================================================="

NPROC_PER_NODE="$NG" "${TIME_LIMIT[@]}" swift rlhf --rlhf_type gkd \
  --model "$MODEL" --adapters "$STUDENT_ADAPTER" \
  --teacher_model "$TEACHER_BACKBONE" \
  --external_plugins "$W/scripts/gsa_mopd_plugin.py" \
  --dataset "${TRAIN_DATASETS[@]}" --split_dataset_ratio 0 "${NATIVE_DATA_ARGS[@]}" \
  --lmbda 1.0 --seq_kd false --beta "${JSD_BETA:-0.5}" --temperature "${TEMPERATURE:-0.9}" \
  "${KD_ARGS[@]}" "${METHOD_ARGS[@]}" --remove_unused_columns false \
  --tuner_type lora --torch_dtype bfloat16 \
  --deepspeed zero2 --teacher_deepspeed "${TEACHER_DEEPSPEED:-zero3}" \
  --max_steps "$STEPS" --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --seed "${SEED:-17}" --data_seed "${DATA_SEED:-17}" \
  --learning_rate "${LR:-5e-6}" --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_length "${MAX_LENGTH:-3072}" --max_completion_length "${MAX_COMPLETION_LENGTH:-512}" \
  --gradient_checkpointing true --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --dataloader_drop_last true --train_dataloader_shuffle false \
  --logging_steps 5 --save_steps "${SAVE_STEPS:-100}" --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --output_dir "$OUT" --dataloader_num_workers "${WORKERS:-8}" --dataset_num_proc "${DATA_PROC:-16}" \
  --report_to "${REPORT_TO:-wandb}" "${RESUME[@]}" \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
if [[ "$RC" -eq 0 ]] && ! grep -qE "'loss':" "$LOG"; then
  echo "FATAL: command exited without a training loss"; RC=1
fi
exit "$RC"
