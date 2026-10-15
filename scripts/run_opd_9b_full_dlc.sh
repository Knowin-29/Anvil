#!/usr/bin/env bash
# run_opd_9b_full_dlc.sh
#   9B  = Qwen3.5-9B student
#   opd = on-policy distillation (GKD). On-policy, not offline KD, because our
#         failure modes are student-specific: a student emitting a rectangular mask
#         must be corrected ON that rectangle. lmbda>0 makes the student sample.
#   dlc = submitted as a Volcengine custom task
#
# ---------------------------------------------------------------------------
# MULTI-TEACHER IS NOT NATIVELY SUPPORTED BY ms-swift. Verified, not assumed:
#   * RLHFArguments.teacher_model is Optional[str] -- a scalar, not a list
#     (swift/arguments/rlhf_args.py:67).
#   * GKDTrainer holds exactly one self.teacher_model and one forward
#     (swift/rlhf_trainers/gkd_trainer.py:113-132, 417-435).
#   * `teacher_adapters: List[str]` EXISTS and is documented, but NO code reads
#     it. The only adapter selection is
#       adapters = args.adapters if key == 'ref' else args.reward_adapters
#     (swift/pipelines/train/rlhf.py:94), which never consults teacher_adapters.
#     Confirmed at runtime: passing --teacher_adapters is accepted silently and
#     the teacher stays un-adapted. So a LoRA teacher CANNOT be attached this way
#     -- it must be merged first.
#
# Therefore ROUTE=seq (default) implements multi-teacher as SEQUENTIAL
# distillation: one GKD stage per dimension teacher, each stage starting from the
# previous stage's student. Order is G -> S -> A, which follows the dependency
# chain in storyline section 4 (S is ill-posed until its arguments are grounded;
# A until the geometry is established), and each stage sees only its own
# dimension slice, so the per-dimension teacher is always matched to the data it
# is expert on.
#
# ROUTE=single distills from one teacher over the pooled corpus (smoke / ablation).
# ---------------------------------------------------------------------------
#
# Platform entrypoint (one line):
#   bash /path/to/Anvil/scripts/run_opd_9b_full_dlc.sh
#
# GPU count, grad accumulation and step count are derived at runtime; the
# effective batch is pinned to 32 so results stay comparable across allocations.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing (check the vePFS mount)"; exit 1; }
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

# caches must not touch read-only /path/to/workspace/foundation
export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$W/logs" WANDB_PROJECT=GSA_Forge
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/output"

HOURS="${HOURS:-12}"
MODEL=/path/to/environment/hf_models/Qwen/Qwen3.5-9B
ROUTE="${ROUTE:-seq}"

# GPU count from torch, NOT nvidia-smi: nvidia-smi -L ignores CUDA_VISIBLE_DEVICES
# and over-reports on a sliced node, which makes torchrun spawn too many ranks.
NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs"; nvidia-smi -L | head -3; exit 1; }

BS="${BS:-4}"
# GA must always be assigned: this script runs under `set -u` and an unset $GA is
# what made E3 exit in 2 seconds with zero steps.
GA="${GA:-$(( 32 / (NG * BS) ))}"; [[ "$GA" -lt 1 ]] && GA=1

# resolve_teacher DIM -> prints a standalone teacher dir for that dimension.
# A merged full-parameter teacher is required; a bare LoRA adapter dir is NOT
# usable because --teacher_adapters is dead (see header).
resolve_teacher() {
  local dim="$1" t
  t=$(ls -d "$W"/output/teacher_"$dim"/checkpoint-* "$W"/output/teacher_"$dim"/v*/checkpoint-* 2>/dev/null \
      | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 | awk 'NF==2' | tail -1 | cut -d' ' -f2)
  if [[ -n "$t" && ! -f "$t/config.json" ]]; then
    echo "FATAL: $t holds no config.json -- looks like a LoRA adapter dir." >&2
    echo "       ms-swift ignores --teacher_adapters, so merge it first:" >&2
    echo "       swift export --adapters $t --merge_lora true" >&2
    return 1
  fi
  # Print (possibly empty) and always succeed: callers distinguish "absent" from
  # "malformed" themselves. Returning 1 on empty would make `T=$(...) || exit 1`
  # abort with no diagnostic.
  echo "$t"
  return 0
}

# One GKD stage. $1=tag $2=teacher $3=dataset $4=init-adapter("" for none) $5=steps
run_stage() {
  local tag="$1" teacher="$2" data="$3" init="$4" steps="$5"
  local out="$W/output/opd_9b/$tag"
  local log="$W/logs/opd_9b_${tag}_$(date -u +%Y%m%d_%H%M%S).log"

  local resume=() start=()
  local last
  last=$(ls -d "$out"/checkpoint-* "$out"/v*/checkpoint-* 2>/dev/null \
         | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 | awk 'NF==2' | tail -1 | cut -d' ' -f2)
  [[ -n "$last" ]] && resume=(--resume_from_checkpoint "$last")
  [[ -n "$init" ]] && start=(--adapters "$init")

  echo "=================================================="
  echo " stage : $tag"
  echo " teach : $teacher"
  echo " data  : $(basename "$data")  ($(wc -l < "$data") rows)   steps=$steps"
  echo " init  : ${init:-base}"
  echo " resume: ${last:-none}"
  echo " log   : $log"
  echo "=================================================="

  NPROC_PER_NODE="$NG" timeout "${HOURS}h" swift rlhf --rlhf_type gkd \
    --teacher_model "$teacher" \
    --lmbda "${LMBDA:-0.5}" --beta "${BETA:-0.5}" \
    --model "$MODEL" \
    --dataset "$data" \
    --split_dataset_ratio 0 \
    --tuner_type lora --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 "${start[@]}" \
    --torch_dtype bfloat16 --deepspeed zero3 --teacher_deepspeed zero3 \
    --max_steps "$steps" \
    --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
    --learning_rate 5e-6 --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --max_length 3072 --max_completion_length 512 \
    --gradient_checkpointing true \
    --logging_steps 20 --save_steps 250 --save_total_limit 3 \
    --output_dir "$out" --dataloader_num_workers 16 --dataset_num_proc 32 \
    --report_to wandb "${resume[@]}" \
    2>&1 | tee -a "$log"

  local rc=${PIPESTATUS[0]}
  # rc=0 alone does not mean the stage trained: a 2-second clean exit also
  # returns 0. Require a loss line.
  if ! grep -qE "'loss':" "$log"; then
    echo " FAIL[$tag]: never reached the training loop"
    grep -iE 'error|traceback|out of memory|assert' "$log" | tail -8
    return 1
  fi
  echo " stage $tag rc=$rc, last loss:"; grep -oE "\{'loss':[^}]*\}" "$log" | tail -1
  # Hand the newest checkpoint to the next stage.
  ls -d "$out"/checkpoint-* "$out"/v*/checkpoint-* 2>/dev/null \
    | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 | awk 'NF==2' | tail -1 | cut -d' ' -f2 \
    > "$W/output/opd_9b/.last_$tag"
  return "$rc"
}

echo "route: $ROUTE   GPUs=$NG   per_device=$BS grad_accum=$GA  eff_batch=$(( NG * BS * GA ))"
free -g 2>/dev/null | awk 'NR==2{print " host RAM: "$2"G"}'
mkdir -p "$W/output/opd_9b"

if [[ "$ROUTE" == "single" ]]; then
  TEACHER="${TEACHER:-}"
  if [[ -z "$TEACHER" ]]; then
    for d in G S A; do TEACHER=$(resolve_teacher "$d") && [[ -n "$TEACHER" ]] && break; done
  fi
  [[ -z "$TEACHER" && -f "$W/output/merged_e1/config.json" ]] && TEACHER="$W/output/merged_e1"
  [[ -n "$TEACHER" ]] || { echo "FATAL: no teacher. Run S1 first, or merge E1 as a stand-in."; exit 1; }
  TRAIN="$W/trains/gsa_bench/train_shuf.jsonl"
  [[ -s "$TRAIN" ]] || TRAIN="$W/trains/gsa_bench/train.jsonl"
  run_stage single "$TEACHER" "$TRAIN" "" "${STEPS:-$(( NG * 150 ))}"
  exit $?
fi

# ---- ROUTE=seq: G -> S -> A, each stage on its own slice ----
MISSING=()
for d in G S A; do
  T=$(resolve_teacher "$d") || exit 1
  [[ -n "$T" ]] || MISSING+=("teacher_$d")
done
if (( ${#MISSING[@]} )); then
  echo "FATAL: missing ${MISSING[*]}. Run S1 first:"
  for m in "${MISSING[@]}"; do echo "  DIM=${m#teacher_} bash $W/scripts/run_teacher_dim_full_dlc.sh"; done
  exit 1
fi

INIT=""
RC=0
for d in G S A; do
  T=$(resolve_teacher "$d")
  D="$W/trains/gsa_bench/train_$d.jsonl"
  [[ -s "$D" ]] || { echo "FATAL: missing slice $D"; exit 1; }
  # Steps proportional to the slice so S (43,873 rows) is not over-trained
  # relative to G (164,090).
  S=$(( $(wc -l < "$D") / 32 )); [[ "$S" -lt 50 ]] && S=50
  run_stage "$d" "$T" "$D" "$INIT" "$S" || { RC=1; break; }
  INIT=$(cat "$W/output/opd_9b/.last_$d" 2>/dev/null || echo "")
  [[ -n "$INIT" ]] || { echo "FATAL: stage $d produced no checkpoint"; RC=1; break; }
done

echo "=================================================="
if [[ "$RC" == 0 ]]; then
  echo " done. final student: $INIT"
else
  echo " FAILED. re-run the SAME command to resume from the last checkpoint."
fi
exit "$RC"
