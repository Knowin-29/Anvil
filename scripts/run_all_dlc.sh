#!/usr/bin/env bash
# run_all_dlc.sh — the whole paper pipeline in one submission.
#
# Platform entrypoint (one line):
#   bash /path/to/Anvil/scripts/run_all_dlc.sh
#
# Runs, in order, skipping anything already finished so a re-submit resumes:
#   E0  base zero-shot eval        (the row every other row is compared against)
#   E1  9B LoRA SFT on GSA-Bench   -> main table
#   E1e eval E1
#   E3  GRPO with verifier reward  -> the methodological claim
#   E3e eval E3
#   E2  9B full-parameter SFT      -> anti-forgetting contrast
#   E2e eval E2
#   T   backfill the LaTeX tables from whatever finished
#
# Every stage is time-capped and writes its own log. If the platform kills the task,
# re-submitting the same line continues from the last completed stage.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing"; exit 1; }

if [[ "${ALLOW_LEGACY_PIPELINE:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
FATAL: this legacy all-in-one pipeline is disabled.

It would retrain the historical E1 recipe and mix development evaluation with
unfinished GRPO/full-SFT stages. Use the claim-driven queue instead:
  experiments/execution_queue.md
  scripts/run_pilot_qualification_l20.sh

Set ALLOW_LEGACY_PIPELINE=1 only to reproduce the historical diagnostic. Results from
that opt-in run are not eligible for the current main method claim.
EOF
  exit 2
fi

# MUST source .env before any python call: the interpreter with torch installed lives on
# the read-only mount, and the container's default python has no torch at all. Omitting
# this made GPU detection report 0 and aborted the whole pipeline.
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

# Caches must not be written to read-only /path/to/workspace/foundation.
export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} logs output eval_results
MASTER="$W/logs/run_all_$(date -u +%Y%m%d_%H%M%S).log"

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$MASTER"; }

NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
if [[ "$NG" -lt 1 ]]; then
  echo "GPU detection failed. Diagnostics:"
  echo "  python  : $(command -v python || echo MISSING)"
  python -c 'import torch;print("  torch   :",torch.__version__,"cuda",torch.cuda.is_available())' 2>&1 | head -2
  echo "  nvidia-smi:"; nvidia-smi -L 2>&1 | head -3
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
say "=================================================="
say " GSA-Forge full pipeline"
say " GPUs detected: $NG"
free -g 2>/dev/null | awk 'NR==2{print "[  ] host RAM: "$2"G"}' | tee -a "$MASTER"
df -h "$W" 2>/dev/null | awk 'NR==2{print "[  ] disk free: "$4}' | tee -a "$MASTER"
say "=================================================="
[[ "$NG" -ge 1 ]] || { say "FATAL: no GPU visible"; exit 1; }

stage () {   # stage <name> <sentinel-path> <command...>
  local name="$1" sentinel="$2"; shift 2
  if [[ -e "$sentinel" ]]; then say "SKIP  $name (found $(basename "$sentinel"))"; return 0; fi
  say "START $name"
  if "$@" >> "$MASTER" 2>&1; then
    say "DONE  $name"
    return 0
  else
    local rc=$?
    say "FAIL  $name rc=$rc — continuing with the next stage"
    return 1        # keep going, but the caller must NOT mark this stage done
  fi
}

# A stage that failed must not be sentinel-stamped: doing so makes the next run SKIP it
# and, worse, makes the eval stage treat a nonexistent checkpoint as available. The E2
# OOM was masked exactly this way — sft_9b_full/.done existed with zero checkpoints.
mark_done () { [[ -d "$1" ]] && ls -d "$1"/checkpoint-* "$1"/v*/checkpoint-* >/dev/null 2>&1 \
               && touch "$1/.done"; }

# ---- E0: base zero-shot ----------------------------------------------------
stage "E0 base eval" "$W/eval_results/base/summary.json" \
  env CKPT=base TAG=base HOURS=4 bash "$W/scripts/run_eval_all_dlc.sh"

# ---- E1: LoRA SFT (main table) --------------------------------------------
# Pick the GPU-specific SFT script. L20 has 48 GiB and no NVLink, so it needs a smaller
# per-device batch and ZeRO-2; A800 has 80 GiB and NVLink and can take a larger batch.
GPU_NAME=$(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo unknown)
if echo "$GPU_NAME" | grep -qiE 'L20|L40|A10|4090'; then
  SFT_SCRIPT="$W/scripts/run_sft_9b_lora_l20_dlc.sh"; SFT_OUT="$W/output/sft_9b_lora_l20"
else
  SFT_SCRIPT="$W/scripts/run_sft_9b_lora_dlc.sh";     SFT_OUT="$W/output/sft_9b_lora"
fi
say " GPU model: $GPU_NAME"
say " SFT script: $(basename "$SFT_SCRIPT")"

stage "E1 9B LoRA SFT" "$SFT_OUT/.done" bash "$SFT_SCRIPT"
mark_done "$SFT_OUT"

stage "E1 eval" "$W/eval_results/sft_9b_lora/summary.json" \
  env CKPT="${SFT_OUT#"$W/"}" TAG=sft_9b_lora HOURS=4 bash "$W/scripts/run_eval_all_dlc.sh"

# ---- E3: verifier-as-reward RL (the method claim) -------------------------
if [[ "${RUN_GRPO:-0}" == "1" ]]; then
  stage "E3 GRPO verifier-reward" "$W/output/grpo_9b_lora/.done" \
    bash "$W/scripts/run_grpo_9b_lora_dlc.sh"
  mark_done "$W/output/grpo_9b_lora"

  stage "E3 eval" "$W/eval_results/grpo_9b_lora/summary.json" \
    env CKPT=output/grpo_9b_lora TAG=grpo_9b_lora HOURS=4 bash "$W/scripts/run_eval_all_dlc.sh"
else
  say "BLOCK E3 GRPO (set RUN_GRPO=1 only after the reward gate passes)"
fi

# ---- E2: full-parameter SFT (anti-forgetting contrast) -------------------
if [[ "${RUN_FULL_SFT:-0}" == "1" ]]; then
  stage "E2 9B full SFT" "$W/output/sft_9b_full/.done" \
    bash "$W/scripts/run_sft_9b_full_dlc.sh"
  mark_done "$W/output/sft_9b_full"

  stage "E2 eval" "$W/eval_results/sft_9b_full/summary.json" \
    env CKPT=output/sft_9b_full TAG=sft_9b_full HOURS=4 bash "$W/scripts/run_eval_all_dlc.sh"
else
  say "BLOCK E2 full SFT (set RUN_FULL_SFT=1 after the L20 memory gate passes)"
fi

# ---- tables --------------------------------------------------------------
say "START backfill tables"
python "$W/eval/backfill_tables.py" --results "$W/eval_results" 2>&1 | tee -a "$MASTER"

say "=================================================="
say " pipeline finished"
for d in "$W"/eval_results/*/summary.json; do
  [[ -s "$d" ]] && say " $(basename "$(dirname "$d")"): $(python -c "
import json,sys
r=json.load(open('$d')); v=[x for x in r.values() if isinstance(x,(int,float))]
print(f'{len(v)} metrics, mean {sum(v)/len(v):.1f}' if v else 'no parsed metrics')" 2>/dev/null)"
done
say " master log: $MASTER"
say "=================================================="
