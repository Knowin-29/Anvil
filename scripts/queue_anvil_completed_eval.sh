#!/usr/bin/env bash
# Wait for a specific live SFT run, then evaluate its validation-selected adapter.
# TRAIN_PID=<torchrun pid> RUN=<version dir> TAG=<new tag> bash this_script
set -euo pipefail
W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
source "$W/.env"
TRAIN_PID="${TRAIN_PID:-}"
QUEUE_FROM_COMPLETED="${QUEUE_FROM_COMPLETED:-0}"
RUN="${RUN:?set the exact versioned training directory}"
TAG="${TAG:?set a fresh result tag}"
STEPS="${STEPS:-8146}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ && "$QUEUE_FROM_COMPLETED" =~ ^[01]$ ]]
if [[ "$QUEUE_FROM_COMPLETED" == 0 ]]; then
  [[ "$TRAIN_PID" =~ ^[1-9][0-9]*$ && -r "/proc/$TRAIN_PID/stat" ]]
else
  [[ -z "$TRAIN_PID" ]] || { echo 'Completed-run mode must not bind a training PID' >&2; exit 2; }
fi
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$ ]]
[[ -f "$RUN/args.json" ]]
RUN=$(realpath "$RUN")
FINAL="$RUN/checkpoint-$STEPS/trainer_state.json"
MERGED="${MERGED_DIR:-$W/output/merged_$TAG}"
RETENTION="$W/output/general_retention_formal_v3_base_vs_$TAG"
[[ ! -e "$MERGED" && ! -e "$W/eval_results/$TAG" && ! -e "$RETENTION" ]] || {
  echo 'FATAL: queue outputs must be fresh' >&2; exit 2;
}
python - "$RUN/args.json" "$STEPS" "$TRAIN_PID" <<'PY'
import json, pathlib, sys
args = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert args['max_steps'] == int(sys.argv[2]), 'training step budget differs'
if sys.argv[3]:
    cmd = pathlib.Path(f'/proc/{sys.argv[3]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert str(pathlib.Path(sys.argv[1]).parent.parent) in cmd, 'PID does not identify this run'
else:
    assert (pathlib.Path(sys.argv[1]).parent / f'checkpoint-{sys.argv[2]}/trainer_state.json').is_file(), 'missing completed state'
print('QUEUE_INPUTS_VERIFIED', flush=True)
PY
if [[ "${QUEUE_CHECK_ONLY:-0}" == 1 ]]; then exit 0; fi
mkdir -p "$W/output"
exec 9>"$W/output/.anvil_completed_eval.lock"
flock -n 9 || { echo 'FATAL: another Anvil completion queue holds the lock' >&2; exit 3; }
# Bind process birth time so a recycled PID cannot be mistaken for this trainer.
if [[ "$QUEUE_FROM_COMPLETED" == 0 ]]; then
  START=$(awk '{print $22}' "/proc/$TRAIN_PID/stat")
  echo "WAITING_FOR_TRAINING pid=$TRAIN_PID run=$RUN steps=$STEPS"
  while [[ -r "/proc/$TRAIN_PID/stat" ]] && [[ "$(awk '{print $22}' "/proc/$TRAIN_PID/stat")" == "$START" ]]; do
    sleep 30
  done
else
  echo "RESUMING_COMPLETED_RUN run=$RUN steps=$STEPS"
fi
[[ -f "$FINAL" ]] || { echo "TRAINING_TERMINATED_WITHOUT_FINAL_CHECKPOINT $FINAL" >&2; exit 4; }
python - "$FINAL" "$STEPS" <<'PY'
import json, pathlib, sys
import torch
from safetensors import safe_open
torch.set_num_threads(2)
state_path = pathlib.Path(sys.argv[1])
state = json.loads(state_path.read_text())
assert state['global_step'] == state['max_steps'] == int(sys.argv[2])
best = pathlib.Path(state.get('best_model_checkpoint') or state_path.parent).resolve()
best.relative_to(state_path.parent.parent.resolve())
with safe_open(best / 'adapter_model.safetensors', framework='pt', device='cpu') as handle:
    assert handle.keys(), 'empty adapter'
    for key in handle.keys():
        assert torch.isfinite(handle.get_tensor(key)).all(), f'nonfinite tensor {key}'
print(f'COMPLETED_ADAPTER_VERIFIED {best}', flush=True)
PY
echo 'WAITING_FOR_IDLE_GPUS'
while :; do
  if ! BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); then
    echo 'GPU_QUERY_FAILED_RETRY'; sleep 30; continue
  fi
  [[ -z "${BUSY//[[:space:]]/}" ]] && break
  sleep 30
done
export DS_IGNORE_CUDA_DETECTION=1 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=PIX
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 THINKING=off \
  REQUIRE_COMPLETED_STEPS="$STEPS" CELL_TIMEOUT=4h MERGED_DIR="$MERGED" \
  bash scripts/run_eval_adapter_dp.sh "$TAG" "$RUN" \
  'cv_bench embspatial refspatial_bench where2place gsa_bench_test' 0 16
echo 'SPATIAL_EVALUATION_COMPLETE'
RETENTION_RC=0
CUDA_VISIBLE_DEVICES=0,1,2,3 BASE_MODEL="$MODEL_9B" E1_MODEL="$MERGED" \
  OUT="$RETENTION" RUN_STAGE=paired RESUME=0 THINKING=off CELL_TIMEOUT=8h \
  bash scripts/run_general_retention_dp.sh || RETENTION_RC=$?
echo "RETENTION_EXIT_CODE=$RETENTION_RC"
if [[ -f "$RETENTION/results/base.json" && -f "$RETENTION/results/e1.json" ]]; then
  python eval/general_retention_mc_compat.py \
    --base-result "$RETENTION/results/base.json" --candidate-result "$RETENTION/results/e1.json" \
    --out "$RETENTION/results/mc_compat_v1.json"
fi
echo 'EVALUATION_QUEUE_FINISHED: inspect results and claim evidence before paper admission'
exit "$RETENTION_RC"
