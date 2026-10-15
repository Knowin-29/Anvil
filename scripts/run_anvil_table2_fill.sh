#!/usr/bin/env bash
# Bounded missing-cell evaluation; explicit architecture, frozen prompts/scorers.
set -euo pipefail
# Freeze the launcher for each running job: source edits must not corrupt a shell
# that is paused inside a long inference command.
case "${BASH_SOURCE[0]}" in
 /tmp/anvil-launch.*/launcher.sh) ;;
 *) SNAPSHOT=$(mktemp -d /tmp/anvil-launch.XXXXXXXX)
    cp -- "${BASH_SOURCE[0]}" "$SNAPSHOT/launcher.sh"
    exec bash "$SNAPSHOT/launcher.sh" "$@";;
esac
cd /path/to/Anvil
ANVIL_PACKAGE=/path/to/Anvil
ANVIL_PY=/path/to/environment/env/gsa_train/bin/python
MODE=${1:?mode: rb25|rynn|mimo|mimo_g_completion|cosmos|w2p_base|w2p_p5|p4_seed23|vacancy_train|vacancy_repair}
GPU_UUID=${2:?one currently free GPU UUID}
SCORE_ONLY=${3:-0}
[[ "$SCORE_ONLY" == 0 || "$SCORE_ONLY" == 1 ]] || exit 2
if [[ "$SCORE_ONLY" == 0 ]]; then
[[ "$GPU_UUID" == GPU-* && "$GPU_UUID" != *,* ]] || exit 2
GPU_WAIT_SECONDS=${ANVIL_GPU_WAIT_SECONDS:-3600}
[[ "$GPU_WAIT_SECONDS" =~ ^[0-9]+$ ]] || exit 2
GPU_DEADLINE=$((SECONDS + GPU_WAIT_SECONDS))
while true; do
 GPU_USED=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)
 [[ "$GPU_USED" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || exit 2
 (( GPU_USED >= 64 )) || break
 (( SECONDS < GPU_DEADLINE )) || { echo "GPU queue expired (${GPU_USED} MiB); no foreign allocation changed"; exit 3; }
 sleep 15
done
fi
if [[ "$MODE" == vacancy_train || "$MODE" == vacancy_train_s44 || "$MODE" =~ ^control_(replay|vacancy)_s(43|44)_train$ ]]; then
 REPAIR=$ANVIL_PACKAGE/engineering/data/vacancy_repair_20260914
 INPUT_REPAIR=$REPAIR
 RUN_SEED=43
 EVAL_MODE=vacancy_repair
 if [[ "$MODE" == vacancy_train_s44 ]]; then
  REPAIR=$REPAIR/seed44
  RUN_SEED=44
  EVAL_MODE=vacancy_seed44
 fi
 if [[ "$MODE" =~ ^control_(replay|vacancy)_s(43|44)_train$ ]]; then
  CONTROL_KIND=${BASH_REMATCH[1]}
  RUN_SEED=${BASH_REMATCH[2]}
  EVAL_MODE=control_${CONTROL_KIND}_s${RUN_SEED}
  REPAIR=$ANVIL_PACKAGE/engineering/data/source_controls_20260915/$EVAL_MODE
  # Concurrent torchrun jobs must not share the default rendezvous port.
  if [[ "$CONTROL_KIND" == replay ]]; then
   export MASTER_PORT=$((29600 + RUN_SEED))
  else
   export MASTER_PORT=$((29700 + RUN_SEED))
  fi
  export MASTER_ADDR=127.0.0.1
 fi
 TRAIN_VARIANT=${REPAIR_TRAIN_VARIANT:-training}
 [[ "$TRAIN_VARIANT" == training || "$TRAIN_VARIANT" == training_uuidfix_v1 || "$TRAIN_VARIANT" == training_portfix_v1 ]] || exit 2
 [[ -s "$INPUT_REPAIR/preparation.json" && ! -e "$REPAIR/$TRAIN_VARIANT" && ! -e "$REPAIR/merged" ]] || exit 4
 export PATH=/path/to/environment/env/gsa_train/bin:$PATH
 export IMAGE_MAX_TOKEN_NUM=1536
 export CFG=anvil_${EVAL_MODE}_seed${RUN_SEED}_v1 MODEL_TYPE=qwen3_5
 export MODEL=$ANVIL_PACKAGE/engineering/data/p5_retention_clean_run_seed17_v2/runtime/merged
 export TRAIN=$INPUT_REPAIR/split/train.jsonl VAL=$INPUT_REPAIR/split/dev.jsonl OUT=$REPAIR/$TRAIN_VARIANT
 if [[ -n "${CONTROL_KIND:-}" ]]; then
  export TRAIN=$ANVIL_PACKAGE/engineering/data/source_controls_20260915/$CONTROL_KIND/train.jsonl
  "$ANVIL_PY" - "$ANVIL_PACKAGE/engineering/data/source_controls_20260915/plan.json" "$CONTROL_KIND" "$TRAIN" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text()); arm=p['arms'][sys.argv[2]]
t=Path(sys.argv[3]); assert len(t.read_text().splitlines()) == arm['training_rows'] == 7264
assert hashlib.sha256(t.read_bytes()).hexdigest() == arm['train']['sha256']
assert hashlib.sha256(Path(p['shared_dev']['path']).read_bytes()).hexdigest() == p['shared_dev']['sha256']
PY
 fi
 export LR=2e-5 RANK=16 ALPHA=32 EPOCHS=1 BS=1 MAXLEN=2048 SEED=$RUN_SEED HOURS=6 GPUS=$GPU_UUID
 "$ANVIL_PY" - "$INPUT_REPAIR" "$ANVIL_PACKAGE" <<'PY'
import hashlib,json,sys
from pathlib import Path
r,a=map(Path,sys.argv[1:])
p=json.loads((r/'preparation.json').read_text())
assert p['p5_train_sha256']==json.loads((a/'engineering/data/p5_retention_clean_split_v1/manifest.json').read_text())['train']['sha256']
m=json.loads((r/'split/manifest.json').read_text())
assert m['train']['rows']==7264 and m['dev']['rows']==360
for split in ('train','dev'):
    assert hashlib.sha256((r/f'split/{split}.jsonl').read_bytes()).hexdigest()==m[split]['sha256']
PY
 bash scripts/run_sweep_train_l20_3gpu.sh
 # Only the predeclared terminal checkpoint; never select using benchmark scores.
 mapfile -t TERMINAL < <(find "$OUT" -maxdepth 2 -type d -name checkpoint-303)
 [[ ${#TERMINAL[@]} == 1 ]] || { echo 'Missing unique terminal checkpoint-303'; exit 5; }
 if [[ -n "${CONTROL_KIND:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1
  exec "$ANVIL_PY" eval/anvil_adapter_resume.py --mode "$EVAL_MODE" --adapter "${TERMINAL[0]}"
 fi
 CUDA_VISIBLE_DEVICES='' DS_IGNORE_CUDA_DETECTION=1 timeout 30m swift export \
  --model "$MODEL" --model_type qwen3_5 --adapters "${TERMINAL[0]}" \
  --merge_lora true --output_dir "$REPAIR/merged" > "$REPAIR/export.log" 2>&1
 exec bash scripts/run_anvil_table2_fill.sh "$EVAL_MODE" "$GPU_UUID"
fi
MODEL_TYPE=qwen3_vl
MAX_TOKENS=0
INFERENCE_HOURS=2
case "$MODE" in
 rb25) MODEL=/path/to/models/RoboBrain2.5-8B-NV; BENCHES=(where2place blink_relative_depth erqa vabench_point gsa_bench_test);;
 rynn) MODEL=/path/to/datasets/models/RynnBrain-8B; BENCHES=(where2place blink_relative_depth vabench_point gsa_bench_test cv_bench embspatial);;
 mimo) MODEL=/path/to/datasets/models/MiMo-Embodied-7B; MODEL_TYPE=qwen2_5_vl; BENCHES=(blink_relative_depth gsa_bench_test);;
 mimo_g_completion) MODEL=/path/to/datasets/models/MiMo-Embodied-7B; MODEL_TYPE=qwen2_5_vl; BENCHES=(gsa_bench_test); INFERENCE_HOURS=3;;
 cosmos) MODEL=/path/to/datasets/models/Cosmos-Reason-1-7B; MODEL_TYPE=qwen2_5_vl; BENCHES=(gsa_bench_test);;
 cosmos_remaining) MODEL=/path/to/datasets/models/Cosmos-Reason-1-7B; MODEL_TYPE=qwen2_5_vl; BENCHES=(cv_bench embspatial refspatial_bench where2place blink_relative_depth erqa vabench_point); INFERENCE_HOURS=3;;
 w2p_base) MODEL=/path/to/environment/hf_models/Qwen/Qwen3.5-9B; MODEL_TYPE=qwen3_5; BENCHES=(where2place); MAX_TOKENS=512;;
 w2p_p5) MODEL=$ANVIL_PACKAGE/engineering/data/p5_retention_clean_run_seed17_v2/runtime/merged; MODEL_TYPE=qwen3_5; BENCHES=(where2place); MAX_TOKENS=512;;
 p4_seed23) MODEL=/tmp/anvil-runtime-20260911.O8s5pn/merged_seed23_v7; MODEL_TYPE=qwen3_5; BENCHES=(blink_relative_depth robospatial_home robospatial_home_context erqa vabench_point);;
 vacancy_repair) MODEL=$ANVIL_PACKAGE/engineering/data/vacancy_repair_20260914/merged; MODEL_TYPE=qwen3_5; BENCHES=(where2place robospatial_home_context vabench_point blink_relative_depth erqa embspatial);;
 vacancy_remaining) MODEL=$ANVIL_PACKAGE/engineering/data/vacancy_repair_20260914/merged; MODEL_TYPE=qwen3_5; BENCHES=(cv_bench refspatial_bench gsa_bench_test robospatial_home);;
 vacancy_seed44) MODEL=$ANVIL_PACKAGE/engineering/data/vacancy_repair_20260914/seed44/merged; MODEL_TYPE=qwen3_5; BENCHES=(where2place robospatial_home_context vabench_point blink_relative_depth erqa embspatial);;
 control_replay_s43|control_replay_s44|control_vacancy_s43|control_vacancy_s44) MODEL=$ANVIL_PACKAGE/engineering/data/source_controls_20260915/$MODE/merged; MODEL_TYPE=qwen3_5; BENCHES=(where2place robospatial_home_context vabench_point blink_relative_depth erqa embspatial);;
 *) exit 2;;
esac
TEMPLATE_TYPE=$MODEL_TYPE
[[ "$MODE" != mimo && "$MODE" != mimo_g_completion ]] || TEMPLATE_TYPE=mimo_vl
TAG=anvil_table2_${MODE}_explicit_20260914
OUT=$ANVIL_PACKAGE/engineering/data/table2_fill_20260914/$MODE
if [[ "$SCORE_ONLY" == 0 ]]; then
[[ ! -e "$OUT/worker.json" && ! -e "$OUT/shards" ]] || { echo "Existing run preserved: $OUT"; exit 4; }
else
 [[ -f "$OUT/worker.json" ]] || { echo 'No complete worker report for CPU recovery'; exit 4; }
fi
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PWD/.cache/hf" HF_HUB_CACHE="$PWD/.cache/hf/hub"
export DS_IGNORE_CUDA_DETECTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CLI=eval/run_benchmark_template_diagnostic.py
START=$(date +%s)
RC=0
ELAPSED=0
if [[ "$SCORE_ONLY" == 0 ]]; then
set +e
timeout "${INFERENCE_HOURS}h" "$ANVIL_PY" "$CLI" --model "$MODEL" --model-type "$MODEL_TYPE" --template-type "$TEMPLATE_TYPE" \
 --benchmarks "${BENCHES[@]}" --shard-dir "$OUT/shards" --out "$OUT/worker.json" \
 --limit 0 --batch-size 4 --no-thinking --max-tokens "$MAX_TOKENS" > "$OUT/generation.log" 2>&1
RC=$?
set -e
ELAPSED=$(( $(date +%s) - START ))
echo "generation exit=$RC elapsed=$ELAPSED"
fi
# Score only complete successful cells, independently; never aggregate partial shards.
for BENCH in "${BENCHES[@]}"; do
 [[ ! -e "$OUT/$BENCH.json" ]] || continue
 if "$ANVIL_PY" - "$OUT/worker.json" "$BENCH" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
sys.exit(0 if p.exists() and json.loads(p.read_text()).get('benchmarks',{}).get(sys.argv[2],{}).get('status')=='ok' else 1)
PY
 then
  BENCH_ELAPSED=$("$ANVIL_PY" - "$OUT/worker.json" "$BENCH" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['benchmarks'][sys.argv[2]]['elapsed_s'])
PY
)
  "$ANVIL_PY" "$CLI" --model "$MODEL" --model-type "$MODEL_TYPE" --template-type "$TEMPLATE_TYPE" \
   --benchmark "$BENCH" --score-shards "$OUT/shards/$BENCH/shard_0.jsonl" \
   --out "$OUT/$BENCH.json" --dump-predictions "$OUT/preds_$BENCH.jsonl" \
   --limit 0 --batch-size 4 --no-thinking --max-tokens "$MAX_TOKENS" --elapsed-s "$BENCH_ELAPSED" || RC=1
 fi
done
"$ANVIL_PY" - "$OUT" "$RC" <<'PY'
import json,sys,datetime
from pathlib import Path
p=Path(sys.argv[1]);w=json.loads((p/'worker.json').read_text()) if (p/'worker.json').exists() else {}
cells={k:{'generation':v.get('status'),'result_exists':(p/f'{k}.json').exists()} for k,v in w.get('benchmarks',{}).items()}
(p/'completion_status.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rc':int(sys.argv[2]),'cells':cells},indent=2)+'\n')
PY
exit "$RC"
