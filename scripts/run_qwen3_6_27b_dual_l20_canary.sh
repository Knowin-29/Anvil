#!/usr/bin/env bash
# Isolated Qwen3.6-27B diagnostic canary on one process and exactly two L20s.
#
# This is deliberately not a formal gsa-eval-v7 run. Qwen3.6-27B supports direct
# responses through enable_thinking=False (not through the /nothink soft instruction),
# which the shared runner exposes as --no-thinking. Parameterized diagnostic subsets
# stay under artifacts/diagnostic_runs and must not enter experiments/result_registry.yaml.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
# shellcheck disable=SC1091
[[ -r "$W/.env" ]] || {
  echo "FATAL: environment activation file is missing or unreadable: $W/.env" >&2
  exit 2
}
source "$W/.env"

PYTHON_BIN="${PYTHON_BIN:-python}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
RUN_BENCHMARK="${RUN_BENCHMARK:-$W/eval/run_benchmark.py}"
MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.6-27B}"
BENCHMARKS="${BENCHMARKS:-cv_bench embspatial refspatial_bench where2place gsa_bench_test}"
LIMIT="${LIMIT:-8}"
BATCH="${BATCH:-1}"
SEED="${SEED:-0}"
CELL_TIMEOUT="${CELL_TIMEOUT:-2h}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
TAG="${TAG:-qwen3_6_27b_dual_l20_diagnostic_limit${LIMIT}_batch${BATCH}_seed${SEED}_${STAMP}}"
OUT="${OUT:-$W/artifacts/diagnostic_runs/qwen3_6_27b_dual_l20_canary/$TAG}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-/tmp/gsa_qwen36_canary_cache_${TAG}}"
PROTOCOL_ID="qwen3.6-27b-dual-l20-diagnostic-canary-v3"
MIN_L20_MIB=$((44 * 1024))
MIN_RUNTIME_CACHE_BYTES=$((8 * 1024 * 1024 * 1024))

fatal() {
  echo "FATAL: $*" >&2
  exit 2
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || fatal "LIMIT must be a positive integer, got '$LIMIT'"
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || fatal "BATCH must be a positive integer, got '$BATCH'"
[[ "$SEED" =~ ^[0-9]+$ ]] || fatal "SEED must be a non-negative integer, got '$SEED'"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || \
  fatal "PREFLIGHT_ONLY must be 0 or 1, got '$PREFLIGHT_ONLY'"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fatal "unsafe diagnostic TAG '$TAG'"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fatal "Python executable not found: $PYTHON_BIN"
command -v "$NVIDIA_SMI_BIN" >/dev/null 2>&1 || fatal "GPU inventory tool not found: $NVIDIA_SMI_BIN"
[[ -r "$RUN_BENCHMARK" ]] || fatal "benchmark entry point is missing: $RUN_BENCHMARK"

# The caller owns the physical-device slice. Requiring it explicitly prevents this
# launcher from seeing all four local GPUs and prevents accidental remapping to 0,1.
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || \
  fatal "set CUDA_VISIBLE_DEVICES to exactly two L20 selectors before invoking this launcher"
UPSTREAM_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
IFS=',' read -r -a GPU_SELECTORS <<< "$UPSTREAM_CUDA_VISIBLE_DEVICES"
[[ "${#GPU_SELECTORS[@]}" == 2 ]] || \
  fatal "exactly two visible GPU selectors are required; got '$UPSTREAM_CUDA_VISIBLE_DEVICES'"

declare -A SEEN_GPU=()
GPU_NAMES=()
GPU_MEMORY_MIB=()
for index in 0 1; do
  selector=$(trim "${GPU_SELECTORS[$index]}")
  [[ -n "$selector" ]] || fatal "empty GPU selector in CUDA_VISIBLE_DEVICES"
  [[ -z "${SEEN_GPU[$selector]:-}" ]] || fatal "duplicate GPU selector '$selector'"
  SEEN_GPU[$selector]=1
  GPU_SELECTORS[$index]="$selector"

  query=$(
    "$NVIDIA_SMI_BIN" -i "$selector" \
      --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null
  ) || fatal "cannot query GPU selector '$selector'"
  mapfile -t rows < <(printf '%s\n' "$query" | awk 'NF')
  [[ "${#rows[@]}" == 1 ]] || \
    fatal "GPU selector '$selector' resolved to ${#rows[@]} devices, expected one"
  name=$(trim "${rows[0]%%,*}")
  memory=$(trim "${rows[0]#*,}")
  memory="${memory//[!0-9]/}"
  [[ "${name^^}" == *L20* ]] || \
    fatal "GPU selector '$selector' is not an L20: '$name'"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_L20_MIB" ]] || \
    fatal "GPU selector '$selector' has ${memory:-unknown} MiB; at least $MIN_L20_MIB MiB is required"
  GPU_NAMES+=("$name")
  GPU_MEMORY_MIB+=("$memory")
done

# Keep this launcher scoped to the five benchmarks whose prompts and scorers are
# already handled by the unified entry point.
# shellcheck disable=SC2206
BENCH_ARRAY=($BENCHMARKS)
[[ "${#BENCH_ARRAY[@]}" -ge 1 ]] || fatal "BENCHMARKS is empty"
declare -A KNOWN_BENCH=(
  [cv_bench]=1 [embspatial]=1 [refspatial_bench]=1
  [where2place]=1 [gsa_bench_test]=1
)
for benchmark in "${BENCH_ARRAY[@]}"; do
  [[ -n "${KNOWN_BENCH[$benchmark]:-}" ]] || fatal "unsupported canary benchmark '$benchmark'"
done

W_ABS=$(realpath -m "$W")
OUT_ABS=$(realpath -m "$OUT")
case "$OUT_ABS" in
  "$W_ABS/eval_results"|"$W_ABS/eval_results/"*|\
  "$W_ABS/runs"|"$W_ABS/runs/"*|\
  "$W_ABS/artifacts/results"|"$W_ABS/artifacts/results/"*)
    fatal "diagnostic output may not be written under a formal/result-indexed tree: $OUT_ABS"
    ;;
esac
OUT="$OUT_ABS"
mkdir -p "$OUT"
RUNTIME_CACHE_ROOT=$(realpath -m "$RUNTIME_CACHE_ROOT")
mkdir -p "$RUNTIME_CACHE_ROOT"
cache_available=$(df -P -B1 "$RUNTIME_CACHE_ROOT" | awk 'NR == 2 {print $4}')
[[ "$cache_available" =~ ^[0-9]+$ && "$cache_available" -ge "$MIN_RUNTIME_CACHE_BYTES" ]] || \
  fatal "runtime cache filesystem has ${cache_available:-unknown} bytes free; at least $MIN_RUNTIME_CACHE_BYTES are required"

MODEL_INVENTORY="$OUT/model_inventory.json"
"$PYTHON_BIN" - "$MODEL" "$MODEL_INVENTORY" "$PROTOCOL_ID" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

model = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
protocol = sys.argv[3]
if not model.is_dir():
    raise SystemExit(f"FATAL: model directory is missing: {model}")

required = ["model.safetensors.index.json", "config.json", "LICENSE", "README.md"]
missing = [name for name in required if not (model / name).is_file()]
empty = [name for name in required if (model / name).is_file() and not (model / name).stat().st_size]
if missing or empty:
    raise SystemExit(f"FATAL: incomplete model metadata; missing={missing}, empty={empty}")

expected = [f"model-{index:05d}-of-00015.safetensors" for index in range(1, 16)]
actual = sorted(path.name for path in model.glob("model-*-of-*.safetensors"))
if actual != expected:
    raise SystemExit(
        f"FATAL: expected exactly 15 canonical weight shards; "
        f"missing={sorted(set(expected) - set(actual))}, extra={sorted(set(actual) - set(expected))}"
    )
empty_shards = [name for name in expected if not (model / name).stat().st_size]
if empty_shards:
    raise SystemExit(f"FATAL: empty model weight shards: {empty_shards}")

try:
    index_record = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"FATAL: unreadable model index/config: {exc}") from exc
weight_map = index_record.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit("FATAL: model index has no non-empty weight_map")
referenced = sorted(set(weight_map.values()))
if referenced != expected:
    raise SystemExit(
        f"FATAL: model index does not bind all 15 shards; "
        f"missing={sorted(set(expected) - set(referenced))}, "
        f"extra={sorted(set(referenced) - set(expected))}"
    )
if config.get("model_type") != "qwen3_5":
    raise SystemExit(
        f"FATAL: unexpected model_type {config.get('model_type')!r}; expected 'qwen3_5'"
    )
if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
    raise SystemExit(
        "FATAL: unexpected model architecture; expected "
        "['Qwen3_5ForConditionalGeneration']"
    )

record = {
    "schema_version": 1,
    "protocol": protocol,
    "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "model_path": str(model),
    "model_type": config["model_type"],
    "architectures": config["architectures"],
    "weight_shard_count": len(expected),
    "index_weight_count": len(weight_map),
    "required_metadata": {
        name: {"path": str(model / name), "size_bytes": (model / name).stat().st_size}
        for name in required
    },
    "weight_shards": [
        {"name": name, "size_bytes": (model / name).stat().st_size}
        for name in expected
    ],
}
out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"model inventory: READY ({len(expected)} shards, index/config/license/readme checked)")
PY

PROTOCOL_MANIFEST="$OUT/diagnostic_protocol.json"
"$PYTHON_BIN" - "$PROTOCOL_MANIFEST" "$PROTOCOL_ID" "$TAG" "$MODEL" \
  "$LIMIT" "$BATCH" "$SEED" "$UPSTREAM_CUDA_VISIBLE_DEVICES" \
  "${GPU_NAMES[0]}" "${GPU_MEMORY_MIB[0]}" \
  "${GPU_NAMES[1]}" "${GPU_MEMORY_MIB[1]}" "$RUNTIME_CACHE_ROOT" \
  "${BENCH_ARRAY[@]}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

(
    path, protocol, tag, model, limit, batch, seed, visible,
    gpu0_name, gpu0_memory, gpu1_name, gpu1_memory, runtime_cache_root, *benchmarks
) = sys.argv[1:]
exclusion_reasons = [
    f"seeded limit-{limit} diagnostic subset rather than a full split",
    "single-process automatic placement across two L20s is a diagnostic topology",
]
if int(batch) != 16:
    exclusion_reasons.append(
        f"generation batch size {batch} differs from frozen formal batch size 16"
    )
record = {
    "schema_version": 1,
    "protocol": protocol,
    "tier": "diagnostic",
    "formal_result_registry_eligible": False,
    "exclusion_reasons": exclusion_reasons,
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "tag": tag,
    "model": str(Path(model).resolve()),
    "benchmarks": benchmarks,
    "subset_limit": int(limit),
    "subset_seed": int(seed),
    "generation_batch_size": int(batch),
    "thinking_control": "enable_thinking_false_with_empty_think_prefix",
    "process_count": 1,
    "device_map": "auto",
    "cuda_visible_devices": visible,
    "runtime_cache_root": runtime_cache_root,
    "hardware": [
        {"visible_index": 0, "name": gpu0_name, "memory_mib": int(gpu0_memory)},
        {"visible_index": 1, "name": gpu1_name, "memory_mib": int(gpu1_memory)},
    ],
    "result_registry": "excluded; diagnostic artifacts only",
}
Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "=================================================="
echo " protocol       : $PROTOCOL_ID"
echo " tier           : diagnostic only; excluded from the formal result registry"
echo " model          : $MODEL"
echo " visible slice  : $UPSTREAM_CUDA_VISIBLE_DEVICES (preserved)"
echo " placement      : one process, device-map auto across exactly two L20s"
echo " generation     : direct response via enable_thinking=false"
echo " benchmarks     : ${BENCH_ARRAY[*]}"
echo " limit/batch    : $LIMIT / $BATCH"
echo " output         : $OUT"
echo " runtime cache  : $RUNTIME_CACHE_ROOT"
echo "=================================================="

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "QWEN3_6_27B_CANARY_PREFLIGHT_READY: no CUDA process was started"
  exit 0
fi

[[ ! -e "$OUT/run.started" && ! -d "$OUT/shards" ]] || \
  fatal "diagnostic run state already exists; choose a fresh TAG or OUT"
for selector in "${GPU_SELECTORS[@]}"; do
  busy=$(
    "$NVIDIA_SMI_BIN" -i "$selector" \
      --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
  )
  [[ -z "$(printf '%s\n' "$busy" | awk '$1 ~ /^[0-9]+$/ {print $1; exit}')" ]] || \
    fatal "GPU selector '$selector' already has a compute process"
done

export HF_HOME="$RUNTIME_CACHE_ROOT/hf" HF_HUB_CACHE="$RUNTIME_CACHE_ROOT/hf/hub"
export TORCH_HOME="$RUNTIME_CACHE_ROOT/torch" TRITON_CACHE_DIR="$RUNTIME_CACHE_ROOT/triton"
export XDG_CACHE_HOME="$RUNTIME_CACHE_ROOT/xdg" TMPDIR="$RUNTIME_CACHE_ROOT/tmp"
export MPLCONFIGDIR="$RUNTIME_CACHE_ROOT/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$TMPDIR" "$MPLCONFIGDIR" \
  "$OUT"/{logs,shards,results,predictions}
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.started"

start_seconds=$(date +%s)
generation_rc=0
# Preserve the caller's slice verbatim. This is one model process spanning the two
# visible devices via automatic placement; there are no per-device worker processes.
CUDA_VISIBLE_DEVICES="$UPSTREAM_CUDA_VISIBLE_DEVICES" timeout "$CELL_TIMEOUT" \
  "$PYTHON_BIN" "$RUN_BENCHMARK" \
    --model "$MODEL" --benchmarks "${BENCH_ARRAY[@]}" \
    --shard-dir "$OUT/shards" --out "$OUT/worker.json" \
    --shard 0 --num-shards 1 --limit "$LIMIT" --batch-size "$BATCH" \
    --seed "$SEED" --device-map auto --no-thinking \
    > "$OUT/logs/generation.log" 2>&1 || generation_rc=$?
elapsed_seconds=$(( $(date +%s) - start_seconds ))
if [[ "$generation_rc" -ne 0 ]]; then
  echo "FATAL: diagnostic generation failed with rc=$generation_rc" >&2
  echo "  log: $OUT/logs/generation.log" >&2
  tail -n 40 "$OUT/logs/generation.log" >&2 || true
  exit "$generation_rc"
fi

for benchmark in "${BENCH_ARRAY[@]}"; do
  shard="$OUT/shards/$benchmark/shard_0.jsonl"
  [[ -s "$shard" ]] || fatal "missing diagnostic shard for $benchmark: $shard"
done

score_rc=0
for benchmark in "${BENCH_ARRAY[@]}"; do
  if "$PYTHON_BIN" "$RUN_BENCHMARK" \
      --model "$MODEL" --benchmark "$benchmark" \
      --out "$OUT/results/$benchmark.json" \
      --limit "$LIMIT" --batch-size "$BATCH" --seed "$SEED" \
      --elapsed-s "$elapsed_seconds" --no-thinking \
      --score-shards "$OUT/shards/$benchmark/shard_0.jsonl" \
      --dump-predictions "$OUT/predictions/$benchmark.jsonl" \
      > "$OUT/logs/score_${benchmark}.log" 2>&1; then
    echo "[$benchmark] diagnostic result: $OUT/results/$benchmark.json"
  else
    score_rc=1
    echo "[$benchmark] diagnostic score withheld or failed; inspect $OUT/logs/score_${benchmark}.log" >&2
  fi
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.completed"
echo "diagnostic canary complete: $OUT"
echo "These artifacts remain excluded from experiments/result_registry.yaml."
exit "$score_rc"
