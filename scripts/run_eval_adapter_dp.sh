#!/usr/bin/env bash
# Merge one LoRA checkpoint and evaluate the standalone model with the formal DP harness.
# Usage:
#   bash scripts/run_eval_adapter_dp.sh <tag> <adapter-or-run-dir> \
#     "<benchmark ...>" [subset] [batch]
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
# Keep adapter merge/evaluation usable on nodes with CUDA runtime libraries but
# without a locally installed CUDA toolkit (DeepSpeed is imported indirectly).
export DS_IGNORE_CUDA_DETECTION=1 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=PIX
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/output"

TAG="${1:?set result tag}"
SOURCE="${2:?set adapter checkpoint or run directory}"
BENCHES="${3:?set space-separated benchmark names}"
SUBSET="${4:-0}"
BATCH="${5:-16}"
REQUIRE_COMPLETED_STEPS="${REQUIRE_COMPLETED_STEPS:-}"

if [[ -f "$SOURCE/adapter_config.json" ]]; then
  ADAPTER="$SOURCE"
elif [[ -n "$REQUIRE_COMPLETED_STEPS" ]]; then
  [[ "$REQUIRE_COMPLETED_STEPS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: REQUIRE_COMPLETED_STEPS must be a positive integer" >&2
    exit 2
  }
  ADAPTER=$(python - "$SOURCE" "$REQUIRE_COMPLETED_STEPS" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
completed = []
for state_path in source.glob("checkpoint-*/trainer_state.json"):
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if state.get("global_step") == expected and state.get("max_steps") == expected:
        completed.append((state_path, state))
if len(completed) != 1:
    raise SystemExit(
        f"FATAL: expected exactly one completed {expected}-step trainer state "
        f"under {source}, found {len(completed)}")
state_path, state = completed[0]
best = pathlib.Path(state.get("best_model_checkpoint") or state_path.parent).resolve()
try:
    best.relative_to(source)
except ValueError:
    raise SystemExit(f"FATAL: best checkpoint escapes run directory: {best}")
if not (best / "adapter_config.json").is_file():
    raise SystemExit(f"FATAL: completed run's best adapter is missing: {best}")
print(best)
PY
  )
else
  ADAPTER=$(find "$SOURCE" -type f -name adapter_config.json -printf '%h\n' 2>/dev/null \
    | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' \
    | sort -n -k1 | awk 'NF==2' | tail -1 | cut -d' ' -f2)
fi
[[ -n "${ADAPTER:-}" && -f "$ADAPTER/adapter_config.json" ]] || {
  echo "FATAL: no LoRA adapter under $SOURCE" >&2
  exit 2
}
echo "[adapter-eval] selected adapter: $ADAPTER"
if [[ "${ADAPTER_RESOLVE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

MERGED="${MERGED_DIR:-$W/output/merged_$TAG}"
if [[ ! -f "$MERGED/config.json" ]]; then
  mkdir -p "$(dirname "$MERGED")"
  echo "[adapter-eval] merging $ADAPTER -> $MERGED"
  swift export --adapters "$ADAPTER" --merge_lora true --output_dir "$MERGED"
fi
[[ -f "$MERGED/config.json" ]] || {
  echo "FATAL: merge did not create $MERGED/config.json" >&2
  exit 1
}
# A failed export can leave config.json plus an incomplete shard. Never treat it
# as a reusable model; safetensors validates each declared shard's complete buffer.
python - "$MERGED" <<'PY'
import json
import pathlib
import sys
from safetensors import safe_open

root = pathlib.Path(sys.argv[1]).resolve()
index = root / 'model.safetensors.index.json'
if index.is_file():
    mapping = json.loads(index.read_text())['weight_map']
    if not mapping:
        raise SystemExit('FATAL: merged model has an empty weight map')
    shards = set(mapping.values())
else:
    mapping = {}
    shards = {'model.safetensors'}
for name in sorted(shards):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SystemExit(f'FATAL: merged checkpoint is incomplete: {name}')
    with safe_open(path, framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        expected = {key for key, shard in mapping.items() if shard == name}
        if not keys or (mapping and keys != expected):
            raise SystemExit(f'FATAL: merged checkpoint index differs from shard: {name}')
print(f'MERGED_SHARDS_VALIDATED count={len(shards)}', flush=True)
PY

exec bash "$W/scripts/run_eval_dp.sh" "$TAG" "$MERGED" "$BENCHES" "$SUBSET" "$BATCH"
