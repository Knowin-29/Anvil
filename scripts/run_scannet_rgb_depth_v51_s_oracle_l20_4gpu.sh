#!/usr/bin/env bash
# Formal held-out ScanNet v5.1 S-oracle evaluation on exactly four L20 GPUs.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || { echo "FATAL: missing $W/.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$W/.env"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_ENV="/path/to/environment/env/gsa_train"
PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
[[ "$PYTHON_PREFIX" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: expected sys.prefix $EXPECTED_ENV, got $PYTHON_PREFIX" >&2
  exit 2
}

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs"

RUNNER="$W/eval/scannet_rgb_depth_v51_dlc.py"
SCORER="$W/eval/scannet_rgb_depth_v51_s_oracle.py"
RELEASE_MANIFEST="${RELEASE_MANIFEST:-$W/artifacts/depth_v51_eval_interface/formal40/release.manifest.json}"
MODEL="${MODEL:?MODEL must name one complete merged checkpoint}"
MODEL_ROLE="${MODEL_ROLE:-evaluated}"
OUT="${OUT:?OUT must be a fresh, model-specific directory}"
CELL_TIMEOUT="${CELL_TIMEOUT:-2h}"
BATCH="${BATCH:-4}"
MAX_TOKENS="${MAX_TOKENS:-16}"

[[ "$BATCH" == 4 ]] || { echo "FATAL: frozen protocol requires BATCH=4" >&2; exit 2; }
[[ "$MAX_TOKENS" == 16 ]] || { echo "FATAL: frozen protocol requires MAX_TOKENS=16" >&2; exit 2; }
case "${THINKING:-off}" in
  off|0|false) ;;
  *) echo "FATAL: frozen protocol requires THINKING=off" >&2; exit 2 ;;
esac

if [[ -e "$OUT" ]]; then
  [[ -d "$OUT" ]] || { echo "FATAL: OUT is not a directory: $OUT" >&2; exit 2; }
  [[ -s "$OUT/formal40_inputs.json" && -s "$OUT/model_binding/model.json" ]] || {
    echo "FATAL: pre-existing OUT lacks a complete input/model binding: $OUT" >&2
    exit 2
  }
fi
mkdir -p "$OUT"/{model_binding,s_oracle,logs}
INPUT_RECEIPT="$OUT/formal40_inputs.json"
MODEL_BINDING="$OUT/model_binding/model.json"

TEMP_PATHS=()
cleanup_temps() {
  local path
  for path in "${TEMP_PATHS[@]:-}"; do rm -f "$path"; done
}
trap cleanup_temps EXIT

receipt_tmp=$(mktemp "$OUT/.formal40_inputs.XXXXXX.json")
TEMP_PATHS+=("$receipt_tmp")
"$PYTHON_BIN" "$RUNNER" preflight \
  --release-manifest "$RELEASE_MANIFEST" --out "$receipt_tmp"
if [[ -e "$INPUT_RECEIPT" ]]; then
  cmp -s "$receipt_tmp" "$INPUT_RECEIPT" || {
    echo "FATAL: pre-existing input receipt differs from the frozen release: $OUT" >&2
    exit 2
  }
else
  mv "$receipt_tmp" "$INPUT_RECEIPT"
fi

binding_tmp=$(mktemp "$OUT/model_binding/.model.XXXXXX.json")
TEMP_PATHS+=("$binding_tmp")
"$PYTHON_BIN" "$RUNNER" bind-model \
  --role "$MODEL_ROLE" --model "$MODEL" \
  --input-receipt "$INPUT_RECEIPT" --out "$binding_tmp"
if [[ -e "$MODEL_BINDING" ]]; then
  cmp -s "$binding_tmp" "$MODEL_BINDING" || {
    echo "FATAL: pre-existing model binding differs from MODEL: $OUT" >&2
    exit 2
  }
else
  mv "$binding_tmp" "$MODEL_BINDING"
fi
cleanup_temps
TEMP_PATHS=()
trap - EXIT

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "SCANNET_RGB_DEPTH_V51_S_ORACLE_PREFLIGHT_READY: no GPU query or process was started"
  exit 0
fi

gpu_pid_output=$(nvidia-smi --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null) || {
  echo "FATAL: nvidia-smi compute-process query failed" >&2
  exit 3
}
mapfile -t GPU_PIDS < <(printf '%s\n' "$gpu_pid_output" | awk 'NF' | sort -u)
if [[ "${#GPU_PIDS[@]}" -gt 0 ]]; then
  echo "FATAL: visible node already has GPU compute processes: ${GPU_PIDS[*]}" >&2
  exit 3
fi

"$PYTHON_BIN" - <<'PY'
import torch

if torch.cuda.device_count() != 4:
    raise SystemExit("FATAL: S-oracle evaluation requires exactly four visible GPUs")
uuids = []
for index in range(4):
    props = torch.cuda.get_device_properties(index)
    name = props.name
    gib = props.total_memory / 2**30
    uuid = str(getattr(props, "uuid", ""))
    if "L20" not in name.upper() or gib < 44 or not uuid or uuid.upper().startswith("MIG"):
        raise SystemExit(f"FATAL: GPU {index} is not a physical >=44 GiB L20: {name}")
    uuids.append(uuid)
if len(set(uuids)) != 4:
    raise SystemExit("FATAL: visible GPUs are not four unique physical devices")
print("4x physical L20 hardware gate: READY")
PY

gpu_inventory_output=$(nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv,noheader,nounits 2>/dev/null) || {
  echo "FATAL: nvidia-smi hardware-inventory query failed" >&2
  exit 3
}
printf '%s\n' "$gpu_inventory_output" > "$OUT/logs/gpu_hardware.csv"
RUN_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_IDS=(0 1 2 3)
fi
[[ "${#GPU_IDS[@]}" == 4 ]] || { echo "FATAL: expected four GPU identifiers" >&2; exit 2; }

CURRENT_PIDS=()
cleanup_workers() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${CURRENT_PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${CURRENT_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup_workers EXIT INT TERM

stage="$OUT/s_oracle"
CURRENT_PIDS=()
for shard in 0 1 2 3; do
  shard_out="$stage/shard_${shard}.jsonl"
  if [[ -e "$shard_out" ]]; then
    "$PYTHON_BIN" "$RUNNER" verify-shard \
      --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
      --track s-oracle --shard "$shard_out" --expected-shard "$shard" >/dev/null
    echo "[s_oracle] reusing verified shard $shard"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" timeout "$CELL_TIMEOUT" \
    "$PYTHON_BIN" "$RUNNER" infer-shard \
      --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
      --track s-oracle --shard "$shard" --num-shards 4 \
      --batch-size "$BATCH" --max-tokens "$MAX_TOKENS" --out "$shard_out" \
      >"$OUT/logs/s_oracle_shard_${shard}.log" 2>&1 &
  CURRENT_PIDS+=("$!")
done

rc=0
for pid in "${CURRENT_PIDS[@]:-}"; do wait "$pid" || rc=1; done
CURRENT_PIDS=()
[[ "$rc" == 0 ]] || {
  echo "FATAL: S-oracle inference failed; inspect $OUT/logs" >&2
  exit 1
}

"$PYTHON_BIN" "$RUNNER" merge-shards \
  --input-receipt "$INPUT_RECEIPT" --model-binding "$MODEL_BINDING" \
  --track s-oracle \
  --predictions "$stage/shard_0.jsonl" "$stage/shard_1.jsonl" \
                "$stage/shard_2.jsonl" "$stage/shard_3.jsonl" \
  --out "$stage/predictions.jsonl" \
  --manifest-out "$stage/predictions.manifest.json"

"$PYTHON_BIN" "$SCORER" score-s-oracle \
  --release-manifest "$RELEASE_MANIFEST" \
  --model-binding "$MODEL_BINDING" \
  --s-oracle-predictions "$stage/predictions.jsonl" \
  --s-oracle-manifest "$stage/predictions.manifest.json" \
  --out "$stage/result.json"

RUN_FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$PYTHON_BIN" - "$W" "$0" "$INPUT_RECEIPT" "$MODEL_BINDING" \
  "$stage/predictions.jsonl" "$stage/predictions.manifest.json" \
  "$stage/result.json" "$OUT/logs/gpu_hardware.csv" \
  "$RUN_STARTED_AT" "$RUN_FINISHED_AT" "$stage/execution_receipt.json" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    root_value,
    launcher_value,
    input_value,
    model_value,
    predictions_value,
    manifest_value,
    result_value,
    hardware_value,
    started_at,
    finished_at,
    out_value,
) = sys.argv[1:]
root = Path(root_value).resolve()


def binding(value: str) -> dict[str, object]:
    path = Path(value).resolve()
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


result_path = Path(result_value).resolve()
result = json.loads(result_path.read_text(encoding="utf-8"))
if (
    result.get("status") != "FORMAL_S_ORACLE_SCORE_COMPLETE"
    or (result.get("trust") or {}).get("score_available") is not True
):
    raise SystemExit("FATAL: scorer did not produce a trusted formal S-oracle result")

document = {
    "schema_version": 1,
    "protocol": "gsa-depth-v5.1-formal40-s-oracle-execution-v1",
    "started_at_utc": started_at,
    "finished_at_utc": finished_at,
    "launcher": binding(launcher_value),
    "standalone_scorer": binding(str(root / "eval/scannet_rgb_depth_v51_s_oracle.py")),
    "input_receipt": binding(input_value),
    "model_binding": binding(model_value),
    "predictions": binding(predictions_value),
    "prediction_manifest": binding(manifest_value),
    "formal_result": binding(result_value),
    "hardware_inventory": binding(hardware_value),
    "physical_gpu_count": 4,
    "training_materialization_eligible": False,
}
unsigned = json.dumps(
    document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
document["identity_sha256"] = hashlib.sha256(unsigned).hexdigest()

out = Path(out_value).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_value = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
temporary = Path(temporary_value)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(out)
except Exception:
    temporary.unlink(missing_ok=True)
    raise
PY

trap - EXIT INT TERM
echo "SCANNET_RGB_DEPTH_V51_S_ORACLE_COMPLETE: $stage/result.json"
