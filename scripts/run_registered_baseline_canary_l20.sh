#!/usr/bin/env bash
# Registry-gated, single-L20 baseline diagnostic. This launcher never writes formal results.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || {
  echo "FATAL: environment activation file is missing or unreadable: $W/.env" >&2
  exit 2
}
# shellcheck disable=SC1091
source "$W/.env"

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

PYTHON_BIN="${PYTHON_BIN:-python}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
RUN_BENCHMARK="${RUN_BENCHMARK:-$W/eval/run_benchmark_template_diagnostic.py}"
REQUIRED_FORMAL_EVAL_FINGERPRINT="${REQUIRED_FORMAL_EVAL_FINGERPRINT:-eb873c0242ad5d08290d13c2608d7c7c0dc79b8521128d6b3e77986c0a459497}"
REQUIRED_FORMAL_ENTRYPOINT_SHA256="${REQUIRED_FORMAL_ENTRYPOINT_SHA256:-044cdfad7aa9c58068b6f05e26a07f95f174bc8540b538c4e3ae9eb0b21f186a}"
REGISTRY="${REGISTRY:-$W/configs/models/baselines.yaml}"
DOWNLOAD_AUDIT_DIR="${DOWNLOAD_AUDIT_DIR:-$W/artifacts/downloads}"
MODEL_ID="${MODEL_ID:-}"
BENCHMARKS="${BENCHMARKS:-cv_bench embspatial refspatial_bench where2place gsa_bench_test}"
LIMIT="${LIMIT:-8}"
BATCH="${BATCH:-1}"
SEED="${SEED:-0}"
THINKING="${THINKING:-off}"
CELL_TIMEOUT="${CELL_TIMEOUT:-2h}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
TAG="${TAG:-${MODEL_ID:-missing_model}_single_l20_limit${LIMIT}_batch${BATCH}_seed${SEED}_${STAMP}}"
OUT="${OUT:-$W/artifacts/diagnostic_runs/baseline_canaries/$TAG}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-/tmp/gsa_registered_baseline_cache_${TAG}}"
PROTOCOL_ID="registered-baseline-single-l20-diagnostic-v1"
DIAGNOSTIC_EVAL_PROTOCOL="gsa-eval-v7-template-diagnostic-v1"
MIN_L20_MIB=$((44 * 1024))
MIN_RUNTIME_CACHE_BYTES=$((4 * 1024 * 1024 * 1024))

[[ -n "$MODEL_ID" ]] || fatal "MODEL_ID is required"
[[ "$MODEL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fatal "unsafe MODEL_ID '$MODEL_ID'"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fatal "unsafe diagnostic TAG '$TAG'"
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || fatal "LIMIT must be a positive integer, got '$LIMIT'"
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || fatal "BATCH must be a positive integer, got '$BATCH'"
[[ "$SEED" =~ ^[0-9]+$ ]] || fatal "SEED must be a non-negative integer, got '$SEED'"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || \
  fatal "PREFLIGHT_ONLY must be 0 or 1, got '$PREFLIGHT_ONLY'"
[[ "${THINKING,,}" == "off" ]] || fatal "THINKING must be off for this diagnostic"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fatal "Python executable not found: $PYTHON_BIN"
command -v "$NVIDIA_SMI_BIN" >/dev/null 2>&1 || fatal "GPU inventory tool not found: $NVIDIA_SMI_BIN"
command -v git >/dev/null 2>&1 || fatal "git is required to pin the ms-swift runtime"
[[ -r "$REGISTRY" ]] || fatal "baseline registry is missing or unreadable: $REGISTRY"
[[ -r "$RUN_BENCHMARK" ]] || fatal "benchmark entry point is missing: $RUN_BENCHMARK"
EXPECTED_DIAGNOSTIC_ENTRYPOINT=$(realpath -m "$W/eval/run_benchmark_template_diagnostic.py")
FORMAL_ENTRYPOINT=$(realpath -m "$W/eval/run_benchmark.py")
RUN_BENCHMARK=$(realpath -m "$RUN_BENCHMARK")
[[ "$RUN_BENCHMARK" == "$EXPECTED_DIAGNOSTIC_ENTRYPOINT" ]] || \
  fatal "RUN_BENCHMARK must be the isolated diagnostic entrypoint: $EXPECTED_DIAGNOSTIC_ENTRYPOINT"
[[ -r "$FORMAL_ENTRYPOINT" ]] || fatal "frozen formal entrypoint is missing: $FORMAL_ENTRYPOINT"

# This is an evidence-isolated diagnostic launcher: even explicit OUT overrides must
# remain below artifacts/diagnostic_runs, never in a formal or result-indexed tree.
W_ABS=$(realpath -m "$W")
DIAGNOSTIC_ROOT=$(realpath -m "$W/artifacts/diagnostic_runs")
OUT=$(realpath -m "$OUT")
case "$OUT" in
  "$DIAGNOSTIC_ROOT"/*) ;;
  *) fatal "OUT must be below $DIAGNOSTIC_ROOT; formal/result paths are forbidden: $OUT" ;;
esac
case "$OUT" in
  "$W_ABS/eval_results"|"$W_ABS/eval_results/"*|\
  "$W_ABS/runs"|"$W_ABS/runs/"*|\
  "$W_ABS/artifacts/results"|"$W_ABS/artifacts/results/"*)
    fatal "diagnostic output may not enter eval_results, runs, or artifacts/results: $OUT"
    ;;
esac
mkdir -p "$OUT"

RUNTIME_CACHE_ROOT=$(realpath -m "$RUNTIME_CACHE_ROOT")
mkdir -p "$RUNTIME_CACHE_ROOT"
cache_available=$(df -P -B1 "$RUNTIME_CACHE_ROOT" | awk 'NR == 2 {print $4}')
[[ "$cache_available" =~ ^[0-9]+$ && "$cache_available" -ge "$MIN_RUNTIME_CACHE_BYTES" ]] || \
  fatal "runtime cache filesystem has ${cache_available:-unknown} bytes free; at least $MIN_RUNTIME_CACHE_BYTES are required"

# Keep the benchmark surface frozen to the unified evaluator's five registered tasks.
# shellcheck disable=SC2206
BENCH_ARRAY=($BENCHMARKS)
[[ "${#BENCH_ARRAY[@]}" -ge 1 ]] || fatal "BENCHMARKS is empty"
declare -A KNOWN_BENCH=(
  [cv_bench]=1 [embspatial]=1 [refspatial_bench]=1
  [where2place]=1 [gsa_bench_test]=1
)
for benchmark in "${BENCH_ARRAY[@]}"; do
  [[ -n "${KNOWN_BENCH[$benchmark]:-}" ]] || fatal "unsupported baseline benchmark '$benchmark'"
done

# Resolve MODEL_ID through the controlled registry and write an immutable launch-time
# checkpoint inventory. Hugging Face entries need both a full commit revision and the
# matching audit emitted by scripts/download_baselines.py.
MODEL_MANIFEST="$OUT/model_manifest.json"
MODEL=$(
  "$PYTHON_BIN" - "$REGISTRY" "$MODEL_ID" "$DOWNLOAD_AUDIT_DIR" "$MODEL_MANIFEST" <<'PY'
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"FATAL: PyYAML is required to read the baseline registry: {exc}") from exc

registry_path = Path(sys.argv[1]).resolve()
model_id = sys.argv[2]
audit_dir = Path(sys.argv[3]).resolve()
manifest_path = Path(sys.argv[4])
try:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"FATAL: unreadable baseline registry {registry_path}: {exc}") from exc
models = registry.get("models") if isinstance(registry, dict) else None
if not isinstance(models, list):
    raise SystemExit("FATAL: baseline registry has no models list")
matches = [item for item in models if isinstance(item, dict) and item.get("id") == model_id]
if len(matches) != 1:
    raise SystemExit(f"FATAL: MODEL_ID {model_id!r} must match exactly one registry entry; found {len(matches)}")
entry = matches[0]
if entry.get("table_eligible") is not True:
    raise SystemExit(f"FATAL: {model_id} is not table_eligible")
if entry.get("harness") != "unified_gsa_eval":
    raise SystemExit(f"FATAL: {model_id} is not assigned to unified_gsa_eval")
if entry.get("protocol") != "image_text_to_text":
    raise SystemExit(f"FATAL: {model_id} has incompatible protocol {entry.get('protocol')!r}")
source = entry.get("source")
if source not in {"local", "huggingface"}:
    raise SystemExit(f"FATAL: unsupported registry source for {model_id}: {source!r}")
template_type = entry.get("template_type")
if template_type is not None and (
    not isinstance(template_type, str)
    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", template_type) is None
):
    raise SystemExit(
        f"FATAL: invalid template_type for {model_id}: {template_type!r}"
    )
model = Path(str(entry.get("path", ""))).resolve()
if not model.is_dir():
    raise SystemExit(f"FATAL: registered checkpoint directory is missing: {model}")
config_path = model / "config.json"
if not config_path.is_file() or not config_path.stat().st_size:
    raise SystemExit(f"FATAL: registered checkpoint has no non-empty config.json: {model}")
try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"FATAL: invalid checkpoint config.json: {exc}") from exc
weights = sorted({
    path for pattern in ("*.safetensors", "*.bin", "*.pth")
    for path in model.glob(pattern) if path.is_file()
})
if not weights or any(not path.stat().st_size for path in weights):
    raise SystemExit(f"FATAL: checkpoint has no complete root-level weight files: {model}")
index_path = model / "model.safetensors.index.json"
index_weight_count = None
if index_path.is_file():
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL: invalid model weight index: {exc}") from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("FATAL: model weight index has no non-empty weight_map")
    referenced = {model / str(name) for name in weight_map.values()}
    missing = sorted(str(path) for path in referenced if not path.is_file() or not path.stat().st_size)
    if missing:
        raise SystemExit(f"FATAL: model index references missing/empty shards: {missing[:5]}")
    index_weight_count = len(weight_map)

download_audit = None
revision = entry.get("revision")
if source == "huggingface":
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise SystemExit(
            f"FATAL: Hugging Face baseline {model_id} requires a full 40-hex pinned revision; got {revision!r}"
        )
    audit_path = audit_dir / f"{model_id}.json"
    if not audit_path.is_file():
        raise SystemExit(f"FATAL: pinned download audit is missing for {model_id}: {audit_path}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL: invalid download audit for {model_id}: {exc}") from exc
    expected = {
        "model_id": model_id,
        "repo_id": entry.get("repo_id"),
        "requested_revision": revision,
    }
    mismatches = {key: (audit.get(key), value) for key, value in expected.items() if audit.get(key) != value}
    try:
        audit_destination = Path(str(audit.get("destination", ""))).resolve()
    except Exception:
        audit_destination = Path("/")
    if audit_destination != model:
        mismatches["destination"] = (str(audit_destination), str(model))
    if mismatches:
        raise SystemExit(f"FATAL: download audit does not match registry for {model_id}: {mismatches}")
    download_audit = {
        "path": str(audit_path),
        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "requested_revision": audit["requested_revision"],
        "downloaded_at_utc": audit.get("downloaded_at_utc"),
        "local_bytes": audit.get("local_bytes"),
    }

identity_files = {}
for name in (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "chat_template.json",
    "chat_template.jinja",
):
    path = model / name
    if path.is_file():
        identity_files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
record = {
    "schema_version": 1,
    "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "registry_path": str(registry_path),
    "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    "model_id": model_id,
    "display_name": entry.get("display_name"),
    "source": source,
    "repo_id": entry.get("repo_id"),
    "revision": revision,
    "path": str(model),
    "protocol": entry["protocol"],
    "harness": entry["harness"],
    "template_type": template_type,
    "table_eligible": True,
    "model_type": config.get("model_type") or config.get("type"),
    "architectures": config.get("architectures") or [],
    "weight_file_count": len(weights),
    "weight_bytes": sum(path.stat().st_size for path in weights),
    "weight_files": [{"name": path.name, "size_bytes": path.stat().st_size} for path in weights],
    "index_weight_count": index_weight_count,
    "identity_files": identity_files,
    "download_audit": download_audit,
}
manifest_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(model)
PY
) || exit $?
[[ -n "$MODEL" ]] || fatal "registry resolution returned an empty model path"
TEMPLATE_TYPE=$(
  "$PYTHON_BIN" - "$MODEL_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(record.get("template_type") or "")
PY
) || exit $?
TEMPLATE_ARGS=()
if [[ -n "$TEMPLATE_TYPE" ]]; then
  TEMPLATE_ARGS=(--template-type "$TEMPLATE_TYPE")
fi

# Exactly one caller-selected, idle L20 is mandatory. Checking emptiness during
# PREFLIGHT_ONLY makes the printed READY token an honest launch-time admission result.
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fatal "set CUDA_VISIBLE_DEVICES to exactly one L20 selector"
GPU_SELECTOR=$(trim "$CUDA_VISIBLE_DEVICES")
[[ -n "$GPU_SELECTOR" && "$GPU_SELECTOR" != *,* ]] || \
  fatal "exactly one visible GPU selector is required; got '$CUDA_VISIBLE_DEVICES'"
GPU_QUERY=$(
  "$NVIDIA_SMI_BIN" -i "$GPU_SELECTOR" \
    --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null
) || fatal "cannot query GPU selector '$GPU_SELECTOR'"
mapfile -t GPU_ROWS < <(printf '%s\n' "$GPU_QUERY" | awk 'NF')
[[ "${#GPU_ROWS[@]}" == 1 ]] || fatal "GPU selector '$GPU_SELECTOR' did not resolve to exactly one device"
IFS=',' read -r GPU_INDEX GPU_UUID GPU_NAME GPU_MEMORY_MIB GPU_DRIVER <<< "${GPU_ROWS[0]}"
GPU_INDEX=$(trim "$GPU_INDEX")
GPU_UUID=$(trim "$GPU_UUID")
GPU_NAME=$(trim "$GPU_NAME")
GPU_MEMORY_MIB=$(trim "$GPU_MEMORY_MIB")
GPU_MEMORY_MIB="${GPU_MEMORY_MIB//[!0-9]/}"
GPU_DRIVER=$(trim "$GPU_DRIVER")
[[ "${GPU_NAME^^}" == *L20* ]] || fatal "GPU selector '$GPU_SELECTOR' is not an L20: '$GPU_NAME'"
[[ "$GPU_MEMORY_MIB" =~ ^[0-9]+$ && "$GPU_MEMORY_MIB" -ge "$MIN_L20_MIB" ]] || \
  fatal "GPU selector '$GPU_SELECTOR' has ${GPU_MEMORY_MIB:-unknown} MiB; at least $MIN_L20_MIB MiB is required"
BUSY=$(
  "$NVIDIA_SMI_BIN" -i "$GPU_SELECTOR" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
)
[[ -z "$(printf '%s\n' "$BUSY" | awk '$1 ~ /^[0-9]+$/ {print $1; exit}')" ]] || \
  fatal "GPU selector '$GPU_SELECTOR' already has a compute process"

formal_eval_fingerprint() {
  {
    local name
    for name in datasets.py parsing.py registry.py runner.py scorers.py; do
      [[ -r "$W/eval/gsa_eval/$name" ]] || return 1
      printf '%s\0' "$name"
      command cat "$W/eval/gsa_eval/$name"
      printf '\0'
    done
  } | sha256sum | awk '{print $1}'
}

diagnostic_eval_fingerprint() {
  {
    local relative
    for relative in \
      gsa_eval/datasets.py gsa_eval/parsing.py gsa_eval/registry.py \
      gsa_eval/runner.py gsa_eval/scorers.py \
      run_benchmark.py \
      gsa_eval/template_diagnostic.py run_benchmark_template_diagnostic.py; do
      [[ -r "$W/eval/$relative" ]] || return 1
      printf '%s\0' "$relative"
      command cat "$W/eval/$relative"
      printf '\0'
    done
  } | sha256sum | awk '{print $1}'
}

PINNED_FORMAL_EVAL_FINGERPRINT=$(formal_eval_fingerprint) || \
  fatal "cannot fingerprint frozen formal evaluator sources"
[[ "$PINNED_FORMAL_EVAL_FINGERPRINT" == "$REQUIRED_FORMAL_EVAL_FINGERPRINT" ]] || \
  fatal "formal evaluator drift: required=$REQUIRED_FORMAL_EVAL_FINGERPRINT current=$PINNED_FORMAL_EVAL_FINGERPRINT"
FORMAL_ENTRYPOINT_SHA256=$(sha256sum "$FORMAL_ENTRYPOINT" | awk '{print $1}')
[[ "$FORMAL_ENTRYPOINT_SHA256" == "$REQUIRED_FORMAL_ENTRYPOINT_SHA256" ]] || \
  fatal "formal entrypoint drift: required=$REQUIRED_FORMAL_ENTRYPOINT_SHA256 current=$FORMAL_ENTRYPOINT_SHA256"
PINNED_EVAL_FINGERPRINT=$(diagnostic_eval_fingerprint) || \
  fatal "cannot fingerprint isolated diagnostic evaluator sources"
export PINNED_FORMAL_EVAL_FINGERPRINT PINNED_EVAL_FINGERPRINT
export PINNED_EVAL_PROTOCOL="$DIAGNOSTIC_EVAL_PROTOCOL"
MS_SWIFT_REPO="${MS_SWIFT_ROOT:-}"
[[ -d "$MS_SWIFT_REPO/.git" ]] || fatal "MS_SWIFT_ROOT is not a git checkout: ${MS_SWIFT_REPO:-unset}"

RUNTIME_MANIFEST="$OUT/runtime_manifest.json"
"$PYTHON_BIN" - "$RUNTIME_MANIFEST" "$PINNED_EVAL_FINGERPRINT" \
  "$PINNED_FORMAL_EVAL_FINGERPRINT" "$DIAGNOSTIC_EVAL_PROTOCOL" "$MS_SWIFT_REPO" \
  "$FORMAL_ENTRYPOINT" "$FORMAL_ENTRYPOINT_SHA256" \
  "$RUN_BENCHMARK" "${BASH_SOURCE[0]}" "$GPU_SELECTOR" "$GPU_INDEX" "$GPU_UUID" \
  "$GPU_NAME" "$GPU_MEMORY_MIB" "$GPU_DRIVER" "$TEMPLATE_TYPE" <<'PY'
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

(
    output, eval_hash, formal_eval_hash, eval_protocol, swift_repo,
    formal_runner, formal_runner_hash, runner, launcher, selector, gpu_index,
    gpu_uuid, gpu_name, gpu_memory, gpu_driver, template_type,
) = sys.argv[1:]
repo = Path(swift_repo).resolve()

def git(*args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True,
        text=not binary, check=False,
    )
    if result.returncode:
        stderr = result.stderr.decode(errors="replace") if binary else result.stderr
        raise SystemExit(f"FATAL: git {' '.join(args)} failed for ms-swift: {stderr}")
    return result.stdout

head = git("rev-parse", "HEAD").strip()
diff = git("diff", "--binary", "HEAD", binary=True)
status = git("status", "--porcelain=v1").splitlines()

def version(*names: str) -> str:
    for name in names:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            pass
    raise SystemExit(f"FATAL: required runtime distribution is not installed: {names[0]}")

runner_path = Path(runner).resolve()
formal_runner_path = Path(formal_runner).resolve()
launcher_path = Path(launcher).resolve()
record = {
    "schema_version": 1,
    "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    },
    "packages": {
        "torch": version("torch"),
        "transformers": version("transformers"),
        "ms-swift": version("ms_swift", "ms-swift"),
        "PyYAML": version("PyYAML"),
    },
    "evaluator": {
        "protocol": eval_protocol,
        "code_sha256": eval_hash,
        "formal_code_sha256": formal_eval_hash,
        "formal_entrypoint": str(formal_runner_path),
        "formal_entrypoint_sha256": formal_runner_hash,
        "entrypoint": str(runner_path),
        "entrypoint_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        "template_type": template_type or None,
        "template_type_source": "model_registry" if template_type else "automatic",
    },
    "launcher": {
        "path": str(launcher_path),
        "sha256": hashlib.sha256(launcher_path.read_bytes()).hexdigest(),
    },
    "ms_swift": {
        "repo": str(repo),
        "git_head": head,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty": bool(status),
        "status_porcelain": status,
    },
    "gpu": {
        "requested_selector": selector,
        "physical_index": gpu_index,
        "uuid": gpu_uuid,
        "name": gpu_name,
        "memory_total_mib": int(gpu_memory),
        "driver_version": gpu_driver,
    },
}
Path(output).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

PROTOCOL_MANIFEST="$OUT/protocol_manifest.json"
"$PYTHON_BIN" - "$PROTOCOL_MANIFEST" "$MODEL_MANIFEST" "$RUNTIME_MANIFEST" \
  "$PROTOCOL_ID" "$TAG" "$OUT" "$LIMIT" "$BATCH" "$SEED" "$THINKING" \
  "$RUNTIME_CACHE_ROOT" "${BENCH_ARRAY[@]}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

(
    output, model_manifest, runtime_manifest, protocol, tag, out_dir,
    limit, batch, seed, thinking, runtime_cache, *benchmarks,
) = sys.argv[1:]
model = json.loads(Path(model_manifest).read_text(encoding="utf-8"))
runtime = json.loads(Path(runtime_manifest).read_text(encoding="utf-8"))
model_template = model.get("template_type")
runtime_template = runtime["evaluator"].get("template_type")
if model_template != runtime_template:
    raise SystemExit(
        f"FATAL: model/runtime template mismatch: {model_template!r} != {runtime_template!r}"
    )
record = {
    "schema_version": 1,
    "protocol": protocol,
    "tier": "diagnostic",
    "formal_result_registry_eligible": False,
    "exclusion_reasons": [
        f"seeded limit-{limit} diagnostic subset rather than a full split",
        "single-process single-L20 canary topology",
    ],
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "tag": tag,
    "output": str(Path(out_dir).resolve()),
    "model_id": model["model_id"],
    "model_path": model["path"],
    "model_registry_sha256": model["registry_sha256"],
    "model_revision": model.get("revision"),
    "template_type": model_template,
    "template_type_source": runtime["evaluator"]["template_type_source"],
    "download_audit_sha256": (model.get("download_audit") or {}).get("sha256"),
    "benchmarks": benchmarks,
    "subset_limit": int(limit),
    "subset_seed": int(seed),
    "generation_batch_size": int(batch),
    "thinking": False,
    "thinking_control": "runner --no-thinking",
    "process_count": 1,
    "device_map": "cuda:0",
    "cuda_visible_devices": runtime["gpu"]["requested_selector"],
    "gpu": runtime["gpu"],
    "runtime_cache_root": str(Path(runtime_cache).resolve()),
    "evaluation_code_sha256": runtime["evaluator"]["code_sha256"],
    "formal_evaluation_code_sha256": runtime["evaluator"]["formal_code_sha256"],
    "formal_entrypoint_sha256": runtime["evaluator"]["formal_entrypoint_sha256"],
    "evaluation_protocol": runtime["evaluator"]["protocol"],
    "ms_swift_git_head": runtime["ms_swift"]["git_head"],
    "ms_swift_git_diff_sha256": runtime["ms_swift"]["git_diff_sha256"],
    "result_registry": "excluded; diagnostic artifacts only",
}
Path(output).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "=================================================="
echo " protocol       : $PROTOCOL_ID"
echo " model          : $MODEL_ID -> $MODEL"
echo " template       : ${TEMPLATE_TYPE:-automatic}"
echo " tier           : diagnostic only; excluded from the formal result registry"
echo " physical GPU   : $GPU_SELECTOR ($GPU_NAME, ${GPU_MEMORY_MIB} MiB, empty)"
echo " placement      : one process, device-map cuda:0"
echo " generation     : THINKING=off via --no-thinking"
echo " benchmarks     : ${BENCH_ARRAY[*]}"
echo " limit/batch    : $LIMIT / $BATCH"
echo " formal eval fp : $PINNED_FORMAL_EVAL_FINGERPRINT"
echo " formal CLI sha : $FORMAL_ENTRYPOINT_SHA256"
echo " diagnostic fp  : $PINNED_EVAL_FINGERPRINT"
echo " output         : $OUT"
echo "=================================================="

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "REGISTERED_BASELINE_CANARY_PREFLIGHT_READY: no CUDA process was started"
  exit 0
fi

[[ ! -e "$OUT/run.started" && ! -d "$OUT/shards" ]] || \
  fatal "diagnostic run state already exists; choose a fresh TAG or OUT"

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

GPU_TRACE="$OUT/gpu_memory_raw.csv"
GPU_SUMMARY="$OUT/gpu_memory_summary.json"
printf '%s\n' 'timestamp,index,name,memory_used_mib,memory_total_mib,utilization_gpu_percent' > "$GPU_TRACE"
MONITOR_PID=""

monitor_gpu() {
  while true; do
    "$NVIDIA_SMI_BIN" -i "$GPU_SELECTOR" \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits >> "$GPU_TRACE" 2>/dev/null || true
    sleep 1
  done
}

stop_monitor() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

write_gpu_summary() {
  "$PYTHON_BIN" - "$GPU_TRACE" "$GPU_SUMMARY" "$GPU_SELECTOR" <<'PY'
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
output = Path(sys.argv[2])
selector = sys.argv[3]
rows = []
with raw.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        try:
            rows.append({
                "timestamp": row["timestamp"].strip(),
                "index": row["index"].strip(),
                "name": row["name"].strip(),
                "memory_used_mib": int(row["memory_used_mib"].strip()),
                "memory_total_mib": int(row["memory_total_mib"].strip()),
                "utilization_gpu_percent": int(row["utilization_gpu_percent"].strip()),
            })
        except (KeyError, TypeError, ValueError):
            continue
peak = max((row["memory_used_mib"] for row in rows), default=0)
total = max((row["memory_total_mib"] for row in rows), default=0)
record = {
    "schema_version": 1,
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "sample_interval_seconds": 1,
    "requested_selector": selector,
    "samples": len(rows),
    "peak_used_mib": peak,
    "total_mib": total,
    "minimum_headroom_mib": max(total - peak, 0) if total else None,
    "raw_trace": str(raw.resolve()),
    "raw_trace_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
}
output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

trap stop_monitor EXIT
trap 'stop_monitor; exit 130' INT TERM
monitor_gpu &
MONITOR_PID=$!

start_seconds=$(date +%s)
generation_rc=0
CUDA_VISIBLE_DEVICES="$GPU_SELECTOR" timeout "$CELL_TIMEOUT" \
  "$PYTHON_BIN" "$RUN_BENCHMARK" \
    --model "$MODEL" --benchmarks "${BENCH_ARRAY[@]}" \
    --shard-dir "$OUT/shards" --out "$OUT/worker.json" \
    --shard 0 --num-shards 1 --limit "$LIMIT" --batch-size "$BATCH" \
    --seed "$SEED" --device-map cuda:0 --no-thinking \
    "${TEMPLATE_ARGS[@]}" \
    > "$OUT/logs/generation.log" 2>&1 || generation_rc=$?
elapsed_seconds=$(( $(date +%s) - start_seconds ))
stop_monitor
write_gpu_summary

if [[ "$generation_rc" -ne 0 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.failed"
  echo "FATAL: diagnostic generation failed with rc=$generation_rc" >&2
  echo "  log: $OUT/logs/generation.log" >&2
  tail -n 40 "$OUT/logs/generation.log" >&2 || true
  exit "$generation_rc"
fi

CURRENT_EVAL_FINGERPRINT=$(diagnostic_eval_fingerprint) || \
  fatal "cannot re-fingerprint diagnostic evaluator after generation"
if [[ "$CURRENT_EVAL_FINGERPRINT" != "$PINNED_EVAL_FINGERPRINT" ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.failed"
  fatal "evaluator changed during generation: pinned=$PINNED_EVAL_FINGERPRINT current=$CURRENT_EVAL_FINGERPRINT"
fi

"$PYTHON_BIN" - "$OUT/worker.json" "$OUT/shards" "${BENCH_ARRAY[@]}" <<'PY'
import json
import sys
from pathlib import Path

worker_path = Path(sys.argv[1])
shard_root = Path(sys.argv[2])
benchmarks = sys.argv[3:]
try:
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"FATAL: invalid worker manifest: {exc}") from exc
if worker.get("n_failed") != 0:
    raise SystemExit(f"FATAL: worker reports n_failed={worker.get('n_failed')}")
for benchmark in benchmarks:
    status = (worker.get("benchmarks") or {}).get(benchmark) or {}
    if status.get("status") != "ok" or not isinstance(status.get("n"), int) or status["n"] <= 0:
        raise SystemExit(f"FATAL: incomplete worker status for {benchmark}: {status}")
    shard = shard_root / benchmark / "shard_0.jsonl"
    if not shard.is_file():
        raise SystemExit(f"FATAL: missing shard for {benchmark}: {shard}")
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != status["n"] or len({str(row.get("uid")) for row in rows}) != len(rows):
        raise SystemExit(
            f"FATAL: shard cardinality/UID mismatch for {benchmark}: rows={len(rows)} expected={status['n']}"
        )
PY

score_rc=0
for benchmark in "${BENCH_ARRAY[@]}"; do
  result="$OUT/results/$benchmark.json"
  cell_rc=0
  CUDA_VISIBLE_DEVICES="$GPU_SELECTOR" "$PYTHON_BIN" "$RUN_BENCHMARK" \
      --model "$MODEL" --benchmark "$benchmark" --out "$result" \
      --limit "$LIMIT" --batch-size "$BATCH" --seed "$SEED" \
      --elapsed-s "$elapsed_seconds" --no-thinking \
      "${TEMPLATE_ARGS[@]}" \
      --score-shards "$OUT/shards/$benchmark/shard_0.jsonl" \
      --dump-predictions "$OUT/predictions/$benchmark.jsonl" \
      > "$OUT/logs/score_${benchmark}.log" 2>&1 || cell_rc=$?
  if [[ "$cell_rc" -ne 0 ]]; then
    score_rc=1
    echo "[$benchmark] trust gate withheld or scoring failed (rc=$cell_rc)" >&2
  fi
  if [[ -f "$result" ]]; then
    if ! "$PYTHON_BIN" - "$result" "$PINNED_EVAL_FINGERPRINT" \
      "$PINNED_FORMAL_EVAL_FINGERPRINT" "$DIAGNOSTIC_EVAL_PROTOCOL" \
      "$LIMIT" "$BATCH" "$SEED" "$TEMPLATE_TYPE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "evaluation_code_sha256": sys.argv[2],
    "formal_evaluation_code_sha256": sys.argv[3],
    "evaluation_protocol": sys.argv[4],
    "subset_limit": int(sys.argv[5]),
    "generation_batch_size": int(sys.argv[6]),
    "subset_seed": int(sys.argv[7]),
    "thinking": False,
    "template_type": sys.argv[8] or None,
}
mismatches = {key: (record.get(key), value) for key, value in expected.items() if record.get(key) != value}
if mismatches:
    raise SystemExit(f"result protocol mismatch: {mismatches}")
PY
    then
      score_rc=1
      echo "[$benchmark] result does not match pinned protocol" >&2
    fi
  else
    score_rc=1
    echo "[$benchmark] result JSON is missing" >&2
  fi
done

CURRENT_EVAL_FINGERPRINT=$(diagnostic_eval_fingerprint) || \
  fatal "cannot re-fingerprint diagnostic evaluator after scoring"
if [[ "$CURRENT_EVAL_FINGERPRINT" != "$PINNED_EVAL_FINGERPRINT" ]]; then
  score_rc=1
  echo "FATAL: evaluator changed during scoring" >&2
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.finished"
if [[ "$score_rc" -eq 0 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.completed"
  echo "registered baseline diagnostic complete: $OUT"
else
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.failed"
  echo "registered baseline diagnostic failed a scoring/trust gate: $OUT" >&2
fi
echo "These artifacts remain excluded from experiments/result_registry.yaml."
exit "$score_rc"
