#!/bin/bash
# Local vLLM OpenAI-compatible VLM server for GSA/A_robot_state.
# Install vLLM in the inference environment before running this script.

set -euo pipefail

VLLM_MODEL="${VLLM_MODEL:-/path/to/workspace/foundation/contributor/hf_models/Qwen/Qwen2.5-VL-7B-Instruct}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-10301}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [ -n "$VLLM_CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES"
fi

VLLM_VISIBLE_GPU_COUNT=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    VLLM_VISIBLE_GPU_COUNT=$(python - <<'PYCOUNT'
import os
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(visible) if visible else "")
PYCOUNT
)
fi

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${VLLM_VISIBLE_GPU_COUNT:-1}}"
if [ -n "$VLLM_VISIBLE_GPU_COUNT" ] && [ "$VLLM_TENSOR_PARALLEL_SIZE" -gt "$VLLM_VISIBLE_GPU_COUNT" ]; then
    echo "ERROR: VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE exceeds visible GPU count=$VLLM_VISIBLE_GPU_COUNT" >&2
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_ENV="${VLLM_ENV:-/path/to/workspace/foundation/contributor/envs/rex-vllm-local}"
VLLM_CACHE_HOME="${VLLM_CACHE_HOME:-/path/to/workspace/foundation/contributor/cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$VLLM_CACHE_HOME}"
export HF_HOME="${HF_HOME:-/path/to/workspace/foundation/contributor/hf-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torch_inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-}"
if [ -z "$VLLM_LIMIT_MM_PER_PROMPT" ]; then
    VLLM_LIMIT_MM_PER_PROMPT='{"image": 5}'
elif [[ "$VLLM_LIMIT_MM_PER_PROMPT" =~ ^image=([0-9]+)$ ]]; then
    VLLM_LIMIT_MM_PER_PROMPT="{\"image\": ${BASH_REMATCH[1]}}"
fi

VLLM_SITE_PACKAGES="$VLLM_ENV/lib/python3.11/site-packages"
VLLM_NVIDIA_LIB_PATHS=""
for lib_dir in \
    "$VLLM_SITE_PACKAGES/nvidia/cuda_runtime/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cublas/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cudnn/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cusolver/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cusparse/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cusparselt/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cufft/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/curand/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/nccl/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/nvjitlink/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cuda_nvrtc/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cuda_cupti/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/nvtx/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/cufile/lib" \
    "$VLLM_SITE_PACKAGES/nvidia/nvshmem/lib"; do
    if [ -d "$lib_dir" ]; then
        VLLM_NVIDIA_LIB_PATHS="${VLLM_NVIDIA_LIB_PATHS:+$VLLM_NVIDIA_LIB_PATHS:}$lib_dir"
    fi
done
export LD_LIBRARY_PATH="${VLLM_NVIDIA_LIB_PATHS:+$VLLM_NVIDIA_LIB_PATHS:}${LD_LIBRARY_PATH:-}"

VLLM_BIN="${VLLM_BIN:-}"
if [ -z "$VLLM_BIN" ]; then
    if [ -x "$VLLM_ENV/bin/vllm" ]; then
        VLLM_BIN="$VLLM_ENV/bin/vllm"
    elif command -v vllm >/dev/null 2>&1; then
        VLLM_BIN="$(command -v vllm)"
    else
        echo "ERROR: vllm command not found." >&2
        echo "Expected it at: $VLLM_ENV/bin/vllm" >&2
        echo "Override with: VLLM_ENV=/path/to/env or VLLM_BIN=/path/to/vllm" >&2
        exit 1
    fi
fi

VLLM_PY="${VLLM_PY:-$VLLM_ENV/bin/python}"
if [ ! -x "$VLLM_PY" ]; then
    VLLM_PY="$(command -v python3 || command -v python)"
fi

# Preflight: vLLM 0.11.x requires transformers<5; rope_scaling must use rope_type.
"$VLLM_PY" - <<'PYPREFLIGHT' "$VLLM_MODEL"
import json
import sys
from pathlib import Path

try:
    import transformers
except ImportError as exc:
    print(f"ERROR: transformers not installed in vLLM env: {exc}", file=sys.stderr)
    sys.exit(1)

version = transformers.__version__
major = int(version.split(".", 1)[0])
if major >= 5:
    print(
        f"ERROR: transformers {version} is incompatible with vLLM 0.11.x "
        f"(needs transformers>=4.56,<5).",
        file=sys.stderr,
    )
    print(
        f"Fix: {sys.executable} -m pip install 'transformers>=4.56.0,<5.0.0'",
        file=sys.stderr,
    )
    sys.exit(1)

model_path = Path(sys.argv[1])
config_path = model_path / "config.json"
if not config_path.exists():
    print(f"ERROR: model config not found: {config_path}", file=sys.stderr)
    sys.exit(1)

cfg = json.loads(config_path.read_text(encoding="utf-8"))
rope = cfg.get("rope_scaling") or {}
if rope.get("type") == "mrope" and "rope_type" not in rope:
    backup = config_path.with_suffix(".json.bak.vllm")
    if not backup.exists():
        backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Patched model config backup: {backup}")
    rope["rope_type"] = "mrope"
    rope.pop("type", None)
    cfg["rope_scaling"] = rope
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched rope_scaling in {config_path}")
PYPREFLIGHT

cat <<EOF
=== Local vLLM VLM Server ===
Model:              $VLLM_MODEL
Served model name:  $VLLM_SERVED_MODEL_NAME
Endpoint:           http://$VLLM_HOST:$VLLM_PORT/v1
CUDA devices:       ${CUDA_VISIBLE_DEVICES:-all}
Visible GPU count:  ${VLLM_VISIBLE_GPU_COUNT:-auto}
Tensor parallel:    $VLLM_TENSOR_PARALLEL_SIZE
GPU mem util:       $VLLM_GPU_MEMORY_UTILIZATION
Max model len:      $VLLM_MAX_MODEL_LEN
Max num seqs:       $VLLM_MAX_NUM_SEQS
DType:              $VLLM_DTYPE
MM limit:           $VLLM_LIMIT_MM_PER_PROMPT
vLLM binary:        $VLLM_BIN
EOF

exec "$VLLM_BIN" serve "$VLLM_MODEL"     --host "$VLLM_HOST"     --port "$VLLM_PORT"     --served-model-name "$VLLM_SERVED_MODEL_NAME"     --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"     --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"     --max-model-len "$VLLM_MAX_MODEL_LEN"     --max-num-seqs "$VLLM_MAX_NUM_SEQS"     --dtype "$VLLM_DTYPE"     --trust-remote-code     --limit-mm-per-prompt "$VLLM_LIMIT_MM_PER_PROMPT"
