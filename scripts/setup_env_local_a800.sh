#!/bin/bash
# =============================================================================
# GSA_Forge LOCAL env setup (this box: 4x A800-80GB, driver 535, CUDA 12.9)
# Goal: one env that can do SFT + GRPO + OPD via ms-swift on the PT/transformers
# backend (NO vLLM).
#
# IMPORTANT — vLLM is intentionally NOT installed here.
#   Qwen3.5/3.6 (GDN hybrid, model_type=qwen3_5) need vLLM>=0.20 + transformers>=5.0,
#   and those vLLM wheels are CUDA-13 / torch-2.11 builds requiring driver >= 575.
#   This box has driver 535 (CUDA 12.9 ceiling; cu121 torch tops out at 2.5.1), so a
#   qwen3_5-capable vLLM cannot run here. GRPO uses --use_vllm false (PT rollout) and
#   OPD/GKD generates with the teacher in-process. See docs/work_summary.md.
#
# Strategy: clone the known-good `qwen36` env (ms-swift 4.2 + transformers 5.5 +
# flash-attn + deepspeed), pin torch to 2.5.1+cu121. That's it.
# =============================================================================
set -uo pipefail

SRC_ENV="${SRC_ENV:-qwen36}"
NEW_ENV="${NEW_ENV:-gsa_train}"
ENV_ROOT="/path/to/environment/env"
export TMPDIR="${TMPDIR:-/path/to/environment/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/path/to/environment/.pipcache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

eval "$(conda shell.bash hook)"

if conda env list | grep -q "${ENV_ROOT}/${NEW_ENV}"; then
    echo "[INFO] env ${NEW_ENV} already exists at ${ENV_ROOT}/${NEW_ENV}"
else
    echo "[Step 1] Cloning ${SRC_ENV} -> ${NEW_ENV} (this can take a while)"
    conda create --prefix "${ENV_ROOT}/${NEW_ENV}" --clone "${ENV_ROOT}/${SRC_ENV}" -y
fi

PY="${ENV_ROOT}/${NEW_ENV}/bin/python"
PIP="${ENV_ROOT}/${NEW_ENV}/bin/pip"

echo "[Step 2] Ensure torch 2.5.1+cu121 (max for driver 535)"
$PIP install --no-cache-dir "torch==2.5.1" "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -4
$PIP install --no-cache-dir "transformers==5.5.*" 2>&1 | tail -3

echo "[Step 3] Verify imports (vLLM intentionally absent)"
$PY - <<'PYEOF'
import importlib
for m in ["torch", "transformers", "swift", "deepspeed", "trl"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK  {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"ERR {m}: {type(e).__name__}: {e}")
import torch
print("cuda_available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
PYEOF
echo "[DONE] env ${NEW_ENV} ready for SFT / GRPO(PT) / OPD(GKD in-process)."
