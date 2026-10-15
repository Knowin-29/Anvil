#!/bin/bash
# =============================================================================
# GSA_Forge 阿里云 H200 环境一键安装
# 适用: 8xH200-141G, NVIDIA Driver >= 550, CUDA 12.4+
# 用途: GRPO / OPD 蒸馏 (依赖 vLLM rollout)
# =============================================================================
set -euo pipefail

ENV_NAME="${GSA_ENV_NAME:-gsa_train}"
PYTHON_VER="3.11"
TORCH_VER="2.7.0"
CUDA_TAG="cu124"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=========================================="
echo " GSA_Forge 阿里云环境安装"
echo " conda env: ${ENV_NAME}"
echo " torch: ${TORCH_VER}+${CUDA_TAG}"
echo "=========================================="

# --- Step 1: 创建 conda 环境 ---
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] conda env '${ENV_NAME}' already exists, skipping create."
else
    echo "[Step 1/7] Creating conda env: ${ENV_NAME} (python ${PYTHON_VER})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# --- Step 2: PyTorch ---
echo "[Step 2/7] Installing PyTorch ${TORCH_VER}+${CUDA_TAG}"
pip install "torch==${TORCH_VER}" "torchvision" --index-url "${TORCH_INDEX}"

# --- Step 3: vLLM (支持 qwen3_5 hybrid 的版本) ---
echo "[Step 3/7] Installing vLLM (>= 0.17.2, qwen3_5 support)"
pip install "vllm>=0.17.2"

# --- Step 4: transformers + ms-swift ---
echo "[Step 4/7] Installing transformers, ms-swift, trl"
pip install "transformers>=5.5.0,<6.0"
pip install "trl>=0.29.0" "peft>=0.18.0" "accelerate>=1.13.0" "datasets>=3.6.0"

# ms-swift 从源码安装
if [ -d "${MS_SWIFT_ROOT:-./ms-swift}" ]; then
    echo "  Installing ms-swift from source: ${MS_SWIFT_ROOT:-./ms-swift}"
    pip install -e "${MS_SWIFT_ROOT:-./ms-swift}"
else
    echo "  ms-swift source not found, cloning..."
    git clone https://github.com/modelscope/ms-swift.git ./ms-swift
    pip install -e ./ms-swift
fi

# Patch ms-swift enable_full_determinism if needed
SWIFT_UTILS="${MS_SWIFT_ROOT:-./ms-swift}/swift/utils/utils.py"
if [ -f "$SWIFT_UTILS" ] && grep -q "from transformers import.*enable_full_determinism" "$SWIFT_UTILS"; then
    echo "  Patching ms-swift enable_full_determinism import..."
    sed -i 's/from transformers import \(.*\), enable_full_determinism, \(.*\)/try:\n    from transformers import \1, enable_full_determinism, \2\nexcept ImportError:\n    enable_full_determinism = None/' "$SWIFT_UTILS" || true
fi

# --- Step 5: flash-attn 2.x + flash-linear-attention + causal-conv1d ---
echo "[Step 5/7] Installing flash-attn 2.x, flash-linear-attention, causal-conv1d"
pip install "flash-attn>=2.7.0,<3.0" --no-build-isolation
pip install causal-conv1d
pip install flash-linear-attention

# --- Step 6: 其他依赖 ---
echo "[Step 6/7] Installing remaining dependencies"
pip install deepspeed wandb qwen_vl_utils decord

# --- Step 7: 验证 ---
echo "[Step 7/7] Verification"
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

echo "--- PyTorch ---"
python -c "import torch; print(f'torch {torch.__version__}, cuda {torch.version.cuda}, gpus {torch.cuda.device_count()}')"

echo "--- transformers ---"
python -c "import transformers; print(f'transformers {transformers.__version__}')"

echo "--- vLLM ---"
python -c "import vllm; print(f'vllm {vllm.__version__}')"

echo "--- flash-linear-attention ---"
python -c "import fla; print('flash-linear-attention OK')" 2>/dev/null || echo "  WARNING: fla import failed"

echo "--- ms-swift ---"
python -c "import swift; print(f'ms-swift {swift.__version__}')" 2>/dev/null || echo "  WARNING: swift import issue"

echo ""
echo "=========================================="
echo " Installation complete!"
echo " Next: source .env && run 5-step smoke test"
echo "=========================================="
