#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/rex-embodied"
REX_OMNI_REPO="${REX_OMNI_REPO:-/path/to/Rex-Omni}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda 未安装或未在 PATH 中。"
  exit 1
fi

if conda run -n rex python -V >/dev/null 2>&1; then
  echo "[INFO] 更新已有环境: rex"
  conda env update -n rex -f "${PROJECT_ROOT}/env/environment.yml" --prune
elif conda run -n rexomni python -V >/dev/null 2>&1; then
  echo "[INFO] 通过 clone rexomni 创建环境: rex"
  conda create -n rex --clone rexomni -y
  conda env update -n rex -f "${PROJECT_ROOT}/env/environment.yml" --prune
else
  echo "[INFO] 创建新环境: rex"
  conda env create -f "${PROJECT_ROOT}/env/environment.yml"
fi

echo "[INFO] 安装 Rex-Omni editable 包: ${REX_OMNI_REPO}"
bash -lc "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate rex && pip install -v -e \"${REX_OMNI_REPO}\""

echo "[INFO] 安装本工程依赖"
bash -lc "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate rex && pip install -r \"${PROJECT_ROOT}/requirements.txt\""

echo "[INFO] 检查 spaCy 英文模型"
bash -lc "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate rex && python -m spacy download en_core_web_sm"

echo "[OK] 环境 rex 就绪"
