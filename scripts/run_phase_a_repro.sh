#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/rex-embodied"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda run -n rex python -V >/dev/null 2>&1; then
  conda activate rex
else
  echo "[WARN] 未找到 rex 环境，回退使用 rexomni"
  conda activate rexomni
fi

python "${PROJECT_ROOT}/scripts/phase_a_repro.py"

