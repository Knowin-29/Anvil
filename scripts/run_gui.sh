#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/rex-embodied"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda run -n rex python -V >/dev/null 2>&1; then
  conda activate rex
else
  conda activate rexomni
fi

cd "${PROJECT_ROOT}"
python -m gui.app

