#!/usr/bin/env bash
# GSA-Forge Volcengine smoke test. Verifies the container has:
#   1. the vePFS /path/to/workspace/foundation mount (models + data live here)
#   2. the expected GPU count
#   3. a working ms-swift + torch install
# Everything is written to the mount so it is readable from the dev instance.
set -uo pipefail

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="/path/to/Anvil/logs/volc_smoke"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/smoke_${STAMP}.log"

{
  echo "=== GSA-Forge volc smoke @ ${STAMP} (UTC) ==="
  echo "[host] $(hostname)"

  echo "--- [1] vePFS mount ---"
  if [[ -d /path/to/workspace/foundation ]]; then
    echo "OK: /path/to/workspace/foundation present"
    df -h /path/to/workspace/foundation | tail -1
    touch "/path/to/Anvil/logs/volc_smoke/.write_probe_${STAMP}" \
      && echo "OK: mount is writable" \
      && rm -f "/path/to/Anvil/logs/volc_smoke/.write_probe_${STAMP}"
  else
    echo "FAIL: /path/to/workspace/foundation missing (Storages block not applied)"
  fi

  echo "--- [2] GPUs ---"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo "gpu_count=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null)"
  else
    echo "FAIL: nvidia-smi not found"
  fi

  echo "--- [3] python / torch / ms-swift ---"
  python -c 'import sys; print("python", sys.version.split()[0])' 2>&1
  python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ndev", torch.cuda.device_count())' 2>&1
  python -c 'import swift; print("ms-swift", getattr(swift, "__version__", "?"))' 2>&1
  echo "--- [4] models present ---"
  for m in Qwen3.5-9B Qwen3.6-27B; do
    p="/path/to/environment/hf_models/Qwen/${m}"
    [[ -d "$p" ]] && echo "OK: ${m} ($(du -sh "$p" 2>/dev/null | cut -f1))" || echo "MISS: ${m}"
  done
  echo "=== smoke complete @ $(date -u +%Y%m%d_%H%M%S) ==="
} 2>&1 | tee "${OUT}"

echo "smoke log written to ${OUT}"
