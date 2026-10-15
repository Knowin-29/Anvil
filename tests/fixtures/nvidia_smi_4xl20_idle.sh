#!/usr/bin/env bash
set -euo pipefail

case "$*" in
  *--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu*)
    printf '%s\n' \
      '0, GPU-fixture-0, NVIDIA L20, 46068, 0, 0' \
      '1, GPU-fixture-1, NVIDIA L20, 46068, 0, 0' \
      '2, GPU-fixture-2, NVIDIA L20, 46068, 0, 0' \
      '3, GPU-fixture-3, NVIDIA L20, 46068, 0, 0'
    ;;
  *--query-gpu=index,name,memory.total,uuid*)
    printf '%s\n' \
      '0, NVIDIA L20, 46068, GPU-fixture-0' \
      '1, NVIDIA L20, 46068, GPU-fixture-1' \
      '2, NVIDIA L20, 46068, GPU-fixture-2' \
      '3, NVIDIA L20, 46068, GPU-fixture-3'
    ;;
  *--query-compute-apps=*)
    ;;
  *)
    echo "unsupported fixture query: $*" >&2
    exit 2
    ;;
esac
