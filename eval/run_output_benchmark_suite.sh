#!/bin/bash
# One-click benchmark suite for checkpoints under ${GSA_FORGE_ROOT}/output.
set -euo pipefail

ROOT="${GSA_FORGE_ROOT}"
cd "${ROOT}"

python3 eval/run_output_benchmark_suite.py "$@"
