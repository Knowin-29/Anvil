#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
python trains/build_clean_registry.py --verify-images
