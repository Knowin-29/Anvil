#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
python trains/index_bridge_images.py --stride "${STRIDE:-3}"
