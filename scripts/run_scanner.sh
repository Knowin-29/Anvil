#!/bin/bash
# Quality Scanner 启动脚本
# 持续扫描已生成数据 + 多模型质量评估
#
# 用法:
#   ./scripts/run_scanner.sh                    # 使用默认配置持续扫描
#   ./scripts/run_scanner.sh --once             # 只扫一轮
#   ./scripts/run_scanner.sh --capabilities G S Aff  # 只扫指定维度
#
# 前置条件: VLM 服务已运行在 http://localhost:10301

set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-rex}"
CONFIG="scripts/configs/scanner.yaml"
LOG_DIR="data/output/flywheel/scanner_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/scanner_${TIMESTAMP}.log"

echo "=========================================="
echo "  Rex-Embodied Quality Scanner"
echo "=========================================="
echo "Config: $CONFIG"
echo "Log:    $LOG_FILE"
echo ""

# 检查 VLM 服务是否可用
VLM_URL=$(grep -oP 'base_url:\s*"\K[^"]+' "$CONFIG" | head -1)
VLM_URL="${VLM_URL:-http://localhost:10301}"
echo "Checking VLM service at $VLM_URL ..."
if curl -sf "${VLM_URL}/health" > /dev/null 2>&1; then
    echo "  VLM service is UP"
else
    echo "  WARNING: VLM service at $VLM_URL not reachable!"
    echo "  Scanner will report errors for VLM calls."
    echo "  Start VLM server first: python services/vlm_server/server.py --model <path>"
fi
echo ""

# 激活 conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

echo "Starting scanner (PID will be logged)..."
echo "Press Ctrl+C to stop gracefully."
echo ""

python -m apps.data_flywheel.scanner.scanner \
    --config "$CONFIG" \
    "$@" \
    2>&1 | tee "$LOG_FILE"
