#!/bin/bash
# 启动 Verifier 推理服务
# 用法: bash scripts/start_verifier.sh [model_path]

set -e

# 代理
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80

# 从 .env 读取凭证
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

MODEL="${1:-checkpoints/verifier_grpo_v0}"
BACKEND="${2:-vllm}"
PORT="${3:-8100}"

echo "Starting Verifier service..."
echo "  Model: $MODEL"
echo "  Backend: $BACKEND"
echo "  Port: $PORT"

python -m apps.data_flywheel.verifier.server \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --port "$PORT"
