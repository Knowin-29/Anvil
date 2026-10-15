#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Rex-Embodied 每日数据飞轮 — 一键启动脚本
# 
# 用法: bash scripts/run_daily_pipeline.sh [max_annotations]
# 默认: 10000 条标注
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

cd /path/to/environment/Rex-Embodied
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

MAX_ANNS=${1:-10000}
TODAY=$(date +%Y%m%d)
LOG_FILE="data/output/flywheel/daily_${TODAY}/pipeline.log"

echo "═══════════════════════════════════════════════════════════"
echo " Rex-Embodied Daily Pipeline"
echo " Date: ${TODAY}"
echo " Target: ${MAX_ANNS} annotations"
echo "═══════════════════════════════════════════════════════════"

# 检查 VLM 服务
echo "[CHECK] VLM Server..."
if ! curl -sf http://localhost:10301/health > /dev/null 2>&1; then
    echo "[ERROR] VLM server not running at localhost:10301"
    echo "  Start it with: bash services/vlm_server/start.sh"
    exit 1
fi
echo "[OK] VLM Server healthy"

# 创建输出目录
mkdir -p "data/output/flywheel/daily_${TODAY}"

# 运行 pipeline
echo "[RUN] Starting pipeline..."
conda run --no-capture-output -n rex python -u -m apps.data_flywheel.daily_pipeline \
    --max-annotations ${MAX_ANNS} \
    --date ${TODAY} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE! Report at: daily_report_${TODAY}.html"
echo "═══════════════════════════════════════════════════════════"
