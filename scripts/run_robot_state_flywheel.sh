#!/bin/bash
# A_robot_state sidecar flywheel runner. Does not modify or invoke the default v3 capability runner.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

LOCAL_NO_PROXY="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_NO_PROXY"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_NO_PROXY"

CONFIG="${CONFIG:-scripts/configs/robot_state_quick.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/output/robot_state}"
RUN_ID="${RUN_ID:-robot_state_$(date +%Y%m%d_%H%M%S)}"
VLM_URL="${VLM_URL:-http://localhost:10301}"
VLM_MODEL="${VLM_MODEL:-qwen2.5-vl-32b-instruct}"
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry) DRY_RUN=true ;;
        --config=*) CONFIG="${arg#*=}" ;;
    esac
done

echo "=============================================="
echo "  A_robot_state sidecar pipeline"
echo "=============================================="
echo "  配置: $CONFIG"
echo "  输出: $OUTPUT_ROOT/$RUN_ID"
echo "  VLM:  $VLM_URL ($VLM_MODEL)"
echo "  日期: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""

echo ">>> 验证 robot state 数据源配置..."
python3 -m apps.data_flywheel.robot_state.pipeline \
    --source-config "$CONFIG" \
    --dry

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo ">>> Dry run 完成，配置验证通过。"
    exit 0
fi

mkdir -p "$OUTPUT_ROOT"

python -m apps.data_flywheel.robot_state.pipeline \
    --source-config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$RUN_ID" \
    --vlm-url "$VLM_URL" \
    --vlm-model "$VLM_MODEL" \
    2>&1 | tee "$OUTPUT_ROOT/${RUN_ID}_console.log"

echo ""
echo "=============================================="
echo "  A_robot_state 运行完成"
echo "=============================================="
echo "  输出目录: $OUTPUT_ROOT/$RUN_ID"
echo "  标注文件: $OUTPUT_ROOT/$RUN_ID/annotations/A_robot_state.jsonl"
echo "  统计文件: $OUTPUT_ROOT/$RUN_ID/robot_state_stats.json"
echo "  控制台日志: $OUTPUT_ROOT/${RUN_ID}_console.log"
