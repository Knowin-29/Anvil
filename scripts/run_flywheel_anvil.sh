#!/bin/bash
# Rex-Embodied v3.0 一键式数据飞轮运行脚本
#
# 功能:
#   1. 从多场景数据源加载图片
#   2. 运行 9 维度能力标注
#   3. 输出按场景+能力分组的 JSONL
#   4. 启动 GUI 监控服务
#   5. 生成带日期的运行日志
#
# 用法:
#   bash scripts/run_flywheel_anvil.sh          # 默认配置
#   bash scripts/run_flywheel_anvil.sh --gui    # 同时启动 GUI
#   bash scripts/run_flywheel_anvil.sh --dry    # 仅验证配置不运行

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 代理
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80

# 从 .env 读取环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 本地 VLM 服务必须绕过代理，否则 localhost 请求会被 ALL_PROXY 劫持。
LOCAL_NO_PROXY="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_NO_PROXY"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_NO_PROXY"

# ---- 参数 ----
CONFIG="${CONFIG:-scripts/configs/flywheel_anvil.yaml}"
CAPABILITIES="${CAPABILITIES:-G,S,A,R,P,F,H,E,Aff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/output/flywheel}"
RUN_ID="${RUN_ID:-v3_run_$(date +%Y%m%d_%H%M%S)}"
VLM_URL="${VLM_URL:-http://localhost:10301}"
VLM_MODEL="${VLM_MODEL:-qwen2.5-vl-32b-instruct}"
START_GUI=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --gui) START_GUI=true ;;
        --dry) DRY_RUN=true ;;
        --config=*) CONFIG="${arg#*=}" ;;
        --capabilities=*) CAPABILITIES="${arg#*=}" ;;
    esac
done

echo "=============================================="
echo "  Rex-Embodied v3.0 数据飞轮"
echo "=============================================="
echo "  配置: $CONFIG"
echo "  能力: $CAPABILITIES"
echo "  输出: $OUTPUT_ROOT/$RUN_ID"
echo "  VLM:  $VLM_URL ($VLM_MODEL)"
echo "  GUI:  $START_GUI"
echo "  日期: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""

# ---- 验证配置 ----
echo ">>> 验证数据源配置..."
python3 -c "
import yaml, os, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

sources = cfg.get('sources', [])
total_found = 0
for i, src in enumerate(sources):
    path = src.get('path', '')
    exists = os.path.isdir(path)
    status = '✓' if exists else '✗'
    print(f'  {status} [{i+1}] {path}')
    if exists:
        total_found += 1

print(f'\n  可用数据源: {total_found}/{len(sources)}')
if total_found == 0:
    print('  ERROR: 无可用数据源!')
    sys.exit(1)
"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo ">>> Dry run 完成，配置验证通过。"
    exit 0
fi

# ---- 启动 GUI (后台) ----
if [ "$START_GUI" = true ]; then
    echo ""
    echo ">>> 启动 GUI 监控服务 (后台, port 7862)..."
    python -m gui.app &
    GUI_PID=$!
    echo "  GUI PID: $GUI_PID"
    echo "  访问: http://localhost:7862"
    sleep 2
fi

# ---- 运行飞轮 ----
echo ""
echo ">>> 启动数据飞轮..."
echo "  Run ID: $RUN_ID"
echo ""

mkdir -p "$OUTPUT_ROOT"

python -m apps.data_flywheel.pipeline \
    --v3 \
    --source-config "$CONFIG" \
    --capabilities "$CAPABILITIES" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$RUN_ID" \
    --vlm-url "$VLM_URL" \
    --vlm-model "$VLM_MODEL" \
    --no-ray \
    2>&1 | tee "$OUTPUT_ROOT/${RUN_ID}_console.log"

echo ""
echo "=============================================="
echo "  飞轮运行完成"
echo "=============================================="
echo "  输出目录: $OUTPUT_ROOT/$RUN_ID"
echo "  日志文件: $OUTPUT_ROOT/$RUN_ID/logs/"
echo "  控制台日志: $OUTPUT_ROOT/${RUN_ID}_console.log"
echo ""

# ---- 运行能力审计 ----
echo ">>> 运行能力覆盖审计..."
python scripts/capability_audit.py --run-dir "$OUTPUT_ROOT/$RUN_ID" 2>/dev/null || true

echo ""
echo ">>> 完成! 可通过 GUI 查看详细结果。"

if [ "$START_GUI" = true ]; then
    echo "  GUI 仍在运行 (PID: $GUI_PID)，按 Ctrl+C 停止。"
    wait $GUI_PID
fi
