#!/usr/bin/env bash
# ============================================================================
# Rex-Embodied 数据飞轮 — 表格场景数据批量处理脚本
# 使用本地 Rex-Omni 3B 模型，不依赖 API
# ============================================================================
set -euo pipefail

# ---- 参数（可通过环境变量覆盖）----
VERSION="${1:?用法: $0 <version> 例如 v9}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-/mnt/data/jiangqixiong/table_data/data_${VERSION}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/output_table_${VERSION}}"
MODEL_PATH="${MODEL_PATH:-IDEA-Research/Rex-Omni}"
BACKEND="${BACKEND:-transformers}"
ROUNDS="${ROUNDS:-5}"
MAX_IMAGES="${MAX_IMAGES:-99999}"      # 默认取全部
AUGMENT="${AUGMENT:-0}"               # 0 = 不增强
QA_LANG="${QA_LANG:-en}"
VERIFIER_POLICY="${VERIFIER_POLICY:-drop}"

# ---- 校验 ----
if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] 输入目录不存在: ${INPUT_DIR}"
    exit 1
fi

IMG_COUNT=$(ls "${INPUT_DIR}"/*.png "${INPUT_DIR}"/*.jpg "${INPUT_DIR}"/*.jpeg 2>/dev/null | wc -l)
echo "============================================"
echo "Rex-Embodied 数据飞轮"
echo "  版本:     ${VERSION}"
echo "  输入目录: ${INPUT_DIR}  (${IMG_COUNT} 张图片)"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  模型:     ${MODEL_PATH}"
echo "  后端:     ${BACKEND}"
echo "  轮数:     ${ROUNDS}"
echo "  增强:     ${AUGMENT}/张"
echo "  QA语言:   ${QA_LANG}"
echo "  Verifier: ${VERIFIER_POLICY}"
echo "============================================"

cd "${PROJECT_ROOT}"

python -m apps.data_flywheel.pipeline \
    --image-dir "${INPUT_DIR}" \
    --rounds "${ROUNDS}" \
    --max-images "${MAX_IMAGES}" \
    --augment "${AUGMENT}" \
    --model-path "${MODEL_PATH}" \
    --backend "${BACKEND}" \
    --detector-verifier-policy "${VERIFIER_POLICY}" \
    --qa-lang "${QA_LANG}" \
    --output-root "${OUTPUT_DIR}"

echo "[DONE] ${VERSION} 飞轮完成，输出: ${OUTPUT_DIR}"
