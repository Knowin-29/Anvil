#!/usr/bin/env bash
# 数据飞轮 Demo — 一键跑通
set -euo pipefail

PROJECT_ROOT="/path/to/rex-embodied"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda run -n rex python -V >/dev/null 2>&1; then
  conda activate rex
else
  echo "[WARN] rex 环境不可用，回退 rexomni"
  conda activate rexomni
fi

cd "${PROJECT_ROOT}"

# 确保有测试图片
DEMO_IMG_DIR="${PROJECT_ROOT}/data/input/images"
mkdir -p "${DEMO_IMG_DIR}"

# 如果目录为空，复制一张示例图
if [ -z "$(ls -A ${DEMO_IMG_DIR} 2>/dev/null)" ]; then
  TEST_IMG="/path/to/Rex-Omni/tutorials/detection_example/test_images/cafe.jpg"
  if [ -f "${TEST_IMG}" ]; then
    cp "${TEST_IMG}" "${DEMO_IMG_DIR}/cafe.jpg"
    echo "[INFO] 已复制示例图片到 ${DEMO_IMG_DIR}"
  else
    echo "[ERROR] 未找到示例图片 ${TEST_IMG}，请手动放置图片到 ${DEMO_IMG_DIR}"
    exit 1
  fi
fi

echo "===== 数据飞轮 Demo 开始 ====="
python -m apps.data_flywheel.pipeline \
  --image-dir "${DEMO_IMG_DIR}" \
  --rounds 1 \
  --max-images 5 \
  --augment 2 \
  --backend transformers

echo "===== 数据飞轮 Demo 完成 ====="
echo "输出目录: ${PROJECT_ROOT}/data/output/flywheel/"
echo "查看报告: cat ${PROJECT_ROOT}/data/output/flywheel/flywheel_summary.json"
