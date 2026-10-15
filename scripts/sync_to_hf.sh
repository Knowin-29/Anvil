#!/bin/bash
# =============================================================================
# 同步飞轮最新数据到 HuggingFace Dataset
# 用法: bash scripts/sync_to_hf.sh
# =============================================================================
set -e

HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN env var must be set}"
HF_REPO="${HF_REPO:?Set the authorized destination dataset repository}"
DATA_DIR="${ANVIL_EXPORT_DATA:?Set the licensed dataset directory to upload}"
PROXY="${ALL_PROXY:-}"

export HF_TOKEN
export ALL_PROXY="$PROXY"

echo "============================================================"
echo " 同步飞轮数据到 HuggingFace"
echo " Repo: https://huggingface.co/datasets/${HF_REPO}"
echo " Source: ${DATA_DIR}"
echo "============================================================"

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# 统计
echo ""
echo "[INFO] 数据统计:"
for sub in annotations metadata images; do
    if [ -d "$DATA_DIR/$sub" ]; then
        count=$(find "$DATA_DIR/$sub" -name "*.jsonl" -exec cat {} + 2>/dev/null | wc -l)
        echo "  $sub: $count lines"
    fi
done
echo ""

# 上传 annotations (训练数据)
echo "[UPLOAD] Uploading annotations..."
hf upload "$HF_REPO" \
    "$DATA_DIR/annotations" "annotations" \
    --repo-type dataset 2>&1 | tail -3

# 上传 metadata
if [ -d "$DATA_DIR/metadata" ]; then
    echo "[UPLOAD] Uploading metadata..."
    hf upload "$HF_REPO" \
        "$DATA_DIR/metadata" "metadata" \
        --repo-type dataset 2>&1 | tail -3
fi

echo ""
echo "============================================================"
echo " DONE! Data synced to:"
echo " https://huggingface.co/datasets/${HF_REPO}"
echo ""
echo " 阿里云下载:"
echo "   huggingface-cli download ${HF_REPO} --repo-type dataset --local-dir ./GSA_volc"
echo "============================================================"
