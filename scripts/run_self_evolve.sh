#!/bin/bash
# Rex-Embodied v3.0 自进化闭环脚本
#
# 每跑完 N 张图:
#   1. 收集本轮 annotations + verifier_scores
#   2. 人工 review 边界样本
#   3. 人工标签合入 verifier_data/vN_train.jsonl
#   4. 触发 ms-swift 增量训练
#   5. 新模型回灌到推理服务
#
# 用法: bash scripts/run_self_evolve.sh [run_dir] [generation]

set -e

# 代理
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80

# 从 .env 读取凭证
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

RUN_DIR="${1:-data/output}"
GENERATION="${2:-0}"
VERIFIER_DATA="verifier_data"
CKPT_DIR="checkpoints"

echo "=== Rex-Embodied Self-Evolution Loop ==="
echo "  Run dir: $RUN_DIR"
echo "  Generation: $GENERATION"
echo ""

# ---- Step 1: 收集 Verifier 训练数据 ----
echo ">>> Step 1: Collecting verifier training data..."
python -c "
from apps.data_flywheel.verifier.collector import VerifierDataCollector
import glob, os

# 找到最新的 run
run_dirs = sorted(glob.glob('${RUN_DIR}/v3_run_*'))
if not run_dirs:
    print('No run directories found')
    exit(1)

latest_run = run_dirs[-1]
print(f'Processing: {latest_run}')

collector = VerifierDataCollector(latest_run, '${VERIFIER_DATA}')
train_path = collector.collect()
dpo_path = collector.collect_dpo_pairs()
print(f'Train data: {train_path}')
print(f'DPO data: {dpo_path}')
"

# ---- Step 2: 等待人工 review ----
echo ""
echo ">>> Step 2: Human review"
echo "  请 review 边界样本: ${VERIFIER_DATA}/v0_train.jsonl"
echo "  标注完成后按 Enter 继续..."
echo "  (跳过 review 直接按 Enter)"
read -r

# ---- Step 3: 增量训练 ----
echo ">>> Step 3: Incremental training (generation ${GENERATION})"

PREV_CKPT="${CKPT_DIR}/verifier_grpo_v${GENERATION}"
NEXT_GEN=$((GENERATION + 1))
NEXT_CKPT="${CKPT_DIR}/verifier_sft_v${NEXT_GEN}"

if [ -d "$PREV_CKPT" ]; then
    echo "  Resuming from: $PREV_CKPT"
    swift sft \
        --model "$PREV_CKPT" \
        --train_type lora \
        --lora_rank 64 \
        --dataset "${VERIFIER_DATA}/v0_sft.jsonl" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --learning_rate 5e-5 \
        --output_dir "$NEXT_CKPT" \
        --report_to wandb \
        --run_name "verifier_sft_v${NEXT_GEN}"
else
    echo "  No previous checkpoint, running full training..."
    bash apps/data_flywheel/verifier/train_v0.sh sft
fi

# ---- Step 4: 回灌到推理服务 ----
echo ""
echo ">>> Step 4: Deploying new verifier model"
echo "  New checkpoint: $NEXT_CKPT"
echo "  Restart verifier service:"
echo "    bash scripts/start_verifier.sh $NEXT_CKPT"

# ---- Step 5: 记录到 W&B ----
echo ""
echo ">>> Step 5: Logging to W&B"
python -c "
try:
    import wandb
    run = wandb.init(project='rex-embodied-flywheel', name='self_evolve_gen_${NEXT_GEN}')
    artifact = wandb.Artifact('verifier-v${NEXT_GEN}', type='model')
    artifact.add_dir('${NEXT_CKPT}')
    run.log_artifact(artifact)
    run.finish()
    print('Logged to W&B')
except Exception as e:
    print(f'W&B logging skipped: {e}')
"

echo ""
echo "=== Self-evolution generation ${NEXT_GEN} complete ==="
echo "  Next: run flywheel with new verifier, then repeat."
