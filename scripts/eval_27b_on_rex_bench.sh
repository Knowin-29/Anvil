#!/bin/bash
# 评测 27B 在 Rex v3 Benchmark 上的表现（用 Verifier 当 judge）
#
# 用法: bash scripts/eval_27b_on_rex_bench.sh [benchmark_dir] [model]

set -e

export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

BENCH_DIR="${1:-benchmarks/rex_v3_bench}"
MODEL="${2:-Qwen/Qwen2.5-VL-27B-Instruct}"
VERIFIER_URL="${3:-http://localhost:8100}"
OUTPUT_DIR="data/output/eval_27b"

echo "=== Rex v3 Benchmark Evaluation ==="
echo "  Benchmark: $BENCH_DIR"
echo "  Model: $MODEL"
echo "  Verifier: $VERIFIER_URL"
echo ""

mkdir -p "$OUTPUT_DIR"

python -c "
import json, os, glob, sys

bench_dir = '${BENCH_DIR}'
output_dir = '${OUTPUT_DIR}'

if not os.path.isdir(bench_dir):
    print(f'Benchmark dir not found: {bench_dir}')
    sys.exit(1)

results = {}
for jsonl_file in sorted(glob.glob(os.path.join(bench_dir, '*.jsonl'))):
    cap = os.path.splitext(os.path.basename(jsonl_file))[0]
    if cap == 'meta':
        continue

    with open(jsonl_file) as f:
        items = [json.loads(l) for l in f if l.strip()]

    results[cap] = {
        'total': len(items),
        'evaluated': 0,
        'pass': 0,
        'fail': 0,
        'avg_score': 0.0,
    }

    # TODO: 实际评测需要:
    # 1. 将 benchmark 样本输入 27B 模型
    # 2. 收集 27B 输出
    # 3. 用 Verifier 评分
    # 当前仅统计 benchmark 规模

    print(f'  {cap}: {len(items)} samples')

# 写结果
output_path = os.path.join(output_dir, 'eval_results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved: {output_path}')
print('NOTE: Full evaluation requires running 27B inference + Verifier scoring.')
print('      This script currently only reports benchmark statistics.')
"

echo ""
echo "=== Evaluation complete ==="
echo "  Results: $OUTPUT_DIR/eval_results.json"
echo "  To run full eval, start Verifier first: bash scripts/start_verifier.sh"
