#!/bin/bash
# ============================================================
# 合并验证通过的 GSA 数据到 Rex-Embodied 主仓库
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$PROJECT_ROOT/data/output/gsa_flywheel_exp"
TARGET_ROOT="/path/to/environment/Rex-Embodied"
TARGET_DIR="$TARGET_ROOT/data/output/flywheel/gsa_exp_may31"

echo "=== GSA Data Merge ==="
echo "  Source: $SOURCE_DIR"
echo "  Target: $TARGET_DIR"
echo ""

# Check source exists
if [ ! -d "$SOURCE_DIR/labels" ]; then
    echo "ERROR: No labels found in $SOURCE_DIR/labels"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR/annotations"
mkdir -p "$TARGET_DIR/verification"
mkdir -p "$TARGET_DIR/reports"

# Copy labels (annotations)
echo "[1/3] Copying labels..."
for dim_dir in "$SOURCE_DIR/labels"/*/; do
    dim_name=$(basename "$dim_dir")
    mkdir -p "$TARGET_DIR/annotations/$dim_name"
    cp -v "$dim_dir"/*.jsonl "$TARGET_DIR/annotations/$dim_name/" 2>/dev/null || true
done

# Copy verification results
echo "[2/3] Copying verification results..."
cp -r "$SOURCE_DIR/verification/"* "$TARGET_DIR/verification/" 2>/dev/null || true

# Generate summary report
echo "[3/3] Generating merge report..."
python3 << 'EOF'
import json, os
from pathlib import Path
from datetime import datetime

source = Path(os.environ.get("SOURCE_DIR", "data/output/gsa_flywheel_exp"))
target = Path(os.environ.get("TARGET_DIR", "/path/to/environment/Rex-Embodied/data/output/flywheel/gsa_exp_may31"))

report = {
    "merge_time": datetime.now().isoformat(),
    "source": str(source),
    "target": str(target),
    "dimensions": {},
    "total_labels": 0,
    "total_images": 0,
}

for jsonl_file in source.rglob("*.jsonl"):
    if "verification" in str(jsonl_file):
        continue
    dim = jsonl_file.stem
    with open(jsonl_file) as f:
        labels = [json.loads(l) for l in f if l.strip()]
    images = set(l.get("image_path", "") for l in labels)
    report["dimensions"][dim] = {
        "labels": len(labels),
        "images": len(images),
    }
    report["total_labels"] += len(labels)
    report["total_images"] += len(images)

report_path = target / "reports" / "merge_report.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"  Total: {report['total_labels']} labels from {report['total_images']} images")
for dim, stats in sorted(report["dimensions"].items()):
    print(f"    {dim}: {stats['labels']} labels, {stats['images']} images")
print(f"  Report: {report_path}")
EOF

echo ""
echo "=== Merge Complete ==="
echo "  Data available at: $TARGET_DIR"
