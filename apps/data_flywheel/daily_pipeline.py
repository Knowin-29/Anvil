#!/usr/bin/env python3
"""每日数据飞轮 Pipeline — 生成 + 校验 + 报告 + 归档。

用法:
    python -m apps.data_flywheel.daily_pipeline [--max-annotations 10000] [--date 2026-05-19]

流程:
    1. 启动 VLM Worker 生成数据 (到达 max_annotations 后停止)
    2. 运行 Dedup 去重
    3. 运行 Scanner 交叉验证
    4. 生成当日 HTML 报告
    5. 更新 WORK_SUMMARY.md
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("daily_pipeline")

# ─── Constants ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = "/path/to/datasets/datasets/VLM_Dataset_Root/01_Raw_Archive"
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data/output/flywheel")
VLM_URL = os.environ.get("VLM_URL", "http://localhost:10301")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2.5-vl-32b")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")


def get_today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def get_run_dir(date_str: str) -> str:
    return os.path.join(OUTPUT_ROOT, f"daily_{date_str}")


# ─── Phase 1: Generate ───────────────────────────────────────────────────────
def phase_generate(run_dir: str, max_annotations: int = 10000) -> Dict:
    """生成数据直到达到 max_annotations 条。"""
    from PIL import Image
    from apps.data_flywheel.workers.vlm_client import VLMClient
    from apps.data_flywheel.workers.grounding_worker import GroundingWorker
    from apps.data_flywheel.workers.attribute_worker import AttributeWorker
    from apps.data_flywheel.workers.spatial_worker import SpatialWorker
    from apps.data_flywheel.workers.reasoning_worker import ReasoningWorker
    from apps.data_flywheel.workers.planning_worker import PlanningWorker
    from apps.data_flywheel.workers.instruction_worker import InstructionWorker
    from apps.data_flywheel.workers.hallucination_worker import HallucinationWorker
    from apps.data_flywheel.workers.multiturn_worker import MultiturnWorker
    from apps.data_flywheel.workers.affordance_worker import AffordanceWorker
    from apps.data_flywheel.sources.base import Sample

    logger.info(f"Phase 1: GENERATE (target={max_annotations:,} annotations)")
    logger.info(f"  Output: {run_dir}")

    vlm = VLMClient(base_url=VLM_URL, model=VLM_MODEL, api_key=VLM_API_KEY or None)

    # Collect images
    images = glob.glob(DATA_ROOT + "/**/*.png", recursive=True)
    images += glob.glob(DATA_ROOT + "/**/*.jpg", recursive=True)
    logger.info(f"  Found {len(images):,} images")

    # Create output dirs
    ann_dir = os.path.join(run_dir, "annotations")
    caps = ["G", "A", "S", "R", "P", "F", "H", "E", "Aff"]
    for cap in caps:
        os.makedirs(os.path.join(ann_dir, cap), exist_ok=True)

    # Init workers
    workers = {
        "G": GroundingWorker(vlm_client=vlm),
        "A": AttributeWorker(vlm_client=vlm),
        "S": SpatialWorker(vlm_client=vlm),
        "R": ReasoningWorker(vlm_client=vlm),
        "P": PlanningWorker(vlm_client=vlm),
        "F": InstructionWorker(vlm_client=vlm),
        "H": HallucinationWorker(vlm_client=vlm),
        "E": MultiturnWorker(vlm_client=vlm),
        "Aff": AffordanceWorker(vlm_client=vlm),
    }

    # Open output files (line-buffered for real-time writes)
    files = {}
    for cap in caps:
        files[cap] = open(os.path.join(ann_dir, cap, f"{cap}.jsonl"), "w", buffering=1)

    manifest_f = open(os.path.join(run_dir, "manifest.jsonl"), "w", buffering=1)
    total_anns = 0
    total_samples = 0
    per_cap = {c: 0 for c in caps}
    start = time.time()

    for img_path in images:
        if total_anns >= max_annotations:
            logger.info(f"  Reached {max_annotations:,} annotations, stopping.")
            break

        sample_id = f"s_{os.path.basename(img_path).replace('.png','').replace('.jpg','')}"
        total_samples += 1

        try:
            pil_img = Image.open(img_path)
            sample = Sample(
                sample_id=sample_id,
                source_type="image_dir",
                frames=[pil_img],
                metadata={"image_path": img_path},
            )
        except Exception as e:
            logger.debug(f"  Skip {img_path}: {e}")
            continue

        manifest_f.write(json.dumps({"sample_id": sample_id, "image_path": img_path}) + "\n")

        # Grounding first
        try:
            g_anns = workers["G"].process(sample)
        except Exception as e:
            logger.warning(f"  G failed for {sample_id}: {e}")
            g_anns = []

        grounding_data = []
        for ann in g_anns:
            c = ann.content
            grounding_data.append({"name": c.get("name", ""), "bbox": c.get("bbox", [])})
            record = {
                "sample_id": sample_id, "capability_tag": "G", "sub_tag": ann.sub_tag,
                "content": ann.content, "confidence": ann.confidence,
                "difficulty": ann.difficulty, "provenance": ann.provenance,
            }
            files["G"].write(json.dumps(record, ensure_ascii=False) + "\n")
            total_anns += 1
            per_cap["G"] += 1

        # Downstream
        for cap in ["A", "S", "R", "P", "F", "H", "E", "Aff"]:
            if total_anns >= max_annotations:
                break
            try:
                anns = workers[cap].process(sample, grounding_data=grounding_data)
                for ann in anns:
                    record = {
                        "sample_id": sample_id, "capability_tag": cap, "sub_tag": ann.sub_tag,
                        "content": ann.content, "confidence": ann.confidence,
                        "difficulty": ann.difficulty, "provenance": ann.provenance,
                    }
                    files[cap].write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_anns += 1
                    per_cap[cap] += 1
            except Exception as e:
                logger.warning(f"  {cap} failed for {sample_id}: {e}")

        if total_samples % 5 == 0:
            logger.info(f"  Progress: {total_samples} images, {total_anns:,} annotations")
            # Flush all files periodically
            for f in files.values():
                f.flush()
            manifest_f.flush()

    # Cleanup
    for f in files.values():
        f.close()
    manifest_f.close()

    elapsed = time.time() - start
    stats = {
        "total_samples": total_samples,
        "total_annotations": total_anns,
        "per_capability": per_cap,
        "elapsed_seconds": round(elapsed, 1),
        "images_per_minute": round(total_samples / (elapsed / 60), 1) if elapsed > 0 else 0,
    }
    logger.info(f"  DONE: {total_anns:,} annotations from {total_samples} images in {elapsed:.0f}s")
    return stats


# ─── Phase 2: Dedup ──────────────────────────────────────────────────────────
def phase_dedup(run_dir: str) -> Dict:
    """运行去重服务。"""
    from apps.data_flywheel.dedup.dedup_service import dedup_annotations

    logger.info("Phase 2: DEDUP")
    ann_dir = os.path.join(run_dir, "annotations")
    results = dedup_annotations(ann_dir, semantic_threshold=0.85, iou_threshold=0.9, dry_run=False)

    total_removed = sum(s.total_removed for s in results.values())
    total_kept = sum(s.unique_kept for s in results.values())
    logger.info(f"  Removed {total_removed:,} duplicates, kept {total_kept:,}")
    return {"removed": total_removed, "kept": total_kept}


# ─── Phase 3: Scanner Verify ─────────────────────────────────────────────────
def phase_verify(run_dir: str) -> Dict:
    """运行 Scanner 交叉验证 (采样验证，每维度最多 50 条)。"""
    import asyncio
    import random
    from apps.data_flywheel.scanner.judges import VLMJudge
    from apps.data_flywheel.scanner.config import JudgeEndpoint

    logger.info("Phase 3: SCANNER VERIFY")
    ann_dir = os.path.join(run_dir, "annotations")
    endpoint = JudgeEndpoint(name="qwen2.5-vl-32b", base_url=VLM_URL, model=VLM_MODEL)
    judge = VLMJudge(endpoint=endpoint)

    caps = ["G", "A", "S", "R", "P", "F", "H", "E", "Aff"]
    verify_stats = {}

    async def verify_cap(cap: str):
        jsonl_file = os.path.join(ann_dir, cap, f"{cap}.jsonl")
        scores_file = os.path.join(ann_dir, cap, f"{cap}.scores.jsonl")
        if not os.path.exists(jsonl_file):
            return

        annotations = []
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))

        if not annotations:
            return

        # Sample up to 50 per cap
        sample_size = min(50, len(annotations))
        sampled = random.sample(annotations, sample_size)

        scores = []
        with open(scores_file, "w") as sf:
            for ann in sampled:
                try:
                    result = await judge.judge(ann)
                    score = result.score if hasattr(result, 'score') else 0.5
                    score_record = {
                        "sample_id": ann.get("sample_id", ""),
                        "capability_tag": cap,
                        "sub_tag": ann.get("sub_tag", ""),
                        "score": score,
                        "rationale": result.rationale if hasattr(result, 'rationale') else "",
                    }
                    sf.write(json.dumps(score_record, ensure_ascii=False) + "\n")
                    scores.append(score)
                except Exception as e:
                    logger.debug(f"  Verify failed for {cap}: {e}")

        if scores:
            avg = sum(scores) / len(scores)
            pass_rate = sum(1 for s in scores if s >= 0.5) / len(scores) * 100
            verify_stats[cap] = {"avg_score": round(avg, 3), "pass_rate": round(pass_rate, 1), "verified": len(scores)}
            logger.info(f"  [{cap}] avg={avg:.3f} pass={pass_rate:.1f}% (n={len(scores)})")

    async def run_all():
        for cap in caps:
            await verify_cap(cap)
        await judge.close()

    asyncio.run(run_all())
    return verify_stats


# ─── Phase 4: Report ─────────────────────────────────────────────────────────
def phase_report(run_dir: str, date_str: str, gen_stats: Dict, dedup_stats: Dict, verify_stats: Dict) -> str:
    """生成当日 HTML 报告。"""
    logger.info("Phase 4: GENERATE REPORT")

    ann_dir = os.path.join(run_dir, "annotations")
    caps = ["G", "A", "S", "R", "P", "F", "H", "E", "Aff"]

    # Collect per-cap data
    cap_data = {}
    for cap in caps:
        jsonl_file = os.path.join(ann_dir, cap, f"{cap}.jsonl")
        if not os.path.exists(jsonl_file):
            continue
        from collections import Counter
        sub_tags = Counter()
        cases = []
        total = 0
        with open(jsonl_file) as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                ann = json.loads(line)
                sub_tags[ann.get("sub_tag", "")] += 1
                total += 1
                if i < 3:
                    cases.append(ann)
        cap_data[cap] = {"total": total, "sub_tags": dict(sub_tags), "cases": cases}

    total_anns = sum(d["total"] for d in cap_data.values())

    # Build HTML
    cap_rows = ""
    for cap in caps:
        if cap not in cap_data: continue
        d = cap_data[cap]
        st = ", ".join(d["sub_tags"].keys())
        vs = verify_stats.get(cap, {})
        pass_rate = vs.get("pass_rate", "-")
        cap_rows += f'<tr><td class="tag tag-{cap.lower()}">{cap}</td><td>{d["total"]:,}</td><td>{st}</td><td>{pass_rate}%</td></tr>\n'

    cases_html = ""
    for cap in caps:
        if cap not in cap_data: continue
        for c in cap_data[cap]["cases"][:2]:
            cs = json.dumps(c.get("content", {}), indent=2, ensure_ascii=False)[:400]
            cases_html += f'<div class="case-card"><b>{cap}</b> | {c.get("sub_tag","")}<pre>{cs}</pre></div>\n'

    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Daily Report {date_str}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#0f0f1a;color:#e0e0e0;padding:30px;max-width:1200px;margin:0 auto}}
h1{{color:#e94560;font-size:24px;margin-bottom:5px}}
h2{{color:#45b7d1;font-size:18px;margin:25px 0 10px;border-bottom:1px solid #333;padding-bottom:5px}}
.subtitle{{color:#888;font-size:13px;margin-bottom:25px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:15px 0}}
.card{{background:#1a1a2e;border-radius:10px;padding:16px;border:1px solid #333}}
.card .num{{font-size:24px;font-weight:bold;color:#e94560}}
.card .label{{font-size:11px;color:#888;margin-top:4px}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}
th{{background:#16213e;padding:8px 10px;text-align:left;font-size:11px;color:#aaa}}
td{{padding:8px 10px;border-bottom:1px solid #222;font-size:12px}}
.tag{{padding:2px 6px;border-radius:8px;font-size:10px}}
.tag-g{{background:#FF6B6B33;color:#FF6B6B}}.tag-a{{background:#4ECDC433;color:#4ECDC4}}
.tag-s{{background:#45B7D133;color:#45B7D1}}.tag-r{{background:#96CEB433;color:#96CEB4}}
.tag-p{{background:#FFEAA733;color:#FFEAA7}}.tag-f{{background:#DDA0DD33;color:#DDA0DD}}
.tag-h{{background:#98D8C833;color:#98D8C8}}.tag-e{{background:#F7DC6F33;color:#F7DC6F}}
.tag-aff{{background:#BB8FCE33;color:#BB8FCE}}
.case-card{{background:#16213e;border-radius:6px;padding:10px;margin:6px 0;border-left:3px solid #e94560}}
.case-card pre{{font-size:10px;color:#7fdbca;white-space:pre-wrap;max-height:150px;overflow-y:auto;margin-top:5px}}
.footer{{margin-top:30px;border-top:1px solid #333;padding-top:15px;color:#555;font-size:11px}}
</style></head><body>
<h1>Daily Report: {date_str}</h1>
<p class="subtitle">Rex-Embodied Data Flywheel | Auto-generated</p>

<h2>Summary</h2>
<div class="grid">
<div class="card"><div class="num">{total_anns:,}</div><div class="label">Annotations</div></div>
<div class="card"><div class="num">{gen_stats.get("total_samples",0):,}</div><div class="label">Images</div></div>
<div class="card"><div class="num">{dedup_stats.get("removed",0):,}</div><div class="label">Dedup Removed</div></div>
<div class="card"><div class="num">{gen_stats.get("elapsed_seconds",0):.0f}s</div><div class="label">Gen Time</div></div>
</div>

<h2>Per-Dimension</h2>
<table><tr><th>Cap</th><th>Count</th><th>Sub-tags</th><th>Pass Rate</th></tr>
{cap_rows}</table>

<h2>Sample Cases</h2>
{cases_html}

<div class="footer">
<p>Pipeline: generate({gen_stats.get("total_annotations",0):,}) -> dedup(-{dedup_stats.get("removed",0)}) -> verify -> report</p>
</div>
</body></html>'''

    report_path = os.path.join(run_dir, f"report_{date_str}.html")
    with open(report_path, "w") as f:
        f.write(html)

    # Also copy to project root for easy access
    root_report = os.path.join(PROJECT_ROOT, f"daily_report_{date_str}.html")
    with open(root_report, "w") as f:
        f.write(html)

    logger.info(f"  Report: {report_path}")
    logger.info(f"  Also at: {root_report}")
    return report_path


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Daily Data Flywheel Pipeline")
    parser.add_argument("--max-annotations", type=int, default=10000, help="Stop after N annotations")
    parser.add_argument("--date", type=str, default=None, help="Override date (YYYYMMDD)")
    parser.add_argument("--skip-generate", action="store_true", help="Skip generation phase")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification phase")
    args = parser.parse_args()

    date_str = args.date or get_today_str()
    run_dir = get_run_dir(date_str)
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"DAILY PIPELINE: {date_str}")
    logger.info(f"  Run dir: {run_dir}")
    logger.info(f"  Max annotations: {args.max_annotations:,}")
    logger.info(f"{'='*60}")

    # Phase 1: Generate
    if not args.skip_generate:
        gen_stats = phase_generate(run_dir, max_annotations=args.max_annotations)
    else:
        gen_stats = {"total_samples": 0, "total_annotations": 0, "per_capability": {}, "elapsed_seconds": 0}

    # Phase 2: Dedup
    dedup_stats = phase_dedup(run_dir)

    # Phase 3: Verify
    if not args.skip_verify:
        verify_stats = phase_verify(run_dir)
    else:
        verify_stats = {}

    # Phase 4: Report
    report_path = phase_report(run_dir, date_str, gen_stats, dedup_stats, verify_stats)

    # Save pipeline metadata
    meta = {
        "date": date_str,
        "run_dir": run_dir,
        "gen_stats": gen_stats,
        "dedup_stats": dedup_stats,
        "verify_stats": verify_stats,
        "report_path": report_path,
        "completed_at": datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "pipeline_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE: {date_str}")
    logger.info(f"  Annotations: {gen_stats.get('total_annotations', 0):,}")
    logger.info(f"  After dedup: {dedup_stats.get('kept', 0):,}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
