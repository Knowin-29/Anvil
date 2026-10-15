"""GSA v0 Production Run — Generate labels, evaluate, and produce reports.

Usage (本地 vLLM 模式, 推荐):
    # 确保本地 vLLM 已启动 (默认 localhost:10301, qwen2.5-vl-32b)
    PYTHONPATH=. python scripts/run_gsa_production.py

Usage (远程 API 模式):
    export VLM_URL="https://your-vlm-endpoint/v1"
    export VLM_API_KEY="your-api-key"
    export VLM_MODEL="MLLM"
    PYTHONPATH=. python scripts/run_gsa_production.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.data_flywheel.workers.vlm_client import VLMClient
from apps.data_flywheel.gsa.grounding_worker import GSAGroundingWorker
from apps.data_flywheel.gsa.spatial_worker import GSASpatialWorker
from apps.data_flywheel.gsa.a_identity_attr_pipeline import AIdentityAttrPipeline
from apps.data_flywheel.gsa.a_obj_state_pipeline import AObjStatePipeline
from apps.data_flywheel.gsa.a_robot_state_pipeline import ARobotStatePipeline
from apps.data_flywheel.gsa.a_action_precondition_pipeline import AActionPreconditionPipeline
from apps.data_flywheel.gsa.a_action_interface_pipeline import AActionInterfacePipeline
from apps.data_flywheel.gsa.visual_evidence import GsaEvidenceBundle, assign_object_ids
from apps.data_flywheel.gsa.eval_harness import GSAEvalHarness
from apps.data_flywheel.gsa.gap_report import GSAGapReporter
from apps.data_flywheel.gsa.manifest import GSAManifest
from apps.data_flywheel.gsa.taxonomy import Bucket
from apps.data_flywheel.gsa.data_adapters.bridge_adapter import is_wrist_camera_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("gsa.production")

TODAY = datetime.now().strftime("%Y%m%d")
RUN_ID = f"gsa_v0_{TODAY}_{int(time.time()) % 100000}"


def discover_images() -> list:
    """Find all available images from multiple diverse sources."""
    # If a curated image list file exists, use it directly
    curated_file = os.environ.get("GSA_IMAGE_LIST", "data/input/high_quality_50.txt")
    if os.path.exists(curated_file):
        with open(curated_file) as f:
            images = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]
        if images:
            logger.info(f"Using curated image list: {curated_file} ({len(images)} images)")
            return images

    patterns = [
        # Local test fixtures
        "tests/fixtures/images/*.png",
        "tests/fixtures/images/*.jpg",
        "tests/fixtures/*.jpg",
        # BridgeDataV2 — robot tabletop manipulation (25k+ images)
        "/path/to/environment/backup_420/bridgedatav2/data/images/*.png",
        # Benchmark eval — diverse indoor scenes
        "/path/to/datasets/datasets/benchmark_eval_all/test_little_data/*.jpg",
        "/path/to/datasets/datasets/benchmark_eval_all/test_little_data/*.png",
        # Washing machine / appliance (limited)
        "/path/to/datasets/datasets/VLM_Dataset_Root/01_Raw_Benchmark/**/*.png",
        # appliance_datasets (appliance panels)
        "/path/to/datasets/datasets/appliance_datasets/**/*.png",
    ]
    images = []
    for p in patterns:
        images.extend(glob(p, recursive=True))

    # NAS data sources (A-dimension relevant)
    nas_root = Path("/path/to/datasets")
    if nas_root.exists():
        nas_dirs = {
            "013_done_P1_A1_mit-states": "mit_states",
            "019_done_P1_A1_visual-genome": "visual_genome",
        }
        for dirname, src_name in nas_dirs.items():
            ds_path = nas_root / dirname
            if ds_path.exists():
                for ext in ("*.jpg", "*.png", "*.jpeg"):
                    found = list(ds_path.rglob(ext))[:100]
                    images.extend(str(f) for f in found)
                    if found:
                        logger.info(f"NAS [{src_name}]: {len(found)} images")

    images = sorted(set(images))
    logger.info(f"Discovered {len(images)} total images from diverse sources")
    return images


def run_production():
    # VLM 配置: 优先本地 vLLM (localhost:10301), 无需 API Key
    # 如果设置了 VLM_URL 环境变量则使用远程 API 模式
    base_url = os.environ.get("VLM_URL", "http://localhost:10301")
    model = os.environ.get("VLM_MODEL", "Qwen2.5-VL-32B-Instruct")
    api_key = os.environ.get("VLM_API_KEY", "")

    # 检测本地 vLLM 是否可用
    import urllib.request
    try:
        urllib.request.urlopen(f"{base_url}/v1/models", timeout=3)
        logger.info(f"Using local vLLM: {base_url} model={model}")
    except Exception:
        # 本地不可用，检查是否有远程 API 配置
        if api_key and "localhost" in base_url:
            # 有 key 但指向本地 -> 本地挂了，报错
            logger.warning(f"Local vLLM at {base_url} not reachable, but VLM_API_KEY is set. "
                          "Set VLM_URL to remote endpoint if you want API mode.")
        elif not api_key and "localhost" in base_url:
            logger.error(f"Local vLLM at {base_url} not reachable. "
                        "Start vLLM first or set VLM_URL + VLM_API_KEY for remote API mode.")
        else:
            logger.info(f"Using remote API: {base_url} model={model}")

    vlm = VLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key or None,
        max_image_size=960,
        timeout=120,
        max_retries=3,
    )

    images = discover_images()
    max_images = int(os.environ.get("GSA_MAX_IMAGES", "10"))
    # Stratified sampling for diversity: group by source, sample from each
    import random
    random.seed(42)
    by_source = {}
    for img in images:
        if "bridgedatav2" in img:
            src = "bridge"
        elif "benchmark_eval" in img or "test_little" in img:
            src = "benchmark"
        elif "appliance_datasets" in img or "washing" in img:
            src = "appliance"
        elif "VLM_Dataset_Root" in img:
            src = "vlm_benchmark"
        else:
            src = "test_fixtures"
        by_source.setdefault(src, []).append(img)

    # Allocate proportionally but ensure diversity
    selected = []
    per_source = max(3, max_images // max(len(by_source), 1))
    for src, src_images in by_source.items():
        random.shuffle(src_images)
        selected.extend(src_images[:per_source])

    random.shuffle(selected)
    images = selected[:max_images]
    logger.info(f"Selected {len(images)} images: " +
                ", ".join(f"{k}={min(per_source, len(v))}" for k, v in by_source.items()))

    # Output dirs
    out_root = Path(f"data/output/gsa_v0/{RUN_ID}")
    (out_root / "labels" / "G").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "S").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "A_state").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "A_action").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "A_identity_attr").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "A_obj_state").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "A_robot_state").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "A_action_precondition").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "A_action_interface").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "pick_place").mkdir(parents=True, exist_ok=True)
    (out_root / "review_queue").mkdir(parents=True, exist_ok=True)
    (out_root / "eval").mkdir(parents=True, exist_ok=True)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    (out_root / "packs").mkdir(parents=True, exist_ok=True)

    # Workers
    g_worker = GSAGroundingWorker(vlm)
    s_worker = GSASpatialWorker(vlm)
    # Fine-grained A pipelines
    a_identity_attr = AIdentityAttrPipeline(vlm)
    a_obj_state = AObjStatePipeline(vlm)
    a_robot_state = ARobotStatePipeline(vlm)
    a_precondition = AActionPreconditionPipeline(vlm)
    a_interface = AActionInterfacePipeline(vlm)

    all_labels = []
    all_reviews = []
    all_pick_place = []
    stats = {"images": 0, "g": 0, "s": 0, "a_identity_attr": 0, "a_obj_state": 0,
             "a_robot_state": 0, "a_precondition": 0, "a_interface": 0, "reviews": 0, "pick_place": 0}

    for idx, img_path in enumerate(images):
        sample_id = f"{RUN_ID}_{idx:04d}"
        scene_family = _classify_scene(img_path)
        scenario_id = "SC.2"  # default tabletop

        logger.info(f"[{idx+1}/{len(images)}] Processing {os.path.basename(img_path)} ...")
        t0 = time.time()

        try:
            # G: Grounding
            g_labels = g_worker.generate(img_path, sample_id, scene_family, scenario_id)
            # Add image_path to all labels for visualization
            for l in g_labels:
                l["image_path"] = img_path
            all_labels.extend(g_labels)
            stats["g"] += len(g_labels)

            # Extract objects
            objs_raw = [l["answer"] for l in g_labels
                        if l.get("subtype") == "simple_grounding"
                        and isinstance(l.get("answer"), dict)]
            obj_data = [{"object_name": o.get("object_name", "x"), "bbox": o.get("bbox")}
                        for o in objs_raw if o.get("bbox")]
            obj_refs = assign_object_ids(obj_data)

            if obj_refs:
                # S: Spatial
                s_labels = s_worker.generate(img_path, sample_id, obj_refs, scene_family, scenario_id)
                for l in s_labels:
                    l["image_path"] = img_path
                all_labels.extend(s_labels)
                stats["s"] += len(s_labels)

                # Build evidence bundle
                bundle = GsaEvidenceBundle(
                    sample_id=sample_id, image_path=img_path,
                    scene_family=scene_family, scenario_id=scenario_id,
                    objects=obj_refs,
                    spatial_facts=s_worker.extract_spatial_facts(s_labels),
                )

                # A_identity_attr (no VLM call, extracts from G)
                id_labels = a_identity_attr.run(bundle)
                for lbl in id_labels:
                    all_labels.append(lbl.to_dict())
                stats["a_identity_attr"] += len(id_labels)

                # A pipelines: run obj_state + precondition + interface concurrently
                from concurrent.futures import ThreadPoolExecutor, as_completed
                a_futures = {}
                with ThreadPoolExecutor(max_workers=3) as pool:
                    a_futures["obj_state"] = pool.submit(a_obj_state.run, bundle)
                    a_futures["precondition"] = pool.submit(a_precondition.run, bundle)
                    a_futures["interface"] = pool.submit(a_interface.run, bundle)

                    # A_robot_state (only for wrist camera images)
                    if is_wrist_camera_image(img_path):
                        a_futures["robot_state"] = pool.submit(a_robot_state.run, bundle)

                for name, future in a_futures.items():
                    try:
                        labels_result = future.result(timeout=300)
                        for lbl in labels_result:
                            all_labels.append(lbl.to_dict())
                        stats[f"a_{name}"] = stats.get(f"a_{name}", 0) + len(labels_result)
                    except Exception as e:
                        logger.warning(f"A pipeline {name} failed: {e}")

                # Pick-Place (unified pipeline — application-aligned)
                from apps.data_flywheel.gsa.pick_place_pipeline import PickPlacePipeline
                pp_pipe = PickPlacePipeline(vlm)
                pp_labels = pp_pipe.run(bundle)
                for ppl in pp_labels:
                    all_pick_place.append(ppl.to_full_dict())
                stats["pick_place"] += len(pp_labels)

            stats["images"] += 1
            elapsed = time.time() - t0
            a_total = stats["a_identity_attr"] + stats["a_obj_state"] + stats["a_precondition"] + stats["a_interface"]
            logger.info(f"  Done in {elapsed:.1f}s: G={len(g_labels)}, objs={len(obj_refs)}, "
                        f"S={len(s_labels) if obj_refs else 0}, A={a_total}")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

    # Save labels by bucket
    by_bucket = {}
    for label in all_labels:
        b = label.get("bucket", "unknown")
        by_bucket.setdefault(b, []).append(label)

    for bucket, labels in by_bucket.items():
        path = out_root / "labels" / bucket / f"{bucket}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for l in labels:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")

    # Save review queue
    with open(out_root / "review_queue" / "A_action.review.jsonl", "w") as f:
        for r in all_reviews:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save pick-place labels (application-aligned format)
    with open(out_root / "labels" / "pick_place" / "pick_place.jsonl", "w") as f:
        for pp in all_pick_place:
            f.write(json.dumps(pp, ensure_ascii=False) + "\n")
    logger.info(f"Pick-Place: {len(all_pick_place)} labels saved")

    # VLM-as-Judge pass: verify a sample of G labels
    vlm_judge_enabled = os.environ.get("GSA_VLM_JUDGE", "1") == "1"
    if vlm_judge_enabled and len(all_labels) > 0:
        from apps.data_flywheel.gsa.vlm_judge import VLMJudge
        judge = VLMJudge(vlm)
        judge_results = []
        g_labels_with_img = [l for l in all_labels
                             if l.get("bucket") == "G" and l.get("image_path")
                             and isinstance(l.get("answer"), dict) and l["answer"].get("bbox")]
        # Judge up to 10 random G labels
        import random
        sample_for_judge = random.sample(g_labels_with_img, min(10, len(g_labels_with_img)))
        rejected = 0
        for label in sample_for_judge:
            from PIL import Image as _PILImg
            try:
                _img = _PILImg.open(label["image_path"])
                _iw, _ih = _img.size
                _img.close()
            except Exception:
                continue
            correct, conf, reason = judge.verify_bbox(
                label["image_path"],
                label["answer"]["object_name"],
                label["answer"]["bbox"],
                _iw, _ih,
            )
            judge_results.append({
                "label_id": f"{label['sample_id']}:{label.get('object_id','')}",
                "correct": correct, "confidence": conf, "reason": reason,
            })
            if not correct:
                rejected += 1
                label["verifier_status"] = "review_required"
                label["judge_reason"] = reason

        logger.info(f"VLM Judge: {len(sample_for_judge)} checked, {rejected} rejected")
        with open(out_root / "eval" / "vlm_judge_results.json", "w") as f:
            f.write(json.dumps(judge_results, indent=2, ensure_ascii=False))

    # Eval
    harness = GSAEvalHarness()
    eval_results = harness.evaluate(all_labels)
    with open(out_root / "eval" / "gsa_eval_results.json", "w") as f:
        f.write(eval_results.to_json())

    # Verifier (aligned with Viewer system)
    from apps.data_flywheel.gsa.verifier import GSAVerifier
    verifier = GSAVerifier()
    verify_report = verifier.verify_batch(all_labels)
    with open(out_root / "eval" / "gsa_verify_report.json", "w") as f:
        import json as _json
        f.write(_json.dumps(verify_report.to_dict(), indent=2, ensure_ascii=False))
    logger.info(f"Verifier: {verify_report.passed}/{verify_report.total} passed "
                f"(score={verify_report.avg_score:.3f})")

    # Gap report
    reporter = GSAGapReporter(target_per_subtype=20)
    gap_report = reporter.generate(all_labels, eval_results)
    with open(out_root / "reports" / "gsa_gap_report_v0.json", "w") as f:
        f.write(gap_report.to_json())
    with open(out_root / "reports" / "gsa_gap_report_v0.html", "w") as f:
        f.write(gap_report.to_html())

    # Manifest
    manifest = GSAManifest()
    manifest.add_from_labels(all_labels)
    manifest.save(str(out_root / "dataset_manifest_v0.jsonl"))

    # Training packs
    pack_names = {
        "G": "G_core_pack_manifest.jsonl",
        "S": "S_core_pack_manifest.jsonl",
        "A_state": "A_state_pack_manifest.jsonl",
        "A_action": "A_action_pack_manifest.jsonl",
    }
    for bucket, filename in pack_names.items():
        stable = [l for l in all_labels
                  if l.get("bucket") == bucket
                  and l.get("verifier_status") == "auto_verifiable"]
        with open(out_root / "packs" / filename, "w") as f:
            for l in stable:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")

    # Generate HTML visualization report with images + annotations
    from apps.data_flywheel.gsa.visualizer import generate_visual_report
    report_path = str(out_root / "reports" / f"gsa_visual_report_{TODAY}.html")
    generate_visual_report(
        labels=all_labels,
        output_path=report_path,
        run_id=RUN_ID,
    )

    # Summary
    logger.info("=" * 60)
    logger.info(f"GSA v0 Production Run Complete: {RUN_ID}")
    logger.info(f"  Images: {stats['images']}/{len(images)}")
    logger.info(f"  G labels: {stats['g']}")
    logger.info(f"  S labels: {stats['s']}")
    logger.info(f"  A_identity_attr: {stats.get('a_identity_attr', 0)}")
    logger.info(f"  A_obj_state: {stats.get('a_obj_state', 0)}")
    logger.info(f"  A_robot_state: {stats.get('a_robot_state', 0)}")
    logger.info(f"  A_precondition: {stats.get('a_precondition', 0)}")
    logger.info(f"  A_interface: {stats.get('a_interface', 0)}")
    logger.info(f"  Pick-Place labels: {stats['pick_place']}")
    logger.info(f"  Reviews: {stats['reviews']}")
    logger.info(f"  Total: {len(all_labels)} labels + {len(all_pick_place)} pick-place")
    logger.info(f"  Verifiable rate: {eval_results.overall_verifiable_rate:.0%}")
    logger.info(f"  Output: {out_root}")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 60)

    return stats


def _classify_scene(path: str) -> str:
    p = path.lower()
    if "bridge" in p or "episode" in p:
        return "tabletop_tidy"  # BridgeData = robot tabletop manipulation
    if "pick" in p or "place" in p or "deliver" in p:
        return "delivery"
    if "block" in p or "stack" in p or "occlu" in p:
        return "tabletop_tidy"
    if "storage" in p or "contain" in p:
        return "storage"
    if "washing" in p or "knob" in p or "screen_button" in p:
        return "storage"  # appliance = storage/operation
    if "benchmark" in p or "test_little" in p:
        return "tabletop_tidy"
    return "tabletop_tidy"


def _generate_html_report(stats, labels, eval_results, gap_report, out_root) -> str:
    """Generate a comprehensive HTML production report."""
    from collections import Counter

    bucket_counts = Counter(l.get("bucket") for l in labels)
    subtype_counts = Counter(l.get("subtype") for l in labels)

    # Sample cases per bucket
    samples_by_bucket = {}
    for l in labels:
        b = l.get("bucket", "")
        if b not in samples_by_bucket:
            samples_by_bucket[b] = []
        if len(samples_by_bucket[b]) < 3:
            samples_by_bucket[b].append(l)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GSA Data Flywheel v0 — Production Report {TODAY}</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:2em;background:#f5f7fa;color:#1a1a2e}}
h1{{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5em}}
h2{{color:#0f3460;margin-top:2em}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1em;margin:1em 0}}
.stat-card{{background:white;border-radius:12px;padding:1.5em;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center}}
.stat-card .value{{font-size:2.5em;font-weight:700;color:#0f3460}}
.stat-card .label{{color:#666;margin-top:0.3em;font-size:0.9em}}
table{{border-collapse:collapse;width:100%;margin:1em 0;background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.05)}}
th{{background:#16213e;color:white;padding:12px 16px;text-align:left}}
td{{padding:10px 16px;border-bottom:1px solid #eee}}
tr:hover td{{background:#f0f4ff}}
.case{{background:white;border-radius:8px;padding:1em;margin:0.5em 0;box-shadow:0 1px 4px rgba(0,0,0,0.06);font-family:monospace;font-size:0.85em;white-space:pre-wrap;overflow-x:auto}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:600}}
.badge-g{{background:#d4edda;color:#155724}}
.badge-s{{background:#cce5ff;color:#004085}}
.badge-astate{{background:#fff3cd;color:#856404}}
.badge-aaction{{background:#f8d7da;color:#721c24}}
.progress{{background:#e9ecef;border-radius:4px;height:8px;margin-top:4px}}
.progress-bar{{background:#0f3460;height:100%;border-radius:4px}}
</style></head><body>
<h1>GSA Data Flywheel v0 — Production Report</h1>
<p>Run ID: <code>{out_root.name}</code> | Date: {TODAY} | Branch: <code>anvil_dev</code></p>

<h2>Overview</h2>
<div class="stats">
<div class="stat-card"><div class="value">{stats['images']}</div><div class="label">Images Processed</div></div>
<div class="stat-card"><div class="value">{len(labels)}</div><div class="label">Total Labels</div></div>
<div class="stat-card"><div class="value">{eval_results.overall_verifiable_rate:.0%}</div><div class="label">Verifiable Rate</div></div>
<div class="stat-card"><div class="value">{stats['reviews']}</div><div class="label">Review Queue</div></div>
</div>

<h2>Per-Bucket Breakdown</h2>
<table>
<tr><th>Bucket</th><th>Count</th><th>Avg Score</th><th>Verifiable Rate</th><th>Subtypes</th></tr>
"""
    for b_name, bs in eval_results.bucket_scores.items():
        badge_cls = {"G": "badge-g", "S": "badge-s", "A_state": "badge-astate", "A_action": "badge-aaction"}.get(b_name, "")
        subtypes = ", ".join(bs.subtype_counts.keys())
        html += f"""<tr><td><span class="badge {badge_cls}">{b_name}</span></td>
<td>{bs.total}</td><td>{bs.avg_score:.3f}</td><td>{bs.verifiable_rate:.0%}</td><td>{subtypes}</td></tr>
"""
    html += "</table>"

    # Subtype distribution
    html += "<h2>Subtype Distribution</h2><table><tr><th>Subtype</th><th>Count</th><th>Coverage</th></tr>"
    for st, count in sorted(subtype_counts.items(), key=lambda x: -x[1]):
        pct = count / len(labels) * 100 if labels else 0
        html += f'<tr><td>{st}</td><td>{count}</td><td><div class="progress"><div class="progress-bar" style="width:{pct}%"></div></div> {pct:.1f}%</td></tr>'
    html += "</table>"

    # Sample cases
    html += "<h2>Sample Cases (per bucket)</h2>"
    for bucket, cases in samples_by_bucket.items():
        badge_cls = {"G": "badge-g", "S": "badge-s", "A_state": "badge-astate", "A_action": "badge-aaction"}.get(bucket, "")
        html += f'<h3><span class="badge {badge_cls}">{bucket}</span></h3>'
        for case in cases:
            case_str = json.dumps(case, indent=2, ensure_ascii=False)[:500]
            html += f'<div class="case">{case_str}</div>'

    # Gap summary
    html += f"""<h2>Gap Analysis</h2>
<p>Total gaps: {gap_report.summary.get('total_gaps', 0)} | Critical: {gap_report.summary.get('critical_gaps', 0)}</p>
<table><tr><th>Bucket</th><th>Subtype</th><th>Scene</th><th>Severity</th><th>Count/Target</th></tr>"""
    for g in gap_report.gaps[:20]:
        html += f"<tr><td>{g.bucket}</td><td>{g.subtype}</td><td>{g.scene_family}</td><td>{g.severity}</td><td>{g.current_count}/{g.target_count}</td></tr>"
    html += "</table>"

    html += f"""
<h2>Next Steps</h2>
<ul>
<li>Increase image diversity (current: {stats['images']} images, need 100+)</li>
<li>Improve G worker prompts for better object detection (avoid generic "rectangle")</li>
<li>Add VLM-driven A_action labels (currently mostly heuristic/review)</li>
<li>When G/S/A quality reaches target: Train Teacher for On Policy Distillation</li>
</ul>
<footer style="margin-top:3em;padding-top:1em;border-top:1px solid #ddd;color:#666;font-size:0.85em">
Generated by GSA Data Flywheel v0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Branch: anvil_dev
</footer></body></html>"""
    return html


if __name__ == "__main__":
    run_production()
