"""GSA v0 Pipeline Runner — End-to-end orchestration.

Runs the full GSA v0 flow:
1. Load images and existing annotations
2. Run G worker (detect + ground objects)
3. Run S worker (spatial relations)
4. Run A_state pipeline (object identity + state)
5. Run A_action pipeline (pick/push/place labels)
6. Run eval harness (offline scoring)
7. Generate gap report
8. Export training packs
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..workers.vlm_client import VLMClient
from .a_action_pipeline import AActionPipeline
from .a_state_pipeline import AStatePipeline
from .a_identity_attr_pipeline import AIdentityAttrPipeline
from .a_obj_state_pipeline import AObjStatePipeline
from .a_robot_state_pipeline import ARobotStatePipeline
from .a_action_precondition_pipeline import AActionPreconditionPipeline
from .a_action_interface_pipeline import AActionInterfacePipeline
from .eval_harness import GSAEvalHarness
from .episode_robot_state_pipeline import ARobotStateEpisodePipeline
from .gap_report import GSAGapReporter
from .grounding_worker import GSAGroundingWorker
from .manifest import GSAManifest
from .spatial_worker import GSASpatialWorker
from .taxonomy import Bucket, SceneFamily
from .visual_evidence import GsaEvidenceBundle, assign_object_ids
from .data_adapters.nas_adapter import discover_available_datasets, get_images_from_dataset
from .data_adapters.bridge_adapter import discover_bridge_images, is_wrist_camera_image
from .data_adapters.public_datasets import discover_public_datasets
from .data_adapters.robot_state_ingest import build_robot_state_registry

logger = logging.getLogger("gsa.runner")


class GSAv0Runner:
    """End-to-end GSA v0 pipeline runner."""

    def __init__(self, config_path: str = "scripts/configs/gsa_v0.yaml"):
        self.config = self._load_config(config_path)
        self.run_id = f"gsa_v0_{int(time.time())}"
        self.output_root = Path(self.config["output"]["root"]) / self.run_id

        # Initialize VLM client
        vlm_cfg = self.config["vlm"]
        api_key = os.environ.get("VLM_API_KEY") or vlm_cfg.get("api_key", "")
        self.vlm = VLMClient(
            base_url=vlm_cfg["base_url"],
            model=vlm_cfg["model"],
            api_key=api_key,
            max_image_size=vlm_cfg.get("max_image_size", 1280),
            timeout=vlm_cfg.get("timeout", 120),
            max_retries=vlm_cfg.get("max_retries", 3),
        )

        # Initialize workers
        self.g_worker = GSAGroundingWorker(self.vlm)
        self.s_worker = GSASpatialWorker(self.vlm)
        # Retain legacy stages for explicitly gated historical configurations.
        self.a_state_pipeline = AStatePipeline(self.vlm)
        self.a_action_pipeline = AActionPipeline(self.vlm)
        # Fine-grained A pipelines
        self.a_identity_attr = AIdentityAttrPipeline(self.vlm)
        self.a_obj_state = AObjStatePipeline(self.vlm)
        self.a_action_precondition = AActionPreconditionPipeline(self.vlm)
        self.a_action_interface = AActionInterfacePipeline(self.vlm)
        self.eval_harness = GSAEvalHarness()
        self.gap_reporter = GSAGapReporter(
            target_per_subtype=self.config.get("quotas", {}).get("G", 100)
        )
        self.manifest = GSAManifest()
        self._ingested_robot_state_ids: set[str] = set()
        verifier_cfg = self.config.get("verifier", {})
        self._auto_pass_confidence = verifier_cfg.get("auto_pass_confidence", 0.8)
        input_cfg = self.config.get("input", {}) or {}
        self.a_robot_state = ARobotStatePipeline(
            self.vlm,
            auto_pass_confidence=self._auto_pass_confidence,
            force_robot_state_vlm=bool(input_cfg.get("force_robot_state_vlm", False)),
            robot_state_vlm_roots=input_cfg.get("robot_state_vlm_roots", []) or [],
        )
        self.a_robot_state_episode = ARobotStateEpisodePipeline(
            self.vlm,
            auto_pass_confidence=self._auto_pass_confidence,
        )

    def run(self, image_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the full GSA v0 pipeline."""
        self._setup_output_dirs()

        if image_paths is None:
            image_paths = self._discover_images()

        raw_max_images = self.config.get("input", {}).get("max_images", 1000)
        try:
            max_images = int(raw_max_images)
        except (TypeError, ValueError):
            logger.warning("Invalid input.max_images=%r; using 1000", raw_max_images)
            max_images = 1000
        image_paths = image_paths[:max_images]

        ingest_only = (
            os.environ.get("GSA_INGEST_ONLY", "").lower() in ("1", "true", "yes")
            or self.config.get("robot_state_datasets", {}).get("ingest_only", False)
        )
        if ingest_only:
            image_paths = []
            logger.info("Ingest-only mode: skipping VLM image pipeline")

        logger.info(f"GSA v0 run {self.run_id}: {len(image_paths)} images")

        all_labels: List[Dict[str, Any]] = []
        all_review: List[Dict[str, Any]] = []
        episode_summaries: List[Dict[str, Any]] = []
        episode_dynamic_labels: List[Dict[str, Any]] = []

        ingested_robot = self._ingest_robot_state_labels()
        if ingested_robot:
            all_labels.extend(ingested_robot)
            logger.info("Structured A_robot_state ingest: %d labels", len(ingested_robot))
            for label in ingested_robot:
                if label.get("verifier_status") == "review_required":
                    all_review.append({
                        "bucket": Bucket.A_ROBOT_STATE.value,
                        "sample_id": label.get("sample_id"),
                        "reason": "missing_image_or_heuristic_only",
                        "image_path": label.get("image_path"),
                        "state_type": label.get("state_type"),
                        "source_dataset": label.get("source_dataset"),
                    })

        episode_labels, episode_summaries, episode_dynamic_labels = self._run_robot_state_episode_pipeline()
        if episode_labels:
            all_labels.extend(episode_labels)
            logger.info(
                "Episode A_robot_state VLM: %d frame labels, %d episode summaries",
                len(episode_labels),
                len(episode_summaries),
            )
            for label in episode_labels:
                if label.get("verifier_status") == "review_required":
                    all_review.append({
                        "bucket": Bucket.A_ROBOT_STATE.value,
                        "sample_id": label.get("sample_id"),
                        "reason": "episode_vlm_quality_flags",
                        "image_path": label.get("image_path"),
                        "state_type": label.get("state_type"),
                        "confidence": label.get("confidence"),
                        "source_dataset": label.get("source_dataset"),
                    })

        processed = self._process_images(image_paths, all_labels, all_review)
        if processed:
            logger.info(
                "Image pipeline complete: %d/%d images, %d labels, %d reviews",
                processed, len(image_paths), len(all_labels), len(all_review)
            )

        # Final save
        self._save_labels(all_labels)
        self._save_episode_summaries(episode_summaries)
        self._save_dynamic_episode_labels(episode_dynamic_labels)
        self._save_review_queue(all_review)
        self._save_dataset_summary(all_labels, episode_summaries, episode_dynamic_labels)

        # Run eval harness
        eval_results = self.eval_harness.evaluate(all_labels)
        eval_path = self.output_root / "eval" / "gsa_eval_results.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(eval_results.to_json())

        # Generate gap report
        gap_report = self.gap_reporter.generate(all_labels, eval_results)
        report_dir = self.output_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "gsa_gap_report_v0.json").write_text(gap_report.to_json())
        (report_dir / "gsa_gap_report_v0.html").write_text(gap_report.to_html())

        # Build manifest
        self.manifest.add_from_labels(all_labels)
        self.manifest.save(str(self.output_root / "dataset_manifest_v0.jsonl"))

        # Generate training packs
        self._generate_packs(all_labels)

        summary = {
            "run_id": self.run_id,
            "images_processed": len(image_paths),
            "total_labels": len(all_labels),
            "total_reviews": len(all_review),
            "episode_summaries": len(episode_summaries),
            "episode_dynamic_labels": len(episode_dynamic_labels),
            "eval_results": eval_results.to_dict(),
            "gap_summary": gap_report.summary,
        }

        logger.info(f"GSA v0 complete: {len(all_labels)} labels, "
                    f"{len(all_review)} reviews, "
                    f"verifiable_rate={eval_results.overall_verifiable_rate:.2%}")

        return summary


    def _configured_concurrency(self) -> int:
        pipeline_cfg = self.config.get("pipeline", {}) or {}
        vlm_cfg = self.config.get("vlm", {}) or {}
        raw = pipeline_cfg.get("concurrency", vlm_cfg.get("concurrency", 1))
        try:
            concurrency = int(raw)
        except (TypeError, ValueError):
            concurrency = 1
        return max(1, concurrency)

    def _enabled_stages(self) -> Optional[set[str]]:
        raw = (self.config.get("pipeline", {}) or {}).get("enabled_stages")
        if not raw:
            return None
        if isinstance(raw, str):
            raw = [s.strip() for s in raw.split(",") if s.strip()]
        return {str(stage) for stage in raw}

    def _stage_enabled(self, stage: str) -> bool:
        enabled = self._enabled_stages()
        return enabled is None or stage in enabled

    def _process_images(
        self,
        image_paths: List[str],
        all_labels: List[Dict[str, Any]],
        all_review: List[Dict[str, Any]],
    ) -> int:
        """Process images sequentially or concurrently; aggregation stays on main thread."""
        if not image_paths:
            return 0

        concurrency = self._configured_concurrency()
        if concurrency <= 1 or len(image_paths) == 1:
            processed = 0
            for idx, img_path in enumerate(image_paths):
                sample_id = f"{self.run_id}_{idx:06d}"
                scene_family = self._classify_scene(img_path)
                scenario_id = self._scene_to_scenario(scene_family)
                try:
                    labels, reviews = self._process_image(
                        img_path, sample_id, scene_family, scenario_id)
                    all_labels.extend(labels)
                    all_review.extend(reviews)
                    processed += 1
                except Exception as e:
                    logger.error("Failed to process %s: %s", img_path, e)
                    continue
                if processed % 10 == 0:
                    logger.info(
                        "Progress: %d/%d images, %d labels, %d reviews",
                        processed, len(image_paths), len(all_labels), len(all_review)
                    )
            return processed

        logger.info("Processing %d images with concurrency=%d", len(image_paths), concurrency)
        processed = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for idx, img_path in enumerate(image_paths):
                sample_id = f"{self.run_id}_{idx:06d}"
                scene_family = self._classify_scene(img_path)
                scenario_id = self._scene_to_scenario(scene_family)
                future = executor.submit(
                    self._process_image, img_path, sample_id, scene_family, scenario_id
                )
                futures[future] = img_path

            for future in as_completed(futures):
                img_path = futures[future]
                try:
                    labels, reviews = future.result()
                except Exception as e:
                    logger.error("Failed to process %s: %s", img_path, e)
                    continue
                all_labels.extend(labels)
                all_review.extend(reviews)
                processed += 1
                if processed % 10 == 0 or processed == len(image_paths):
                    logger.info(
                        "Progress: %d/%d images, %d labels, %d reviews",
                        processed, len(image_paths), len(all_labels), len(all_review)
                    )
        return processed

    def _process_image(self, image_path: str, sample_id: str,
                       scene_family: str, scenario_id: str):
        """Process a single image through configured GSA stages."""
        labels = []
        reviews = []

        g_labels = []
        if self._stage_enabled(Bucket.G.value):
            g_labels = self.g_worker.generate(image_path, sample_id, scene_family, scenario_id)
            labels.extend(g_labels)

        objects_raw = [l.get("answer", {}) for l in g_labels
                       if l.get("subtype") == "simple_grounding"]
        objects_for_ids = []
        for obj in objects_raw:
            if isinstance(obj, dict) and obj.get("bbox"):
                objects_for_ids.append({
                    "object_name": obj.get("object_name", "unknown"),
                    "bbox": obj.get("bbox"),
                    "source_provenance": "vlm",
                })
        obj_refs = assign_object_ids(objects_for_ids)
        obj_refs = self._filter_downstream_objects(obj_refs, image_path)

        s_labels = []
        spatial_facts = []
        if self._stage_enabled(Bucket.S.value):
            s_labels = self.s_worker.generate(image_path, sample_id, obj_refs, scene_family, scenario_id)
            labels.extend(s_labels)
            spatial_facts = self.s_worker.extract_spatial_facts(s_labels)

        bundle = GsaEvidenceBundle(
            sample_id=sample_id,
            image_path=image_path,
            scene_family=scene_family,
            scenario_id=scenario_id,
            objects=obj_refs,
            spatial_facts=spatial_facts,
        )

        if self._stage_enabled(Bucket.A_STATE.value):
            a_state_labels = self.a_state_pipeline.run(bundle)
            for asl in a_state_labels:
                labels.append(asl.to_dict())

        if self._stage_enabled(Bucket.A_ACTION.value):
            a_action_labels, a_action_reviews = self.a_action_pipeline.run(bundle)
            for aal in a_action_labels:
                labels.append(aal.to_dict())
            for rev in a_action_reviews:
                reviews.append(rev.to_dict())

        if self._stage_enabled(Bucket.A_IDENTITY_ATTR.value):
            id_attr_labels = self.a_identity_attr.run(bundle)
            for lbl in id_attr_labels:
                labels.append(lbl.to_dict())

        if self._stage_enabled(Bucket.A_OBJ_STATE.value):
            obj_state_labels = self.a_obj_state.run(bundle)
            for lbl in obj_state_labels:
                labels.append(lbl.to_dict())

        if self._stage_enabled(Bucket.A_ROBOT_STATE.value) and (
            is_wrist_camera_image(image_path) or self.a_robot_state.accepts_image_path(image_path)
        ):
            robot_labels = self.a_robot_state.run(bundle)
            for lbl in robot_labels:
                d = lbl.to_dict()
                d["bucket"] = Bucket.A_ROBOT_STATE.value
                labels.append(d)
                if d.get("verifier_status") == "review_required":
                    reviews.append({
                        "bucket": Bucket.A_ROBOT_STATE.value,
                        "sample_id": d.get("sample_id"),
                        "reason": "vlm_low_confidence",
                        "image_path": d.get("image_path"),
                        "state_type": d.get("state_type"),
                        "confidence": d.get("confidence"),
                        "source_dataset": d.get("source_dataset"),
                    })

        if self._stage_enabled(Bucket.A_ACTION_PRECONDITION.value):
            precond_labels = self.a_action_precondition.run(bundle)
            for lbl in precond_labels:
                labels.append(lbl.to_dict())

        if self._stage_enabled(Bucket.A_ACTION_INTERFACE.value):
            interface_labels = self.a_action_interface.run(bundle)
            for lbl in interface_labels:
                labels.append(lbl.to_dict())

        return labels, reviews

    def _filter_downstream_objects(self, obj_refs: List[Any], image_path: str) -> List[Any]:
        """Bound downstream inference without changing the raw grounding labels."""
        cfg = self.config.get("input", {})
        normalize = lambda name: str(name).strip().lower().replace(" ", "_")
        excluded = {normalize(n) for n in cfg.get("downstream_exclude_object_names", [])}
        preferred = {normalize(n) for n in cfg.get("downstream_preferred_object_names", [])}
        area_limit = cfg.get("max_downstream_bbox_area_ratio")
        image_area = None
        if area_limit:
            from PIL import Image
            try:
                with Image.open(image_path) as image:
                    image_area = image.width * image.height
            except (OSError, ValueError):
                pass
        def matches(obj, names):
            return any(n in normalize(obj.object_name) or n in normalize(obj.raw_object_name) for n in names)
        result = []
        for obj in obj_refs:
            if matches(obj, excluded):
                continue
            if area_limit and image_area and obj.bbox and len(obj.bbox) == 4:
                x1, y1, x2, y2 = obj.bbox
                if max(0, x2-x1) * max(0, y2-y1) / image_area > float(area_limit):
                    continue
            result.append(obj)
        if preferred:
            result.sort(key=lambda obj: not matches(obj, preferred))
        limit = cfg.get("max_objects_per_image")
        return result[:limit] if isinstance(limit, int) and limit > 0 else result

    def _ingest_robot_state_labels(self) -> List[Dict[str, Any]]:
        """Ingest structured A_robot_state labels from registered dataset adapters."""
        cfg = self.config.get("robot_state_datasets", {})
        enabled = cfg.get("enabled") or []
        if not enabled:
            return []

        registry = build_robot_state_registry()
        overrides = cfg.get("nas_overrides", {})
        max_per = cfg.get("max_samples_per_source", 2000)
        labels = registry.ingest_all(enabled, overrides, max_per)

        ingested_ids = {lbl.sample_id for lbl in labels}
        self._ingested_robot_state_ids = ingested_ids

        out: List[Dict[str, Any]] = []
        for label in labels:
            d = label.to_dict()
            d["bucket"] = Bucket.A_ROBOT_STATE.value
            out.append(d)
        return out

    def _run_robot_state_episode_pipeline(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run optional multi-frame episode A_robot_state inference."""
        episode_cfg = self.config.get("robot_state_episode", {}) or {}
        if not episode_cfg.get("enabled", False):
            return [], [], []

        dataset_cfg = self.config.get("robot_state_datasets", {}) or {}
        overrides = dataset_cfg.get("nas_overrides", {}) or {}
        include_dynamic = bool(episode_cfg.get("output_dynamic_labels", False))
        result = self.a_robot_state_episode.run_from_config(
            episode_cfg, overrides, include_dynamic=include_dynamic
        )
        if include_dynamic and len(result) == 3:
            labels, summaries, dynamic_labels = result
        else:
            labels, summaries = result
            dynamic_labels = []
        if not episode_cfg.get("output_episode_summary", True):
            summaries = []

        out: List[Dict[str, Any]] = []
        for label in labels:
            d = label.to_dict()
            d["bucket"] = Bucket.A_ROBOT_STATE.value
            out.append(d)
        return out, summaries, dynamic_labels

    def _save_labels(self, labels: List[Dict[str, Any]]):
        """Save labels grouped by bucket."""
        by_bucket: Dict[str, List[Dict]] = {}
        for label in labels:
            bucket = label.get("bucket", "unknown")
            by_bucket.setdefault(bucket, []).append(label)

        for bucket, bucket_labels in by_bucket.items():
            out_dir = self.output_root / "labels" / bucket
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{bucket}.jsonl"
            with open(out_path, "w") as f:
                for label in bucket_labels:
                    f.write(json.dumps(label, ensure_ascii=False) + "\n")

    def _save_review_queue(self, reviews: List[Dict[str, Any]]):
        """Save review queue grouped by bucket."""
        by_bucket: Dict[str, List[Dict]] = {}
        for rev in reviews:
            bucket = rev.get("bucket", "unknown")
            by_bucket.setdefault(bucket, []).append(rev)

        for bucket, bucket_reviews in by_bucket.items():
            out_dir = self.output_root / "review_queue"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{bucket}.review.jsonl"
            with open(out_path, "w") as f:
                for rev in bucket_reviews:
                    f.write(json.dumps(rev, ensure_ascii=False) + "\n")

    def _save_episode_summaries(self, summaries: List[Dict[str, Any]]) -> None:
        """Save optional episode-level robot-state summaries."""
        if not summaries:
            return
        out_dir = self.output_root / "labels" / "A_robot_state_episode"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "episode_summaries.jsonl"
        with open(out_path, "w") as f:
            for summary in summaries:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def _save_dynamic_episode_labels(self, dynamic_labels: List[Dict[str, Any]]) -> None:
        """Save optional dynamic episode labels separately from A_robot_state."""
        if not dynamic_labels:
            return
        out_dir = self.output_root / "labels" / "A_robot_state_dynamic"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "A_robot_state_dynamic.jsonl"
        with open(out_path, "w") as f:
            for label in dynamic_labels:
                f.write(json.dumps(label, ensure_ascii=False) + "\n")

    def _save_dataset_summary(
        self,
        labels: List[Dict[str, Any]],
        episode_summaries: Optional[List[Dict[str, Any]]] = None,
        episode_dynamic_labels: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Write dataset_summary.json for A_robot_state reporting."""
        from collections import Counter

        robot_labels = [
            l for l in labels if l.get("bucket") == Bucket.A_ROBOT_STATE.value
        ]
        quality_flags: Counter[str] = Counter()
        for label in robot_labels:
            for flag in label.get("quality_flags") or []:
                quality_flags[str(flag)] += 1

        verifier_status_counts = Counter(
            l.get("verifier_status", "?") for l in robot_labels
        )
        auto_count = verifier_status_counts.get("auto_verifiable", 0)
        robot_count = len(robot_labels)
        episode_summaries = episode_summaries or []
        episode_dynamic_labels = episode_dynamic_labels or []
        summary = {
            "run_id": self.run_id,
            "total_labels": len(labels),
            "A_robot_state": {
                "candidate_count": robot_count,
                "stable_training_label_count": sum(
                    1 for l in robot_labels if l.get("stable_training_label")
                ),
                "auto_verifiable": auto_count,
                "review_required": verifier_status_counts.get("review_required", 0),
                "verifiable_rate": round(auto_count / robot_count, 4) if robot_count else 0.0,
                "subtype_counts": dict(Counter(
                    l.get("state_type", "?") for l in robot_labels
                )),
                "verifier_status_counts": dict(verifier_status_counts),
                "verifier_result_counts": dict(Counter(
                    l.get("verifier_result", "missing") for l in robot_labels
                )),
                "state_value_counts": dict(Counter(
                    l.get("state_value", "?") for l in robot_labels
                )),
                "label_status_counts": dict(Counter(
                    l.get("label_status", "missing") for l in robot_labels
                )),
                "source_counts": dict(Counter(
                    l.get("source_dataset", "?") for l in robot_labels
                )),
                "scene_family_counts": dict(Counter(
                    l.get("scene_family", "") for l in robot_labels
                )),
                "quality_flag_counts": dict(quality_flags),
            },
            "A_robot_state_episode": {
                "summary_count": len(episode_summaries),
                "source_counts": dict(Counter(
                    s.get("source_dataset", "?") for s in episode_summaries
                )),
                "episode_outcome_counts": dict(Counter(
                    s.get("episode_outcome", "?") for s in episode_summaries
                )),
                "quality_flag_counts": dict(Counter(
                    flag
                    for s in episode_summaries
                    for flag in (s.get("quality_flags") or [])
                )),
            },
            "A_robot_state_dynamic": {
                "label_count": len(episode_dynamic_labels),
                "source_counts": dict(Counter(
                    d.get("source_dataset", "?") for d in episode_dynamic_labels
                )),
                "verifier_status_counts": dict(Counter(
                    ((d.get("quality") or {}).get("verifier_status", "?"))
                    for d in episode_dynamic_labels
                )),
                "quality_flag_counts": dict(Counter(
                    flag
                    for d in episode_dynamic_labels
                    for flag in ((d.get("quality") or {}).get("quality_flags") or [])
                )),
            },
        }
        path = self.output_root / "dataset_summary.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def _generate_packs(self, labels: List[Dict[str, Any]]):
        """Generate training pack manifests (only auto_verifiable labels)."""
        packs_dir = self.output_root / "packs"
        packs_dir.mkdir(parents=True, exist_ok=True)

        pack_names = {
            Bucket.G.value: "G_core_pack_manifest.jsonl",
            Bucket.S.value: "S_core_pack_manifest.jsonl",
            Bucket.A_STATE.value: "A_state_pack_manifest.jsonl",
            Bucket.A_ACTION.value: "A_action_pack_manifest.jsonl",
            Bucket.A_IDENTITY_ATTR.value: "A_identity_attr_pack_manifest.jsonl",
            Bucket.A_OBJ_STATE.value: "A_obj_state_pack_manifest.jsonl",
            Bucket.A_ROBOT_STATE.value: "A_robot_state_pack_manifest.jsonl",
            Bucket.A_ACTION_PRECONDITION.value: "A_action_precondition_pack_manifest.jsonl",
            Bucket.A_ACTION_INTERFACE.value: "A_action_interface_pack_manifest.jsonl",
        }

        for bucket, filename in pack_names.items():
            stable = [l for l in labels
                      if l.get("bucket") == bucket
                      and l.get("verifier_status") == "auto_verifiable"]
            if not stable:
                continue
            with open(packs_dir / filename, "w") as f:
                for label in stable:
                    f.write(json.dumps(label, ensure_ascii=False) + "\n")
            logger.info(f"Pack {filename}: {len(stable)} stable labels")

    def _setup_output_dirs(self):
        """Create output directory structure."""
        dirs = ["labels", "review_queue", "eval", "reports", "packs"]
        for d in dirs:
            (self.output_root / d).mkdir(parents=True, exist_ok=True)

    def _discover_images(self) -> List[str]:
        """Find images from configured directories, glob patterns, and NAS data sources."""
        from glob import glob

        input_cfg = self.config.get("input", {}) or {}
        image_dirs = input_cfg.get("image_dirs", [])
        image_globs = input_cfg.get("image_globs", [])
        images: List[str] = []

        for pattern in image_dirs:
            for d in glob(pattern):
                if os.path.isdir(d):
                    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                        images.extend(glob(os.path.join(d, ext)))
                elif os.path.isfile(d):
                    images.append(d)

        for pattern in image_globs:
            images.extend(glob(pattern, recursive=True))

        if input_cfg.get("discover_external_sources", True):
            # Add NAS data sources
            nas_datasets = discover_available_datasets()
            for ds in nas_datasets:
                if ds["ready"]:
                    nas_images = get_images_from_dataset(ds["name"], max_images=50)
                    images.extend(nas_images)
                    if nas_images:
                        logger.info(f"NAS [{ds['name']}]: {len(nas_images)} images -> {ds['pipeline']}")

            # Add BridgeDataV2
            bridge_images = discover_bridge_images(max_images=50)
            images.extend(bridge_images)
            if bridge_images:
                logger.info(f"BridgeDataV2: {len(bridge_images)} images")

        deduped = sorted(dict.fromkeys(images))
        if input_cfg.get("balance_image_sources", False):
            try:
                max_images = int(input_cfg.get("max_images", len(deduped)))
            except (TypeError, ValueError):
                max_images = len(deduped)
            return self._balance_images_by_source(deduped, max_images)
        return deduped

    def _image_source_key(self, image_path: str) -> str:
        lp = image_path.lower()
        if "bridge_data_v2" in lp or "bridgedatav2" in lp:
            return "bridge_data_v2"
        if "fmb_extracted" in lp or "/fmb/" in lp:
            return "fmb"
        if "droid_100" in lp or "/droid/" in lp:
            return "droid_100"
        if "embodied_reasoner" in lp:
            return "embodied_reasoner"
        if "robo2vlm" in lp:
            return "robo2vlm"
        if "sharerobot" in lp:
            return "sharerobot"
        return "other"

    def _balance_images_by_source(self, images: List[str], max_images: int) -> List[str]:
        if max_images <= 0:
            return []
        input_cfg = self.config.get("input", {}) or {}
        priority = input_cfg.get("image_source_priority") or [
            "bridge_data_v2", "fmb", "droid_100", "embodied_reasoner", "robo2vlm", "sharerobot", "other"
        ]
        buckets: Dict[str, List[str]] = {str(key): [] for key in priority}
        for image in images:
            key = self._image_source_key(image)
            buckets.setdefault(key, []).append(image)
        for bucket_images in buckets.values():
            bucket_images.sort()

        balanced: List[str] = []
        while len(balanced) < max_images:
            added = False
            for key in priority:
                bucket = buckets.get(str(key), [])
                if bucket:
                    balanced.append(bucket.pop(0))
                    added = True
                    if len(balanced) >= max_images:
                        break
            if not added:
                break
        return balanced

    def _classify_scene(self, image_path: str) -> str:
        """Classify image into a scene family based on path keywords."""
        path_lower = image_path.lower()
        families = self.config.get("scene_families", {})
        for family, cfg in families.items():
            for kw in cfg.get("keywords", []):
                if kw in path_lower:
                    return family
        return SceneFamily.TABLETOP_TIDY.value  # default

    def _scene_to_scenario(self, scene_family: str) -> str:
        families = self.config.get("scene_families", {})
        for family, cfg in families.items():
            if family == scene_family:
                return cfg.get("scenario", "SC.2")
        return "SC.2"

    def _expand_config_env(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._expand_config_env(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._expand_config_env(v) for v in value]
        return VLMClient.expand_env(value)

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            config = yaml.safe_load(f)
        return self._expand_config_env(config)
