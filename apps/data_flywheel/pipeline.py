"""飞轮编排主入口 — 串联 Step1-8，支持多轮迭代，支持全 API 模式。"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from apps.data_flywheel.config import FlywheelConfig

logger = logging.getLogger("flywheel")


def _to_relative(abs_path: str, base: str) -> str:
    try:
        return os.path.relpath(abs_path, base)
    except ValueError:
        return abs_path


def _relativize_jsonl(jsonl_path: str, base: str, image_keys: tuple = ("image",)) -> None:
    if not os.path.exists(jsonl_path):
        return
    lines: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            for key in image_keys:
                if key in item and item[key] and os.path.isabs(item[key]):
                    item[key] = _to_relative(item[key], base)
            lines.append(json.dumps(item, ensure_ascii=False))
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _relativize_eval_report(report_path: str, base: str) -> None:
    if not os.path.exists(report_path):
        return
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    for sample in report.get("samples", []):
        if "image" in sample and sample["image"] and os.path.isabs(sample["image"]):
            sample["image"] = _to_relative(sample["image"], base)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _sanitize_detector_tag(raw: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_-]+", "_", (raw or "").strip())
    return tag.strip("_").lower() or "detector"


def _load_jsonl_records(jsonl_path: Optional[str]) -> List[Dict[str, Any]]:
    if not jsonl_path or not os.path.exists(jsonl_path):
        return []
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                records.append(json.loads(raw))
    return records


def _bbox_polygon(bbox: List[float]) -> List[List[float]]:
    return [
        [float(bbox[0]), float(bbox[1])],
        [float(bbox[2]), float(bbox[1])],
        [float(bbox[2]), float(bbox[3])],
        [float(bbox[0]), float(bbox[3])],
    ]


def _bbox_center_point(bbox: List[float]) -> List[float]:
    return [
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ]


def _legacy_scene_record_to_scene_graph(record: Dict[str, Any]):
    from apps.data_flywheel.types import (
        InstanceAttributes,
        PlacementRegion,
        SceneEdge,
        SceneGraph,
        SceneNode,
        SupportRelation,
    )

    objects = list(record.get("objects") or [])
    index_to_node_id: Dict[Any, str] = {}
    nodes: List[SceneNode] = []
    for obj in objects:
        node_id = str(obj.get("node_id") or obj.get("id", len(nodes)))
        index_to_node_id[obj.get("id")] = node_id
        placement = obj.get("placement") or {}
        perception = obj.get("perception") or {}
        center_3d = obj.get("center_3d")
        size_3d = obj.get("size_3d")
        attributes = obj.get("attributes") or {}
        bbox = [float(value) for value in (obj.get("bbox") or [0.0, 0.0, 0.0, 0.0])]
        mask_polygon = obj.get("mask_polygon") or _bbox_polygon(bbox)
        mask_source = obj.get("mask_source")
        has_true_mask = bool(obj.get("has_true_mask"))
        if mask_source in {"mask_polygon", "mask_file"}:
            has_true_mask = True
        if not has_true_mask and obj.get("mask_file"):
            mask_source = mask_source or "mask_file"
            has_true_mask = True
        if not has_true_mask and obj.get("mask_polygon"):
            normalized_mask = [[float(point[0]), float(point[1])] for point in mask_polygon]
            if normalized_mask != _bbox_polygon(bbox):
                mask_source = mask_source or "mask_polygon"
                has_true_mask = True
        if mask_source is None:
            mask_source = "bbox_fallback"
        grasp_point = obj.get("grasp_point") or _bbox_center_point(bbox)
        nodes.append(
            SceneNode(
                id=node_id,
                class_name=str(obj.get("name") or obj.get("category") or "object"),
                bbox=bbox,
                mask_polygon=[[float(point[0]), float(point[1])] for point in mask_polygon],
                mask_file=obj.get("mask_file"),
                mask_source=mask_source,
                has_true_mask=has_true_mask,
                grasp_point=[float(value) for value in grasp_point],
                center_3d=list(center_3d) if center_3d else None,
                size_3d=list(size_3d) if size_3d else None,
                mean_depth=obj.get("mean_depth"),
                valid_3d=bool(obj.get("valid")) and bool(center_3d) and bool(size_3d),
                operation=obj.get("operation"),
                role_ids=list(obj.get("role_ids") or []),
                allowed_operations=list(obj.get("allowed_operations") or []),
                preferred_operation=obj.get("preferred_operation"),
                is_container=bool(obj.get("is_container")),
                container_placeability=placement.get("container_placeability"),
                placement_policy=placement.get("placement_policy"),
                extra_tags=list(perception.get("extra_tags") or []),
                canonical_name=obj.get("canonical_name"),
                object_class_id=obj.get("object_class_id"),
                ontology_status=str(obj.get("ontology_status") or "unknown"),
                triggered_constraints=list(obj.get("triggered_constraints") or []),
                descriptor_axes=list(obj.get("descriptor_axes") or []),
                point_cloud_sample=[
                    [float(value) for value in point]
                    for point in (obj.get("point_cloud_sample") or [])
                ],
                support_parent=obj.get("support_parent"),
                support_type=obj.get("support_type"),
                support_excluded=bool(obj.get("support_excluded")),
                directly_reachable=obj.get("directly_reachable"),
                blocked_by=list(obj.get("blocked_by") or []),
                blocking=list(obj.get("blocking") or []),
                excluded_from_action_graph=bool(obj.get("excluded_from_action_graph")),
                hierarchical_name=obj.get("hierarchical_name"),
                referring_expressions=list(obj.get("referring_expressions") or []),
                attributes=InstanceAttributes(
                    color_primary=attributes.get("color_primary"),
                    color_primary_hsv=list(attributes.get("color_primary_hsv"))
                    if attributes.get("color_primary_hsv") is not None
                    else None,
                    color_secondary=attributes.get("color_secondary"),
                    size=attributes.get("size"),
                    shape=attributes.get("shape"),
                ),
            )
        )

    edges: List[SceneEdge] = []
    for edge in record.get("relations") or []:
        subject = index_to_node_id.get(edge.get("subject"), str(edge.get("subject")))
        object_ = index_to_node_id.get(edge.get("object"), str(edge.get("object")))
        edges.append(
            SceneEdge(
                subject=subject,
                object=object_,
                predicate=str(edge.get("predicate") or "related"),
                source=str(edge.get("source") or "derived"),
                confidence=float(edge.get("confidence") or 0.0),
                frame=str(edge.get("frame") or "untyped"),
            )
        )

    support_relations: List[SupportRelation] = []
    for relation in record.get("support_relations") or []:
        subject = index_to_node_id.get(relation.get("subject"), str(relation.get("subject")))
        object_ = index_to_node_id.get(relation.get("object"), str(relation.get("object")))
        support_relations.append(
            SupportRelation(
                subject=subject,
                object=object_,
                relation=str(relation.get("relation") or "on-top-of"),
                confidence=float(relation.get("confidence") or 0.0),
            )
        )

    placement_regions: List[PlacementRegion] = []
    for region in record.get("placement_regions") or []:
        support_id = index_to_node_id.get(region.get("support_id"), str(region.get("support_id")))
        anchor_ids = [
            index_to_node_id.get(anchor_id, str(anchor_id))
            for anchor_id in (region.get("anchor_ids") or [])
        ]
        polygon = [
            [float(point[0]), float(point[1])]
            for point in (region.get("polygon_2d") or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        point_2d = region.get("point_2d")
        placement_regions.append(
            PlacementRegion(
                region_id=str(region.get("region_id") or ""),
                region_type=str(region.get("region_type") or "free-space"),
                support_id=support_id,
                anchor_ids=anchor_ids,
                polygon_2d=polygon,
                point_2d=[float(point_2d[0]), float(point_2d[1])]
                if isinstance(point_2d, (list, tuple)) and len(point_2d) >= 2
                else None,
                reason=str(region.get("reason") or ""),
            )
        )

    image_path = str(record.get("image") or "")
    image_size_raw = record.get("image_size") or [1280, 800]
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    blocking_dag = [
        (
            index_to_node_id.get(edge[0], str(edge[0])),
            index_to_node_id.get(edge[1], str(edge[1])),
        )
        for edge in (record.get("blocking_dag") or [])
        if isinstance(edge, (list, tuple)) and len(edge) >= 2
    ]
    orientation_facts = [
        SceneEdge(
            subject=index_to_node_id.get(edge.get("subject"), str(edge.get("subject"))),
            object=index_to_node_id.get(edge.get("object"), str(edge.get("object"))),
            predicate=str(edge.get("predicate") or "related"),
            source=str(edge.get("source") or "derived"),
            confidence=float(edge.get("confidence") or 0.0),
            frame=str(edge.get("frame") or "untyped"),
        )
        for edge in (record.get("orientation_facts") or [])
        if isinstance(edge, dict)
    ]
    image_id = os.path.splitext(os.path.basename(image_path))[0] or "unknown"
    return SceneGraph(
        image_id=image_id,
        image_path=image_path,
        image_size=image_size,
        nodes=nodes,
        edges=edges,
        ternary_relations=[],
        blocking_dag=blocking_dag,
        support_relations=support_relations,
        placement_regions=placement_regions,
        orientation_facts=orientation_facts,
    )


def _legacy_scene_record_to_ontology_labels(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    for obj in record.get("objects") or []:
        perception = obj.get("perception") or {}
        labels.append(
            {
                "observation_name": obj.get("name") or obj.get("category") or "object",
                "canonical_name": obj.get("canonical_name") or obj.get("name") or obj.get("category") or "object",
                "object_class_id": obj.get("object_class_id") or obj.get("canonical_name") or obj.get("name") or obj.get("category") or "object",
                "object_subtype_id": obj.get("object_subtype_id"),
                "matched_alias": obj.get("matched_alias"),
                "operation": obj.get("operation"),
                "role_ids": list(obj.get("role_ids") or []),
                "allowed_operations": list(obj.get("allowed_operations") or []),
                "preferred_operation": obj.get("preferred_operation"),
                "extra_tags": list(perception.get("extra_tags") or []),
                "ontology_status": obj.get("ontology_status") or "unknown",
            }
        )
    return labels


def _grounding_item_to_detections(grounding_item: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not grounding_item:
        return detections
    for phrase in grounding_item.get("phrases") or []:
        class_name = str(phrase.get("phrase") or "object")
        for box in phrase.get("boxes") or []:
            bbox = box.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            detections.append(
                {
                    "class_name": class_name,
                    "bbox": [float(value) for value in bbox],
                    "confidence": float(box.get("confidence") or 1.0),
                    "source": "grounding",
                }
            )
    return detections


def _fallback_detections_from_scene_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for obj in record.get("objects") or []:
        detections.append(
            {
                "class_name": str(obj.get("name") or obj.get("category") or "object"),
                "bbox": [float(value) for value in (obj.get("bbox") or [0.0, 0.0, 0.0, 0.0])],
                "confidence": float(obj.get("confidence") or 1.0),
                "source": "scene_graph",
            }
        )
    return detections


def _run_training_data_for_round(
    scene_graph_jsonl: str,
    round_dir: str,
    ontology_store_path: str,
) -> Dict[str, Any]:
    from apps.data_flywheel.ontology.store import OntologyStore, resolve_store_path
    from apps.data_flywheel.training_data.runtime import build_training_artifacts

    output_dir = os.path.join(round_dir, "training_data")
    os.makedirs(output_dir, exist_ok=True)

    ontology = OntologyStore.load(resolve_store_path(ontology_store_path))
    scene_graphs = [
        _legacy_scene_record_to_scene_graph(record)
        for record in _load_jsonl_records(scene_graph_jsonl)
    ]
    artifacts = build_training_artifacts(
        scene_graphs,
        ontology=ontology,
        output_dir=output_dir,
    )

    return {
        "count": artifacts["images"],
        "instructions": artifacts["instruction_count"],
        "plan_count": artifacts["plan_results"],
        "sample_count": artifacts["accepted_samples"],
        "rejected_count": artifacts["rejected_samples"],
        "training_samples_path": artifacts["training_samples_path"],
        "rejected_samples_path": artifacts["rejected_samples_path"],
        "training_artifacts_by_image": {
            image_path: {
                "plan_results": artifacts["plan_results_by_image"].get(image_path, []),
                "training_samples": artifacts["training_samples_by_image"].get(image_path, []),
                "rejected_entries": artifacts["rejected_entries_by_image"].get(image_path, []),
            }
            for image_path in artifacts["plan_results_by_image"].keys()
        },
    }


def _generate_round_reports(
    grounding_jsonl: str,
    scene_graph_jsonl: str,
    training_artifacts_by_image: Optional[Dict[str, Dict[str, Any]]] = None,
    training_by_image: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    from apps.data_flywheel.config import PROJECT_ROOT
    from apps.data_flywheel.visualization import generate_report

    grounding_by_image = {
        str(item.get("image")): item
        for item in _load_jsonl_records(grounding_jsonl)
        if item.get("image")
    }
    reports_output = str(PROJECT_ROOT / "data" / "output" / "reports")
    report_paths: List[str] = []

    for record in _load_jsonl_records(scene_graph_jsonl):
        scene_graph = _legacy_scene_record_to_scene_graph(record)
        grounding_item = grounding_by_image.get(scene_graph.image_path)
        detections = _grounding_item_to_detections(grounding_item) or _fallback_detections_from_scene_record(record)
        training_artifacts = (training_artifacts_by_image or training_by_image or {}).get(scene_graph.image_path, {})
        report_paths.append(
            generate_report(
                image_path=scene_graph.image_path,
                image_size=scene_graph.image_size,
                ontology_labels=_legacy_scene_record_to_ontology_labels(record),
                detections=detections,
                segmentations=[],
                scene_graph=scene_graph,
                plan_results=training_artifacts.get("plan_results"),
                training_samples=training_artifacts.get("training_samples"),
                validation_report={"rejected_entries": training_artifacts.get("rejected_entries", [])},
                output_dir=reports_output,
            )
        )

    return report_paths


def _resolve_detector_entries(cfg: FlywheelConfig, default_use_api: bool) -> List[Dict[str, Any]]:
    """构建检测器列表。支持本地模型与 API 检测器混合。"""
    entries: List[Dict[str, Any]] = []
    models = [m for m in (cfg.detector_models or []) if str(m).strip()]
    if cfg.backend == "detvlm" and not default_use_api:
        if len(models) > 1:
            logger.warning("detvlm 后端不支持基于 model_path 的多 detector A/B，对比已折叠为单一远端服务。")
        return [{"tag": "detvlm", "use_api": False, "model_path": cfg.model_path}]
    if not models:
        if default_use_api:
            return [{"tag": "api", "use_api": True, "model_path": None}]
        return [{"tag": "primary", "use_api": False, "model_path": cfg.model_path}]

    for idx, m in enumerate(models):
        mstr = str(m).strip()
        if mstr.lower() in {"api", "qwen-vl-max"}:
            entries.append({"tag": "api" if idx == 0 else f"api_{idx}", "use_api": True, "model_path": None})
            continue
        base = os.path.basename(mstr.rstrip("/")) or f"model_{idx}"
        entries.append({"tag": _sanitize_detector_tag(base), "use_api": False, "model_path": mstr})
    return entries


def run_pipeline(cfg: Optional[FlywheelConfig] = None) -> Dict[str, Any]:
    """执行完整数据飞轮 pipeline。

    cfg.use_api=True 时全程用 VLM API，无需加载本地模型。
    """
    if cfg is None:
        cfg = FlywheelConfig()
    cfg.ensure_dirs()

    from apps.data_flywheel.augmentor import augment_directory
    from apps.data_flywheel.captioner import generate_captions
    from apps.data_flywheel.evaluator import evaluate_grounding
    from apps.data_flywheel.exporter import export_dataset
    from apps.data_flywheel.grounding import run_grounding
    from apps.data_flywheel.hard_mining import get_hard_image_paths, mine_hard_samples
    from apps.data_flywheel.pointing import run_pointing
    from apps.data_flywheel.referring import run_referring

    use_api = getattr(cfg, "use_api", False)
    use_detvlm = cfg.backend == "detvlm"
    detector_entries = _resolve_detector_entries(cfg, use_api)
    model_cache: Dict[str, Any] = {}

    def _get_model(model_path: str):
        if model_path in model_cache:
            return model_cache[model_path]
        if use_detvlm:
            from apps.data_flywheel.detvlm_adapter import DetVLMAdapter
            detvlm_url = getattr(cfg, "detvlm_url", "YOUR_DETVLM_URL")
            model_cache[model_path] = DetVLMAdapter(base_url=detvlm_url)
            logger.info(f"使用远程 DetVLM 服务: {detvlm_url}")
            return model_cache[model_path]
        from rex_omni import RexOmniWrapper
        logger.info(f"加载检测模型: {model_path}")
        model_cache[model_path] = RexOmniWrapper(
            model_path=model_path,
            backend=cfg.backend,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
        )
        return model_cache[model_path]

    # API 模式下 caption 强制用 API
    caption_backend = cfg.caption_backend
    if use_api and caption_backend == "local":
        caption_backend = "auto"

    all_reports: List[Dict] = []
    t0 = time.time()

    # 进度回调（GUI 可注入 cfg._progress_fn = lambda step, detail: ...）
    progress_fn = getattr(cfg, "_progress_fn", None)

    def _step(name: str, detail: str = ""):
        if progress_fn:
            progress_fn(name, detail)

    for round_idx in range(cfg.rounds):
        round_tag = f"round_{round_idx}"
        logger.info(f"===== Flywheel {round_tag} =====")
        step_timings: Dict[str, float] = {}

        # ---- Step 1: 图片收集 + 增强 ----
        _step("① 图片增强", "收集图片...")
        t_step = time.time()
        logger.info(f"[{round_tag}] Step1: 图片增强")
        all_images: List[str] = []
        aug_dir = os.path.join(cfg.output_root, round_tag, "augmented")
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for img_dir in cfg.image_dirs:
            if os.path.isdir(img_dir):
                if cfg.enable_augment:
                    augmented = augment_directory(
                        image_dir=img_dir,
                        output_dir=aug_dir,
                        augment_per_image=cfg.augment_per_image,
                        max_images=cfg.max_images_per_round,
                    )
                    all_images.extend(augmented)
                else:
                    originals = sorted(
                        os.path.join(img_dir, name)
                        for name in os.listdir(img_dir)
                        if os.path.splitext(name)[1].lower() in image_exts
                    )
                    remaining = max(0, int(cfg.max_images_per_round) - len(all_images))
                    all_images.extend(originals[:remaining or None])
                if len(all_images) >= int(cfg.max_images_per_round):
                    all_images = all_images[: int(cfg.max_images_per_round)]
                    break

        if not all_images:
            logger.warning(f"[{round_tag}] 没有找到图片，跳过")
            continue

        logger.info(f"[{round_tag}] 共 {len(all_images)} 张图片")
        step_timings["augment"] = round(time.time() - t_step, 2)

        # ---- Step 2: Caption 生成 ----
        _step("② Caption", f"{len(all_images)} 张图片...")
        t_step = time.time()
        logger.info(f"[{round_tag}] Step2: Caption 生成")
        if cfg.enable_caption:
            pairs = generate_captions(
                image_paths=all_images,
                caption_backend=caption_backend,
                model_path=cfg.model_path,
                backend=cfg.backend,
            )
        else:
            pairs = [(img, "") for img in all_images]
        logger.info(f"[{round_tag}] 生成 {len(pairs)} 条 caption")
        step_timings["caption"] = round(time.time() - t_step, 2)

        # ---- Step 3: Grounding 标注 ----
        _step("③ Grounding", f"{len(pairs)} 条 caption...")
        t_step = time.time()
        logger.info(f"[{round_tag}] Step3: Grounding 标注（支持 A/B）")
        round_dir = os.path.join(cfg.output_root, round_tag)
        grounding_jsonl = os.path.join(round_dir, "grounding.jsonl")
        vis_dir = os.path.join(cfg.output_root, round_tag, "visualizations")
        primary_tag = cfg.detector_primary.strip() if cfg.detector_primary else detector_entries[0]["tag"]
        detector_groundings: Dict[str, str] = {}
        detector_grounding_results: Dict[str, Dict[str, Any]] = {}
        detector_eval_results: Dict[str, Dict[str, Any]] = {}
        ontology_review_result: Dict[str, Any] = {"enabled": False}
        detector_compare_path = os.path.join(round_dir, "detector_compare.json")

        if cfg.enable_grounding:
            ab_pairs = pairs
            if getattr(cfg, "detection_ab_max_images", 0) > 0:
                ab_pairs = pairs[: int(cfg.detection_ab_max_images)]
            for det in detector_entries:
                det_tag = det["tag"]
                det_is_primary = det_tag == primary_tag
                det_use_api = bool(det.get("use_api", False))
                det_model_path = det.get("model_path", cfg.model_path)
                det_out = grounding_jsonl if det_is_primary else os.path.join(round_dir, f"grounding_{det_tag}.jsonl")
                det_vis_dir = vis_dir if det_is_primary else os.path.join(vis_dir, det_tag)

                det_model = None
                if not det_use_api:
                    det_model = _get_model(det_model_path)

                det_result = run_grounding(
                    image_caption_pairs=pairs if det_is_primary else ab_pairs,
                    model_path=det_model_path or cfg.model_path,
                    backend=cfg.backend,
                    output_jsonl=det_out,
                    vis_dir=det_vis_dir,
                    visualize=det_is_primary,
                    filter_adjectives=cfg.filter_adjectives,
                    model=det_model,
                    use_api=det_use_api,
                    enable_verifier=cfg.enable_detection_verifier,
                    verifier_policy=cfg.detection_verifier_policy,
                    min_area_ratio=cfg.grounding_min_area_ratio,
                    max_area_ratio=cfg.grounding_max_area_ratio,
                    min_aspect_ratio=cfg.grounding_min_aspect_ratio,
                    max_aspect_ratio=cfg.grounding_max_aspect_ratio,
                    nms_iou_thresh=cfg.grounding_nms_iou_thresh,
                )
                detector_groundings[det_tag] = det_out
                detector_grounding_results[det_tag] = det_result

                # A/B 评测每个检测器的自洽指标
                if cfg.enable_eval:
                    eval_out = os.path.join(round_dir, f"eval_{det_tag}.json")
                    det_eval = evaluate_grounding(
                        grounding_jsonl=det_out,
                        model_path=det_model_path or cfg.model_path,
                        backend=cfg.backend,
                        iou_thresh=cfg.iou_thresh,
                        eval_report_path=eval_out,
                        model=det_model,
                        use_api=det_use_api,
                    )
                else:
                    det_eval = {"avg_iou": 0.0, "pass_rate": 0.0, "total_boxes": 0}
                detector_eval_results[det_tag] = det_eval

            if primary_tag not in detector_groundings:
                # 容错：未命中时回退到第一个检测器
                primary_tag = detector_entries[0]["tag"]

            grounding_result = detector_grounding_results[primary_tag]

            if cfg.enable_eval:
                compare = {
                    "primary": primary_tag,
                    "detectors": {},
                }
                for det in detector_entries:
                    tag = det["tag"]
                    g = detector_grounding_results.get(tag, {})
                    e = detector_eval_results.get(tag, {})
                    q = g.get("quality_stats", {})
                    compare["detectors"][tag] = {
                        "use_api": bool(det.get("use_api", False)),
                        "model_path": det.get("model_path"),
                        "grounding_count": g.get("count", 0),
                        "avg_iou": e.get("avg_iou", 0),
                        "pass_rate": e.get("pass_rate", 0),
                        "total_boxes": e.get("total_boxes", 0),
                        "avg_boxes_per_phrase": q.get("avg_boxes_per_phrase", 0),
                        "dropped_shape": q.get("dropped_shape", 0),
                        "dropped_nms": q.get("dropped_nms", 0),
                    }
                with open(os.path.join(round_dir, "detector_compare.json"), "w", encoding="utf-8") as f:
                    json.dump(compare, f, ensure_ascii=False, indent=2)
            elif os.path.exists(detector_compare_path):
                os.remove(detector_compare_path)
        else:
            grounding_result = {"count": 0}
            os.makedirs(os.path.dirname(grounding_jsonl), exist_ok=True)
            open(grounding_jsonl, "w").close()
            if os.path.exists(detector_compare_path):
                os.remove(detector_compare_path)
        logger.info(f"[{round_tag}] Grounding: {grounding_result['count']} 条标注")
        step_timings["grounding"] = round(time.time() - t_step, 2)

        if cfg.enable_grounding and getattr(cfg, "enable_ontology_bootstrap", True):
            _step("③.1 词表候选", "抽取 observation / cluster...")
            t_step = time.time()
            try:
                from apps.data_flywheel.review.bootstrap import prepare_review_artifacts

                ontology_review_result = prepare_review_artifacts(
                    grounding_jsonl=grounding_jsonl,
                    round_dir=round_dir,
                    ontology_store_path=getattr(cfg, "ontology_store_path", ""),
                )
                ontology_review_result["enabled"] = True
                logger.info(
                    f"[{round_tag}] Ontology Review: {ontology_review_result['observations']['count']} observations, "
                    f"{ontology_review_result['clusters']['count']} candidate clusters"
                )
            except Exception as exc:
                ontology_review_result = {"enabled": True, "error": str(exc)}
                logger.warning(f"[{round_tag}] Ontology Review 产物生成失败: {exc}")
            step_timings["ontology_review"] = round(time.time() - t_step, 2)

        primary_detector = next((d for d in detector_entries if d["tag"] == primary_tag), detector_entries[0])
        downstream_use_api = bool(primary_detector.get("use_api", use_api))
        shared_model = None
        if not downstream_use_api and primary_detector.get("model_path"):
            shared_model = _get_model(primary_detector["model_path"])

        # ---- Step 4 + 5: Referring + Pointing 并行 ----
        _step("④⑤ Referring+Pointing", "并行处理...")
        t_step = time.time()
        logger.info(f"[{round_tag}] Step4+5: Referring + Pointing（并行）")
        referring_jsonl = os.path.join(cfg.output_root, round_tag, "referring.jsonl")
        pointing_jsonl = os.path.join(cfg.output_root, round_tag, "pointing.jsonl")

        if cfg.enable_referring or cfg.enable_pointing:
            from concurrent.futures import ThreadPoolExecutor

            def _run_referring():
                if not cfg.enable_referring:
                    os.makedirs(os.path.dirname(referring_jsonl), exist_ok=True)
                    open(referring_jsonl, "w").close()
                    return {"count": 0}
                return run_referring(
                    grounding_jsonl=grounding_jsonl,
                    model_path=primary_detector.get("model_path") or cfg.model_path,
                    backend=cfg.backend,
                    output_jsonl=referring_jsonl,
                    model=shared_model,
                    use_api=downstream_use_api,
                )

            def _run_pointing():
                if not cfg.enable_pointing:
                    os.makedirs(os.path.dirname(pointing_jsonl), exist_ok=True)
                    open(pointing_jsonl, "w").close()
                    return {"count": 0}
                return run_pointing(
                    grounding_jsonl=grounding_jsonl,
                    output_jsonl=pointing_jsonl,
                    sam_checkpoint=cfg.sam_checkpoint,
                    sam_model_type=cfg.sam_model_type,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_ref = pool.submit(_run_referring)
                fut_pt = pool.submit(_run_pointing)
                referring_result = fut_ref.result()
                pointing_result = fut_pt.result()
        else:
            os.makedirs(os.path.dirname(referring_jsonl), exist_ok=True)
            open(referring_jsonl, "w").close()
            open(pointing_jsonl, "w").close()
            referring_result = {"count": 0}
            pointing_result = {"count": 0}

        logger.info(f"[{round_tag}] Referring: {referring_result['count']} 条标注")
        logger.info(f"[{round_tag}] Pointing: {pointing_result['count']} 条标注")
        step_timings["referring_pointing"] = round(time.time() - t_step, 2)

        # ---- Step 5.5: Scene Graph + Spatial QA ----
        scene_graph_jsonl = None
        spatial_qa_jsonl = None
        depth_info_jsonl = None
        scene_graph_result = {"count": 0}
        spatial_qa_result = {"count": 0, "type_counts": {}}
        depth_result = {"count": 0, "success": 0, "failed": 0}
        training_result: Dict[str, Any] = {
            "count": 0,
            "instructions": 0,
            "plan_count": 0,
            "sample_count": 0,
            "rejected_count": 0,
            "training_samples_path": "",
            "rejected_samples_path": "",
        }
        training_artifacts_by_image: Dict[str, Dict[str, Any]] = {}

        if cfg.enable_scene_graph:
            # ---- Step 3.5: 深度估计 (可选) ----
            if cfg.enable_depth_estimation:
                _step("③.5 深度估计", "FFS API...")
                t_step = time.time()
                logger.info(f"[{round_tag}] Step3.5: 深度估计 (FFS API)")
                from apps.data_flywheel.depth_client import estimate_depth_batch

                depth_info_jsonl = os.path.join(cfg.output_root, round_tag, "depth_info.jsonl")
                depth_result = estimate_depth_batch(
                    grounding_jsonl=grounding_jsonl,
                    output_jsonl=depth_info_jsonl,
                    server_url=cfg.ffs_server_url,
                    fx=cfg.default_fx, fy=cfg.default_fy,
                    cx=cfg.default_cx, cy=cfg.default_cy,
                    timeout=cfg.ffs_timeout,
                    max_workers=cfg.ffs_max_workers,
                )
                logger.info(f"[{round_tag}] 深度估计: {depth_result['count']} 张图, "
                            f"成功 {depth_result['success']}, 失败 {depth_result['failed']}")
                step_timings["depth"] = round(time.time() - t_step, 2)

            _step("③.6 场景图", "构建中...")
            t_step = time.time()
            logger.info(f"[{round_tag}] Step5.5: Scene Graph + Spatial QA")
            from apps.data_flywheel.scene_understanding.builder import (
                build_scene_graphs_from_grounding,
            )
            from apps.data_flywheel.spatial_qa import generate_spatial_qa

            scene_graph_jsonl = os.path.join(cfg.output_root, round_tag, "scene_graph.jsonl")
            scene_graph_result = build_scene_graphs_from_grounding(
                grounding_jsonl=grounding_jsonl,
                output_jsonl=scene_graph_jsonl,
                depth_info_jsonl=depth_info_jsonl,
                enable_3d=cfg.enable_3d_relations,
                ontology_store_path=cfg.ontology_store_path if getattr(cfg, "enable_ontology_runtime_matching", True) else None,
                enable_mask_segmentation=bool(getattr(cfg, "sam3_enabled", False)),
                sam3_url=getattr(cfg, "sam3_url", "YOUR_SAM3_URL"),
                mask_output_dir=getattr(cfg, "mask_output_dir", "data/output/masks"),
            )
            logger.info(f"[{round_tag}] Scene Graph: {scene_graph_result['count']} 张图, "
                        f"{scene_graph_result.get('total_objects', 0)} 物体, "
                        f"{scene_graph_result.get('total_relations', 0)} 关系, "
                        f"来源: {scene_graph_result.get('source_counts', {})}")
            step_timings["scene_graph"] = round(time.time() - t_step, 2)

            _step("③.7 空间QA", "生成中...")
            t_step = time.time()
            spatial_qa_jsonl = os.path.join(cfg.output_root, round_tag, "spatial_qa.jsonl")
            spatial_qa_result = generate_spatial_qa(
                scene_graph_jsonl=scene_graph_jsonl,
                output_jsonl=spatial_qa_jsonl,
                max_qa_per_image=cfg.max_qa_per_image,
                include_negatives=cfg.include_negative_qa,
                negative_ratio=cfg.negative_qa_ratio,
                enable_thinking=cfg.enable_thinking_chain,
                lang=cfg.qa_language,
                enable_bbox_single=cfg.enable_bbox_single_qa,
                enable_bbox_relational=cfg.enable_bbox_relational_qa,
                enable_point_single=cfg.enable_point_single_qa,
                enable_point_relational=cfg.enable_point_relational_qa,
                enable_verifier=cfg.enable_qa_verifier,
                verify_iou_thresh=cfg.qa_verify_iou_thresh,
                verify_point_margin=cfg.qa_verify_point_margin,
            )
            logger.info(f"[{round_tag}] Spatial QA: {spatial_qa_result['count']} 条, "
                        f"类型: {spatial_qa_result.get('type_counts', {})}")
            step_timings["spatial_qa"] = round(time.time() - t_step, 2)

        if scene_graph_jsonl:
            _step("⑥ M4 训练样本", "规划/合成/校验...")
            t_step = time.time()
            logger.info(f"[{round_tag}] Step6: M4 训练数据生成")
            training_result = _run_training_data_for_round(
                scene_graph_jsonl=scene_graph_jsonl,
                round_dir=round_dir,
                ontology_store_path=cfg.ontology_store_path if getattr(cfg, "enable_ontology_runtime_matching", True) else "",
            )
            training_artifacts_by_image = dict(training_result.pop("training_artifacts_by_image"))
            logger.info(
                f"[{round_tag}] Training Data: {training_result['plan_count']} plans, "
                f"{training_result['sample_count']} passed samples, "
                f"{training_result['rejected_count']} rejected"
            )
            step_timings["training_data"] = round(time.time() - t_step, 2)

        # ---- Step 6.5: 导出 ----
        _step("⑥.5 导出", "合并数据...")
        t_step = time.time()
        logger.info(f"[{round_tag}] Step6.5: 数据集导出")
        export_jsonl = os.path.join(cfg.output_root, round_tag, "train_export.jsonl")
        export_result = export_dataset(
            grounding_jsonl=grounding_jsonl,
            referring_jsonl=referring_jsonl,
            pointing_jsonl=pointing_jsonl,
            output_jsonl=export_jsonl,
            spatial_qa_jsonl=spatial_qa_jsonl,
            scene_graph_jsonl=scene_graph_jsonl,
            keep_only_verified=cfg.keep_only_verified,
        )
        logger.info(f"[{round_tag}] 导出: {export_result['total']} 条")
        step_timings["export"] = round(time.time() - t_step, 2)

        # ---- Step 7: 评估 ----
        eval_report_path = os.path.join(cfg.output_root, round_tag, "eval_report.json")
        if cfg.enable_eval:
            _step("⑦ 评估", "自洽检验...")
            t_step = time.time()
            logger.info(f"[{round_tag}] Step7: 评估" + (" (API)" if downstream_use_api else ""))
            # 如果第3步已做 A/B 评测，则主检测器结果直接复用
            eval_result = detector_eval_results.get(primary_tag)
            if eval_result is None:
                eval_result = evaluate_grounding(
                    grounding_jsonl=grounding_jsonl,
                    model_path=primary_detector.get("model_path") or cfg.model_path,
                    backend=cfg.backend,
                    iou_thresh=cfg.iou_thresh,
                    eval_report_path=eval_report_path,
                    model=shared_model,
                    use_api=downstream_use_api,
                )
            else:
                with open(eval_report_path, "w", encoding="utf-8") as f:
                    json.dump(eval_result, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[{round_tag}] 评估: avg_iou={eval_result['avg_iou']}, "
                f"pass_rate={eval_result['pass_rate']}"
            )
            step_timings["eval"] = round(time.time() - t_step, 2)
        else:
            eval_result = {"avg_iou": 0.0, "pass_rate": 0.0, "total_boxes": 0}
            if os.path.exists(eval_report_path):
                os.remove(eval_report_path)
            logger.info(f"[{round_tag}] Step7: 评估已跳过")

        # ---- Step 8: 难例回流 ----
        hard_jsonl = os.path.join(cfg.output_root, round_tag, "hard_samples.jsonl")
        if cfg.enable_hard_mining and cfg.enable_eval:
            _step("⑧ 难例挖掘", "分析中...")
            t_step = time.time()
            logger.info(f"[{round_tag}] Step8: 难例挖掘")
            hard_result = mine_hard_samples(
                eval_report_path=eval_report_path,
                hard_iou_thresh=cfg.hard_sample_iou_thresh,
                output_jsonl=hard_jsonl,
                scene_graph_jsonl=scene_graph_jsonl,
                depth_info_jsonl=depth_info_jsonl,
            )
            logger.info(f"[{round_tag}] 难例: {hard_result['hard_count']} 条")
            step_timings["hard_mining"] = round(time.time() - t_step, 2)
        else:
            os.makedirs(os.path.dirname(hard_jsonl), exist_ok=True)
            open(hard_jsonl, "w").close()
            hard_result = {"hard_count": 0}
            logger.info(f"[{round_tag}] Step8: 难例挖掘已跳过")

        # ---- Step 9: Visualization report generation (non-fatal) ----
        report_paths: List[str] = []
        if scene_graph_jsonl:
            try:
                report_paths = _generate_round_reports(
                    grounding_jsonl=grounding_jsonl,
                    scene_graph_jsonl=scene_graph_jsonl,
                    training_artifacts_by_image=training_artifacts_by_image,
                )
                logger.info(f"[{round_tag}] Reports: generated {len(report_paths)} HTML files")
            except Exception as exc:
                logger.warning(f"[{round_tag}] Report generation failed (non-fatal): {exc}")

        # 打印本轮耗时汇总
        round_elapsed = sum(step_timings.values())
        logger.info(f"[{round_tag}] 耗时汇总 (总计 {round_elapsed:.1f}s):")
        for sname, stime in step_timings.items():
            logger.info(f"  {sname:25s} {stime:6.2f}s")
        _step("完成", f"Round {round_idx} 完成，耗时 {round_elapsed:.1f}s")

        # 将难例图片加入下一轮的输入
        if round_idx < cfg.rounds - 1:
            hard_paths = get_hard_image_paths(hard_jsonl)
            if hard_paths:
                hard_dir = os.path.join(cfg.output_root, round_tag, "hard_images")
                os.makedirs(hard_dir, exist_ok=True)
                for hp in hard_paths:
                    abs_hp = hp if os.path.isabs(hp) else os.path.join(cfg.output_root, hp)
                    link_name = os.path.join(hard_dir, os.path.basename(abs_hp))
                    if not os.path.exists(link_name):
                        try:
                            os.symlink(os.path.abspath(abs_hp), link_name)
                        except OSError:
                            pass
                cfg.image_dirs.append(hard_dir)
                logger.info(f"[{round_tag}] 难例图片 {len(hard_paths)} 张加入下一轮")

        # ---- 路径转换 ----
        if cfg.relative_paths:
            from apps.data_flywheel.config import PROJECT_ROOT
            base = str(PROJECT_ROOT)
            logger.info(f"[{round_tag}] 转换为相对路径 (base={base})")
            _relativize_jsonl(grounding_jsonl, base)
            _relativize_jsonl(referring_jsonl, base)
            _relativize_jsonl(pointing_jsonl, base)
            _relativize_jsonl(export_jsonl, base)
            _relativize_jsonl(hard_jsonl, base)
            _relativize_eval_report(eval_report_path, base)
            if scene_graph_jsonl:
                _relativize_jsonl(scene_graph_jsonl, base)
            if spatial_qa_jsonl:
                _relativize_jsonl(spatial_qa_jsonl, base)
            if depth_info_jsonl:
                _relativize_jsonl(depth_info_jsonl, base)
            if training_result.get("training_samples_path"):
                _relativize_jsonl(
                    training_result["training_samples_path"],
                    base,
                    image_keys=("image_path",),
                )
            if training_result.get("rejected_samples_path"):
                _relativize_jsonl(
                    training_result["rejected_samples_path"],
                    base,
                    image_keys=("image_path",),
                )

        round_report = {
            "round": round_idx,
            "images": len(all_images),
            "caption_backend": caption_backend,
            "use_api": use_api,
            "detector_primary": primary_tag,
            "detector_models": detector_entries,
            "grounding": grounding_result,
            "detector_grounding": detector_grounding_results,
            "detector_eval": detector_eval_results,
            "ontology_review": ontology_review_result,
            "referring": referring_result,
            "pointing": pointing_result,
            "depth_estimation": depth_result,
            "scene_graph": scene_graph_result,
            "spatial_qa": spatial_qa_result,
            "training_data": training_result,
            "export": export_result,
            "eval": {
                "avg_iou": eval_result.get("avg_iou", 0.0),
                "pass_rate": eval_result.get("pass_rate", 0.0),
                "total_boxes": eval_result.get("total_boxes", 0),
            },
            "hard_mining": hard_result,
            "reports": report_paths,
            "step_timings": step_timings,
        }
        all_reports.append(round_report)

    elapsed = time.time() - t0
    summary = {
        "rounds": len(all_reports),
        "elapsed_seconds": round(elapsed, 1),
        "use_api": use_api,
        "reports": all_reports,
    }

    summary_path = os.path.join(cfg.output_root, "flywheel_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"飞轮完成，总报告: {summary_path}")

    return summary


# =====================================================================
# v3.0 适配层 — 新增 CLI 标志和 orchestrator 调用入口
# =====================================================================

def run_v3_pipeline(
    source_paths: list = None,
    source_type: str = "image_dir",
    source_config: str = None,
    capabilities: str = "G,S,A,R,P,F,H,E,Aff",
    num_gpus: int = None,
    no_ray: bool = False,
    output_root: str = "data/output",
    run_id: str = None,
    **kwargs,
):
    """v3.0 飞轮入口 — 能力维度驱动 + 多源 + 多卡并行。

    向后兼容: 如果仅传 source_paths + image_dir 类型，行为等价于 v2.1。
    """
    from apps.data_flywheel.sources import get_source
    from apps.data_flywheel.sources.multi_path import MultiPathSource, SourcesConfig
    from apps.data_flywheel.orchestrator.runner import RunConfig, run
    from apps.data_flywheel.workers.base import CapabilityTag

    if source_config:
        config = SourcesConfig.from_yaml(source_config)
        source = MultiPathSource(config)
    elif source_paths:
        source = get_source(source_type, paths=source_paths, **kwargs)
    else:
        source = get_source("image_dir", paths=["./raw_data/test/"])

    cap_set = set()
    for c in capabilities.split(","):
        c = c.strip()
        if c:
            try:
                cap_set.add(CapabilityTag(c))
            except ValueError:
                logger.warning(f"Unknown capability tag: {c}")

    scheduler_backend = "sequential" if no_ray else "auto"
    run_config = RunConfig(
        capabilities=cap_set,
        scheduler_backend=scheduler_backend,
        num_gpus=num_gpus,
        output_root=output_root,
        run_id=run_id,
    )

    return run(source, run_config)


def main() -> None:
    """CLI 入口。"""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Rex-Embodied 数据飞轮")
    parser.add_argument("--image-dir", type=str, default=None, help="图片目录")
    parser.add_argument("--rounds", type=int, default=1, help="飞轮轮数")
    parser.add_argument("--max-images", type=int, default=50, help="每轮最大图片数")
    parser.add_argument("--augment", type=int, default=2, help="每张图增强数")
    parser.add_argument("--model-path", type=str, default="IDEA-Research/Rex-Omni")
    parser.add_argument("--detector-models", type=str, default="",
                        help="检测器列表，逗号分隔。可混合本地路径和 'api'")
    parser.add_argument("--detector-primary", type=str, default="",
                        help="主检测器 tag（为空则使用第一个）")
    parser.add_argument("--detector-verifier-policy", type=str, default="drop",
                        choices=["drop", "rescore", "flag"], help="检测框校验策略")
    parser.add_argument("--keep-only-verified", action="store_true",
                        help="导出时仅保留 verified 的空间 QA")
    parser.add_argument("--backend", type=str, default="transformers")
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--caption-backend", type=str, default="auto",
                        choices=["auto", "dashscope", "openai", "anthropic", "local"],
                        help="Caption 生成后端 (auto=自动选最快可用API)")
    parser.add_argument("--use-api", action="store_true",
                        help="全 API 模式：Caption + 检测 + 评估全部用 VLM API，无需加载本地模型")
    parser.add_argument("--no-depth", action="store_true",
                        help="禁用深度估计（仅用 2D 规则推理空间关系）")
    parser.add_argument("--no-3d", action="store_true",
                        help="禁用 3D 空间关系（仅用 2D fallback）")
    parser.add_argument("--no-scene-graph", action="store_true",
                        help="禁用场景图构建和空间 QA 生成")
    parser.add_argument("--no-thinking", action="store_true",
                        help="QA 中不包含思维链")
    parser.add_argument("--qa-lang", type=str, default="en",
                        choices=["en", "zh"], help="QA 语言")
    parser.add_argument("--ffs-url", type=str, default=None,
                        help="FFS 深度估计服务 URL")
    parser.add_argument("--output-root", type=str, default=None)
    # ---- v3.0 新增标志 ----
    parser.add_argument("--capabilities", type=str, default="G,S,A,R,P,F,H,E,Aff",
                        help="启用的能力维度，逗号分隔 (默认全开)")
    parser.add_argument("--source-type", type=str, default="image_dir",
                        choices=["image_dir", "image_seq", "video", "rgbd", "hf", "multi"],
                        help="数据源类型")
    parser.add_argument("--source-paths", type=str, default=None,
                        help="数据源路径，逗号分隔")
    parser.add_argument("--source-config", type=str, default=None,
                        help="数据源 YAML 配置文件路径")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="使用的 GPU 数量 (默认全部可见)")
    parser.add_argument("--no-ray", action="store_true",
                        help="强制降级模式，不使用 Ray")
    parser.add_argument("--v3", action="store_true",
                        help="使用 v3.0 能力维度驱动模式")
    args = parser.parse_args()

    cfg = FlywheelConfig()
    if args.image_dir:
        cfg.image_dirs = [args.image_dir]
    if args.output_root:
        cfg.output_root = args.output_root
    cfg.rounds = args.rounds
    cfg.max_images_per_round = args.max_images
    cfg.augment_per_image = args.augment
    cfg.model_path = args.model_path
    if args.detector_models.strip():
        cfg.detector_models = [x.strip() for x in args.detector_models.split(",") if x.strip()]
    cfg.detector_primary = args.detector_primary.strip()
    cfg.detection_verifier_policy = args.detector_verifier_policy
    cfg.keep_only_verified = args.keep_only_verified
    cfg.backend = args.backend
    cfg.sam_checkpoint = args.sam_checkpoint
    cfg.caption_backend = args.caption_backend
    cfg.use_api = args.use_api
    if args.no_depth:
        cfg.enable_depth_estimation = False
    if args.no_3d:
        cfg.enable_3d_relations = False
    if args.no_scene_graph:
        cfg.enable_scene_graph = False
    if args.no_thinking:
        cfg.enable_thinking_chain = False
    cfg.qa_language = args.qa_lang
    if args.ffs_url:
        cfg.ffs_server_url = args.ffs_url

    # v3.0 模式: 使用能力维度驱动
    if args.v3:
        source_paths = args.source_paths.split(",") if args.source_paths else (
            [args.image_dir] if args.image_dir else None
        )
        result = run_v3_pipeline(
            source_paths=source_paths,
            source_type=args.source_type,
            source_config=args.source_config,
            capabilities=args.capabilities,
            num_gpus=args.num_gpus,
            no_ray=args.no_ray,
            output_root=args.output_root or cfg.output_root,
        )
        print(json.dumps({
            "mode": "v3",
            "run_id": result.run_id,
            "total_samples": result.total_samples,
            "total_annotations": result.total_annotations,
            "per_capability": result.per_capability,
            "elapsed_seconds": result.elapsed_seconds,
        }, ensure_ascii=False, indent=2))
        return

    result = run_pipeline(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

