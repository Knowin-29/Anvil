import copy
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from apps.data_flywheel.geometry import bbox_iou
from apps.data_flywheel.ontology.matcher import RuntimeInstanceContext
from apps.data_flywheel.ontology.store import OntologyStore, resolve_store_path
from apps.data_flywheel.scene_understanding.blocking import (
    derive_blocking_dag,
    populate_node_actionability,
)
from apps.data_flywheel.scene_understanding.enrich.color import extract_color_attributes
from apps.data_flywheel.scene_understanding.enrich.hierarchical import generate_hierarchical_name
from apps.data_flywheel.scene_understanding.pointcloud import (
    extract_masked_point_cloud_sample,
    extract_masked_xyz_point_cloud_sample,
    load_xyz_map,
    point_cloud_pose_from_sample,
    polygon_from_mask_file,
)
from apps.data_flywheel.scene_understanding.placement import compute_placement_regions
from apps.data_flywheel.scene_understanding.enrich.referring import generate_referring_expressions
from apps.data_flywheel.scene_understanding.enrich.shape import extract_shape_hint, extract_size
from apps.data_flywheel.scene_understanding.fusion import fuse_edges
from apps.data_flywheel.scene_understanding.support import compute_support_relations
from apps.data_flywheel.scene_understanding.relations_2d import compute_2d_relations
from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations
from apps.data_flywheel.scene_understanding.relations_round2 import (
    compute_between_relations,
    compute_round2_binary_relations,
)
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds
from apps.data_flywheel.scene_understanding.vlm.backend import VLMBackend
from apps.data_flywheel.types import (
    Detection,
    InstanceAttributes,
    SceneGraph,
    SceneNode,
    Segmentation,
)

_LOGGER = logging.getLogger("flywheel.scene_understanding.builder")


def _allocate_ids(segmentations: Sequence[Segmentation]) -> List[str]:
    counts: Dict[str, int] = {}
    ids: List[str] = []
    for seg in segmentations:
        class_slug = re.sub(r"[^\w]+", "_", seg.detection.class_name)
        idx = counts.get(class_slug, 0)
        ids.append(f"{class_slug}_{idx}")
        counts[class_slug] = idx + 1
    return ids


def _safe_mask_image_id(image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0] or "scene"
    safe = re.sub(r"[^\w]+", "_", stem).strip("_")
    return safe or "scene"


def _enrich_box_with_sam_mask(
    image: Image.Image,
    image_path: str,
    phrase: str,
    box_data: Dict[str, Any],
    det_idx: int,
    segmenter: Optional[Any],
) -> Dict[str, Any]:
    if segmenter is None:
        return dict(box_data)

    bbox = box_data.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return dict(box_data)

    try:
        segmentations = segmenter.segment(
            image,
            bbox=[float(coord) for coord in bbox[:4]],
            image_id=_safe_mask_image_id(image_path),
            class_name=phrase,
            det_idx=det_idx,
        )
    except Exception as exc:
        _LOGGER.warning("SAM3 mask segmentation failed for %s[%s]: %s", image_path, phrase, exc)
        return dict(box_data)

    if not segmentations:
        return dict(box_data)

    segmentation = segmentations[0]
    enriched = dict(box_data)
    enriched["bbox"] = list(segmentation.detection.bbox)
    enriched["source"] = segmentation.detection.source or box_data.get("source", "grounding")
    if segmentation.mask_polygon:
        enriched["mask_polygon"] = [list(point) for point in segmentation.mask_polygon]
    if segmentation.mask_file:
        enriched["mask_file"] = segmentation.mask_file
    if segmentation.grasp_point is not None:
        enriched["grasp_point"] = list(segmentation.grasp_point)
    return enriched


def _segmentations_from_grounding_item(
    item: Dict[str, Any],
    image: Image.Image,
    segmenter: Optional[Any],
) -> List[Segmentation]:
    image_path = str(item.get("image", ""))
    class_counts: Dict[str, int] = {}
    segmentations: List[Segmentation] = []
    for phrase_data in item.get("phrases", []):
        phrase = str(phrase_data.get("phrase", ""))
        for box_data in phrase_data.get("boxes", []):
            det_idx = class_counts.get(phrase, 0)
            class_counts[phrase] = det_idx + 1
            enriched_box = _enrich_box_with_sam_mask(
                image=image,
                image_path=image_path,
                phrase=phrase,
                box_data=box_data,
                det_idx=det_idx,
                segmenter=segmenter,
            )
            segmentations.append(_segmentation_from_box(phrase, enriched_box))
    return segmentations


def _match_depth_object(
    bbox: List[float],
    depth_objects: Sequence[Dict[str, Any]],
    used_indices: set[int],
    iou_threshold: float = 0.8,
) -> Optional[Dict[str, Any]]:
    best_idx: Optional[int] = None
    best_iou = 0.0
    for idx, depth_obj in enumerate(depth_objects):
        if idx in used_indices:
            continue
        if not isinstance(depth_obj, dict):
            continue
        depth_bbox = depth_obj.get("bbox")
        if not isinstance(depth_bbox, list) or len(depth_bbox) != 4:
            continue
        if _coerce_depth_payload(depth_obj) is None:
            continue
        iou = bbox_iou(bbox, depth_bbox)
        if iou > iou_threshold and iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_idx is None:
        return None
    used_indices.add(best_idx)
    return depth_objects[best_idx]


def _coerce_vector3(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    vector: List[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        vector.append(float(item))
    return vector


def _coerce_float(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _coerce_depth_payload(
    depth_obj: Optional[Dict[str, Any]],
) -> Optional[Tuple[List[float], List[float], float]]:
    if not depth_obj or not depth_obj.get("valid"):
        return None
    candidate_center = _coerce_vector3(depth_obj.get("center_3d"))
    candidate_size = _coerce_vector3(depth_obj.get("size_3d"))
    candidate_depth = _coerce_float(depth_obj.get("mean_depth"))
    if (
        candidate_center is None
        or candidate_size is None
        or candidate_depth is None
    ):
        return None
    return candidate_center, candidate_size, candidate_depth


def _estimate_max_dimension_cm(size_3d: Optional[List[float]]) -> Optional[float]:
    if not size_3d:
        return None
    dims = [abs(float(value)) for value in size_3d if value is not None]
    if not dims:
        return None
    max_dim = max(dims)
    if max_dim <= 2:
        return max_dim * 100.0
    if max_dim >= 10:
        return max_dim / 10.0
    return max_dim


def _build_runtime_context(
    bbox: List[float],
    mask_polygon: Optional[List[List[float]]],
    has_true_mask: bool,
    size_3d: Optional[List[float]],
    mean_depth: Optional[float],
    image_size: Optional[Tuple[int, int]],
) -> RuntimeInstanceContext:
    bbox_w = max(0.0, float(bbox[2]) - float(bbox[0]))
    bbox_h = max(0.0, float(bbox[3]) - float(bbox[1]))
    extras: Dict[str, Any] = {
        "bbox_width_px": bbox_w,
        "bbox_height_px": bbox_h,
        "bbox_max_side_px": max(bbox_w, bbox_h),
    }
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    if image_size is not None:
        image_width, image_height = image_size
        img_area = max(1.0, float(image_width) * float(image_height))
        extras["bbox_area_ratio"] = (bbox_w * bbox_h) / img_area
        extras["bbox_max_side_ratio"] = max(
            bbox_w / max(1.0, float(image_width)),
            bbox_h / max(1.0, float(image_height)),
        )
    return RuntimeInstanceContext(
        bbox=list(bbox),
        image_width=image_width,
        image_height=image_height,
        estimated_max_dimension_cm=_estimate_max_dimension_cm(size_3d),
        mean_depth=mean_depth,
        mask_polygon=[list(point) for point in mask_polygon] if mask_polygon and has_true_mask else None,
        extras=extras,
    )


def _build_nodes(
    image_id: str,
    segmentations: List[Segmentation],
    ontology: OntologyStore,
    depth_info: Optional[Dict[str, Any]],
    image_size: Optional[Tuple[int, int]] = None,
) -> List[SceneNode]:
    del image_id  # Reserved for later Task assembly; not needed in Task 2 node build.

    depth_objects: List[Dict[str, Any]] = []
    if depth_info and depth_info.get("depth_valid"):
        depth_objects = list(depth_info.get("objects") or [])
    xyz_map = load_xyz_map(depth_info)
    dense_depth_map = depth_info.get("depth_map") if depth_info else None
    dense_intrinsics = depth_info.get("intrinsics") if depth_info else None

    used_depth_indices: set[int] = set()
    ids = _allocate_ids(segmentations)
    nodes: List[SceneNode] = []

    for node_id, seg in zip(ids, segmentations):
        detection = seg.detection
        matched_depth = _match_depth_object(
            bbox=detection.bbox,
            depth_objects=depth_objects,
            used_indices=used_depth_indices,
        )
        center_3d: Optional[List[float]] = None
        size_3d: Optional[List[float]] = None
        mean_depth: Optional[float] = None
        depth_valid = False
        depth_payload = _coerce_depth_payload(matched_depth)
        if depth_payload is not None:
            center_3d, size_3d, mean_depth = depth_payload
            depth_valid = True

        runtime = ontology.resolve_observation_name(
            detection.class_name,
            context=_build_runtime_context(
                bbox=detection.bbox,
                mask_polygon=seg.mask_polygon,
                has_true_mask=seg.mask_source in {"mask_polygon", "mask_file"},
                size_3d=size_3d,
                mean_depth=mean_depth,
                image_size=image_size,
            ),
        )
        object_class = (
            ontology.get_class(runtime.object_class_id)
            if runtime.object_class_id is not None
            else None
        )
        descriptor_axes = (
            list(object_class.descriptor_schema.priority)
            if object_class is not None
            else []
        )
        point_cloud_sample: List[List[float]] = []
        if xyz_map is not None:
            point_cloud_sample = extract_masked_xyz_point_cloud_sample(
                xyz_map=xyz_map,
                mask_polygon=seg.mask_polygon,
                source_image_size=image_size,
            )
        elif (
            dense_depth_map is not None
            and isinstance(dense_intrinsics, dict)
        ):
            point_cloud_sample = extract_masked_point_cloud_sample(
                depth_map=dense_depth_map,
                mask_polygon=seg.mask_polygon,
                intrinsics=dense_intrinsics,
                source_image_size=image_size,
            )
        if not depth_valid:
            sample_pose = point_cloud_pose_from_sample(point_cloud_sample)
            if sample_pose is not None:
                center_3d, size_3d, mean_depth = sample_pose
                depth_valid = True
        is_container = (
            bool(runtime.is_container) if runtime.is_container is not None else False
        )
        stable_class_name = (
            runtime.object_class_id
            if runtime.object_class_id is not None
            else detection.class_name
        )
        has_true_mask = seg.mask_source in {"mask_polygon", "mask_file"}

        nodes.append(
            SceneNode(
                id=node_id,
                class_name=stable_class_name,
                bbox=list(detection.bbox),
                mask_polygon=[list(point) for point in seg.mask_polygon]
                if seg.mask_polygon is not None
                else None,
                mask_file=seg.mask_file,
                mask_source=seg.mask_source,
                has_true_mask=has_true_mask,
                grasp_point=list(seg.grasp_point) if seg.grasp_point is not None else None,
                center_3d=center_3d,
                size_3d=size_3d,
                mean_depth=mean_depth,
                valid_3d=depth_valid,
                operation=runtime.operation,
                role_ids=list(runtime.role_ids or []),
                allowed_operations=list(runtime.allowed_operations or []),
                preferred_operation=runtime.preferred_operation,
                is_container=is_container,
                container_placeability=runtime.container_placeability,
                placement_policy=runtime.placement_policy,
                extra_tags=list(runtime.extra_tags or []),
                canonical_name=runtime.canonical_name,
                object_class_id=runtime.object_class_id,
                ontology_status=runtime.ontology_status,
                triggered_constraints=list(runtime.triggered_constraints or []),
                descriptor_axes=descriptor_axes,
                point_cloud_sample=point_cloud_sample,
                hierarchical_name=None,
                referring_expressions=[],
                attributes=InstanceAttributes(),
            )
        )

    return nodes


def _validate_image_size(image_size: Tuple[int, int]) -> Tuple[int, int]:
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must be a tuple of (width, height)")
    width, height = image_size
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise ValueError("image_size must contain numeric width and height")
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive")
    return int(width), int(height)


def _validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance")
    return image


def _resolve_image_path(image_id: str, image: Optional[Image.Image]) -> str:
    if image is not None:
        image_path = getattr(image, "filename", "")
        if isinstance(image_path, str) and image_path:
            return image_path
    return image_id


def build_phase_a(
    image_id: str,
    image_size: Tuple[int, int],
    segmentations: List[Segmentation],
    ontology: OntologyStore,
    *,
    depth_info: Optional[Dict[str, Any]] = None,
    thresholds: Optional[SceneGraphThresholds] = None,
) -> SceneGraph:
    """Assemble Phase A: nodes, relations, fusion, ternary relations, blocking."""

    normalized_size = _validate_image_size(image_size)
    thresholds = thresholds or SceneGraphThresholds()

    nodes = _build_nodes(
        image_id=image_id,
        segmentations=segmentations,
        ontology=ontology,
        depth_info=depth_info,
        image_size=normalized_size,
    )

    support_relations = compute_support_relations(nodes)
    edges_2d = compute_2d_relations(nodes, normalized_size, thresholds)
    edges_3d = compute_3d_relations(nodes, normalized_size, thresholds)
    fused_edges = fuse_edges(edges_2d, edges_3d, strategy=thresholds.fusion_strategy)
    round2_edges = compute_round2_binary_relations(nodes, normalized_size, thresholds)
    ternary_relations = compute_between_relations(nodes, thresholds)
    placement_regions = compute_placement_regions(nodes, normalized_size)
    edges = [*fused_edges, *round2_edges]
    blocking_dag = derive_blocking_dag(nodes, edges)
    populate_node_actionability(nodes, blocking_dag)

    return SceneGraph(
        image_id=image_id,
        image_path=image_id,
        image_size=normalized_size,
        nodes=nodes,
        edges=edges,
        ternary_relations=ternary_relations,
        blocking_dag=blocking_dag,
        support_relations=support_relations,
        placement_regions=placement_regions,
    )


def _fallback_referring_expressions(node: SceneNode) -> List[str]:
    if node.hierarchical_name:
        return [node.hierarchical_name]
    return []


def enrich_phase_b(
    scene_graph: SceneGraph,
    image: Image.Image,
    vlm_backend: Optional[VLMBackend],
) -> SceneGraph:
    """Enrich Phase B fields on a new SceneGraph copy."""

    if not isinstance(scene_graph, SceneGraph):
        raise TypeError("scene_graph must be a SceneGraph instance")
    validated_image = _validate_image(image)

    enriched = copy.deepcopy(scene_graph)
    thresholds = SceneGraphThresholds()

    for node in enriched.nodes:
        color_attrs = extract_color_attributes(
            validated_image,
            node.mask_polygon,
            n_colors=thresholds.hsv_cluster_k,
            secondary_min_ratio=thresholds.hsv_secondary_min_ratio,
        )
        node.attributes.color_primary = color_attrs.get("color_primary")
        node.attributes.color_primary_hsv = color_attrs.get("color_primary_hsv")
        node.attributes.color_secondary = color_attrs.get("color_secondary")
        node.attributes.size = extract_size(node.bbox, enriched.image_size, thresholds)
        node.attributes.shape = extract_shape_hint(node.bbox)

    for node in enriched.nodes:
        node.hierarchical_name = generate_hierarchical_name(node, enriched.nodes)

    if vlm_backend is None:
        for node in enriched.nodes:
            node.referring_expressions = []
        return enriched

    for node in enriched.nodes:
        try:
            node.referring_expressions = generate_referring_expressions(
                node,
                enriched.nodes,
                validated_image,
                vlm_backend,
            )
        except Exception:
            node.referring_expressions = _fallback_referring_expressions(node)

    return enriched


def build_scene_graph(
    image_id: str,
    image: Image.Image,
    image_size: Tuple[int, int],
    segmentations: List[Segmentation],
    ontology: OntologyStore,
    *,
    depth_info: Optional[Dict[str, Any]] = None,
    vlm_backend: Optional[VLMBackend] = None,
    thresholds: Optional[SceneGraphThresholds] = None,
    skip_phase_b: bool = False,
) -> SceneGraph:
    """End-to-end build of a scene graph across Phase A and Phase B."""

    validated_image = _validate_image(image)
    phase_a = build_phase_a(
        image_id=image_id,
        image_size=image_size,
        segmentations=segmentations,
        ontology=ontology,
        depth_info=depth_info,
        thresholds=thresholds,
    )
    phase_a.image_path = _resolve_image_path(image_id, validated_image)
    if skip_phase_b:
        return phase_a
    return enrich_phase_b(phase_a, validated_image, vlm_backend)


def _bbox_polygon(bbox: Sequence[float]) -> List[List[float]]:
    return [
        [float(bbox[0]), float(bbox[1])],
        [float(bbox[2]), float(bbox[1])],
        [float(bbox[2]), float(bbox[3])],
        [float(bbox[0]), float(bbox[3])],
    ]


def _bbox_center_point(bbox: Sequence[float]) -> List[float]:
    return [
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ]


def _segmentation_from_box(phrase: str, box_data: Dict[str, Any]) -> Segmentation:
    bbox = list(box_data.get("bbox") or [])
    mask_polygon = box_data.get("mask_polygon")
    if isinstance(mask_polygon, list) and len(mask_polygon) >= 3:
        normalized_polygon = [
            [float(point[0]), float(point[1])]
            for point in mask_polygon
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(normalized_polygon) >= 3:
            return Segmentation(
                detection=Detection(
                    class_name=phrase,
                    bbox=bbox,
                    confidence=float(box_data.get("confidence", box_data.get("score", 1.0))),
                    source=str(box_data.get("source", "grounding")),
                ),
                mask_polygon=normalized_polygon,
                mask_file=str(box_data.get("mask_file", "")),
                grasp_point=list(box_data.get("grasp_point") or _bbox_center_point(bbox)),
                mask_source="mask_polygon",
            )

    mask_file = str(box_data.get("mask_file", ""))
    polygon_from_file = polygon_from_mask_file(mask_file)
    if polygon_from_file is not None:
        return Segmentation(
            detection=Detection(
                class_name=phrase,
                bbox=bbox,
                confidence=float(box_data.get("confidence", box_data.get("score", 1.0))),
                source=str(box_data.get("source", "grounding")),
            ),
            mask_polygon=polygon_from_file,
            mask_file=mask_file,
            grasp_point=list(box_data.get("grasp_point") or _bbox_center_point(bbox)),
            mask_source="mask_file",
        )

    return Segmentation(
        detection=Detection(
            class_name=phrase,
            bbox=bbox,
            confidence=float(box_data.get("confidence", box_data.get("score", 1.0))),
            source=str(box_data.get("source", "grounding")),
        ),
        mask_polygon=_bbox_polygon(bbox),
        mask_file=mask_file,
        grasp_point=list(box_data.get("grasp_point") or _bbox_center_point(bbox)),
        mask_source="bbox_fallback",
    )


def _load_image(image_path: str, image_size: Tuple[int, int]) -> Image.Image:
    try:
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        image.filename = image_path
        return image
    except Exception as exc:
        _LOGGER.warning("Falling back to blank image for %s: %s", image_path, exc)
        image = Image.new("RGB", image_size, color="white")
        image.filename = image_path
        return image


def _load_grounding_items(grounding_jsonl: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(grounding_jsonl, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                items.append(json.loads(raw))
    return items


def _load_depth_map(depth_info_jsonl: Optional[str]) -> Dict[str, Dict[str, Any]]:
    depth_map: Dict[str, Dict[str, Any]] = {}
    if not depth_info_jsonl or not os.path.exists(depth_info_jsonl):
        return depth_map
    with open(depth_info_jsonl, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            image_path = entry.get("image")
            if isinstance(image_path, str) and image_path:
                depth_map[image_path] = entry
    return depth_map


def _legacy_can_pick(node: SceneNode) -> bool:
    return node.operation == "pick"


def _legacy_can_push(node: SceneNode) -> bool:
    return (
        node.operation in {"pick", "push"}
        or node.placement_policy == "push_only"
        or "push_only" in node.extra_tags
    )


def _legacy_can_stack(node: SceneNode) -> bool:
    return "stackable_2d" in node.extra_tags or "planar" in node.extra_tags


def _legacy_is_surface(node: SceneNode) -> bool:
    return "planar" in node.extra_tags


def scene_graph_to_legacy_record(
    scene_graph: SceneGraph,
    *,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    id_to_index = {node.id: idx for idx, node in enumerate(scene_graph.nodes)}
    objects: List[Dict[str, Any]] = []
    for idx, node in enumerate(scene_graph.nodes):
        can_stack = _legacy_can_stack(node)
        objects.append(
            {
                "id": idx,
                "node_id": node.id,
                "name": node.class_name,
                "category": node.class_name,
                "bbox": list(node.bbox),
                "confidence": 1.0,
                "mask_polygon": [list(point) for point in (node.mask_polygon or [])],
                "mask_file": node.mask_file,
                "mask_source": node.mask_source,
                "has_true_mask": bool(node.has_true_mask),
                "grasp_point": list(node.grasp_point) if node.grasp_point is not None else None,
                "center_3d": list(node.center_3d) if node.center_3d is not None else None,
                "size_3d": list(node.size_3d) if node.size_3d is not None else None,
                "mean_depth": node.mean_depth,
                "valid": bool(node.valid_3d),
                "can_pick": _legacy_can_pick(node),
                "can_push": _legacy_can_push(node),
                "can_stack": can_stack,
                "is_container": bool(node.is_container),
                "is_stackable": can_stack,
                "is_surface": _legacy_is_surface(node),
                "operation": node.operation,
                "role_ids": list(node.role_ids),
                "allowed_operations": list(node.allowed_operations),
                "preferred_operation": node.preferred_operation,
                "canonical_name": node.canonical_name or node.class_name,
                "object_class_id": node.object_class_id,
                "object_subtype_id": None,
                "matched_alias": None,
                "ontology_status": node.ontology_status,
                "descriptor_axes": list(node.descriptor_axes),
                "point_cloud_sample": [list(point) for point in node.point_cloud_sample],
                "support_parent": node.support_parent,
                "support_type": node.support_type,
                "support_excluded": bool(node.support_excluded),
                "directly_reachable": node.directly_reachable,
                "blocked_by": list(node.blocked_by),
                "blocking": list(node.blocking),
                "excluded_from_action_graph": bool(node.excluded_from_action_graph),
                "placement": {
                    "container_placeability": node.container_placeability,
                    "placement_policy": node.placement_policy,
                    "default_container_types": [],
                    "placement_modes": [node.placement_policy] if node.placement_policy else [],
                },
                "perception": {"extra_tags": list(node.extra_tags)},
                "triggered_constraints": list(node.triggered_constraints),
                "hierarchical_name": node.hierarchical_name,
                "referring_expressions": list(node.referring_expressions),
                "attributes": {
                    "color_primary": node.attributes.color_primary,
                    "color_primary_hsv": list(node.attributes.color_primary_hsv)
                    if node.attributes.color_primary_hsv is not None
                    else None,
                    "color_secondary": node.attributes.color_secondary,
                    "size": node.attributes.size,
                    "shape": node.attributes.shape,
                },
            }
        )

    relations: List[Dict[str, Any]] = []
    for edge in scene_graph.edges:
        subject = id_to_index.get(edge.subject)
        obj = id_to_index.get(edge.object)
        if subject is None or obj is None:
            continue
        relations.append(
            {
                "subject": subject,
                "object": obj,
                "predicate": edge.predicate,
                "source": edge.source,
                "confidence": edge.confidence,
                "frame": edge.frame,
            }
        )

    return {
        "image": image_path or scene_graph.image_path or scene_graph.image_id,
        "image_size": list(scene_graph.image_size),
        "objects": objects,
        "relations": relations,
        "support_relations": [
            {
                "subject": id_to_index.get(rel.subject, rel.subject),
                "object": id_to_index.get(rel.object, rel.object),
                "relation": rel.relation,
                "confidence": rel.confidence,
            }
            for rel in scene_graph.support_relations
            if rel.subject in id_to_index and rel.object in id_to_index
        ],
        "blocking_dag": [[blocker, blocked] for blocker, blocked in scene_graph.blocking_dag],
        "placement_regions": [
            {
                "region_id": region.region_id,
                "region_type": region.region_type,
                "support_id": region.support_id,
                "anchor_ids": list(region.anchor_ids),
                "polygon_2d": [list(point) for point in region.polygon_2d],
                "point_2d": list(region.point_2d) if region.point_2d is not None else None,
                "reason": region.reason,
            }
            for region in scene_graph.placement_regions
        ],
        "orientation_facts": [
            {
                "subject": id_to_index.get(edge.subject, edge.subject),
                "object": id_to_index.get(edge.object, edge.object),
                "predicate": edge.predicate,
                "source": edge.source,
                "confidence": edge.confidence,
                "frame": edge.frame,
            }
            for edge in scene_graph.orientation_facts
        ],
    }


def build_scene_graphs_from_grounding(
    grounding_jsonl: str,
    output_jsonl: str = "data/output/flywheel/scene_graph.jsonl",
    depth_info_jsonl: Optional[str] = None,
    thresholds: Optional[SceneGraphThresholds] = None,
    enable_3d: bool = True,
    ontology_store_path: Optional[str] = None,
    enable_mask_segmentation: bool = False,
    sam3_url: str = "YOUR_SAM3_URL",
    mask_output_dir: str = "data/output/masks",
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    items = _load_grounding_items(grounding_jsonl)
    if not items:
        with open(output_jsonl, "w", encoding="utf-8") as handle:
            handle.write("")
        return {
            "count": 0,
            "total_objects": 0,
            "total_relations": 0,
            "predicate_counts": {},
            "source_counts": {},
            "output_jsonl": output_jsonl,
        }

    depth_map = _load_depth_map(depth_info_jsonl)
    ontology = OntologyStore.load(resolve_store_path(ontology_store_path))
    segmenter = None
    if enable_mask_segmentation:
        from apps.data_flywheel.perception.backends.sam3 import SAM3Backend

        segmenter = SAM3Backend(
            base_url=sam3_url,
            mask_output_dir=mask_output_dir,
        )

    records: List[Dict[str, Any]] = []
    predicate_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    total_objects = 0
    total_relations = 0

    for item in items:
        image_path = str(item.get("image", ""))
        image_size = _validate_image_size(
            (
                int(item.get("width", 1280)),
                int(item.get("height", 800)),
            )
        )
        image = _load_image(image_path, image_size)
        segmentations = _segmentations_from_grounding_item(item, image, segmenter)
        depth_info = depth_map.get(image_path) if enable_3d else None
        scene_graph = build_scene_graph(
            image_id=image_path,
            image=image,
            image_size=image_size,
            segmentations=segmentations,
            ontology=ontology,
            depth_info=depth_info,
            thresholds=thresholds,
            skip_phase_b=False,
        )
        record = scene_graph_to_legacy_record(scene_graph, image_path=image_path)
        records.append(record)
        total_objects += len(record["objects"])
        total_relations += len(record["relations"])
        for relation in record["relations"]:
            predicate = relation["predicate"]
            source = relation.get("source", "2d")
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1

    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "count": len(records),
        "total_objects": total_objects,
        "total_relations": total_relations,
        "predicate_counts": predicate_counts,
        "source_counts": source_counts,
        "output_jsonl": output_jsonl,
    }
