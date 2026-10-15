from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw

from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.types import SceneGraph, SceneNode

from .phase_a.destination import resolve_cleanup_destination
from .phase_a.core import PlanResult, plan
from .phase_b.sample import synthesize_samples
from .phase_c.base import RejectedSampleEntry
from .phase_c.main import validate_samples


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _preferred_instruction_phrase(node: SceneNode) -> str:
    for phrase in node.referring_expressions:
        if _has_chinese(phrase):
            return phrase
    if node.hierarchical_name:
        return node.hierarchical_name
    return node.class_name or node.id


def _node_center_x(node: SceneNode) -> float:
    return (float(node.bbox[0]) + float(node.bbox[2])) / 2.0


def _node_center_y(node: SceneNode) -> float:
    return (float(node.bbox[1]) + float(node.bbox[3])) / 2.0


def _is_push_node(node: SceneNode) -> bool:
    return (
        node.operation == "push"
        or node.placement_policy == "push_only"
        or "push_only" in node.extra_tags
    )


def _is_cleanup_candidate(node: SceneNode) -> bool:
    return (
        node.operation == "pick"
        and node.grasp_point is not None
        and not node.is_container
        and "support_surface" not in node.role_ids
        and "base_support_root" not in node.role_ids
    )


def _candidate_names(node: SceneNode) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in [node.class_name, node.canonical_name, node.hierarchical_name, *(node.referring_expressions or [])]:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _hydrate_node_from_ontology(node: SceneNode, ontology: OntologyStore) -> None:
    if node.operation and node.object_class_id:
        return

    for candidate in _candidate_names(node):
        resolved = ontology.resolve_observation_name(candidate)
        if resolved.ontology_status != "resolved" or not resolved.operation:
            continue
        node.operation = node.operation or resolved.operation
        if resolved.is_container is not None:
            node.is_container = bool(resolved.is_container)
        node.container_placeability = node.container_placeability or resolved.container_placeability
        node.placement_policy = node.placement_policy or resolved.placement_policy
        node.extra_tags = list(dict.fromkeys([*(node.extra_tags or []), *(resolved.extra_tags or [])]))
        node.canonical_name = resolved.canonical_name or node.canonical_name
        node.object_class_id = resolved.object_class_id or node.object_class_id
        node.ontology_status = resolved.ontology_status
        if not node.hierarchical_name and resolved.canonical_name and _has_chinese(resolved.canonical_name):
            node.hierarchical_name = resolved.canonical_name
        break


def _hydrate_scene_graph_from_ontology(scene_graph: SceneGraph, ontology: OntologyStore) -> SceneGraph:
    enriched = copy.deepcopy(scene_graph)
    for node in enriched.nodes:
        _hydrate_node_from_ontology(node, ontology)
    return enriched


def _scene_prefers_chinese(scene_graph: SceneGraph) -> bool:
    return any(_has_chinese(_preferred_instruction_phrase(node)) for node in scene_graph.nodes)


def _build_place_instruction(
    node: SceneNode,
    scene_graph: SceneGraph,
    node_by_id: Dict[str, SceneNode],
) -> str:
    if not _is_cleanup_candidate(node):
        return ""

    resolution = resolve_cleanup_destination(node, scene_graph, OntologyStore())
    if resolution.status != "resolved" or resolution.region is None:
        return ""
    if resolution.region.region_type != "inside-container":
        return ""

    destination_node = node_by_id.get(resolution.destination_id or "")
    if destination_node is None:
        return ""

    target_phrase = _preferred_instruction_phrase(node)
    destination_phrase = _preferred_instruction_phrase(destination_node)
    if not target_phrase or not destination_phrase or target_phrase == destination_phrase:
        return ""

    if _has_chinese(target_phrase) or _has_chinese(destination_phrase):
        return f"把{target_phrase}放到{destination_phrase}里"
    return f"put {target_phrase} in {destination_phrase}"


def _has_cleanup_destination(node: SceneNode, scene_graph: SceneGraph) -> bool:
    if not _is_cleanup_candidate(node):
        return False
    resolution = resolve_cleanup_destination(node, scene_graph, OntologyStore())
    return resolution.status == "resolved" and resolution.region is not None


def generate_candidate_instructions(
    scene_graph: SceneGraph,
    max_instructions: int = 0,
) -> List[str]:
    node_by_id = {node.id: node for node in scene_graph.nodes}
    blockers_by_target: Dict[str, List[SceneNode]] = {}
    for blocker_id, blocked_id in scene_graph.blocking_dag:
        blocker = node_by_id.get(blocker_id)
        if blocker is None:
            continue
        blockers_by_target.setdefault(blocked_id, []).append(blocker)

    ordered_nodes = sorted(
        scene_graph.nodes,
        key=lambda node: (
            0 if node.operation == "pick" and blockers_by_target.get(node.id) else 1 if node.operation == "pick" else 2,
            _node_center_x(node),
            node.id,
        ),
    )

    instructions: List[str] = []

    def _append_instruction(text: str) -> bool:
        if max_instructions > 0 and len(instructions) >= max_instructions:
            return True
        if text and text not in instructions:
            instructions.append(text)
        return max_instructions > 0 and len(instructions) >= max_instructions

    for node in ordered_nodes:
        target_phrase = _preferred_instruction_phrase(node)
        instruction = ""
        blockers = sorted(blockers_by_target.get(node.id, []), key=_node_center_x)
        if node.operation == "pick" and blockers:
            blocker_phrase = _preferred_instruction_phrase(blockers[0])
            instruction = f"拿被{blocker_phrase}压住的{target_phrase}"
        elif node.operation == "pick":
            instruction = f"拿{target_phrase}"
        elif _is_push_node(node):
            instruction = f"推{target_phrase}"

        if _append_instruction(instruction):
            break

        if _append_instruction(_build_place_instruction(node, scene_graph, node_by_id)):
            break

    if any(_has_cleanup_destination(node, scene_graph) for node in ordered_nodes):
        cleanup_instruction = "收拾桌面" if _scene_prefers_chinese(scene_graph) else "tidy the desk"
        _append_instruction(cleanup_instruction)

    return instructions


def _bbox_center(node: SceneNode) -> List[float]:
    return [
        (float(node.bbox[0]) + float(node.bbox[2])) / 2.0,
        (float(node.bbox[1]) + float(node.bbox[3])) / 2.0,
    ]


def _mask_filename(scene_graph: SceneGraph, node: SceneNode) -> str:
    image_stem = Path(scene_graph.image_path or scene_graph.image_id or "scene").stem
    return f"{image_stem}_{node.id}.png"


def materialize_missing_masks(
    scene_graph: SceneGraph,
    mask_output_dir: str,
) -> SceneGraph:
    enriched = copy.deepcopy(scene_graph)
    output_dir = Path(mask_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = enriched.image_size
    for node in enriched.nodes:
        if node.grasp_point is None:
            node.grasp_point = _bbox_center(node)

        if node.mask_file and Path(node.mask_file).exists():
            continue

        mask_path = output_dir / _mask_filename(enriched, node)
        image = Image.new("L", (int(width), int(height)), 0)
        draw = ImageDraw.Draw(image)
        polygon = node.mask_polygon or [
            [float(node.bbox[0]), float(node.bbox[1])],
            [float(node.bbox[2]), float(node.bbox[1])],
            [float(node.bbox[2]), float(node.bbox[3])],
            [float(node.bbox[0]), float(node.bbox[3])],
        ]
        draw.polygon([tuple(point[:2]) for point in polygon], fill=255)
        image.save(mask_path)
        node.mask_file = str(mask_path)
        if node.mask_polygon is None:
            node.mask_polygon = [list(point[:2]) for point in polygon]

    return enriched


def run_training_generation(
    scene_graph: SceneGraph,
    ontology: OntologyStore,
    *,
    variants_per_plan: int = 3,
    max_instructions: int = 0,
    mask_output_dir: str = "data/output/flywheel/runtime_masks",
    data_source: str = "real",
) -> Dict[str, Any]:
    runtime_scene_graph = materialize_missing_masks(scene_graph, mask_output_dir)
    runtime_scene_graph = _hydrate_scene_graph_from_ontology(runtime_scene_graph, ontology)
    instructions = generate_candidate_instructions(runtime_scene_graph, max_instructions=max_instructions)

    plan_results: List[PlanResult] = [
        plan(runtime_scene_graph, instruction, ontology)
        for instruction in instructions
    ]

    training_samples = []
    for result in plan_results:
        training_samples.extend(
            synthesize_samples(
                image_path=runtime_scene_graph.image_path,
                image_size=runtime_scene_graph.image_size,
                scene_graph=runtime_scene_graph,
                plan_result=result,
                variants_per_plan=variants_per_plan,
                data_source=data_source,
            )
        )

    passed_samples, rejected_entries = validate_samples(training_samples)
    return {
        "scene_graph": runtime_scene_graph,
        "instructions": instructions,
        "plan_results": plan_results,
        "training_samples": passed_samples,
        "rejected_entries": rejected_entries,
    }


def build_training_artifacts(
    scene_graphs: Iterable[SceneGraph],
    *,
    ontology: OntologyStore,
    output_dir: str,
    variants_per_plan: int = 3,
    max_instructions: int = 0,
    data_source: str = "real",
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_results_path = out_dir / "training_plan_results.jsonl"
    training_samples_path = out_dir / "training_samples.jsonl"
    rejected_samples_path = out_dir / "rejected_samples.jsonl"
    validation_metrics_path = out_dir / "validation_metrics.json"
    mask_output_dir = out_dir / "runtime_masks"

    plan_results_by_image: Dict[str, List[PlanResult]] = {}
    training_samples_by_image: Dict[str, List[Any]] = {}
    rejected_entries_by_image: Dict[str, List[RejectedSampleEntry]] = {}

    instruction_count = 0
    accepted_samples = 0
    rejected_samples = 0

    with plan_results_path.open("w", encoding="utf-8") as plans_handle, training_samples_path.open(
        "w", encoding="utf-8"
    ) as samples_handle, rejected_samples_path.open("w", encoding="utf-8") as rejected_handle:
        for scene_graph in scene_graphs:
            result = run_training_generation(
                scene_graph=scene_graph,
                ontology=ontology,
                variants_per_plan=variants_per_plan,
                max_instructions=max_instructions,
                mask_output_dir=str(mask_output_dir),
                data_source=data_source,
            )
            image_key = scene_graph.image_path
            plan_results_by_image[image_key] = list(result["plan_results"])
            training_samples_by_image[image_key] = list(result["training_samples"])
            rejected_entries_by_image[image_key] = list(result["rejected_entries"])

            instruction_count += len(result["instructions"])
            accepted_samples += len(result["training_samples"])
            rejected_samples += len(result["rejected_entries"])

            for item in result["plan_results"]:
                plans_handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            for item in result["training_samples"]:
                samples_handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            for item in result["rejected_entries"]:
                rejected_handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    metrics = {
        "images": len(plan_results_by_image),
        "instruction_count": instruction_count,
        "plan_results": sum(len(items) for items in plan_results_by_image.values()),
        "accepted_samples": accepted_samples,
        "rejected_samples": rejected_samples,
    }
    validation_metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "plan_results_path": str(plan_results_path),
        "training_samples_path": str(training_samples_path),
        "rejected_samples_path": str(rejected_samples_path),
        "validation_metrics_path": str(validation_metrics_path),
        "plan_results_by_image": plan_results_by_image,
        "training_samples_by_image": training_samples_by_image,
        "rejected_entries_by_image": rejected_entries_by_image,
        **metrics,
    }
