from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from apps.data_flywheel.types import SceneGraph, SceneNode, SceneObjectSample, TrainingSample
from apps.data_flywheel.training_data.phase_a.core import PlanResult, parse_instruction
from apps.data_flywheel.training_data.phase_b.prompts import (
    PROMPT_TEMPLATES_V1,
    PromptTemplate,
    _pick_variant_templates_from_pool,
    fill_template,
)
from apps.data_flywheel.training_data.phase_b.quantize import quantize_bbox, quantize_point
from apps.data_flywheel.training_data.phase_b.rex_omni_format import format_rex_omni_text


_LOGGER = logging.getLogger(__name__)


def _get_node(scene_graph: SceneGraph, node_id: Optional[str]) -> SceneNode:
    if node_id is None:
        raise KeyError("node_id is None")
    for node in scene_graph.nodes:
        if node.id == node_id:
            return node
    raise KeyError(f"Unknown node id: {node_id}")


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _resolve_target_phrase(node: SceneNode, language: str) -> str:
    if language == "zh":
        for phrase in node.referring_expressions:
            if _has_chinese(phrase):
                return phrase
        return node.hierarchical_name or node.class_name or node.id
    return node.hierarchical_name or node.class_name or node.id


def _resolve_blocker_phrase(node: SceneNode, language: str) -> str:
    return _resolve_target_phrase(node, language)


def _resolve_destination_phrase(
    scene_graph: SceneGraph,
    plan_result: PlanResult,
    language: str,
) -> Optional[str]:
    place_action = next((action for action in plan_result.actions if action.type == "place"), None)
    if place_action is None or place_action.destination is None:
        return None
    destination_node = _get_node(scene_graph, place_action.destination)
    return _resolve_target_phrase(destination_node, language)


def build_scene_object_sample(
    node: SceneNode,
    image_size: Tuple[int, int],
) -> SceneObjectSample:
    return SceneObjectSample(
        id=node.id,
        class_name=node.class_name,
        bbox_pixel=list(node.bbox),
        bbox_quantized=quantize_bbox(node.bbox, image_size),
        grasp_point_pixel=list(node.grasp_point) if node.grasp_point else None,
        grasp_point_quantized=quantize_point(node.grasp_point, image_size) if node.grasp_point else None,
        mask_file=node.mask_file,
        operation=node.operation,
        is_container=node.is_container,
        placement_policy=node.placement_policy,
        extra_tags=list(node.extra_tags),
        hierarchical_name=node.hierarchical_name,
        color_primary=node.attributes.color_primary,
    )


def _build_training_sample(
    image_path: str,
    image_size: Tuple[int, int],
    scene_graph: SceneGraph,
    plan_result: PlanResult,
    template: PromptTemplate,
    variant_idx: int,
    data_source: str,
) -> TrainingSample:
    target_node = _get_node(scene_graph, plan_result.resolved_target_id)
    target_phrase = _resolve_target_phrase(target_node, template.language)

    blocker_phrase = None
    if template.task_type == "pick_under_blocker":
        blocker_action = next(
            (action for action in plan_result.actions if "blocker cleanup" in (action.reason or "")),
            None,
        )
        if blocker_action is not None:
            blocker_node = _get_node(scene_graph, blocker_action.target)
            blocker_phrase = _resolve_blocker_phrase(blocker_node, template.language)
    destination_phrase = None
    if template.task_type in {"place_in_container", "cleanup_surface"}:
        destination_phrase = _resolve_destination_phrase(scene_graph, plan_result, template.language)

    instruction_text = fill_template(
        template,
        target_phrase,
        blocker=blocker_phrase,
        destination=destination_phrase,
    )
    sample_id = f"{scene_graph.image_id}_{template.id}_{variant_idx}"
    scene_objects = [build_scene_object_sample(node, image_size) for node in scene_graph.nodes]
    rex_omni_text = format_rex_omni_text(instruction_text, plan_result.thinking, plan_result.actions)

    return TrainingSample(
        sample_id=sample_id,
        image_id=scene_graph.image_id,
        image_path=image_path,
        image_size=image_size,
        data_source=data_source,
        instruction=instruction_text,
        instruction_template_id=template.id,
        instruction_language=template.language,
        scene_objects=scene_objects,
        blocking_dag=list(scene_graph.blocking_dag),
        thinking=plan_result.thinking,
        actions=list(plan_result.actions),
        rex_omni_text=rex_omni_text,
        metadata={
            "num_blockers": sum(1 for action in plan_result.actions if "blocker cleanup" in (action.reason or "")),
            "has_card": any("push_only" in obj.extra_tags for obj in scene_objects),
            "has_place": any(action.type == "place" for action in plan_result.actions),
            "destination_types": [
                action.destination_type
                for action in plan_result.actions
                if action.type == "place" and action.destination_type is not None
            ],
            "num_actions": len(plan_result.actions),
            "variant_group_id": f"{scene_graph.image_id}_{template.task_type}",
        },
    )


def synthesize_samples(
    image_path: str,
    image_size: Tuple[int, int],
    scene_graph: SceneGraph,
    plan_result: PlanResult,
    templates: Optional[List[PromptTemplate]] = None,
    variants_per_plan: int = 3,
    data_source: str = "real",
) -> List[TrainingSample]:
    if plan_result.status != "success":
        return []

    parsed = parse_instruction(plan_result.instruction)
    if parsed.task_type == "scope_violation":
        return []

    pool = templates if templates is not None else PROMPT_TEMPLATES_V1
    selected_templates = _pick_variant_templates_from_pool(parsed.task_type, variants_per_plan, pool)
    if not selected_templates:
        _LOGGER.warning("No prompt templates available for task_type=%s", parsed.task_type)
        return []

    return [
        _build_training_sample(
            image_path=image_path,
            image_size=image_size,
            scene_graph=scene_graph,
            plan_result=plan_result,
            template=template,
            variant_idx=index,
            data_source=data_source,
        )
        for index, template in enumerate(selected_templates, start=1)
    ]


def write_samples_jsonl(
    samples: List[TrainingSample],
    output_path: str,
    append: bool = False,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
