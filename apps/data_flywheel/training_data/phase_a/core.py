from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from apps.data_flywheel.geometry import bbox_center
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.types import Action, SceneGraph, SceneNode, ThinkingStep, ThinkingTrace

from apps.data_flywheel.training_data.phase_a.blocking import _topological_sort_blockers
from apps.data_flywheel.training_data.phase_a.destination import (
    destination_type_for_region,
    resolve_cleanup_destination,
    resolve_explicit_destination,
)
from apps.data_flywheel.training_data.phase_a.resolver import resolve_reference
from apps.data_flywheel.training_data.phase_b.quantize import quantize_point


@dataclass
class ParsedInstruction:
    """Template-match result for Phase A."""

    task_type: str
    target_phrase: str
    blocker_phrase: Optional[str] = None
    destination_phrase: Optional[str] = None
    language: str = "en"


@dataclass
class PlanResult:
    instruction: str
    resolved_target_id: Optional[str]
    thinking: ThinkingTrace
    actions: List[Action]
    status: str
    notes: List[str] = field(default_factory=list)


PatternExtractor = Callable[[re.Match[str]], Dict[str, str]]
PatternEntry = Tuple[str, str, str, PatternExtractor]


INSTRUCTION_PATTERNS: List[PatternEntry] = [
    (r"^拿被(.+?)压住的(.+?)$", "pick_under_blocker", "zh", lambda m: {"blocker_phrase": m.group(1), "target_phrase": m.group(2)}),
    (r"^pick (.+?) under (.+?)$", "pick_under_blocker", "en", lambda m: {"target_phrase": m.group(1), "blocker_phrase": m.group(2)}),
    (r"^拿(.+?)$", "pick_named", "zh", lambda m: {"target_phrase": m.group(1)}),
    (r"^pick (.+?)$", "pick_named", "en", lambda m: {"target_phrase": m.group(1)}),
    (r"^取(.+?)$", "pick_named", "zh", lambda m: {"target_phrase": m.group(1)}),
    (r"^get me (.+?)$", "pick_named", "en", lambda m: {"target_phrase": m.group(1)}),
    (r"^帮我把(.+?)拿起来$", "pick_named", "zh", lambda m: {"target_phrase": m.group(1)}),
    (r"^推(.+?)$", "push_named", "zh", lambda m: {"target_phrase": m.group(1)}),
    (r"^push (.+?)(?: aside)?$", "push_named", "en", lambda m: {"target_phrase": m.group(1)}),
    (r"^把(.+?)推开$", "push_named", "zh", lambda m: {"target_phrase": m.group(1)}),
    (
        r"^(?:把|拿)(.+?)放到(.+?)里$",
        "place_in_container",
        "zh",
        lambda m: {"target_phrase": m.group(1), "destination_phrase": m.group(2)},
    ),
    (
        r"^put (.+?) in (.+?)$",
        "place_in_container",
        "en",
        lambda m: {"target_phrase": m.group(1), "destination_phrase": m.group(2)},
    ),
    (r"^(?:tidy|clean)\s*(?:the\s*)?(?:desk|table)$", "cleanup_surface", "en", lambda m: {"target_phrase": ""}),
    (r"^(?:收拾|整理)(?:桌面|书桌|桌子)$", "cleanup_surface", "zh", lambda m: {"target_phrase": ""}),
]


_LOGGER = logging.getLogger("training_data.planner")


def _detect_lang(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def parse_instruction(text: str) -> ParsedInstruction:
    """Match an instruction to one of the supported v1 templates."""

    normalized_text = text.strip()
    for pattern, task_type, lang, extractor in INSTRUCTION_PATTERNS:
        match = re.match(pattern, normalized_text, re.IGNORECASE)
        if match:
            fields = extractor(match)
            return ParsedInstruction(
                task_type=task_type,
                target_phrase=fields.get("target_phrase", ""),
                blocker_phrase=fields.get("blocker_phrase"),
                destination_phrase=fields.get("destination_phrase"),
                language=lang,
            )

    return ParsedInstruction(
        task_type="scope_violation",
        target_phrase=normalized_text,
        destination_phrase=None,
        language=_detect_lang(normalized_text),
    )


def _get_node(scene_graph: SceneGraph, node_id: str) -> SceneNode:
    for node in scene_graph.nodes:
        if node.id == node_id:
            return node
    raise KeyError(f"Unknown node id: {node_id}")


def _append_affordance_step(steps: List[ThinkingStep], node: SceneNode, content_suffix: str = "") -> None:
    content = f"operation={node.operation}, is_container={node.is_container}, extra_tags={node.extra_tags}"
    if content_suffix:
        content = f"{content}{content_suffix}"
    steps.append(
        ThinkingStep(
            perception_type="Affordance",
            target=node.id,
            content=content,
            output={
                "operation": node.operation,
                "is_container": node.is_container,
                "extra_tags": list(node.extra_tags),
            },
        )
    )


def _append_position_step(steps: List[ThinkingStep], node: SceneNode, image_size: Tuple[int, int]) -> None:
    center_x, center_y = bbox_center(node.bbox)
    qx, qy = quantize_point([center_x, center_y], image_size)
    steps.append(
        ThinkingStep(
            perception_type="Position",
            target=node.id,
            content=f"({qx}, {qy})",
            output={"x": qx, "y": qy},
        )
    )


def _append_grasp_step(steps: List[ThinkingStep], node: SceneNode, image_size: Tuple[int, int]) -> Tuple[List[float], List[int]]:
    if node.grasp_point is None:
        raise ValueError(f"Node '{node.id}' has no grasp_point")
    grasp_pixel = list(node.grasp_point)
    grasp_quantized = quantize_point(grasp_pixel, image_size)
    steps.append(
        ThinkingStep(
            perception_type="GraspPoint",
            target=node.id,
            content=f"({grasp_quantized[0]}, {grasp_quantized[1]})",
            output={"x": grasp_quantized[0], "y": grasp_quantized[1]},
        )
    )
    return grasp_pixel, grasp_quantized


def _blocked_result(
    instruction: str,
    resolved_target_id: Optional[str],
    steps: List[ThinkingStep],
    actions: List[Action],
    note: str,
) -> PlanResult:
    return PlanResult(
        instruction=instruction,
        resolved_target_id=resolved_target_id,
        thinking=ThinkingTrace(steps=steps),
        actions=actions,
        status="blocked_unresolvable",
        notes=[note],
    )


def _select_cleanup_target(scene_graph: SceneGraph) -> Optional[SceneNode]:
    candidates = [
        node
        for node in scene_graph.nodes
        if node.operation == "pick"
        and node.grasp_point is not None
        and not node.is_container
        and "support_surface" not in node.role_ids
        and "base_support_root" not in node.role_ids
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda node: (bbox_center(node.bbox)[0], bbox_center(node.bbox)[1], node.id))


def plan(scene_graph: SceneGraph, instruction: str, ontology: OntologyStore) -> PlanResult:
    steps: List[ThinkingStep] = []
    actions: List[Action] = []
    notes: List[str] = []

    parsed = parse_instruction(instruction)
    if parsed.task_type == "scope_violation":
        rejected = parsed.target_phrase or instruction.strip()
        return PlanResult(
            instruction=instruction,
            resolved_target_id=None,
            thinking=ThinkingTrace(steps=[]),
            actions=[],
            status="scope_violation",
            notes=[f"Instruction not supported in v1: {rejected}"],
        )

    if parsed.task_type == "cleanup_surface":
        target_node = _select_cleanup_target(scene_graph)
        if target_node is None:
            return PlanResult(
                instruction=instruction,
                resolved_target_id=None,
                thinking=ThinkingTrace(steps=[]),
                actions=[],
                status="unresolved",
                notes=["No movable tabletop target available for cleanup"],
            )
        steps.append(
            ThinkingStep(
                perception_type="ReferenceResolution",
                target=target_node.id,
                content='"cleanup_surface" → '
                f"{target_node.id} (cleanup_candidate)",
                output={
                    "status": "resolved",
                    "node_id": target_node.id,
                    "candidates": [target_node.id],
                    "rule": "cleanup_candidate",
                },
            )
        )
    else:
        target_result = resolve_reference(parsed.target_phrase, scene_graph.nodes, language=parsed.language)
        steps.append(
            ThinkingStep(
                perception_type="ReferenceResolution",
                target=target_result.node_id or "",
                content=f'"{parsed.target_phrase}" → {target_result.node_id or target_result.status} ({target_result.matched_rule})',
                output={
                    "status": target_result.status,
                    "node_id": target_result.node_id,
                    "candidates": list(target_result.candidates),
                    "rule": target_result.matched_rule,
                },
            )
        )

        if target_result.status == "not_found":
            return PlanResult(
                instruction=instruction,
                resolved_target_id=None,
                thinking=ThinkingTrace(steps=steps),
                actions=[],
                status="unresolved",
                notes=[f"Target phrase '{parsed.target_phrase}' not matched to any node"],
            )
        if target_result.status == "ambiguous":
            return PlanResult(
                instruction=instruction,
                resolved_target_id=None,
                thinking=ThinkingTrace(steps=steps),
                actions=[],
                status="ambiguous",
                notes=[f"Multiple candidates for '{parsed.target_phrase}': {target_result.candidates}"],
            )

        target_node = _get_node(scene_graph, target_result.node_id)

    ontology_op = ontology.get_operation(target_node.class_name)
    if ontology_op != target_node.operation:
        _LOGGER.warning(
            "Ontology drift for %s: scene_graph.operation=%s, ontology.get_operation=%s. Using scene_graph value.",
            target_node.id,
            target_node.operation,
            ontology_op,
        )

    _append_affordance_step(steps, target_node)

    if parsed.task_type in {"pick_named", "pick_under_blocker", "place_in_container", "cleanup_surface"}:
        if "push_only" in target_node.extra_tags:
            return _blocked_result(
                instruction,
                target_node.id,
                steps,
                actions,
                f"Cannot pick '{target_node.id}': push_only object",
            )
        if target_node.operation == "push":
            return _blocked_result(
                instruction,
                target_node.id,
                steps,
                actions,
                f"Cannot pick '{target_node.id}': operation=push",
            )
        if target_node.operation == "non_operable":
            return _blocked_result(
                instruction,
                target_node.id,
                steps,
                actions,
                f"Cannot pick '{target_node.id}': operation=non_operable",
            )
    elif parsed.task_type == "push_named" and target_node.operation == "non_operable":
        return _blocked_result(
            instruction,
            target_node.id,
            steps,
            actions,
            f"Cannot push '{target_node.id}': operation=non_operable",
        )

    stacked_on = [edge.subject for edge in scene_graph.edges if edge.predicate == "stacked-on" and edge.object == target_node.id]
    if stacked_on:
        steps.append(
            ThinkingStep(
                perception_type="Stacking",
                target=target_node.id,
                content=f"has_on_top={stacked_on}",
                output={"has_on_top": stacked_on},
            )
        )

    occluded_by = [edge.subject for edge in scene_graph.edges if edge.predicate == "occluding" and edge.object == target_node.id]
    if occluded_by:
        steps.append(
            ThinkingStep(
                perception_type="Occlusion",
                target=target_node.id,
                content=f"occluded_by={occluded_by}",
                output={"occluded_by": occluded_by},
            )
        )

    direct_blockers = [blocker_id for blocker_id, blocked_id in scene_graph.blocking_dag if blocked_id == target_node.id]
    steps.append(
        ThinkingStep(
            perception_type="BlockingDependency",
            target=target_node.id,
            content=f"blocked_by={direct_blockers}" if direct_blockers else "blocked_by=[]",
            output={"blocked_by": direct_blockers},
        )
    )

    resolved_blockers: set[str] = set()
    blocker_order = _topological_sort_blockers(direct_blockers, scene_graph, resolved_blockers)

    for blocker_id in blocker_order:
        blocker_node = _get_node(scene_graph, blocker_id)
        if blocker_node.operation == "non_operable":
            return _blocked_result(
                instruction,
                target_node.id,
                steps,
                actions,
                f"Blocker '{blocker_id}' is non_operable",
            )

        _append_affordance_step(steps, blocker_node, content_suffix=" (blocker cleanup)")
        _append_position_step(steps, blocker_node, scene_graph.image_size)
        try:
            grasp_pixel, grasp_quantized = _append_grasp_step(steps, blocker_node, scene_graph.image_size)
        except ValueError:
            return _blocked_result(
                instruction,
                target_node.id,
                steps,
                actions,
                f"Blocker '{blocker_id}' has no grasp_point",
            )

        action_type = "push" if blocker_node.operation == "push" else "pick"
        actions.append(
            Action(
                type=action_type,
                target=blocker_id,
                grasp_point_pixel=grasp_pixel,
                grasp_point_quantized=grasp_quantized,
                reason=f"blocker cleanup for {target_node.id}",
            )
        )
        resolved_blockers.add(blocker_id)

    if parsed.task_type in {"place_in_container", "cleanup_surface"}:
        if parsed.task_type == "place_in_container":
            destination_result = resolve_explicit_destination(
                scene_graph,
                parsed.destination_phrase or "",
                language=parsed.language,
            )
        else:
            destination_result = resolve_cleanup_destination(target_node, scene_graph, ontology)

        steps.append(
            ThinkingStep(
                perception_type="ContainerLookup",
                target=destination_result.destination_id or "",
                content=(
                    f"{destination_result.destination_id or destination_result.status} "
                    f"({destination_result.placement_reason or destination_result.matched_rule or 'no_region'})"
                ),
                output={
                    "status": destination_result.status,
                    "destination_id": destination_result.destination_id,
                    "placement_reason": destination_result.placement_reason,
                    "candidates": list(destination_result.candidates),
                },
            )
        )

        if destination_result.status == "not_found":
            return PlanResult(
                instruction=instruction,
                resolved_target_id=target_node.id,
                thinking=ThinkingTrace(steps=steps),
                actions=[],
                status="unresolved",
                notes=[destination_result.note or "Destination not found"],
            )
        if destination_result.status == "ambiguous":
            return PlanResult(
                instruction=instruction,
                resolved_target_id=target_node.id,
                thinking=ThinkingTrace(steps=steps),
                actions=[],
                status="ambiguous",
                notes=[
                    destination_result.note
                    or f"Multiple destination candidates: {destination_result.candidates}"
                ],
            )

        if destination_result.region is None or destination_result.region.point_2d is None:
            return PlanResult(
                instruction=instruction,
                resolved_target_id=target_node.id,
                thinking=ThinkingTrace(steps=steps),
                actions=[],
                status="blocked_unresolvable",
                notes=["Destination region missing place point"],
            )

    _append_position_step(steps, target_node, scene_graph.image_size)
    try:
        target_grasp_pixel, target_grasp_quantized = _append_grasp_step(steps, target_node, scene_graph.image_size)
    except ValueError:
        return _blocked_result(
            instruction,
            target_node.id,
            steps,
            [],
            f"Target '{target_node.id}' has no grasp_point",
        )

    main_action_type = "push" if parsed.task_type == "push_named" else "pick"
    actions.append(
        Action(
            type=main_action_type,
            target=target_node.id,
            grasp_point_pixel=target_grasp_pixel,
            grasp_point_quantized=target_grasp_quantized,
            reason="main target",
        )
    )

    if parsed.task_type in {"place_in_container", "cleanup_surface"}:
        place_pixel = list(destination_result.region.point_2d)
        place_quantized = quantize_point(place_pixel, scene_graph.image_size)
        actions.append(
            Action(
                type="place",
                target=target_node.id,
                destination=destination_result.destination_id,
                place_point_pixel=place_pixel,
                place_point_quantized=place_quantized,
                reason="main target placement",
                destination_type=destination_type_for_region(destination_result.region),
                destination_anchor_ids=list(destination_result.region.anchor_ids),
                placement_reason=destination_result.placement_reason,
            )
        )

    steps.append(
        ThinkingStep(
            perception_type="PlanOrder",
            target="",
            content=" → ".join(f"{action.type} {action.target}" for action in actions),
            output=[f"{action.type}:{action.target}" for action in actions],
        )
    )

    return PlanResult(
        instruction=instruction,
        resolved_target_id=target_node.id,
        thinking=ThinkingTrace(steps=steps),
        actions=actions,
        status="success",
        notes=notes,
    )
