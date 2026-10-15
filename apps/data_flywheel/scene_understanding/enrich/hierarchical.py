"""Rule-based hierarchical name generation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from apps.data_flywheel.geometry import bbox_center, bbox_distance
from apps.data_flywheel.types import SceneNode


_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}


def _group_key(node: SceneNode) -> str:
    return node.object_class_id or node.canonical_name or node.class_name


def _display_name(node: SceneNode) -> str:
    if node.object_class_id:
        return node.object_class_id.replace("_", " ")
    return node.class_name


def _same_class_nodes(node: SceneNode, all_nodes: Iterable[SceneNode]) -> List[SceneNode]:
    target_key = _group_key(node)
    return [candidate for candidate in all_nodes if _group_key(candidate) == target_key]


def _ordered_by_x(nodes: List[SceneNode]) -> List[SceneNode]:
    return sorted(nodes, key=lambda candidate: (bbox_center(candidate.bbox)[0], bbox_center(candidate.bbox)[1], candidate.id))


def _ordinal_word(index: int) -> str:
    return _ORDINALS.get(index, f"{index}th")


def _distinct_color_name(node: SceneNode, same_class: List[SceneNode]) -> Optional[str]:
    color = node.attributes.color_primary
    if not color:
        return None
    colors = [candidate.attributes.color_primary for candidate in same_class if candidate.attributes.color_primary]
    if len(colors) != len(same_class):
        return None
    if colors.count(color) == 1:
        return color
    return None


def _same_support_context_name(node: SceneNode, all_nodes: Iterable[SceneNode]) -> Optional[str]:
    if not node.support_parent or not node.support_type:
        return None
    node_by_id: Dict[str, SceneNode] = {candidate.id: candidate for candidate in all_nodes}
    parent = node_by_id.get(node.support_parent)
    if parent is None or "base_support_root" in set(parent.role_ids or []):
        return None
    parent_name = _display_name(parent)
    if node.support_type == "inside":
        return f"in the {parent_name}"
    return f"on the {parent_name}"


def _position_name(
    node: SceneNode, same_class: List[SceneNode], *, compact: bool = False
) -> Optional[str]:
    ordered = _ordered_by_x(same_class)
    positions = {candidate.id: idx for idx, candidate in enumerate(ordered)}
    rank = positions[node.id]
    xs = [bbox_center(candidate.bbox)[0] for candidate in ordered]
    if max(xs) - min(xs) < 1e-6:
        return None

    if len(ordered) == 2:
        return "left" if rank == 0 else "right"

    if len(ordered) == 3:
        labels = ["left", "middle", "right"] if compact else ["leftmost", "middle", "rightmost"]
        return labels[rank]

    if len(ordered) >= 4:
        return _ordinal_word(rank + 1)

    return None


def _compose_name(
    node: SceneNode,
    *,
    position: Optional[str] = None,
    color: Optional[str] = None,
    ordinal_suffix: bool = False,
) -> str:
    parts: List[str] = []
    if position:
        parts.append(position)
    if color:
        parts.append(color)
    body = " ".join([*parts, _display_name(node)])
    if ordinal_suffix and position:
        return f"the {body} from the left"
    return f"the {body}"


def _descriptor_layered_name(node: SceneNode, same_class: List[SceneNode]) -> Optional[str]:
    descriptor_axes = {axis.lower() for axis in node.descriptor_axes}
    if not descriptor_axes:
        return None

    color = node.attributes.color_primary if "color" in descriptor_axes else None
    all_have_colors = all(candidate.attributes.color_primary for candidate in same_class)
    color_is_unique = bool(
        color and all_have_colors and sum(
            1 for candidate in same_class if candidate.attributes.color_primary == color
        )
        == 1
    )

    position = None
    if "relative_position" in descriptor_axes and len(same_class) > 1 and not color_is_unique:
        position = _position_name(node, same_class, compact=True)

    if position:
        return _compose_name(
            node,
            position=position,
            color=color,
            ordinal_suffix=len(same_class) >= 4,
        )
    if color_is_unique and color:
        return _compose_name(node, color=color)

    return None


def _neighbor_fallback(node: SceneNode, all_nodes: Iterable[SceneNode]) -> Optional[str]:
    target_key = _group_key(node)
    others = [
        candidate
        for candidate in all_nodes
        if candidate.id != node.id and _group_key(candidate) != target_key
    ]
    if node.support_parent:
        others = [candidate for candidate in others if candidate.id != node.support_parent]
    if not others:
        return None
    neighbor = min(others, key=lambda candidate: bbox_distance(candidate.bbox, node.bbox))
    return f"the {_display_name(node)} next to the {_display_name(neighbor)}"


def _descriptor_token(axis: str, node: SceneNode, same_class: List[SceneNode], all_nodes: List[SceneNode]) -> Optional[str]:
    normalized = axis.lower()
    if normalized == "color":
        return _distinct_color_name(node, same_class) or node.attributes.color_primary
    if normalized == "relative_position" and len(same_class) > 1:
        if _distinct_color_name(node, same_class):
            return None
        return _position_name(node, same_class, compact=True)
    if normalized == "support_context":
        return _same_support_context_name(node, all_nodes)
    if normalized in {"near_anchor", "nearby_anchor"}:
        target_key = _group_key(node)
        support_parent = node.support_parent
        candidates = [
            candidate
            for candidate in all_nodes
            if candidate.id != node.id
            and _group_key(candidate) != target_key
            and candidate.id != support_parent
        ]
        if not candidates:
            return None
        neighbor = min(candidates, key=lambda candidate: bbox_distance(candidate.bbox, node.bbox))
        neighbor = f"the {_display_name(node)} next to the {_display_name(neighbor)}"
        return neighbor.replace(f"the {_display_name(node)} ", "", 1)
    return None


def _descriptor_priority_name(node: SceneNode, same_class: List[SceneNode], all_nodes: List[SceneNode]) -> Optional[str]:
    if not node.descriptor_axes:
        return None

    parts: List[str] = []
    seen = set()
    for axis in node.descriptor_axes:
        token = _descriptor_token(axis, node, same_class, all_nodes)
        if not token:
            continue
        normalized = token.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(token)

    if not parts:
        return None

    positional = [part for part in parts if part in {"left", "right", "middle"} or part.endswith("most") or part.endswith("from the left")]
    non_positional = [part for part in parts if part not in positional and not part.startswith("next to ")]
    context = [part for part in parts if part.startswith("in the ") or part.startswith("on the ") or part.startswith("next to ")]
    name_parts = positional + non_positional
    body = " ".join([part for part in name_parts if not part.startswith(("in the ", "on the ", "next to "))] + [_display_name(node)])
    suffix = f" {' '.join(context)}" if context else ""
    return f"the {body}{suffix}".strip()


def generate_hierarchical_name(node: SceneNode, all_nodes: List[SceneNode]) -> str:
    same_class = _same_class_nodes(node, all_nodes)
    if len(same_class) <= 1:
        descriptor_name = _descriptor_priority_name(node, same_class, all_nodes)
        return descriptor_name or f"the {_display_name(node)}"

    if node.descriptor_axes:
        layered_name = _descriptor_priority_name(node, same_class, all_nodes)
        if layered_name:
            return layered_name
        layered_name = _descriptor_layered_name(node, same_class)
        if layered_name:
            return layered_name
        neighbor_name = _neighbor_fallback(node, all_nodes)
        if neighbor_name:
            return neighbor_name
        return f"the {_display_name(node)} ({node.id})"

    color_name = _distinct_color_name(node, same_class)
    if color_name:
        return f"the {color_name} {_display_name(node)}"

    position_name = _position_name(node, same_class, compact=False)
    if position_name:
        return _compose_name(
            node,
            position=position_name,
            ordinal_suffix=len(same_class) >= 4,
        )

    if len(same_class) > 1:
        neighbor_name = _neighbor_fallback(node, all_nodes)
        if neighbor_name:
            return neighbor_name
        return f"the {_display_name(node)} ({node.id})"

    return f"the {_display_name(node)} ({node.id})"
