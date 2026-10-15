"""2D + 3D edge fusion for scene understanding."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

from apps.data_flywheel.types import SceneEdge


_VALID_STRATEGIES = {"higher_confidence", "3d_wins", "2d_wins"}


def _merge_same_predicate(edges: List[SceneEdge]) -> SceneEdge:
    base = edges[0]
    confidence = min(1.0, sum(edge.confidence for edge in edges))
    sources = {edge.source for edge in edges}
    source = "2d+3d" if sources == {"2d", "3d"} else base.source
    frames = {edge.frame for edge in edges}
    frame = "multi" if sources == {"2d", "3d"} else (base.frame if len(frames) == 1 else "multi")
    return SceneEdge(
        subject=base.subject,
        object=base.object,
        predicate=base.predicate,
        source=source,
        confidence=confidence,
        frame=frame,
    )


def _choose_conflict(edges: List[SceneEdge], strategy: str) -> SceneEdge:
    if strategy == "3d_wins":
        for edge in edges:
            if edge.source == "3d":
                return edge
    elif strategy == "2d_wins":
        for edge in edges:
            if edge.source == "2d":
                return edge

    # higher_confidence or fallback: pick the max-confidence edge.
    return max(edges, key=lambda edge: edge.confidence)


def fuse_edges(
    edges_2d: List[SceneEdge],
    edges_3d: List[SceneEdge],
    strategy: str = "higher_confidence",
) -> List[SceneEdge]:
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Invalid fusion strategy '{strategy}'. Must be one of {sorted(_VALID_STRATEGIES)}"
        )

    grouped: "OrderedDict[Tuple[str, str], List[SceneEdge]]" = OrderedDict()
    for edge in [*edges_2d, *edges_3d]:
        grouped.setdefault((edge.subject, edge.object), []).append(edge)

    fused: List[SceneEdge] = []
    for pair_edges in grouped.values():
        by_predicate: Dict[str, List[SceneEdge]] = OrderedDict()
        for edge in pair_edges:
            by_predicate.setdefault(edge.predicate, []).append(edge)

        merged_edges: List[SceneEdge] = []
        for group in by_predicate.values():
            merged_edges.append(_merge_same_predicate(group))

        if len(merged_edges) == 1:
            fused.append(merged_edges[0])
            continue

        if len(merged_edges) == 2:
            sources = {edge.source for edge in merged_edges}
            predicates = {edge.predicate for edge in merged_edges}
            if sources == {"2d", "3d"} and len(predicates) == 2:
                fused.append(_choose_conflict(merged_edges, strategy))
                continue

        fused.extend(merged_edges)

    return fused
