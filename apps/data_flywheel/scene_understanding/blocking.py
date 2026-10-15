"""Blocking DAG derivation for scene understanding."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

from apps.data_flywheel.types import SceneEdge, SceneNode


logger = logging.getLogger("scene_understanding.blocking")


def _build_graph(edges: Iterable[Tuple[str, str]]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for blocker, blocked in edges:
        graph[blocker].add(blocked)
    return graph


def _has_path(graph: Dict[str, Set[str]], start: str, target: str) -> bool:
    if start == target:
        return True
    queue: deque[str] = deque([start])
    seen: Set[str] = {start}
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, set()):
            if nxt == target:
                return True
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


def _would_create_cycle(dag: List[Tuple[str, str]], candidate: Tuple[str, str]) -> bool:
    blocker, blocked = candidate
    graph = _build_graph(dag)
    return _has_path(graph, blocked, blocker)


def _candidate_block_edge(node_by_id: Dict[str, SceneNode], edge: SceneEdge) -> Tuple[str, str] | None:
    blocker = node_by_id.get(edge.subject)
    if blocker is None:
        return None
    if "base_support_root" in set(blocker.role_ids or []):
        return None

    if edge.predicate == "stacked-on":
        return edge.subject, edge.object
    if edge.predicate != "occluding":
        return None

    operation = blocker.preferred_operation or blocker.operation
    if operation not in {"pick", "push"}:
        return None
    return edge.subject, edge.object


def derive_blocking_dag(
    nodes: List[SceneNode],
    edges: List[SceneEdge],
) -> List[Tuple[str, str]]:
    """Derive direct blocking edges from scene relations."""

    node_by_id = {node.id: node for node in nodes}
    dag: List[Tuple[str, str]] = []

    for edge in edges:
        candidate = _candidate_block_edge(node_by_id, edge)
        if candidate is None:
            continue
        if candidate[0] == candidate[1]:
            continue
        if _would_create_cycle(dag, candidate):
            logger.warning("Cycle detected in blocking_dag; dropping edge %s -> %s", candidate[0], candidate[1])
            continue
        dag.append(candidate)

    return dag


def populate_actionability_state(
    nodes: List[SceneNode],
    blocking_dag: List[Tuple[str, str]],
) -> None:
    node_by_id = {node.id: node for node in nodes}

    for node in nodes:
        node.blocked_by = []
        node.blocking = []
        node.excluded_from_action_graph = (
            node.excluded_from_action_graph
            or "base_support_root" in set(node.role_ids or [])
        )

    for blocker_id, blocked_id in blocking_dag:
        blocker = node_by_id.get(blocker_id)
        blocked = node_by_id.get(blocked_id)
        if blocker is None or blocked is None:
            continue
        if blocker_id not in blocked.blocked_by:
            blocked.blocked_by.append(blocker_id)
        if blocked_id not in blocker.blocking:
            blocker.blocking.append(blocked_id)

    for node in nodes:
        preferred_operation = node.preferred_operation or node.operation
        actionable = preferred_operation in {"pick", "push"}
        node.directly_reachable = (
            actionable
            and not node.blocked_by
            and not node.excluded_from_action_graph
        )


def populate_node_actionability(
    nodes: List[SceneNode],
    blocking_dag: List[Tuple[str, str]],
) -> None:
    blocked_by: Dict[str, List[str]] = defaultdict(list)
    blocking: Dict[str, List[str]] = defaultdict(list)

    for blocker, blocked in blocking_dag:
        blocked_by[blocked].append(blocker)
        blocking[blocker].append(blocked)

    for node in nodes:
        node.blocked_by = list(blocked_by.get(node.id, []))
        node.blocking = list(blocking.get(node.id, []))
        node.excluded_from_action_graph = node.excluded_from_action_graph or "base_support_root" in set(node.role_ids or [])
        if node.excluded_from_action_graph:
            node.directly_reachable = False
        elif node.grasp_point is None and (node.preferred_operation or node.operation) == "pick":
            node.directly_reachable = False
        else:
            node.directly_reachable = not bool(node.blocked_by)
