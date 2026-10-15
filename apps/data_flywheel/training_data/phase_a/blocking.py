from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

from apps.data_flywheel.types import SceneGraph, SceneNode


logger = logging.getLogger("training_data.blocking")


@dataclass(frozen=True)
class _Priority:
    x: float
    y: float
    node_id: str


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _priority(node: SceneNode) -> _Priority:
    center_x, center_y = _bbox_center(node.bbox)
    return _Priority(center_x, center_y, node.id)


def _node_by_id(scene_graph: SceneGraph) -> Dict[str, SceneNode]:
    return {node.id: node for node in scene_graph.nodes}


def _build_reverse_index(
    scene_graph: SceneGraph,
    resolved: Set[str],
) -> DefaultDict[str, Set[str]]:
    parents: DefaultDict[str, Set[str]] = defaultdict(set)
    for blocker_id, blocked_id in scene_graph.blocking_dag:
        if blocker_id in resolved or blocked_id in resolved:
            continue
        parents[blocked_id].add(blocker_id)
    return parents


def _collect_blocker_closure(
    direct_blockers: Iterable[str],
    parents: Dict[str, Set[str]],
    resolved: Set[str],
) -> Set[str]:
    relevant: Set[str] = set()
    stack = [blocker_id for blocker_id in direct_blockers if blocker_id not in resolved]

    while stack:
        blocker_id = stack.pop()
        if blocker_id in resolved or blocker_id in relevant:
            continue
        relevant.add(blocker_id)
        stack.extend(parent for parent in parents.get(blocker_id, set()) if parent not in resolved)

    return relevant


def _topological_sort_blockers(
    direct_blockers: List[str],
    scene_graph: SceneGraph,
    already_resolved: Set[str],
) -> List[str]:
    """Return blocker ids in execution order.

    The order is deterministic:
    1. recursively expands ancestor blockers from ``scene_graph.blocking_dag``
    2. runs a Kahn-style topological sort on the induced subgraph
    3. breaks ties by left-to-right bbox center X, then Y, then node id
    4. filters out blockers already present in ``already_resolved``
    """

    resolved = set(already_resolved)
    node_by_id = _node_by_id(scene_graph)
    parents = _build_reverse_index(scene_graph, resolved)
    relevant = _collect_blocker_closure(direct_blockers, parents, resolved)

    if not relevant:
        return []

    adjacency: DefaultDict[str, Set[str]] = defaultdict(set)
    indegree: Dict[str, int] = {node_id: 0 for node_id in relevant}

    for blocker_id, blocked_id in scene_graph.blocking_dag:
        if blocker_id in resolved or blocked_id in resolved:
            continue
        if blocker_id not in relevant or blocked_id not in relevant:
            continue
        if blocked_id not in adjacency[blocker_id]:
            adjacency[blocker_id].add(blocked_id)
            indegree[blocked_id] += 1

    heap: List[Tuple[float, float, str]] = []
    for node_id, degree in indegree.items():
        if degree == 0:
            priority = _priority(node_by_id[node_id])
            heappush(heap, (priority.x, priority.y, priority.node_id))

    ordered: List[str] = []
    seen: Set[str] = set()

    while heap:
        _, _, node_id = heappop(heap)
        if node_id in seen or node_id in resolved:
            continue
        seen.add(node_id)
        ordered.append(node_id)

        for child_id in sorted(adjacency.get(node_id, set()), key=lambda candidate: (_priority(node_by_id[candidate]).x, _priority(node_by_id[candidate]).y, candidate)):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                priority = _priority(node_by_id[child_id])
                heappush(heap, (priority.x, priority.y, priority.node_id))

    if len(ordered) != len(relevant):
        missing = [node_id for node_id in relevant if node_id not in seen and node_id not in resolved]
        logger.warning("Cycle detected in blocker subgraph; falling back to deterministic ordering.")
        ordered.extend(sorted(missing, key=lambda candidate: (_priority(node_by_id[candidate]).x, _priority(node_by_id[candidate]).y, candidate)))

    return ordered
