"""Legacy scene-graph compatibility entrypoints.

This module remains for older callers that still import
`apps.data_flywheel.scene_graph.build_scene_graphs`, but the implementation is
now delegated to the M3 scene-understanding builder.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional

from apps.data_flywheel.scene_understanding.builder import (
    build_scene_graphs_from_grounding,
)
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds

logger = logging.getLogger("flywheel.scene_graph")


def build_scene_graphs(
    grounding_jsonl: str,
    output_jsonl: str = "data/output/flywheel/scene_graph.jsonl",
    depth_info_jsonl: Optional[str] = None,
    thresholds: Optional[SceneGraphThresholds] = None,
    enable_3d: bool = True,
    ontology_store_path: Optional[str] = None,
) -> Dict[str, Any]:
    warnings.warn(
        "apps.data_flywheel.scene_graph.build_scene_graphs is deprecated; "
        "use apps.data_flywheel.scene_understanding.builder.build_scene_graph "
        "or build_scene_graphs_from_grounding instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_scene_graphs_from_grounding(
        grounding_jsonl=grounding_jsonl,
        output_jsonl=output_jsonl,
        depth_info_jsonl=depth_info_jsonl,
        thresholds=thresholds,
        enable_3d=enable_3d,
        ontology_store_path=ontology_store_path,
    )


__all__ = ["SceneGraphThresholds", "build_scene_graphs"]
