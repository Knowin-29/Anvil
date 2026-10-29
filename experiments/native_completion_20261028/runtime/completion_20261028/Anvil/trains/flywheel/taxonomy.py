#!/usr/bin/env python3
"""GSA taxonomy aligned to what the pipeline actually produces (7 dimensions).

Grounding (G): object localization/reference.
Spatial   (S): 2D/3D relations between grounded objects.
Action    (A): identity+attributes, object state, precondition, action interface.

We report SEVEN dimensions (the 3 G/S/A layers expand into per-pipeline dims).
The abandoned `A_affordance` (0 samples in v0) is intentionally NOT a dimension.
"""
from __future__ import annotations

# The seven production dimensions, grouped by layer.
DIMENSIONS = {
    "G": ["G_grounding"],
    "S": ["S_spatial"],
    "A": ["A_identity_attr", "A_obj_state", "A_action_precondition",
          "A_robot_state", "A_action_interface"],
}
ALL_DIMS = [d for layer in DIMENSIONS.values() for d in layer]

# Which dimensions carry a bbox that Layer-1 cross-check can verify.
BBOX_DIMS = {"G_grounding", "A_identity_attr", "A_action_interface"}

# Layer dependency: S consumes G context; A consumes G (+S) context.
LAYER_DEPENDS = {"G": [], "S": ["G"], "A": ["G", "S"]}
