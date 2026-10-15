#!/usr/bin/env python3
"""GSA task-form taxonomy (RynnBrain-style output forms) + record classifier.

Nine output forms, each with its own validation contract:
  point      : {"point_2d":[x,y]} / <point>[[x,y]]</point>      (pointing)
  bbox       : {"bbox_2d":[x1,y1,x2,y2]}                        (grounding/REC)
  detection  : list of >=2 bbox with labels                     (multi-object)
  mask       : polygon / RLE / seg reference
  qa         : free-form or multiple-choice text
  dof        : gripper/jaw axis, rotation, 6-DoF-ish interface
  depth      : metric/relative depth or distance ordering
  affordance  : where-to-act / graspable region / functional part
  action_cot : multi-step action plan with explicit reasoning

Coordinates are thousandth-scale ints in [0,1000), origin top-left.
"""
from __future__ import annotations

import re

FORMS = ["point", "bbox", "detection", "mask", "qa", "dof", "depth", "affordance", "action_cot"]

# Scene buckets (for balance reporting)
SCENES = ["grounding", "spatial", "action_planning", "embodied_home",
          "exploration", "ego", "general_visual"]

_RE_POINT = re.compile(r'"point_2d"\s*:\s*\[|<point>')
_RE_BBOX = re.compile(r'"bbox_2d"\s*:\s*\[|<bbox>|"bbox"\s*:\s*\[')
_RE_MASK = re.compile(r'"(polygon|segmentation|mask|rle)"\s*:', re.I)
_RE_DOF = re.compile(r'jaw_axis|gripper|rotation|yaw|pitch|roll|6dof|grasp_axis', re.I)
_RE_DEPTH = re.compile(r'\bdepth\b|closer to the (observer|camera)|farther|distance', re.I)
_RE_AFFORD = re.compile(r'afford|graspable|where (can|should) .*(place|put|grasp)|unoccupied space|free space|handle|functional part', re.I)
_RE_COT = re.compile(r'step\s*1|first,.*then|<think>|because|in order to|plan:', re.I)


def classify(user_text: str, asst_text: str) -> str:
    """Assign one output form. Order matters: most specific first."""
    a, u = asst_text or "", user_text or ""
    both = u + "\n" + a
    if _RE_MASK.search(a):
        return "mask"
    if _RE_DOF.search(both):
        return "dof"
    # detection = multiple boxes
    if _RE_BBOX.search(a):
        n = len(re.findall(r'"bbox_2d"\s*:\s*\[', a)) or len(re.findall(r'"bbox"\s*:\s*\[', a))
        return "detection" if n >= 2 else "bbox"
    if _RE_AFFORD.search(u):
        return "affordance"
    if _RE_POINT.search(a):
        return "point"
    if _RE_DEPTH.search(u):
        return "depth"
    if _RE_COT.search(a) and len(a) > 160:
        return "action_cot"
    return "qa"


# Which sources belong to which scene bucket. Anything not listed -> general_visual.
SOURCE_SCENE = {
    "pixmo_points": "general_visual", "pixmo_pointing_explanations": "general_visual",
    "pixmo_count_counting": "general_visual", "pixmo_cap_qa": "general_visual",
    "pixmo_ask_model_anything": "general_visual", "visual7w": "general_visual",
    "localized_narratives": "general_visual", "vqav2": "general_visual",
    "okvqa": "general_visual", "aokvqa": "general_visual", "cocoqa": "general_visual",
    "tallyqa": "general_visual", "nlvr2": "general_visual", "clevr": "spatial",
    "refspatial_2d_choice": "spatial", "refspatial_2d_reasoning": "spatial",
    "refspatial_3d_choice": "spatial", "refspatial_3d_reasoning": "spatial",
    "refspatial_3d_vacant": "spatial", "refspatial_3d_visual_choice": "spatial",
    "sat_spatial_qa": "spatial", "vsr": "spatial", "whatsup_spatial": "spatial",
    "prism": "action_planning", "robopoint_ref": "action_planning",
    "roborefit_rec": "action_planning", "fsd_free_point": "action_planning",
    "bridge_flywheel": "embodied_home",
    # newly registered open sources. Visual Genome is object-level bbox grounding =
    # the G layer of the GSA hierarchy, so it is embodied-relevant (its own bucket),
    # NOT general_visual (which is the capped anti-forgetting bucket).
    "visual_genome": "grounding", "visual_genome_cleaned": "grounding",
    "gs_spatial_qa": "spatial", "vsi_590k": "spatial", "mimic_cgd": "spatial",
    "pixmo_points_label": "grounding",
    "robopoint_ref_40k": "action_planning", "roborefit_rec_105k": "action_planning",
    "fsd_free_point_98k": "action_planning",
}

# Sources to EXCLUDE from the embodied corpus (chart/doc/OCR: not embodied).
NON_EMBODIED = {
    "chartqa", "docvqa", "plotqa", "dvqa", "figureqa", "ocrvqa", "textvqa",
    "robut_wikisql", "mapqa", "iconqa", "ai2d", "scienceqa", "raven",
}
