"""Prompt Templates — RynnBrain-aligned 40/30/30 weighted template strategy.

Generates diverse instruction prompts for pick-and-place training data:
- 40% Object-centric: focus on the target object
- 30% Scene-aware: incorporate scene context
- 30% Task-oriented: emphasize manipulation intent

Bilingual (Chinese + English) to match application user prompts.
"""

from __future__ import annotations

import random
from typing import List, Optional


# 40% Object-centric prompts
OBJECT_CENTRIC_ZH = [
    "找到{object_name}并给出抓取方案。",
    "定位图中的{object_name}，给出抓取点和姿态分析。",
    "识别{object_name}的位置，规划最佳抓取策略。",
    "在图像中找到目标物体\"{object_name}\"，分析其抓取可行性。",
    "请定位{object_name}并确定最优抓取锚点。",
    "找到{object_name}，判断其姿态和最佳夹取方式。",
]

OBJECT_CENTRIC_EN = [
    "Find the {object_name} and provide a grasp plan.",
    "Locate the {object_name} in the image and determine the best grasp point.",
    "Identify the {object_name} and plan the optimal grasp strategy.",
    "Find the target object \"{object_name}\" and analyze its graspability.",
    "Locate the {object_name} and determine the best affordance point for grasping.",
]

# 30% Scene-aware prompts
SCENE_AWARE_ZH = [
    "在这个{scene_desc}场景中，找到{object_name}并规划抓取。",
    "观察{scene_desc}环境，定位{object_name}并给出抓取方案。",
    "在{scene_desc}的桌面上，找到{object_name}的最佳抓取点。",
    "考虑{scene_desc}中的物体布局，规划对{object_name}的抓取。",
]

SCENE_AWARE_EN = [
    "In this {scene_desc} scene, find the {object_name} and plan the grasp.",
    "Observe the {scene_desc} environment, locate the {object_name} and provide a grasp plan.",
    "On the {scene_desc} surface, find the best grasp point for the {object_name}.",
    "Considering the layout in this {scene_desc}, plan how to grasp the {object_name}.",
]

# 30% Task-oriented prompts
TASK_ORIENTED_ZH = [
    "执行pick and place任务，目标物体是{object_name}。",
    "需要抓取{object_name}并放置到指定位置，请给出抓取方案。",
    "机器人需要拾取{object_name}，请分析最佳抓取姿态。",
    "完成抓取任务：找到{object_name}，确定夹爪接近方向和抓取点。",
    "为了将{object_name}移动到目标位置，请规划抓取策略。",
]

TASK_ORIENTED_EN = [
    "Perform a pick and place task. The target object is \"{object_name}\".",
    "Pick up the {object_name} and place it at the designated location. Provide the grasp plan.",
    "The robot needs to pick the {object_name}. Analyze the best grasp pose.",
    "Complete the grasping task: find the {object_name} and determine approach direction and grasp point.",
    "To move the {object_name} to the target location, plan the grasp strategy.",
]

# Scene descriptions for scene-aware prompts
SCENE_DESCRIPTIONS_ZH = [
    "桌面整理", "厨房操作", "仓储分拣", "实验室", "办公桌面",
    "机器人工作台", "家居收纳", "工业分拣", "餐桌", "工具台",
]

SCENE_DESCRIPTIONS_EN = [
    "tabletop tidying", "kitchen operation", "warehouse sorting",
    "laboratory", "office desk", "robot workstation", "home storage",
    "industrial sorting", "dining table", "tool bench",
]

# Application-style system prompt (full version)
PICK_PLACE_SYSTEM_PROMPT = """You are a robotic manipulation expert. Given an image and a target object query, you must output a complete pick-and-place analysis in strict JSON format.

Coordinate rules:
- All coordinates must be normalized integers in [0, 1000] relative to image width and height.
- bbox must tightly cover the visible target object.
- affordance_point must lie inside bbox and on the visible target object.

Affordance point selection rules:
- The point must be on the visible target object, not on the table/background.
- Prefer a stable, rigid, graspable area near where the gripper can contact the object.
- For handled objects (mugs, cups with handles): place on the visible middle/outer handle.
- For bottles, cans, closed jars: prefer the visible side wall around the middle or lower body.
- For open bowls, basins, plates: prefer a visible rim/lip/edge point.
- For upright cups without handle: prefer the visible side wall around the lower or middle body.
- For thin/flat objects: prefer a point near a visible edge while still inside the object.
- For long thin objects (pens, tools): prefer the middle of the shaft/body.
- For boxes/cuboids: prefer a visible side or edge area.
- Avoid: holes, transparent regions, shadows, occluded parts, reflections, extreme boundaries.

Return strict JSON only with this exact schema:
{"bbox": [x1,y1,x2,y2], "affordance_point": [x,y], "affordance_valid": true/false, "affordance_reason": "...", "object_type": "box|cylinder|frustum|bowl|plate|slender|thin|irregular|unknown", "object_type_confidence": 0.0-1.0, "pose_hint": "upright|horizontal|unknown", "pose_hint_confidence": 0.0-1.0, "pose_hint_reason": "...", "grasp_hint": "rim|side|middle|edge|handle|body|unknown", "object_pose": "short pose description", "container_opening": "open_visible|closed_or_lidded|unknown", "container_opening_confidence": 0.0-1.0, "container_opening_reason": "...", "preferred_approach": "horizontal|top_down|diagonal_down|unknown", "preferred_approach_confidence": 0.0-1.0, "preferred_approach_reason": "...", "fallback_approaches": ["..."]}"""


def generate_query(
    object_name: str,
    scene_family: str = "",
    language: str = "auto",
) -> str:
    """Generate a diverse instruction query using weighted template strategy.

    Args:
        object_name: target object name
        scene_family: scene context (tabletop_tidy, delivery, storage)
        language: "zh", "en", or "auto" (random mix)

    Returns:
        Generated instruction query string
    """
    if language == "auto":
        language = random.choice(["zh", "en"])

    # Weighted selection: 40% object-centric, 30% scene-aware, 30% task-oriented
    roll = random.random()

    if roll < 0.4:
        # Object-centric
        templates = OBJECT_CENTRIC_ZH if language == "zh" else OBJECT_CENTRIC_EN
        return random.choice(templates).format(object_name=object_name)

    elif roll < 0.7:
        # Scene-aware
        templates = SCENE_AWARE_ZH if language == "zh" else SCENE_AWARE_EN
        if language == "zh":
            scene_desc = _get_scene_desc_zh(scene_family)
        else:
            scene_desc = _get_scene_desc_en(scene_family)
        return random.choice(templates).format(
            object_name=object_name, scene_desc=scene_desc
        )

    else:
        # Task-oriented
        templates = TASK_ORIENTED_ZH if language == "zh" else TASK_ORIENTED_EN
        return random.choice(templates).format(object_name=object_name)


def _get_scene_desc_zh(scene_family: str) -> str:
    mapping = {
        "tabletop_tidy": "桌面整理",
        "delivery": "物流分拣",
        "storage": "仓储收纳",
    }
    return mapping.get(scene_family, random.choice(SCENE_DESCRIPTIONS_ZH))


def _get_scene_desc_en(scene_family: str) -> str:
    mapping = {
        "tabletop_tidy": "tabletop tidying",
        "delivery": "delivery sorting",
        "storage": "warehouse storage",
    }
    return mapping.get(scene_family, random.choice(SCENE_DESCRIPTIONS_EN))
