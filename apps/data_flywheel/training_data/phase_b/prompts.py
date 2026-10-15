from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    language: str
    task_type: str
    template: str


PROMPT_TEMPLATES_V1: list[PromptTemplate] = [
    PromptTemplate("tpl_001_pick_en_simple", "en", "pick_named", "Please pick {target}"),
    PromptTemplate("tpl_002_pick_en_get", "en", "pick_named", "Get me {target}"),
    PromptTemplate("tpl_003_pick_zh_simple", "zh", "pick_named", "拿{target}"),
    PromptTemplate("tpl_004_pick_zh_polite", "zh", "pick_named", "帮我把{target}拿起来"),
    PromptTemplate("tpl_005_push_en", "en", "push_named", "Push {target} aside"),
    PromptTemplate("tpl_006_push_zh", "zh", "push_named", "把{target}推开"),
    PromptTemplate("tpl_007_blocker_en", "en", "pick_under_blocker", "Please pick {target} which is under {blocker}"),
    PromptTemplate("tpl_008_blocker_zh", "zh", "pick_under_blocker", "拿被{blocker}压住的{target}"),
    PromptTemplate("tpl_009_place_en", "en", "place_in_container", "Put {target} in {destination}"),
    PromptTemplate("tpl_010_place_zh", "zh", "place_in_container", "把{target}放到{destination}里"),
    PromptTemplate("tpl_011_cleanup_en", "en", "cleanup_surface", "Tidy the desk by putting {target} in {destination}"),
    PromptTemplate("tpl_012_cleanup_zh", "zh", "cleanup_surface", "收拾桌面，把{target}放到{destination}里"),
]


def fill_template(
    template: PromptTemplate,
    target: str,
    blocker: Optional[str] = None,
    destination: Optional[str] = None,
) -> str:
    text = template.template.replace("{target}", target)
    if blocker is not None and "{blocker}" in text:
        text = text.replace("{blocker}", blocker)
    if destination is not None and "{destination}" in text:
        text = text.replace("{destination}", destination)
    return text


def _pick_variant_templates_from_pool(
    task_type: str,
    variants: int,
    pool: Sequence[PromptTemplate],
) -> List[PromptTemplate]:
    candidates = [template for template in pool if template.task_type == task_type]
    if not candidates or variants <= 0:
        return []

    selected: List[PromptTemplate] = []

    def add_first(language: str) -> None:
        for template in candidates:
            if template.language == language and template not in selected:
                selected.append(template)
                break

    add_first("en")
    add_first("zh")

    for template in candidates:
        if template not in selected:
            selected.append(template)

    if len(selected) >= variants:
        return selected[:variants]

    repeat_index = 0
    while len(selected) < variants:
        selected.append(candidates[repeat_index % len(candidates)])
        repeat_index += 1

    return selected[:variants]


def pick_variant_templates(
    task_type: str,
    variants: int = 3,
) -> List[PromptTemplate]:
    return _pick_variant_templates_from_pool(task_type, variants, PROMPT_TEMPLATES_V1)
