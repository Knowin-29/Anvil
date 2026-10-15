from apps.data_flywheel.training_data.phase_b.prompts import (
    PROMPT_TEMPLATES_V1,
    PromptTemplate,
    fill_template,
    pick_variant_templates,
)


def test_fill_template_single_slot():
    tpl = PromptTemplate("test", "en", "pick_named", "Please pick {target}")
    assert fill_template(tpl, target="the red pen") == "Please pick the red pen"


def test_fill_template_two_slots_blocker():
    tpl = PromptTemplate("test", "zh", "pick_under_blocker", "拿被{blocker}压住的{target}")
    assert fill_template(tpl, target="笔", blocker="书") == "拿被书压住的笔"


def test_fill_template_with_destination():
    tpl = PromptTemplate("test", "zh", "place_in_container", "把{target}放到{destination}里")
    assert fill_template(tpl, target="笔", destination="碗") == "把笔放到碗里"


def test_pick_variant_templates_returns_at_least_one_en_and_one_zh():
    variants = pick_variant_templates("pick_named", variants=3)
    langs = {template.language for template in variants}
    assert "en" in langs and "zh" in langs
    assert len(variants) == 3


def test_all_templates_have_unique_ids():
    ids = [template.id for template in PROMPT_TEMPLATES_V1]
    assert len(ids) == len(set(ids))


def test_templates_cover_all_task_types():
    types = {template.task_type for template in PROMPT_TEMPLATES_V1}
    assert types == {
        "pick_named",
        "push_named",
        "pick_under_blocker",
        "place_in_container",
        "cleanup_surface",
    }
