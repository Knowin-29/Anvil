from apps.data_flywheel.types import Action, ThinkingStep, ThinkingTrace
from apps.data_flywheel.training_data.phase_b.rex_omni_format import (
    _format_action_line,
    format_rex_omni_text,
)


def test_rex_omni_pick_format():
    action = Action(
        type="pick",
        target="pen_0",
        grasp_point_pixel=[326, 550],
        grasp_point_quantized=[254, 687],
    )
    line = _format_action_line(action, 1)
    assert "<|object_ref_start|>pen_0<|object_ref_end|>" in line
    assert "<254>" in line and "<687>" in line


def test_rex_omni_push_format():
    action = Action(
        type="push",
        target="card_0",
        grasp_point_pixel=[100, 200],
        grasp_point_quantized=[257, 687],
    )
    line = _format_action_line(action, 2)
    assert line.startswith("Action 2: push ")
    assert "<|object_ref_start|>card_0<|object_ref_end|>" in line
    assert "<257>" in line and "<687>" in line


def test_rex_omni_full_sample_has_all_tokens():
    thinking = ThinkingTrace(
        steps=[
            ThinkingStep(
                perception_type="ReferenceResolution",
                target="pen_0",
                content='"红笔" → pen_0 (matched by hierarchical_name)',
            )
        ]
    )
    actions = [
        Action(
            type="pick",
            target="pen_0",
            grasp_point_pixel=[326, 550],
            grasp_point_quantized=[254, 687],
        )
    ]

    text = format_rex_omni_text("拿那支笔", thinking, actions)
    assert "<|user|>拿那支笔<|end|>" in text
    assert "<|thinking_start|>" in text and "<|thinking_end|>" in text
    assert "<|action_start|>" in text and "<|action_end|>" in text
    assert "Action 1: pick" in text
