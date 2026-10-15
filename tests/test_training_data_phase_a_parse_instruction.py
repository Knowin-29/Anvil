from apps.data_flywheel.training_data.phase_a.core import parse_instruction


def test_parse_instruction_pick_en():
    result = parse_instruction("pick the red pen")
    assert result.task_type == "pick_named"
    assert result.target_phrase == "the red pen"
    assert result.language == "en"


def test_parse_instruction_pick_zh():
    result = parse_instruction("拿红色的笔")
    assert result.task_type == "pick_named"
    assert result.target_phrase == "红色的笔"
    assert result.language == "zh"


def test_parse_instruction_push():
    result = parse_instruction("推那张卡片")
    assert result.task_type == "push_named"


def test_parse_instruction_blocker():
    result = parse_instruction("拿被书压住的笔")
    assert result.task_type == "pick_under_blocker"
    assert result.blocker_phrase == "书"
    assert result.target_phrase == "笔"


def test_parse_instruction_place_is_supported():
    result = parse_instruction("把笔放到笔筒里")
    assert result.task_type == "place_in_container"
    assert result.target_phrase == "笔"
    assert result.destination_phrase == "笔筒"
    assert result.language == "zh"


def test_parse_instruction_tidy_is_supported():
    result = parse_instruction("收拾桌面")
    assert result.task_type == "cleanup_surface"
    assert result.target_phrase == ""
    assert result.language == "zh"


def test_parse_instruction_unknown_fallback_to_scope_violation():
    result = parse_instruction("dance for me")
    assert result.task_type == "scope_violation"
