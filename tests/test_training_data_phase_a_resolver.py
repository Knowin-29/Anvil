from apps.data_flywheel.training_data.phase_a.resolver import resolve_reference
from apps.data_flywheel.types import InstanceAttributes, SceneNode


def _node(
    node_id: str,
    class_name: str,
    bbox,
    *,
    hierarchical_name=None,
    referring_expressions=None,
    color=None,
):
    return SceneNode(
        id=node_id,
        class_name=class_name,
        bbox=bbox,
        hierarchical_name=hierarchical_name,
        referring_expressions=list(referring_expressions or []),
        attributes=InstanceAttributes(color_primary=color),
    )


def test_resolve_by_hierarchical_name_exact():
    exact = _node("pen_exact", "pen", [0, 0, 10, 10], hierarchical_name="the red pen")
    fallback = _node("pen_other", "pen", [20, 0, 30, 10], color="red")

    result = resolve_reference("the red pen", [exact, fallback])

    assert result.status == "resolved"
    assert result.node_id == "pen_exact"
    assert result.matched_rule == "hierarchical_name_exact"


def test_resolve_by_referring_expressions():
    exact = _node(
        "pen_ref",
        "pen",
        [0, 0, 10, 10],
        referring_expressions=["the pen next to the book"],
    )
    fallback = _node("pen_other", "pen", [20, 0, 30, 10], color="blue")

    result = resolve_reference("the pen next to the book", [exact, fallback])

    assert result.status == "resolved"
    assert result.node_id == "pen_ref"
    assert result.matched_rule == "referring_expressions_exact"


def test_resolve_by_class_and_color_unique():
    red_pen = _node("pen_red", "pen", [0, 0, 10, 10], color="red")
    blue_pen = _node("pen_blue", "pen", [20, 0, 30, 10], color="blue")

    result = resolve_reference("red pen", [red_pen, blue_pen])

    assert result.status == "resolved"
    assert result.node_id == "pen_red"
    assert result.matched_rule == "class_color"


def test_resolve_by_class_and_color_ambiguous_returns_candidates():
    left = _node("pen_left", "pen", [0, 0, 10, 10], color="red")
    right = _node("pen_right", "pen", [20, 0, 30, 10], color="red")
    other = _node("pen_blue", "pen", [40, 0, 50, 10], color="blue")

    result = resolve_reference("red pen", [left, right, other])

    assert result.status == "ambiguous"
    assert result.node_id is None
    assert result.candidates == ["pen_left", "pen_right"]
    assert result.matched_rule == "class_ambiguous"


def test_resolve_by_position_leftmost():
    left = _node("pen_left", "pen", [0, 0, 10, 10])
    middle = _node("pen_middle", "pen", [20, 0, 30, 10])
    right = _node("pen_right", "pen", [40, 0, 50, 10])

    result = resolve_reference("leftmost pen", [left, middle, right])

    assert result.status == "resolved"
    assert result.node_id == "pen_left"
    assert result.matched_rule == "class_position"


def test_resolve_by_ordinal():
    first = _node("pen_1", "笔", [0, 0, 10, 10])
    second = _node("pen_2", "笔", [20, 0, 30, 10])
    third = _node("pen_3", "笔", [40, 0, 50, 10])

    result = resolve_reference("左边第二个笔", [first, second, third], language="zh")

    assert result.status == "resolved"
    assert result.node_id == "pen_2"
    assert result.matched_rule == "class_position"


def test_resolve_by_ordinal_english_from_left():
    first = _node("pen_1", "pen", [0, 0, 10, 10])
    second = _node("pen_2", "pen", [20, 0, 30, 10])
    third = _node("pen_3", "pen", [40, 0, 50, 10])

    result = resolve_reference("third pen from the left", [first, second, third])

    assert result.status == "resolved"
    assert result.node_id == "pen_3"
    assert result.matched_rule == "class_position"


def test_resolve_not_found():
    first = _node("pen_1", "pen", [0, 0, 10, 10], color="red")
    second = _node("pen_2", "pen", [20, 0, 30, 10], color="blue")

    result = resolve_reference("green cup", [first, second])

    assert result.status == "not_found"
    assert result.node_id is None
    assert result.candidates == []

