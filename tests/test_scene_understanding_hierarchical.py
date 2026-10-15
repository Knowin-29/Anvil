from apps.data_flywheel.scene_understanding.enrich.hierarchical import generate_hierarchical_name
from apps.data_flywheel.types import InstanceAttributes, SceneNode


def _node(node_id: str, class_name: str, bbox, *, color=None, descriptor_axes=None):
    attrs = InstanceAttributes(color_primary=color)
    return SceneNode(
        id=node_id,
        class_name=class_name,
        bbox=bbox,
        attributes=attrs,
        descriptor_axes=list(descriptor_axes or []),
    )


def test_single_instance_uses_class_name():
    node = _node("pen_0", "pen", [0, 0, 10, 10])

    assert generate_hierarchical_name(node, [node]) == "the pen"


def test_two_distinct_colors():
    left = _node("pen_0", "pen", [0, 0, 10, 10], color="red")
    right = _node("pen_1", "pen", [20, 0, 30, 10], color="blue")

    assert generate_hierarchical_name(left, [left, right]) == "the red pen"
    assert generate_hierarchical_name(right, [left, right]) == "the blue pen"


def test_two_same_colors_use_position():
    left = _node("pen_0", "pen", [0, 0, 10, 10], color="red")
    right = _node("pen_1", "pen", [20, 0, 30, 10], color="red")

    assert generate_hierarchical_name(left, [left, right]) == "the left pen"
    assert generate_hierarchical_name(right, [left, right]) == "the right pen"


def test_three_instances_use_leftmost_middle_rightmost():
    left = _node("pen_0", "pen", [0, 0, 10, 10])
    middle = _node("pen_1", "pen", [20, 0, 30, 10])
    right = _node("pen_2", "pen", [40, 0, 50, 10])

    assert generate_hierarchical_name(left, [left, middle, right]) == "the leftmost pen"
    assert generate_hierarchical_name(middle, [left, middle, right]) == "the middle pen"
    assert generate_hierarchical_name(right, [left, middle, right]) == "the rightmost pen"


def test_four_plus_use_ordinal_from_left():
    nodes = [
        _node("pen_0", "pen", [0, 0, 10, 10]),
        _node("pen_1", "pen", [20, 0, 30, 10]),
        _node("pen_2", "pen", [40, 0, 50, 10]),
        _node("pen_3", "pen", [60, 0, 70, 10]),
    ]

    assert generate_hierarchical_name(nodes[0], nodes) == "the first pen from the left"
    assert generate_hierarchical_name(nodes[3], nodes) == "the fourth pen from the left"


def test_fallback_uses_neighbor():
    left = _node("cup_0", "cup", [0, 0, 10, 10])
    right = _node("cup_1", "cup", [0, 20, 10, 30])
    neighbor = _node("book_0", "book", [20, 0, 30, 10])

    assert "next to the book" in generate_hierarchical_name(left, [left, right, neighbor])


def test_fallback_final_uses_id_suffix():
    left = _node("cup_9", "cup", [0, 0, 10, 10])
    right = _node("cup_10", "cup", [0, 20, 10, 30])

    assert generate_hierarchical_name(left, [left, right]) == "the cup (cup_9)"


def test_descriptor_axes_layer_color_and_position_for_same_color_objects():
    left = _node(
        "plate_0",
        "plate",
        [0, 0, 10, 10],
        color="white",
        descriptor_axes=["color", "support_context", "relative_position"],
    )
    right = _node(
        "plate_1",
        "plate",
        [20, 0, 30, 10],
        color="white",
        descriptor_axes=["color", "support_context", "relative_position"],
    )

    assert generate_hierarchical_name(left, [left, right]) == "the left white plate"
    assert generate_hierarchical_name(right, [left, right]) == "the right white plate"


def test_descriptor_axes_use_compact_position_labels_for_three_colored_objects():
    left = _node(
        "pen_0",
        "pen",
        [0, 0, 10, 10],
        color="red",
        descriptor_axes=["color", "relative_position"],
    )
    middle = _node(
        "pen_1",
        "pen",
        [20, 0, 30, 10],
        color="red",
        descriptor_axes=["color", "relative_position"],
    )
    right = _node(
        "pen_2",
        "pen",
        [40, 0, 50, 10],
        color="red",
        descriptor_axes=["color", "relative_position"],
    )

    assert generate_hierarchical_name(left, [left, middle, right]) == "the left red pen"
    assert (
        generate_hierarchical_name(middle, [left, middle, right])
        == "the middle red pen"
    )
    assert generate_hierarchical_name(right, [left, middle, right]) == "the right red pen"


def test_ontology_identity_groups_aliases_for_layered_naming():
    left = SceneNode(
        id="cup_0",
        class_name="cup",
        bbox=[0, 0, 10, 10],
        object_class_id="cup",
        canonical_name="杯子",
        descriptor_axes=["relative_position"],
        attributes=InstanceAttributes(color_primary="white"),
    )
    right = SceneNode(
        id="mug_0",
        class_name="mug",
        bbox=[20, 0, 30, 10],
        object_class_id="cup",
        canonical_name="杯子",
        descriptor_axes=["relative_position"],
        attributes=InstanceAttributes(color_primary="white"),
    )

    assert generate_hierarchical_name(left, [left, right]) == "the left cup"
    assert generate_hierarchical_name(right, [left, right]) == "the right cup"


def test_descriptor_priority_can_use_support_context_when_color_is_not_unique():
    tray = SceneNode(
        id="tray_0",
        class_name="tray",
        bbox=[0, 40, 80, 100],
        role_ids=["container", "support_surface"],
    )
    plate = SceneNode(
        id="plate_0",
        class_name="plate",
        bbox=[10, 50, 30, 70],
        support_parent="tray_0",
        support_type="inside",
        descriptor_axes=["color", "support_context", "near_anchor"],
        attributes=InstanceAttributes(color_primary="white"),
    )

    assert generate_hierarchical_name(plate, [plate, tray]) == "the white plate in the tray"
