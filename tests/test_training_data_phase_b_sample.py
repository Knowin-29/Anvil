import json

from apps.data_flywheel.types import (
    Action,
    InstanceAttributes,
    SceneEdge,
    SceneGraph,
    SceneNode,
    ThinkingStep,
    ThinkingTrace,
)
from apps.data_flywheel.training_data.phase_a.core import PlanResult
from apps.data_flywheel.training_data.phase_b.sample import (
    build_scene_object_sample,
    synthesize_samples,
    write_samples_jsonl,
)


def _make_scene_graph() -> SceneGraph:
    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[10, 10, 20, 20],
        grasp_point=[15, 15],
        operation="pick",
        hierarchical_name="red pen",
        referring_expressions=["红笔", "红色的笔"],
        attributes=InstanceAttributes(color_primary="red"),
    )
    card = SceneNode(
        id="card_0",
        class_name="card",
        bbox=[30, 30, 40, 40],
        grasp_point=[35, 35],
        operation="push",
        extra_tags=["push_only"],
        hierarchical_name="card",
        referring_expressions=["卡片"],
    )
    return SceneGraph(
        image_id="scene_1",
        image_path="/tmp/scene_1.png",
        image_size=(100, 100),
        nodes=[pen, card],
        edges=[SceneEdge(subject="card_0", object="pen_0", predicate="occluding", source="test", confidence=1.0)],
        blocking_dag=[("card_0", "pen_0")],
    )


def _make_pick_place_scene_graph() -> SceneGraph:
    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[10, 10, 20, 20],
        grasp_point=[15, 15],
        operation="pick",
        hierarchical_name="red pen",
        referring_expressions=["红笔", "红色的笔"],
        attributes=InstanceAttributes(color_primary="red"),
    )
    bowl = SceneNode(
        id="bowl_0",
        class_name="bowl",
        bbox=[60, 60, 90, 90],
        operation="pick",
        is_container=True,
        hierarchical_name="blue bowl",
        referring_expressions=["蓝碗", "碗"],
        attributes=InstanceAttributes(color_primary="blue"),
    )
    return SceneGraph(
        image_id="scene_place",
        image_path="/tmp/scene_place.png",
        image_size=(100, 100),
        nodes=[pen, bowl],
        edges=[],
        blocking_dag=[],
    )


def _make_pick_plan_result(status: str = "success") -> PlanResult:
    thinking = ThinkingTrace(
        steps=[
            ThinkingStep(perception_type="ReferenceResolution", target="pen_0", content='"笔" → pen_0'),
            ThinkingStep(perception_type="Affordance", target="pen_0", content="operation=pick"),
            ThinkingStep(perception_type="PlanOrder", target="", content="pick pen_0"),
        ]
    )
    actions = [
        Action(
            type="pick",
            target="pen_0",
            grasp_point_pixel=[15, 15],
            grasp_point_quantized=[150, 150],
            reason="main target",
        ),
    ]
    return PlanResult(
        instruction="拿笔",
        resolved_target_id="pen_0",
        thinking=thinking,
        actions=actions,
        status=status,
        notes=[],
    )


def _make_blocker_plan_result(status: str = "success") -> PlanResult:
    thinking = ThinkingTrace(
        steps=[
            ThinkingStep(perception_type="ReferenceResolution", target="pen_0", content='"笔" → pen_0'),
            ThinkingStep(perception_type="Affordance", target="pen_0", content="operation=pick"),
            ThinkingStep(perception_type="PlanOrder", target="", content="push card_0 → pick pen_0"),
        ]
    )
    actions = [
        Action(
            type="push",
            target="card_0",
            grasp_point_pixel=[35, 35],
            grasp_point_quantized=[350, 350],
            reason="blocker cleanup for pen_0",
        ),
        Action(
            type="pick",
            target="pen_0",
            grasp_point_pixel=[15, 15],
            grasp_point_quantized=[150, 150],
            reason="main target",
        ),
    ]
    return PlanResult(
        instruction="拿被卡片压住的笔",
        resolved_target_id="pen_0",
        thinking=thinking,
        actions=actions,
        status=status,
        notes=[],
    )


def _make_pick_place_plan_result(status: str = "success") -> PlanResult:
    thinking = ThinkingTrace(
        steps=[
            ThinkingStep(perception_type="ReferenceResolution", target="pen_0", content='"红笔" → pen_0'),
            ThinkingStep(perception_type="Affordance", target="pen_0", content="operation=pick"),
            ThinkingStep(perception_type="ContainerLookup", target="bowl_0", content="bowl_0 (explicit_destination)"),
            ThinkingStep(perception_type="PlanOrder", target="", content="pick pen_0 → place pen_0"),
        ]
    )
    actions = [
        Action(
            type="pick",
            target="pen_0",
            grasp_point_pixel=[15, 15],
            grasp_point_quantized=[150, 150],
            reason="main target",
        ),
        Action(
            type="place",
            target="pen_0",
            destination="bowl_0",
            place_point_pixel=[75, 75],
            place_point_quantized=[750, 750],
            reason="main target placement",
            destination_type="container",
            destination_anchor_ids=["bowl_0"],
            placement_reason="explicit_destination",
        ),
    ]
    return PlanResult(
        instruction="把红笔放到蓝碗里",
        resolved_target_id="pen_0",
        thinking=thinking,
        actions=actions,
        status=status,
        notes=[],
    )


def test_build_scene_object_sample_copies_fields():
    scene_graph = _make_scene_graph()
    sample = build_scene_object_sample(scene_graph.nodes[0], scene_graph.image_size)

    assert sample.id == "pen_0"
    assert sample.bbox_pixel == [10, 10, 20, 20]
    assert sample.bbox_quantized == [100, 100, 200, 200]
    assert sample.grasp_point_quantized == [150, 150]
    assert sample.hierarchical_name == "red pen"
    assert sample.color_primary == "red"


def test_synthesize_three_variants_per_plan():
    scene_graph = _make_scene_graph()
    samples = synthesize_samples(
        image_path=scene_graph.image_path,
        image_size=scene_graph.image_size,
        scene_graph=scene_graph,
        plan_result=_make_pick_plan_result(),
    )

    assert len(samples) == 3
    assert len({sample.sample_id for sample in samples}) == 3
    assert all(sample.data_source == "real" for sample in samples)


def test_synthesize_zh_template_uses_zh_referring_expression():
    scene_graph = _make_scene_graph()
    samples = synthesize_samples(
        image_path=scene_graph.image_path,
        image_size=scene_graph.image_size,
        scene_graph=scene_graph,
        plan_result=_make_blocker_plan_result(),
    )

    assert len(samples) == 3
    zh_sample = next(sample for sample in samples if sample.instruction_language == "zh")
    assert "红笔" in zh_sample.instruction
    assert "卡片" in zh_sample.instruction
    assert zh_sample.metadata["num_blockers"] == 1
    assert zh_sample.metadata["has_card"] is True
    assert zh_sample.metadata["num_actions"] == 2
    assert zh_sample.metadata["variant_group_id"] == "scene_1_pick_under_blocker"


def test_synthesize_skip_when_plan_not_success():
    scene_graph = _make_scene_graph()
    assert (
        synthesize_samples(
            image_path=scene_graph.image_path,
            image_size=scene_graph.image_size,
            scene_graph=scene_graph,
            plan_result=_make_pick_plan_result(status="unresolved"),
        )
        == []
    )


def test_jsonl_round_trip(tmp_path):
    scene_graph = _make_scene_graph()
    sample = synthesize_samples(
        image_path=scene_graph.image_path,
        image_size=scene_graph.image_size,
        scene_graph=scene_graph,
        plan_result=_make_pick_plan_result(),
    )[0]

    output_path = tmp_path / "samples.jsonl"
    write_samples_jsonl([sample], str(output_path))

    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1

    row = rows[0]
    assert row["sample_id"] == sample.sample_id
    assert row["rex_omni_text"] == sample.rex_omni_text
    assert row["metadata"] == sample.metadata
    assert row["scene_objects"][0]["hierarchical_name"] == "red pen"


def test_build_training_sample_persists_place_metadata():
    scene_graph = _make_pick_place_scene_graph()
    sample = synthesize_samples(
        image_path=scene_graph.image_path,
        image_size=scene_graph.image_size,
        scene_graph=scene_graph,
        plan_result=_make_pick_place_plan_result(),
        variants_per_plan=1,
    )[0]

    place_action = sample.actions[-1]

    assert place_action.type == "place"
    assert place_action.place_point_pixel is not None
    assert place_action.destination_type == "container"
    assert place_action.destination_anchor_ids == ["bowl_0"]
    assert sample.metadata["num_actions"] == 2
    assert sample.metadata["has_place"] is True
    assert sample.metadata["destination_types"] == ["container"]
