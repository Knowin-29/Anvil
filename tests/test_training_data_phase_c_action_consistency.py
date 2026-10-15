from apps.data_flywheel.training_data.phase_c.validators.action_consistency import ActionConsistencyValidator
from apps.data_flywheel.types import Action, SceneObjectSample, ThinkingStep, ThinkingTrace, TrainingSample


def _sample(
    actions: list[Action],
    plan_order_output: list[str] | None = None,
    scene_objects: list[SceneObjectSample] | None = None,
) -> TrainingSample:
    if plan_order_output is None:
        plan_order_output = [f"{action.type}:{action.target}" for action in actions]
    if scene_objects is None:
        scene_objects = [
            SceneObjectSample(
                id="pen_0",
                class_name="pen",
                bbox_pixel=[10.0, 10.0, 30.0, 30.0],
                bbox_quantized=[10, 10, 30, 30],
                operation="pick",
            ),
            SceneObjectSample(
                id="card_0",
                class_name="card",
                bbox_pixel=[40.0, 40.0, 50.0, 50.0],
                bbox_quantized=[40, 40, 50, 50],
                operation="push",
                extra_tags=["push_only"],
            ),
        ]
    return TrainingSample(
        sample_id="sample-1",
        image_id="img-1",
        image_path="img-1.jpg",
        image_size=(100, 100),
        data_source="real",
        instruction="pick pen",
        instruction_template_id="tpl",
        instruction_language="en",
        scene_objects=scene_objects,
        blocking_dag=[],
        thinking=ThinkingTrace(
            steps=[
                ThinkingStep(perception_type="ReferenceResolution", target="pen_0", content="resolved"),
                ThinkingStep(perception_type="Affordance", target="pen_0", content="operation=pick"),
                ThinkingStep(perception_type="PlanOrder", target="", content=" | ".join(plan_order_output), output=plan_order_output),
            ]
        ),
        actions=actions,
        rex_omni_text="",
    )


def test_action_target_must_exist():
    sample = _sample(
        [
            Action(
                type="pick",
                target="missing",
                grasp_point_pixel=[1.0, 1.0],
                grasp_point_quantized=[1, 1],
            )
        ]
    )

    result = ActionConsistencyValidator().check(sample, {})

    assert result.passed is False
    assert "target 'missing'" in result.reason


def test_pick_without_grasp_point_fails():
    sample = _sample(
        [
            Action(
                type="pick",
                target="pen_0",
            )
        ]
    )

    result = ActionConsistencyValidator().check(sample, {})

    assert result.passed is False
    assert "missing grasp_point_pixel" in result.reason


def test_push_with_destination_fails():
    sample = _sample(
        [
            Action(
                type="push",
                target="card_0",
                grasp_point_pixel=[1.0, 1.0],
                grasp_point_quantized=[1, 1],
                destination="bin_0",
            )
        ]
    )

    result = ActionConsistencyValidator().check(sample, {})

    assert result.passed is False
    assert "should not have destination" in result.reason


def test_place_without_destination_fails():
    sample = _sample(
        [
            Action(
                type="place",
                target="pen_0",
                place_point_pixel=[1.0, 1.0],
                place_point_quantized=[1, 1],
            )
        ],
        plan_order_output=["place:pen_0"],
    )

    result = ActionConsistencyValidator().check(sample, {})

    assert result.passed is False
    assert "place missing destination" in result.reason


def test_place_without_destination_type_fails():
    sample = _sample(
        [
            Action(
                type="place",
                target="pen_0",
                destination="card_0",
                place_point_pixel=[1.0, 1.0],
                place_point_quantized=[1, 1],
            )
        ],
        plan_order_output=["place:pen_0"],
    )

    result = ActionConsistencyValidator().check(sample, {})

    assert result.passed is False
    assert "place missing destination_type" in result.reason


def test_pick_with_place_point_quantized_fails():
    sample = _sample(
        [
            Action(
                type="pick",
                target="pen_0",
                grasp_point_pixel=[1.0, 1.0],
                grasp_point_quantized=[1, 1],
                place_point_quantized=[2, 2],
            )
        ]
    )

    result = ActionConsistencyValidator().check(sample, {})

    assert result.passed is False
    assert "should not have place_point_quantized" in result.reason


def test_plan_order_mismatch_fails():
    sample = _sample(
        [
            Action(
                type="pick",
                target="pen_0",
                grasp_point_pixel=[1.0, 1.0],
                grasp_point_quantized=[1, 1],
            )
        ],
        plan_order_output=["push:card_0"],
    )

    result = ActionConsistencyValidator().check(sample, {})

    assert result.passed is False
    assert "PlanOrder mismatch" in result.reason


def test_place_accepts_additive_placement_metadata():
    sample = _sample(
        [
            Action(
                type="place",
                target="pen_0",
                destination="bowl_0",
                place_point_pixel=[12.0, 14.0],
                place_point_quantized=[12, 14],
                destination_type="container",
                destination_anchor_ids=["bowl_0"],
                placement_reason="preferred_container",
            )
        ],
        plan_order_output=["place:pen_0"],
        scene_objects=[
            SceneObjectSample(
                id="pen_0",
                class_name="pen",
                bbox_pixel=[10.0, 10.0, 30.0, 30.0],
                bbox_quantized=[10, 10, 30, 30],
                operation="pick",
            ),
            SceneObjectSample(
                id="bowl_0",
                class_name="bowl",
                bbox_pixel=[60.0, 60.0, 90.0, 90.0],
                bbox_quantized=[60, 60, 90, 90],
                operation="pick",
                is_container=True,
            ),
        ],
    )

    result = ActionConsistencyValidator().check(sample, {})

    assert result.passed is True
    assert result.reason is None
