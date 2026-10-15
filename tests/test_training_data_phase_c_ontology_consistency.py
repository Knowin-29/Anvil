from apps.data_flywheel.training_data.phase_c.validators.ontology_consistency import OntologyConsistencyValidator
from apps.data_flywheel.types import Action, SceneObjectSample, ThinkingTrace, TrainingSample


def _sample(actions: list[Action], scene_objects: list[SceneObjectSample] | None = None) -> TrainingSample:
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
        thinking=ThinkingTrace(steps=[]),
        actions=actions,
        rex_omni_text="",
    )


def test_pick_of_pick_passes():
    sample = _sample(
        [
            Action(
                type="pick",
                target="pen_0",
                grasp_point_pixel=[1.0, 1.0],
                grasp_point_quantized=[1, 1],
            )
        ]
    )

    result = OntologyConsistencyValidator().check(sample, {})

    assert result.passed is True


def test_pick_of_push_only_fails():
    sample = _sample(
        [
            Action(
                type="pick",
                target="card_0",
                grasp_point_pixel=[1.0, 1.0],
                grasp_point_quantized=[1, 1],
            )
        ]
    )

    result = OntologyConsistencyValidator().check(sample, {})

    assert result.passed is False
    assert "push_only" in result.reason


def test_push_of_card_passes():
    sample = _sample(
        [
            Action(
                type="push",
                target="card_0",
                grasp_point_pixel=[1.0, 1.0],
                grasp_point_quantized=[1, 1],
            )
        ]
    )

    result = OntologyConsistencyValidator().check(sample, {})

    assert result.passed is True


def test_place_allowed_when_target_is_pickable():
    sample = _sample(
        [
            Action(
                type="place",
                target="pen_0",
                destination="card_0",
                place_point_pixel=[2.0, 2.0],
                place_point_quantized=[2, 2],
                destination_type="free-space",
                destination_anchor_ids=["card_0"],
            )
        ]
    )

    result = OntologyConsistencyValidator().check(sample, {})

    assert result.passed is True


def test_pick_blocker_with_push_operation_allowed():
    sample = _sample(
        [
            Action(
                type="push",
                target="card_0",
                grasp_point_pixel=[1.0, 1.0],
                grasp_point_quantized=[1, 1],
            )
        ]
    )

    result = OntologyConsistencyValidator().check(sample, {})

    assert result.passed is True
