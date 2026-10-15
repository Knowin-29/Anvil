from apps.data_flywheel.gsa.a_action_interface_pipeline import AActionInterfacePipeline
from apps.data_flywheel.gsa.visual_evidence import GsaEvidenceBundle, GsaObjectRef


class _FakeVLM:
    def chat_json(self, **kwargs):
        return {
            "grasp_point_2d": [5.0, 5.0],
            "jaw_axis_2d": [1.0, 0.0],
            "push_contact_point_2d": [1.0, 5.0],
            "push_direction_2d": [-1.0, 0.0],
        }


class _ReasoningThenJsonVLM:
    def chat(self, **kwargs):
        if "push_contact_point_2d" in kwargs["prompt"]:
            return (
                "I considered the object boundary.\n"
                'FINAL_JSON: {"coordinate_frame":"pixel",'
                '"push_contact_point_2d":[1.0,5.0],"push_direction_2d":[-2.0,0.0]}'
            )
        return (
            "I considered the block shape first.\n"
            'FINAL_JSON: {"coordinate_frame":"pixel",'
            '"grasp_point_2d":[5.0,5.0],"jaw_axis_2d":[2.0,0.0]}'
        )


class _BBoxOnlyVLM:
    def chat(self, **kwargs):
        return "The object bbox is [0.0, 0.0, 10.0, 10.0]."


class _NormalizedVLM:
    def chat(self, **kwargs):
        return (
            'FINAL_JSON: {"coordinate_frame":"normalized_1000",'
            '"grasp_point_2d":[500,500],"jaw_axis_2d":[0,10]}'
        )


class _RepairOnlyVLM:
    def chat(self, **kwargs):
        prompt = kwargs["prompt"]
        if "Model response to convert" not in prompt:
            return "The best action point is the object center, around x=5 and y=5."
        if "push_contact_point_2d" in prompt:
            return (
                'FINAL_JSON: {"coordinate_frame":"pixel",'
                '"push_contact_point_2d":[1,5],"push_direction_2d":[-3,0]}'
            )
        return (
            'FINAL_JSON: {"coordinate_frame":"pixel",'
            '"grasp_point_2d":[5,5],"jaw_axis_2d":[3,0]}'
        )


def _bundle(image_path=None):
    return GsaEvidenceBundle(
        sample_id="sample_001",
        image_path=image_path,
        scene_family="tabletop_tidy",
        scenario_id="SC.2",
        objects=[
            GsaObjectRef(
                object_id="cup_0",
                object_name="cup",
                raw_object_name="cup",
                bbox=[0.0, 0.0, 10.0, 10.0],
            )
        ],
    )


def test_heuristic_interface_labels_are_not_auto_verifiable():
    labels = AActionInterfacePipeline().run(_bundle())

    assert labels
    assert all(label.source_provenance == "adapter" for label in labels)
    assert all(label.verifier_status == "heuristic_only" for label in labels)
    assert all(label.label_status == "candidate" for label in labels)
    assert all(label.stable_training_label is False for label in labels)
    assert all(label.label_id for label in labels)


def test_vlm_interface_labels_require_review():
    labels = AActionInterfacePipeline(_FakeVLM()).run(_bundle(image_path="/tmp/nonexistent.jpg"))

    assert labels
    assert all(label.source_provenance == "vlm" for label in labels)
    assert all(label.verifier_status == "review_required" for label in labels)
    assert all(label.label_status == "teacher_generated" for label in labels)
    assert all(label.stable_training_label is False for label in labels)
    assert all("not_auto_verifiable" in label.promotion_blockers for label in labels)
    assert all(label.object_name == "cup" for label in labels)
    assert all(label.bbox == [0.0, 0.0, 10.0, 10.0] for label in labels)


def test_reasoning_text_final_json_is_used_without_denormalizing_pixel_points():
    labels = AActionInterfacePipeline(_ReasoningThenJsonVLM()).run(
        _bundle(image_path="tests/fixtures/desk_sample.jpg")
    )

    pick = next(label for label in labels if label.action_type == "pick")
    push = next(label for label in labels if label.action_type == "push")

    assert pick.source_provenance == "vlm"
    assert pick.grasp_point_2d == [5.0, 5.0]
    assert pick.jaw_axis_2d == [1.0, 0.0]
    assert push.source_provenance == "vlm"
    assert push.push_contact_point_2d == [1.0, 5.0]
    assert push.push_direction_2d == [-1.0, 0.0]


def test_bbox_only_vlm_response_does_not_get_mistaken_for_action_json():
    labels = AActionInterfacePipeline(_BBoxOnlyVLM()).run(
        _bundle(image_path="tests/fixtures/desk_sample.jpg")
    )

    assert labels
    assert all(label.source_provenance == "adapter" for label in labels)
    assert all(label.verifier_status == "heuristic_only" for label in labels)


def test_explicit_normalized_1000_points_are_denormalized_to_pixels():
    label = AActionInterfacePipeline(_NormalizedVLM()).run(
        _bundle(image_path="tests/fixtures/desk_sample.jpg")
    )[0]

    assert label.source_provenance == "vlm"
    assert label.grasp_point_2d == [80.0, 60.0]
    assert label.jaw_axis_2d == [0.0, 1.0]


def test_reasoning_without_json_can_be_repaired_with_second_vlm_call():
    labels = AActionInterfacePipeline(_RepairOnlyVLM()).run(
        _bundle(image_path="tests/fixtures/desk_sample.jpg")
    )

    assert labels
    assert all(label.source_provenance == "vlm" for label in labels)
    pick = next(label for label in labels if label.action_type == "pick")
    push = next(label for label in labels if label.action_type == "push")
    assert pick.grasp_point_2d == [5.0, 5.0]
    assert pick.jaw_axis_2d == [1.0, 0.0]
    assert push.push_contact_point_2d == [1.0, 5.0]
    assert push.push_direction_2d == [-1.0, 0.0]
