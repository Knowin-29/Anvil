from pathlib import Path

from apps.data_flywheel.gsa.a_action_precondition_pipeline import AActionPreconditionPipeline
from apps.data_flywheel.gsa.a_action_interface_pipeline import AActionInterfacePipeline
from apps.data_flywheel.gsa.a_action_interface_review import (
    promote_action_interface_candidates,
    read_jsonl,
    write_jsonl,
)
from apps.data_flywheel.gsa.runner import GSAv0Runner
from apps.data_flywheel.gsa.taxonomy import Bucket, GSubtype, SourceProvenance, VerifierStatus
from apps.data_flywheel.gsa.visual_evidence import GsaEvidenceBundle, GsaObjectRef


def _vlm_candidate():
    return {
        "label_id": "aif_unit_pick",
        "schema_version": "a_action_interface_candidate_v1",
        "sample_id": "sample_001",
        "image_path": "tests/fixtures/desk_sample.jpg",
        "image_width": 160,
        "image_height": 120,
        "object_id": "cup_0",
        "object_name": "cup",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "bucket": "A_action_interface",
        "subtype": "grasp_point",
        "action_type": "pick",
        "label_type": "point",
        "source_provenance": "vlm",
        "verifier_status": "review_required",
        "label_status": "teacher_generated",
        "stable_training_label": False,
        "grasp_point_2d": [5.0, 5.0],
        "jaw_axis_2d": [1.0, 0.0],
        "risk_flags": [],
    }


def test_vlm_candidate_does_not_promote_without_approval():
    result = promote_action_interface_candidates([_vlm_candidate()])

    assert result.stable_labels == []
    assert result.summary["stable_training_label_count"] == 0
    assert "review_decision_pending" in result.candidates[0]["promotion_blockers"]


def test_approved_valid_vlm_candidate_promotes_to_stable_label():
    candidate = _vlm_candidate()
    result = promote_action_interface_candidates(
        [candidate],
        {candidate["label_id"]: {"label_id": candidate["label_id"], "decision": "approve"}},
    )

    assert len(result.stable_labels) == 1
    stable = result.stable_labels[0]
    assert stable["label_status"] == "stable_training_label"
    assert stable["stable_training_label"] is True
    assert stable["verifier_status"] == "auto_verifiable"


def test_heuristic_candidate_cannot_promote_even_if_approved():
    candidate = _vlm_candidate()
    candidate["label_id"] = "aif_unit_heuristic"
    candidate["source_provenance"] = "adapter"
    candidate["verifier_status"] = "heuristic_only"
    candidate["label_status"] = "candidate"

    result = promote_action_interface_candidates(
        [candidate],
        {candidate["label_id"]: {"label_id": candidate["label_id"], "decision": "approve"}},
    )

    assert result.stable_labels == []
    assert "heuristic_only" in result.candidates[0]["promotion_blockers"]
    assert "source_not_vlm" in result.candidates[0]["promotion_blockers"]


def test_reject_and_invalid_geometry_do_not_enter_pack(tmp_path):
    rejected = _vlm_candidate()
    invalid = _vlm_candidate()
    invalid["label_id"] = "aif_unit_invalid"
    invalid["grasp_point_2d"] = [50.0, 50.0]
    result = promote_action_interface_candidates(
        [rejected, invalid],
        {
            rejected["label_id"]: {"label_id": rejected["label_id"], "decision": "reject"},
            invalid["label_id"]: {"label_id": invalid["label_id"], "decision": "approve"},
        },
    )

    pack_path = tmp_path / "A_action_interface_pack_manifest.jsonl"
    write_jsonl(pack_path, result.stable_labels)

    assert read_jsonl(pack_path) == []
    assert "review_decision_reject" in result.candidates[0]["promotion_blockers"]
    assert "grasp_point_2d outside bbox" in result.candidates[1]["promotion_blockers"]


class _CountingVLM:
    def __init__(self):
        self.calls = 0

    def chat(self, prompt, **kwargs):
        self.calls += 1
        return '{"can_do_now": true, "blocked_by": [], "need_pre_action": "none", "confidence": 0.9}'


class _ListVLM:
    def chat(self, prompt, **kwargs):
        return '[{"unexpected": true}]'


def test_action_precondition_queries_once_per_object_not_once_per_action():
    vlm = _CountingVLM()
    bundle = GsaEvidenceBundle(
        sample_id="sample_001",
        image_path="tests/fixtures/desk_sample.jpg",
        objects=[
            GsaObjectRef(
                object_id="cup_0",
                object_name="cup",
                raw_object_name="cup",
                bbox=[0.0, 0.0, 10.0, 10.0],
            )
        ],
    )

    labels = AActionPreconditionPipeline(vlm).run(bundle)

    # Current spatial-aware contract assesses pick; do not reuse one answer
    # as unverified push and place feasibility.
    assert len(labels) == 1
    assert vlm.calls == 1


def test_action_precondition_ignores_non_object_vlm_json():
    bundle = GsaEvidenceBundle(
        sample_id="sample_001",
        image_path="tests/fixtures/desk_sample.jpg",
        objects=[
            GsaObjectRef(
                object_id="cup_0",
                object_name="cup",
                raw_object_name="cup",
                bbox=[0.0, 0.0, 10.0, 10.0],
            )
        ],
    )

    labels = AActionPreconditionPipeline(_ListVLM()).run(bundle)

    assert len(labels) == 1
    assert all(label.can_do_now for label in labels)


class _FakeGroundingWorker:
    def generate(self, image_path, sample_id, scene_family="", scenario_id=""):
        return [
            {
                "sample_id": sample_id,
                "bucket": Bucket.G.value,
                "subtype": GSubtype.SIMPLE_GROUNDING.value,
                "scene_family": scene_family,
                "scenario_id": scenario_id,
                "object_id": "cup_0",
                "question": "Find the cup.",
                "answer": {"bbox": [0.0, 0.0, 10.0, 10.0], "object_name": "cup"},
                "verifier_status": VerifierStatus.AUTO_VERIFIABLE.value,
                "source_provenance": SourceProvenance.VLM.value,
            }
        ]


class _FakeSpatialWorker:
    def generate(self, image_path, sample_id, objects, scene_family="", scenario_id=""):
        return []

    def extract_spatial_facts(self, labels):
        return []


class _EmptyPipeline:
    def run(self, bundle):
        return []


class _EmptyActionPipeline:
    def run(self, bundle):
        return [], []


class _FakeActionVLM:
    def chat(self, **kwargs):
        if "push_contact_point_2d" in kwargs["prompt"]:
            return (
                'FINAL_JSON: {"coordinate_frame":"pixel",'
                '"push_contact_point_2d":[1,5],"push_direction_2d":[-1,0]}'
            )
        return (
            'FINAL_JSON: {"coordinate_frame":"pixel",'
            '"grasp_point_2d":[5,5],"jaw_axis_2d":[1,0]}'
        )


def test_full_runner_output_can_feed_action_interface_promotion(tmp_path):
    runner = GSAv0Runner(config_path="configs/anvil.yaml")
    runner.config["pipeline"]["enabled_stages"] = ["G", "S", "A_action_interface"]
    runner.run_id = "gsa_v0_unit"
    runner.output_root = tmp_path / "gsa_v0_unit"
    runner.g_worker = _FakeGroundingWorker()
    runner.s_worker = _FakeSpatialWorker()
    runner.a_state_pipeline = _EmptyPipeline()
    runner.a_action_pipeline = _EmptyActionPipeline()
    runner.a_identity_attr = _EmptyPipeline()
    runner.a_obj_state = _EmptyPipeline()
    runner.a_robot_state = _EmptyPipeline()
    runner.a_action_precondition = _EmptyPipeline()
    runner.a_action_interface = AActionInterfacePipeline(_FakeActionVLM())

    runner.run(image_paths=["tests/fixtures/desk_sample.jpg"])

    labels_path = runner.output_root / "labels" / "A_action_interface" / "A_action_interface.jsonl"
    labels = read_jsonl(labels_path)
    assert labels
    assert all(row["bucket"] == "A_action_interface" for row in labels)

    first = labels[0]
    result = promote_action_interface_candidates(
        labels,
        {first["label_id"]: {"label_id": first["label_id"], "decision": "approve"}},
    )

    assert len(result.stable_labels) == 1


def test_runner_filters_background_objects_before_downstream_stages():
    runner = GSAv0Runner(config_path="scripts/configs/gsa_v0.yaml")
    runner.config.setdefault("input", {})["downstream_exclude_object_names"] = ["wall", "table"]
    runner.config["input"]["max_objects_per_image"] = 1
    objects = [
        GsaObjectRef("wall_0", "wall", "wall", bbox=[0.0, 0.0, 100.0, 100.0]),
        GsaObjectRef("cup_0", "cup", "cup", bbox=[10.0, 10.0, 20.0, 20.0]),
        GsaObjectRef("can_0", "can", "can", bbox=[30.0, 30.0, 40.0, 40.0]),
    ]

    filtered = runner._filter_downstream_objects(objects, "tests/fixtures/desk_sample.jpg")

    assert [obj.object_id for obj in filtered] == ["cup_0"]


def test_runner_prefers_manipulable_objects_before_downstream_limit():
    runner = GSAv0Runner(config_path="scripts/configs/gsa_v0.yaml")
    runner.config.setdefault("input", {})["downstream_preferred_object_names"] = ["cup", "can"]
    runner.config["input"]["max_objects_per_image"] = 1
    objects = [
        GsaObjectRef("stand_0", "stand", "stand", bbox=[0.0, 0.0, 10.0, 10.0]),
        GsaObjectRef("cup_0", "cup", "cup", bbox=[10.0, 10.0, 20.0, 20.0]),
        GsaObjectRef("can_0", "can", "can", bbox=[30.0, 30.0, 40.0, 40.0]),
    ]

    filtered = runner._filter_downstream_objects(objects, "tests/fixtures/desk_sample.jpg")

    assert [obj.object_id for obj in filtered] == ["cup_0"]
