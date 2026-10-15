"""Offline contract and orchestration tests; not a model-quality benchmark."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from PIL import Image
from pydantic import ValidationError

from trains.flywheel.alignment_schema import (
    FORMS, SPATIAL_TASKS, NATIVE_CONTRACTS, Candidate, Scene, Task, student_prompt, validate_candidate,
)
from trains.flywheel.alignment import Ledger, align_scene, export_records, score_predictions
from trains.flywheel.alignment_schema import validate_training_record, Spatial, answer_schema
from trains.flywheel.alignment_sources import spatialsense_scenes


@pytest.fixture
def prism_depth_export_fixture(tmp_path):
    """Synthetic policy/data fixture; never a production scene approval."""
    import random
    from trains.flywheel.alignment_schema import Candidate
    from tools.source_runtime import sha, write_rows
    image_root = tmp_path / 'raw'
    image_root.mkdir()
    packets, views = [], []
    groups = ['PRISM:' + char * 32 for char in 'abc']
    for index, group in enumerate(groups):
        scene_id = group.removeprefix('PRISM:')
        folder = image_root / scene_id
        folder.mkdir()
        image = folder / 'view_0.png'
        pixels = random.Random(index + 80).randbytes(32 * 32 * 3)
        Image.frombytes('RGB', (32, 32), pixels).save(image)
        entities = [{'entity_id': 'one', 'label': 'object_one', 'bbox': [0, 0, 400, 400]},
                    {'entity_id': 'two', 'label': 'object_two', 'bbox': [500, 500, 900, 900]}]
        relation = {'task': 'relative_depth', 'subject_id': 'one', 'object_id': 'two',
                    'predicate': 'nearer', 'frame': 'camera', 'truth': True}
        raw_scene = {'source': 'PRISM', 'scene_id': scene_id + ':view_0', 'group_id': group,
            'images': [str(image)], 'source_split': 'train', 'training_approved': False,
            'entities': entities, 'evidence': [{'evidence_id': 'camera_depth', 'kind': 'derived_geometry',
                'description': 'Synthetic test depth intervals.', 'entity_ids': ['one', 'two'], 'relation': relation}],
            'tasks': [{'task_id': 'ground', 'form': 'detection', 'native_form': 'detection',
                'question': 'Locate the two visible instances.', 'entity_ids': ['one', 'two']},
                {'task_id': 'depth', 'form': 'spatial', 'native_form': 'depth_order',
                'spatial_task': 'relative_depth', 'question': 'Is one nearer or farther than two?',
                'entity_ids': ['one', 'two']}]}
        scene_value = Scene.model_validate(raw_scene)
        reference = Candidate.model_validate({'task_id': 'depth', 'form': 'spatial',
            'answer': [{**relation, 'evidence_ids': ['camera_depth']}],
            'reasoning_trace': [{'evidence_ids': ['camera_depth'], 'rule': 'fixture_depth', 'conclusion': 'one is nearer.'}]})
        source_binding = {'hdf5': str(folder / (scene_id + '.hdf5')), 'sha256': 'd' * 64,
            'view_id': 'view_0', 'instances': [{**entities[0], 'depth_quantiles': [1., 2., 3.]},
                                            {**entities[1], 'depth_quantiles': [5., 6., 7.]}], 'interval_gap': 2.}
        packets.append({'id': hashlib.sha256(group.encode()).hexdigest(), 'native_form': 'depth_order',
            'partition': 'unassigned_source_only', 'image_sha256': sha(image), 'scene': scene_value.model_dump(),
            'reference': reference.model_dump(), 'student_prompt': student_prompt(scene_value, scene_value.tasks[1]),
            'source_binding': source_binding, 'verification': 'source_rendered_depth_and_deterministic_checks'})
        views.append({'scene_id': scene_id, 'view_id': 'view_0', 'image': str(image),
                      'image_sha256': sha(image), 'raw_hdf5': source_binding['hdf5'], 'raw_sha256': 'd' * 64})
    source, source_manifest = tmp_path / 'records.jsonl', tmp_path / 'source_manifest.json'
    write_rows(source, packets)
    source_manifest.write_text(json.dumps({'state': 'native_source_candidates_validated_partition_pending',
        'records_sha256': sha(source), 'counts': {'records': 3}, 'physical_scenes': 3, 'unique_images': 3}))
    write_rows(image_root / 'views.jsonl', views)
    views_hash = sha(image_root / 'views.jsonl')
    (image_root / 'manifest.json').write_text(json.dumps({'files': {'views.jsonl': views_hash}}))
    policy_source = 'e' * 64
    assignments_path, exclusions_path, audit_path = (tmp_path / name for name in ('assignments.json', 'exclusions.json', 'audit.json'))
    assignments = {groups[0]: {'split': 'train', 'training_approved': True},
                   groups[1]: {'split': 'validation', 'training_approved': False}}
    assignments_path.write_text(json.dumps({'schema': 'anvil-prism-scene-assignments-v1',
        'contract': 'anvil-prism-gripper-se3-v1', 'source_records_sha256': policy_source,
        'assignments': assignments, 'seed': 'fixture', 'heldout_fraction': .5}))
    evidence = [{'sha256': str(i) * 64, 'role': role, 'rows': 1} for i, role in enumerate(
        ('protected_records', 'protected_fingerprints', 'historical_training_pool'), 1)]
    exclusions_path.write_text(json.dumps({'schema': 'anvil-prism-training-exclusions-v1',
        'contract': 'anvil-prism-gripper-se3-v1', 'source_records_sha256': policy_source,
        'protected_group_ids': [], 'protected_image_sha256': [], 'training_seen_group_ids': [],
        'evidence_sha256': [e['sha256'] for e in evidence]}))
    audit_path.write_text(json.dumps({'schema': 'anvil-prism-programmatic-partition-audit-v1',
        'source_records_sha256': policy_source, 'seed': 'fixture', 'heldout_fraction': .5,
        'scene_decisions': [{'group_id': group, 'split': a['split']} for group, a in assignments.items()],
        'candidate_physical_scenes': 2, 'candidate_views_checked': 2, 'views_index_sha256': views_hash,
        'counts': {'train': 1, 'validation': 1}, 'evidence': evidence}))
    return (source, source_manifest, assignments_path, audit_path, exclusions_path, image_root, tmp_path / 'export')


def test_prism_depth_sft_reuses_split_and_excludes_unapproved_scene(prism_depth_export_fixture):
    from tools.arxiv_source_semantic import export_prism_sft
    result = export_prism_sft(*prism_depth_export_fixture)
    assert result['counts']['training_records'] == result['counts']['evaluation_records'] == 1
    assert result['counts']['excluded_records'] == 1
    assert result['counts']['audited_views_rechecked'] == 2
    assert result['excluded_by_reason'] == {'scene_absent_from_frozen_assignments': 1}
    output = prism_depth_export_fixture[-1]
    train = json.loads((output / 'train.jsonl').read_text())
    evaluation = json.loads((output / 'evaluation.jsonl').read_text())
    assert train['anvil_meta']['group_id'] != evaluation['anvil_meta']['group_id']
    assert train['anvil_meta']['training_approved'] and not evaluation['anvil_meta']['training_approved']
    assert train['messages'][0]['content'].startswith('<image>\n')
    assert 'camera_depth' not in train['messages'][0]['content']
    assert 'evidence_ids' not in train['messages'][1]['content']
    assert not train['anvil_meta']['human_reviewed'] and not train['anvil_meta']['legacy_v3_admitted']


@pytest.mark.parametrize('field,value,error', [
    ('candidate_views_checked', 3, 'not all frozen audited source views'),
    ('views_index_sha256', '0' * 64, 'not the frozen audited index'),
    ('source_records_sha256', '0' * 64, 'policy contract or source checksum'),
    ('seed', 'changed', 'seed/fraction'),
])
def test_prism_depth_sft_rejects_changed_policy(prism_depth_export_fixture, field, value, error):
    from tools.arxiv_source_semantic import export_prism_sft
    path = prism_depth_export_fixture[3]
    audit = json.loads(path.read_text())
    audit[field] = value
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match=error):
        export_prism_sft(*prism_depth_export_fixture)
    assert not prism_depth_export_fixture[-1].exists()


def test_prism_depth_sft_rechecks_actual_image_bytes(prism_depth_export_fixture):
    from tools.arxiv_source_semantic import export_prism_sft
    image = prism_depth_export_fixture[5] / ('a' * 32) / 'view_0.png'
    Image.new('RGB', (32, 32), 'black').save(image)
    with pytest.raises(ValueError, match='view image SHA'):
        export_prism_sft(*prism_depth_export_fixture)


def test_prism_depth_sft_rejects_calibrated_interval_contradiction(prism_depth_export_fixture):
    from tools.arxiv_source_semantic import export_prism_sft
    from tools.source_runtime import rows, sha
    source, manifest = prism_depth_export_fixture[:2]
    records = list(rows(source))
    records[0]['source_binding']['interval_gap'] = 0.5
    source.write_text(''.join(json.dumps(r) + '\n' for r in records))
    data = json.loads(manifest.read_text())
    data['records_sha256'] = sha(source)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='contradicts calibrated source depth intervals'):
        export_prism_sft(*prism_depth_export_fixture)


def test_documented_offline_scene_preflight(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ledger = tmp_path / "run.sqlite"
    command = [sys.executable, "-m", "anvil", "align"]
    result = subprocess.run(command + ["run", "--manifest", "configs/alignment_smoke.jsonl",
        "--config", "configs/alignment.yaml", "--ledger", str(ledger)],
        cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert events[-1]["execute"] is False
    expected = [{"split": "dev", "status": "pending", "scenes": 1}]
    assert events[-1]["partitions"] == expected
    result = subprocess.run(command + ["status", "--ledger", str(ledger)],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    output = tmp_path / "dev.jsonl"
    result = subprocess.run(command + ["export", "--ledger", str(ledger), "--split", "dev",
                            "--out", str(output)], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == ""
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT count(*) FROM call_receipts").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0


def scene(tmp_path, group="room-1", split="train"):
    path = tmp_path / (group + ".png")
    Image.new("RGB", (8, 8), "red").save(path)
    return Scene.model_validate({
        "source": "fixture", "scene_id": group, "group_id": group,
        "images": [str(path)], "source_split": split, "training_approved": True,
        "entities": [{"entity_id": "mug", "label": "mug", "bbox": [100, 100, 300, 400]},
                     {"entity_id": "plate", "label": "plate", "bbox": [600, 100, 800, 400]}],
        "evidence": [{"evidence_id": "e1", "kind": "source_annotation",
                      "description": "PRIVATE_SOURCE_SENTINEL", "entity_ids": ["mug", "plate"]}],
        "tasks": [{"task_id": "g", "form": "bbox", "question": "Locate the mug.", "entity_ids": ["mug"]},
                  {"task_id": "s", "form": "spatial", "question": "Where is mug relative to plate?",
                   "entity_ids": ["mug", "plate"], "spatial_task": "direction"}],
    })


def candidate(task):
    answer = [{"label": "mug", "bbox_2d": [100, 100, 300, 400]}]
    if task.form == "spatial":
        answer = [{"task": "direction", "subject_id": "mug", "object_id": "plate",
                   "predicate": "left_of", "frame": "image", "evidence_ids": ["e1"]}]
    return {"task_id": task.task_id, "form": task.form, "answer": answer,
            "reasoning_trace": [{"evidence_ids": ["e1"], "rule": "source_geometry",
                                 "conclusion": "Source boxes support this answer."}]}


class FakeModels:
    def __init__(self, strong="accept", student=None):
        self.calls = []
        self.strong = strong
        self.student = student

    def has_role(self, role):
        return role != "student_verifier" or self.student is not None

    def call(self, role, stage, scene, tasks, candidates, *, advice=None, feedback=None):
        self.calls.append((role, stage))
        if role == "strong_verifier":
            self.advice = advice
        if stage == "generate":
            payload = candidate(tasks[0])
        else:
            decision = self.strong if role == "strong_verifier" else (
                self.student if role == "student_verifier" else "accept")
            payload = {"decision": decision, "task_ids": [t.task_id for t in tasks], "reason": "fixture"}
        return payload, {"role": role, "model": role + "-fixture", "stage": stage}


def test_eight_formats_spatial_is_not_depth():
    assert len(FORMS) == 8 and "spatial" in FORMS and "depth" not in FORMS
    assert len(SPATIAL_TASKS) >= 10


def test_eleven_native_contracts_are_explicit_and_not_spatial_predicates(tmp_path):
    from collections import Counter
    assert len(NATIVE_CONTRACTS) == 11
    assert Counter(item["family"] for item in NATIVE_CONTRACTS.values()) == {"G": 4, "S": 4, "A": 3}
    assert set(NATIVE_CONTRACTS) != set(SPATIAL_TASKS)
    s = scene(tmp_path)
    assert all(task.native_form is None for task in s.tasks)
    raw = s.tasks[1].model_dump()
    assert Task.model_validate({**raw, "native_form": "direction"}).native_form == "direction"
    with pytest.raises(ValidationError, match="native contract"):
        Task.model_validate({**raw, "native_form": "depth_order"})
    with pytest.raises(ValidationError, match="native contract"):
        Task.model_validate({**s.tasks[0].model_dump(), "native_form": "point"})


def native_graph_scene(tmp_path):
    raw = scene(tmp_path).model_dump()
    raw["entities"].append({"entity_id": "spoon", "label": "spoon", "bbox": [100, 600, 300, 800]})
    raw["evidence"].append({"evidence_id": "e2", "kind": "sensor", "description": "Registered ordinal depth",
                            "entity_ids": ["mug", "spoon"], "relation": {
                                "task": "relative_depth", "subject_id": "mug", "object_id": "spoon",
                                "predicate": "nearer", "frame": "camera", "truth": True}})
    raw["tasks"][1].update(native_form="relation_graph", spatial_task=None, entity_ids=["mug", "plate", "spoon"])
    s = Scene.model_validate(raw)
    item = candidate(s.tasks[1])
    item["answer"].append({"task": "relative_depth", "subject_id": "mug", "object_id": "spoon",
                           "predicate": "nearer", "frame": "camera", "truth": True, "evidence_ids": ["e2"]})
    return s, item


def test_native_graph_keeps_typed_edges_and_all_entities(tmp_path):
    s, item = native_graph_scene(tmp_path)
    validate_candidate(Candidate.model_validate(item), s, s.tasks[1], [])
    for answer in [item["answer"][:1], [item["answer"][0], item["answer"][0]]]:
        with pytest.raises(ValueError, match="nonduplicate typed edges"):
            validate_candidate(Candidate.model_validate({**item, "answer": answer}), s, s.tasks[1], [])
    missing = copy.deepcopy(item)
    missing["answer"][1] = {**missing["answer"][0], "subject_id": "plate", "object_id": "mug", "predicate": "right_of"}
    with pytest.raises(ValueError, match="preserve all"):
        validate_candidate(Candidate.model_validate(missing), s, s.tasks[1], [])
    wrong = copy.deepcopy(item)
    wrong["answer"][1]["predicate"] = "farther"
    with pytest.raises(ValueError, match="source annotation"):
        validate_candidate(Candidate.model_validate(wrong), s, s.tasks[1], [])
    with pytest.raises(ValidationError, match="per-edge types"):
        Task.model_validate({**s.tasks[1].model_dump(), "spatial_task": "direction"})


def test_native_metric_never_invents_physical_units(tmp_path):
    raw = scene(tmp_path).model_dump()
    raw["tasks"][1].update(native_form="metric_comparison", spatial_task="distance")
    s = Scene.model_validate(raw)
    item = candidate(s.tasks[1])
    item["answer"][0].update(task="distance", predicate="distance", frame="camera", value=0.5, unit="m")
    with pytest.raises(ValueError, match="typed source evidence"):
        validate_candidate(Candidate.model_validate(item), s, s.tasks[1], [])
    raw["evidence"][0]["relation"] = {k: v for k, v in item["answer"][0].items() if k != "evidence_ids"}
    raw["evidence"][0]["relation"]["truth"] = True
    s = Scene.model_validate(raw)
    validate_candidate(Candidate.model_validate(item), s, s.tasks[1], [])
    raw["evidence"][0]["relation"].pop("value")
    raw["evidence"][0]["relation"].pop("unit")
    with pytest.raises(ValueError, match="explicit source value and unit"):
        validate_candidate(Candidate.model_validate(item), Scene.model_validate(raw), s.tasks[1], [])
    raw["evidence"][0]["relation"] = None
    image_s = Scene.model_validate(raw)
    item["answer"][0].update(frame="image", unit="normalized_image")
    validate_candidate(Candidate.model_validate(item), image_s, image_s.tasks[1], [])
    item["answer"][0]["value"] = 0.4
    with pytest.raises(ValueError, match="image-plane distance"):
        validate_candidate(Candidate.model_validate(item), image_s, image_s.tasks[1], [])


def test_native_export_admission_and_legacy_null(tmp_path):
    ledger = Ledger(tmp_path / "native.sqlite", seed="s", benchmark_fraction=0)
    s, item = native_graph_scene(tmp_path)
    s.tasks[0].native_form = "bbox"
    key, split = ledger.reserve(s)
    result = {"status": "accepted", "candidates": [candidate(s.tasks[0]), item],
              "audit": [{"role": "strong_verifier", "model": "fixture"}]}
    ledger.finish(key, result)
    records = list(export_records(ledger, split))
    for record in records:
        assert record["gsa_native_form"] == record["gsa_meta"]["native_form"]
        assert "native_form" not in record  # gsa_ prefix is stripped by Swift plugin.
        validate_training_record(record)
    assert records[1]["gsa_native_form"] == "relation_graph"
    bad = copy.deepcopy(records[1])
    bad["gsa_native_form"] = None
    with pytest.raises(ValueError, match="native contract and metadata"):
        validate_training_record(bad)
    bad = copy.deepcopy(records[1])
    bad["messages"][1]["content"] = json.dumps(json.loads(bad["messages"][1]["content"])[:1])
    with pytest.raises(ValueError, match="nonduplicate typed edges"):
        validate_training_record(bad)
    legacy = scene(tmp_path, "legacy")
    key, split = ledger.reserve(legacy)
    ledger.finish(key, align_scene(legacy, split, FakeModels()))
    legacy_records = [r for r in export_records(ledger, split) if r["group_id"] == "legacy"]
    assert all(r["gsa_native_form"] is None for r in legacy_records)
    for record in legacy_records:
        record.pop("gsa_native_form")
        record["gsa_meta"].pop("native_form")
        prompt = json.loads(record["messages"][0]["content"])
        prompt.pop("native_form")
        record["messages"][0]["content"] = json.dumps(prompt)
        validate_training_record(record)


def test_native_schema_cli(tmp_path, capsys):
    from trains.flywheel.alignment import main
    main(["schema"])
    assert len(json.loads(capsys.readouterr().out)["native_contracts"]) == 11
    main(["schema", "--native-form", "relation_graph"])
    assert json.loads(capsys.readouterr().out)["minItems"] == 2
    main(["schema", "--native-form", "metric_comparison"])
    schema = json.loads(capsys.readouterr().out)
    assert schema["$defs"]["Spatial"]["properties"]["task"]["enum"] == ["distance", "size"]


def test_strict_schema_and_source_direction(tmp_path):
    s = scene(tmp_path)
    item = candidate(s.tasks[1])
    validate_candidate(Candidate.model_validate(item), s, s.tasks[1], [])
    item["answer"][0]["predicate"] = "right_of"
    with pytest.raises(ValueError, match="contradicts"):
        validate_candidate(Candidate.model_validate(item), s, s.tasks[1], [])
    item["answer"][0]["confidence"] = 0.99
    with pytest.raises(ValidationError):
        Candidate.model_validate(item)


def test_training_prompt_omits_oracle(tmp_path):
    s = scene(tmp_path)
    for task in s.tasks:
        prompt = student_prompt(s, task)
        assert "PRIVATE_SOURCE_SENTINEL" not in prompt
        assert "e1" not in prompt and "evidence_ids" not in prompt
    assert "bbox" not in json.loads(student_prompt(s, s.tasks[0]))["entities"][0]


def test_strong_verifier_is_required_even_with_student_accept(tmp_path):
    result = align_scene(scene(tmp_path), "train", FakeModels("reject", "accept"))
    assert result["status"] == "rejected"


def test_benchmark_never_uses_student(tmp_path):
    models = FakeModels(student="accept")
    result = align_scene(scene(tmp_path), "benchmark", models)
    assert result["status"] == "accepted"
    assert not any(role == "student_verifier" for role, _ in models.calls)


def test_ledger_resume_and_duplicate_image_boundary(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", seed="v1", benchmark_fraction=0.0)
    s = scene(tmp_path)
    key, split = ledger.reserve(s)
    assert split == "train"
    result = align_scene(s, split, FakeModels())
    ledger.finish(key, result)
    assert ledger.completed(key)
    heldout = scene(tmp_path, "different-id", "test")
    with pytest.raises(ValueError, match="image.*split"):
        ledger.reserve(heldout)
    with pytest.raises(ValueError, match="configuration"):
        Ledger(tmp_path / "ledger.sqlite", seed="changed", benchmark_fraction=0.0)


def test_training_and_benchmark_exports_separate(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", seed="v1", benchmark_fraction=0.0)
    s = scene(tmp_path)
    key, split = ledger.reserve(s)
    ledger.finish(key, align_scene(s, split, FakeModels()))
    exported = list(export_records(ledger, "train"))
    assert len(exported) == 2 and list(export_records(ledger, "benchmark")) == []
    assert exported[1]["gsa_form"] == "spatial"
    assert exported[1]["gsa_family"] == "S"
    assert "PRIVATE_SOURCE_SENTINEL" not in json.dumps(exported)
    assert "evidence_ids" not in exported[1]["messages"][-1]["content"]


def test_missing_predictions_count_as_wrong(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", seed="v1", benchmark_fraction=1.0)
    s = scene(tmp_path)
    key, split = ledger.reserve(s)
    ledger.finish(key, align_scene(s, split, FakeModels()))
    targets = list(export_records(ledger, "benchmark"))
    result = score_predictions(targets, [])
    assert result["total"] == 2 and result["exact_match"] == 0
    correct = [{"record_id": t["record_id"], "answer": copy.deepcopy(t["target"])} for t in targets]
    assert score_predictions(targets, correct)["exact_match"] == 1.0
    correct[0]["answer"][0]["bbox_2d"][0] = True
    assert score_predictions(targets, correct)["exact_match"] == 0.5


def test_student_advice_reaches_independent_verifier(tmp_path):
    models = FakeModels(student="reject")
    result = align_scene(scene(tmp_path), "train", models)
    assert models.advice["decision"] == "reject"
    assert result["status"] == "accepted"  # independent verdict remains final


def test_invalid_generation_retry_is_bounded_and_named(tmp_path):
    class BrokenModels(FakeModels):
        generation_attempts = 2
        def call(self, role, stage, scene, tasks, candidates, **kwargs):
            payload, meta = super().call(role, stage, scene, tasks, candidates, **kwargs)
            self.last_feedback = kwargs.get("feedback")
            if stage == "generate":
                payload["answer"][0]["bbox_2d"] = [-1, 0, 1, 2]
            return payload, meta
    models = BrokenModels()
    result = align_scene(scene(tmp_path), "train", models)
    assert result["status"] == "rejected"
    assert len(models.calls) == 2
    assert models.last_feedback["failure"] == "schema_or_rule"
    assert models.last_feedback["attempt"] == 1


@pytest.mark.parametrize("task,predicate,frame", [
    ("relative_depth", "nearer", "camera"), ("direction", "above", "image"),
    ("distance", "near", "world"), ("size", "larger", "world"),
    ("topology", "touching", "world"), ("occlusion", "occludes", "camera"),
    ("support", "supported_by", "world"), ("containment", "inside", "world"),
    ("orientation", "parallel", "world"), ("viewpoint", "left_of", "observer"),
    ("motion", "approaching", "world"),
])
def test_spatial_subtypes_have_distinct_contracts(task, predicate, frame):
    value = {"task": task, "predicate": predicate, "frame": frame, "subject_id": "a",
             "object_id": "b", "evidence_ids": ["e"]}
    if frame == "observer":
        value["observer_id"] = "camera_1"
    assert Spatial.model_validate(value).task == task
    value["predicate"] = "invented_predicate"
    with pytest.raises(ValidationError):
        Spatial.model_validate(value)


@pytest.mark.parametrize("form,answer", [
    ("bbox", [{"label": "cup", "bbox_2d": [100, 100, 200, 300]}]),
    ("detection", [{"label": "cup", "bbox_2d": [100, 100, 200, 300]},
                   {"label": "bowl", "bbox_2d": [400, 100, 500, 300]}]),
    ("mask", [{"label": "cup", "polygon": [[100, 100], [150, 100], [200, 150], [200, 200], [150, 250], [100, 200]]}]),
    ("point", [{"label": "cup", "point_2d": [150, 150]}]),
    ("affordance", [{"label": "handle", "bbox_2d": [100, 100, 150, 150],
                    "affordance": "grasp", "contact_point_2d": [125, 125]}]),
    ("dof", [{"label": "cup", "grasp_point_2d": [125, 125], "jaw_axis_2d": [1, 0],
              "approach": "top", "wrist_deg": 0}]),
    ("action_cot", [{"goal": "grasp cup", "steps": ["Approach the visible cup", "Close jaws on the handle"],
                    "preconditions": ["handle visible"], "failure_modes": ["slip"], "target_point_2d": [125, 125]}]),
])
def test_other_formats_remain_strict(form, answer):
    value = {"task_id": "t", "form": form, "answer": answer,
             "reasoning_trace": [{"rule": "visible", "evidence_ids": ["e"], "conclusion": "visible"}]}
    assert Candidate.model_validate(value).form == form
    assert answer_schema(form)["type"] == "array"
    value["answer"][0]["undeclared_key"] = 1
    with pytest.raises(ValidationError):
        Candidate.model_validate(value)


def test_nonfinite_and_out_of_range_are_rejected(tmp_path):
    s = scene(tmp_path)
    for bad in [float("nan"), float("inf"), 1001, -1, True, "200"]:
        item = candidate(s.tasks[0])
        item["answer"][0]["bbox_2d"][0] = bad
        with pytest.raises(ValidationError):
            Candidate.model_validate(item)


def test_source_scene_split_must_be_known(tmp_path):
    s = scene(tmp_path)
    s.source_split = "unknown"
    ledger = Ledger(tmp_path / "db", seed="s", benchmark_fraction=0.1)
    with pytest.raises(ValueError, match="resolved"):
        ledger.reserve(s)


def test_ledger_freezes_model_identity_and_scene_metadata(tmp_path):
    ledger = Ledger(tmp_path / "db", seed="s", benchmark_fraction=0)
    ledger.bind_models({"teacher": "v1"})
    ledger.bind_models({"teacher": "v1"})
    with pytest.raises(ValueError, match="deployments"):
        ledger.bind_models({"teacher": "v2"})
    s = scene(tmp_path)
    ledger.reserve(s)
    s.tasks[0].question = "changed question"
    with pytest.raises(ValueError, match="metadata"):
        ledger.reserve(s)


def test_export_is_mopd_eligible_only_on_train(tmp_path):
    ledger = Ledger(tmp_path / "db", seed="s", benchmark_fraction=0)
    s = scene(tmp_path)
    key, split = ledger.reserve(s)
    ledger.finish(key, align_scene(s, split, FakeModels()))
    for record in export_records(ledger, "train"):
        validate_training_record(record)
        record["split"] = "benchmark"
        with pytest.raises(ValueError, match="train split"):
            validate_training_record(record)


def test_spatialsense_axis_order_negative_labels_and_official_split(tmp_path):
    url = "https://example.org/source.png"
    rows = [{"url": url, "width": 200, "height": 100, "split": "test", "annotations": [
        {"predicate": "to the right of", "label": False,
         "subject": {"name": "cup", "bbox": [10, 30, 20, 60]},
         "object": {"name": "bowl", "bbox": [10, 30, 120, 160]}}]}]
    s = next(spatialsense_scenes(rows, {url: {"image": "local.png", "group_id": "original-scene"}}))
    assert s.entities[0].bbox == [100, 100, 300, 300]
    assert s.source_split == "test" and s.group_id == "original-scene"
    assert s.evidence[0].relation.truth is False
    value = {"task_id": "relation_0", "form": "spatial", "answer": [{
        **s.evidence[0].relation.model_dump(), "evidence_ids": ["annotation_0"]}],
        "reasoning_trace": [{"rule": "source", "evidence_ids": ["annotation_0"], "conclusion": "false"}]}
    validate_candidate(Candidate.model_validate(value), s, s.tasks[1], [])
    value["answer"][0]["truth"] = True
    with pytest.raises(ValueError, match="source annotation"):
        validate_candidate(Candidate.model_validate(value), s, s.tasks[1], [])


def test_request_cache_roundtrip_and_scoring_duplicates(tmp_path):
    ledger = Ledger(tmp_path / "db", seed="s", benchmark_fraction=0)
    ledger.cache("request", {"payload": {"answer": 1}, "metadata": {"model": "t"}})
    assert ledger.cached("request")["metadata"]["model"] == "t"
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions([], [{"record_id": "a"}, {"record_id": "a"}])


def test_strict_vlm_rejects_truncation_and_hidden_reasoning(monkeypatch):
    from apps.data_flywheel.workers.vlm_client import VLMClient
    client = VLMClient(base_url="http://localhost", model="test", max_retries=0)
    response = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
    class Reply:
        def raise_for_status(self): pass
        def json(self): return response
    class Session:
        def post(self, *args, **kwargs): return Reply()
    monkeypatch.setattr(client, "_get_session", lambda: Session())
    with pytest.raises(ValueError, match="incomplete"):
        client.chat("test", require_complete=True)
    assert len(client._thread_local.attempt_receipts) == 1
    assert client._thread_local.attempt_receipts[0]["finish_reason"] == "length"
    assert client._thread_local.attempt_receipts[0]["content"] == "{}"
    response["choices"][0] = {"finish_reason": "stop", "message": {"content": None, "reasoning_content": "secret"}}
    assert client.chat("test", require_complete=True) == ""
    assert client._thread_local.response_metadata["model"] == "test"
    assert len(client._thread_local.attempt_receipts) == 1
    assert client._thread_local.attempt_receipts[0]["finish_reason"] == "stop"


def test_real_cli_transport_resume_export_and_preflight(tmp_path, monkeypatch):
    """Loopback fake model service exercises HTTP, not model-quality evidence."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from types import SimpleNamespace
    from trains.flywheel.alignment import main
    from scripts.validate_grpo_dataset import main as preflight
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = json.loads(request["messages"][-1]["content"][-1]["text"])
            calls.append((request["model"], prompt["stage"]))
            if prompt["stage"] == "generate":
                answer = candidate(SimpleNamespace(**prompt["tasks"][0]))
            else:
                answer = {"decision": "accept", "task_ids": [t["task_id"] for t in prompt["tasks"]], "reason": "fixture"}
            data = json.dumps({"model": request["model"], "usage": {"total_tokens": 10},
                               "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
        roles = {}
        for name in ["teacher_G", "teacher_S", "teacher_A", "strong_verifier"]:
            monkeypatch.setenv(name+"_URL", f"http://127.0.0.1:{server.server_port}")
            monkeypatch.setenv(name+"_MODEL", name+"-v1")
            roles[name] = {"url_env": name+"_URL", "model_env": name+"_MODEL"}
        config = {"contract": "anvil-gen-v3", "split_seed": "test", "benchmark_fraction": 0,
                  "max_tokens": 512, "timeout_seconds": 3, "transport_retries": 0, "roles": roles}
        cfg, manifest = tmp_path / "config.json", tmp_path / "scenes.jsonl"
        cfg.write_text(json.dumps(config))
        manifest.write_text(scene(tmp_path).model_dump_json()+"\n")
        ledger_path = str(tmp_path / "run.sqlite")
        arguments = ["run", "--manifest", str(manifest), "--config", str(cfg), "--ledger", ledger_path]
        main(arguments)
        assert calls == []
        main(arguments+["--execute", "--workers", "2"])
        assert len(calls) == 5
        main(arguments+["--execute"])
        assert len(calls) == 5
        out = str(tmp_path / "train.jsonl")
        main(["export", "--ledger", ledger_path, "--split", "train", "--out", out])
        monkeypatch.setattr("sys.argv", ["validate", out, "--alignment"])
        assert preflight() == 0
        with pytest.raises(FileExistsError):
            main(["export", "--ledger", ledger_path, "--split", "train", "--out", out])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_changed_trainer_input_filter_without_optional_swift_stack():
    """Exercise the changed method's AST against its parent interface on CPU.

    This is not a substitute for importing/training the full ms-swift plugin.
    """
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "scripts/gsa_mopd_plugin.py"
    module = ast.parse(source.read_text())
    trainer = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "GSARoutedMOPDTrainer")
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_batch_inputs")
    probe = ast.ClassDef(name="Probe", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                         keywords=[], body=[method], decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    class Parent:
        def _prepare_batch_inputs(self, inputs, encode_prompt_only=False):
            return dict(inputs)
    namespace = {"Parent": Parent}
    exec(compile(tree, str(source), "exec"), namespace)
    inputs = {"input_ids": [1], "pixel_values": [2], "record_id": "r", "group_id": "s",
              "source": "src", "split": "train", "gsa_form": "spatial", "spatial_task": "direction"}
    assert namespace["Probe"]()._prepare_batch_inputs(inputs) == {"input_ids": [1], "pixel_values": [2]}
    from collections import UserDict
    assert namespace["Probe"]()._prepare_batch_inputs(UserDict(inputs)) == {"input_ids": [1], "pixel_values": [2]}
    assert inputs["record_id"] == "r"
    with pytest.raises(ValueError, match="list of source records"):
        namespace["Probe"]()._prepare_batch_inputs({"gsa_contract_version": "anvil-source-native-train-v1"})
    assert inputs["record_id"] == "r"


def test_production_goal_counts_only_accepted_train_and_new_pixels(tmp_path):
    from trains.flywheel.alignment import ProductionGoal
    ledger = Ledger(tmp_path / "goal.sqlite", seed="test", benchmark_fraction=0)
    first = scene(tmp_path)
    baseline = tmp_path / "old.jsonl"
    baseline.write_text(json.dumps({"images": first.images})+"\n")
    key, split = ledger.reserve(first)
    goal = ProductionGoal(ledger, 4, 1, baseline)
    result = align_scene(first, split, FakeModels())
    ledger.finish(key, result)
    goal.observe(key, split, result)
    assert goal.summary()["accepted_train_records"] == 2
    assert goal.new_images == 0 and not goal.reached()
    second = scene(tmp_path, "room-2")
    Image.new("RGB", (8, 8), "blue").save(second.images[0])
    key, split = ledger.reserve(second)
    result = align_scene(second, split, FakeModels())
    ledger.finish(key, result)
    goal.observe(key, split, result)
    assert goal.reached() and goal.images == 2 and goal.new_images == 1
    heldout = scene(tmp_path, "test-room", "test")
    Image.new("RGB", (8, 8), "green").save(heldout.images[0])
    key, split = ledger.reserve(heldout)
    result = align_scene(heldout, split, FakeModels())
    ledger.finish(key, result)
    goal.observe(key, split, result)
    assert goal.count == 4 and goal.images == 2
    resumed = ProductionGoal(ledger, 4, 1, baseline)
    assert resumed.summary() == goal.summary()
    baseline.write_text(json.dumps({"image": second.images[0]})+"\n")
    with pytest.raises(ValueError, match="baseline changed"):
        ProductionGoal(ledger, 4, 1, baseline)


def test_production_image_deduplication_and_rejection(tmp_path):
    from trains.flywheel.alignment import ProductionGoal
    ledger = Ledger(tmp_path / "goal.sqlite", seed="test", benchmark_fraction=0)
    goal = ProductionGoal(ledger, 4, 2)
    for name, decision in [("first", "accept"), ("copy", "accept"), ("bad", "reject")]:
        value = scene(tmp_path, name)  # Identical decoded pixels, different paths.
        key, split = ledger.reserve(value)
        result = align_scene(value, split, FakeModels(strong=decision))
        ledger.finish(key, result)
        goal.observe(key, split, result)
    assert goal.count == 4 and goal.images == 1 and not goal.reached()


def test_production_baseline_blocks_heldout_reuse(tmp_path):
    from trains.flywheel.alignment import ProductionGoal
    ledger = Ledger(tmp_path / "goal.sqlite", seed="test", benchmark_fraction=0)
    value = scene(tmp_path, split="test")
    ledger.reserve(value)
    baseline = tmp_path / "old.jsonl"
    baseline.write_text(json.dumps({"image": value.images[0]})+"\n")
    with pytest.raises(ValueError, match="held-out"):
        ProductionGoal(ledger, 2, 1, baseline)


def test_goal_cli_stops_resumes_and_reports_unmet(tmp_path, monkeypatch):
    from trains.flywheel import alignment
    models = FakeModels()
    monkeypatch.setattr(alignment, "Models", lambda *args: models)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"contract": "anvil-gen-v3", "split_seed": "test", "benchmark_fraction": 0,
                                 "max_tokens": 512, "timeout_seconds": 3}))
    inputs = []
    for name in ["a", "b", "c"]:
        path = tmp_path / (name+".jsonl")
        path.write_text(scene(tmp_path, name).model_dump_json()+"\n")
        inputs += ["--manifest", str(path)]
    args = ["run", *inputs, "--config", str(config), "--ledger", str(tmp_path/"run.sqlite"),
            "--target-records", "2", "--execute"]
    # Different scene paths with identical pixels count as ONE source image.
    assert alignment.main(args+["--min-source-images", "2"]) == 3
    assert len(models.calls) == 0
    assert alignment.main(args) == 0 and len(models.calls) == 5
    monkeypatch.setattr(alignment, "Models", lambda *args: pytest.fail("completed goal must not initialize endpoints"))
    assert alignment.main(args) == 0 and len(models.calls) == 5
    monkeypatch.setattr(alignment, "Models", lambda *args: models)
    assert alignment.main(args+["--target-records", "10"]) == 3
    assert len(models.calls) == 15  # Three distinct scene tasks, no duplicate accepted calls.


@pytest.mark.parametrize("background", [False, True])
def test_root_production_script_exports_completed_ledger(tmp_path, background):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    from trains.flywheel.alignment import digest
    root = Path(__file__).resolve().parents[1]
    config = {"contract": "anvil-gen-v3", "split_seed": "test", "benchmark_fraction": 0,
              "max_tokens": 512, "timeout_seconds": 3}
    cfg, manifest, out = tmp_path/"config.json", tmp_path/"scenes.jsonl", tmp_path/"production"
    cfg.write_text(json.dumps(config))
    value = scene(tmp_path)
    manifest.write_text(value.model_dump_json()+"\n")
    ledger = Ledger(out/"ledger.sqlite", seed="test", benchmark_fraction=0, fingerprint=digest(config))
    key, split = ledger.reserve(value)
    ledger.finish(key, align_scene(value, split, FakeModels()))
    ledger.db.close()
    command = ["bash", str(root/"generate_to_scale.sh"), "--manifest", str(manifest), "--config", str(cfg),
               "--out", str(out), "--target-records", "2", "--target-source-images", "1", "--execute"]
    if background:
        command += ["--background"]
    run = subprocess.run(command, env={**os.environ, "ANVIL_PYTHON": sys.executable},
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr+run.stdout
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        status = out/"exit_code"
        if status.exists() and status.read_text().strip() != "running":
            break
        time.sleep(.05)
    assert status.read_text().strip() == "0"
    exports = list(out.glob("export.*/train.jsonl"))
    assert len(exports) == 1 and ".partial" not in str(exports[0])
    assert len(exports[0].read_text().splitlines()) == 2
