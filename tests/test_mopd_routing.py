from collections import Counter
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch

from trains.mopd.routing import FamilyBalancedBatchSampler, dataset_families
from trains.mopd.anvil_runtime import ContractV2BatchRouter, decoded_token_spans, parse_stage_segments, structural_ranges


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join(chr(value - 1) for value in ids if value)


def _native_fixture(form):
    from tests.test_native_training import evaluation_fixture_row, serialize
    from trains.flywheel.native_training import bind_training_record
    # Synthetic fixtures only; production held-out rows must never be relabelled.
    row = evaluation_fixture_row(form)
    row['anvil_meta'].update(split='train', training_approved=True)
    return bind_training_record(*serialize(row))


@pytest.mark.parametrize('form', ['bbox', 'detection', 'mask', 'point', 'direction',
    'metric_comparison', 'relation_graph', 'depth_order', 'dof', 'affordance', 'action_cot'])
def test_native_atomic_router_preserves_format_and_current_sample_ids(form):
    from trains.mopd.anvil_runtime import AnvilBatchRouter, FAMILIES
    from tools.response_text import SWIFT_NON_THINKING_PREFIX
    row = _native_fixture(form); original = deepcopy(row)
    router = AnvilBatchRouter()
    expected = FAMILIES.index(row['gsa_family'])
    target = row['messages'][1]['content']
    reference = router.plan([row], _encode_responses([SWIFT_NON_THINKING_PREFIX + target]),
        CharacterTokenizer(), reference=True, qualified={}, reference_prefix=SWIFT_NON_THINKING_PREFIX)
    assert (reference.stage_ids == expected).sum() == len(target)
    assert reference.selected_stages.tolist() == [-1]
    inputs = _encode_responses(['malformed fresh sample'])
    plan = router.plan([row], inputs, CharacterTokenizer(), reference=False, qualified={row['gsa_family']: True})
    assert torch.equal(plan.token_ids, inputs['input_ids'][:, 1:])
    assert plan.selected_stages.tolist() == [expected]
    assert (plan.stage_ids == expected).sum() == len('malformed fresh sample')
    assert plan.decisions[0]['supervision_span'] == 'atomic_response'
    assert row == original


@pytest.mark.parametrize('qualified', [{}, {'G': False}, {'G': 1}])
def test_native_router_cannot_qualify_teacher(qualified):
    from trains.mopd.anvil_runtime import AnvilBatchRouter
    plan = AnvilBatchRouter().plan([_native_fixture('bbox')], _encode_responses(['bad']),
        CharacterTokenizer(), reference=False, qualified=qualified)
    assert plan.selected_stages.tolist() == [-1]
    assert plan.decisions[0]['reason'] == 'teacher_not_qualified'


def test_native_router_empty_response_and_reference_tampering():
    from trains.mopd.anvil_runtime import AnvilBatchRouter
    row = _native_fixture('bbox'); router = AnvilBatchRouter()
    plan = router.plan([row], _encode_responses(['']), CharacterTokenizer(), reference=False, qualified={'G':True})
    assert plan.selected_stages.tolist() == [-1] and plan.decisions[0]['reason'] == 'empty_rollout_span'
    with pytest.raises(ValueError, match='differs from source'):
        router.plan([row], _encode_responses(['[]']), CharacterTokenizer(), reference=True, qualified={})
    inputs = _encode_responses(['bad']); inputs['attention_mask'][0,1] = 0
    with pytest.raises(ValueError, match='attended'):
        router.plan([row], inputs, CharacterTokenizer(), reference=False, qualified={'G':True})


@pytest.mark.parametrize('mutation', ['split', 'content', 'linked', 'QA'])
def test_native_router_rejects_changed_provenance_and_qa(mutation):
    from trains.mopd.anvil_runtime import AnvilBatchRouter
    row = _native_fixture('bbox')
    if mutation == 'split': row['gsa_meta']['split'] = 'benchmark'
    elif mutation == 'content': row['messages'][0]['content'] += ' changed'
    elif mutation == 'linked': row['gsa_stages'] = ['G','S']
    else:
        from tests.test_native_training import fixture_qa, serialize
        from trains.flywheel.native_training import bind_training_record
        row = bind_training_record(*serialize(fixture_qa()))
    with pytest.raises(ValueError):
        AnvilBatchRouter().plan([row], _encode_responses(['bad']), CharacterTokenizer(), reference=False, qualified={'G':True})


def test_native_router_marks_native_geometry_and_action_values():
    from trains.mopd.anvil_runtime import AnvilBatchRouter
    row = _native_fixture('affordance')
    target = row['messages'][1]['content']
    plan = AnvilBatchRouter().plan([row], _encode_responses([target]), CharacterTokenizer(), reference=True, qualified={})
    for value in ('10', '20', '30', '40'):
        start = target.index(value)
        assert bool(plan.structural_mask[0,start:start+len(value)].all())


def test_native_teacher_gate_rejects_legacy_verdict_and_changed_weights(tmp_path, monkeypatch):
    import hashlib
    from trains.mopd.anvil_runtime import validate_teacher_eligibility
    export = tmp_path/'export'; export.mkdir(); (export/'manifest.json').write_text('{}')
    paths = {}
    for family in ('G','S','A','joint'):
        path = tmp_path/family; path.mkdir(); (path/'adapter_model.safetensors').write_bytes(family.encode())
        paths[family] = str(path)
    report = {'verdict':'PASS', 'joint_adapter':paths['joint'],
        'teachers':{f:{'eligible':True,'adapter':paths[f]} for f in ('G','S','A')}}
    location = tmp_path/'quality.json'; location.write_text(json.dumps(report))
    monkeypatch.setenv('ANVIL_DATA_CONTRACT','anvil-source-native-train-v1')
    monkeypatch.setenv('TRAIN',str(export))
    adapters = {f:paths[f] for f in ('G','S','A')}
    with pytest.raises(ValueError,match='native training-export'):
        validate_teacher_eligibility(location,adapters,paths['joint'])
    report.update(data_contract='anvil-source-native-train-v1',
        training_export_manifest_sha256=hashlib.sha256(b'{}').hexdigest(),
        joint_adapter_sha256=hashlib.sha256(b'joint').hexdigest())
    for family,item in report['teachers'].items(): item['adapter_sha256'] = hashlib.sha256(family.encode()).hexdigest()
    location.write_text(json.dumps(report))
    assert validate_teacher_eligibility(location,adapters,paths['joint']) == {'G':True,'S':True,'A':True}
    (tmp_path/'G'/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='teacher bytes'):
        validate_teacher_eligibility(location,adapters,paths['joint'])


def test_native_training_step_cannot_reuse_legacy_qualified_initialization():
    student = TinyTokenModel(); trainer = _cpu_anvil_trainer(student)
    with pytest.raises(ValueError, match='native-qualified trainer'):
        trainer.training_step(student, [_native_fixture('bbox')])


@pytest.mark.parametrize('name,value', [
    ('MAX_LENGTH', None), ('MAX_COMPLETION_LENGTH', None),
    ('MAX_LENGTH', '0'), ('MAX_LENGTH', '-1'), ('MAX_LENGTH', '1.5'),
    ('MAX_COMPLETION_LENGTH', ''), ('MAX_COMPLETION_LENGTH', 'auto'),
])
def test_native_launcher_rejects_implicit_or_invalid_budget_before_cuda(tmp_path, name, value):
    import os
    import subprocess
    from pathlib import Path
    script = Path(__file__).parents[1] / 'scripts/run_mopd_9b_lora_l20.sh'
    fakebin = tmp_path / 'bin'; fakebin.mkdir()
    python = fakebin / 'python'
    python.write_text('#!/bin/sh\necho UNEXPECTED_PYTHON_CALL\nexit 91\n'); python.chmod(0o755)
    env = {**os.environ, 'GSA_ROOT': str(tmp_path), 'MODEL': str(tmp_path/'model'),
           'ANVIL_DATA_CONTRACT': 'anvil-source-native-train-v1', 'ANVIL_MTD': '1',
           'MAX_LENGTH': '16384', 'MAX_COMPLETION_LENGTH': '16384',
           'PATH': str(fakebin) + os.pathsep + os.environ['PATH']}
    if value is None: env.pop(name)
    else: env[name] = value
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and 'explicit positive-integer ' + name in result.stderr
    assert 'UNEXPECTED_PYTHON_CALL' not in result.stdout + result.stderr


def test_native_launcher_passes_strict_flags_without_changing_legacy_defaults():
    import subprocess
    from pathlib import Path
    script = Path(__file__).parents[1] / 'scripts/run_mopd_9b_lora_l20.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    source = script.read_text()
    assert 'NATIVE_DATA_ARGS=()' in source
    assert 'NATIVE_DATA_ARGS=(--strict true --truncation_strategy delete)' in source
    assert '--split_dataset_ratio 0 "${NATIVE_DATA_ARGS[@]}"' in source
    assert '--max_length "${MAX_LENGTH:-3072}" --max_completion_length "${MAX_COMPLETION_LENGTH:-512}"' in source


@pytest.mark.parametrize('policy', ['left', 'right', 'split', None])
@pytest.mark.parametrize('prompt_only', [False, True])
def test_native_mtd_rejects_truncating_encoder_before_touching_row(policy, prompt_only):
    from scripts.gsa_mopd_plugin import GSARoutedMOPDTrainer
    trainer = GSARoutedMOPDTrainer.__new__(GSARoutedMOPDTrainer)
    trainer.template = SimpleNamespace(truncation_strategy=policy)
    row = _native_fixture('point'); original = deepcopy(row)
    with pytest.raises(ValueError, match='non-truncating template encoding'):
        trainer._prepare_batch_inputs([row], encode_prompt_only=prompt_only)
    assert row == original


@pytest.mark.parametrize('prompt_only', [False, True])
def test_native_mtd_encoding_error_propagates_without_resampling(prompt_only):
    from scripts.gsa_mopd_plugin import GSARoutedMOPDTrainer
    trainer = GSARoutedMOPDTrainer.__new__(GSARoutedMOPDTrainer)
    row = _native_fixture('point'); original = deepcopy(row); calls = []
    failure = ValueError('fixture exceeds the declared context budget')
    def encode(data, return_length):
        calls.append(data['id'])
        raise failure
    trainer.template = SimpleNamespace(truncation_strategy='raise', max_length=128, encode=encode)
    trainer._template_context = lambda *args, **kwargs: nullcontext()
    with pytest.raises(ValueError) as caught:
        trainer._prepare_batch_inputs([deepcopy(row)], encode_prompt_only=prompt_only)
    assert caught.value is failure and calls == [row['id']] and row == original


@pytest.mark.parametrize('prompt_only', [False, True])
def test_native_mtd_nontruncating_encoder_keeps_original_messages(prompt_only):
    from scripts.gsa_mopd_plugin import GSARoutedMOPDTrainer
    trainer = GSARoutedMOPDTrainer.__new__(GSARoutedMOPDTrainer)
    trainer.model = SimpleNamespace(device=torch.device('cpu'))
    row = _native_fixture('point'); original = deepcopy(row)
    def encode(data, return_length):
        assert data['messages'][0] == original['messages'][0]
        assert data['messages'][1]['content'] == (None if prompt_only else original['messages'][1]['content'])
        return {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones(1, 2),
                'gsa_meta': data['gsa_meta']}
    trainer.template = SimpleNamespace(truncation_strategy='raise', max_length=128, encode=encode, data_collator=lambda rows: rows[0])
    trainer._template_context = lambda *args, **kwargs: nullcontext()
    result = trainer._prepare_batch_inputs([deepcopy(row)], encode_prompt_only=prompt_only)
    assert set(result) == {'input_ids', 'attention_mask'} and row == original


@pytest.mark.parametrize('prompt_only', [False, True])
def test_native_mtd_keeps_context_limit_inside_real_swift_context(prompt_only):
    from scripts.gsa_mopd_plugin import GSARoutedMOPDTrainer
    from swift.template.base import MaxLengthError
    trainer = GSARoutedMOPDTrainer.__new__(GSARoutedMOPDTrainer)
    trainer.model = SimpleNamespace(device=torch.device('cpu'))
    template = SimpleNamespace(truncation_strategy='raise', max_length=8, mode='train')
    template.set_mode = lambda mode: setattr(template, 'mode', mode)
    observed = []
    def encode(data, return_length):
        observed.append(template.max_length)
        if template.max_length is not None and 12 > template.max_length:
            raise MaxLengthError('fixture needs 12 tokens')
        return {'input_ids': torch.arange(12).reshape(1, 12)}
    template.encode = encode; template.data_collator = lambda rows: rows[0]
    trainer.template = template
    # Deliberately use the actual inherited context manager: it normally sets
    # max_length=None when GKD's encoder does not pass an explicit override.
    with pytest.raises(MaxLengthError):
        trainer._prepare_batch_inputs([_native_fixture('point')], encode_prompt_only=prompt_only)
    assert observed == [8] and template.max_length == 8 and template.mode == 'train'


def test_mixed_native_and_linked_real_trainer_loss_keeps_teacher_gradients_frozen():
    from trains.mopd.anvil_runtime import AnvilBatchRouter
    torch.set_num_threads(1)
    student = TinyTokenModel(); trainer = _cpu_anvil_trainer(student)
    trainer.anvil_router = AnvilBatchRouter(_exact_fixture_score)
    rows = [_native_fixture('bbox'), _linked_row(), _native_fixture('action_cot')]
    reference = [row['messages'][1]['content'] for row in rows]
    sampled = ['bad G', reference[1].replace('"rank": 1', '"rank": 2', 1), 'bad A']
    trainer._anvil_rows = rows; trainer._anvil_reference = _encode_responses(reference)
    loss = trainer.compute_loss(student, _encode_responses(sampled))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.weight.grad).all()
    assert trainer.activated == ['G','S','A']
    assert all(parameter.grad is None for teacher in trainer.teachers.values() for parameter in teacher.parameters())
    assert trainer._anvil_last_budget['teacher_forward_calls'] == 3


def _encode_responses(texts):
    ids = [[ord("P") + 1] + [ord(char) + 1 for char in text] + [0] for text in texts]
    length = max(map(len, ids))
    encoded = torch.tensor([row + [0] * (length - len(row)) for row in ids])
    labels = encoded.clone()
    labels[:, 0] = -100
    for row, values in enumerate(ids):
        labels[row, len(values):] = -100
    return {"input_ids": encoded, "labels": labels, "attention_mask": (encoded != 0).long()}


def _linked_row():
    grounding = [{"label": "mug", "bbox_2d": [100, 100, 200, 200]},
                 {"label": "book", "bbox_2d": [400, 400, 600, 600]}]
    depth = [{"label": "mug", "rank": 1, "relative_depth": 0.0},
             {"label": "book", "rank": 2, "relative_depth": 1.0}]
    text = "<GROUND>" + json.dumps(grounding) + "</GROUND><RELATE>" + json.dumps(depth) + "</RELATE>"
    row = {"gsa_form": "depth", "gsa_family": "S", "gsa_contract_version": "gsa-contract-v2",
           "messages": [{"role": "user", "content": "Locate the mug and book, then order their depths."},
                        {"role": "assistant", "content": text}],
           "gsa_meta": {"image_sha256": "a" * 64, "split": "train"},
           "gsa_stages": ["G", "S"], "gsa_stage_forms": ["detection", "depth"],
           "gsa_edges": [{"upstream": "G", "downstream": "S", "consumed_field": "label"}],
           "gsa_stage_span_unit": "reference_characters",
           "gsa_stage_spans": [{"start": start, "end": end}
                               for _, start, end, _, _ in parse_stage_segments(text, ["G", "S"])],
           "gsa_references": [grounding, depth], "gsa_thresholds": [.5, .5], "gsa_uncertainty_bands": [.05, .05]}
    return row


def _exact_fixture_score(**kwargs):
    return float(kwargs["payload"] == kwargs["reference"])


def test_reference_geometry_routes_changed_depth_after_nonidentical_correct_grounding():
    from trains.flywheel.contracts import context_for
    from trains.mopd.anvil_runtime import reference_geometry_evaluator
    row = _linked_row()
    groundings = [{"object_class": item["label"], "bbox": item["bbox_2d"]} for item in row["gsa_references"][0]]
    row["gsa_stage_contexts"] = [context_for(form, groundings) for form in ("detection", "depth")]
    predicted_g = deepcopy(row["gsa_references"][0])
    predicted_g[0]["bbox_2d"][0] += 5
    predicted_g.reverse()
    predicted_s = deepcopy(row["gsa_references"][1])
    for item in predicted_s:
        item["rank"] = 3 - item["rank"]
        item["relative_depth"] = 1 - item["relative_depth"]
    text = "<GROUND>" + json.dumps(predicted_g) + "</GROUND><RELATE>" + json.dumps(predicted_s) + "</RELATE>"
    plan = ContractV2BatchRouter(reference_geometry_evaluator).plan(
        [row], _encode_responses([text]), CharacterTokenizer(), reference=False, qualified={"G": True, "S": True})
    assert plan.selected_stages.tolist() == [1]
    assert plan.decisions[0]["state"] == "semantic_failure"
    assert predicted_g != row["gsa_references"][0]


def test_reference_geometry_abstains_without_context_and_requires_passed_predecessors():
    from trains.flywheel.contracts import context_for
    from trains.mopd.anvil_runtime import reference_geometry_evaluator
    row = _linked_row()
    call = dict(row=row, stage="S", form="relative_depth", payload=row["gsa_references"][1],
                upstream={"G": row["gsa_references"][0]}, reference=row["gsa_references"][1])
    assert reference_geometry_evaluator(**call) is None
    gs = [{"object_class": item["label"], "bbox": item["bbox_2d"]} for item in row["gsa_references"][0]]
    row["gsa_stage_contexts"] = [context_for("detection", gs), context_for("depth", gs)]
    assert reference_geometry_evaluator(**call) == 1.0
    assert reference_geometry_evaluator(**{**call, "upstream": {"G": row["gsa_references"][0][:1]}}) is None
    assert reference_geometry_evaluator(**{**call, "upstream": {"G": row["gsa_references"][0][:1]},
                                           "payload": row["gsa_references"][1][:1]}) is None
    with pytest.raises(ValueError, match="exact passed direct predecessors"):
        reference_geometry_evaluator(**{**call, "upstream": {}})


def test_reference_geometry_does_not_treat_single_grasp_reference_as_complete_semantics():
    from trains.flywheel.contracts import context_for
    from trains.mopd.anvil_runtime import reference_geometry_evaluator
    context = context_for("affordance", [{"object_class": "mug", "bbox": [100, 100, 800, 800]}])
    first = [{"label": "handle", "affordance": "grasp", "bbox_2d": [150, 200, 250, 300], "contact_point_2d": [200, 250]}]
    second = [{"label": "rim", "affordance": "grasp", "bbox_2d": [300, 150, 500, 250], "contact_point_2d": [400, 200]}]
    row = {"gsa_stages": ["A"], "gsa_edges": [], "gsa_stage_contexts": [context]}
    call = dict(row=row, stage="A", form="affordance", payload=second, upstream={}, reference=first)
    assert reference_geometry_evaluator(**call) is None
    row.update(gsa_reference_coverage=["complete_for_task"], gsa_reference_alternatives=[[first, second]])
    assert reference_geometry_evaluator(**call) == 1.0
    row["gsa_reference_alternatives"] = [[second]]
    with pytest.raises(ValueError, match="incomplete or invalid"):
        reference_geometry_evaluator(**call)


def test_anvil_routes_current_semantic_failure_and_stops_on_upstream_ambiguity():
    row = _linked_row()
    response = row["messages"][-1]["content"].replace('"rank": 1', '"rank": 2', 1)
    calls = []
    def evaluator(**kwargs):
        calls.append((kwargs["stage"], kwargs["upstream"]))
        return _exact_fixture_score(**kwargs)
    router = ContractV2BatchRouter(evaluator)
    inputs = _encode_responses([response])
    plan = router.plan([row], inputs, CharacterTokenizer(), reference=False, qualified={"G": True, "S": True})
    assert plan.selected_stages.tolist() == [1]
    assert plan.decisions[0]["state"] == "semantic_failure"
    assert [stage for stage, _ in calls] == ["G", "S"]
    assert calls[0][1] == {} and calls[1][1] == {"G": row["gsa_references"][0]}
    assert torch.equal(plan.token_ids, inputs["input_ids"][:, 1:])
    calls.clear()
    router.stage_evaluator = lambda **kwargs: calls.append(kwargs["stage"]) or .52
    abstain = router.plan([row], inputs, CharacterTokenizer(), reference=False, qualified={"G": True, "S": True})
    assert abstain.selected_stages.tolist() == [-1]
    assert calls == ["G"]
    assert abstain.decisions[0]["state"] == "indeterminate_low_confidence"


def test_anvil_missing_semantics_and_empty_suffix_do_not_invent_a_route_span():
    row = _linked_row()
    tokenizer = CharacterTokenizer()
    router = ContractV2BatchRouter()
    complete = _encode_responses([row["messages"][-1]["content"]])
    unknown = router.plan([row], complete, tokenizer, reference=False, qualified={"G": True, "S": True})
    assert unknown.selected_stages.tolist() == [-1]
    assert unknown.decisions[0]["state"] == "evaluation_unavailable"
    router.stage_evaluator = _exact_fixture_score
    prefix = row["messages"][-1]["content"].split("<RELATE>")[0]
    empty = router.plan([row], _encode_responses([prefix]), tokenizer, reference=False, qualified={"G": True, "S": True})
    assert empty.selected_stages.tolist() == [-1]
    assert empty.decisions[0]["state"] == "structural_invalid"
    assert empty.decisions[0]["reason"] == "empty_rollout_span"
    malformed = router.plan([row], _encode_responses([prefix + "broken relation"]), tokenizer,
                            reference=False, qualified={"G": True, "S": True})
    assert malformed.selected_stages.tolist() == [1]
    assert (malformed.stage_ids == 1).sum() == len("broken relation")


@pytest.mark.parametrize("selector,expected", [("semantic", 0), ("structural", -1), ("declared", 1), ("flat", 1)])
def test_control_selectors_separate_semantic_error_from_structural_validity(selector, expected):
    row = _linked_row()
    response = row["messages"][-1]["content"].replace('"mug"', '"cup"', 1)
    calls = []
    def evaluator(**kwargs):
        calls.append(kwargs["stage"])
        return _exact_fixture_score(**kwargs)
    plan = ContractV2BatchRouter(evaluator, selector=selector).plan(
        [row], _encode_responses([response]), CharacterTokenizer(), reference=False,
        qualified={"G": True, "S": True, "A": True})
    assert plan.selected_stages.tolist() == [expected]
    assert calls == (["G"] if selector == "semantic" else [])
    assert plan.decisions[0]["selector"] == selector
    if selector == "flat":
        assert (plan.stage_ids == 1).sum() == len(response)
    elif selector == "structural":
        assert plan.decisions[0]["state"] == "structural_all_pass"


def test_declared_control_uses_terminal_boundaries_without_evaluating_upstream_json():
    row = _linked_row()
    relation = row["messages"][-1]["content"].split("<RELATE>")[1]
    response = "<GROUND>{bad json}</GROUND><RELATE>" + relation
    def forbidden(**kwargs):
        raise AssertionError("reference evaluator must not enter either control selector")
    structural = ContractV2BatchRouter(forbidden, selector="structural").plan(
        [row], _encode_responses([response]), CharacterTokenizer(), reference=False, qualified={"G": True})
    declared = ContractV2BatchRouter(forbidden, selector="declared").plan(
        [row], _encode_responses([response]), CharacterTokenizer(), reference=False, qualified={"S": True})
    assert structural.selected_stages.tolist() == [0]
    assert declared.selected_stages.tolist() == [1]
    assert (declared.stage_ids == 1).sum() == len("<RELATE>" + relation)
    assert structural.decisions[0]["state"] == "structural_invalid"
    # With an ambiguous/missing delimiter, the declared control cannot borrow
    # a reference span or relabel the malformed grounding as a relation.
    missing = ContractV2BatchRouter(selector="declared").plan(
        [row], _encode_responses(["malformed grounding"]), CharacterTokenizer(), reference=False, qualified={"S": True})
    assert missing.selected_stages.tolist() == [-1]
    assert missing.decisions[0]["reason"] == "empty_rollout_span"


@pytest.mark.parametrize("selector", ["semantic", "structural", "declared", "flat"])
def test_all_control_selectors_preserve_record_and_teacher_eligibility(selector):
    row = _linked_row()
    response = "<GROUND>{bad json}</GROUND><RELATE>" + row["messages"][-1]["content"].split("<RELATE>")[1]
    router = ContractV2BatchRouter(_exact_fixture_score, selector=selector)
    inputs = _encode_responses([response])
    unqualified = router.plan([row], inputs, CharacterTokenizer(), reference=False, qualified={"G": False, "S": False})
    assert unqualified.selected_stages.tolist() == [-1]
    assert unqualified.decisions[0]["reason"] == "teacher_not_qualified"
    row["gsa_route_eligible"] = False
    ineligible = router.plan([row], inputs, CharacterTokenizer(), reference=False, qualified={"G": True, "S": True})
    assert ineligible.selected_stages.tolist() == [-1]
    assert ineligible.decisions[0]["reason"] == "record_not_route_eligible"


def test_anvil_reference_spans_and_train_split_are_not_inferred_or_reused_for_rollout():
    row = _linked_row()
    router = ContractV2BatchRouter(_exact_fixture_score)
    reference = _encode_responses([row["messages"][-1]["content"]])
    plan = router.plan([row], reference, CharacterTokenizer(), reference=True, qualified={})
    assert (plan.stage_ids == 0).any() and (plan.stage_ids == 1).any()
    assert plan.structural_mask.any()
    corrupted = deepcopy(row)
    corrupted["gsa_stage_spans"][0]["end"] -= 1
    with pytest.raises(ValueError, match="character units"):
        router.plan([corrupted], reference, CharacterTokenizer(), reference=True, qualified={})
    row["gsa_meta"]["split"] = "qualification"
    with pytest.raises(ValueError, match="explicit train split"):
        router.plan([row], reference, CharacterTokenizer(), reference=True, qualified={})


def test_anvil_decoder_preserves_sampled_segmentation_and_unicode_bytes():
    class Tokenizer:
        # Simulate two byte tokens that jointly complete one Unicode character.
        def decode(self, ids, **kwargs):
            return {(): "", (1,): "�", (1, 2): "杯", (1, 2, 3): "杯子"}[tuple(ids)]
    text, spans = decoded_token_spans(Tokenizer(), [1, 2, 3])
    assert text == "杯子" and spans == [(0, 1), (0, 1), (1, 2)]
    text = '{"label":"chair12","point_2d":[100,200],"goal":"grasp chair12"}'
    ranges = structural_ranges(text)
    assert any(text[start:end] == "grasp" for start, end in ranges)
    assert any(text[start:end] == "100" for start, end in ranges)
    assert not any(text[start:end] == "12" for start, end in ranges)


def test_anvil_template_prefix_and_post_eos_newline_keep_actual_reference_token_positions():
    row = _linked_row()
    prefix = "<think>\n\n</think>\n\n"
    reference_text = row["messages"][-1]["content"]
    encoded = _encode_responses([prefix + reference_text])
    newline = torch.tensor([[ord("\n") + 1]])
    encoded["input_ids"] = torch.cat([encoded["input_ids"], newline], dim=1)
    encoded["labels"] = torch.cat([encoded["labels"], newline], dim=1)
    encoded["attention_mask"] = torch.cat([encoded["attention_mask"], torch.ones_like(newline)], dim=1)
    router = ContractV2BatchRouter(_exact_fixture_score)
    plan = router.plan([row], encoded, CharacterTokenizer(), reference=True, qualified={}, reference_prefix=prefix)
    assert (plan.stage_ids[:, :len(prefix)] == -1).all()
    assert (plan.stage_ids[:, -2:] == -1).all()
    assert (plan.stage_ids >= 0).sum() == len(reference_text)
    assert torch.equal(plan.token_ids, encoded["input_ids"][:, 1:])
    # The same prefix freshly generated by the student is not template padding.
    sampled = router.plan([row], encoded, CharacterTokenizer(), reference=False,
                          qualified={"G": True}, reference_prefix=prefix)
    assert sampled.decisions[0]["state"] == "structural_invalid"
    assert sampled.selected_stages.tolist() == [0]
    encoded["input_ids"][0, -1] = encoded["labels"][0, -1] = ord("X") + 1
    with pytest.raises(ValueError, match="after the end-of-turn"):
        router.plan([row], encoded, CharacterTokenizer(), reference=True, qualified={}, reference_prefix=prefix)


def test_mopd_preflight_accepts_native_reference_without_rescuing_wrong_tags():
    from scripts.validate_grpo_dataset import reference_tasks
    payload = [{"label": "chair", "point_2d": [50, 50]}]
    row = {"gsa_form": "point", "gsa_family": "G", "gsa_payload": payload,
           "gsa_context": {}, "messages": [{"role": "assistant", "content": json.dumps(payload)}]}
    assert reference_tasks(row)[0][1] == payload
    row["messages"][-1]["content"] = "<GROUND>" + json.dumps(payload)
    assert reference_tasks(row)[0][1] == payload
    for answer in ("<ACT>" + json.dumps(payload), "<GROUND>" + json.dumps(payload) + "</GROUND>"):
        row["messages"][-1]["content"] = answer
        with pytest.raises(ValueError):
            reference_tasks(row)
    row["messages"][-1]["content"] = "<GROUND>[]"
    with pytest.raises(ValueError, match="payload differ"):
        reference_tasks(row)


def test_anvil_launcher_preflight_inspects_each_linked_reference_and_requires_context():
    from scripts.validate_grpo_dataset import reference_tasks
    from trains.flywheel.contracts import context_for
    row = _linked_row()
    grounding = [{"object_class": item["label"], "bbox": item["bbox_2d"]}
                 for item in row["gsa_references"][0]]
    row["gsa_stage_contexts"] = [context_for("detection", grounding), context_for("depth", grounding)]
    router = ContractV2BatchRouter()
    tasks = reference_tasks(row, router)
    assert [item[0] for item in tasks] == ["detection", "depth"]
    assert [item[1] for item in tasks] == row["gsa_references"]
    row["gsa_stage_contexts"][1] = {}
    with pytest.raises(ValueError, match="missing_multiple_groundings"):
        reference_tasks(row, router)
    row.pop("gsa_stage_contexts")
    with pytest.raises(ValueError, match="gsa_stage_contexts"):
        reference_tasks(row, router)


class TinyTokenModel(torch.nn.Module):
    def __init__(self, seed=1):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(256, 256, generator=generator) * .03)
        self.forward_inputs = []
        self.forward_output_shapes = []

    def forward(self, input_ids, logits_to_keep=None, **kwargs):
        self.forward_inputs.append(input_ids.detach().clone())
        logits = self.weight[input_ids]
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep]
        self.forward_output_shapes.append(tuple(logits.shape))
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("prompt_only", [False, True])
def test_mopd_model_inputs_preserve_images_but_exclude_routing_metadata(prompt_only):
    from scripts.gsa_mopd_plugin import GSARoutedMOPDTrainer
    trainer = GSARoutedMOPDTrainer.__new__(GSARoutedMOPDTrainer)
    trainer.model = SimpleNamespace(device=torch.device("cpu"))
    pixels = torch.randn(1, 3, 8, 8)
    def encode(data, return_length):
        assert return_length
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2),
                "pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 2, 2]]),
                **{k: v for k, v in data.items() if k in {"id", "canonical_uid", "sample_id"} or k.startswith("gsa_")}}
    trainer.template = SimpleNamespace(encode=encode, data_collator=lambda rows: rows[0])
    trainer._template_context = lambda *args, **kwargs: nullcontext()
    row = {"id": "native-id", "canonical_uid": "native-id", "sample_id": "private-lineage", "gsa_family": "G", "gsa_payload": [{"label": "chair"}],
           "gsa_form": "point", "gsa_context": {"private": True},
           "messages": [{"role": "user", "content": "<image>point"}, {"role": "assistant", "content": "[]"}]}
    encoded = trainer._prepare_batch_inputs([deepcopy(row)], encode_prompt_only=prompt_only)
    assert set(encoded) == {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    torch.testing.assert_close(encoded["pixel_values"], pixels)
    assert GSARoutedMOPDTrainer._batch_family([row]) == "G"
    assert row["gsa_context"] == {"private": True} and row["messages"][-1]["content"] == "[]"


def _cpu_anvil_trainer(student):
    # Use the real plugin compute_loss with a tiny CPU model and a deterministic
    # fixture scorer. This validates optimization mechanics, not model quality.
    from scripts.gsa_mopd_plugin import AnvilOnPolicyTrainer
    trainer = AnvilOnPolicyTrainer.__new__(AnvilOnPolicyTrainer)
    trainer.anvil_router = ContractV2BatchRouter(_exact_fixture_score)
    trainer.anvil_qualified = {family: True for family in ("G", "S", "A")}
    trainer.processing_class = CharacterTokenizer()
    trainer.template = SimpleNamespace(
        forward_context=lambda *args: nullcontext(), _get_response_prefix=lambda: "")
    trainer.args = SimpleNamespace(offload_teacher_model=False, gradient_checkpointing_kwargs={})
    trainer.anvil_options = {"topk": 8, "beta": .5, "structural_alpha": 2., "temperature": .9, "chunk_size": 16}
    trainer.anvil_sft_weight = trainer.anvil_mtd_weight = 1.
    trainer.anvil_route_log = None
    trainer._metrics = {"train": defaultdict(list)}
    trainer.teachers = {family: TinyTokenModel(seed=10 + index).requires_grad_(False)
                        for index, family in enumerate(("G", "S", "A"))}
    trainer.activated = []
    def activate(family):
        trainer.activated.append(family)
        trainer.teacher_model = trainer.teachers[family]
    trainer._activate_teacher = activate
    return trainer


def test_anvil_real_compute_loss_retains_reference_sft_and_scores_only_routed_teacher():
    from trains.mopd.firstfault_core import anvil_stage_sft_loss
    torch.set_num_threads(1)
    student = TinyTokenModel()
    trainer = _cpu_anvil_trainer(student)
    row = _linked_row()
    original = deepcopy(row)
    trainer._anvil_rows = [row]
    trainer._anvil_reference = _encode_responses([row["messages"][-1]["content"]])
    rollout_text = row["messages"][-1]["content"].replace('"rank": 1', '"rank": 2', 1)
    rollout = _encode_responses([rollout_text])
    loss = trainer.compute_loss(student, rollout)
    assert trainer.activated == ["S"]
    assert torch.equal(student.forward_inputs[0], trainer._anvil_reference["input_ids"])
    assert torch.equal(student.forward_inputs[1], rollout["input_ids"])
    assert torch.equal(trainer.teachers["S"].forward_inputs[0], rollout["input_ids"])
    before = student.weight.detach().clone()
    optimizer = torch.optim.SGD(student.parameters(), lr=.05)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, student.weight)
    assert all(parameter.grad is None for teacher in trainer.teachers.values() for parameter in teacher.parameters())
    assert row == original
    # All-pass still optimizes reference SFT, with no teacher forward at all.
    trainer.activated.clear()
    trainer._anvil_rows = [row]
    reference_plan = trainer.anvil_router.plan([row], trainer._anvil_reference, CharacterTokenizer(), reference=True, qualified={})
    expected = anvil_stage_sft_loss(student(**trainer._anvil_reference).logits[:, :-1],
                                    reference_plan.token_ids, reference_plan.stage_ids)
    all_pass = trainer.compute_loss(student, trainer._anvil_reference)
    torch.testing.assert_close(all_pass, expected)
    assert trainer.activated == []
    budget = trainer._anvil_last_budget
    assert budget["teacher_forward_calls"] == budget["teacher_scored_positions"] == budget["mtd_loss_positions"] == 0
    assert budget["reference_loss_positions"] > 0


@pytest.mark.parametrize("selector,alpha,teacher", [
    ("flat", 0., "S"), ("structural", 2., None), ("declared", 2., "S"),
    ("semantic", 0., "G"), ("semantic", 2., "G")])
def test_five_core_controls_share_reference_sft_and_score_their_actual_spans(selector, alpha, teacher):
    from trains.mopd.firstfault_core import anvil_stage_sft_loss
    torch.set_num_threads(1)
    student = TinyTokenModel()
    trainer = _cpu_anvil_trainer(student)
    trainer.anvil_router = ContractV2BatchRouter(_exact_fixture_score, selector=selector)
    trainer.anvil_options["structural_alpha"] = alpha
    row = _linked_row()
    text = row["messages"][-1]["content"]
    sampled_text = text.replace('"mug"', '"cup"', 1)
    trainer._anvil_rows = [row]
    trainer._anvil_reference = _encode_responses([text])
    inputs = _encode_responses([sampled_text])
    ref = ContractV2BatchRouter().plan([row], trainer._anvil_reference, CharacterTokenizer(), reference=True, qualified={})
    expected_sft = anvil_stage_sft_loss(student(**trainer._anvil_reference).logits[:, :-1], ref.token_ids, ref.stage_ids)
    loss = trainer.compute_loss(student, inputs)
    assert trainer._metrics["train"]["anvil/sft"][-1] == pytest.approx(expected_sft.item())
    assert trainer.activated == ([] if teacher is None else [teacher])
    loss.backward()
    assert torch.isfinite(student.weight.grad).all()
    assert all(p.grad is None for model in trainer.teachers.values() for p in model.parameters())
    budget = trainer._anvil_last_budget
    assert budget["reference_loss_positions"] == len(text)
    if selector == "flat":
        assert budget["mtd_loss_positions"] == len(sampled_text)
    elif teacher is None:
        assert budget["mtd_loss_positions"] == budget["teacher_forward_calls"] == 0
        torch.testing.assert_close(loss, expected_sft)
    else:
        segment = next(s for s in parse_stage_segments(sampled_text, ["G", "S"]) if s[0] == teacher)
        assert budget["mtd_loss_positions"] == segment[2] - segment[1]


def test_anvil_budget_matches_actual_mixed_teacher_forwards_and_saved_log(tmp_path):
    torch.set_num_threads(1)
    student = TinyTokenModel()
    trainer = _cpu_anvil_trainer(student)
    row = _linked_row()
    reference_text = row["messages"][-1]["content"]
    # Same declared S family, but one G error and one S error. Both teachers
    # score a padded batch; only one row per teacher receives MTD supervision.
    sampled = _encode_responses([reference_text.replace('"mug"', '"cup"', 1),
                                 reference_text.replace('"rank": 1', '"rank": 2', 1)])
    trainer._anvil_rows = [deepcopy(row), deepcopy(row)]
    trainer._anvil_reference = _encode_responses([reference_text, reference_text])
    trainer.anvil_route_log = tmp_path / "routes.jsonl"
    trainer.state = SimpleNamespace(global_step=7)
    loss = trainer.compute_loss(student, sampled)
    loss.backward()
    assert trainer.activated == ["G", "S"]
    actual_shapes = [shape for teacher in trainer.teachers.values() for shape in teacher.forward_output_shapes]
    budget = trainer._anvil_last_budget
    assert budget["teacher_forward_calls"] == len(actual_shapes) == 2
    assert budget["teacher_scored_rows"] == sum(shape[0] for shape in actual_shapes) == 4
    assert budget["teacher_scored_positions"] == sum(shape[0] * shape[1] for shape in actual_shapes)
    assert budget["teacher_padding_or_dummy_positions"] == budget["mtd_loss_positions"] > 0
    assert budget["sampled_attended_response_positions"] == sum(len(text) for text in
        [reference_text.replace('"mug"', '"cup"', 1), reference_text.replace('"rank": 1', '"rank": 2', 1)])
    assert budget["reference_loss_positions"] == 2 * len(reference_text)
    assert budget["teacher_attended_prefix_positions"] == sum(int((tokens != 0).sum())
        for teacher in trainer.teachers.values() for tokens in teacher.forward_inputs)
    saved = json.loads(trainer.anvil_route_log.read_text())
    assert saved["step"] == 7 and saved["budget"] == budget
    assert all(p.grad is None for teacher in trainer.teachers.values() for p in teacher.parameters())


def test_anvil_eligibility_preflight_rejects_before_trainer_initialization(tmp_path, monkeypatch):
    import scripts.gsa_mopd_plugin as plugin
    from trains.mopd.anvil_runtime import validate_teacher_eligibility
    paths = {family: str(tmp_path / family) for family in ("G", "S", "A")}
    student = str(tmp_path / "joint")
    path = tmp_path / "eligibility.json"
    report = {"verdict": "PASS", "joint_adapter": student,
              "teachers": {family: {"eligible": True, "adapter": value} for family, value in paths.items()}}
    path.write_text(json.dumps(report))
    assert validate_teacher_eligibility(path, paths, student) == {family: True for family in paths}
    with pytest.raises(ValueError, match="student initialization"):
        validate_teacher_eligibility(path, paths, str(tmp_path / "other"))
    with pytest.raises(ValueError, match="selected S adapter"):
        validate_teacher_eligibility(path, {**paths, "S": str(tmp_path / "other")}, student)
    report["verdict"] = "PENDING"
    path.write_text(json.dumps(report))
    monkeypatch.setenv("TEACHER_ELIGIBILITY_JSON", str(path))
    monkeypatch.setenv("STUDENT_ADAPTER", student)
    monkeypatch.setattr(plugin, "_adapter_paths", lambda: paths)
    def forbidden(*args, **kwargs):
        raise AssertionError("teacher/trainer initialization ran before eligibility validation")
    monkeypatch.setattr(plugin.GSARoutedMOPDTrainer, "__init__", forbidden)
    with pytest.raises(ValueError, match="not passing"):
        plugin.AnvilOnPolicyTrainer()


@pytest.mark.parametrize("backward_failure", [False, True])
def test_anvil_training_step_preserves_references_and_clears_context(monkeypatch, backward_failure):
    import scripts.gsa_mopd_plugin as plugin
    torch.set_num_threads(1)
    student = TinyTokenModel()
    trainer = _cpu_anvil_trainer(student)
    trainer.args.ds3_gather_for_generation = False
    trainer.accelerator = object()
    trainer.generation_config = object()
    row = _linked_row()
    original = deepcopy(row)
    response = row["messages"][-1]["content"].replace('"rank": 1', '"rank": 2', 1)
    sampled = _encode_responses([response])
    def encode(rows, encode_prompt_only):
        assert rows == [original]
        if encode_prompt_only:
            rows[0]["messages"][-1]["content"] = None
            return _encode_responses([""])
        return _encode_responses([rows[0]["messages"][-1]["content"]])
    trainer._prepare_batch_inputs = encode
    def generate(model, inputs, config, pad_token_id):
        assert model is student and not model.training and not torch.is_grad_enabled()
        assert inputs["input_ids"].shape[1] == 2
        return sampled["input_ids"], sampled["attention_mask"], sampled["labels"]
    trainer.generate_on_policy_outputs = generate
    monkeypatch.setattr(plugin, "unwrap_model_for_generation", lambda model, *args, **kwargs: nullcontext(model))
    def backward(self, model, inputs, num_items):
        assert model.training and torch.equal(inputs["input_ids"], sampled["input_ids"])
        loss = self.compute_loss(model, inputs)
        if backward_failure:
            raise RuntimeError("fixture backward failure")
        loss.backward()
        return loss.detach()
    monkeypatch.setattr(plugin.HFSFTTrainer, "training_step", backward)
    if backward_failure:
        with pytest.raises(RuntimeError, match="fixture backward failure"):
            trainer.training_step(student, [row])
    else:
        assert torch.isfinite(trainer.training_step(student, [row]))
        assert student.weight.grad is not None
    assert row == original and trainer._anvil_rows is None and trainer._anvil_reference is None
    assert trainer.activated == ["S"] and student.training


def _distributed_anvil_worker(rank, directory):
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    path = Path(directory)
    dist.init_process_group("gloo", init_method=(path / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        model = TinyTokenModel(seed=4)
        wrapped = DistributedDataParallel(model)
        trainer = _cpu_anvil_trainer(model)
        row = _linked_row()
        good = row["messages"][-1]["content"]
        responses = [good.replace('"bbox_2d": [100, 100, 200, 200]', '"point_2d": [100, 100]', 1),
                     good.replace('"rank": 1', '"rank": 2', 1)] if rank == 0 else [good, good]
        trainer._anvil_rows = [deepcopy(row), deepcopy(row)]
        trainer._anvil_reference = _encode_responses([good, good])
        loss = trainer.compute_loss(wrapped, _encode_responses(responses))
        loss.backward()
        torch.save({"gradient": model.weight.grad, "teacher_families": trainer.activated,
                    "eligible": trainer._metrics["train"]["anvil/eligible_local"]}, path / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_anvil_two_rank_optimizer_gradient_matches_global_batch_with_empty_rank(tmp_path):
    import torch.multiprocessing as mp
    torch.set_num_threads(1)
    mp.spawn(_distributed_anvil_worker, args=(str(tmp_path),), nprocs=2, join=True)
    left, right = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in (0, 1)]
    assert left["eligible"] == [2] and right["eligible"] == [0]
    assert left["teacher_families"] == right["teacher_families"] == ["G", "S"]
    student = TinyTokenModel(seed=4)
    trainer = _cpu_anvil_trainer(student)
    row = _linked_row()
    good = row["messages"][-1]["content"]
    responses = [good.replace('"bbox_2d": [100, 100, 200, 200]', '"point_2d": [100, 100]', 1),
                 good.replace('"rank": 1', '"rank": 2', 1), good, good]
    trainer._anvil_rows = [deepcopy(row) for _ in range(4)]
    trainer._anvil_reference = _encode_responses([good] * 4)
    trainer.compute_loss(student, _encode_responses(responses)).backward()
    torch.testing.assert_close(left["gradient"], student.weight.grad, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(right["gradient"], student.weight.grad, rtol=2e-5, atol=1e-7)


def _families():
    return ["G"] * 48 + ["S"] * 16 + ["A"] * 32


def test_distributed_batches_are_homogeneous_and_disjoint():
    families = _families()
    left = FamilyBalancedBatchSampler(families, 4, seed=7, rank=0, world_size=2)
    right = FamilyBalancedBatchSampler(families, 4, seed=7, rank=1, world_size=2)
    exposure = Counter()
    for batch_left, batch_right in zip(left, right):
        merged = batch_left + batch_right
        labels = {families[index] for index in merged}
        assert len(labels) == 1
        assert len(set(merged)) == len(merged)
        exposure[next(iter(labels))] += len(merged)
    assert exposure["G"] == exposure["S"] == exposure["A"]


def test_sampler_is_deterministic_per_epoch_and_changes_order():
    sampler = FamilyBalancedBatchSampler(_families(), 4, seed=11)
    first = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_sampler_requires_every_family_and_one_global_batch():
    with pytest.raises(ValueError, match="missing"):
        FamilyBalancedBatchSampler(["G"] * 8 + ["S"] * 8, 4)
    sampler = FamilyBalancedBatchSampler(["G"] * 8 + ["S"] * 8 + ["A"] * 3, 4)
    with pytest.raises(ValueError, match="global batch"):
        list(sampler)


def test_dataset_families_rejects_missing_metadata():
    with pytest.raises(ValueError, match="invalid or missing"):
        dataset_families([{"gsa_family": "G"}, {}])


def test_native_reference_uses_resolved_nonthinking_prefix():
    from trains.mopd.anvil_runtime import AnvilBatchRouter
    torch.set_num_threads(1)
    student = TinyTokenModel()
    trainer = _cpu_anvil_trainer(student)
    prefix = "<think>\n\n</think>\n\n"
    trainer.template.response_prefix = None
    trainer.template._get_response_prefix = lambda: prefix
    trainer.anvil_router = AnvilBatchRouter()
    row = _native_fixture("point")
    reference = row["messages"][-1]["content"]
    trainer._anvil_rows = [row]
    trainer._anvil_reference = _encode_responses([prefix + reference])
    loss = trainer.compute_loss(student, _encode_responses([reference]))
    assert torch.isfinite(loss)
    assert trainer._anvil_last_budget["reference_loss_positions"] == len(reference)
    loss.backward()
    assert torch.isfinite(student.weight.grad).all()
