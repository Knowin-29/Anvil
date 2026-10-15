"""CPU representation/binding tests; fixtures are not production evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from trains.flywheel import native_training as native


def fixture_record(form="bbox"):
    edge = {"subject": "cup", "object": "plate", "predicate": "left_of", "frame": "image_box_centers"}
    answers = {
        "bbox": [{"label": "cup", "bbox_2d": [10, 20, 30, 40]}],
        "detection": [{"label": "cup", "bbox_2d": [10, 20, 30, 40]}],
        "point": [{"label": "cup", "point_2d": [10, 20]}],
        "mask": [{"label": "cup", "polygons": [[[10, 20], [30, 40], [30, 20]]]}],
        "direction": edge,
        "metric_comparison": {**edge, "predicate": "distance", "value": .25, "unit": "normalized_image"},
        "relation_graph": [edge, {**edge, "subject": "fork"}],
        "depth_order": [{"task": "relative_depth", "subject_id": "cup", "object_id": "plate",
                         "predicate": "nearer", "frame": "camera", "truth": True}],
        "affordance": {"affordance": "open drawer", "region_bbox_2d": [10, 20, 30, 40]},
        "action_cot": {"target_object_type": "Cup", "target_role": "interaction_object",
                       "next_expert_action": "CleanObject", "remaining_high_level_actions": [
                           {"action": "CleanObject", "args": [""]}]},
    }
    variant = {"depth_order": "anvil-prism-depth-order-v1", "affordance": "anvil-affordance-region-v1",
               "action_cot": "anvil-alfred-expert-continuation-sft-v1"}.get(form, "anvil-source-native-v1")
    meta = {"native_form": form, "native_contract": variant, "source": "fixture",
            "split": "train", "training_approved": True, "group_id": "scene:fixture",
            "image_sha256": "1" * 64, "verification": "fixture_only"}
    return {"id": "fixture:0", "images": ["/fixture/image.png"], "anvil_meta": meta,
            "messages": [{"role": "user", "content": "<image>Source task"},
                         {"role": "assistant", "content": json.dumps(answers[form])}]}


def serialize(record):
    raw = (json.dumps(record) + "\n").encode()
    meta = record["anvil_meta"]
    binding = {"selection_sha256": "2" * 64, "source_sha256": "3" * 64,
               "source_line_sha256": hashlib.sha256(raw).hexdigest(), "image_sha256": meta.get("image_sha256", "1" * 64),
               "decoded_rgb_sha256": meta.get("decoded_rgb_sha256", "4" * 64), "source_offset": 0, "source_size": len(raw),
               "split": "train", "source_uid": record["id"], "source": meta["source"],
               "native_form": "QA" if meta.get("record_type") == "qa_replay" else meta["native_form"],
               "group_id": meta["group_id"], "image_path": record["images"][0]}
    return raw, binding


@pytest.mark.parametrize("form", sorted(set(native.FAMILIES) - {"dof"}))
def test_source_targets_and_messages_are_preserved(form):
    original = fixture_record(form)
    bound = native.bind_training_record(*serialize(original))
    native.validate_training_record(bound)
    assert bound["messages"] == original["messages"] and bound["images"] == original["images"]
    assert bound["gsa_family"] == native.FAMILIES[form]
    assert not bound["gsa_meta"]["human_reviewed"]
    assert not bound["gsa_meta"]["strong_model_verified"]
    assert not bound["gsa_meta"]["public_release_eligible"]
    assert "anvil_meta" not in bound


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '[NaN]', '[Infinity]', '[1e999]'])
def test_ambiguous_and_nonfinite_json_rejected(text):
    with pytest.raises(ValueError):
        native.strict_json(text)


@pytest.mark.parametrize("field,value", [("source_uid", "other"), ("source_offset", -1),
    ("source_size", 0), ("source_sha256", "bad"), ("source_line_sha256", "5" * 64),
    ("selection_sha256", ""), ("split", "benchmark"), ("image_path", "/other.png"),
    ("image_sha256", "6" * 64), ("native_form", "point"), ("group_id", "other"),
    ("source", "other"), ("decoded_rgb_sha256", "bad")])
def test_invalid_bindings_rejected(field, value):
    raw, binding = serialize(fixture_record())
    binding[field] = value
    with pytest.raises(ValueError):
        native.bind_training_record(raw, binding)


@pytest.mark.parametrize("split", ["evaluation", "benchmark", "test", "unassigned_source_only"])
def test_frozen_native_nontraining_cannot_be_relabelled(split):
    source = fixture_record()
    source["anvil_meta"]["split"] = split
    with pytest.raises(ValueError, match="nontraining"):
        native.bind_training_record(*serialize(source))


def test_dense_candidate_requires_explicit_training_selection():
    source = fixture_record()
    source["anvil_meta"].update(source="coco_train2017", split="evaluation_candidate")
    raw, binding = serialize(source)
    native.bind_training_record(raw, binding)
    binding["split"] = "benchmark"
    with pytest.raises(ValueError, match="not training"):
        native.bind_training_record(raw, binding)


@pytest.mark.parametrize("field,value", [("images", []), ("messages", [None, None]),
                                        ("messages", []), ("anvil_meta", None)])
def test_malformed_record_fails_closed(field, value):
    source = fixture_record()
    _, binding = serialize(source)
    source[field] = value
    raw = (json.dumps(source) + "\n").encode()
    binding.update(source_line_sha256=hashlib.sha256(raw).hexdigest(), source_size=len(raw))
    with pytest.raises(ValueError):
        native.bind_training_record(raw, binding)


@pytest.mark.parametrize("claim", ["human_reviewed", "strong_model_verified", "public_release_eligible"])
def test_source_binding_cannot_claim_model_human_or_release_approval(claim):
    bound = native.bind_training_record(*serialize(fixture_record()))
    bound["gsa_meta"][claim] = True
    with pytest.raises(ValueError, match="unsupported"):
        native.validate_training_record(bound)


def test_changed_content_or_linked_relabel_is_rejected():
    bound = native.bind_training_record(*serialize(fixture_record()))
    changed = deepcopy(bound)
    changed["messages"][0]["content"] += " changed"
    with pytest.raises(ValueError, match="content changed"):
        native.validate_training_record(changed)
    bound["gsa_stages"] = ["G", "S", "A"]
    with pytest.raises(ValueError, match="linked"):
        native.validate_training_record(bound)


def test_decoded_source_fingerprint_cannot_be_rebound():
    source = fixture_record()
    source["anvil_meta"]["decoded_rgb_sha256"] = "4" * 64
    raw, binding = serialize(source)
    native.bind_training_record(raw, binding)
    binding["decoded_rgb_sha256"] = "5" * 64
    with pytest.raises(ValueError, match="decoded image"):
        native.bind_training_record(raw, binding)


@pytest.mark.parametrize("field,value", [("bbox_2d", [True, 2, 3, 4]), ("bbox_2d", [1, 2, 1, 4]),
                                        ("bbox_2d", [1, 2, 1001, 4]), ("label", "")])
def test_invalid_geometry_or_label_rejected(field, value):
    answer = [{"label": "cup", "bbox_2d": [10, 20, 30, 40]}]
    answer[0][field] = value
    with pytest.raises(ValueError):
        native.validate_target("bbox", "anvil-source-native-v1", answer, "")


def test_empty_source_argument_preserved_but_nonstring_rejected():
    source = fixture_record("action_cot")
    bound = native.bind_training_record(*serialize(source))
    answer = json.loads(bound["messages"][1]["content"])
    assert answer["remaining_high_level_actions"][0]["args"] == [""]
    answer["remaining_high_level_actions"][0]["args"] = [None]
    with pytest.raises(ValueError):
        native.validate_target("action_cot", source["anvil_meta"]["native_contract"], answer, "")


def test_metric_units_cannot_be_promoted_to_meters():
    source = fixture_record("metric_comparison")
    answer = json.loads(source["messages"][1]["content"])
    answer["unit"] = "m"
    with pytest.raises(ValueError, match="physical"):
        native.validate_target("metric_comparison", "anvil-source-native-v1", answer, "")


def test_gripper_uses_original_source_geometry_validator():
    from tools.map_prism_native import CONTRACT, LENGTH_UNIT
    answer = {"contract": CONTRACT, "entity_id": "cup", "label": "Cup",
              "camera_from_gripper": [[1., 0., 0., 0.], [0., 1., 0., 0.],
                                      [0., 0., 1., 1.], [0., 0., 0., 1.]],
              "contact_point_camera": [0., 0., 1.], "contact_point_px": [4., 3.],
              "approach_axis_camera": [0., 0., 1.], "jaw_axis_camera": [1., 0., 0.],
              "frame": "camera_x_right_y_down_z_forward", "length_unit": LENGTH_UNIT,
              "pixel_coordinates": "zero_based_xy"}
    prompt = "<image>" + json.dumps({"contract": CONTRACT, "input": {
        "intrinsics": [[4., 0., 4.], [0., 4., 3.], [0., 0., 1.]],
        "image_size_wh": [8, 6], "entity": {"entity_id": "cup"}}})
    native.validate_target("dof", CONTRACT, answer, prompt)
    answer["length_unit"] = "m"
    with pytest.raises(ValueError):
        native.validate_target("dof", CONTRACT, answer, prompt)


def fixture_qa():
    row = fixture_record()
    row["anvil_meta"] = {"schema": "anvil-clevr-official-qa-replay-v1", "source": "CLEVR",
        "record_type": "qa_replay", "split": "train", "training_approved": True,
        "independent_question": True, "official_question_id": 12, "group_id": "CLEVR:train:1",
        "official_archive_sha256": "a" * 64, "official_questions_sha256": "b" * 64,
        "image_sha256": "1" * 64, "decoded_rgb_sha256": "4" * 64, "verification": "fixture_official_match"}
    row["messages"][1]["content"] = "rubber"
    return row


@pytest.mark.parametrize('corruption', [None, 'hash', 'selection', 'missing_form'])
def test_native_mtd_preflight_all_spatial_streams_and_qa_reserved(tmp_path, monkeypatch, corruption):
    from scripts.validate_grpo_dataset import main
    rows = {}
    for form in [*sorted(native.FAMILIES), 'QA']:
        row = fixture_qa() if form == 'QA' else evaluation_fixture_row(form)
        row['anvil_meta'].update(split='train', training_approved=True)
        rows[form] = native.bind_training_record(*serialize(row))
    manifest = {'state':'all_selected_training_rows_exported', 'dropped_records':0,
        'records':12, 'selection_sha256':'2'*64, 'files':{}}
    for form,row in rows.items():
        name = f'train_{form}.jsonl'; raw = (native.canonical(row)+'\n').encode()
        (tmp_path/name).write_bytes(raw)
        manifest['files'][name] = {'form':form,'family':row['gsa_family'],'records':1,
                                 'sha256':hashlib.sha256(raw).hexdigest()}
    if corruption == 'hash': manifest['files']['train_bbox.jsonl']['sha256'] = 'f'*64
    if corruption == 'selection': manifest['selection_sha256'] = 'f'*64
    if corruption == 'missing_form': del manifest['files']['train_bbox.jsonl']
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    output = tmp_path/'report.json'
    monkeypatch.setattr('sys.argv', ['validate_grpo_dataset.py', str(tmp_path), '--source-native', '--out', str(output)])
    if corruption:
        with pytest.raises(ValueError): main()
        assert not output.exists()
    else:
        assert main() == 0
        report = json.loads(output.read_text())
        assert report['records'] == 11 and report['qa_replay_records_reserved_for_sft'] == 1
        assert len(report['dataset_files']) == 11 and all('train_QA' not in p for p in report['dataset_files'])
        assert set(report['by_form']) == set(native.FAMILIES)


def test_qa_plaintext_is_reference_only_not_native_json():
    row = fixture_qa()
    result = native.bind_training_record(*serialize(row))
    native.validate_training_record(result)
    assert result["messages"] == row["messages"]
    assert result["gsa_family"] == "QA" and result["gsa_meta"]["reference_only"] is True
    assert result["gsa_contract_version"] == native.QA_VERSION
    assert "gsa_stages" not in result


@pytest.mark.parametrize("key,value", [("split", "test"), ("training_approved", False),
    ("independent_question", False), ("source", "other"), ("official_question_id", True),
    ("official_questions_sha256", "bad"), ("official_archive_sha256", "bad")])
def test_qa_requires_official_independent_training_source(key, value):
    row = fixture_qa(); row["anvil_meta"][key] = value
    with pytest.raises(ValueError): native.bind_training_record(*serialize(row))


@pytest.mark.parametrize("key,value", [("split", "benchmark"), ("native_form", "point"),
    ("source_line_sha256", "f" * 64), ("source_size", 2), ("decoded_rgb_sha256", "e" * 64),
    ("image_path", "/other.png"), ("source_offset", -1)])
def test_qa_rejects_changed_binding(key, value):
    raw, binding = serialize(fixture_qa()); binding[key] = value
    with pytest.raises(ValueError): native.bind_training_record(raw, binding)


@pytest.mark.parametrize("key,value", [("reference_only", False), ("strong_model_verified", True),
    ("human_reviewed", True), ("public_release_eligible", True), ("binding", None)])
def test_qa_admission_cannot_be_promoted(key, value):
    bound = native.bind_training_record(*serialize(fixture_qa()))
    bound["gsa_meta"][key] = value
    with pytest.raises(ValueError): native.validate_training_record(bound)


def test_export_checks_image_alias_and_rejects_changed_pixels(tmp_path):
    from PIL import Image
    from tools.arxiv_source_scale import fingerprint
    from tools.export_native_training import checked_image
    original, alias, changed = [tmp_path / name for name in ("image.png", "alias.bmp", "other.png")]
    Image.new("RGB", (8, 6), "red").save(original)
    Image.new("RGB", (8, 6), "red").save(alias)
    Image.new("RGB", (8, 6), "blue").save(changed)
    fp = fingerprint(original)
    assert checked_image(str(alias), str(original), fp["sha256"], fp["decoded_rgb_sha256"]) != fp["sha256"]
    with pytest.raises(ValueError, match="differs"):
        checked_image(str(changed), str(original), fp["sha256"], fp["decoded_rgb_sha256"])


def test_export_dense_group_uses_index_namespace(tmp_path):
    from PIL import Image
    from tools.arxiv_source_scale import fingerprint
    from tools.export_native_training import convert_selected
    image = tmp_path / "dense.png"; Image.new("RGB", (8, 6), "red").save(image)
    fp = fingerprint(image); row = fixture_record()
    row["images"] = [str(image)]
    row["anvil_meta"].update(source="coco_train2017", group_id=fp["decoded_rgb_sha256"],
                              image_sha256=fp["sha256"], decoded_rgb_sha256=fp["decoded_rgb_sha256"])
    raw, _ = serialize(row)
    entry = {"uid": row["id"], "split": "train", "source": "coco_train2017", "form": "bbox",
        "variant": "anvil-source-native-v1", "scene_group": "decoded_image:" + fp["decoded_rgb_sha256"],
        "image_path": str(image), "file_sha256": fp["sha256"], "decoded_sha256": fp["decoded_rgb_sha256"],
        "byte_offset": 0, "byte_length": len(raw), "source_sha256": "a" * 64}
    form, line = convert_selected(raw, entry, "b" * 64)
    assert form == "bbox" and json.loads(line)["gsa_meta"]["binding"]["group_id"] == fp["decoded_rgb_sha256"]
    entry["scene_group"] = fp["decoded_rgb_sha256"]
    with pytest.raises(ValueError, match="group"): convert_selected(raw, entry, "b" * 64)


@pytest.fixture
def training_export_fixture(tmp_path):
    from argparse import Namespace
    import sqlite3
    from PIL import Image
    from tools.arxiv_source_scale import fingerprint, sha
    image = tmp_path / "source.png"; Image.new("RGB", (8, 6), "red").save(image)
    fp = fingerprint(image)
    rows = [fixture_record(), fixture_qa(), fixture_record()]
    for i, row in enumerate(rows):
        row["id"] = f"fixture:{i}"; row["images"] = [str(image)]
        row["anvil_meta"].update(image_sha256=fp["sha256"], decoded_rgb_sha256=fp["decoded_rgb_sha256"])
    source = tmp_path / "source.jsonl"
    raw = [(json.dumps(row) + "\n").encode() for row in rows]; source.write_bytes(b"".join(raw))
    source_manifest = tmp_path / "source_manifest.json"; source_manifest.write_text('{}')
    index = tmp_path / "selection.sqlite"
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE images(image_id INTEGER,file_sha256 TEXT)')
        db.execute('INSERT INTO images VALUES(1,?)', (fp["sha256"],))
        db.execute('CREATE TABLE selection(uid,split,source,form,variant,scene_group,byte_offset,byte_length,'
                   'source_path,source_sha256,image_path,decoded_sha256,image_id,file_id)')
        offset = 0
        for i, row in enumerate(rows):
            meta = row["anvil_meta"]; qa = i == 1
            db.execute('INSERT INTO selection VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (row["id"], "benchmark" if i == 2 else "train", meta["source"], "QA" if qa else "bbox",
                 meta["schema"] if qa else meta["native_contract"],
                 meta["group_id"] if qa else meta["source"] + ":" + meta["group_id"],
                 offset, len(raw[i]), str(source), sha(source), str(image), fp["decoded_rgb_sha256"], 1, 1))
            offset += len(raw[i])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"selection_complete": True, "selection_sqlite_sha256": sha(index),
        "observed": {"rows_by_split": {"train": 2}}, "inputs": [{"path": str(source), "expected_sha256": sha(source),
            "manifest": str(source_manifest), "manifest_sha256": sha(source_manifest)}]}))
    return Namespace(index=index, manifest=manifest, output=tmp_path / "export", workers=1)


@pytest.mark.parametrize("workers", [1, 2])
def test_full_training_export_preserves_all_selected_rows_and_qa(training_export_fixture, workers):
    from tools.export_native_training import export
    args = training_export_fixture; args.workers = workers; result = export(args)
    assert result["records"] == 2 and result["dropped_records"] == 0
    assert result["by_form"] == {"bbox": 1, "QA": 1}
    ids = set()
    for name in result["files"]:
        for line in (args.output / name).read_text().splitlines():
            row = json.loads(line); native.validate_training_record(row); ids.add(row["id"])
    assert ids == {"fixture:0", "fixture:1"}
    assert not list(args.output.glob('*.pending'))
    assert not result["tokenization_complete"] and not result["public_release_eligible"]
    with pytest.raises(ValueError, match="immutable"): export(args)


def test_training_export_source_tamper_never_writes_complete_manifest(training_export_fixture):
    from tools.export_native_training import export
    args = training_export_fixture
    (args.index.parent / "source.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="changed before"): export(args)
    assert not (args.output / "manifest.json").exists()


def fixture_export_manifest():
    files = {f'train_{form}.jsonl': {'form': form, 'family': family, 'records': 100}
             for form, family in native.FAMILIES.items()}
    files['train_QA.jsonl'] = {'form': 'QA', 'family': 'QA', 'records': 200}
    return {'state': 'all_selected_training_rows_exported', 'dropped_records': 0, 'files': files}


@pytest.mark.parametrize('scope', ['G', 'S', 'A', 'joint'])
@pytest.mark.parametrize('sampling', ['record', 'family-form'])
def test_native_sft_full_scope_sampling_and_qa_share(scope, sampling):
    from tools.train_native_specialist import selected_streams
    streams, probabilities = selected_streams(fixture_export_manifest(), scope, sampling)
    assert sum(probabilities) == pytest.approx(1)
    if scope == 'joint':
        assert len(streams) == 12
        qa_probability = next(p for (_, info), p in zip(streams, probabilities) if info['form'] == 'QA')
        assert qa_probability == pytest.approx(200 / 1300)
    else:
        assert all(info['family'] == scope and info['form'] != 'QA' for _, info in streams)


def test_native_sft_rejects_missing_forms_or_unsafe_filename():
    from tools.train_native_specialist import selected_streams
    manifest = fixture_export_manifest(); del manifest['files']['train_dof.jsonl']
    with pytest.raises(ValueError, match='lacks'): selected_streams(manifest, 'A', 'record')
    manifest = fixture_export_manifest()
    manifest['files']['../train_QA.jsonl'] = manifest['files'].pop('train_QA.jsonl')
    with pytest.raises(ValueError, match='Invalid'): selected_streams(manifest, 'joint', 'record')


def test_native_sft_command_uses_all_files_strict_encoding_and_full_shuffle(tmp_path):
    from argparse import Namespace
    from tools.train_native_specialist import selected_streams, training_command
    args = Namespace(export=tmp_path, model=tmp_path / 'base', output=tmp_path / 'out', steps=1024,
        gradient_accumulation=8, seed=17, learning_rate=1e-4, max_length=8192)
    streams, probabilities = selected_streams(fixture_export_manifest(), 'A', 'family-form')
    command = training_command(args, streams, probabilities)
    for name, _ in streams: assert str(tmp_path / name) in command
    for flag, value in [('strict', 'true'), ('streaming', 'false'), ('lazy_tokenize', 'true'),
                        ('eval_strategy', 'no'), ('split_dataset_ratio', '0'), ('max_steps', '1024')]:
        assert command[command.index('--' + flag) + 1] == value
    assert '--resume_from_checkpoint' not in command and '--val_dataset' not in command
    assert command[command.index('--remove_unused_columns') + 1] == 'true'


def evaluation_fixture_row(form):
    if form != 'dof':
        row = fixture_record(form)
    else:
        from tools.map_prism_native import CONTRACT, LENGTH_UNIT
        row = fixture_record()
        row['anvil_meta'].update(native_form='dof', native_contract=CONTRACT)
        answer = {'contract': CONTRACT, 'entity_id': 'cup', 'label': 'Cup',
            'camera_from_gripper': [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 1.], [0., 0., 0., 1.]],
            'contact_point_camera': [0., 0., 1.], 'contact_point_px': [4., 3.],
            'approach_axis_camera': [0., 0., 1.], 'jaw_axis_camera': [1., 0., 0.],
            'frame': 'camera_x_right_y_down_z_forward', 'length_unit': LENGTH_UNIT, 'pixel_coordinates': 'zero_based_xy'}
        row['messages'][0]['content'] = '<image>\n' + json.dumps({'contract': CONTRACT, 'input': {
            'intrinsics': [[4., 0., 4.], [0., 4., 3.], [0., 0., 1.]], 'image_size_wh': [8, 6], 'entity': {'entity_id': 'cup'}}})
        row['messages'][1]['content'] = json.dumps(answer)
    row['id'] = 'eval:' + form
    row['anvil_meta'].update(split='validation', training_approved=False)
    return row


def make_eval_entry(row, image, fp, raw):
    meta = row['anvil_meta']; group = meta['group_id']; source = meta['source']
    selected_group = 'decoded_image:' + group if source in native.DENSE else (group if group.startswith(source + ':') else source + ':' + group)
    return {'uid': row['id'], 'split': 'benchmark', 'source': source, 'form': meta['native_form'],
        'variant': meta['native_contract'], 'scene_group': selected_group,
        'image_path': str(image), 'file_sha256': fp['sha256'], 'decoded_sha256': fp['decoded_rgb_sha256'],
        'byte_offset': 0, 'byte_length': len(raw), 'source_sha256': hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize('form', sorted(native.FAMILIES))
def test_evaluation_conversion_preserves_every_native_form(tmp_path, form):
    from PIL import Image
    from tools.arxiv_source_scale import fingerprint
    from tools.export_native_evaluation import convert_selected
    image = tmp_path / 'eval.png'; Image.new('RGB', (8, 6), 'red').save(image); fp = fingerprint(image)
    row = evaluation_fixture_row(form); row['images'] = [str(image)]
    row['anvil_meta']['image_sha256'] = fp['sha256']; raw = (json.dumps(row) + '\n').encode()
    entry = make_eval_entry(row, image, fp, raw)
    got_form, line = convert_selected(raw, entry, 'b' * 64, {})
    result = json.loads(line)
    assert got_form == form and result['messages'] == row['messages']
    assert result['training_approved'] is False and result['evaluation_meta']['split'] == 'benchmark'
    assert result['image_size_wh'] == [8, 6]
    if form != 'action_cot':
        assert convert_selected(raw, entry, 'b' * 64, {}, 'explicit-schema-v1') == (got_form, line)
    else:
        with pytest.raises(ValueError, match='only supports the source ALFRED'):
            convert_selected(raw, entry, 'b' * 64, {}, 'explicit-schema-v1')
    entry['split'] = 'train'
    with pytest.raises(ValueError, match='training row'): convert_selected(raw, entry, 'b' * 64, {})


def test_evaluation_alfred_rendering_must_bind_exact_original(tmp_path):
    from PIL import Image
    from tools.arxiv_source_scale import fingerprint
    from tools.export_native_evaluation import convert_selected
    from tools.map_alfred_native import SFT_INSTRUCTION, SFT_OUTPUT_SCHEMA
    from tools.run_native_evaluation import validate_row
    image = tmp_path / 'eval.png'; Image.new('RGB', (8, 6), 'red').save(image); fp = fingerprint(image)
    row = evaluation_fixture_row('action_cot'); row['images'] = [str(image)]
    row['messages'][0]['content'] = '<image>\n' + SFT_INSTRUCTION + 'put the mug away'
    row['anvil_meta'].update(source='ALFRED', group_id='ALFRED:scene:fixture', image_sha256=fp['sha256'])
    original = {'id': row['id'], 'schema_version': 'anvil-alfred-source-native-action-v1',
        'training_approved': False, 'input': {'image': str(image), 'image_sha256': fp['sha256']}}
    raw = (json.dumps(original) + '\n').encode()
    row['anvil_meta'].update(source_record_id=row['id'], source_native_contract=original['schema_version'],
        source_record_canonical_sha256=hashlib.sha256(json.dumps(original, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
        source_binding={'records_sha256': hashlib.sha256(raw).hexdigest()})
    entry = make_eval_entry(row, image, fp, raw); entry['variant'] = original['schema_version']
    _, line = convert_selected(raw, entry, 'c' * 64, {row['id']: row})
    assert json.loads(line)['messages'] == row['messages']
    assert 'prompt_rendering' not in json.loads(line)['evaluation_meta']
    before = deepcopy(row)
    _, explicit_line = convert_selected(raw, entry, 'c' * 64, {row['id']: row}, 'explicit-schema-v1')
    explicit = json.loads(explicit_line)
    assert row == before
    assert explicit['images'] == row['images'] and explicit['id'] == row['id']
    assert explicit['messages'][1] == row['messages'][1]
    assert explicit['messages'][0]['content'].endswith('\nTask goal: put the mug away')
    assert explicit['evaluation_meta']['evaluation_content_sha256'] != json.loads(line)['evaluation_meta']['evaluation_content_sha256']
    assert explicit['evaluation_meta']['prompt_rendering'] == {
        'policy': 'explicit-schema-v1', 'reference_changed': False, 'goal_changed': False,
        'source_prompt_sha256': hashlib.sha256(row['messages'][0]['content'].encode()).hexdigest(),
        'output_schema_sha256': native.digest(SFT_OUTPUT_SCHEMA)}
    validate_row(explicit, 'action_cot', 'c' * 64)
    with pytest.raises(ValueError, match='Unknown ALFRED'):
        convert_selected(raw, entry, 'c' * 64, {row['id']: row}, 'unknown')
    changed = deepcopy(row)
    changed['messages'][0]['content'] = 'changed prompt'
    with pytest.raises(ValueError, match='unchanged source'):
        convert_selected(raw, entry, 'c' * 64, {row['id']: changed}, 'explicit-schema-v1')
    row['anvil_meta']['source_record_canonical_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='bind this raw'): convert_selected(raw, entry, 'c' * 64, {row['id']: row})


def test_evaluation_export_exact_all_forms_and_no_qa(tmp_path):
    from argparse import Namespace
    import sqlite3
    from PIL import Image
    from tools.arxiv_source_scale import fingerprint, sha
    from tools.export_native_evaluation import export
    image = tmp_path / 'eval.png'; Image.new('RGB', (8, 6), 'red').save(image); fp = fingerprint(image)
    rows = [evaluation_fixture_row(form) for form in sorted(native.FAMILIES)]
    for row in rows:
        row['images'] = [str(image)]; row['anvil_meta']['image_sha256'] = fp['sha256']
    raw_rows = [(json.dumps(row) + '\n').encode() for row in rows]
    source = tmp_path / 'source.jsonl'; source.write_bytes(b''.join(raw_rows))
    sm = tmp_path / 'source_manifest.json'; sm.write_text('{}')
    index = tmp_path / 'selection.sqlite'
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE images(image_id INTEGER,file_sha256 TEXT)')
        db.execute('INSERT INTO images VALUES(1,?)', (fp['sha256'],))
        db.execute('CREATE TABLE selection(uid,split,source,form,variant,scene_group,byte_offset,byte_length,'
                   'source_path,source_sha256,image_path,decoded_sha256,image_id,file_id)')
        offset = 0
        for row, raw in zip(rows, raw_rows):
            e = make_eval_entry(row, image, fp, raw)
            db.execute('INSERT INTO selection VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (e['uid'], e['split'], e['source'], e['form'], e['variant'], e['scene_group'], offset, len(raw),
                 str(source), sha(source), str(image), fp['decoded_rgb_sha256'], 1, 1))
            offset += len(raw)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'selection_complete': True, 'selection_sqlite_sha256': sha(index),
        'observed': {'rows_by_split': {'benchmark': 11}}, 'inputs': [{'path': str(source), 'expected_sha256': sha(source),
            'manifest': str(sm), 'manifest_sha256': sha(sm)}]}))
    alfred = tmp_path / 'alfred'; alfred.mkdir(); (alfred / 'evaluation.jsonl').write_text('')
    (alfred / 'manifest.json').write_text(json.dumps({'state': 'source_native_expert_continuation_sft_exported',
        'contract': 'anvil-alfred-expert-continuation-sft-v1', 'files': {'evaluation.jsonl': sha(alfred / 'evaluation.jsonl')},
        'counts': {'evaluation': {'records': 0}}, 'source_binding': {'evaluation': {'records_path': str(source),
            'records_sha256': sha(source), 'manifest_path': str(sm), 'manifest_sha256': sha(sm)}}}))
    args = Namespace(index=index, manifest=manifest, alfred_sft=alfred, output=tmp_path / 'export')
    result = export(args)
    assert result['records'] == 11 and set(result['by_form']) == set(native.FAMILIES)
    assert result['dropped_records'] == 0 and result['model_evaluation_performed'] is False
    assert not list(args.output.glob('*.pending')) and 'eval_QA.jsonl' not in result['files']


def native_scoring_fixture(form, source='pixmo_points'):
    row = evaluation_fixture_row(form)
    if form == 'mask':
        row['messages'][1]['content'] = json.dumps([{'label': 'cup', 'polygons': [[[0, 0], [500, 0], [500, 800], [0, 800]]]}])
    row.update(native_form=form, native_variant=row['anvil_meta']['native_contract'],
               image_size_wh=[8, 6], evaluation_meta={'source': source})
    return row


def reference_check_fixture(tmp_path):
    from tools.arxiv_source_scale import sha
    root = tmp_path / 'references'; root.mkdir()
    files = {}; selection = 'a' * 64
    for form in native.FAMILIES:
        row = native_scoring_fixture(form)
        row.pop('anvil_meta')
        row.update(schema='anvil-source-native-evaluation-v1', family=native.FAMILIES[form], training_approved=False)
        row['evaluation_meta'].update(split='benchmark', selection_sha256=selection)
        row['evaluation_meta']['evaluation_content_sha256'] = native.digest({k: v for k, v in row.items() if k != 'evaluation_meta'})
        path = root / ('eval_' + form + '.jsonl'); path.write_text(native.canonical(row) + '\n')
        files[path.name] = {'form': form, 'family': native.FAMILIES[form], 'records': 1, 'sha256': sha(path)}
    manifest = {'state': 'all_selected_native_evaluation_rows_exported', 'dropped_records': 0, 'targets_repaired': 0,
                'files': files, 'by_form': dict.fromkeys(native.FAMILIES, 1), 'records': 11,
                'source_inputs': [], 'selection_sha256': selection}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root, manifest


def test_full_reference_check_is_not_model_evaluation(tmp_path):
    from argparse import Namespace
    from tools.export_native_evaluation import check_references
    root, manifest = reference_check_fixture(tmp_path)
    args = Namespace(export=root, output=tmp_path/'checked', workers=2)
    result = check_references(args)
    assert result['records'] == 11 and set(result['forms']) == set(native.FAMILIES)
    assert result['all_references_score_ideally'] and result['model_calls'] == 0
    assert not any(result[k] for k in ('semantic_quality_established', 'teacher_qualified', 'model_evaluation_complete', 'references_changed'))
    with pytest.raises(ValueError, match='new reference-check'): check_references(args)
    assert json.loads((root/'manifest.json').read_text()) == manifest


@pytest.mark.parametrize('problem', ['file_hash', 'duplicate', 'binding', 'truncated_line', 'count'])
def test_reference_check_rejects_changed_or_incomplete_input(tmp_path, problem):
    from tools.export_native_evaluation import check_reference_form
    from tools.arxiv_source_scale import sha
    root, manifest = reference_check_fixture(tmp_path)
    name = 'eval_bbox.jsonl'; path = root/name; info = manifest['files'][name]
    if problem == 'file_hash': info['sha256'] = '0' * 64
    if problem == 'count': info['records'] = 2
    if problem == 'duplicate': path.write_text(path.read_text() * 2)
    if problem == 'truncated_line': path.write_text(path.read_text().rstrip('\n'))
    if problem == 'binding':
        row = json.loads(path.read_text()); row['messages'][0]['content'] += 'changed'
        path.write_text(native.canonical(row) + '\n')
    if problem not in {'file_hash', 'count'}: info['sha256'] = sha(path)
    with pytest.raises(ValueError): check_reference_form(root, name, info, manifest)


def test_reference_check_exposes_source_region_mismatch(tmp_path, monkeypatch):
    import numpy as np
    from eval import native_metrics
    from tools.export_native_evaluation import check_reference_form
    from tools.arxiv_source_scale import sha
    root, manifest = reference_check_fixture(tmp_path)
    name = 'eval_point.jsonl'; path = root/name
    row = json.loads(path.read_text()); row['evaluation_meta']['source'] = 'coco_train2017'
    path.write_text(native.canonical(row) + '\n'); manifest['files'][name]['sha256'] = sha(path)
    manifest['source_inputs'] = [{'source': 'coco_train2017'}]
    closed = []
    class Regions:
        def __init__(self, manifest): pass
        def __call__(self, row):
            region = np.ones((6, 8), dtype=bool); region[0, 0] = False
            return region
        def close(self): closed.append(True)
    monkeypatch.setattr(native_metrics, 'CocoPointReference', Regions)
    result, identities = check_reference_form(root, name, manifest['files'][name], manifest)
    assert result['records'] == 1 and result['metric_issues'] == {'interior_point_f1': 1}
    assert result['groups'][0]['nonideal_records'] == 1 and closed == [True]
    assert identities == {row['id']}


def test_reference_check_failure_does_not_emit_complete_report(tmp_path):
    from argparse import Namespace
    from tools.export_native_evaluation import check_references
    root, manifest = reference_check_fixture(tmp_path)
    (root/'eval_bbox.jsonl').write_text('changed\n')
    output = tmp_path/'checked'
    with pytest.raises(ValueError, match='Reference file changed'):
        check_references(Namespace(export=root, output=output, workers=2))
    assert (output/'failure.json').is_file() and not (output/'report.json').exists()


@pytest.mark.parametrize('form', sorted(native.FAMILIES))
def test_native_metrics_exact_reference_and_failure_denominator(form):
    from eval.native_metrics import score_row, metric_names
    row = native_scoring_fixture(form)
    perfect = score_row(row, row['messages'][1]['content'])
    assert perfect['json_valid'] and perfect['contract_valid']
    errors = {'rotation_error_degrees', 'pi_symmetric_rotation_error_degrees',
              'normalized_projected_contact_l2', 'normalized_image_distance_mae'}
    for key in metric_names(row): assert perfect[key] == pytest.approx(0 if key in errors else 1, abs=1e-6)
    for response, reason in [('not JSON', 'stop'), (row['messages'][1]['content'], 'length')]:
        failed = score_row(row, response, finish_reason=reason)
        for key, value in metric_names(row).items(): assert failed[key] == value


def test_coco_interior_point_credits_alternative_valid_location():
    import numpy as np
    from eval.native_metrics import score_row
    row = native_scoring_fixture('point', source='coco_train2017')
    region = np.ones((6, 8), dtype=bool)
    prediction = json.dumps([{'label': 'cup', 'point_2d': [900, 900]}])
    assert score_row(row, prediction, point_region=region)['interior_point_f1'] == 1
    region[5, 7] = False
    assert score_row(row, prediction, point_region=region)['interior_point_f1'] == 0
    with pytest.raises(ValueError, match='source instance mask'): score_row(row, prediction)
    prediction = json.dumps([{'label': 'cup', 'point_2d': [1000, 500]}])
    assert score_row(row, prediction, point_region=region)['interior_point_f1'] == 0


def test_point_matching_counts_false_positives_and_uses_explicit_scale():
    from eval.native_metrics import score_row
    row = native_scoring_fixture('point'); target = json.loads(row['messages'][1]['content'])
    assert score_row(row, json.dumps(target + target))['annotated_point_f1_r50'] == pytest.approx(2/3)
    target[0]['point_2d'] = [900, 900]
    assert score_row(row, json.dumps(target))['annotated_point_f1_r50'] == 0


def test_bbox_matching_is_order_invariant_and_penalizes_extra_instances():
    from eval.native_metrics import score_row
    row = native_scoring_fixture('detection')
    targets = [{'label': 'cup', 'bbox_2d': [0, 0, 400, 400]}, {'label': 'plate', 'bbox_2d': [500, 500, 900, 900]}]
    row['messages'][1]['content'] = json.dumps(targets)
    score = score_row(row, json.dumps(targets[::-1]))
    assert score['matched_box_iou'] == 1 and score['box_f1_iou50'] == 1
    extra = score_row(row, json.dumps(targets + targets[:1]))
    assert extra['matched_box_iou'] == pytest.approx(2/3) and extra['box_f1_iou50'] == pytest.approx(4/5)
    targets[0]['label'] = 'a different cup'
    assert score_row(row, json.dumps(targets))['box_f1_iou50'] == .5


@pytest.mark.parametrize('response', ['{"x":NaN}', '{"x":1,"x":2}', '[1e999]', 'null', '[]'])
def test_native_metrics_reject_invalid_or_ambiguous_answers(response):
    from eval.native_metrics import score_row
    score = score_row(native_scoring_fixture('bbox'), response)
    assert not score['contract_valid'] and score['matched_box_iou'] == 0


def test_native_summary_retains_failures_and_rejects_duplicate_ids():
    from eval.native_metrics import score_row, summarize
    row = native_scoring_fixture('bbox')
    valid = score_row(row, row['messages'][1]['content']); invalid = score_row(row, 'bad')
    scored = [{'id': str(i), 'native_form': 'bbox', 'source': 'fixture', 'metrics': values}
              for i, values in enumerate([valid, invalid])]
    result = summarize(scored)
    assert result['n'] == 2 and result['groups'][0]['mean']['matched_box_iou'] == .5
    assert result['global_composite_score'] is None
    with pytest.raises(ValueError, match='duplicated'): summarize(scored + scored[:1])


def test_rotation_identity_is_not_penalized_by_source_roundoff():
    from eval.native_metrics import score_row
    row = native_scoring_fixture('dof'); target = json.loads(row['messages'][1]['content'])
    target['camera_from_gripper'][0][0] -= 1e-8
    target['jaw_axis_camera'][0] -= 1e-8
    row['messages'][1]['content'] = json.dumps(target)
    score = score_row(row, row['messages'][1]['content'])
    assert score['contract_valid']
    assert score['rotation_error_degrees'] == pytest.approx(0., abs=1e-9)


def test_native_eval_binds_optional_kernel_versions(monkeypatch):
    from tools import run_native_evaluation as runner
    optional = {'triton', 'flash-linear-attention', 'fla-core', 'causal-conv1d', 'pycocotools'}
    installed = {}
    def version(name):
        if name in optional and name not in installed:
            raise runner.importlib.metadata.PackageNotFoundError(name)
        return installed.get(name, 'fixture-version')
    monkeypatch.setattr(runner.importlib.metadata, 'version', version)
    before = runner.runtime_environment()
    assert all(before[name] is None for name in optional)
    installed['fla-core'] = 'fixture-new-kernel'
    after = runner.runtime_environment()
    assert after != before and after['fla-core'] == 'fixture-new-kernel'
    assert before['torch'] == after['torch'] == 'fixture-version'


def test_native_eval_missing_mask_binary_fails_before_model_inference(monkeypatch):
    from tools import run_native_evaluation as runner
    calls = []
    def missing(name):
        calls.append(name)
        raise ImportError('fixture missing mask extension')
    monkeypatch.setattr(runner.importlib, 'import_module', missing)
    runner.verify_metric_dependencies([('depth', {'form': 'depth_order'})])
    assert not calls
    for form in ('mask', 'point'):
        with pytest.raises(ValueError, match='before inference'):
            runner.verify_metric_dependencies([(form, {'form': form})])
    assert calls == ['pycocotools.mask', 'pycocotools.mask']


def test_native_eval_requested_fast_path_never_silently_falls_back():
    from types import SimpleNamespace
    from tools.run_native_evaluation import verify_fast_kernels
    names = ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
             'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated')
    module = SimpleNamespace(is_fast_path_available=True, **dict.fromkeys(names, lambda: None))
    verify_fast_kernels(module)
    for name in names:
        original = getattr(module, name); setattr(module, name, None)
        with pytest.raises(ValueError, match='unavailable'): verify_fast_kernels(module)
        setattr(module, name, original)
    module.is_fast_path_available = False
    with pytest.raises(ValueError, match='unavailable'): verify_fast_kernels(module)


def test_native_eval_shards_partition_every_form_without_sampling():
    from tools.run_native_evaluation import selected_files
    counts = {form: i + 1 for i, form in enumerate(native.FAMILIES)}
    manifest = {'state': 'all_selected_native_evaluation_rows_exported', 'dropped_records': 0,
        'targets_repaired': 0, 'records': sum(counts.values()), 'by_form': counts,
        'files': {f'eval_{form}.jsonl': {'form': form, 'family': native.FAMILIES[form], 'records': count}
                  for form, count in counts.items()}}
    for scope in ['G', 'S', 'A', 'joint']:
        observed = {}
        for shard in range(4):
            for _, info in selected_files(manifest, scope, shard, 4):
                observed[info['form']] = observed.get(info['form'], 0) + info['shard_records']
        assert observed == {f: n for f, n in counts.items() if scope == 'joint' or native.FAMILIES[f] == scope}
    with pytest.raises(ValueError, match='shard'): selected_files(manifest, 'joint', 4, 4)


def test_eval_gpu_pool_skips_busy_card_without_eviction(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import run_native_evaluation as runner
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'GPU-busy\nGPU-idle\n')
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(runner, 'gpu_snapshot', lambda uuid: {'uuid': uuid,
        'compute_pid_present': uuid == 'GPU-busy', 'memory_mib': 9000 if uuid == 'GPU-busy' else 0,
        'utilization_percent': 50 if uuid == 'GPU-busy' else 0})
    args = SimpleNamespace(gpu='0,1',lock_root=tmp_path/'locks',output=tmp_path)
    lock, uuid = runner.acquire_gpu(args)
    try:
        assert uuid == 'GPU-idle'
        assert len(json.loads((tmp_path/'gpu_allocation.json').read_text())['checks']) == 2
    finally: lock.close()


def test_prediction_resume_rejects_wrong_weight_reference_order_and_changed_score():
    from tools.run_native_evaluation import prediction_item, validate_saved
    row = native_scoring_fixture('bbox')
    row['evaluation_meta']['evaluation_content_sha256'] = native.digest({k:v for k,v in row.items() if k != 'evaluation_meta'})
    item = prediction_item(row, 7, 'bad JSON', 'stop', None, 'a' * 64)
    validate_saved(item, row, 7, 'a' * 64)
    for key, value in [('ordinal', 8), ('id', 'different'), ('protocol_sha256', 'b'*64),
                       ('evaluation_content_sha256', 'c'*64), ('source', 'different')]:
        altered = deepcopy(item); altered[key] = value
        with pytest.raises(ValueError): validate_saved(altered, row, 7, 'a'*64)
    altered = deepcopy(item); altered['metrics']['matched_box_iou'] = 1.
    with pytest.raises(ValueError): validate_saved(altered, row, 7, 'a'*64)


def test_native_terminal_checkpoint_requires_exact_step_base_and_receipt(tmp_path):
    from tools.run_native_evaluation import terminal_adapter, sha
    protocol = {'steps': 9, 'model': '/fixture/base'}
    (tmp_path/'status.json').write_text(json.dumps({'stage': 'training'}))
    assert terminal_adapter(tmp_path, protocol) is None
    checkpoint = tmp_path/'training/checkpoint-9'; checkpoint.mkdir(parents=True)
    adapter = checkpoint/'adapter_model.safetensors'; adapter.write_bytes(b'fixture-only')
    (checkpoint/'adapter_config.json').write_text(json.dumps({'peft_type': 'LORA', 'base_model_name_or_path': '/fixture/base'}))
    (checkpoint/'args.json').write_text('{}')
    (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step': 9}))
    (tmp_path/'training.result.json').write_text(json.dumps({'exit_code': 0}))
    (tmp_path/'status.json').write_text(json.dumps({'stage': 'sft_complete_evaluation_pending',
        'terminal_adapter': str(adapter), 'adapter_sha256': sha(adapter)}))
    assert terminal_adapter(tmp_path, protocol)['files_sha256']['adapter_model.safetensors'] == sha(adapter)
    (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step': 8}))
    with pytest.raises(ValueError, match='terminal'): terminal_adapter(tmp_path, protocol)


@pytest.mark.parametrize('generation_version,pretrained', [
    ('anvil-native-evaluation-run-v2',False), ('anvil-native-evaluation-run-v3',False),
    ('anvil-native-evaluation-run-v4',False), ('anvil-native-evaluation-run-v4',True)])
def test_native_full_scope_driver_with_fake_engine_and_exact_resume(tmp_path, monkeypatch, generation_version,pretrained):
    """CPU fixture integration, explicitly not a GPU/model performance result."""
    from argparse import Namespace
    from contextlib import nullcontext
    from dataclasses import dataclass
    from types import ModuleType, SimpleNamespace
    import sys
    from PIL import Image
    from tools import run_native_evaluation as runner
    current_version = runner.VERSION
    monkeypatch.setattr(runner, 'VERSION', generation_version)
    export = tmp_path/'export'; export.mkdir()
    from tools.response_text import SWIFT_NON_THINKING_PREFIX
    prefix='' if pretrained else SWIFT_NON_THINKING_PREFIX
    image = tmp_path/'image.png'; Image.new('RGB', (8, 6)).save(image)
    files = {}; answers = {}; selection = 'a'*64
    for form in native.FAMILIES:
        row = native_scoring_fixture(form)
        if form == 'bbox': row['messages'][1]['content'] = row['messages'][1]['content'].replace('cup', 'caf\u00e9 \u676f\u5b50')
        row.update(schema='anvil-source-native-evaluation-v1', family=native.FAMILIES[form],
                   training_approved=False, images=[str(image)], id='fixture:'+form)
        row['messages'][0]['content'] += '\nTask fixture ' + form if form != 'dof' else ''
        row['evaluation_meta'].update(split='benchmark', selection_sha256=selection, image_sha256=runner.sha(image))
        row['evaluation_meta']['evaluation_content_sha256'] = native.digest({k:v for k,v in row.items() if k != 'evaluation_meta'})
        name = f'eval_{form}.jsonl'; (export/name).write_text(native.canonical(row)+'\n', encoding='utf-8')
        files[name] = {'form':form, 'family':native.FAMILIES[form], 'records':1, 'sha256':runner.sha(export/name)}
        answers[row['messages'][0]['content']] = row['messages'][1]['content']
    (export/'manifest.json').write_text(json.dumps({'state':'all_selected_native_evaluation_rows_exported',
        'dropped_records':0, 'targets_repaired':0, 'selection_sha256':selection, 'records':11,
        'by_form':dict.fromkeys(native.FAMILIES,1), 'files':files, 'source_inputs':[]}))
    train = tmp_path/'train'; train.mkdir()
    (train/'manifest.json').write_text(json.dumps({'state':'all_selected_training_rows_exported','selection_sha256':selection}))
    model = tmp_path/'model'; model.mkdir(); (model/'config.json').write_text('{}')
    proto = tmp_path/'base_protocol.json'; proto.write_text(json.dumps({'model':str(model),
        'export':str(train), 'export_manifest_sha256':runner.sha(train/'manifest.json'),
        'base_files_sha256':{'config.json':runner.sha(model/'config.json')}}))
    baseline=None
    if pretrained:
        (model/'config.json').write_text(json.dumps({'model_type':'qwen3_vl','architectures':['Qwen3VLForConditionalGeneration']}))
        (model/'model.safetensors').write_bytes(b'fixture')
        (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'model.safetensors'}}))
        baseline=tmp_path/'baseline.json'
        baseline.write_text(json.dumps({'schema':'anvil-pretrained-baseline-binding-v1',
            'state':'local_files_match_pinned_hub_revision','model':str(model),'model_type':'qwen3_vl',
            'repository':'fixture/baseline','revision':'a'*40,
            'base_files_sha256':{p.name:runner.sha(p) for p in model.iterdir()}}))
    calls = []
    @dataclass
    class Usage:
        prompt_tokens:int = 1
        completion_tokens:int = 1
        total_tokens:int = 2
    class Engine:
        def __init__(self,*a,**kw):
            assert json.loads((args.output/'status.json').read_text())['stage']=='loading_model'
            assert kw['model_type']==('qwen3_vl' if pretrained else 'qwen3_5')
            self.model=SimpleNamespace(eval=lambda:None)
        def infer(self, requests, config, **kwargs):
            assert json.loads((args.output/'status.json').read_text())['stage']=='inference'
            calls.extend(requests)
            assert all(len(r.messages)==1 and r.messages[0]['role']=='user' for r in requests)
            return [SimpleNamespace(usage=Usage(),choices=[SimpleNamespace(finish_reason='stop',
                    message=SimpleNamespace(content=prefix+answers[r.messages[0]['content']]))]) for r in requests]
    for module_name in ['swift','swift.infer_engine','swift.model','swift.template','torch']:
        monkeypatch.setitem(sys.modules,module_name,ModuleType(module_name))
    sys.modules['torch'].bfloat16='fixture'
    sys.modules['swift.infer_engine'].TransformersEngine=Engine
    sys.modules['swift.infer_engine'].InferRequest=SimpleNamespace
    sys.modules['swift.infer_engine'].RequestConfig=SimpleNamespace
    sys.modules['swift.model'].get_model_processor=lambda *a,**k:(None,object())
    sys.modules['swift.template'].get_template=lambda *a,**k:SimpleNamespace(response_prefix=prefix)
    monkeypatch.setattr(runner,'acquire_gpu',lambda args:(nullcontext(),'GPU-fixture'))
    args=Namespace(export=export,model=model,output=tmp_path/'run',lock_root=tmp_path/'locks',
        training_run=None,base_protocol=None if pretrained else proto,pretrained_binding=baseline,
        scope='joint',shard=0,shards=1,batch_size=4,resume=False)
    # Reproduce a C/ASCII locale: model text must specify UTF-8 explicitly.
    from pathlib import Path
    original_open = Path.open
    def ascii_default(path, mode='r', buffering=-1, encoding=None, *extra, **kwargs):
        if path.name.startswith(('predictions_', 'rescored_')) and 'b' not in mode:
            assert encoding == 'utf-8', 'Prediction I/O must not inherit an ASCII locale'
        return original_open(path,mode,buffering,encoding,*extra,**kwargs)
    monkeypatch.setattr(Path,'open',ascii_default)
    runner.evaluate(args)
    result=json.loads((args.output/'metrics.json').read_text())
    assert result['n']==11 and len(calls)==11 and result['state']=='complete_scope_shard'
    assert result['teacher_qualified'] is False and result['global_composite_score'] is None
    request=json.loads((args.output/'request.json').read_text())
    if pretrained:
        assert request['training_protocol_sha256'] is None and request['training_run'] is None
        assert request['pretrained_binding_sha256']==runner.sha(baseline)
        assert request['response_text_policy']['sdk_response_prefix']==''
    args.resume=True; runner.evaluate(args)
    assert len(calls)==11, 'Saved exact prefix should not be inferred a second time'
    assert json.loads((args.output/'metrics.json').read_text())==result
    monkeypatch.setattr(runner, 'VERSION', current_version)
    # Re-score immutable completed outputs without invoking the fake model again.
    import shutil
    from tools.rescore_native_evaluation import rescore
    for name in runner.CODE:
        destination=args.output/'code'/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(__import__('pathlib').Path(runner.__file__).parents[1]/name,destination)
    revised=rescore(args.output,export,tmp_path/'derived')
    assert revised['n']==11 and revised['scope_complete'] and revised['model_calls']==0
    assert revised['groups']==result['groups'] and len(calls)==11
    partial=rescore(args.output,export,tmp_path/'one_form',['bbox'])
    assert partial['n']==1 and not partial['scope_complete']
    with pytest.raises(ValueError,match='new derived'):rescore(args.output,export,tmp_path/'derived')
    from tools import rescore_native_evaluation as rescorer
    watched=tmp_path/'form_watch'; watched.mkdir()
    requested=rescorer.form_wait_request(args.output,export,['depth_order','bbox'])
    binding={'generation_request_sha256':runner.sha(args.output/'request.json'),
             'form_request':requested,'code_sha256':{},'producer':{'pid':123}}
    # A producer may have exited after writing every selected completion receipt.
    monkeypatch.setattr(runner,'process_identity',lambda pid:None)
    accepted=rescorer.wait_for_forms(args.output,export,watched,binding)
    assert accepted['n']==2 and accepted['scope_complete'] is False
    assert accepted['state']=='complete_selected_forms_rescored' and len(calls)==11
    for form,info in accepted['completed_forms'].items():
        assert runner.sha(watched/info['metrics'])==info['metrics_sha256']
        assert json.loads((watched/info['metrics']).read_text())['groups']==[
            v for v in result['groups'] if v['form']==form]
    receipt=args.output/'complete_bbox.json'; old_receipt=receipt.read_bytes()
    receipt.write_text('{}')
    (tmp_path/'bad_watch').mkdir()
    with pytest.raises(ValueError,match='incomplete'):
        rescorer.wait_for_forms(args.output,export,tmp_path/'bad_watch',binding)
    receipt.write_bytes(old_receipt)
    with pytest.raises(ValueError,match='Invalid or repeated'):
        rescorer.form_wait_request(args.output,export,['bbox','bbox'])
    with pytest.raises(ValueError,match='Invalid or repeated'):
        rescorer.form_wait_request(args.output,export,['../bad'])
    data=export/'eval_bbox.jsonl'; data.write_text(data.read_text()+'\n')
    monkeypatch.setattr(runner, 'VERSION', generation_version)
    with pytest.raises(ValueError,match='file changed'): runner.evaluate(args)


def test_portable_data_private_terms_do_not_rewrite_content():
    from tools.export_portable_data import private_pattern, validate_portable, image_path
    pattern = private_pattern(['PrivateOwner', '/internal/server/'])
    assert pattern.search(b'PrivateOwner') and pattern.search(b'/internal/server/data.png')
    assert not pattern.search(b'PrivateOwnership')
    assert image_path('a'*64) == 'images/aa/' + 'a'*64 + '.img'
    with pytest.raises(ValueError): image_path('../escape')


def test_portable_images_keep_exact_bytes_and_reject_identity_metadata(tmp_path):
    from PIL import Image, PngImagePlugin
    from tools.arxiv_source_scale import fingerprint
    from tools.export_portable_data import read_image, private_pattern
    image = tmp_path/'image.png'; Image.new('RGB', (7, 5), 'red').save(image)
    fp = fingerprint(image)
    binding = (fp['sha256'], {'paths': {str(image)}, 'decoded': fp['decoded_rgb_sha256'],
                             'split': 'train', 'sources': {'CLEVR'}})
    record, payload = read_image(binding, private_pattern([]))
    assert payload == image.read_bytes() and record['image_size_wh'] == [7, 5]
    metadata = PngImagePlugin.PngInfo(); metadata.add_text('Author', 'PrivateOwner')
    Image.new('RGB', (7,5), 'red').save(image, pnginfo=metadata)
    fp = fingerprint(image); binding = (fp['sha256'], {**binding[1], 'decoded':fp['decoded_rgb_sha256']})
    with pytest.raises(ValueError, match='Private term'): read_image(binding, private_pattern(['PrivateOwner']))


def test_portable_complete_fixture_roundtrip_and_corruption_rejection(tmp_path):
    from argparse import Namespace
    import gzip
    from PIL import Image
    from tools.arxiv_source_scale import fingerprint
    from tools import export_portable_data as portable
    selection = '2'*64; roots = {}; source_messages = {}
    for split in ['train','benchmark']:
        root=tmp_path/split; root.mkdir(); roots[split]=root
        image=tmp_path/(split+'.png'); Image.new('RGB',(8,6),'red' if split=='train' else 'blue').save(image)
        fp=fingerprint(image); files={}
        forms=list(native.FAMILIES)+(['QA'] if split=='train' else [])
        for form in forms:
            row=fixture_qa() if form=='QA' else evaluation_fixture_row(form)
            row['id']=split+':'+form;row['images']=[str(image)]
            row['anvil_meta'].update(source='CLEVR' if form=='QA' else 'pixmo_points',
                image_sha256=fp['sha256'],decoded_rgb_sha256=fp['decoded_rgb_sha256'],
                split='train' if split=='train' else 'evaluation_candidate',training_approved=split=='train')
            if split=='train':
                raw,binding=serialize(row);binding['selection_sha256']=selection
                row=native.bind_training_record(raw,binding)
            else:
                from tools.export_native_evaluation import convert_selected
                raw=(json.dumps(row)+'\n').encode();entry=make_eval_entry(row,image,fp,raw)
                # Dense candidates are selected by canonical decoded-image group.
                entry['scene_group']='decoded_image:'+row['anvil_meta']['group_id']
                _,line=convert_selected(raw,entry,selection,{})
                row=json.loads(line)
            name=('train' if split=='train' else 'eval')+'_'+form+'.jsonl'
            (root/name).write_text(native.canonical(row)+'\n')
            files[name]={'form':form,'family':native.FAMILIES.get(form,'QA'),'records':1,'sha256':portable.sha(root/name)}
            source_messages[row['id']]=row['messages']
        m={'state':'all_selected_training_rows_exported' if split=='train' else 'all_selected_native_evaluation_rows_exported',
           'dropped_records':0,'targets_repaired':0,'records':len(forms),'unique_images':1,'selection_sha256':selection,
           'by_form':dict.fromkeys(forms,1),'files':files,'source_inputs':[]}
        (root/'manifest.json').write_text(json.dumps(m))
    args=Namespace(training=roots['train'],evaluation=roots['benchmark'],output=tmp_path/'package',
        private_work=tmp_path/'private',private_term=['PrivateOwner'],rows_per_shard=2,images_per_shard=1,workers=2)
    result=portable.build(args)
    assert result['records']==23 and result['unique_images']==result['image_files']==2
    assert result['messages_changed']==result['records_dropped']==0 and not result['public_release_eligible']
    for p in args.output.glob('records/*/*.gz'):
        with gzip.open(p,'rt') as s:
            for line in s:
                r=json.loads(line);assert r['messages']==source_messages[r['id']]
                assert not r['images'][0].startswith('/') and str(tmp_path) not in line
    assert portable.verify(args.output,['PrivateOwner'])['records']==23
    image_archive=next((args.output/'media').glob('*.tar'))
    with image_archive.open('ab') as s:s.write(b'corrupt')
    with pytest.raises(ValueError,match='file identity'):portable.verify(args.output)


def coco_license_fixture(tmp_path, *, missing=False, unknown_license=False):
    """Synthetic metadata only; no claim about production image permissions."""
    import zipfile
    from tools import export_portable_data as portable
    package, evaluation, source = (tmp_path / name for name in ('package', 'evaluation', 'source'))
    for p in (package, evaluation, source): p.mkdir()
    key, decoded = 'a' * 64, 'b' * 64
    row = {'sha256': key, 'canonical_rgb_sha256': decoded, 'sources': ['coco_train2017'],
           'image_size_wh': [8, 6], 'split': 'train'}
    (package/'image_index.jsonl').write_text(native.canonical(row) + '\n')
    originals = [{'id': i, 'width': 8, 'height': 6, 'license': 999 if unknown_license else i,
                  'flickr_url': 'https://example.org/image/' + str(i)} for i in (1, 2)]
    archive = source/'annotations.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('annotations/instances_train2017.json', native.canonical({
            'images': originals, 'licenses': [{'id': 1, 'name': 'fixture-by'}, {'id': 2, 'name': 'fixture-nc'}]}))
    rows = [{'source': 'coco_train2017', 'sha256': 'c' * 64 if missing else key,
             'decoded_rgb_sha256': decoded, 'source_image_id': str(i)} for i in (1, 2)]
    (source/'images.jsonl').write_text(''.join(native.canonical(r) + '\n' for r in rows))
    sm = source/'manifest.json'
    sm.write_text(native.canonical({'source': 'coco_train2017', 'input_binding': {'annotations': {
        'path': str(archive), 'sha256': portable.sha(archive)}}, 'files': {
        'images.jsonl': {'sha256': portable.sha(source/'images.jsonl')}}}))
    em = evaluation/'manifest.json'
    em.write_text(native.canonical({'state': 'all_selected_native_evaluation_rows_exported',
        'source_inputs': [{'source': 'coco_train2017', 'manifest_sha256': portable.sha(sm)}]}))
    (package/'manifest.json').write_text(native.canonical({'state': 'complete_local_portable_candidate',
        'parent_manifest_sha256': {'benchmark': portable.sha(em)}, 'image_files': 1,
        'file_inventory': {'image_index.jsonl': {'sha256': portable.sha(package/'image_index.jsonl')}}}))
    return package, evaluation, sm


def test_coco_license_inventory_keeps_conflicting_identities_without_permission(tmp_path):
    from tools import export_portable_data as portable
    package, evaluation, sm = coco_license_fixture(tmp_path)
    before = portable.sha(package/'manifest.json'); output = tmp_path/'inventory'
    result = portable.audit_coco_licenses(package, evaluation, sm, output)
    assert result['coco_image_files'] == result['coco_multi_id_images'] == 1
    assert result['coco_license_sets'] == {'1,2': 1}
    assert not result['creator_attribution_complete'] and not result['redistribution_review_complete']
    assert not result['public_release_eligible'] and not result['images_or_targets_changed']
    row = json.loads((output/'coco_image_licenses.jsonl').read_text())
    assert row['license_ids'] == [1, 2] and len(row['official_images']) == 2
    assert portable.sha(package/'manifest.json') == before
    with pytest.raises(ValueError, match='new license inventory'):
        portable.audit_coco_licenses(package, evaluation, sm, output)


@pytest.mark.parametrize('kind', ['missing', 'unknown_license', 'changed_bytes'])
def test_coco_license_inventory_rejects_unbound_or_missing_evidence(tmp_path, kind):
    from tools import export_portable_data as portable
    package, evaluation, sm = coco_license_fixture(tmp_path, missing=kind == 'missing',
                                                   unknown_license=kind == 'unknown_license')
    if kind == 'changed_bytes':
        with (sm.parent/'annotations.zip').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='no bound license|unknown official license|input bytes changed'):
        portable.audit_coco_licenses(package, evaluation, sm, tmp_path/'inventory')
    assert not (tmp_path/'inventory').exists()


def test_portable_image_privacy_distinguishes_upstream_paths_and_pixel_bytes(tmp_path):
    import struct
    import zlib
    from PIL import Image, PngImagePlugin
    from tools.arxiv_source_scale import fingerprint
    from tools.export_portable_data import read_image, private_pattern
    image = tmp_path/'upstream.png'
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('File', '/mnt/upstream-public-renderer/scene.blend')
    # An uncompressed IDAT deliberately contains the name as RGB bytes, not text.
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    image.write_bytes(b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', 5, 1, 8, 2, 0, 0, 0))
        + chunk(b'tEXt', b'File\x00/mnt/upstream-public-renderer/scene.blend')
        + chunk(b'IDAT', zlib.compress(b'\x00PrivateOwner\x00\x00\x00', level=0))
        + chunk(b'IEND', b''))
    pattern = private_pattern(['PrivateOwner'], generic_paths=False)
    assert pattern.search(image.read_bytes()) is not None
    def binding():
        fp = fingerprint(image)
        return (fp['sha256'], {'paths':{str(image)}, 'decoded':fp['decoded_rgb_sha256'],
                               'split':'train', 'sources':{'CLEVR'}})
    assert read_image(binding(), pattern)[1] == image.read_bytes()
    # A real owner path in compressed metadata must still fail.
    metadata.add_text('PrivateLocation', '/mnt/PrivateOwner/run', zip=True)
    Image.new('RGB', (4, 1), 'red').save(image, pnginfo=metadata)
    with pytest.raises(ValueError, match='Private term in source image metadata'):
        read_image(binding(), pattern)
    assert private_pattern([]).search(b'/mnt/upstream-public-renderer/scene.blend')


@pytest.mark.parametrize('body', ['{}', '```json\n{}\n```', '<think>reason</think>{}',
                                 '<think>\n\n</think>\n\n{}', ' not JSON ', ''])
def test_swift_response_scaffold_removes_exactly_one_transport_prefix(body):
    from tools.response_text import generated_text, SWIFT_NON_THINKING_PREFIX as prefix
    assert generated_text(prefix+body,prefix)==body
    assert generated_text(body,'')==body
    assert generated_text(prefix+body,'')==prefix+body
    with pytest.raises(ValueError):generated_text('{}',prefix)
    with pytest.raises(ValueError):generated_text('<think>reason', '<think>')


def test_rescore_legacy_keeps_raw_and_rejects_prediction_tampering():
    from tools.run_native_evaluation import prediction_item
    from tools.rescore_native_evaluation import validate_and_rescore
    from tools.response_text import SWIFT_NON_THINKING_PREFIX as prefix
    row=native_scoring_fixture('bbox')
    row['evaluation_meta']['evaluation_content_sha256']=native.digest({k:v for k,v in row.items() if k!='evaluation_meta'})
    raw=prefix+row['messages'][1]['content']
    old=prediction_item(row,3,raw,'stop',None,'a'*64)
    assert not old['metrics']['json_valid']
    revised=validate_and_rescore(old,row,3,'a'*64,True)
    assert revised['response']==raw and revised['metrics']['contract_valid']
    assert revised['metrics']['matched_box_iou']==pytest.approx(1.)
    for key,value in [('ordinal',4),('id','different'),('protocol_sha256','b'*64)]:
        wrong=deepcopy(old);wrong[key]=value
        with pytest.raises(ValueError):validate_and_rescore(wrong,row,3,'a'*64,True)
    fenced=prefix+'```json\n'+row['messages'][1]['content']+'\n```'
    old=prediction_item(row,3,fenced,'stop',None,'a'*64)
    assert not validate_and_rescore(old,row,3,'a'*64,True)['metrics']['json_valid']


@pytest.mark.parametrize('fault',[None,'revision','size','lfs','blob','architecture','index','extra'])
def test_public_pretrained_binding_verifies_complete_files_without_downloading(tmp_path,monkeypatch,fault):
    from types import SimpleNamespace
    import huggingface_hub
    from tools import run_native_evaluation as tool
    model=tmp_path/'model'; model.mkdir()
    config={'model_type':'qwen3_vl','architectures':['Qwen3VLForConditionalGeneration']}
    if fault=='architecture': config['model_type']='different'
    (model/'config.json').write_text(json.dumps(config))
    (model/'model.safetensors').write_bytes(b'fixture weights')
    index={'weight_map':{'weight':'absent.safetensors' if fault=='index' else 'model.safetensors'}}
    (model/'model.safetensors.index.json').write_text(json.dumps(index))
    entries=[]
    for path in model.iterdir():
        raw=path.read_bytes(); lfs=path.suffix=='.safetensors'
        entries.append(SimpleNamespace(rfilename=path.name,size=len(raw)+(1 if fault=='size' else 0),
            blob_id='0'*40 if fault=='blob' else hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
            lfs=SimpleNamespace(sha256='0'*64 if fault=='lfs' else tool.sha(path)) if lfs else None))
    # An SDK cache marker is not an input to model inference.
    (model/'.model-metadata.json').write_text('{}')
    if fault=='extra': (model/'unbound.json').write_text('{}')
    calls=[]
    class Api:
        def __init__(self,token): assert token is False
        def model_info(self,repository,revision,files_metadata):
            calls.append(repository); assert revision=='a'*40 and files_metadata is True
            return SimpleNamespace(sha='b'*40 if fault=='revision' else revision,siblings=entries)
    monkeypatch.setattr(huggingface_hub,'HfApi',Api)
    output=tmp_path/'binding.json'
    before={p.name:p.read_bytes() for p in model.iterdir()}
    if fault:
        with pytest.raises(ValueError): tool.bind_public_model(model,'fixture/model','a'*40,output)
        assert not output.exists()
    else:
        result=tool.bind_public_model(model,'fixture/model','a'*40,output)
        assert tool.pretrained_binding(model,output)==result
        assert len(result['base_files_sha256'])==3 and result['model_calls']==0
        assert not result['training_claimed'] and not result['full_evaluation_complete']
        with pytest.raises(ValueError,match='new pretrained'):tool.bind_public_model(model,'fixture/model','a'*40,output)
    assert calls==['fixture/model']
    assert before=={p.name:p.read_bytes() for p in model.iterdir()}


def test_rescore_watcher_follows_exact_process_and_never_restarts_gpu(tmp_path,monkeypatch):
    from tools import rescore_native_evaluation as tool
    run=tmp_path/'run';run.mkdir();out=tmp_path/'watch';out.mkdir()
    (run/'request.json').write_text('{}')
    producer={'pid':123,'start_ticks':'99','cmdline_sha256':'a'*64}
    binding={'producer':producer,'generation_request_sha256':tool.sha(run/'request.json'),'code_sha256':{}}
    monkeypatch.setattr(tool.runner,'process_identity',lambda pid:dict(producer))
    calls=[]
    def complete(seconds):
        assert seconds==20
        (run/'metrics.json').write_text('{}')
        (run/'status.json').write_text(json.dumps({'stage':'complete_scope_shard'}))
    monkeypatch.setattr(tool.time,'sleep',complete)
    monkeypatch.setattr(tool,'rescore',lambda *args:calls.append(args) or {'n':11})
    assert tool.wait_for_scope(run,tmp_path/'data',out,binding)=={'n':11}
    assert len(calls)==1 and json.loads((out/'status.json').read_text())['stage']=='complete'
    (run/'status.json').write_text(json.dumps({'stage':'inference'}))
    monkeypatch.setattr(tool.runner,'process_identity',lambda pid:{**producer,'start_ticks':'100'})
    with pytest.raises(ValueError,match='PID changed'):tool.wait_for_scope(run,tmp_path/'data',out,binding)
    assert len(calls)==1


@pytest.fixture
def incremental_watch(tmp_path,monkeypatch):
    from tools import rescore_native_evaluation as tool
    run=tmp_path/'run'; run.mkdir(); export=tmp_path/'export'; export.mkdir()
    output=tmp_path/'watch'; output.mkdir()
    for path in (run/'request.json',run/'protocol.json',export/'manifest.json'): path.write_text('{}')
    producer={'pid':123,'start_ticks':'99','cmdline_sha256':'a'*64}
    requested={'forms':['bbox','depth_order'],
               'generation_protocol_sha256':tool.sha(run/'protocol.json'),
               'export_manifest_sha256':tool.sha(export/'manifest.json')}
    binding={'producer':producer,'generation_request_sha256':tool.sha(run/'request.json'),
             'code_sha256':{},'form_request':requested}
    monkeypatch.setattr(tool,'form_wait_request',lambda *args:requested)
    monkeypatch.setattr(tool.runner,'process_identity',lambda pid:dict(producer))
    calls=[]
    def accept(run,export,destination,forms):
        assert len(forms)==1
        calls.append(forms[0]); destination.mkdir(parents=True)
        metrics={'n':1,'scope_complete':False,'state':'complete_forms_rescored'}
        (destination/'metrics.json').write_text(json.dumps(metrics))
        return metrics
    monkeypatch.setattr(tool,'rescore',accept)
    return tool,run,export,output,binding,calls


def test_incremental_watch_accepts_each_form_once_before_later_completion(incremental_watch,monkeypatch):
    tool,run,export,output,binding,calls=incremental_watch
    (run/'complete_bbox.json').write_text('{}')
    def advance(seconds):
        assert seconds==20 and calls==['bbox']
        index=json.loads((output/'index.json').read_text())
        assert index['n']==1 and index['scope_complete'] is False
        assert json.loads((output/'status.json').read_text())['pending_forms']==['depth_order']
        (run/'complete_depth_order.json').write_text('{}')
    monkeypatch.setattr(tool.time,'sleep',advance)
    result=tool.wait_for_forms(run,export,output,binding)
    assert calls==['bbox','depth_order'] and result['n']==2
    assert result['model_calls']==0 and result['teacher_qualified'] is False
    assert result['scope_complete'] is False and not result['public_release_eligible']
    assert json.loads((output/'status.json').read_text())['stage']=='complete_selected_forms'


@pytest.mark.parametrize('cause',['pid','failure','request','protocol','export','code'])
def test_incremental_watch_stops_on_changed_identity_and_retains_accepted_forms(incremental_watch,monkeypatch,cause):
    tool,run,export,output,binding,calls=incremental_watch
    (run/'complete_bbox.json').write_text('{}')
    def break_source(seconds):
        assert calls==['bbox']
        if cause=='pid': monkeypatch.setattr(tool.runner,'process_identity',lambda pid:None)
        elif cause=='failure': (run/'failure.json').write_text('{}')
        elif cause=='request': (run/'request.json').write_text('{"changed":true}')
        elif cause=='protocol': (run/'protocol.json').write_text('{"changed":true}')
        elif cause=='export': (export/'manifest.json').write_text('{"changed":true}')
        else:
            binding['code_sha256']['tools/rescore_native_evaluation.py']='0'*64
            (run/'complete_depth_order.json').write_text('{}')
    monkeypatch.setattr(tool.time,'sleep',break_source)
    with pytest.raises(ValueError): tool.wait_for_forms(run,export,output,binding)
    assert calls==['bbox']
    accepted=json.loads((output/'index.json').read_text())
    assert list(accepted['completed_forms'])==['bbox'] and not accepted['scope_complete']
    assert (output/'forms/bbox/metrics.json').is_file()


@pytest.fixture
def native_merge_fixture(tmp_path):
    """Two unequal CPU fixture shards; never an experimental model result."""
    import shutil
    from pathlib import Path
    from tools import rescore_native_evaluation as tool
    runner=tool.runner; export=tmp_path/'export';export.mkdir()
    files={}; rows={}; selection='a'*64
    for form in native.FAMILIES:
        rows[form]=[]
        for ordinal in range(3):
            row=native_scoring_fixture(form)
            row.update(schema='anvil-source-native-evaluation-v1',family=native.FAMILIES[form],
                       training_approved=False,images=['/fixture/image.png'],id=f'{form}:{ordinal}')
            row['evaluation_meta'].update(split='benchmark',selection_sha256=selection,image_sha256='b'*64)
            row['evaluation_meta']['evaluation_content_sha256']=native.digest({k:v for k,v in row.items() if k!='evaluation_meta'})
            rows[form].append(row)
        path=export/f'eval_{form}.jsonl';path.write_text(''.join(native.canonical(r)+'\n' for r in rows[form]),encoding='utf-8')
        files[path.name]={'form':form,'family':native.FAMILIES[form],'records':3,'sha256':tool.sha(path)}
    manifest={'state':'all_selected_native_evaluation_rows_exported','dropped_records':0,'targets_repaired':0,
              'selection_sha256':selection,'records':33,'by_form':dict.fromkeys(native.FAMILIES,3),'files':files,'source_inputs':[]}
    (export/'manifest.json').write_text(json.dumps(manifest))
    runs=[];producer={'pid':123,'start_ticks':'456','cmdline_sha256':'c'*64}
    for shard in range(2):
        run=tmp_path/f'run{shard}';run.mkdir();runs.append(run)
        for name in runner.CODE:
            target=run/'code'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(runner.__file__).parents[1]/name,target)
        protocol={'schema':runner.VERSION,'thinking':False,'context_length':32768,'response_text_policy':tool.RAW_POLICY,
                  'model_type':'qwen3_vl','scope':'joint','shard':shard,'shards':2,'model':'fixture/model','adapter':None,
                  'export_manifest_sha256':tool.sha(export/'manifest.json'),
                  'code_sha256':{name:tool.sha(run/'code'/name) for name in runner.CODE},
                  'files':dict(runner.selected_files(manifest,'joint',shard,2))}
        for filename,value in [('protocol.json',protocol),('request.json',protocol),('process.json',producer),
                               ('status.json',{'stage':'complete_scope_shard'})]: (run/filename).write_text(json.dumps(value))
        run_sha=tool.sha(run/'protocol.json'); all_items=[]; predictions={}
        for form in native.FAMILIES:
            items=[]
            for ordinal in range(shard,3,2):
                row=rows[form][ordinal]; response=row['messages'][1]['content'] if ordinal!=1 else 'invalid fixture JSON'
                items.append(runner.prediction_item(row,ordinal,response,'stop',None,run_sha,sdk_response_prefix=''))
            path=run/f'predictions_{form}.jsonl';path.write_text(''.join(native.canonical(r)+'\n' for r in items),encoding='utf-8')
            predictions[form]=tool.sha(path);all_items.extend(items)
            (run/f'complete_{form}.json').write_text(json.dumps({'records':len(items),'prediction_sha256':tool.sha(path),
                'dataset_sha256':files[f'eval_{form}.jsonl']['sha256'],'protocol_sha256':run_sha}))
        final=tool.native_metrics.summarize(all_items)
        final.update(state='complete_scope_shard',run_protocol_sha256=run_sha,prediction_files=predictions)
        (run/'metrics.json').write_text(json.dumps(final))
    out=tmp_path/'merge';out.mkdir();request=tool.merge_request(runs,export)
    binding={'request':request,'producers':{str(run):producer for run in runs},'code_sha256':{}}
    (out/'merge_binding.json').write_text(json.dumps(binding))
    return tool,runs,export,out,binding


def test_merge_streams_complete_unequal_shards_without_averaging_percentages(native_merge_fixture):
    tool,runs,export,out,binding=native_merge_fixture
    result=tool.wait_for_merge(runs,export,out,binding,False)
    assert result['state']=='complete_scope' and result['scope_complete'] and result['full_benchmark_complete']
    assert result['n']==33 and len(result['groups'])==11 and len(result['terminal_shards'])==2
    assert all(g['n']==3 and g['mean']['json_valid']==pytest.approx(2/3) for g in result['groups'])
    assert result['global_composite_score'] is None and result['model_calls']==0 and not result['teacher_qualified']
    assert not list(out.rglob('predictions_*')) and not list(out.rglob('rescored_*'))
    partial=tool.merge_request(runs,export,['bbox']);binding['request']=partial
    subset=out/'partial';subset.mkdir();(subset/'merge_binding.json').write_text(json.dumps(binding))
    result=tool.wait_for_merge(runs,export,subset,binding,False)
    assert result['n']==3 and result['state']=='complete_selected_forms' and not result['scope_complete']


@pytest.mark.parametrize('fault',['missing','repeated','model','adapter','template','schema','code','export'])
def test_merge_requires_exact_model_protocol_and_complete_partition(native_merge_fixture,fault):
    tool,runs,export,out,binding=native_merge_fixture
    if fault=='missing':runs=runs[:1]
    elif fault=='repeated':runs=[runs[0],runs[0]]
    elif fault=='code':(runs[1]/'code/tools/run_native_evaluation.py').write_text('changed')
    elif fault=='export':(export/'manifest.json').write_text('{}')
    else:
        path=runs[1]/'protocol.json';protocol=json.loads(path.read_text())
        protocol[{'template':'response_text_policy'}.get(fault,fault)]='different'
        path.write_text(json.dumps(protocol))
    with pytest.raises((ValueError,KeyError)):tool.merge_request(runs,export)


@pytest.mark.parametrize('fault',['id','ordinal','metric','extra','missing','tail','receipt'])
def test_merge_rejects_changed_rows_even_with_a_rewritten_file_checksum(native_merge_fixture,fault):
    tool,runs,export,out,binding=native_merge_fixture
    path=runs[0]/'predictions_bbox.jsonl'; lines=path.read_text().splitlines();row=json.loads(lines[0])
    if fault=='id':row['id']='different'
    elif fault=='ordinal':row['ordinal']=1
    elif fault=='metric':row['metrics']['json_valid']=False
    lines[0]=native.canonical(row)
    if fault=='extra':lines.append(lines[0])
    if fault=='missing':lines=lines[:1]
    path.write_text('\n'.join(lines)+('' if fault=='tail' else '\n'),encoding='utf-8')
    receipt=runs[0]/'complete_bbox.json';value=json.loads(receipt.read_text())
    value['prediction_sha256']=tool.sha(path)
    if fault=='receipt':value['records']=9
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):tool.merge_form(binding['request'],export,'bbox')


def test_merge_wait_accepts_early_forms_then_waits_for_terminal_checks(native_merge_fixture,monkeypatch):
    tool,runs,export,out,binding=native_merge_fixture
    producer=binding['producers'][str(runs[0])]
    monkeypatch.setattr(tool.runner,'process_identity',lambda pid:producer)
    receipt=runs[1]/'complete_depth_order.json';old=receipt.read_bytes();receipt.unlink()
    terminal=runs[1]/'metrics.json';old_terminal=terminal.read_bytes();terminal.unlink()
    sleeps=[]
    def advance(seconds):
        assert seconds==20
        index=json.loads((out/'index.json').read_text());assert not index['scope_complete'] and not (out/'metrics.json').exists()
        sleeps.append(index['n'])
        if len(sleeps)==1:receipt.write_bytes(old)
        else:terminal.write_bytes(old_terminal)
    monkeypatch.setattr(tool.time,'sleep',advance)
    result=tool.wait_for_merge(runs,export,out,binding,True)
    assert sleeps==[30,33] and result['n']==33 and result['scope_complete']


def test_partial_merge_keeps_completed_source_when_its_other_forms_fail(native_merge_fixture,monkeypatch):
    tool,runs,export,out,binding=native_merge_fixture
    binding['request']=tool.merge_request(runs,export,['bbox'])
    (out/'merge_binding.json').write_text(json.dumps(binding))
    (runs[0]/'failure.json').write_text('{}')
    producer=binding['producers'][str(runs[1])]
    monkeypatch.setattr(tool.runner,'process_identity',lambda pid:producer)
    receipt=runs[1]/'complete_bbox.json';old=receipt.read_bytes();receipt.unlink()
    monkeypatch.setattr(tool.time,'sleep',lambda seconds:receipt.write_bytes(old))
    result=tool.wait_for_merge(runs,export,out,binding,True)
    assert result['n']==3 and not result['scope_complete'] and result['state']=='complete_selected_forms'


@pytest.mark.parametrize('fault',['pid','failure','request','protocol','export','code','accepted'])
def test_merge_wait_stops_without_gpu_restart_and_retains_accepted_forms(native_merge_fixture,monkeypatch,fault):
    tool,runs,export,out,binding=native_merge_fixture
    producer=binding['producers'][str(runs[0])]
    monkeypatch.setattr(tool.runner,'process_identity',lambda pid:producer)
    receipt=runs[1]/'complete_depth_order.json';old=receipt.read_bytes();receipt.unlink()
    def break_run(seconds):
        assert json.loads((out/'index.json').read_text())['n']==30
        if fault=='pid':monkeypatch.setattr(tool.runner,'process_identity',lambda pid:None)
        elif fault=='failure':(runs[1]/'failure.json').write_text('{}')
        elif fault in ('request','protocol'):(runs[1]/(fault+'.json')).write_text('{}')
        elif fault=='export':(export/'manifest.json').write_text('{}')
        elif fault=='code':binding['code_sha256']['tools/rescore_native_evaluation.py']='0'*64
        else:
            (out/'forms/bbox.json').write_text('{}');receipt.write_bytes(old)
    monkeypatch.setattr(tool.time,'sleep',break_run)
    with pytest.raises((ValueError,KeyError)):tool.wait_for_merge(runs,export,out,binding,True)
    assert (out/'index.json').is_file() and not (out/'metrics.json').exists()


def test_merge_cli_freezes_cpu_code_and_accepts_already_finished_producers(native_merge_fixture):
    import subprocess,sys
    from pathlib import Path
    tool,runs,export,out,binding=native_merge_fixture;target=out.parent/'cli_merge'
    before={str(p):tool.sha(p) for run in runs for p in run.glob('predictions_*.jsonl')}
    command=[sys.executable,'-B',str(Path(tool.__file__)),'merge','--export',str(export),'--output',str(target)]
    for run in runs:command.extend(['--run',str(run)])
    completed=subprocess.run(command,capture_output=True,text=True,timeout=45)
    assert completed.returncode==0,completed.stderr
    result=json.loads((target/'metrics.json').read_text());assert result['n']==33 and result['scope_complete']
    assert (target/'process.json').exists() and (target/'code/tools/rescore_native_evaluation.py').exists()
    assert before=={str(p):tool.sha(p) for run in runs for p in run.glob('predictions_*.jsonl')}


@pytest.fixture
def calibration_export(tmp_path):
    from argparse import Namespace
    from PIL import Image
    from tools import calibrate_native_teachers as tool
    from tools.arxiv_source_scale import fingerprint
    root = tmp_path/'train'; root.mkdir(); image = tmp_path/'image.png'
    Image.new('RGB', (8,6), 'red').save(image); fp = fingerprint(image)
    manifest = fixture_export_manifest(); manifest.update(records=60, selection_sha256='2'*64)
    for name, info in manifest['files'].items():
        rows = []; form = info['form']
        for i in range(5):
            row = fixture_qa() if form == 'QA' else evaluation_fixture_row(form)
            row['id'] = f'calibration:{form}:{i}'; row['images'] = [str(image)]
            row['anvil_meta'].update(split='train', training_approved=True,
                                    image_sha256=fp['sha256'], decoded_rgb_sha256=fp['decoded_rgb_sha256'])
            rows.append(native.bind_training_record(*serialize(row)))
        (root/name).write_text(''.join(native.canonical(row)+'\n' for row in rows), encoding='utf-8')
        info.update(records=5, sha256=tool.sha(root/name))
    (root/'manifest.json').write_text(json.dumps(manifest))
    return Namespace(export=root, output=tmp_path/'panel', per_group=2, seed=17)


def test_training_calibration_selection_complete_hashes_and_all_groups(calibration_export):
    from tools import calibrate_native_teachers as tool
    args = calibration_export; tool.select(args)
    manifest, rows = tool.load_panel(args.output)
    assert len(rows) == 22 and len(manifest['groups']) == 11
    assert manifest['complete_export_records'] == 60 and manifest['qa_records_excluded'] == 5
    assert not manifest['heldout'] and not manifest['teacher_qualified']
    assert all(g['population'] == 5 and g['selected'] == 2 for g in manifest['groups'])
    for g in manifest['groups']:
        candidates = [json.loads(s) for s in (args.export/f"train_{g['form']}.jsonl").read_text().splitlines()]
        expected = sorted(candidates, key=lambda r:tool.rank(r,17))[:2]
        assert [r for r in rows if r['gsa_form']==g['form']] == expected
    with pytest.raises(ValueError, match='new output'): tool.select(args)
    (args.output/'panel.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='bytes changed'): tool.load_panel(args.output)


@pytest.mark.parametrize('change', ['QA_hash', 'native_hash', 'benchmark', 'image'])
def test_training_calibration_rejects_tampering(calibration_export, change):
    from tools import calibrate_native_teachers as tool
    args = calibration_export
    if change in ('QA_hash', 'native_hash'):
        name = 'train_QA.jsonl' if change=='QA_hash' else 'train_bbox.jsonl'
        with (args.export/name).open('a') as stream: stream.write('{}\n')
    elif change == 'benchmark':
        path=args.export/'train_action_cot.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]['gsa_meta']['binding']['split']='benchmark'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    else:
        (args.export.parent/'image.png').write_bytes(b'changed')
    with pytest.raises(ValueError): tool.select(args)
    assert not (args.output/'manifest.json').exists()


def test_training_calibration_next_token_nll_mask_and_chunks():
    import torch
    from tools import calibrate_native_teachers as tool
    positions, target = tool.response_positions(torch.tensor([[-100,-100,2,1,0]]),torch.tensor([[1,1,1,1,0]]))
    assert positions.tolist()==[1,2] and target.tolist()==[2,1]
    logits=torch.tensor([[[0.,0.,3.],[0.,3.,0.]]])
    value=tool.token_scores(logits,target)
    assert value['tokens']==value['correct_tokens']==2
    assert value['nll_sum']==pytest.approx(float(torch.nn.functional.cross_entropy(logits[0],target,reduction='sum')))
    logits=logits.repeat(1,129,1);target=target.repeat(129)
    assert tool.token_scores(logits,target)['nll_sum']==pytest.approx(value['nll_sum']*129,rel=1e-5)
    with pytest.raises(ValueError,match='misaligned'):tool.token_scores(logits[:,:1],target)
    with pytest.raises(ValueError,match='No supervised'):tool.response_positions(torch.tensor([[-100,-100]]),torch.ones(1,2))


def test_training_calibration_paired_binding_and_no_performance_promotion():
    from tools import calibrate_native_teachers as tool
    row=native.bind_training_record(*serialize(fixture_record()))
    item={'ordinal':0,'row_sha256':native.digest(row),'protocol_sha256':'a'*64,'group':list(tool.group(row)),
          'expert':{'nll_sum':1.,'tokens':4,'correct_tokens':3},'joint':{'nll_sum':2.,'tokens':4,'correct_tokens':2}}
    tool.check_score(item,row,0,'a'*64)
    report=tool.summarize([item])
    assert report['families']['G']['equal_source_form_mean_nll_difference']==-.25
    assert report['families']['G']['training_reference_nll_improved']
    assert not report['teacher_qualified'] and not report['heldout_accuracy']
    assert report['verdict'] not in ('PASS','APPROVED','ELIGIBLE')
    wrong=deepcopy(item);wrong['expert']['tokens']=3
    with pytest.raises(ValueError,match='lengths differ'):tool.check_score(wrong,row,0,'a'*64)
    with pytest.raises(ValueError,match='binding mismatch'):tool.check_score(item,row,1,'a'*64)
