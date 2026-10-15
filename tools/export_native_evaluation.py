#!/usr/bin/env python3
"""Export every frozen native benchmark record without introducing a new split.

ALFRED's raw native interactions use their already source-verified SFT rendering;
the raw record hash, source binding, unchanged image and selected UID must agree.
Other native references remain unchanged. These local-path exports are not the
portable anonymous dataset release and do not establish a model's performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import export_native_training as training_export
from tools import map_alfred_native as alfred
from tools.arxiv_source_scale import sha
from tools.build_anvil_pool import write_json
from trains.flywheel import native_training as native

VERSION = 'anvil-source-native-evaluation-v1'
require = native.require


def load_alfred_rendering(root):
    manifest_path = root / 'manifest.json'; records_path = root / 'evaluation.jsonl'
    manifest = native.strict_json(manifest_path.read_bytes())
    require(manifest.get('state') == 'source_native_expert_continuation_sft_exported'
            and manifest.get('contract') == 'anvil-alfred-expert-continuation-sft-v1', 'Unsupported ALFRED rendering')
    require(sha(records_path) == manifest['files']['evaluation.jsonl'], 'ALFRED rendering changed')
    binding = manifest['source_binding']['evaluation']
    require(sha(binding['records_path']) == binding['records_sha256']
            and sha(binding['manifest_path']) == binding['manifest_sha256'], 'ALFRED raw binding changed')
    rows = {}
    with records_path.open('rb') as stream:
        for raw in stream:
            row = native.strict_json(raw)
            require(row['id'] not in rows, 'Duplicate ALFRED rendered identity')
            require(row['anvil_meta']['source_binding'] == binding, 'ALFRED row binding mismatch')
            rows[row['id']] = row
    require(len(rows) == manifest['counts']['evaluation']['records'], 'ALFRED rendering count mismatch')
    return rows, {'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
                  'records_path': str(records_path), 'records_sha256': sha(records_path),
                  'raw_binding': binding}


@lru_cache(maxsize=50000)
def image_size(path):
    with Image.open(path) as image: return list(image.size)


def convert_selected(raw, entry, selection_sha, alfred_rows, action_prompt="source"):
    require(action_prompt in alfred.SFT_PROMPT_POLICIES, 'Unknown ALFRED prompt policy')
    require(entry['split'] == 'benchmark', 'Evaluation cannot admit a selected training row')
    original = native.strict_json(raw); row = original
    require(isinstance(original, dict) and original.get('id') == entry['uid'], 'Selected source identity mismatch')
    mapping = None
    if entry['source'] == 'ALFRED':
        require(entry['form'] == 'action_cot' and entry['variant'] == original.get('schema_version')
                and original.get('training_approved') is False, 'Unexpected ALFRED selected variant/scope')
        row = alfred_rows.get(original['id'])
        require(row is not None, 'Selected ALFRED interaction lacks its verified rendering')
        meta = row['anvil_meta']
        canonical_source_sha = hashlib.sha256(json.dumps(original, sort_keys=True, ensure_ascii=False,
                                                        allow_nan=False).encode()).hexdigest()
        require(meta.get('source_record_id') == original['id']
                and meta.get('source_record_canonical_sha256') == canonical_source_sha
                and meta['source_binding']['records_sha256'] == entry['source_sha256']
                and meta.get('source_native_contract') == entry['variant'], 'ALFRED rendering does not bind this raw record')
        require(row['images'] == [original['input']['image']]
                and meta['image_sha256'] == original['input']['image_sha256'], 'ALFRED rendering image changed')
        mapping = {'method': 'existing_source_verified_expert_continuation_rendering',
                   'raw_record_canonical_sha256': canonical_source_sha, 'rendered_record_sha256': native.digest(row)}
    meta = row.get('anvil_meta', {})
    require(isinstance(meta, dict), 'Source metadata absent')
    form, source, group = meta.get('native_form'), meta.get('source'), meta.get('group_id')
    variant = meta.get('native_contract', meta.get('native_variant', meta.get('schema')))
    require(source == entry['source'] and form == entry['form'] and form in native.FAMILIES,
            'Evaluation source/form mismatch; QA is training-only')
    require(mapping is not None or variant == entry['variant'], 'Evaluation variant mismatch')
    if source in native.DENSE:
        require(meta.get('split') in {'train', 'evaluation_candidate'}, 'Unknown dense candidate origin')
        selected_group = 'decoded_image:' + str(group)
    else:
        require(meta.get('split') in {'validation', 'evaluation', 'test', 'heldout', 'valid_seen', 'valid_unseen'}
                and meta.get('training_approved') is False, 'Frozen native training cannot become evaluation')
        selected_group = group if str(group).startswith(source + ':') else source + ':' + str(group)
    require(selected_group == entry['scene_group'], 'Selected evaluation group mismatch')
    images = row.get('images')
    require(isinstance(images, list) and len(images) == 1 and isinstance(images[0], str), 'One source image required')
    actual_image_sha = training_export.checked_image(images[0], entry['image_path'], entry['file_sha256'], entry['decoded_sha256'])
    source_image_sha = (meta.get('image_sha256') or meta.get('input_image_sha256')
                        or meta.get('source_provenance', {}).get('image_sha256'))
    require(source_image_sha is None or source_image_sha == actual_image_sha, 'Source image bytes disagree')
    require(meta.get('decoded_rgb_sha256', entry['decoded_sha256']) == entry['decoded_sha256'], 'Source decoded identity differs')
    messages = row.get('messages')
    require(isinstance(messages, list) and len(messages) == 2 and all(isinstance(m, dict) for m in messages)
            and [m.get('role') for m in messages] == ['user', 'assistant']
            and all(isinstance(m.get('content'), str) and m['content'].strip() for m in messages), 'Native prompt/reference pair required')
    native.validate_target(form, variant, native.strict_json(messages[1]['content']), messages[0]['content'])
    result = {'schema': VERSION, 'id': original['id'], 'native_form': form, 'native_variant': variant,
        'family': native.FAMILIES[form], 'messages': deepcopy(messages), 'images': deepcopy(images),
        'image_size_wh': image_size(images[0]), 'training_approved': False,
        'evaluation_meta': {'split': 'benchmark', 'selection_sha256': selection_sha, 'source': source,
            'source_sha256': entry['source_sha256'], 'source_line_sha256': hashlib.sha256(raw).hexdigest(),
            'source_offset': entry['byte_offset'], 'source_size': entry['byte_length'],
            'selected_variant': entry['variant'], 'selected_group_id': entry['scene_group'], 'source_group_id': group,
            'source_metadata_sha256': native.digest(meta), 'image_sha256': actual_image_sha,
            'decoded_rgb_sha256': entry['decoded_sha256'], 'source_mapping': mapping,
            'public_release_eligible': False, 'model_evaluation_performed': False}}
    if action_prompt != 'source' and form == 'action_cot':
        require(source == 'ALFRED' and variant == alfred.SFT_CONTRACT,
                'Explicit Action prompt only supports the source ALFRED contract')
        original_prompt = messages[0]['content']
        result['messages'][0]['content'] = alfred.render_sft_prompt(original_prompt, action_prompt)
        result['evaluation_meta']['prompt_rendering'] = {
            'policy': action_prompt,
            'source_prompt_sha256': hashlib.sha256(original_prompt.encode()).hexdigest(),
            'output_schema_sha256': native.digest(alfred.SFT_OUTPUT_SCHEMA),
            'reference_changed': False, 'goal_changed': False,
        }
    result['evaluation_meta']['evaluation_content_sha256'] = native.digest({k: v for k, v in result.items() if k != 'evaluation_meta'})
    return form, native.canonical(result) + '\n'


def export(args):
    start = time.monotonic()
    action_prompt = getattr(args, 'action_prompt', 'source')
    require(action_prompt in alfred.SFT_PROMPT_POLICIES, 'Unknown ALFRED prompt policy')
    code_paths = [Path(__file__), Path(training_export.__file__), Path(native.__file__), Path(alfred.__file__)]
    code_hashes = {str(path): sha(path) for path in code_paths}
    require(not args.output.exists(), 'Use a new immutable evaluation export directory')
    manifest_sha = sha(args.manifest); manifest = native.strict_json(args.manifest.read_bytes())
    require(manifest.get('selection_complete') is True, 'Incomplete frozen pool selection')
    selection_sha = sha(args.index)
    require(selection_sha == manifest['selection_sqlite_sha256'], 'Frozen pool SHA mismatch')
    for item in manifest['inputs']:
        require(sha(item['path']) == item['expected_sha256'] and sha(item['manifest']) == item['manifest_sha256'], 'Bound source changed')
    alfred_rows, alfred_binding = load_alfred_rendering(args.alfred_sft)
    db = sqlite3.connect(':memory:'); db.execute('PRAGMA temp_store=MEMORY')
    source = sqlite3.connect(f'{args.index.resolve().as_uri()}?mode=ro', uri=True)
    source.backup(db, pages=4096); source.close()
    counts = dict(db.execute("SELECT form,count(*) FROM selection WHERE split='benchmark' GROUP BY form"))
    require(set(counts) == set(native.FAMILIES) and sum(counts.values()) == manifest['observed']['rows_by_split']['benchmark'],
            'Incomplete native benchmark coverage or counts')
    images = db.execute("SELECT count(DISTINCT image_id) FROM selection WHERE split='benchmark'").fetchone()[0]
    require(db.execute("SELECT count(*) FROM images WHERE image_id IN (SELECT image_id FROM selection WHERE split='train') "
                       "AND image_id IN (SELECT image_id FROM selection WHERE split='benchmark')").fetchone()[0] == 0,
            'Selected training/evaluation image overlap')
    args.output.mkdir(parents=True, exist_ok=False); written = Counter()
    def progress(stage):
        value = {'stage': stage, 'seconds': time.monotonic() - start, 'records': sum(written.values()), 'by_form': dict(written)}
        write_json(args.output / 'progress.json', value); print(json.dumps(value), flush=True)
    progress('source_bound')
    try:
        with ExitStack() as stack:
            handles = {form: stack.enter_context((args.output / f'eval_{form}.jsonl.pending').open('x')) for form in counts}
            for _, batch in training_export.source_batches(db, selection_sha, split='benchmark'):
                previous = sum(written.values())
                for raw, entry in batch:
                    try: form, line = convert_selected(raw, entry, selection_sha, alfred_rows, action_prompt)
                    except Exception as error: raise ValueError(f"Selected evaluation row {entry['uid']} ({entry['form']}): {error}") from error
                    handles[form].write(line); written[form] += 1
                if sum(written.values()) // 10000 != previous // 10000: progress('exporting')
        require(dict(written) == counts, 'Selected evaluation records omitted or duplicated')
        require(sha(args.index) == selection_sha and sha(args.manifest) == manifest_sha, 'Frozen selection changed during export')
        for item in manifest['inputs']:
            require(sha(item['path']) == item['expected_sha256'] and sha(item['manifest']) == item['manifest_sha256'], 'Source changed during export')
        for prefix in ('manifest', 'records'):
            require(sha(alfred_binding[prefix + '_path']) == alfred_binding[prefix + '_sha256'], 'ALFRED rendering changed during export')
        require(all(sha(path) == digest for path, digest in code_hashes.items()), 'Exporter code changed during export')
        files = {}
        for form, count in sorted(written.items()):
            name = f'eval_{form}.jsonl'; (args.output / (name + '.pending')).rename(args.output / name)
            files[name] = {'records': count, 'sha256': sha(args.output / name), 'form': form, 'family': native.FAMILIES[form]}
        result = {'schema': VERSION, 'state': 'all_selected_native_evaluation_rows_exported', 'records': sum(written.values()),
            'unique_images': images, 'by_form': dict(written), 'files': files, 'dropped_records': 0, 'targets_repaired': 0,
            'selection_sha256': selection_sha, 'pool_manifest_sha256': manifest_sha, 'source_inputs': manifest['inputs'],
            'alfred_rendering': alfred_binding, 'action_prompt': action_prompt,
            'code_sha256': code_hashes, 'seconds': time.monotonic() - start,
            'model_evaluation_performed': False, 'public_release_eligible': False,
            'scope': 'Complete local benchmark input/reference rendering; not final scores, physical execution or a public data release.'}
        write_json(args.output / 'manifest.json', result); progress('complete'); return result
    except BaseException as error:
        write_json(args.output / 'failure.json', {'state': 'failed_no_complete_evaluation_export', 'type': type(error).__name__,
                   'reason': str(error), 'written_records': sum(written.values())})
        raise
    finally: db.close()


def check_reference_form(root, name, info, manifest):
    """Exercise the production scorer with unchanged references, never a model."""
    from eval import native_metrics
    from tools.run_native_evaluation import validate_row
    path = root / name
    require(sha(path) == info['sha256'], 'Reference file changed: ' + name)
    regions = None
    if info['form'] == 'point' and any(i['source'] == 'coco_train2017' for i in manifest['source_inputs']):
        regions = native_metrics.CocoPointReference(manifest)
    identities = set(); groups = {}; issues = Counter(); examples = []
    tolerances = {'rotation_error_degrees': 1e-4, 'pi_symmetric_rotation_error_degrees': 1e-4,
                  'normalized_projected_contact_l2': 1e-8, 'normalized_image_distance_mae': 1e-12}
    try:
        with path.open('rb') as stream:
            for raw in stream:
                require(raw.endswith(b'\n'), 'Incomplete reference record')
                row = native.strict_json(raw); validate_row(row, info['form'], manifest['selection_sha256'])
                require(row['id'] not in identities, 'Duplicate reference identity')
                identities.add(row['id'])
                region = regions(row) if regions and row['evaluation_meta']['source'] == 'coco_train2017' else None
                values = native_metrics.score_row(row, row['messages'][1]['content'], point_region=region)
                source = row['evaluation_meta']['source']
                group = groups.setdefault(source, {'n': 0, 'sums': Counter(), 'nonideal_records': 0})
                group['n'] += 1; group['sums'].update(values)
                bad = []
                for key, value in values.items():
                    require(type(value) in (float, int, bool) and math.isfinite(value), 'Nonfinite reference metric')
                    target = 0. if key in tolerances or key == 'truncated' else 1.
                    if abs(value - target) > tolerances.get(key, 1e-12):
                        issues[key] += 1; bad.append(key)
                if bad:
                    group['nonideal_records'] += 1
                    if len(examples) < 20: examples.append({'id': row['id'], 'metrics': {k: values[k] for k in bad}})
        require(len(identities) == info['records'] and sha(path) == info['sha256'],
                'Reference count/hash changed during scoring')
    finally:
        if regions: regions.close()
    return {'form': info['form'], 'records': len(identities), 'dataset_sha256': info['sha256'],
        'metric_issues': dict(issues), 'examples': examples,
        'groups': [{'source': source, 'n': group['n'], 'nonideal_records': group['nonideal_records'],
                    'mean': {key: value / group['n'] for key, value in sorted(group['sums'].items())}}
                   for source, group in sorted(groups.items())]}, identities


def check_references(args):
    """Full reference/scorer compatibility audit; not a semantic-quality rate."""
    from eval import native_metrics
    from tools import run_native_evaluation as runner
    from tools import map_prism_native
    require(1 <= args.workers <= 8, 'Use between one and eight CPU workers')
    require(not args.output.exists(), 'Use a new reference-check output directory')
    manifest_path = args.export / 'manifest.json'; manifest_sha = sha(manifest_path)
    manifest = native.strict_json(manifest_path.read_bytes())
    files = runner.selected_files(manifest, 'joint', 0, 1)
    runner.verify_metric_dependencies(files)
    code = {str(Path(p).resolve()): sha(p) for p in
            (Path(__file__), native_metrics.__file__, native.__file__, runner.__file__, map_prism_native.__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    request = {'export': str(args.export.resolve()), 'manifest_sha256': manifest_sha,
               'workers': args.workers, 'code_sha256': code, 'metrics_version': native_metrics.VERSION,
               'expected_records': manifest['records'], 'process': runner.process_identity(os.getpid()),
               'model_calls': 0, 'purpose': 'reference/scorer self-consistency only'}
    write_json(args.output / 'request.json', request)
    completed = {}; seen = set(); start = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(check_reference_form, args.export, name, info, manifest) for name, info in files]
            for future in as_completed(futures):
                result, identities = future.result()
                require(not seen.intersection(identities), 'Duplicate reference identity across forms')
                seen.update(identities); completed[result['form']] = result
                write_json(args.output / (result['form'] + '.json'), result)
                progress = {'completed_forms': sorted(completed), 'records': len(seen), 'seconds': time.monotonic() - start}
                write_json(args.output / 'progress.json', progress); print(json.dumps(progress), flush=True)
        require(len(seen) == manifest['records'], 'Reference audit did not cover the whole export')
        require(sha(manifest_path) == manifest_sha and all(sha(args.export / n) == i['sha256'] for n, i in files),
                'Reference export changed before completion')
        require(all(sha(p) == value for p, value in code.items()), 'Reference-check code changed')
        result = {'schema': 'anvil-native-reference-self-check-v1', 'state': 'complete_reference_check',
            'records': len(seen), 'forms': completed, 'export_manifest_sha256': manifest_sha,
            'all_references_score_ideally': all(not item['metric_issues'] for item in completed.values()),
            'model_calls': 0, 'semantic_quality_established': False, 'teacher_qualified': False,
            'model_evaluation_complete': False, 'references_changed': False, 'seconds': time.monotonic() - start}
        write_json(args.output / 'report.json', result); return result
    except BaseException as error:
        write_json(args.output / 'failure.json', {'type': type(error).__name__, 'reason': str(error),
                   'completed_forms': sorted(completed), 'model_evaluation_complete': False})
        raise


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'check-references':
        parser = argparse.ArgumentParser(description=check_references.__doc__)
        parser.add_argument('--export', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--workers', type=int, default=4)
        result = check_references(parser.parse_args(sys.argv[2:]))
        raise SystemExit(0 if result['all_references_score_ideally'] else 2)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'manifest', 'alfred-sft', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--action-prompt', choices=alfred.SFT_PROMPT_POLICIES, default='source',
                        help='Opt-in schema clarification; use a new export and matched runs, never resume old predictions')
    export(parser.parse_args())


if __name__ == '__main__': main()
