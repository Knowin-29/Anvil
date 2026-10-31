"""Audit the delivered student's actual ancestry and its separate teacher lineage.

Historical warm starts lack per-row logs. Their unknown coverage is disclosed,
never filled in from the available pool or from unrelated experimental arms.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from common import R, bound, check_binding, formal_training, load_spec, read, require, save, sha
from exposure_audit import audit
from initialization_lineage import trace


def unique_evidence(items):
    result = {}
    for item in items:
        previous = result.setdefault(item['path'], item)
        require(previous == item, 'Conflicting evidence hashes: ' + item['path'])
    return list(result.values())


def logged_totals(stages, original_selection):
    counts, uids, images = Counter(), defaultdict(set), defaultdict(set)
    runs = set()
    for stage in stages:
        run = Path(stage['run'])
        require(run not in runs and not stage['execution_only'], 'Repeated or execution-only ancestry stage')
        runs.add(run)
        protocol = read(run/'protocol.json')
        world = protocol.get('world_size', protocol.get('world'))
        for rank in range(world):
            for line in (run/f'exposure_rank{rank}.jsonl').open():
                row = json.loads(line)
                binding = row['source_binding']
                source = binding.get('source_dataset', binding.get('source'))
                original = binding.get('selection_sha256') == original_selection
                key = ('original_replay:' if original else 'supplemental:') + source
                counts[key] += 1
                uids[key].add(row['canonical_uid'])
                images[key].add(binding['decoded_rgb_sha256'])
    supplemental = set().union(*(ids for key, ids in uids.items() if key.startswith('supplemental:')))
    return dict(
        exposures=sum(counts.values()), unique_supplemental_records=len(supplemental),
        groups={key: dict(exposures=n, unique_records=len(uids[key]), unique_images=len(images[key]),
                         repetitions_within_logged_stages=n-len(uids[key]),
                         unique_ids_sha256=hashlib.sha256('\n'.join(sorted(uids[key])).encode()).hexdigest())
                for key, n in sorted(counts.items())},
        historical_unlogged_warm_start_included=False,
        original_replays_are_additional_unique_training_records=False)


def teacher_trace(family, checkpoint):
    require(family in 'GSA' and len(family) == 1, 'Unknown teacher family')
    checkpoint = Path(checkpoint).resolve()
    tip, stages, evidence, seen = checkpoint, [], [], set()
    while (checkpoint.parent/'protocol.json').is_file():
        run = checkpoint.parent
        require(run not in seen, 'Teacher ancestry cycle')
        seen.add(run)
        protocol, complete = read(run/'protocol.json'), read(run/'complete.json')
        require(not protocol['smoke'] and not complete['smoke'], 'Execution-only teacher ancestor')
        require(Path(complete['checkpoint']).resolve() == checkpoint, 'Non-final repair ancestor')
        require(sha(checkpoint/'adapter_model.safetensors') == complete['adapter_sha256'], 'Changed teacher adapter')
        stage = audit(run)
        stages.append(stage)
        evidence.extend(stage['evidence'])
        evidence.append(bound(checkpoint/'adapter_model.safetensors'))
        initial = protocol['initial']
        checkpoint = Path(initial['checkpoint'] if isinstance(initial, dict) else initial).resolve()
        if isinstance(initial, dict):
            require(sha(checkpoint/'adapter_model.safetensors') == initial['adapter_sha256'], 'Changed teacher initialization')
    root = R.parent/'a800_repair_20261020'/('repair_' + family)
    complete, protocol = read(root/'complete.json'), read(root/'training_protocol.json')
    require(Path(complete['checkpoint']).resolve() == checkpoint, 'Teacher does not trace to its specialist SFT')
    require(sha(checkpoint/'adapter_model.safetensors') == complete['adapter_sha256'], 'Changed specialist SFT')
    state = read(checkpoint/'trainer_state.json')
    require(complete['epoch'] == state['epoch'] == 1.0 and not complete['smoke'] and not protocol['smoke'],
            'Specialist SFT did not complete its declared epoch')
    require(complete['global_step'] == state['global_step'] and protocol['scope'] == family, 'Specialist scope mismatch')
    export = R.parent/'arxiv/native_training_export_20261012_v4_r2'
    manifest = read(export/'manifest.json')
    expected = {str(export/name): info for name, info in manifest['files'].items() if info['family'] == family}
    require(set(protocol['data']) == set(expected), 'Specialist source files differ')
    argv = protocol['argv']
    warm = Path(argv[argv.index('--adapters')+1]).resolve()
    warm_root = warm.parent.parent
    warm_protocol, warm_state = read(warm_root/'protocol.json'), read(warm/'trainer_state.json')
    require(warm_state['global_step'] == warm_state['max_steps'] == 1024, 'Incomplete specialist warm start')
    require(warm_protocol['scope'] == family and warm_protocol['export_manifest_sha256'] == sha(export/'manifest.json'),
            'Specialist warm-start source mismatch')
    require(read(warm_root/'training.result.json')['exit_code'] == 0, 'Failed specialist warm start')
    evidence.extend(bound(p) for p in [root/'complete.json', root/'training_protocol.json',
        checkpoint/'trainer_state.json', checkpoint/'adapter_model.safetensors', export/'manifest.json',
        warm_root/'protocol.json', warm_root/'training.command.json', warm_root/'training.result.json',
        warm/'trainer_state.json', warm/'adapter_model.safetensors'])
    stages.reverse()
    return dict(family=family, checkpoint=str(tip), adapter_sha256=sha(tip/'adapter_model.safetensors'),
        warm_start=dict(checkpoint=str(warm), updates=1024, per_row_exposure_log_available=False,
                        exact_unique_records=None),
        specialist_sft=dict(checkpoint=str(checkpoint), epoch=1.0, updates=state['global_step'],
            available_training_records=sum(info['records'] for info in expected.values()),
            source_manifest_sha256=sha(export/'manifest.json'),
            coverage_evidence='Completed map-style specialist epoch; historical per-row exposure logs unavailable.'),
        post_sft_stages=stages, post_sft_actual=logged_totals(stages, manifest['selection_sha256']),
        counted_as_direct_student_exposure=False, evidence=unique_evidence(evidence))


def build(spec_path, eligibility_path):
    spec_path, eligibility_path = Path(spec_path), Path(eligibility_path)
    spec = load_spec(spec_path, verify_model=False)
    complete, protocol, checkpoint, _ = formal_training(spec['training_run'])
    pipeline = Path(spec['training_run']).parent.parent
    pipeline_spec = read(pipeline/'spec.json')
    require(eligibility_path.resolve() == pipeline/'spec.json', 'Use the continuation pipeline specification')
    require(sha(eligibility_path) == protocol['spec_sha256'], 'Pipeline specification changed')
    parent_binding = pipeline_spec['continuation_parent']['lineage']
    check_binding(parent_binding)
    parent = read(parent_binding['path'])
    require(parent['schema'] == 'anvil-final-lineage-audit-v1' and parent['state'] == 'complete',
            'Incomplete parent training lineage')
    require(parent['checkpoint'] == pipeline_spec['initial']['adapter'] and
            parent['adapter_sha256'] == pipeline_spec['initial']['adapter_sha256'],
            'The corrective stage must inherit the completed parent MOPD weights')
    for item in parent['evidence']:
        check_binding(item)
    repair = audit(pipeline/'runs/repair')
    stage = audit(spec['training_run'])
    repair_protocol = read(pipeline/'runs/repair/protocol.json')
    require(repair_protocol['initial'] == parent['checkpoint'], 'Corrective ancestry differs')
    require(repair['checkpoint'] == protocol['initial'], 'MOPD ancestry differs')
    export = R.parent/'arxiv/native_training_export_20261012_v4_r2'
    manifest = read(export/'manifest.json')
    teachers = parent['teachers']
    require(set(teachers) == set(protocol['teachers']) == set('GSA'), 'Incomplete teacher lineage')
    for family, adapter in protocol['teachers'].items():
        item = pipeline_spec['teachers'][family]
        require(Path(adapter).resolve() == Path(teachers[family]['checkpoint']).resolve() and
                item['adapter_sha256'] == teachers[family]['adapter_sha256'] ==
                sha(Path(adapter)/'adapter_model.safetensors'), 'Different continuation teacher')
    stages = parent['post_sft_stages'] + [repair, stage]
    totals = logged_totals(stages, manifest['selection_sha256'])
    evidence = parent['evidence'] + repair['evidence'] + stage['evidence'] + [
        bound(spec_path), bound(eligibility_path), parent_binding,
        bound(checkpoint/'adapter_model.safetensors'), bound(Path(__file__))]
    return dict(schema='anvil-final-lineage-audit-v1', state='complete', final_spec_sha256=sha(spec_path),
        checkpoint=str(checkpoint), adapter_sha256=sha(checkpoint/'adapter_model.safetensors'),
        warm_start=parent['warm_start'], initial_sft=parent['initial_sft'],
        recovery=parent.get('recovery'), continuation_parent=pipeline_spec['continuation_parent'],
        optimizer_transitions=[dict(stage='targeted_repair', initial_checkpoint=parent['checkpoint'],
            model_weights_inherited=True, optimizer_state_inherited=False,
            reason='New corrective optimization stage, not resumption of an interrupted update.'),
            dict(stage='targeted_mopd', initial_checkpoint=repair['checkpoint'],
                 model_weights_inherited=True, optimizer_state_inherited=False,
                 reason='New on-policy distillation stage with its declared schedule.')],
        post_sft_stages=stages, post_sft_actual=totals, mopd=stage, teachers=teachers,
        available_inventory=dict(initial=parent['available_inventory']['initial'],
            direct_stage_datasets=pipeline_spec['datasets']), inventory_is_actual_exposure=False,
        original_sft_records_plus_unique_logged_supplemental_records=5220000+totals['unique_supplemental_records'],
        exact_total_exposures_including_historical_warm_start=None,
        other_experiment_arms_counted_as_final_training=False, teachers_training_counted_as_student_exposure=False,
        limitations=parent['limitations'] + [
            'Native form balancing intentionally changes sampling proportions; the original inventory is unchanged.',
            'The parent candidate failed the public benchmark review. Its weights are an ancestor, not an accepted release.'],
        evidence=unique_evidence(evidence))


def main():
    ap = argparse.ArgumentParser()
    for name in ['spec', 'eligibility', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), 'Use a new immutable final lineage report')
    report = build(args.spec, args.eligibility)
    save(args.output, report)
    print(json.dumps(dict(state=report['state'], checkpoint=report['checkpoint'],
                         post_sft_actual=report['post_sft_actual']), indent=2))


if __name__ == '__main__':
    main()
