"""Seal exactly the committed exposure prefix of a formal checkpoint.

This is provenance evidence only. It does not mark the planned run completed,
choose a delivery checkpoint, change final admission, or accept model quality.
Later training rows may be appended without changing the chosen prefix.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

C = Path(__file__).resolve().parent
sys.path.insert(0, str(C / 'formal_mopd'))
from formal_common import bound, check_binding, load_plan, read, require, save, sealed_checkpoint, sha


def prefix_rows(path, steps, rank, accumulation):
    """Read exact ordered committed rows; never consume an in-progress suffix."""
    raw_rows = []
    with Path(path).open('rb') as stream:
        for step in range(1, steps + 1):
            for micro in range(accumulation):
                raw = stream.readline()
                require(raw.endswith(b'\n'), 'Missing or partial committed exposure row')
                row = json.loads(raw)
                require((row['step'], row['micro'], row['rank']) == (step, micro, rank),
                        'Duplicated, reordered or missing committed exposure')
                require(row['smoke'] is False, 'Execution-only row in formal prefix')
                raw_rows.append((raw, row))
    return raw_rows


def summarize(rows):
    counts = Counter(); categories = Counter(); forms = Counter()
    uids = defaultdict(set); images = defaultdict(set)
    for row in rows:
        binding = row['source_binding']
        source = binding.get('source_dataset', binding.get('source'))
        category = row['source_category']
        require(source and category in {'native_distillation', 'general_qa', 'spatial_reference'},
                'Missing source identity/category')
        require(binding.get('decoded_rgb_sha256'), 'Missing decoded image identity')
        if category == 'native_distillation':
            require(row['rollout_tokens'] > 0 and row['teacher_scored_positions'] > 0,
                    'Formal native row lacks real teacher distillation')
        else:
            require(row['rollout_tokens'] == row['teacher_scored_positions'] == 0,
                    'Reference-only replay unexpectedly distilled')
        key = category + ':' + source
        counts[key] += 1; categories[category] += 1; forms[row['form']] += 1
        uids[key].add(row['canonical_uid']); images[key].add(binding['decoded_rgb_sha256'])
    return dict(exposures=sum(counts.values()), source_exposures=dict(categories), by_form=dict(forms),
        unique_records=len(set().union(*uids.values())), unique_images=len(set().union(*images.values())),
        groups={key:dict(exposures=n, unique_records=len(uids[key]), repeated_exposures=n-len(uids[key]),
                         unique_images=len(images[key]),
                         unique_ids_sha256=hashlib.sha256('\n'.join(sorted(uids[key])).encode()).hexdigest())
                for key,n in sorted(counts.items())})


def validate(path, check_current_prefix=True):
    path = Path(path).resolve(); report = read(path)
    require(report['schema'] == 'anvil-formal-checkpoint-prefix-audit-v1'
            and report['final_quality_accepted'] is False and report['selected_for_delivery'] is False,
            'Prefix provenance is not quality admission')
    for item in report['evidence'].values():
        check_binding(item)
    state, protocol, receipt = sealed_checkpoint(report['checkpoint'], full=False)
    require(state['global_step'] == report['selected_step'] and protocol['updates'] == report['planned_steps'],
            'Checkpoint training endpoint changed')
    require(report['adapter_sha256'] == receipt['files']['adapter_model.safetensors']['sha256'], 'Changed checkpoint')
    require(len(report['rank_prefixes']) == protocol['world_size'], 'Incomplete rank prefixes')
    rows = []; seen = set()
    for item in report['rank_prefixes']:
        rank = item['rank']; require(rank not in seen and 0 <= rank < protocol['world_size'], 'Duplicate/wrong rank')
        seen.add(rank); prefix = Path(item['snapshot'])
        require(prefix.parent == path.parent / 'exposures', 'Prefix snapshot outside bundle')
        require(prefix.stat().st_size == item['bytes'] and sha(prefix) == item['sha256'], 'Changed frozen prefix')
        parsed = prefix_rows(prefix, report['selected_step'], rank, protocol['global_batch'] // protocol['world_size'])
        require(sum(len(raw) for raw, row in parsed) == item['bytes'], 'Unexpected prefix suffix')
        rows.extend(row for raw,row in parsed)
        if check_current_prefix:
            with Path(item['original_source']).open('rb') as stream:
                current = stream.read(item['bytes'])
            require(hashlib.sha256(current).hexdigest() == item['sha256'], 'Original committed prefix changed')
    actual = summarize(rows)
    require(actual == report['actual'], 'Prefix totals changed')
    require(actual['source_exposures'] == state['source_exposures']
            and actual['exposures'] == report['selected_step'] * protocol['global_batch'], 'Checkpoint exposure mismatch')
    return report


def build(plan_path, step, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = load_plan(plan_path); run = Path(plan['training_run'])
    require(step in plan['evaluation_steps'], 'Use a declared evaluated checkpoint endpoint')
    checkpoint = run / ('checkpoint-%d' % step)
    state, protocol, receipt = sealed_checkpoint(checkpoint)
    require(state['global_step'] == step and protocol['updates'] == plan['steps'], 'Wrong formal plan')
    require(protocol['eligibility_sha256'] == sha(plan['eligibility']), 'Changed teacher evidence')
    require(not output.exists(), 'Use a new immutable prefix audit directory')
    (output / 'exposures').mkdir(parents=True)
    rows = []; prefixes = []
    for rank in range(protocol['world_size']):
        original = run / ('exposure_rank%d.jsonl' % rank)
        parsed = prefix_rows(original, step, rank, protocol['global_batch'] // protocol['world_size'])
        snapshot = output / 'exposures' / original.name
        raw = b''.join(line for line,row in parsed); snapshot.write_bytes(raw)
        rows.extend(row for line,row in parsed)
        prefixes.append(dict(rank=rank, original_source=str(original), snapshot=str(snapshot),
            bytes=len(raw), rows=len(parsed), sha256=sha(snapshot)))
    actual = summarize(rows)
    require(actual['source_exposures'] == state['source_exposures'], 'Audited counts differ from checkpoint')
    require(actual['exposures'] == step * plan['global_batch'], 'Incomplete checkpoint prefix')
    initial = protocol['student_initialization_lineage']; check_binding(initial)
    original = read(initial['path'])
    require(original['initial_sft']['training_records'] == 5220000, 'Original full SFT lineage changed')
    report = dict(schema='anvil-formal-checkpoint-prefix-audit-v1', state='verified_committed_prefix',
        checkpoint=str(checkpoint), adapter_sha256=receipt['files']['adapter_model.safetensors']['sha256'],
        selected_step=step, planned_steps=plan['steps'], global_batch=plan['global_batch'],
        planned_training_completion_inferred=False, selected_for_delivery=False, final_quality_accepted=False,
        actual=actual, rank_prefixes=prefixes, original_full_sft_records=5220000,
        initial_student_lineage=initial, later_updates_counted=False, other_experiment_arms_counted=False,
        required_before_delivery=['Explicit checkpoint selection and training-budget disposition.',
            'Complete580K native and seven-public/general evaluation of this exact checkpoint.',
            'Final quality review, full selected-tip lineage, complete HF package and remote hash verification.'],
        evidence={k:bound(p) for k,p in dict(plan=plan_path, eligibility=Path(plan['eligibility']),
            protocol=run / 'protocol.json', checkpoint=checkpoint / 'checkpoint_complete.json',
            trainer_state=checkpoint / 'trainer_state.json', audit_code=Path(__file__)).items()},
        created=time.time())
    save(output / 'report.json', report)
    validate(output / 'report.json')
    return report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--step', type=int, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); report = build(args.plan, args.step, args.output)
    print(json.dumps(dict(state=report['state'], step=report['selected_step'], actual=report['actual']), indent=2))


if __name__ == '__main__':
    main()
