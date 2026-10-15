#!/usr/bin/env python3
"""CPU-only, immutable re-scoring of completed native prediction forms.

No inference, model download, answer repair, or modification of a source run.
Legacy migration is limited to the two verified explicit-template v1 runners.
Use --form for an already completed whole form while other forms are running;
add --wait to accept each requested form as soon as it completes. Omit --form
only when waiting for or processing the complete scope/shard.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import native_metrics
from tools import run_native_evaluation as runner
from tools.response_text import POLICY, RAW_POLICY, SWIFT_NON_THINKING_PREFIX
from tools.train_native_specialist import save, sha
from trains.flywheel.native_training import canonical, require, strict_json

VERSION = 'anvil-native-generated-text-rescore-v1'
LEGACY_RUNNERS = {
    '3fb2f83bc568defeaee8a5a84a893d1800a36f1230fc6cce771f80b79ae24da2',
    '9a39d6d3f140c0f5df78502dc31b28ee3698205370a35fb0a256236cdd1ec567',
}
CODE = tuple(sorted(set(('tools/rescore_native_evaluation.py', 'tools/response_text.py', *runner.CODE))))


def validate_and_rescore(item, row, ordinal, run_sha, legacy, region=None, *, sdk_response_prefix=SWIFT_NON_THINKING_PREFIX):
    expected = runner.prediction_item(row, ordinal, item['response'], item['finish_reason'],
        item['usage'], run_sha, region, sdk_response_prefix=None if legacy else sdk_response_prefix)
    require(item == expected, 'Original prediction identity, order, protocol or metrics changed')
    return runner.prediction_item(row, ordinal, item['response'], item['finish_reason'],
        item['usage'], run_sha, region, sdk_response_prefix=sdk_response_prefix)


def checked_run(run, export):
    """Validate the immutable generation protocol before accepting any rows."""
    protocol_path = run/'protocol.json'; protocol_sha = sha(protocol_path)
    protocol = strict_json(protocol_path.read_bytes())
    legacy = protocol['schema'] == 'anvil-native-evaluation-run-v1'
    require(legacy or protocol['schema'] in {'anvil-native-evaluation-run-v2', 'anvil-native-evaluation-run-v3', runner.VERSION},
            'Unsupported generation protocol')
    require(protocol['thinking'] is False and protocol['context_length'] == 32768,
            'Explicit non-thinking generation protocol required')
    if legacy:
        require(protocol['code_sha256']['tools/run_native_evaluation.py'] in LEGACY_RUNNERS,
                'Legacy template implementation has not been verified for this migration')
    response_policy = POLICY if legacy else protocol['response_text_policy']
    if not legacy:
        require(response_policy == POLICY or (protocol['schema'] == 'anvil-native-evaluation-run-v4'
                and response_policy == RAW_POLICY and protocol.get('model_type') == 'qwen3_vl'), 'Different SDK text policy')
    for name, expected in protocol['code_sha256'].items():
        path = run/'code'/name
        require(path.resolve().is_relative_to(run/'code') and sha(path) == expected,
                'Frozen generation code changed')
    require(protocol['code_sha256']['eval/native_metrics.py'] == sha(Path(native_metrics.__file__)),
            'Metric formulas changed; this migration only changes SDK transport decoding')
    manifest_path = export/'manifest.json'
    require(sha(manifest_path) == protocol['export_manifest_sha256'], 'Different evaluation export')
    manifest = strict_json(manifest_path.read_bytes())
    all_files = runner.selected_files(manifest, protocol['scope'], protocol['shard'], protocol['shards'])
    require(dict(all_files) == protocol['files'], 'Generation scope/shard differs from complete export')
    return protocol, protocol_sha, legacy, response_policy, manifest, all_files


def rescore(run, export, output, forms=()):
    run, export, output = (Path(p).resolve() for p in (run, export, output))
    require(not output.exists(), 'Use a new derived output directory; source predictions are never overwritten')
    protocol_path = run/'protocol.json'; manifest_path = export/'manifest.json'
    protocol, protocol_sha, legacy, response_policy, manifest, all_files = checked_run(run, export)
    require(len(set(forms)) == len(forms) and set(forms) <= {i['form'] for _, i in all_files}, 'Invalid or repeated form')
    full_scope = not forms
    files = [(n, i) for n, i in all_files if not forms or i['form'] in forms]
    terminal = None
    if full_scope:
        terminal = strict_json((run/'metrics.json').read_bytes())
        require(terminal['state'] == 'complete_scope_shard' and terminal['run_protocol_sha256'] == protocol_sha
                and terminal['n'] == sum(i['shard_records'] for _, i in files), 'Source scope is not complete')
    bindings = {}
    for name, info in files:
        form = info['form']; predictions = run/f'predictions_{form}.jsonl'
        receipt_path = run/f'complete_{form}.json'; receipt = strict_json(receipt_path.read_bytes())
        require(receipt == {'records':info['shard_records'], 'prediction_sha256':sha(predictions),
                           'dataset_sha256':info['sha256'], 'protocol_sha256':protocol_sha},
                'Form is incomplete or its prediction receipt changed')
        require(sha(export/name) == info['sha256'], 'Evaluation records changed')
        if terminal: require(terminal['prediction_files'][form] == receipt['prediction_sha256'], 'Final prediction binding differs')
        bindings[form] = {'prediction_sha256':receipt['prediction_sha256'], 'completion_sha256':sha(receipt_path),
                          'dataset_sha256':info['sha256'], 'records':info['shard_records']}
    output.mkdir(parents=True)
    code_sha = {n:sha(Path(__file__).parents[1]/n) for n in CODE}
    save(output/'request.json', {'schema':VERSION, 'generation_protocol_sha256':protocol_sha,
        'response_text_policy':response_policy, 'input_files':bindings, 'code_sha256':code_sha, 'scope_complete':full_scope})
    loader = None
    try:
        if any(i['form']=='point' for _, i in files) and any(s['source']=='coco_train2017' for s in manifest['source_inputs']):
            loader = native_metrics.CocoPointReference(manifest)
        def scored_rows():
            for name, info in files:
                form = info['form']; predictions = run/f'predictions_{form}.jsonl'
                count = 0
                with predictions.open('rb') as saved, (output/f'rescored_{form}.jsonl').open('x', encoding='utf-8') as destination:
                    for ordinal, row in runner.shard_rows(export/name, form, manifest['selection_sha256'],
                                                         protocol['shard'], protocol['shards']):
                        raw = saved.readline(); require(raw.endswith(b'\n'), 'Missing or incomplete prediction')
                        item = strict_json(raw)
                        region = loader(row) if loader and form=='point' and row['evaluation_meta']['source']=='coco_train2017' else None
                        revised = validate_and_rescore(item, row, ordinal, protocol_sha, legacy, region,
                                                      sdk_response_prefix=response_policy['sdk_response_prefix'])
                        # The unchanged SDK response remains in the original bound file.
                        revised['raw_response_sha256'] = hashlib.sha256(revised.pop('response').encode()).hexdigest()
                        destination.write(canonical(revised)+'\n'); count += 1
                        yield revised
                    require(not saved.read(1), 'Extra predictions outside the declared shard')
                require(count == info['shard_records'], 'Derived denominator mismatch')
        result = native_metrics.summarize(scored_rows())
        for name, info in files:
            form = info['form']
            require(sha(run/f'predictions_{form}.jsonl') == bindings[form]['prediction_sha256']
                    and sha(run/f'complete_{form}.json') == bindings[form]['completion_sha256']
                    and sha(export/name) == info['sha256'], 'Inputs changed during re-scoring')
        require(sha(protocol_path)==protocol_sha and sha(manifest_path)==protocol['export_manifest_sha256']
                and all(sha(Path(__file__).parents[1]/n)==value for n,value in code_sha.items()),
                'Protocol, export or rescorer changed during re-scoring')
        result.update(schema=VERSION, state='complete_scope_shard_rescored' if full_scope else 'complete_forms_rescored',
            scope_complete=full_scope, scope=protocol['scope'], shard=protocol['shard'], shards=protocol['shards'],
            source_generation_protocol_sha256=protocol_sha, response_text_policy=response_policy,
            input_files=bindings, derived_files={p.name:sha(p) for p in output.glob('rescored_*.jsonl')},
            original_predictions_changed=False, model_calls=0, teacher_qualified=False,
            mopd_complete=False, public_release_eligible=False)
        require(result['n']==sum(i['shard_records'] for _,i in files), 'Incomplete derived scope')
        save(output/'metrics.json',result)
        return result
    except BaseException as error:
        save(output/'failure.json', {'type':type(error).__name__, 'reason':str(error), 'partial_outputs_retained':True})
        raise
    finally:
        if loader: loader.close()


def wait_for_scope(run, export, output, binding):
    """Wait for an exact existing producer; never start/restart GPU inference."""
    while True:
        require(sha(run/'request.json')==binding['generation_request_sha256'], 'Generation request changed while waiting')
        require(not (run/'failure.json').exists(), 'Generation failed; keep its predictions and inspect the source run')
        if (run/'metrics.json').exists():
            state = strict_json((run/'status.json').read_bytes())
            if state.get('stage')=='complete_scope_shard': break
        require(runner.process_identity(binding['producer']['pid'])==binding['producer'],
                'Producer stopped or PID changed without a completed scope')
        save(output/'status.json', {'stage':'waiting_for_complete_generation', 'pid':os.getpid(),
                                   'producer':binding['producer']})
        time.sleep(20)
    for name, expected in binding['code_sha256'].items():
        require(sha(Path(__file__).parents[1]/name)==expected, 'Frozen CPU re-scoring code changed')
    save(output/'status.json', {'stage':'rescoring_complete_generation', 'pid':os.getpid()})
    result = rescore(run,export,output/'derived')
    save(output/'status.json', {'stage':'complete', 'pid':os.getpid(), 'records':result['n'], 'model_calls':0})
    return result


def form_wait_request(run, export, forms):
    """Bind a requested subset before creating a background CPU worker."""
    protocol = strict_json((run/'protocol.json').read_bytes())
    require(sha(export/'manifest.json') == protocol['export_manifest_sha256'], 'Different evaluation export')
    manifest = strict_json((export/'manifest.json').read_bytes())
    files = runner.selected_files(manifest, protocol['scope'], protocol['shard'], protocol['shards'])
    require(dict(files) == protocol['files'], 'Generation scope/shard differs from complete export')
    require(bool(forms) and len(set(forms)) == len(forms)
            and set(forms) <= {info['form'] for _, info in files}, 'Invalid or repeated form')
    return {'forms': [info['form'] for _, info in files if info['form'] in forms],
            'generation_protocol_sha256': sha(run/'protocol.json'),
            'export_manifest_sha256': sha(export/'manifest.json')}


def wait_for_forms(run, export, output, binding):
    """Accept immutable whole forms incrementally; never call them a full scope."""
    requested = binding['form_request']
    require(form_wait_request(run, export, requested['forms']) == requested, 'Form request changed')
    result = {'schema': 'anvil-native-form-rescore-watch-v1',
              'state': 'waiting_for_complete_forms', 'scope_complete': False,
              'generation_protocol_sha256': requested['generation_protocol_sha256'],
              'export_manifest_sha256': requested['export_manifest_sha256'],
              'requested_forms': requested['forms'], 'completed_forms': {}, 'n': 0,
              'model_calls': 0, 'teacher_qualified': False, 'mopd_complete': False,
              'public_release_eligible': False, 'original_predictions_changed': False}
    while True:
        require(sha(run/'request.json') == binding['generation_request_sha256'],
                'Generation request changed while waiting')
        require(sha(run/'protocol.json') == requested['generation_protocol_sha256']
                and sha(export/'manifest.json') == requested['export_manifest_sha256'],
                'Protocol or export changed while waiting')
        for form in requested['forms']:
            if form in result['completed_forms'] or not (run/f'complete_{form}.json').exists(): continue
            for name, expected in binding['code_sha256'].items():
                require(sha(Path(__file__).parents[1]/name) == expected, 'Frozen CPU re-scoring code changed')
            save(output/'status.json', {'stage': 'rescoring_complete_form', 'form': form, 'pid': os.getpid()})
            derived = output/'forms'/form
            metrics = rescore(run, export, derived, [form])
            require(metrics['scope_complete'] is False and metrics['state'] == 'complete_forms_rescored',
                    'Incremental result must not claim a completed scope')
            result['completed_forms'][form] = {'records': metrics['n'],
                'metrics': f'forms/{form}/metrics.json', 'metrics_sha256': sha(derived/'metrics.json')}
            result['n'] += metrics['n']
            save(output/'index.json', result)
        if len(result['completed_forms']) == len(requested['forms']):
            result['state'] = 'complete_selected_forms_rescored'
            save(output/'index.json', result)
            save(output/'status.json', {'stage': 'complete_selected_forms', 'pid': os.getpid(),
                                       'records': result['n'], 'scope_complete': False, 'model_calls': 0})
            return result
        # Completed receipts remain usable if the producer later fails elsewhere.
        # Missing forms, however, must never cause an implicit GPU restart.
        require(not (run/'failure.json').exists(), 'Generation failed; completed forms remain preserved')
        require(runner.process_identity(binding['producer']['pid']) == binding['producer'],
                'Producer stopped or PID changed before the requested forms completed')
        save(output/'status.json', {'stage': 'waiting_for_complete_forms', 'pid': os.getpid(),
            'producer': binding['producer'], 'completed_forms': list(result['completed_forms']),
            'pending_forms': [f for f in requested['forms'] if f not in result['completed_forms']]})
        time.sleep(20)


def launch_wait(args):
    if not args.watch_worker:
        require(not args.output.exists(), 'Use a new watcher output directory')
        producer = strict_json((args.run/'process.json').read_bytes())
        require(runner.process_identity(producer['pid'])==producer, 'Expected generation worker is not currently live')
        requested = form_wait_request(args.run, args.export, args.form) if args.form else None
        args.output.mkdir(parents=True)
        snapshot=args.output/'code'
        for name in CODE:
            destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(__file__).parents[1]/name,destination)
        for name in ('tools','eval','trains','trains/flywheel'):(snapshot/name/'__init__.py').touch()
        binding={'producer':producer,'generation_request_sha256':sha(args.run/'request.json'),
                 'code_sha256':{name:sha(snapshot/name) for name in CODE}}
        if requested: binding['form_request'] = requested
        save(args.output/'watch_binding.json',binding)
        command=[sys.executable,'-B',str(snapshot/'tools/rescore_native_evaluation.py'),
                 '--run',str(args.run),'--export',str(args.export),'--output',str(args.output),'--wait','--watch-worker']
        if requested:
            for form in requested['forms']: command.extend(('--form', form))
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
        raise SystemExit(subprocess.call(command,env=env))
    binding=strict_json((args.output/'watch_binding.json').read_bytes())
    require(args.form == binding.get('form_request', {}).get('forms', []), 'Watcher form arguments changed')
    save(args.output/'process.json',runner.process_identity(os.getpid()))
    try:
        wait = wait_for_forms if args.form else wait_for_scope
        return wait(args.run,args.export,args.output,binding)
    except BaseException as error:
        save(args.output/'failure.json',{'type':type(error).__name__,'reason':str(error),
                                       'gpu_restarted':False,'source_predictions_preserved':True})
        raise


def merge_request(runs, export, forms=()):
    """Require the complete ordinal partition of one exact model/protocol."""
    require(len(runs) >= 2 and len(set(runs)) == len(runs), 'Use distinct, complete generation shards')
    checked = sorted([(run, checked_run(run, export)) for run in runs], key=lambda pair: pair[1][0]['shard'])
    first = checked[0][1][0]
    common = {k:v for k,v in first.items() if k not in {'files', 'shard'}}
    require(first['shards'] == len(runs)
            and [data[0]['shard'] for _,data in checked] == list(range(len(runs))), 'Missing or repeated shard ordinal')
    for _, data in checked:
        require({k:v for k,v in data[0].items() if k not in {'files','shard'}} == common,
                'Shards use different models, templates, code, or generation protocols')
    available = [i['form'] for _,i in checked[0][1][-1]]
    require(len(set(forms)) == len(forms) and set(forms) <= set(available), 'Invalid or repeated merge form')
    return {'schema':'anvil-native-shard-merge-request-v1', 'common_protocol':common,
        'forms':[f for f in available if not forms or f in forms], 'full_scope_requested':not forms,
        'sources':[{'run':str(run), 'shard':data[0]['shard'], 'protocol_sha256':data[1],
                    'request_sha256':sha(run/'request.json')} for run,data in checked]}


def merge_form(request, export, form):
    """Re-score every ordinal once, streaming across shards without duplicating raw files."""
    require(form in request['forms'], 'Form not requested')
    sources = request['sources']; name = f'eval_{form}.jsonl'
    runs = [Path(s['run']) for s in sources]
    checked = [checked_run(run, export) for run in runs]
    manifest = checked[0][4]; info = manifest['files'][name]
    require(sha(export/name) == info['sha256'], 'Evaluation file changed')
    bindings = []
    for run, source, data in zip(runs, sources, checked):
        protocol, run_sha = data[:2]
        require(run_sha == source['protocol_sha256'] and sha(run/'request.json') == source['request_sha256'],
                'Generation request or protocol changed')
        predictions = run/f'predictions_{form}.jsonl'; receipt = run/f'complete_{form}.json'
        expected = {'records':protocol['files'][name]['shard_records'], 'prediction_sha256':sha(predictions),
                    'dataset_sha256':info['sha256'], 'protocol_sha256':run_sha}
        require(strict_json(receipt.read_bytes()) == expected, 'Incomplete form or changed prediction receipt')
        bindings.append({**source, **expected, 'completion_sha256':sha(receipt)})
    loader = None
    try:
        if form == 'point' and any(s['source']=='coco_train2017' for s in manifest['source_inputs']):
            loader = native_metrics.CocoPointReference(manifest)
        def rows():
            counts = [0]*len(runs)
            with ExitStack() as stack:
                streams = [stack.enter_context((run/f'predictions_{form}.jsonl').open('rb')) for run in runs]
                for ordinal, row in runner.shard_rows(export/name, form, manifest['selection_sha256'], 0, 1):
                    shard = ordinal % len(runs); raw = streams[shard].readline()
                    require(raw.endswith(b'\n'), 'Missing or incomplete shard prediction')
                    item = strict_json(raw); data = checked[shard]
                    region = loader(row) if loader and row['evaluation_meta']['source']=='coco_train2017' else None
                    counts[shard] += 1
                    yield validate_and_rescore(item, row, ordinal, data[1], data[2], region,
                                               sdk_response_prefix=data[3]['sdk_response_prefix'])
                require(all(not stream.read(1) for stream in streams), 'Extra shard predictions')
                require(counts == [b['records'] for b in bindings], 'Shard denominator mismatch')
        result = native_metrics.summarize(rows())
        require(result['n'] == info['records'], 'Merged form denominator mismatch')
        for run,b in zip(runs,bindings):
            require(sha(run/f'predictions_{form}.jsonl') == b['prediction_sha256']
                    and sha(run/f'complete_{form}.json') == b['completion_sha256']
                    and sha(run/'protocol.json') == b['protocol_sha256']
                    and sha(run/'request.json') == b['request_sha256'], 'Source changed during shard merge')
        require(sha(export/name) == info['sha256']
                and sha(export/'manifest.json') == request['common_protocol']['export_manifest_sha256'],
                'Evaluation export changed during shard merge')
        result.update(schema='anvil-native-merged-form-v1', state='complete_form_all_shards', form=form,
                      source_bindings=bindings, scope_complete=False, model_calls=0,
                      original_predictions_changed=False, teacher_qualified=False, mopd_complete=False)
        return result
    finally:
        if loader: loader.close()


def wait_for_merge(runs, export, output, binding, wait):
    request = binding['request']; forms = request['forms']; common = request['common_protocol']
    result = {'schema':'anvil-native-shard-merge-v1', 'state':'waiting_for_shards', 'scope':common['scope'],
        'scope_complete':False, 'full_benchmark_complete':False, 'n':0, 'groups':[],
        'global_composite_score':None, 'completed_forms':{}, 'model_calls':0,
        'original_predictions_changed':False, 'teacher_qualified':False, 'mopd_complete':False,
        'public_release_eligible':False, 'merge_request_sha256':sha(output/'merge_binding.json')}
    while True:
        require(merge_request(runs, export, () if request['full_scope_requested'] else forms) == request,
                'Merge input protocol changed')
        require(all(sha(Path(__file__).parents[1]/n) == value for n,value in binding['code_sha256'].items()),
                'Frozen merger code changed')
        for form in forms:
            if form in result['completed_forms'] or not all((run/f'complete_{form}.json').exists() for run in runs): continue
            save(output/'status.json', {'stage':'merging_complete_form', 'form':form, 'pid':os.getpid()})
            merged = merge_form(request, export, form)
            path = output/'forms'/f'{form}.json'; path.parent.mkdir(exist_ok=True); save(path, merged)
            result['completed_forms'][form] = {'records':merged['n'], 'metrics':f'forms/{form}.json', 'sha256':sha(path)}
            result['n'] += merged['n']; result['groups'].extend(merged['groups'])
            result['groups'].sort(key=lambda group:(group['form'],group['source']))
            save(output/'index.json', result)
        forms_done = len(result['completed_forms']) == len(forms)
        terminal = []
        for source in request['sources']:
            run = Path(source['run']); pending = [f for f in forms if not (run/f'complete_{f}.json').exists()]
            if not pending and not request['full_scope_requested']: continue
            require(not (run/'failure.json').exists(), 'Generation failed; accepted complete forms remain preserved')
            if not pending and (run/'metrics.json').exists() and strict_json((run/'status.json').read_bytes()).get('stage')=='complete_scope_shard':
                final = strict_json((run/'metrics.json').read_bytes())
                protocol = strict_json((run/'protocol.json').read_bytes())
                require(final['state']=='complete_scope_shard' and final['run_protocol_sha256']==source['protocol_sha256']
                        and final['n']==sum(i['shard_records'] for i in protocol['files'].values()), 'Invalid terminal shard receipt')
                terminal.append({'shard':source['shard'], 'metrics_sha256':sha(run/'metrics.json')})
                continue
            producer = binding['producers'][source['run']]
            require(producer is not None and runner.process_identity(producer['pid'])==producer,
                    'Producer stopped or changed before terminal shard acceptance')
        if forms_done and (not request['full_scope_requested'] or len(terminal)==len(runs)):
            identities = set()
            for form, item in result['completed_forms'].items():
                require(sha(output/item['metrics'])==item['sha256'], 'Accepted merged metrics changed')
                merged = strict_json((output/item['metrics']).read_bytes())
                require(sha(export/f'eval_{form}.jsonl') == merged['source_bindings'][0]['dataset_sha256'],
                        'Accepted evaluation data changed')
                manifest = strict_json((export/'manifest.json').read_bytes())
                for _, row in runner.shard_rows(export/f'eval_{form}.jsonl', form, manifest['selection_sha256'], 0, 1):
                    require(row['id'] not in identities, 'Repeated identity across merged forms')
                    identities.add(row['id'])
                for b in merged['source_bindings']:
                    run = Path(b['run'])
                    require(sha(run/f'predictions_{form}.jsonl')==b['prediction_sha256']
                            and sha(run/f'complete_{form}.json')==b['completion_sha256'], 'Accepted source predictions changed')
                    if request['full_scope_requested']:
                        require(strict_json((run/'metrics.json').read_bytes())['prediction_files'][form]==b['prediction_sha256'],
                                'Terminal prediction binding differs')
            require(len(identities)==result['n'] and all(sha(Path(__file__).parents[1]/n)==value
                    for n,value in binding['code_sha256'].items()), 'Merged identity count or code changed')
            require(merge_request(runs, export, () if request['full_scope_requested'] else forms)==request,
                    'Merge input protocol changed during final acceptance')
            result.update(state='complete_scope' if request['full_scope_requested'] else 'complete_selected_forms',
                scope_complete=request['full_scope_requested'], full_benchmark_complete=request['full_scope_requested'] and common['scope']=='joint',
                terminal_shards=terminal)
            save(output/'metrics.json',result); save(output/'index.json',result)
            save(output/'status.json',{'stage':result['state'],'pid':os.getpid(),'records':result['n'],'model_calls':0})
            return result
        require(wait, 'Some shards or terminal checks are incomplete; use --wait with live producers')
        save(output/'status.json',{'stage':'waiting_for_shards','pid':os.getpid(),
            'completed_forms':list(result['completed_forms']), 'pending_forms':[f for f in forms if f not in result['completed_forms']],
            'terminal_shards':len(terminal),'scope_complete':False})
        time.sleep(20)


def merge_main():
    parser = argparse.ArgumentParser(description='CPU-only complete ordinal-shard aggregation; no GPU restart')
    parser.add_argument('--run', action='append', type=Path, required=True)
    for name in ('export','output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--form', action='append', default=[])
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--merge-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(sys.argv[2:])
    args.run=[p.resolve() for p in args.run];args.export=args.export.resolve();args.output=args.output.resolve()
    if not args.merge_worker:
        require(not args.output.exists() and not args.output.is_relative_to(args.export)
                and all(not args.output.is_relative_to(run) for run in args.run), 'Use a new output outside source runs/export')
        request = merge_request(args.run,args.export,args.form)
        producers={s['run']:strict_json((Path(s['run'])/'process.json').read_bytes()) for s in request['sources']}
        for run in args.run:
            p=producers[str(run)]
            require(runner.process_identity(p['pid'])==p or all((run/f'complete_{f}.json').exists() for f in request['forms']),
                    'A requested producer is not live and its forms are incomplete')
        snapshot=args.output/'code'; snapshot.mkdir(parents=True)
        for name in CODE:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(__file__).parents[1]/name,target)
        for name in ('tools','eval','trains','trains/flywheel'):(snapshot/name/'__init__.py').touch()
        save(args.output/'merge_binding.json',{'request':request,'producers':producers,
             'code_sha256':{name:sha(snapshot/name) for name in CODE}})
        env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','GITHUB_TOKEN','GH_TOKEN'):env.pop(key,None)
        command=[sys.executable,'-B',str(snapshot/'tools/rescore_native_evaluation.py'),*sys.argv[1:],'--merge-worker']
        raise SystemExit(subprocess.call(command,env=env))
    save(args.output/'process.json',runner.process_identity(os.getpid()))
    try:
        result=wait_for_merge(args.run,args.export,args.output,strict_json((args.output/'merge_binding.json').read_bytes()),args.wait)
        print(json.dumps({'state':result['state'],'n':result['n'],'scope_complete':result['scope_complete']}))
    except BaseException as error:
        save(args.output/'failure.json',{'type':type(error).__name__,'reason':str(error),'gpu_restarted':False,'accepted_forms_preserved':True})
        raise


def main():
    if len(sys.argv)>1 and sys.argv[1]=='merge': return merge_main()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','export','output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--form', action='append', default=[])
    parser.add_argument('--wait', action='store_true', help='Freeze this CPU tool and follow one exact live worker; --form enables incremental acceptance')
    parser.add_argument('--watch-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ('run','export','output'):setattr(args,name,getattr(args,name).resolve())
    require(not args.watch_worker or args.wait, 'Internal watcher requires --wait')
    result = launch_wait(args) if args.wait else rescore(args.run,args.export,args.output,args.form)
    print(json.dumps(result))


if __name__ == '__main__': main()
