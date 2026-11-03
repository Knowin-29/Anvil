"""Trace actual student ancestry; separate warm start, full SFT and later exposure."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from common import C,D,R,bound,read,require,save,sha
from exposure_audit import audit


def trace(checkpoint):
    checkpoint=Path(checkpoint).resolve();tip=checkpoint;stages=[];evidence=[];seen=set();logs=[]
    while (checkpoint.parent/'protocol.json').is_file():
        run=checkpoint.parent;protocol=read(run/'protocol.json')
        # Campaign repair stages have explicit initial adapters and actual row logs.
        require(run not in seen,'Ancestry cycle');seen.add(run)
        require(not protocol['smoke'],'Execution-only checkpoint in student lineage')
        complete=read(run/'complete.json');require(Path(complete['checkpoint']).resolve()==checkpoint,'Non-final stage checkpoint')
        require(sha(checkpoint/'adapter_model.safetensors')==complete['adapter_sha256'],'Changed ancestor adapter')
        report=audit(run);stages.append(report);evidence.extend(report['evidence']);evidence.append(bound(checkpoint/'adapter_model.safetensors'))
        world=protocol.get('world',protocol.get('world_size'));logs.extend(run/f'exposure_rank{rank}.jsonl' for rank in range(world))
        initial=protocol.get('initial');require(initial,'Missing stage initialization')
        checkpoint=Path(initial['checkpoint'] if isinstance(initial,dict) else initial).resolve()
        if isinstance(initial,dict):require(sha(checkpoint/'adapter_model.safetensors')==initial['adapter_sha256'],'Changed initial adapter')
    root=R.parent/'a800_repair_20261020/repair_joint';complete=read(root/'complete.json');protocol=read(root/'training_protocol.json')
    require(checkpoint==Path(complete['checkpoint']).resolve(),'Student is not descended from full joint SFT')
    require(sha(checkpoint/'adapter_model.safetensors')==complete['adapter_sha256'],'Changed full-SFT adapter')
    state=read(checkpoint/'trainer_state.json');require(complete['epoch']==state['epoch']==1. and complete['global_step']==state['global_step']==81563,'Full-SFT completion mismatch')
    preflight=R.parent/'a800_repair_20261020/data_preflight.json';data=read(preflight);export=Path(data['export']);manifest=read(export/'manifest.json')
    require(data['records']==manifest['records']==5220000 and sha(export/'manifest.json')==data['manifest_sha256'],'Original SFT scale mismatch')
    require(protocol['world_size']==8 and not protocol['smoke'],'Unexpected initial SFT protocol')
    argv=protocol['argv'];warm=Path(argv[argv.index('--adapters')+1]).resolve();warm_root=warm.parent.parent
    warm_protocol=read(warm_root/'protocol.json');warm_state=read(warm/'trainer_state.json');warm_command=read(warm_root/'training.command.json')
    require(warm_state['global_step']==1024 and warm_state['max_steps']==1024,'Warm start endpoint mismatch')
    require(warm_protocol['export_manifest_sha256']==data['manifest_sha256'],'Warm start source differs')
    require(read(warm_root/'training.result.json')['exit_code']==0,'Warm start did not complete')
    evidence.extend(bound(p) for p in [root/'complete.json',root/'training_protocol.json',checkpoint/'trainer_state.json',checkpoint/'adapter_model.safetensors',preflight,export/'manifest.json',warm_root/'protocol.json',warm_root/'training.command.json',warm_root/'training.result.json',warm_root/'gpu_allocation.json',warm/'trainer_state.json',warm/'adapter_model.safetensors'])
    # Keep direct student exposures distinct from teacher training and all controls.
    counts=Counter();uids=defaultdict(set);images=defaultdict(set);original_selection=manifest['selection_sha256']
    for file in logs:
        for line in file.open():
            row=json.loads(line);binding=row['source_binding'];source=binding.get('source_dataset',binding.get('source'));require(source,'Unidentified source')
            category='original_replay' if binding.get('selection_sha256')==original_selection else 'supplemental'
            key=category+':'+source;counts[key]+=1;uids[key].add(row['canonical_uid']);images[key].add(binding['decoded_rgb_sha256'])
    groups={k:dict(exposures=n,unique_records=len(uids[k]),unique_images=len(images[k]),repeats_within_post_sft=n-len(uids[k])) for k,n in sorted(counts.items())}
    supplement=set().union(*(v for k,v in uids.items() if k.startswith('supplemental:')))
    return dict(schema='anvil-initialization-lineage-v1',state='complete_checkpoint_ancestry',checkpoint=str(tip),adapter_sha256=sha(tip/'adapter_model.safetensors'),
        warm_start=dict(checkpoint=str(warm),updates=1024,source_manifest_sha256=data['manifest_sha256'],
            configured_per_device_batch=int(warm_command[warm_command.index('--per_device_train_batch_size')+1]),
            configured_accumulation=int(warm_command[warm_command.index('--gradient_accumulation_steps')+1]),
            per_row_exposure_log_available=False,exact_unique_warm_start_rows=None,
            note='Earlier 1024-update warm start precedes the complete SFT epoch; its unknown row-level coverage is not counted as additional unique data.'),
        initial_sft=dict(checkpoint=str(checkpoint),training_records=5220000,training_images=142616,epoch=1.,updates=81563,
            source_manifest_sha256=data['manifest_sha256'],coverage_evidence='Completed full map-style epoch and bound dataset; historical trainer did not record one UID per exposure.'),
        post_sft_stages=list(reversed(stages)),post_sft_groups=groups,post_sft_logged_exposures=sum(counts.values()),
        unique_supplemental_student_records=len(supplement),other_experiment_arms_counted_as_final_training=False,
        teachers_training_counted_as_student_exposure=False,complete_final_model_lineage=False,
        inventory=read(D/'augmented_data_scale_reconciliation_v2.json'),evidence=evidence)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    require(not args.output.exists(),'Use a new immutable ancestry report')
    report=trace(args.checkpoint);save(args.output,report)
    print(json.dumps({k:report[k] for k in ['state','checkpoint','post_sft_logged_exposures','unique_supplemental_student_records']},indent=2))


if __name__=='__main__':main()
