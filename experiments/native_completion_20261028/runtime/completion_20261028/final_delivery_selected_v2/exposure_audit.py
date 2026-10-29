"""Count actual completed stage exposures, unique source records and repetitions.

This audits one stage. It never adds unrelated experimental arms to final model
lineage, or equates the available source inventory with training coverage.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from common import bound,read,require,save,sha


def audit(run,allow_execution_only=False,selection_path=None):
    if selection_path is not None:
        from selection import validate
        selection,prefix=validate(selection_path)
        require(selection['training_run']==str(Path(run).resolve()),'Wrong selected exposure run')
        actual=prefix['actual']
        evidence=list(prefix['evidence'].values())+[bound(selection_path),selection['evidence']['prefix']]
        files=[item['snapshot'] for item in sorted(prefix['rank_prefixes'],key=lambda v:v['rank'])]
        evidence.extend(bound(path) for path in files)
        return dict(schema='anvil-stage-exposure-audit-v1',state='complete_selected_prefix_audit',
            run=selection['training_run'],checkpoint=selection['checkpoint'],execution_only=False,
            steps=selection['selected_step'],planned_steps=selection['planned_steps'],
            global_batch=prefix['global_batch'],exposures=actual['exposures'],groups=actual['groups'],
            unique_records=actual['unique_records'],unique_images=actual['unique_images'],by_form=actual['by_form'],
            evidence=evidence,exposure_files=files,other_experiment_arms_included=False,
            complete_final_model_lineage=False,original_planned_run_completed=selection['original_planned_run_completed'],
            budget_disposition=selection['budget_disposition'],later_training_rows_counted=False)
    run=Path(run).resolve();protocol=read(run/'protocol.json');complete=read(run/'complete.json')
    smoke=protocol['smoke']
    require(not smoke or allow_execution_only,'Execution-only exposures must be explicitly identified')
    steps=complete.get('steps',complete.get('step'))
    require(isinstance(steps,int) and steps>0,'Missing completed update count')
    world=protocol.get('world_size',protocol.get('world'));batch=protocol['global_batch']
    require(isinstance(world,int) and batch%world==0,'Invalid distributed batch')
    require(smoke or steps==protocol.get('steps',protocol.get('updates')),'Incomplete scheduled training')
    checkpoint=Path(complete['checkpoint']);sealed=read(checkpoint/'checkpoint_complete.json')
    require(sealed['global_step']==steps,'Checkpoint and exposure endpoint differ')
    seen=set();by_step=Counter();counts=Counter();forms=Counter();uids=defaultdict(set);images=defaultdict(set)
    examples=[];evidence=[bound(run/'protocol.json'),bound(run/'complete.json'),bound(checkpoint/'checkpoint_complete.json')]
    for rank in range(world):
        path=run/f'exposure_rank{rank}.jsonl';evidence.append(bound(path))
        for line in path.open():
            row=json.loads(line);require(row['smoke']==smoke and row['rank']==rank,'Wrong exposure scope')
            require(1<=row['step']<=steps,'Uncommitted trailing exposure: explicit recovery required')
            key=(row['step'],row['micro'],rank);require(key not in seen,'Duplicate exposure from replayed update')
            seen.add(key);by_step[(row['step'],rank)]+=1
            binding=row['source_binding']
            # The source identity is retained even for supervised auxiliary replay.
            source=binding.get('source_dataset',binding.get('source'))
            require(bool(source),'Exposure lacks source identity')
            category=row.get('source_category','original_native' if 'selection_sha256' in binding else 'supplemental_reference')
            group=category+':'+source
            counts[group]+=1;forms[row['form']]+=1;uids[group].add(row['canonical_uid'])
            rgb=binding.get('decoded_rgb_sha256');require(bool(rgb),'Missing decoded image identity')
            images[group].add(rgb)
            if row.get('source_category') in {'general_qa','spatial_reference'}:
                require(row['rollout_tokens']==row['teacher_scored_positions']==0,'Replay unexpectedly used teacher rollout')
    require(len(seen)==steps*batch,'Exposure count mismatch')
    require(len(by_step)==steps*world and all(n==batch//world for n in by_step.values()),
            'Missing or repeated rank/update exposures')
    actual_categories=Counter()
    for group,n in counts.items():actual_categories[group.split(':',1)[0]]+=n
    if 'source_exposures' in complete:
        require(dict(actual_categories)==complete['source_exposures'],'Completion exposure totals differ')
    groups={group:dict(exposures=n,unique_records=len(uids[group]),repeated_exposures=n-len(uids[group]),
        unique_images=len(images[group]),unique_ids_sha256=hashlib.sha256('\n'.join(sorted(uids[group])).encode()).hexdigest())
        for group,n in sorted(counts.items())}
    return dict(schema='anvil-stage-exposure-audit-v1',state='complete_stage_audit',run=str(run),checkpoint=str(checkpoint),
        execution_only=smoke,steps=steps,global_batch=batch,exposures=len(seen),groups=groups,
        unique_records=len(set().union(*uids.values())),unique_images=len(set().union(*images.values())),
        by_form=dict(forms),evidence=evidence,other_experiment_arms_included=False,
        complete_final_model_lineage=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-execution-only',action='store_true');args=ap.parse_args()
    save(args.output,audit(args.run,args.allow_execution_only));print(args.output)


if __name__=='__main__':main()
