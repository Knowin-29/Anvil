"""One preparation of substantive 32-GPU repair→MOPD, no extra GPU tests."""
from pathlib import Path
import sys,json,time,hashlib,shlex,collections,math
O=Path(__file__).resolve().parent;C=O.parent;D=C/'diagnosis_32gpu';R=C.parent
sys.path[:0]=[str(O),str(C/'Anvil'),str(C),str(D),str(R),str(R/'cloud_a800')]
from checkpoints import sha,save,verify
import yaml,torch
import train,pool
from mopd_representation import Representation
from training_data import encode,processor
from repair_train import token_unlikelihood
from retention_node import environment


def main():
    assert not (O/'spec.json').exists(),'Preserve submitted immutable spec'
    eligibility=json.loads((C/'formal_mopd/runs/candidate1_20261029/eligibility.json').read_text())
    initial=Path(eligibility['joint_adapter']);receipt=verify(initial,8)
    datasets={}
    for name,path,n in [
        ('vacancy',C/'vacancy_expansion_candidate_20261029/train.jsonl',182878),
        ('fsd',D/'spatial_replay/fsd_goal.jsonl',29448),
        ('prism',D/'prism_expansion/train_bound.jsonl',13536)]:
        datasets[name]=dict(path=str(path),sha256=sha(path),records=n)
    qa=json.loads((D/'retention_replay/manifest.json').read_text())
    for name in ['ai2d','scienceqa','textvqa']:
        path=D/'retention_replay'/(name+'.jsonl');assert sha(path)==qa['files_sha256'][name]
        datasets[name]=dict(path=str(path),sha256=sha(path),records=qa['additional_training_records'][name])
    lineage=C/'diagnosis_32gpu/mopd_v6_candidate1/initialization_lineage.json'
    spec=dict(schema='anvil-direct-corrective-MOPD-v1',world=32,global_batch=64,
        initial=dict(adapter=str(initial),adapter_sha256=receipt['files']['adapter_model.safetensors']['sha256']),
        teachers=eligibility['teachers'],datasets=datasets,
        initialization_lineage=dict(path=str(lineage),sha256=sha(lineage)),
        teacher_assessment=dict(existing_review=str(C/'formal_mopd/candidate1_quality_review.json'),
            review_sha256=sha(C/'formal_mopd/candidate1_quality_review.json'),
            decision='Reuse the exact reviewed G/S/A teachers while directly correcting the student; no new teacher-superiority or final-quality claim.',
            new_student_paired_qualification_claimed=False),
        stages=dict(repair=dict(steps=2048,lr=2e-5,save_every=256,ul_weight=.1,ul_rollout_cap=768),
                    mopd=dict(steps=1536,lr=5e-6,save_every=128,native_sft_weight=.1,mtd_weight=1.,ul_weight=.1,ul_rollout_cap=768)),
        rationale='User prioritizes full corrective training over more small ablations. Greedy mask-cycle UL previously improved full81250mask validity to98.30%; spatial replay improvedVA26.33→71.00. Use independently screened182878vacancy records; preserve QA/native/PRISM. Formal MOPD continues after repair without a small-test gate.',
        deadline_utc='2026-11-02T05:48:00+00:00',final_reserve_hours=24,
        original_corpus_records=5800000,original_full_sft_records=5220000,original_spatial_pool=4756000,
        candidate_data_inventory_not_actual_exposure=True,final_quality_accepted=False,
        final_required=['One exact finalcheckpoint full580K native,7public and10105general evaluation',
            'Upload final full weights and key checkpoints; verify all remote files at pinned commit'],
        key_checkpoints=['repair/checkpoint-2048','mopd/checkpoint-512','mopd/checkpoint-1024','mopd/checkpoint-1536'])
    p=pool.Pool(spec);rep=Representation('rotation128');rep.validate_adapters({k:v['adapter'] for k,v in spec['teachers'].items()},str(initial))
    checked=[];processor_=processor();torch.set_num_threads(4)
    for stage in ['repair','mopd']:
        seen=set();natives=set();counts=collections.Counter()
        for step in range(len(pool.SCHEDULES[stage])*3):
            for micro in range(2):
                for rank in range(32):
                    row,kind,source=p.sample(stage,step,micro,rank,32,64)
                    counts[source]+=1
                    key=(kind,row['canonical_uid']);assert key not in seen,'Unexpected within-prefix repeated source';seen.add(key)
                    if kind=='native':natives.add(row['gsa_family'])
                    if rank==0 and micro==0:
                        encoded=encode(processor_,rep.row(row));assert encoded['input_ids'].shape[1]<=16384
                        checked.append(dict(stage=stage,step=step,source=source,uid=row['canonical_uid'],tokens=encoded['input_ids'].shape[1]))
        assert natives==set('GSA')
        assert sum(counts.values())==len(pool.SCHEDULES[stage])*3*64
        planned=collections.Counter()
        for step in range(spec['stages'][stage]['steps']):
            _,_,source=p.sample(stage,step,0,0,32,64)
            planned[source]+=64
        spec.setdefault('planned_source_exposures',{})[stage]=dict(planned)
        # Counts need full schedule periods also for QA round-robin. Use direct schedule arithmetic.
        expected={kind:spec['stages'][stage]['steps']//len(pool.SCHEDULES[stage])*pool.SCHEDULES[stage].count(kind)*64 for kind in set(pool.SCHEDULES[stage])}
        spec.setdefault('planned_role_exposures',{})[stage]=expected
    # UL gradient is compared against the defining -log(1-p) on an actual dense head.
    torch.manual_seed(17)
    hidden=torch.randn(1,9,5,requires_grad=True);head=torch.nn.Linear(5,13,bias=False);head.requires_grad_(False)
    ids=torch.randint(0,13,(1,9));positions=[3,5,8]
    actual=token_unlikelihood(hidden,head,ids,positions)
    logits=head(hidden[0,torch.tensor(positions)-1]);prob=logits.softmax(-1).gather(1,ids[0,positions,None]).squeeze(1)
    expected=-torch.log1p(-prob).mean()
    assert torch.allclose(actual,expected,atol=1e-6)
    a=torch.autograd.grad(actual,hidden,retain_graph=True)[0];b=torch.autograd.grad(expected,hidden)[0];assert torch.allclose(a,b,atol=1e-6)
    # Validate imports in the same paths used by worker and node before freezing.
    import node
    validation=dict(state='passed',encoded_real_rows=len(checked),schedule_rows_checked=4608,
        all_three_native_families_reachable=True,ul_loss_and_gradient_parity=True,
        initial_checkpoint_verified=True,extra_gpu_smoke_tasks=0,
        execution_proof='First real production optimizer steps/checkpoint; no synthetic smoke is promoted.',created=time.time())
    save(O/'cpu_validation.json',validation)
    source_files=[*O.glob('*.py'),R/'training_data.py',R/'eval_core.py',R/'cloud_a800/checkpoints.py',R/'cloud_a800/memory_loss.py',
        C/'native_mopd_v6.py',C/'native_mopd_v2.py',C/'mopd_representation.py',C/'mopd_lineage.py',C/'completion_policy.py',C/'completion_policy.json',
        D/'repair_train.py',D/'audit_targets.py',D/'dof_pose_loss.py',D/'retention_node.py',lineage,
        C/'formal_mopd/candidate1_quality_review.json',O/'cpu_validation.json']
    source_files+=list((C/'Anvil/trains/mopd').glob('*.py'))
    source_files+=list((C/'Anvil/trains/flywheel').glob('*.py'))
    source_files+=list(map(Path,rep.descriptor['files_sha256']))
    for directory in ['spatial_replay','retention_replay','prism_expansion']:source_files.append(D/directory/'manifest.json')
    source_files+=[C/'vacancy_expansion_candidate_20261029/manifest.json',C/'vacancy_expansion_candidate_20261029/policy.json']
    spec['frozen_files']={str(f):sha(f) for f in sorted(set(source_files))}
    save(O/'spec.json',spec);train.load_spec(O/'spec.json')
    conf=yaml.safe_load((C/'formal_mopd/runs/candidate1_20261029/training_cli.yaml').read_text())
    conf['TaskName']='task_train_anvil_corrective_then_MOPD_A800x32'
    conf['Description']='32GPU substantive student repair2048 then genuine MOPD1536; fullpool,QA,expandedspatial,PRISM,greedy-cycleUL; preserve key checkpoints'
    conf['TaskRoleSpecs'][0]['RoleReplicas']=4;conf['ActiveDeadlineSeconds']=240000
    conf['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(R/'.venv-train/bin/python'),'-u',str(O/'node.py'),'--spec',str(O/'spec.json'),'--spec-sha256',sha(O/'spec.json')])
    (O/'cli.yaml').write_text(yaml.safe_dump(conf,sort_keys=False))
    print(json.dumps(dict(validation=validation,planned_roles=spec['planned_role_exposures'],spec_sha256=sha(O/'spec.json')),indent=2))

if __name__=='__main__':main()
