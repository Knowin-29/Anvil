"""One bounded step512 native diagnostic; preserve original runs and scorers."""
import ast
import fcntl
from pathlib import Path
import sys
import time

C=Path(__file__).resolve().parent
sys.path[:0]=[str(C),str(C/'formal_mopd')]
import controller
from formal_common import read,require,save,sha,bound
from final_delivery_handoff_v1 import observe
import midrun_gs_diagnostic_512 as diagnostic

ROOT=C/'native_diagnostics/candidate1_step512'
FORMAL=C/'formal_mopd/runs/candidate1_20261029'
SCOPE='formal_step512_fixed_GS35989_diagnostic'


def capacity(cloud,step):
    active=sum(j['gpus'] for j in cloud if j['status'] not in controller.TERMINAL)
    public=sum(j['gpus'] for j in cloud if j['status'] not in controller.TERMINAL
               and j.get('scope','').startswith('formal_MOPD_public_step'))
    require(active<=32 and public<=8,'Existing campaign allocation exceeds policy')
    # Submit only well before the next public checkpoint. The task has a90min
    # deadline and is expected to finish before1024; future formal submissions
    # still perform their own fresh cap checks and cannot exceed32.
    reserve=max(0,8-public) if step>=768 else 0
    return step<768 and active+8+reserve<=32,dict(active=active,requested=8,
        public_active=public,public_reserved=reserve,cap=32,formal_step=step)


def register(receipt,gpus,scope,config):
    require(gpus==8 and scope==SCOPE,'Unexpected supplementary registration')
    jobs=read(C/'jobs_v2.json');identity=receipt['response']['Id']
    if identity not in {j['id'] for j in jobs}:
        require(identity not in {j['id'] for j in read(C/'diagnosis_32gpu/jobs.json')},'Duplicate registry')
        jobs.append(dict(id=identity,gpus=8,scope=scope,config=str(config)))
        save(C/'jobs_v2.json',jobs)


def preflight():
    require(not ROOT.exists(),'Use a fresh declared diagnostic directory')
    ROOT.mkdir(parents=True)
    def body(path,name):
        tree=ast.parse(path.read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    for name in ['checked_rows','paired_group_rows']:
        require(body(C/'midrun_gs_diagnostic.py',name)==body(C/'midrun_gs_diagnostic_512.py',name),'Scorer/pairing logic changed')
    validated=read(C/'midrun_gs_cpu_validation.json')
    require(validated['state']=='passed' and validated['real_baseline_rows']==35989,'Original baseline CPU validation missing')
    for path,h in validated['real_baseline_evidence_sha256'].items():require(sha(path)==h,'Original baseline predictions changed')
    train=dict(id='t',gpus=16,status='Running',scope='formal_MOPDv6')
    control=dict(id='c',gpus=8,status='Running',scope='objective_control')
    require(capacity([train,control],600)[0],'Expected24+8 capacity')
    require(not capacity([train,control,dict(id='u',gpus=8,status='UNKNOWN',scope='unknown')],600)[0],'UNKNOWN not charged')
    require(not capacity([train,control],768)[0],'Late launch allowed')
    require(not capacity([train,control],1024)[0],'Public checkpoint collision')
    spec=diagnostic.prepare(FORMAL/'evaluation/512/spec.json',ROOT)
    require(diagnostic.load_spec(spec)['selection']['expected_records']==35989,'Unexpected diagnostic denominator')
    report=dict(state='passed_cpu_only',real_baseline_rows=35989,
        original_pairing_and_scorer_functions_unchanged=True,real512_model_spec_verified=True,
        checks=['24+8allowed_before768','UNKNOWNcharged','step768launchrefused','step1024launchrefused'],
        original_baseline_validation=bound(C/'midrun_gs_cpu_validation.json'),
        new_code={str(p):sha(p) for p in [Path(__file__),C/'midrun_gs_diagnostic_512.py']},
        gpu_submitted=False,created=time.time())
    save(ROOT/'cpu_validation.json',report)
    policy=dict(schema='anvil-step512-paired-GS-diagnostic-policy-v1',step=512,native_GS_records=35989,
        gpu_budget=8,launch_before_formal_step=768,task_deadline_seconds=5400,
        purpose='Distinguish native-geometry learning from public regression after formal512 and negative objective controls.',
        decision_after_observing_public512=True,original_fixed_sample_unchanged=True,
        full_final_validation=False,original_training_modified=False,
        evidence={k:bound(p) for k,p in {
            'formal512_review':C/'formal512_review_20261029.json',
            'low_sft_no_mtd':C/'objective_controls_phase2_20261029/evaluation/low_native_sft_no_mtd/report.json',
            'cpu_validation':ROOT/'cpu_validation.json','spec':spec}.items()},
        frozen_files={str(p):sha(p) for p in [Path(__file__),C/'midrun_gs_diagnostic_512.py',FORMAL/'plan.json']},created=time.time())
    save(ROOT/'policy.json',policy)
    print('Prepared verified512 diagnostic without GPU submission',flush=True)


def launch():
    policy=read(ROOT/'policy.json')
    for path,h in policy['frozen_files'].items():require(sha(path)==h,'Changed diagnostic policy dependency')
    for item in policy['evidence'].values():require(sha(item['path'])==item['sha256'],'Changed declaration evidence')
    # These finished supplementary controllers must not race jobs_v2 writes.
    for pid in [2324260,2272013,1887396]:
        require(not Path('/proc',str(pid)).exists(),'Earlier supplementary submitter still alive')
    controller.check=observe;controller.register=register
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cloud,_=controller.cloud_snapshot()
        statuses={j['id']:j['status'] for j in cloud}
        require(statuses['t-20261029095820-px5rk']=='Running','Formal run no longer live')
        require(statuses['t-20261029151550-n7h6w']=='Success','Control has not released its8GPUs')
        step=read(FORMAL/'training/status.json')['step']
        allowed,allocation=capacity(cloud,step)
        save(ROOT/'reserved_capacity.json',dict(cloud=cloud,**allocation,checked=time.time()))
        require(allowed,'No safe diagnostic capacity/window; no submission')
        identity=controller.submit(ROOT,ROOT/'cli.yaml',8,SCOPE,FORMAL/'plan.json')
        require(identity,'Capacity changed before submission; no retry')
        save(ROOT/'controller_status.json',dict(stage='submitted',id=identity,updated=time.time(),
            final_validation=False,original_training_modified=False))
        print(identity,flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--prepare']:preflight()
    elif sys.argv[1:]==['--submit']:launch()
    else:raise SystemExit('Use --prepare or --submit')
