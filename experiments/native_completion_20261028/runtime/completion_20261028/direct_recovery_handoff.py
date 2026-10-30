"""After the real direct pipeline finishes, run one complete final evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
C=Path(__file__).resolve().parent
H=C/'final_delivery_recovery_v1'
sys.path.insert(0,str(H))
from common import TRAIN,formal_training,load_spec,read,require,save,sha
sys.path.insert(0,str(C))
from final_delivery_handoff_v1 import cloud_snapshot,submission_response
sys.path.insert(0,str(C/'formal_mopd'))
from controller import register,TERMINAL


def validate_policy(policy):
    require(policy['schema']=='anvil-direct-final-handoff-v1','Different policy')
    for filename,digest in policy['files'].items():require(sha(filename)==digest,'Changed handoff source '+filename)
    require(sha(policy['pipeline_spec'])==policy['pipeline_spec_sha256'],'Changed training pipeline')
    require(policy['gpus']==32 and policy['nodes']==4,'Wrong capacity')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',type=Path,required=True);args=ap.parse_args()
    policy=read(args.policy);validate_policy(policy)
    root=args.policy.parent;pipeline=Path(policy['pipeline_spec']).parent
    identity=read(pipeline/'submission.json')['response']['Id']
    own=(root/'supervisor.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**fields):save(root/'status.json',dict(stage=stage,updated=time.time(),pid=os.getpid(),final_quality_accepted=False,weights_uploaded=False,**fields))
    while time.time()<policy['deadline']:
        cloud,active=cloud_snapshot();job=next(j for j in cloud if j['id']==identity)
        require(job['status'] not in TERMINAL or job['status']=='Success','Training terminal without success: '+job['status'])
        if job['status']=='Success':break
        status('waiting_direct_training',task_id=identity,cloud_status=job['status'],active_campaign_gpus=active)
        time.sleep(60)
    else:raise TimeoutError('Training has not completed before final deadline')
    validate_policy(policy)
    complete,protocol,checkpoint,receipt=formal_training(pipeline/'runs/mopd')
    def run_tool(script,*argv):
        log=root/'logs'/(Path(script).stem+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:subprocess.run([TRAIN,'-u',str(H/script),*map(str,argv)],stdout=f,stderr=subprocess.STDOUT,check=True)
    merged=root/'merged_model';spec_path=root/'evaluation_spec.json';evaluation=root/'evaluation'
    status('preparing_final_model',checkpoint=str(checkpoint))
    if not merged.exists():run_tool('merge.py','--adapter',checkpoint,'--output',merged)
    if not spec_path.exists():run_tool('prepare.py','--training-run',pipeline/'runs/mopd','--model',merged,'--output',spec_path,'--native-shards',32,'--public-shards',8)
    load_spec(spec_path);run_tool('configs.py','--spec',spec_path,'--output',evaluation,'--nodes',4)
    lineage=root/'training_lineage.json'
    if not lineage.exists():run_tool('lineage.py','--spec',spec_path,'--eligibility',pipeline/'spec.json','--output',lineage)
    shared=open('/tmp/anvil-formal-mopd-submit.lock','a');fcntl.flock(shared,fcntl.LOCK_EX)
    identities={}
    for node in range(4):
        folder=root/'submissions'/str(node);folder.mkdir(parents=True,exist_ok=True)
        config=evaluation/f'node_{node}_cli.yaml'
        if (folder/'submission.json').exists():
            submitted=read(folder/'submission.json');require(submitted['final_spec_sha256']==sha(spec_path),'Different final spec')
        else:
            require(not (folder/'submission_intent.json').exists(),'Ambiguous submission; inspect cloud before retry')
            while time.time()<policy['deadline']:
                cloud,active=cloud_snapshot()
                if active+8<=32:break
                status('waiting_final_capacity',active_campaign_gpus=active);time.sleep(60)
            else:raise TimeoutError('No capacity before deadline')
            save(folder/'capacity.json',dict(active=active,requested=8,cloud=cloud,checked=time.time()))
            validate_policy(policy)
            binding=dict(final_spec_sha256=sha(spec_path),config_sha256=sha(config),gpus=8,time=time.time())
            save(folder/'submission_intent.json',binding)
            proc=subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','submit','--conf',str(config),'--output','json'],capture_output=True,text=True,timeout=90)
            (folder/'submission.log').write_text(proc.stdout+proc.stderr)
            submitted=dict(binding,response=submission_response(proc.stdout,proc.returncode));save(folder/'submission.json',submitted)
        register(submitted,8,'direct_full_final_evaluation_node'+str(node),config)
        identities[node]=submitted['response']['Id']
    fcntl.flock(shared,fcntl.LOCK_UN)
    while time.time()<policy['deadline']:
        cloud,active=cloud_snapshot();by_id={j['id']:j for j in cloud};finished=[]
        for node,identity in identities.items():
            observed=by_id[identity]['status']
            require(observed not in TERMINAL or observed=='Success','Final evaluation failed: '+identity)
            if observed=='Success':
                state=read(evaluation/f'node_{node}.json');require(state['stage']=='node_complete' and not state['failures'],'Incomplete final node')
                finished.append(node)
        status('full_final_evaluation_running',jobs=identities,finished_nodes=finished,active_campaign_gpus=active)
        if len(finished)==4:break
        time.sleep(60)
    else:raise TimeoutError('Final evaluation deadline')
    report=evaluation/'final_report.json'
    if not report.exists():run_tool('score.py','--spec',spec_path,'--output',evaluation)
    result=read(report);require(result['state']=='complete_requires_quality_review' and result['final_spec_sha256']==sha(spec_path),'Different final report')
    status('full_final_report_ready_requires_quality_review',report=str(report),report_sha256=sha(report),lineage=str(lineage))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if '--policy' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--policy')+1]).parent
            save(root/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
