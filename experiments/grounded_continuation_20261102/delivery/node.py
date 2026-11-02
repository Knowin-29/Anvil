"""Run a static shard partition on one eight-GPU node; no cross-host GPU locks.

Different nodes receive disjoint task indices before submission. Existing valid
results are checked and retained; incomplete worker output uses its own resume
protocol. This never submits cloud tasks or accepts model quality implicitly.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from common import C, D, HERE, TRAIN, EVAL, check_protocol, load_spec, read, require, save, sha


def task_list(spec):
    return ([dict(kind='public', shard=i) for i in range(spec['public_shards'])]
            + [dict(kind='native', shard=i) for i in range(spec['native_shards'])])


def existing_complete(task, out, spec_path, spec):
    file = out / ('complete.json' if task['kind']=='public' else 'metrics.json')
    if not file.exists():
        return False
    result = read(file)
    require(result['state'] == ('complete' if task['kind']=='public' else 'complete_scope_shard'),
            'Incomplete result cannot be skipped')
    require(result['protocol_sha256'] == sha(out/'protocol.json'), 'Protocol receipt mismatch')
    check_protocol(read(out/'protocol.json'),spec_path,spec)
    for name,digest in result['prediction_sha256'].items():
        file = out/(name+'.jsonl') if task['kind']=='public' else out/name
        require(sha(file)==digest,'Saved predictions changed')
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spec',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--node',type=int,required=True)
    ap.add_argument('--nodes',type=int,required=True)
    args=ap.parse_args()
    spec=load_spec(args.spec)
    require(1<=args.nodes<=4 and 0<=args.node<args.nodes,'Expected at most 32 GPUs')
    args.output.mkdir(parents=True,exist_ok=True)
    state_path=args.output/f'node_{args.node}.json'
    topology=dict(spec_sha256=sha(args.spec),nodes=args.nodes)
    topology_path=args.output/'topology.json'
    # Controller writes this before submission; mismatched task partitioning is fatal.
    require(topology_path.exists() and read(topology_path)==topology,'Missing/mismatched sealed node topology')
    lock_path=Path('/tmp')/('anvil-final-node-'+sha(args.spec)+'.lock')
    lease=lock_path.open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    devices=[]
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    require(len(rows)==8,'Expected eight GPUs per node')
    for row in rows:
        index,uuid,name,memory=map(str.strip,row.split(','))
        require('A800' in name and int(memory)>=79000,'Unexpected GPU type')
        devices.append((int(index),uuid))
    tasks=queue.Queue()
    for i,task in enumerate(task_list(spec)):
        if i%args.nodes==args.node:
            tasks.put(task)
    sys.path.insert(0,str(D))
    from retention_node import environment,run
    save(state_path,dict(stage='running',started=time.time(),tasks=tasks.qsize(),devices=devices))
    def worker(device):
        gpu,uuid=device;failures=[]
        while True:
            try:task=tasks.get_nowait()
            except queue.Empty:return failures
            out=args.output/task['kind']/str(task['shard'])
            try:
                if existing_complete(task,out,args.spec,spec):continue
                env=environment([gpu],f'final_{args.node}_{gpu}')
                env['ANVIL_EXPECTED_GPU_UUID']=uuid
                env['ANVIL_FINAL_REPRESENTATION']=spec['representation']
                common=['--model',spec['model'],'--spec',str(args.spec.resolve()),'--output',str(out),
                        '--shard',str(task['shard'])]
                if task['kind']=='public':
                    argv=[TRAIN,str(HERE/'public_worker.py'),*common,'--role','final_mopd',
                          '--shards',str(spec['public_shards'])]
                else:
                    argv=[EVAL,str(HERE/'native_worker.py'),*common,'--name','final_mopd',
                          '--scope','joint','--gpu-uuid',uuid,'--shards',str(spec['native_shards']),
                          '--concurrency','64','--memory','.88']
                    if (out/'protocol.json').exists():argv+=['--resume']
                    elif out.exists():
                        require(not any(out.iterdir()),'Unbound prior native output')
                        out.rmdir()
                log=args.output/'logs'/f"{task['kind']}_{task['shard']}.log"
                launch=args.output/'launches'/f"{task['kind']}_{task['shard']}.json"
                launch.parent.mkdir(exist_ok=True)
                save(launch,dict(argv=argv,gpu=gpu,uuid=uuid,started=time.time()))
                run(argv,env,log)
                require(existing_complete(task,out,args.spec,spec),'Worker returned without complete evidence')
            except BaseException as exc:
                failures.append(dict(task=task,error=str(exc)))
                out.mkdir(parents=True,exist_ok=True)
                save(out/'node_failure.json',dict(error=str(exc),time=time.time()))
            finally:tasks.task_done()
    failures=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(worker,d) for d in devices]):failures.extend(future.result())
    save(state_path,dict(stage='failed' if failures else 'node_complete',failures=failures,completed=time.time()))
    require(not failures,'Final evaluation worker failed; original output retained')


if __name__=='__main__':main()
