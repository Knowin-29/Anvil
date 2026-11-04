"""Prepare reviewable DLC configs for static final-evaluation partitions."""
import argparse
from pathlib import Path
import shlex
import yaml
from common import D,HERE,TRAIN,load_spec,read,require,save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--nodes',type=int,choices=[1,2,3,4],required=True)
    args=ap.parse_args();load_spec(args.spec)
    args.output.mkdir(parents=True,exist_ok=True)
    topology=dict(spec_sha256=sha(args.spec),nodes=args.nodes)
    file=args.output/'topology.json'
    if file.exists():require(read(file)==topology,'Existing topology differs')
    else:save(file,topology)
    template=yaml.safe_load((D/'spatial_joint_cli.yaml').read_text())
    for node in range(args.nodes):
        config=dict(template)
        config['TaskName']=f'task_anvil_final_eval_{sha(args.spec)[:8]}_node{node}_A800x8'
        config['Description']='Full final model validation; exact checkpoint bound across native/public/general sets'
        config['Tags']=['eval'];config['ActiveDeadlineSeconds']=172800
        argv=[TRAIN,'-u',str(HERE/'node.py'),'--spec',str(args.spec.resolve()),'--output',str(args.output.resolve()),'--node',str(node),'--nodes',str(args.nodes)]
        config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join(argv)
        path=args.output/f'node_{node}_cli.yaml'
        text=yaml.safe_dump(config,allow_unicode=True,sort_keys=False)
        if path.exists():require(path.read_text()==text,'Config already exists with different contents')
        else:path.write_text(text)
        print(shlex.join(['/root/.volc/bin/volc','--log-level','error','ml_task','submit','--conf',str(path),'--output','json']))
    print(f'Prepared {args.nodes*8} GPUs. Submit only after checking total active campaign allocation stays <=32.')


if __name__=='__main__':main()
