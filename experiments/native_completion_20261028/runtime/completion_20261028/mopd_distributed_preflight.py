"""Real cross-node NCCL and GPU identity check, without model/data mutation."""
import argparse
from datetime import timedelta
import json,os,socket,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'cloud_a800'))
from checkpoints import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    assert world==16 and 0<=local<8
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180),device_id=device)
    prop=torch.cuda.get_device_properties(device);assert 'A800' in prop.name and prop.total_memory>=79*10**9
    identity=dict(rank=rank,local_rank=local,host=socket.gethostname(),uuid=str(prop.uuid))
    identities=[None]*world;dist.all_gather_object(identities,identity)
    assert len({x['uuid'] for x in identities})==16 and len({x['host'] for x in identities})==2
    assert all(sum(x['host']==host for x in identities)==8 for host in {x['host'] for x in identities})
    tensor=torch.empty(16*1024*1024,dtype=torch.float32,device=device);times=[]
    for iteration in range(5):
        tensor.fill_(rank);torch.cuda.synchronize();t0=time.monotonic()
        dist.all_reduce(tensor);torch.cuda.synchronize();elapsed=time.monotonic()-t0
        assert torch.all(tensor==sum(range(world)))
        if iteration:times.append(elapsed)
    args.output.mkdir(parents=True,exist_ok=True)
    save(args.output/f'rank-{rank}.json',dict(state='passed_real_16GPU_collectives',identity=identity,world=world,
        identities=identities,allreduce_bytes=tensor.numel()*tensor.element_size(),seconds=times))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
