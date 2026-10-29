"""Four fixed base shards and four fixed joint shards on one eight-GPU node."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
PY=str(ROOT.parent.parent/'.venv-train/bin/python')

def main():
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(gpus)==8
    for g in gpus:
        _,_,name,memory=map(str.strip,g.split(','));assert 'A800' in name and int(memory)>=79000
    def worker(g):
        index,uuid,_,_=map(str.strip,g.split(','));i=int(index);role='base' if i<4 else 'joint';shard=i%4
        model='/Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B' if role=='base' else str(ROOT.parent/'merged_joint')
        env=os.environ.copy();cache=ROOT/'cache'/role/str(shard);cache.mkdir(parents=True,exist_ok=True)
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=index,ANVIL_EXPECTED_GPU_UUID=uuid,
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
            USE_HF='1',USE_HUB_KERNELS='NO',IMAGE_MAX_TOKEN_NUM='1536',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',
            TRITON_CACHE_DIR=str(cache/'triton'),XDG_CACHE_HOME=str(cache/'xdg'),PYTHONDONTWRITEBYTECODE='1')
        for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','GH_TOKEN','GITHUB_TOKEN'):env.pop(key,None)
        out=ROOT/'benchmarks'/role/str(shard);out.mkdir(parents=True,exist_ok=True)
        cmd=[PY,'-u',str(ROOT/'benchmark_worker.py'),'--model',model,'--role',role,'--shard',str(shard)]
        with (out/'console.log').open('a') as log:
            return subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes=list(pool.map(worker,gpus))
    assert not any(codes),codes
    subprocess.run([PY,str(ROOT/'score_benchmarks.py')],check=True)

if __name__=='__main__':main()
