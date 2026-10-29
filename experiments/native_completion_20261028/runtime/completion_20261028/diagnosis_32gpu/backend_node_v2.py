from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from repair_node import environment,run,TRAIN
jobs=json.loads((ROOT/'backend_jobs.json').read_text())
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines();assert len(gpus)==len(jobs)==8
def worker(pair):
    line,arm=pair;i,u,n,m=map(str.strip,line.split(','));assert 'A800' in n and int(m)>=79000
    env=environment([i],'backend_'+i);env['ANVIL_EXPECTED_GPU_UUID']=u
    run([TRAIN,str(ROOT/'backend_probe_v2.py'),'--arm',arm],env,ROOT/'backend_probes_v2'/arm/'console.log')
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,zip(gpus,jobs)))
(ROOT/'backend_complete_v2.json').write_text(json.dumps({arm:json.loads((ROOT/'backend_probes_v2'/arm/'complete.json').read_text()) for arm in jobs},indent=2)+'\n')
