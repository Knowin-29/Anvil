"""CPU source/encoding checks for the changed loss, not a model-quality trial."""
import hashlib
import json
from pathlib import Path
import sys
import time

O=Path(__file__).resolve().parent;C=O.parent;R=C.parent
sys.path[:0]=[str(O),str(C),str(C/'Anvil'),str(C/'diagnosis_32gpu'),str(R)]
import torch
from native_io_pool import Pool
from geometry_loss import GeometryLoss
from training_data import processor,encode,EMPTY_THINK
from mopd_representation import Representation


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    torch.set_num_threads(2)
    spec=json.loads((O/'candidate_settings.json').read_text())
    p=processor();loss=GeometryLoss(p.tokenizer,EMPTY_THINK);representation=Representation('rotation128')
    pool=Pool(spec);results={}
    for form in ['mask','affordance','bbox','detection','point','dof']:
        count=0;scalars=[];positions=[]
        for i in range(min(128,len(pool.offsets[form]))):
            source=pool.native_row('repair',form,i,'native:'+form)
            assert source['gsa_meta']['split']=='train'
            row=representation.row(source)
            before=json.dumps(row,sort_keys=True)
            meta=loss.plan(row['messages'][1]['content'])
            assert before==json.dumps(row,sort_keys=True),'Objective must not mutate source annotations'
            if form in {'mask','affordance','bbox','detection','dof'}:
                assert meta['scalars']>0,(form,row['canonical_uid'])
            scalars.append(meta['scalars']);positions.append(len(meta['positions']));count+=1
            if i==0:
                data=encode(p,row)
                start=data['labels'].shape[1]-len(meta['ids'])
                assert data['labels'][0,start:].tolist()==meta['ids']
                assert (data['labels'][0,:start]==-100).all()
                assert data['pixel_values'].numel()>0
                assert min(meta['positions'],default=1)+start>0
        results[form]=dict(rows_checked=count,min_scalars=min(scalars),max_scalars=max(scalars),
                           mean_coordinate_tokens=sum(positions)/count,actual_image_encoding_passed=True)
    output=dict(state='cpu_geometry_objective_source_checks_passed',checked=time.time(),
                model_quality_validated=False,gpu_jobs_submitted=False,forms=results,
                files={str(p):sha(p) for p in [O/'geometry_loss.py',O/'train.py',O/'validate_sources.py',O/'candidate_settings.json']})
    (O/'source_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':main()
