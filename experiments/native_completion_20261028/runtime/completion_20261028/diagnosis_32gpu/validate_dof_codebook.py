"""Real CPU target/decoder/encoding checks; no claim of learned pose accuracy."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'eval_code/score_code'),str(ROOT.parent.parent)]
from eval.native_metrics import score_row
from dof_codebook_interface import training_row,decode_response,context,PATH
from training_data import encode,processor
import hashlib

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    path=ROOT.parent.parent.parent/'arxiv/native_training_export_20261012_v4_r2/train_dof.jsonl'
    stats=[];representatives=[]
    for row in map(json.loads,path.open()):
        transformed=training_row(row);prompt=row['messages'][0]['content']
        assert context(prompt)['entity']['entity_id'] not in transformed['messages'][0]['content']
        response=decode_response(transformed['messages'][1]['content'],prompt)
        evaluation=dict(row,native_form='dof',native_variant=row['gsa_native_variant'])
        score=score_row(evaluation,response);assert score['contract_valid']
        stats.append(score)
        if len(representatives)<2:representatives.append(transformed)
    assert len(stats)==3008
    invalid=['{"rotation_bin":128,"position":[0,0,0,0,0,0]}',
             '{"rotation_bin":true,"position":[0,0,0,0,0,0]}',
             '{"rotation_bin":0,"position":[0,0,0,0,0,NaN]}',
             '{"rotation_bin":0,"position":[0,0]}']
    for response in invalid:
        try:decode_response(response,prompt)
        except ValueError:pass
        else:raise AssertionError('Malformed output accepted')
    p=processor();encodings=[]
    for row in representatives:
        encoded=encode(p,row)
        encodings.append(dict(tokens=encoded['input_ids'].shape[1],supervised_tokens=int((encoded['labels']!=-100).sum())))
    report=dict(state='passed_cpu',train_target_roundtrips=3008,all_geometry_valid=True,
        quantization_train_only={k:float(np.mean([s[k] for s in stats])) for k in stats[0]},
        malformed_rejected=len(invalid),real_encodings=encodings,
        model_performance_verified=False,heldout_fitted=False,production_mopd_integrated=False,
        bindings={str(p):sha(p) for p in [path,PATH,Path(__file__),ROOT/'dof_codebook_interface.py',ROOT/'dof_codebook_train.py',ROOT/'dof_codebook_node.py',ROOT/'dof_codebook_eval.py']})
    (ROOT/'dof_codebook_cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
