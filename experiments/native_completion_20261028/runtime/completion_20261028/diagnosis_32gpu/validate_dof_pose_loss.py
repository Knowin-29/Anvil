"""Verify actual pose-token positions and the weighted objective before GPU use."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path[:0]=[str(ROOT),str(C),str(R),str(R/'cloud_a800')]
import numpy as np
import torch
from checkpoints import save
from training_data import processor,encode
from memory_loss import chunked_ce
from dof_codebook_interface import training_row
from dof_pose_loss import PoseLoss


def main():
    torch.set_num_threads(8);torch.manual_seed(17)
    p=processor();hard=PoseLoss(p.tokenizer,'hard',weight=1.);smooth=PoseLoss(p.tokenizer,'angular')
    source=R.parent/'arxiv/native_training_export_20261012_v4_r2/train_dof.jsonl'
    rows=[json.loads(line) for line in source.open()];assert len(rows)==3008
    entropies=[];lengths=[]
    for row in rows:
        derived=training_row(row);original=row['messages'][1]['content']
        meta=smooth.target(derived,original);ids=meta['ids'];weights=meta['code_probabilities']
        one=hard.target(derived,original)['code_probabilities']
        assert np.isfinite(weights).all() and np.isclose(weights.sum(),1.) and weights.argmax()==one.argmax()
        assert (weights>=0).all()
        entropies.append(float(-(weights*np.log(weights.clip(1e-300))).sum()));lengths.append(len(ids))
    representative=sorted(set([0,1,17,int(np.argmax(lengths)),int(np.argmin(lengths))]))
    encoded=[]
    for i in representative:
        row=rows[i];derived=training_row(row);data=encode(p,derived)
        meta=hard.target(derived,row['messages'][1]['content']);ids=meta['ids'];start=data['labels'].shape[1]-len(ids)
        assert data['labels'][0,start:].tolist()==ids
        encoded.append(dict(index=i,sequence_tokens=data['labels'].shape[1],target_tokens=len(ids),pose_token_positions=[start+j for j in meta['positions']]))
    row=rows[0];derived=training_row(row);original=row['messages'][1]['content']
    meta=hard.target(derived,original);ids=meta['ids'];indices=meta['positions']
    labels=torch.tensor([[-100]*4+ids]);hidden=torch.randn(1,len(ids)+4,4,requires_grad=True)
    head=torch.nn.Linear(4,len(p.tokenizer),bias=False)
    reference=chunked_ce(hidden,head,labels)
    ref_gradient=torch.autograd.grad(reference,hidden,retain_graph=True)[0]
    weighted,_=hard(hidden,head,labels,derived,original)
    got_gradient=torch.autograd.grad(weighted,hidden,retain_graph=True)[0]
    assert torch.allclose(reference,weighted,rtol=2e-6,atol=2e-6)
    assert torch.allclose(ref_gradient,got_gradient,rtol=2e-5,atol=2e-6)
    angular,pose=smooth(hidden,head,labels,derived,original)
    gradient=torch.autograd.grad(angular,hidden)[0]
    assert torch.isfinite(angular) and torch.isfinite(pose) and torch.isfinite(gradient).all()
    assert all(gradient[0,4+i-1].abs().sum()>0 for i in indices)
    report=dict(state='passed_cpu',train_rows=3008,actual_multitoken_codes_and_terminators=True,
        angular_loss='Geodesic Gaussian code weights, conditional on actual reference prefix; full-vocabulary CE at each digit/terminator position',
        exact_reference_token_positions=True,real_image_encodings=encoded,
        weight1_hard_equivalent_to_original_CE=dict(loss_absolute_error=float(abs(reference-weighted)),gradient_max_absolute_error=float((ref_gradient-got_gradient).abs().max())),
        angular_target_entropy_mean=float(np.mean(entropies)),geodesic_sigma_degrees=15.,pose_token_weight=32.,
        finite_angular_gradient=True,source_targets_changed=False,heldout_labels_used=False,
        gpu_training_validated=False,
        files_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'dof_pose_loss.py',ROOT/'validate_dof_pose_loss.py',source]})
    save(ROOT/'dof_pose_cpu_validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
