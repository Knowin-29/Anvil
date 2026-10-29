"""Actual-data preflight; never submit controls before the supplement is sealed."""
from collections import Counter,defaultdict
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
from dof_codebook_interface import training_row
from dof_pose_loss import PoseLoss
from prism_expanded_train import Pool


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    torch.set_num_threads(8);p=processor();loss=PoseLoss(p.tokenizer,'angular');result={}
    manifest=json.loads((ROOT/'prism_expansion/manifest.json').read_text());assert manifest['state']=='verified_train_only_supplement'
    for arm in ['A_prism_union','A_prism_balanced']:
        pool=Pool('A',arm);groups=Counter();unique=defaultdict(set);samples={}
        for step in range(512):
            for micro in range(8):
                for rank in range(4):
                    row=pool.sample(step*8+micro,rank,4)
                    source='original' if row['gsa_meta']['binding']['selection_sha256']==pool.original_selection_sha else 'supplemental_prism'
                    groups[source]+=1;unique[source].add(row['canonical_uid'])
                    samples.setdefault((source,row['gsa_form']),row)
        assert sum(groups.values())==16384 and groups['supplemental_prism']>0
        encodings=[]
        for (source,form),row in samples.items():
            original=row['messages'][1]['content'];derived=training_row(row) if form=='dof' else row
            encoded=encode(p,derived);length=encoded['input_ids'].shape[1]
            if form=='dof':
                meta=loss.target(derived,original)
                assert encoded['labels'][0,-len(meta['ids']):].tolist()==meta['ids']
            encodings.append(dict(source=source,form=form,sequence_tokens=length))
        # Check every supplementary target under the unchanged original codebook.
        for row in pool.additional:
            derived=training_row(row);meta=loss.target(derived,row['messages'][1]['content'])
            assert np.isfinite(meta['code_probabilities']).all() and np.isclose(sum(meta['code_probabilities']),1.)
        result[arm]=dict(planned_exposures=dict(groups),planned_unique_records={k:len(v) for k,v in unique.items()},
            actual_image_encodings=encodings,all_additional_targets_checked=len(pool.additional),training_started=False)
    save(ROOT/'prism_expanded_cpu_validation.json',dict(state='passed_cpu',results=result,
        supplement_manifest_sha256=sha(ROOT/'prism_expansion/manifest.json'),
        original_source_export_unchanged=True,codebook_refitted=False,gpu_runtime_validation=False,
        files_sha256={str(path):sha(path) for path in [Path(__file__),ROOT/'prism_expanded_train.py',ROOT/'dof_pose_loss.py']}))
    print(ROOT/'prism_expanded_cpu_validation.json')


if __name__=='__main__':main()
