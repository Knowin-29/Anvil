"""Explicit pose-token weighting/geometry smoothing; source answers stay fixed."""
import json
import re
from collections import defaultdict
import numpy as np
import torch
from memory_loss import chunked_ce
from training_data import EMPTY_THINK
from dof_codebook_interface import ROTATIONS
from prepare_dof_rotation_codebook import distances


class PoseLoss:
    def __init__(self,tokenizer,mode,weight=32.,sigma_degrees=15.):
        assert mode in {'hard','angular'} and weight>0 and sigma_degrees>0
        self.tokenizer,self.mode,self.weight,self.sigma=tokenizer,mode,weight,sigma_degrees
        # Qwen3.5 numbers use multiple actual digit tokens. Include the first
        # delimiter token so a one-digit bin is distinguishable from a longer
        # bin sharing its prefix. No synthetic single-token code is assumed.
        self.sequences=[];starts=[]
        for code in range(128):
            text=EMPTY_THINK+'{"rotation_bin":'+str(code)+',"position":'
            encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            match=re.search(r'"rotation_bin":(\d+)',text)
            indices=[i for i,(left,right) in enumerate(encoded['offset_mapping'])
                     if left<match.end(1) and right>match.start(1)]
            start,end=indices[0],indices[-1]+1
            assert encoded['offset_mapping'][start][0]==match.start(1)
            assert encoded['offset_mapping'][end][0]==match.end(1)
            starts.append(start);self.sequences.append(encoded['input_ids'][start:end+1])
        assert len(set(starts))==1 and len({tuple(x) for x in self.sequences})==128
        self.start=starts[0]

    def target(self,derived,original_target):
        target=EMPTY_THINK+derived['messages'][1]['content']+'<|im_end|>\n'
        encoded=self.tokenizer(target,add_special_tokens=False,return_offsets_mapping=True)
        match=re.search(r'"rotation_bin":(\d+)',target);assert match is not None
        code=int(match.group(1));sequence=self.sequences[code]
        positions=list(range(self.start,self.start+len(sequence)))
        assert encoded['input_ids'][self.start:self.start+len(sequence)]==sequence
        rotation=np.asarray(json.loads(original_target)['camera_from_gripper'])[:3,:3]
        angle=distances(rotation[None],ROTATIONS)[0]
        assert int(angle.argmin())==code,'Source pose/code mismatch'
        if self.mode=='hard':
            weights=np.zeros(128);weights[code]=1.
        else:
            logits=-.5*(angle/np.radians(self.sigma))**2
            weights=np.exp(logits-logits.max());weights/=weights.sum()
        distributions=[]
        for offset in range(len(sequence)):
            distribution=defaultdict(float)
            for candidate,probability in zip(self.sequences,weights):
                if len(candidate)>offset and candidate[:offset]==sequence[:offset]:
                    distribution[candidate[offset]]+=float(probability)
            total=sum(distribution.values());assert total>0
            distributions.append({token:mass/total for token,mass in distribution.items() if mass>0})
        return dict(ids=encoded['input_ids'],positions=positions,code_probabilities=weights,
                    token_probabilities=distributions)

    def __call__(self,hidden,head,labels,derived,original_target):
        meta=self.target(derived,original_target);ids=meta['ids']
        assert labels.shape[0]==1
        start=labels.shape[1]-len(ids);positions=[start+i for i in meta['positions']]
        assert labels[0,start:].tolist()==ids and min(positions)>0
        normal=labels.clone();normal[0,positions]=-100
        n=int((normal[:,1:]!=-100).sum())
        token_ce=chunked_ce(hidden,head,normal)
        states=hidden[0,torch.tensor(positions,device=hidden.device)-1]
        log_prob=head(states).float().log_softmax(-1)
        pose_total=log_prob.sum()*0
        for i,distribution in enumerate(meta['token_probabilities']):
            tokens=torch.tensor(list(distribution),device=log_prob.device)
            weights=torch.tensor(list(distribution.values()),device=log_prob.device,dtype=log_prob.dtype)
            pose_total=pose_total-(weights*log_prob[i,tokens]).sum()
        pose_ce=pose_total/len(positions)
        loss=(token_ce*n+self.weight*pose_total)/(n+self.weight*len(positions))
        assert torch.isfinite(loss)
        return loss,pose_ce
