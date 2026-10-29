"""Source-preserving corrective replay; native MOPD keeps the full spatial pool."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import numpy as np

O=Path(__file__).resolve().parent
C=O.parent
D=C/'diagnosis_32gpu'
R=C.parent
sys.path[:0]=[str(C/'Anvil'),str(C),str(D),str(R),str(R/'cloud_a800')]
from native_mopd_v6 import FullPool
from checkpoints import sha

SCHEDULES={
    'repair':['mask','fsd','vacancy','qa','native','fsd','vacancy','prism'],
    'mopd':['native','fsd','native','vacancy','native','qa','native','mask',
            'native','fsd','native','vacancy','native','qa','native','prism'],
}


def permute(index,count):
    stride=1000003
    while math.gcd(stride,count)!=1:stride+=2
    return (index*stride+17017)%count


class Pool:
    def __init__(self,spec):
        self.native=FullPool()
        self.export=self.native.export
        self.index=json.loads((R/'training_repair/full_training_index/manifest.json').read_text())
        self.rows={}
        for name,item in spec['datasets'].items():
            assert sha(item['path'])==item['sha256'],name
            self.rows[name]=[json.loads(line) for line in open(item['path'])]
            assert len(self.rows[name])==item['records'],name
            assert all(row['gsa_meta']['split']=='train' for row in self.rows[name]),name
        self.rows['prism']=self.rows['prism']+[json.loads(line) for line in (self.export/'train_dof.jsonl').open()]
        assert len(self.rows['prism'])==16544
        info=self.index['files']['train_mask.jsonl']
        assert sha(info['index_file'])==info['index_sha256']
        self.mask_offsets=np.load(info['index_file'],mmap_mode='r')
        self.mask_handle=(self.export/'train_mask.jsonl').open('rb')

    def sample(self,stage,step,micro,rank,world,global_batch):
        schedule=SCHEDULES[stage]
        cycle,slot=divmod(step,len(schedule))
        kind=schedule[slot]
        occurrence=cycle*schedule.count(kind)+schedule[:slot].count(kind)
        accumulation=global_batch//world
        ordinal=occurrence*global_batch+micro*world+rank
        if kind=='native':
            # One family per update; all records remain reachable without truncation.
            family=occurrence%3
            within=(occurrence//3)*accumulation+micro
            row=self.native.sample(within*3+family,rank,world)
            source='native_distillation' if stage=='mopd' else 'original_native'
        elif kind=='mask':
            index=permute(ordinal,len(self.mask_offsets));offset,size=self.mask_offsets[index]
            self.mask_handle.seek(int(offset));row=json.loads(self.mask_handle.read(int(size)))
            source='mask_replay'
        else:
            name=kind
            if kind=='qa':
                name=['ai2d','scienceqa','textvqa'][occurrence%3]
                ordinal=(occurrence//3)*global_batch+micro*world+rank
            rows=self.rows[name];row=rows[permute(ordinal,len(rows))]
            source=name
        assert row['gsa_meta']['split']=='train'
        return deepcopy(row),kind,source
