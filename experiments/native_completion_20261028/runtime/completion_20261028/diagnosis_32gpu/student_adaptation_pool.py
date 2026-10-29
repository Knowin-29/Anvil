"""Student representation adaptation, independently sampled with preserved replay."""
from copy import deepcopy
import json
from pathlib import Path
from spatial_train import Pool as ReplayPool, sha
from prism_expanded_train import Pool as PrismPool
from dof_codebook_interface import training_row
ROOT=Path(__file__).resolve().parent

class Pool:
    schedule=['dof','native','dof','qa','dof','native','dof','spatial']
    def __init__(self):
        self.replay=ReplayPool('joint_spatial')
        self.prism=PrismPool('A','A_prism_union')
        self.index=self.replay.index
        self.source_manifests=dict(self.replay.source_manifests,prism_expansion=sha(ROOT/'prism_expansion/manifest.json'))
        self.union=self.prism.union
    def sample(self,micro,rank,world):
        cycle,slot=divmod(micro,len(self.schedule));kind=self.schedule[slot]
        occurrence=cycle*self.schedule.count(kind)+self.schedule[:slot].count(kind)
        ordinal=occurrence*world+rank
        if kind=='dof':
            row=self.union[self.replay.permute(ordinal,len(self.union))]
            source='original_dof' if row['gsa_meta']['binding']['selection_sha256']==self.prism.original_selection_sha else 'supplemental_prism'
        elif kind=='native':
            row=self.replay.native.sample(occurrence,rank,world);source='original_native'
        elif kind=='qa':
            source=['ai2d','scienceqa','textvqa'][occurrence%3]
            rows=self.replay.rows[source];index=(occurrence//3)*world+rank
            row=rows[self.replay.permute(index,len(rows))]
        else:
            sub=occurrence%4;source='refspatial_vacancy' if sub==3 else 'fsd_goal'
            index=(occurrence//4 if sub==3 else (occurrence//4)*3+sub)*world+rank
            rows=self.replay.rows[source];row=rows[self.replay.permute(index,len(rows))]
        assert row['gsa_meta']['split']=='train'
        return deepcopy(row),source

def load_spec(path):
    spec=json.loads(Path(path).read_text())
    assert spec['schema']=='anvil-student-representation-adaptation-v1'
    assert spec['world']==8 and spec['global_batch']==32 and spec['steps'] in (512,1024)
    assert spec['pose_mode'] in ('hard','angular') and spec['final_mopd'] is False
    assert spec['learning_rate']>0 and spec['learning_rate']<=2e-5
    assert sha(ROOT/'student_adaptation_cpu_validation.json')==spec['cpu_validation_sha256']
    for name,digest in spec['frozen_files'].items():assert sha(name)==digest,name
    initial=spec['initial'];receipt=json.loads(Path(initial['completion_receipt']).read_text())
    assert receipt['smoke'] is False and receipt['state']=='repair_complete_requires_evaluation'
    assert receipt['checkpoint']==initial['checkpoint']
    assert sha(initial['completion_receipt'])==initial['completion_receipt_sha256']
    assert sha(Path(initial['checkpoint'])/'adapter_model.safetensors')==initial['adapter_sha256']==receipt['adapter_sha256']
    return spec
