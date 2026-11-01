"""CPU-only failure-path checks; fixtures are not a release or training result."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent
C=ROOT.parent
module_spec=importlib.util.spec_from_file_location('balanced_hub_verifier',ROOT/'verify_entire_balanced_hub.py')
m=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(m)
prior=json.loads((C/'targeted_checkpoint_backup_20261031/mopd_768/verification_receipt.json').read_text())
pipeline=json.loads((C/'balanced_repair_20261102/spec.json').read_text())
backup=copy.deepcopy(prior)
source_prefix='checkpoints/mopd/checkpoint-768/'
for key in sorted(m.KEYS-{r['path'] for r in backup['key_checkpoints']}):
    prefix='checkpoints/'+key+'/'
    for name,item in prior['files'].items():
        if name.startswith(source_prefix):backup['files'][prefix+name[len(source_prefix):]]=copy.deepcopy(item)
    backup['key_checkpoints'].append(dict(path=key,step=int(key.split('-')[-1]),adapter_sha256=prior['files'][source_prefix+'adapter_model.safetensors']['sha256']))
files=copy.deepcopy(backup['files']);files.pop('.gitattributes')
final_spec_hash='f'*64
files['evidence/corrective_pipeline_spec.json']=dict(bytes=100,sha256=m.PIPELINE_SPEC_SHA256)
files['evidence/evaluation_spec.json']=dict(bytes=100,sha256=final_spec_hash)
files['adapter/adapter_model.safetensors']=copy.deepcopy(files[source_prefix+'adapter_model.safetensors'])
files['model-00001-of-00001.safetensors']=dict(bytes=100,sha256='a'*64)
manifest=dict(schema='anvil-sealed-final-release-v1',repository=m.REPOSITORY,quality_review_accepted=True,
              final_spec_sha256=final_spec_hash,files=files)
identity=dict(bytes=100,sha256='b'*64)
final=dict(state='complete_remote_verified',repository=m.REPOSITORY,revision='c'*40,manifest_sha256=identity['sha256'],
           files={**files,'release_manifest.json':identity})
ckpt=pipeline['key_checkpoint_sources']['balanced_mopd/checkpoint-512']
evaluation=dict(checkpoint=ckpt,training_run=str(Path(ckpt).parent),model_files={'model-00001-of-00001.safetensors':'a'*64})
checks=[]
def accept():
    expected=m.inventory(manifest,identity,final,backup)
    assert set(expected)==set(final['files'])|{'.gitattributes'}
    m.bind_final_model(manifest,pipeline,m.PIPELINE_SPEC_SHA256,evaluation,final_spec_hash)
accept();checks.append('complete_19_checkpoint_fixture_passes')
def reject(name,edit):
    data=copy.deepcopy((manifest,identity,final,backup,pipeline,evaluation))
    edit(*data)
    try:
        m.inventory(*data[:4]);m.bind_final_model(data[0],data[4],m.PIPELINE_SPEC_SHA256,data[5],final_spec_hash)
    except (ValueError,KeyError):checks.append(name)
    else:raise AssertionError('Invalid release accepted: '+name)
reject('missing_nineteenth_checkpoint_rejected',lambda a,i,f,b,p,e:b['key_checkpoints'].pop())
reject('missing_rng_state_rejected',lambda a,i,f,b,p,e:b['files'].pop('checkpoints/balanced_mopd/checkpoint-512/rng_state_31.pth'))
reject('changed_checkpoint_after_backup_rejected',lambda a,i,f,b,p,e:b['files']['checkpoints/balanced_mopd/checkpoint-512/optimizer.pt'].update(sha256='d'*64))
reject('unaccepted_release_rejected',lambda a,i,f,b,p,e:a.update(quality_review_accepted=False))
reject('wrong_final_stage_rejected',lambda a,i,f,b,p,e:e.update(checkpoint=p['key_checkpoint_sources']['balanced_mopd/checkpoint-256']))
reject('changed_merged_model_rejected',lambda a,i,f,b,p,e:e['model_files'].update({'model-00001-of-00001.safetensors':'e'*64}))
reject('changed_packaged_pipeline_rejected',lambda a,i,f,b,p,e:a['files']['evidence/corrective_pipeline_spec.json'].update(sha256='1'*64))
reject('unexpected_old_remote_payload_rejected',lambda a,i,f,b,p,e:b['files'].update({'unreviewed.bin':dict(bytes=1,sha256='0'*64)}))
receipt=dict(state='passed',checks=checks,cpu_only=True,synthetic_contract_fixtures=True,
             final_release_verified=False,training_quality_assessed=False,completed=time.time(),
             verifier_sha256=m.sha(ROOT/'verify_entire_balanced_hub.py'),script_sha256=m.sha(__file__))
(ROOT/'balanced_release_contract_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
