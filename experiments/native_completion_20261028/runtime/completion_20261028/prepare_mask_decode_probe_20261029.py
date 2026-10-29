"""Prepare immutable paired inference probe; no GPU submission or training."""
from pathlib import Path
import sys,json,hashlib,shlex,time,copy
C=Path(__file__).resolve().parent;D=C/'diagnosis_32gpu';R=C.parent;O=C/'mask_decode_probe_20261029'
O.mkdir(exist_ok=True)
assert not (O/'spec.json').exists()
sys.path[:0]=[str(C),str(D),str(R),str(R/'cloud_a800'),str(C/'eval_code/score_code')]
import torch,yaml
from training_data import processor,encode,EMPTY_THINK
from checkpoints import sha,save,verify
import mask_decode_probe_20261029 as module
import eval_core
rows=[json.loads(line) for line in (D/'backend_inputs_G.jsonl').open()]
chosen=[]
for split in ['train','heldout']:
    group=[x for x in rows if x['split']==split]
    group.sort(key=lambda x:hashlib.sha256(('decode-probe-v1:'+x['row']['id']).encode()).hexdigest())
    chosen.extend(group[:16])
assert len({x['row']['id'] for x in chosen})==32
inputs=O/'inputs.jsonl';inputs.write_text(''.join(json.dumps(x)+'\n' for x in chosen))
p=processor();torch.set_num_threads(4);verified=[]
for item in chosen:
    row=item['row'];assert row['native_form']=='mask' and sha(row['images'][0])==item['image_sha256']
    shell=dict(id=row['id'],messages=[row['messages'][0],dict(role='assistant',content='')],images=row['images'],gsa_meta=dict(split='train'))
    empty=encode(p,shell);fullrow=copy.deepcopy(shell);fullrow['messages'][1]=row['messages'][1];full=encode(p,fullrow)
    lengths=[]
    for data in [empty,full]:lengths.append(int((data['labels'][0]!=-100).nonzero()[0])+len(p.tokenizer.encode(EMPTY_THINK,add_special_tokens=False)))
    assert lengths[0]==lengths[1]
    for key in ['input_ids','attention_mask','mm_token_type_ids']:
        if key in empty:assert torch.equal(empty[key][:,:lengths[0]],full[key][:,:lengths[1]])
    assert torch.equal(empty['pixel_values'],full['pixel_values']);verified.append(row['id'])
for tokens,want in [([1]*511,False),([1]*512,True),(list(range(1024)),False)]:
    guard=module.CycleStop(3)
    assert bool(guard(torch.tensor([[4,5,6]+tokens]),None)[0])==want
models={}
for name,ck in [('initial',D/'student_adaptation_run/main/checkpoint-1024'),('step512',C/'formal_mopd/runs/candidate1_20261029/training/checkpoint-512')]:
    receipt=verify(ck,8 if name=='initial' else 16)
    models[name]=dict(adapter=str(ck),adapter_sha256=receipt['files']['adapter_model.safetensors']['sha256'])
validation=dict(state='passed',actual_encode_prompt_and_pixels_match_full_reference_encoding=len(verified),rows=32,models_sealed_files_verified=True,original_scorer_unchanged=True,worker_module_imported=True,cycle_detector_boundary_cases=3,created=time.time())
save(O/'cpu_validation.json',validation)
files=[Path(__file__),Path(module.__file__),R/'training_data.py',Path(eval_core.__file__),R/'cloud_a800/checkpoints.py',D/'retention_node.py',D/'repair_node.py',D/'backend_inputs_G.jsonl',inputs,O/'cpu_validation.json']
files+=list((C/'eval_code').rglob('*.py'))
tasks={f'{name}_{mode}_{shard}':dict(model=name,mode=mode,shard=shard) for name in models for mode in ['greedy','sample'] for shard in range(2)}
spec=dict(schema='anvil-paired-mask-decoding-probe-v1',inputs=str(inputs),models=models,tasks=tasks,files={str(f):sha(f) for f in sorted(set(files))},planned_decodes=128,unique_examples=32,training_performed=False,final_validation=False,limitations=['Adaptive diagnostic after observing formal512 mask regression; not a blind benchmark.','Sixteen training and sixteen heldout examples reused from the existing fixed backend diagnostic; these are not the same as all8192 committed MOPD rollouts.','One seeded sample per example per checkpoint; sampling variance and dataset effects cannot be excluded.','Both modes use the same HF adapter runtime; this isolates decoding settings here, not general backend parity.'])
save(O/'spec.json',spec);module.load_spec(O/'spec.json',sha(O/'spec.json'))
conf=yaml.safe_load((D/'backend_v2_cli.yaml').read_text());conf['TaskName']='task_anvil_mask_decode_paired_A800x8';conf['Description']='Read-only32fixed masks x2checkpoints x2decoders; isolate sampling versus greedy failure; no training';conf['ActiveDeadlineSeconds']=3600
conf['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(R/'.venv-train/bin/python'),'-u',str(Path(module.__file__)),'--spec',str(O/'spec.json'),'--spec-sha256',sha(O/'spec.json'),'--node'])
(O/'cli.yaml').write_text(yaml.safe_dump(conf,sort_keys=False))
print(json.dumps(validation,indent=2))
