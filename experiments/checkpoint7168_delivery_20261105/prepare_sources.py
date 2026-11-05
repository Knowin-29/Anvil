from pathlib import Path
import shutil,hashlib,json,py_compile
C=Path(__file__).resolve().parent.parent;Z=C/'checkpoint7168_delivery_20261105';H=C/'final_delivery_checkpoint7168_v1';OLD=C/'final_delivery_visual_geometry_v1';H.mkdir(exist_ok=False)
for name in ['common.py','merge.py','prepare.py','node.py','configs.py','native_worker.py','public_worker.py','score.py']:
 shutil.copyfile(OLD/name,H/name)
contract='''def formal_training(run, full_checkpoint=True):
    """Validate the user's selected saved state without asserting planned-stage completion."""
    run=Path(run).resolve();pipeline=C/'visual_geometry_candidate_20261104'
    selected=C/'checkpoint7168_delivery_20261105/selection.json'
    choice=read(selected);checkpoint=Path(choice['checkpoint']).resolve()
    require(run==pipeline/'runs/repair' and checkpoint==run/'checkpoint-7168','Different selected checkpoint')
    protocol=read(run/'protocol.json');spec=read(pipeline/'spec.json')
    require(sha(pipeline/'spec.json')==choice['pipeline_spec_sha256']==protocol['spec_sha256'],'Training identity changed')
    require(sha(checkpoint/'checkpoint_complete.json')==choice['seal_sha256'],'Selected state changed')
    for file,digest in spec['frozen_files'].items():require(sha(file)==digest,'Training source changed')
    require(protocol['stage']=='repair' and protocol['scope']=='all_11_native_forms' and not protocol['smoke'],'Unexpected training scope')
    require(protocol['spatial_pool']==4756000 and protocol['sft_pool_already_covered']==5220000,'Changed dataset ancestry')
    require(protocol['initial']==spec['initial']['adapter'] and protocol['initial_adapter_sha256']==spec['initial']['adapter_sha256'],'Broken corrective ancestry')
    require(sha(Path(protocol['initial'])/'adapter_model.safetensors')==protocol['initial_adapter_sha256'],'Changed initial adapter')
    binding=spec['continuation_parent']['lineage'];check_binding(binding);ancestor=read(binding['path'])
    require(ancestor['state']=='complete','Parent MOPD lineage incomplete');initialization_transform(spec,ancestor)
    receipt=verify(checkpoint,32) if full_checkpoint else read(checkpoint/'checkpoint_complete.json')
    state=read(checkpoint/'trainer_state.json');audit=read(choice['integrity']['path']);check_binding(choice['integrity'])
    require(receipt['global_step']==state['global_step']==audit['step']==7168,'Selected step changed')
    require(state['max_steps']==8192 and state['stage']=='repair' and state['final_mopd'] is False,'Incorrect stage identity')
    require(audit['all_file_hashes_verified'] and audit['all32_rng_files_loadable'] and audit['optimizer_step']==7168,'Incomplete integrity audit')
    require(len(state['log_history'])==7168 and state['source_exposures']==audit['committed_exposure_prefix']['source_exposures'],'Exposure prefix differs')
    require(sum(state['source_exposures'].values())==7168*64,'Incorrect actual training coverage')
    complete=dict(stage='user_selected_repair_checkpoint',checkpoint=str(checkpoint),steps=7168,
        planned_stage_steps=8192,stage_plan_completed=False,smoke=False,final_mopd=False,
        source_exposures=state['source_exposures'],adapter_sha256=receipt['files']['adapter_model.safetensors']['sha256'],spec_sha256=choice['pipeline_spec_sha256'])
    return complete,protocol,checkpoint,receipt
'''
p=H/'common.py';s=p.read_text();start=s.index('def formal_training(');end=s.index('\ndef load_spec(',start);s=s[:start]+contract+'\n'+s[end:];p.write_text(s)
p=H/'prepare.py';s=p.read_text().replace('Seal the exact completed MOPD model','Seal the exact user-selected model').replace("'training_complete':args.training_run/'complete.json'", "'selected_checkpoint':C/'checkpoint7168_delivery_20261105/selection.json'")
s=s.replace("spec = dict(schema='anvil-final-evaluation-spec-v1',", "spec = dict(schema='anvil-final-evaluation-spec-v1', selected_training_stage='repair', selected_step=7168, final_mopd=False,")
p.write_text(s)
p=H/'merge.py';s=p.read_text().replace('"formal_mopd_protocol_sha256":', '"training_protocol_sha256":');s=s.replace('"representation": mopd_protocol["representation"]}', '"representation": mopd_protocol["representation"], "selected_training_stage":"repair", "selected_step":7168, "final_mopd":False}');p.write_text(s)
p=H/'node.py';s=p.read_text().replace("'final_mopd'", "'visual_repair_7168'");p.write_text(s)
p=H/'native_worker.py';s=p.read_text().replace('"final_mopd": True','"final_mopd": False, "selected_training_stage": "repair", "selected_step": 7168');p.write_text(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
V=C/'visual_geometry_candidate_20261104';check=V/'runs/repair/checkpoint-7168';audit=C/'visual_resume7168_20261105/resume7168_verification.json'
selection=dict(schema='anvil-user-selected-checkpoint-v1',checkpoint=str(check),stage='repair',step=7168,planned_stage_steps=8192,stage_plan_completed=False,final_mopd=False,pipeline_spec_sha256=sha(V/'spec.json'),seal_sha256=sha(check/'checkpoint_complete.json'),integrity=dict(path=str(audit),sha256=sha(audit)),parent_mopd_retained=True,eval_gpus=8,full_native_records=580000,full_public_records=7366,full_general_records=10105,publish_checkpoint_now=True,final_acceptance_relative_degradation_max=.1)
(Z/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
for p in H.glob('*.py'):py_compile.compile(str(p),doraise=True)
(Z/'source_derivation.json').write_text(json.dumps({p.name:dict(original_sha256=sha(OLD/p.name),derived_sha256=sha(p)) for p in H.glob('*.py')},indent=2)+'\n')
print(H)
