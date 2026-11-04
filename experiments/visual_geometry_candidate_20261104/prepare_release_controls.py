"""Derive new release controls without changing frozen training/evaluation sources."""
from pathlib import Path
import ast,hashlib,json,re,shutil,time
C=Path(__file__).resolve().parent.parent;V=C/'visual_geometry_candidate_20261104';A=C/'verification_20261101'
J=C/'visual_delivery_control_20261104';N=C/'hub_presentation_visual_20261104';S=C/'visual_native_score_20261104'
for p in [J,N,S]:p.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
spec_sha=sha(V/'spec.json');spec=read(V/'spec.json')
replacements=[('delivery_runs/geometry_objective_v2_20261104','delivery_runs/visual_geometry_20261104'),('geometry_objective_v2_20261104','visual_geometry_candidate_20261104'),('final_delivery_geometry_objective_v2','final_delivery_visual_geometry_v1'),('coordinate_delivery_control_v2_20261104','visual_delivery_control_20261104'),('coordinate_checkpoint_backup_v2_20261104','visual_checkpoint_backup_20261104'),('coordinate_v2_merged_blob_preupload','visual_merged_blob_preupload'),('preupload_coordinate_merged_blobs_v2.py','preupload_visual_merged_blobs.py'),('verify_entire_coordinate_hub_v2.py','verify_entire_visual_hub.py'),('backup_coordinate_checkpoints_v2.py','backup_visual_checkpoints.py'),('hub_presentation_v2_20261104','hub_presentation_visual_20261104'),('release_policy_v2.json','release_policy.json'),('01b438ee02b653394ff744979f83ddfc2ca80e396a6eee1f2198b1cb7519356c',spec_sha)]
def convert(text):
 for a,b in replacements:text=text.replace(a,b)
 text=re.sub(r'\b35\b','39',text).replace('and35_','and39_').replace('all35','all39').replace('thirty-five','thirty-nine')
 return text
files={}
def derived(source,destination,transform=convert):
 text=transform(source.read_text());ast.parse(text)
 with destination.open('x') as f:f.write(text)
 files[str(destination)]=dict(source=str(source),source_sha256=sha(source),sha256=sha(destination))
derived(C/'coordinate_delivery_control_v2_20261104/quality_gate.py',J/'quality_gate.py',lambda s:s)
derived(C/'coordinate_delivery_control_v2_20261104/finish_delivery.py',J/'finish_delivery.py')
derived(A/'preupload_coordinate_merged_blobs_v2.py',A/'preupload_visual_merged_blobs.py',lambda s:convert(s).replace('coordinate_mopd/checkpoint-256','visual_mopd/checkpoint-256'))
def verifier(s):
 s=convert(s);s=re.sub(r'^KEYS = .*$', 'KEYS = '+repr(set(spec['key_checkpoints'])),s,flags=re.M)
 s=s.replace("pipeline['key_checkpoint_sources']['coordinate_mopd/checkpoint-256']","pipeline['key_checkpoint_sources']['visual_mopd/checkpoint-256']").replace("manifest['files']['checkpoints/coordinate_mopd/checkpoint-256/adapter_model.safetensors']","manifest['files']['checkpoints/visual_mopd/checkpoint-256/adapter_model.safetensors']")
 return s
derived(A/'verify_entire_coordinate_hub_v2.py',A/'verify_entire_visual_hub.py',verifier)
# Use the tested quota ledger handling, retaining the OLD ledger's actual scope.
def cleanup(s):
 s=convert(s).replace("RESUME = Path(__file__).resolve().parent","RESUME = C / 'coordinate_resume_storage_20261104'")
 s=s.replace("path.parent.parent==PIPELINE/'runs/repair'","path.parent.parent==C/'geometry_objective_v2_20261104/runs/repair'")
 s=s.replace("    require(by_id[read(PIPELINE / 'submission.json')['response']['Id']]['status'] == 'Killed', 'Original failed task not stopped')\n    identities = [read(RESUME / 'submission.json')['response']['Id']]","    identities = [read(PIPELINE / 'submission.json')['response']['Id']]")
 return s
derived(C/'coordinate_resume_storage_20261104/cleanup_verified_weights.py',J/'cleanup_verified_weights.py',cleanup)
# Keep the exact pre-sealed score definitions; only route the waiter to this model.
ORIG=C/'anvil_eval_score_v1_1_20261104'
for name in ['score_records.py','PROTOCOL.md','test_score_records.py','protocol_seal.json']:shutil.copyfile(ORIG/name,S/name)
derived(ORIG/'wait_and_score.py',S/'wait_and_score.py')
def presentation(s):
 s=convert(s).replace("'coordinate_repair/checkpoint-8192': ('geometry-repair', 'Geometry-focused training')","'visual_repair/checkpoint-8192': ('visual-geometry', 'Visual geometry adaptation')").replace("'coordinate_mopd/checkpoint-256': ('final-mopd', 'Final MOPD')","'visual_mopd/checkpoint-256': ('final-mopd', 'Final MOPD')")
 return s
derived(C/'hub_presentation_v2_20261104/model_presentation.py',N/'model_presentation.py',presentation)
def finalizer(s):
 s=convert(s).replace("score_path=C/'anvil_eval_score_v1_1_20261104/final_native_score.json'","score_path=C/'visual_native_score_20261104/final_native_score.json'")
 s=s.replace("(C/'anvil_eval_score_v1_1_20261104/final_native_score.json').read_bytes()","(C/'visual_native_score_20261104/final_native_score.json').read_bytes()")
 s=s.replace('Native paper-aligned tolerance not verified','Native versioned score does not meet the user-designated threshold')
 return s
derived(C/'hub_presentation_v2_20261104/finalize_model.py',N/'finalize_model.py',finalizer)
policy=read(C/'hub_presentation_v2_20261104/user_acceptance_policy.json');save(N/'user_acceptance_policy.json',policy)
for old,dest in [(C/'coordinate_delivery_control_v2_20261104/test_release.py',J/'test_release.py'),(C/'coordinate_resume_storage_20261104/test_cleanup_verified_weights.py',J/'test_cleanup_verified_weights.py'),(C/'hub_presentation_v2_20261104/test_model_presentation.py',N/'test_model_presentation.py')]:derived(old,dest,presentation)
# Enforce numeric user acceptance before any publication, not just the final layout.
p=J/'quality_gate.py';s=p.read_text();s=s.replace('import math','import math\nimport hashlib\nimport json\nfrom pathlib import Path')
needle="    require(review.get('native_quality_required') is True and"
check="""    root=Path(__file__).resolve().parent.parent
    policy_path=root/'hub_presentation_visual_20261104/user_acceptance_policy.json'
    policy=json.loads(policy_path.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    require(review.get('user_acceptance_policy_sha256')==digest(policy_path),'Different user tolerance policy')
    require(all(score>=.9*report['paper_target'][name] for name,score in report['paper_aligned_public'].items()),'Public benchmark below tolerance')
    score_path=root/'visual_native_score_20261104/final_native_score.json'
    score=json.loads(score_path.read_text())
    native=review.get('paper_native_tolerance_assessment',{})
    require(score['state']=='complete' and score['n']==580000 and score['schema']==policy['native_aggregate_protocol'],'Incomplete versioned native score')
    require(score['report_sha256']==report_sha and score['final_spec_sha256']==spec_sha,'Native score model mismatch')
    require(score['score']>=policy['native_minimum'] and native.get('accepted') is True and native.get('measured_score')==score['score'],'Native score below user tolerance')
    require(native.get('score_artifact')==str(score_path) and native.get('score_artifact_sha256')==digest(score_path),'Different native score artifact')
    sealed=root/'anvil_eval_score_v1_1_20261104'
    require(digest(sealed/'protocol_seal.json')==policy['native_protocol_seal_sha256'],'Scoring seal changed')
    seal=json.loads((sealed/'protocol_seal.json').read_text())
    require(all(digest(sealed/name)==value for name,value in seal['files'].items()),'Scoring implementation changed')
    require(score['scorer_sha256']==digest(sealed/'score_records.py') and score['protocol_sha256']==digest(sealed/'PROTOCOL.md'),'Different scoring definitions')
"""
assert needle in s;s=s.replace(needle,check+needle);ast.parse(s);p.write_text(s);files[str(p)]['sha256']=sha(p)
save(J/'derivation.json',dict(state='prepared_requires_cpu_validation',created=time.time(),files=files,scoring_definition_unchanged=True,publication_requires_explicit_review=True,source_pipeline_sha256=spec_sha))
print(json.dumps(dict(files=len(files),control=str(J),presentation=str(N),scorer=str(S))))
