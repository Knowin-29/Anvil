"""Snapshot and publish only the visual campaign source; auth remains transient."""
import ast,base64,hashlib,json,os,re,shutil,subprocess,time,zipfile
from pathlib import Path
C=Path(__file__).resolve().parent.parent;V=C/'visual_geometry_candidate_20261104';H=C/'final_delivery_visual_geometry_v1'
repo=Path('/Knowin/demo/weilinruan/Anvil/01_Code');dest=repo/'experiments/visual_geometry_candidate_20261104'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def run(*args,env=None):
 p=subprocess.run(list(args),capture_output=True,text=True,env=env,timeout=180)
 if p.returncode:raise RuntimeError('Command failed: '+args[0])
 return p.stdout.strip()
assert run('git','-C',str(repo),'branch','--show-current')=='main'
assert not run('git','-C',str(repo),'status','--porcelain')
files=[*V.glob('*.py')]
files += [V/n for n in ['cpu_validation.json','optimizer_cpu_tests.json','merge_cpu_validation.json','trainer_interface_validation.json','trainer_source_derivation.json','delivery_source_derivation.json','candidate_notes.json','protocol.json','spec.json','candidate_settings.json','submit_policy.json','backup_policy.json','cli.yaml','public_trigger.json','schedules.json']]
files += [V/'initialization/initialization.json',V/'initialization/adapter_config.json']
mapping={p:dest/p.relative_to(V) for p in files}
mapping.update({p:dest/'delivery'/p.relative_to(H) for p in H.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.json'}})
for name in ['visual_continuation_controller.py','visual_final_handoff.py','backup_visual_checkpoints.py']:mapping[C/name]=dest/'controllers'/name
mapping[C/'coordinate_delivery_control_v2_20261104/quality_review.json']=dest/'parent_quality_review.json'
mapping[C/'delivery_runs/visual_geometry_20261104/policy.json']=dest/'delivery_policy.json'
for folder,label in [(C/'visual_delivery_control_20261104','release_control'),(C/'hub_presentation_visual_20261104','presentation'),(C/'visual_native_score_20261104','scoring')]:
 for source in folder.iterdir():
  if source.is_file() and source.suffix in {'.py','.json','.md'} and source.name not in {'status.json','worker_launch.json','workers_launched.json','first_checkpoint_status.json','first_checkpoint_failure.json','model_status.json','model_failure.json'}:mapping[source]=dest/label/source.name
for name in ['preupload_visual_merged_blobs.py','verify_entire_visual_hub.py']:mapping[C/'verification_20261101'/name]=dest/'controllers'/name
manifest=[]
for source,target in mapping.items():
 text=source.read_text();assert not re.search(r'(?:ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})',text),'Credential-like source contents'
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);manifest.append(dict(path=str(target.relative_to(dest)),source=str(source),sha256=sha(source)))
save(dest/'SOURCE_MANIFEST.json',dict(schema='anvil-visual-campaign-source-snapshot-v1',created=time.time(),files=manifest,spec_sha256=sha(V/'spec.json'),trained_weights_included=False))
run('git','-C',str(repo),'add','--',str(dest.relative_to(repo)))
run('git','-C',str(repo),'commit','-m','Track production visual checkpoint verification and campaign handoff')
helper=C/'verification_20261101/sync_format_source_git.py';tree=ast.parse(helper.read_text());nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in {'strings','user_texts'}) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SESSION' for t in n.targets))]
ns={'Path':Path};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(helper),'exec'),ns)
token=None
for line in ns['SESSION'].open():
 for text in ns['user_texts'](json.loads(line)):
  if 'Knowin-29' in text and 'git token:' in text:
   m=re.search(r'git token:\s*(ghp_[A-Za-z0-9]+)',text)
   if m:token=m.group(1);break
 if token:break
assert token
transient={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
transient.update(GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='3',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('Knowin-29:'+token).encode()).decode(),GIT_CONFIG_KEY_1='credential.helper',GIT_CONFIG_VALUE_1='',GIT_CONFIG_KEY_2='credential.interactive',GIT_CONFIG_VALUE_2='false')
remote='https://github.com/Knowin-29/Anvil.git';run('git','-C',str(repo),'push',remote,'HEAD:refs/heads/main',env=transient)
commit=run('git','-C',str(repo),'rev-parse','HEAD');assert run('git','-C',str(repo),'ls-remote',remote,'refs/heads/main',env=transient).split()[0]==commit
save(V/'monitor_source_git_sync.json',dict(branch='main',commit=commit,remote=remote,verified=time.time(),credentials_persisted=False,files=len(mapping)))
bundles=C/'source_bundles_20261104';bundles.mkdir(exist_ok=True);bundle=bundles/('Anvil_main_'+commit[:12]+'.zip')
run('git','-C',str(repo),'archive','--format=zip','--prefix=Anvil/','--output='+str(bundle),commit)
with zipfile.ZipFile(bundle) as z:assert z.testzip() is None;entries=len(z.namelist())
link=repo.parent/'03_Delivery/Anvil_Source_Clean.zip';assert link.is_symlink();temporary=link.with_suffix('.pending');temporary.symlink_to(bundle);temporary.replace(link)
save(V/'monitor_source_bundle.json',dict(path=str(bundle),sha256=sha(bundle),bytes=bundle.stat().st_size,entries=entries,commit=commit,link=str(link)))
print(json.dumps(dict(commit=commit,files=len(mapping),bundle=str(bundle))))
