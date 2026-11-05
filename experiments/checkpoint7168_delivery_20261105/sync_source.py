from pathlib import Path
import ast,base64,hashlib,json,os,re,shutil,subprocess,time,zipfile
W=Path(__file__).resolve().parent;C=W.parent;repo=Path('/Knowin/demo/weilinruan/Anvil/01_Code');dest=repo/'experiments/checkpoint7168_delivery_20261105'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def run(*args,env=None):
 p=subprocess.run(list(args),capture_output=True,text=True,env=env,timeout=180)
 if p.returncode:raise RuntimeError('Command failed: '+args[0])
 return p.stdout.strip()
assert run('git','-C',str(repo),'branch','--show-current')=='main'
assert not run('git','-C',str(repo),'status','--porcelain')
mapping={p:dest/p.name for p in W.glob('*.py')}
for name in ['selection.json','source_derivation.json','release_source_derivation.json','release_layout_validation.json','cleanup_guard_validation.json','user_selection.json','evaluation_spec.json','OPERATIONS.md']:
 mapping[W/name]=dest/name
h=C/'final_delivery_checkpoint7168_v1'
for p in h.glob('*.py'):mapping[p]=repo/'experiments/final_delivery_checkpoint7168_v1'/p.name
mapping[W/'evaluation/node_0_cli.yaml']=dest/'evaluation_cli.yaml'
manifest=[]
for source,target in mapping.items():
 text=source.read_text();assert not re.search(r'(?:ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})',text)
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);manifest.append(dict(path=str(target.relative_to(repo)),source=str(source),sha256=sha(source)))
save(dest/'SOURCE_MANIFEST.json',dict(schema='anvil-selected-checkpoint-delivery-source-v1',files=manifest,created=time.time(),weights_included=False))
run('git','-C',str(repo),'add','--',str(dest.relative_to(repo)),'experiments/final_delivery_checkpoint7168_v1')
run('git','-C',str(repo),'commit','-m','Deliver selected checkpoint7168 with full eight-GPU evaluation and staged Hub release')
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
save(W/'source_git_sync.json',dict(branch='main',commit=commit,remote=remote,verified=time.time(),credentials_persisted=False,files=len(mapping)))
bundles=C/'source_bundles_20261104';bundles.mkdir(exist_ok=True);bundle=bundles/('Anvil_main_'+commit[:12]+'.zip')
run('git','-C',str(repo),'archive','--format=zip','--prefix=Anvil/','--output='+str(bundle),commit)
with zipfile.ZipFile(bundle) as z:assert z.testzip() is None;entries=len(z.namelist())
link=repo.parent/'03_Delivery/Anvil_Source_Clean.zip';assert link.is_symlink();temporary=link.with_suffix('.pending');temporary.symlink_to(bundle);temporary.replace(link)
save(W/'source_bundle.json',dict(path=str(bundle),sha256=sha(bundle),bytes=bundle.stat().st_size,entries=entries,commit=commit,link=str(link)))
print(json.dumps(dict(commit=commit,files=len(mapping),bundle=str(bundle))))
