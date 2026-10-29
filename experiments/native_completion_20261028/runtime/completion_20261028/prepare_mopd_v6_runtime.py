"""Seal exact candidate checkpoints for real on-policy resume/throughput measurement.

Preparing or passing this execution task never fabricates teacher qualification.
"""
import argparse,hashlib,json,sys,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent;D=ROOT/'diagnosis_32gpu';R=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'Anvil'),str(R),str(R/'cloud_a800'),str(ROOT/'final_delivery')]
from checkpoints import save,verify
from initialization_lineage import trace
from mopd_representation import Representation
from trains.flywheel.native_training import validate_training_record
from native_mopd_v6 import ReplayPool

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def repaired(run):
 run=Path(run).resolve();receipt=json.loads((run/'complete.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
 assert receipt['state']=='repair_complete_requires_evaluation' and receipt['smoke'] is False
 assert receipt['step']==protocol['steps'];checkpoint=Path(receipt['checkpoint']);assert checkpoint.parent==run
 assert sha(checkpoint/'adapter_model.safetensors')==receipt['adapter_sha256']
 verify(checkpoint,protocol['world'])
 return dict(adapter=str(checkpoint),adapter_sha256=receipt['adapter_sha256'],completion_receipt=str(run/'complete.json'),completion_sha256=sha(run/'complete.json'))

def validate_fixtures(fixtures):
 rows=[json.loads(l) for l in Path(fixtures).open()]
 assert len(rows)==5 and {r['gsa_form'] for r in rows}=={'QA','bbox','direction','dof','point'}
 pool=ReplayPool(True)
 for row in rows:
  if row['gsa_family']=='QA':
   name=row['gsa_meta']['binding']['source_dataset'];matches=[r for r in pool.qa[name] if r['canonical_uid']==row['canonical_uid']]
   assert len(matches)==1 and matches[0]==row,'QA fixture differs from sealed reference pool'
  elif row.get('anvil_replay_source'):
   matches=[r for r in pool.spatial[row['anvil_replay_source']] if r['canonical_uid']==row['canonical_uid']]
   assert len(matches)==1 and matches[0]==row,'Spatial reference fixture differs from sealed pool'
  else:validate_training_record(row)
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--student-run',type=Path,required=True);ap.add_argument('--a-run',type=Path,required=True);ap.add_argument('--g-run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 cpu=ROOT/'mopd_v6_cpu_validation.json';validation=json.loads(cpu.read_text());assert validation['state']=='passed_cpu'
 fixture_receipt=ROOT/'mopd_v6_fixture_validation_v2.json';fixture_check=json.loads(fixture_receipt.read_text())
 assert fixture_check['state']=='passed_cpu'
 for path,digest in fixture_check['files_sha256'].items():assert sha(path)==digest,path
 for path,digest in validation['files_sha256'].items():assert sha(path)==digest,path
 student=repaired(args.student_run);teachers={'G':repaired(args.g_run),'A':repaired(args.a_run)}
 original=R.parent/'a800_repair_20261020/pipeline_status.json';s=json.loads(original.read_text())['checkpoints']['S']
 assert sha(Path(s['checkpoint'])/'adapter_model.safetensors')==s['adapter_sha256']
 teachers['S']=dict(adapter=s['checkpoint'],adapter_sha256=s['adapter_sha256'],completion_receipt=str(original),completion_sha256=sha(original))
 representation=Representation('rotation128');interfaces=representation.validate_adapters({k:v['adapter'] for k,v in teachers.items()},student['adapter'])
 assert interfaces['student']['student_requires_representation_adaptation'] is False
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 lineage=trace(student['adapter']);save(out/'initialization_lineage.json',lineage)
 fixtures=D/'mopd_v5_runtime_fixtures.jsonl';rows=validate_fixtures(fixtures)
 shutil.copyfile(fixtures,out/'fixtures.jsonl')
 binding=dict(schema='anvil-execution-only-checkpoint-binding-v1',student=student,teachers=teachers,
  teachers_qualified=False,formal_mopd=False,representation_sha256=representation.digest,
  note='Exact prospective candidate weights; no quality eligibility is created by this execution probe.')
 save(out/'bindings.json',binding)
 files=[Path(__file__),D/'mopd_v6_runtime_node.py',ROOT/'native_mopd_v6.py',ROOT/'mopd_lineage.py',ROOT/'mopd_representation.py',ROOT/'completion_policy.py',D/'repair_node.py',D/'repair_train.py',R/'training_data.py',R/'eval_core.py',R/'cloud_a800/checkpoints.py',R/'cloud_a800/memory_loss.py',cpu,fixture_receipt,out/'bindings.json',out/'fixtures.jsonl',out/'initialization_lineage.json']
 files+=list((ROOT/'Anvil/trains/mopd').rglob('*.py'))+list((ROOT/'Anvil/trains/flywheel').rglob('*.py'))+[Path(p) for p in representation.descriptor['files_sha256']]
 for item in [student,*teachers.values()]:files.append(Path(item['completion_receipt']))
 experiment=dict(schema='anvil-exact-candidate-v6-runtime-v1',runtime_bindings=str(out/'bindings.json'),fixtures=str(out/'fixtures.jsonl'),
  initialization_lineage=str(out/'initialization_lineage.json'),representation='rotation128',runtime_output=str(out/'resume'),probe_output=str(out/'probe'),
  world=8,probe_steps=32,teacher_qualified=False,formal_mopd=False,
  frozen_files={str(path):sha(path) for path in sorted(set(files))})
 save(out/'experiment.json',experiment)
 config=yaml.safe_load((D/'prism_expanded_cli.yaml').read_text());config['TaskName']='task_anvil_MOPDv6_exact_candidate_runtime_A800x8'
 config['Description']='Actual optimizer2to3resume and32full-budget on-policy updates with exact candidate weights; not formal MOPD or teacher eligibility'
 config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+str(R/'.venv-train/bin/python')+' -u '+str(D/'mopd_v6_runtime_node.py')+' --experiment '+str(out/'experiment.json')
 (out/'cli.yaml').write_text(yaml.safe_dump(config,sort_keys=False));print(out/'cli.yaml')

if __name__=='__main__':main()
