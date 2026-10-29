"""Bind real student ancestry and exact execution evidence without quality invention."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def inventory():
 path=ROOT/'diagnosis_32gpu/augmented_data_scale_reconciliation_v2.json';value=json.loads(path.read_text())
 assert value['original_train_records']==5220000 and value['available_training_records']==5316931
 for file,digest in value['bindings'].items():assert sha(file)==digest
 return dict(path=str(path),sha256=sha(path),counts=value)

def bind_lineage(path,student,required):
 if path is None:
  assert not required,'Formal training requires audited exact student ancestry'
  return None
 path=Path(path);value=json.loads(path.read_text())
 assert value['schema']=='anvil-initialization-lineage-v1' and value['state']=='complete_checkpoint_ancestry'
 assert Path(value['checkpoint']).resolve()==Path(student).resolve(),'Lineage belongs to another student'
 assert value['adapter_sha256']==sha(Path(student)/'adapter_model.safetensors')
 assert value['initial_sft']['training_records']==5220000
 assert value['other_experiment_arms_counted_as_final_training'] is False
 for item in value['evidence']:assert sha(item['path'])==item['sha256']
 return dict(path=str(path.resolve()),sha256=sha(path),checkpoint=value['checkpoint'],adapter_sha256=value['adapter_sha256'],
  post_sft_logged_exposures=value['post_sft_logged_exposures'],unique_supplemental_student_records=value['unique_supplemental_student_records'],
  warm_start_disclosed=True,full_final_lineage_required_separately=True)

def validate_runtime(report,trainer,student,teachers):
 assert report['trainer_code_sha256']==sha(trainer),'Runtime proof is for another trainer version'
 assert report['student_adapter_sha256']==sha(Path(student)/'adapter_model.safetensors'),'Runtime proof is for another student'
 assert report['teacher_adapter_sha256']=={f:sha(Path(p)/'adapter_model.safetensors') for f,p in teachers.items()},'Runtime proof is for other teachers'
