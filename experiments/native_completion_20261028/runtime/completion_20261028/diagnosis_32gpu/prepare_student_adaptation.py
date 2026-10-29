"""Seal explicit initial-checkpoint choice after reviewing candidate scores."""
import argparse,hashlib,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--initial-run',type=Path,required=True);ap.add_argument('--pose-mode',choices=['hard','angular'],required=True)
 ap.add_argument('--steps',type=int,choices=[512,1024],default=1024);ap.add_argument('--rationale',required=True);args=ap.parse_args()
 cpu=ROOT/'student_adaptation_cpu_validation.json';validation=json.loads(cpu.read_text());assert validation['state']=='passed_cpu'
 for path,digest in validation['files_sha256'].items():assert sha(path)==digest
 complete=args.initial_run.resolve()/'complete.json';receipt=json.loads(complete.read_text());assert not receipt['smoke'] and receipt['state']=='repair_complete_requires_evaluation'
 checkpoint=Path(receipt['checkpoint']);assert sha(checkpoint/'adapter_model.safetensors')==receipt['adapter_sha256']
 names=['student_adaptation_pool.py','student_adaptation_train.py','student_adaptation_node.py','student_adaptation_benchmark_worker.py','merge_student_adaptation.py','score_student_adaptation_public.py','score_student_adaptation_A.py','dof_pose_loss.py','dof_codebook_interface.py','dof_rotation_codebook.json','dof_codec_interface.py','dof_codec_prototype.py','dof_codebook_eval.py','dof_pose_node.py','spatial_train.py','prism_expanded_train.py','prepare_dof_rotation_codebook.py','audit_targets.py','augmented_data_scale_reconciliation_v2.json']
 spec=dict(schema='anvil-student-representation-adaptation-v1',world=8,global_batch=32,steps=args.steps,learning_rate=1e-5,pose_mode=args.pose_mode,
  initial=dict(checkpoint=str(checkpoint),adapter_sha256=receipt['adapter_sha256'],completion_receipt=str(complete),completion_receipt_sha256=sha(complete)),
  selection_rationale=args.rationale,cpu_validation_sha256=sha(cpu),final_mopd=False,
  frozen_files={str(ROOT/name):sha(ROOT/name) for name in names})
 target=ROOT/'student_adaptation_spec.json';assert not target.exists(),'Never replace a sealed live spec'
 target.write_text(json.dumps(spec,indent=2)+'\n')
 config=yaml.safe_load((ROOT/'prism_expanded_cli.yaml').read_text());config['TaskName']='task_anvil_student_representation_adaptation_A800x8'
 config['Description']='One student DoF representation adaptation with original native, broadQA and spatial replay; true optimizer resume and full public/general/A evaluation; not formal MOPD'
 config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+str(ROOT.parent.parent/'.venv-train/bin/python')+' -u '+str(ROOT/'student_adaptation_node.py')+' --spec '+str(target)
 (ROOT/'student_adaptation_cli.yaml').write_text(yaml.safe_dump(config,sort_keys=False));print(target)

if __name__=='__main__':main()
