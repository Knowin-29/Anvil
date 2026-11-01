"""Prepare separate post-training workers without mutating frozen sources."""
from pathlib import Path
import ast,hashlib,json
V=Path(__file__).resolve().parent;C=V.parent
P=C/'balanced_repair_20261102';digest=hashlib.sha256((P/'spec.json').read_bytes()).hexdigest()
OLD_SHA='5e3d25f7d59d2028d2c47bb4b2b433b6da0703994fad38ec464baac45a95e5bc'

def save(path,text):
    assert not path.exists(),path
    ast.parse(text,filename=str(path));path.write_text(text)

def common(s):
    return s.replace(OLD_SHA,digest).replace('focus_repair_20261101','balanced_repair_20261102').replace('final_delivery_focus_v1','final_delivery_balanced_v1').replace('focus_checkpoint_backup_20261101','balanced_checkpoint_backup_20261102')
s=common((V/'verify_focus_checkpoint.py').read_text()).replace('focus_checkpoint_integrity_verified','balanced_checkpoint_integrity_verified')
save(V/'verify_balanced_checkpoint.py',s)
s=common((C/'backup_focus_checkpoints.py').read_text())
s=s.replace("C/'targeted_checkpoint_backup_20261031/mopd_768'","C/'focus_checkpoint_backup_20261101/mopd_512'").replace('verify_focus_checkpoint.py','verify_balanced_checkpoint.py')
s=s.replace("[9:]","[14:]").replace("[:9]","[:14]")
s=s.replace("focus_repair/checkpoint-4096","balanced_repair/checkpoint-1280").replace('focus_repair/checkpoint-8192','balanced_repair/checkpoint-2560').replace('focus_mopd/','balanced_mopd/').replace("removeprefix('focus_')","removeprefix('balanced_')")
s=s.replace("eaf724a6af7f0f5e1c898ae58842de56e43c6962","3463ded9ba5fc24939206f986c6a3bafc1e776a0")
s=s.replace('focus_checkpoint_integrity_verified','balanced_checkpoint_integrity_verified').replace('intermediate-focus-checkpoint','intermediate-balanced-checkpoint')
s=s.replace('==14','==19').replace('verified_checkpoints=14','verified_checkpoints=19').replace('Focus checkpoint backup','Balanced checkpoint backup')
s=s.replace("key_records.append(dict(path=relative,step=step,adapter_sha256=seal['files']['adapter_model.safetensors']['sha256'],final_evaluated_checkpoint=False))", "key_records.append(dict(path=relative,step=step,adapter_sha256=seal['files']['adapter_model.safetensors']['sha256'],final_evaluated_checkpoint=False))\n        for record in key_records:\n            record['in_final_optimizer_ancestry']=record['path'] not in spec['retained_nonancestor_checkpoints']")
s=s.replace('This continuation inherits the completed MOPD768 weights with fresh stage optimizers.', 'The rejected focus branch scored67.6957% and is retained solely as a nonancestor experiment backup. This balanced continuation inherits the stronger completed MOPD768 weights with fresh stage optimizers; rejected-focus exposures are excluded from its model ancestry.')
save(C/'backup_balanced_checkpoints.py',s)
s=common((V/'preupload_focus_merged_blobs.py').read_text()).replace('focus_merged_blob_preupload','balanced_merged_blob_preupload').replace('checkpoints/focus_mopd/','checkpoints/balanced_mopd/')
s=s.replace('== 14','== 19').replace('all14','all19').replace('fourteen','nineteen').replace('required_backups_verified=14','required_backups_verified=19').replace('uncommitted-focus','uncommitted-balanced')
save(V/'preupload_balanced_merged_blobs.py',s)
s=common((V/'verify_entire_focus_hub.py').read_text())
s=s.replace("    'focus_mopd/checkpoint-128', 'focus_mopd/checkpoint-256', 'focus_mopd/checkpoint-512',\n}","    'focus_mopd/checkpoint-128', 'focus_mopd/checkpoint-256', 'focus_mopd/checkpoint-512',\n    'balanced_repair/checkpoint-1280', 'balanced_repair/checkpoint-2560',\n    'balanced_mopd/checkpoint-128', 'balanced_mopd/checkpoint-256', 'balanced_mopd/checkpoint-512',\n}")
s=s.replace('fourteen','nineteen').replace('== 14','== 19')
s=s.replace("['focus_mopd/checkpoint-512']","['balanced_mopd/checkpoint-512']").replace('checkpoints/focus_mopd/checkpoint-512/adapter_model','checkpoints/balanced_mopd/checkpoint-512/adapter_model').replace('completed focus MOPD512','completed balanced MOPD512')
save(V/'verify_entire_balanced_hub.py',s)
s=common((V/'validate_focus_release_contract.py').read_text()).replace('focus_hub_verifier','balanced_hub_verifier').replace('verify_entire_focus_hub.py','verify_entire_balanced_hub.py').replace('focus_mopd/','balanced_mopd/').replace('complete_14','complete_19').replace('fourteenth','nineteenth').replace('focus_release_contract_validation','balanced_release_contract_validation')
save(V/'validate_balanced_release_contract.py',s)
s=common((C/'campaign_monitor_focus.py').read_text()).replace('anvil-focus','anvil-balanced').replace('campaign_focus_status','campaign_balanced_status').replace('monitor_focus','monitor_balanced').replace('focus continuation','balanced continuation')
save(C/'campaign_monitor_balanced.py',s)
print(json.dumps(dict(spec_sha256=digest,prepared=['checkpoint verifier','sequential19-checkpoint backup','exact-merged preupload','whole-Hub verifier','contract validation','read-only monitor'])))
