"""Copy maintained source/evidence only; no live training files are edited."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import time
C=Path(__file__).resolve().parent.parent
REPO=Path('/Knowin/demo/weilinruan/Anvil/01_Code')
OUT=REPO/'experiments/focus_continuation_20261101'
N=C/'focus_repair_20261101'
V=C/'verification_20261101'
OUT.mkdir(exist_ok=True)
files={}
def copy(source,relative):
    assert source.is_file(),source
    payload=source.read_bytes()
    assert len(payload)<20_000_000,'Only compact source/evidence may be synchronized'
    text=payload.decode()
    assert not re.search(r'\b(?:hf_|ghp_)[A-Za-z0-9]{20,}\b',text),'Credential-like source content'
    if source.suffix=='.py':ast.parse(text)
    destination=OUT/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(payload)
    files[relative]=dict(source=str(source),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
for suffix in ['*.py','*.json','*.yaml']:
    for source in sorted(N.glob(suffix)):
        if source.name.startswith(('node','controller')) and source.suffix!='.py':continue
        if re.match(r'(repair|mopd)_checkpoint\d+_verification\.json$',source.name):continue
        if source.name in ['submission_intent.json','capacity.json']:continue
        copy(source,'training/'+source.name)
for source in sorted((C/'final_delivery_focus_v1').glob('*.py')):copy(source,'delivery/'+source.name)
for name in ['cpu_validation.json','rotation128_interface_validation.json']:
    copy(C/'final_delivery_focus_v1/interface_validation'/name,'evidence/'+name)
for name in ['focus_continuation_controller.py','focus_final_handoff.py','backup_focus_checkpoints.py',
             'campaign_monitor_focus.py','prepare_refspatial_grounding_20261101.py','cleanup_targeted_inference_20261101.py']:
    copy(C/name,'operations/'+name)
for name in ['verify_focus_checkpoint.py','verify_entire_focus_hub.py','validate_focus_release_contract.py',
             'focus_release_contract_validation.json','snapshot_focus_source.py',
             'preupload_focus_merged_blobs.py','focus_preupload_contract_validation.json',
             'published_native_metric_review.json']:
    copy(V/name,'operations/'+name)
for source,relative in [
    (C/'refspatial_grounding_20261101/manifest.json','evidence/supplement_manifest.json'),
    (C/'delivery_runs/focus_repair_20261101/policy.json','delivery_policy.json'),
    (C/'delivery_runs/targeted_repair_20261031/evaluation/final_report.json','evidence/parent_final_report.json'),
    (C/'delivery_runs/targeted_repair_20261031/training_lineage.json','evidence/parent_training_lineage.json'),
    (C/'targeted_repair_20261031/mopd_checkpoint768_verification.json','evidence/parent_mopd768_integrity.json'),
    (C/'targeted_checkpoint_backup_20261031/mopd_768/verification_receipt.json','evidence/nine_checkpoint_backup_receipt.json'),
    (C/'targeted_checkpoint_backup_20261031/mopd_768/manifest.json','evidence/nine_checkpoint_backup_manifest.json'),
    (N/'submission.json','operations/submission.json'),
]:copy(source,relative)
for source,relative in [
    (N/'repair_checkpoint64_verification.json','evidence/first_production_checkpoint_integrity.json'),
    (C/'storage_cleanup_20261101/deletion_receipt.json','operations/inference_cleanup_receipt.json'),
    (C/'storage_cleanup_20261101/preflight.json','operations/inference_cleanup_preflight.json'),
    (V/'failure_resume_storage_review_focus.md','operations/failure_resume_storage_review.md'),
]:
    if source.exists():copy(source,relative)
(OUT/'README.md').write_text('''# Grounding continuation from completed MOPD768

Task `t-20261101095411-z67h9` runs on four nodes / 32 A800 80GB GPUs. It inherits the completed MOPD768 adapter, following the earlier MOPD1536 ancestor; the initial 5,220,000-record SFT is not rerun. The new stages use fresh optimizers: repair8192 then genuine on-policy MOPD512, batch 64. The original 5,800,000-record corpus and reachable 4,756,000 native training pool remain unchanged.

The preceding complete evaluation scored 70.1171% on the seven-benchmark macro against the paper target 74.27%. Placement, relative-depth and pointing gaps and weak native geometry prevented final acceptance. Original scores are retained in `evidence/parent_final_report.json`; no score or generation-budget change is used to meet the target.

The added sources contain 97,674 natural-language grounding pairs and 116,175 relative-depth QA pairs, drawn only from the previously admitted training-image set. Their union contains 35,510 images and adds no new images. Source annotations are retained. The source does not establish an official split or independently verified semantic correctness for every annotation. Source-pool inventory and planned exposures are not actual training coverage.

Prior auxiliary mask generation consumed about 5.33 hours while producing zero nonzero unlikelihood updates. The new frozen schedule retains all mask supervised updates and runs auxiliary generation every eighth mask update. Genuine native MOPD rollouts and inference budgets remain unchanged. The original estimate is 12.21 training hours plus queue/loading variation; eight hours are reserved for evaluation and upload before the 2026-11-02 13:48 CST deadline. This is a schedule estimate, not a promised quality result.

Repair saves every 64 updates and MOPD every 32. A sequential backup worker retains the nine previously verified checkpoint paths and appends five distinct `focus_repair/` or `focus_mopd/` paths. Each contains the adapter, optimizer, scheduler and 32 RNG states. Final publication requires all fourteen declared checkpoints, exact-model full evaluation, evidence-backed quality review and pinned remote inventory/checksum verification. Intermediate checkpoint uploads do not constitute final acceptance. Server HF credentials are not changed.

`source_manifest.json` binds this archive to the actual immutable runtime files. The Python sources preserve their runtime layout assumptions and must be used at the documented runtime paths with the recorded dependencies; do not launch this archival copy blindly. CPU contract checks are not GPU or quality validation. Full native/public/general evaluation is automatic after training Success; final quality review remains explicit.

The historical Anvil-Eval 65.35 aggregate cannot currently be reconstructed from the available mapping. Distinct-unit native metrics are reported separately. Validation results have informed the repairs, so the eventual results are not a new blind test. No reproduction claim is made before final review.

A separate CPU worker waits for the exact final merged model and all fourteen verified checkpoint backups, then preuploads its four LFS shards during final evaluation. It creates no repository commit or public release. It verifies shard hashes and that the Hub revision/inventory remain unchanged, uses only the transient authorized token, and holds the campaign Hub-writer lock. Final publication must wait for this worker to terminate as well as for quality acceptance.
''')
if (N/'repair_checkpoint64_verification.json').exists():
    with (OUT/'README.md').open('a') as stream:
        stream.write('\nThe first production checkpoint (repair64) passed CPU verification of all 38 payload hashes, 496 finite updated adapter tensors, optimizer/scheduler step 64, all 32 RNG states and all 4,096 committed exposure records. Training continued beyond the save point. This verifies saved state integrity; it is not a resumed GPU experiment or held-out quality result.\n')
manifest=dict(schema='anvil-focus-source-snapshot-v1',created=time.time(),runtime_root=str(C),files=files,
              live_sources_modified=False,includes_weights_or_source_data=False)
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copyfile(V/'verify_entire_focus_hub.py',REPO/'tools/verify_anvil_final_hub.py')
print(json.dumps(dict(files=len(files),source_snapshot=str(OUT),final_verifier_requires=14)))
