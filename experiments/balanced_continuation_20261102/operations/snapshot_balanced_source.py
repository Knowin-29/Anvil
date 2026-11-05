"""Archive current authorized production source and compact provenance, never weights/data."""
from pathlib import Path
import ast,hashlib,json,re,shutil,time
V=Path(__file__).resolve().parent;C=V.parent;N=C/'balanced_repair_20261102'
REPO=Path('/workspace/Anvil');OUT=REPO/'experiments/balanced_continuation_20261102';OUT.mkdir(exist_ok=True)
files={}
def copy(source,relative):
 assert source.is_file(),source
 payload=source.read_bytes();assert len(payload)<20_000_000
 text=payload.decode();assert not re.search(r'\b(?:hf_|ghp_)[A-Za-z0-9]{20,}\b',text)
 if source.suffix=='.py':ast.parse(text)
 d=OUT/relative;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(payload)
 files[relative]=dict(source=str(source),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
for ext in ['*.py','*.json','*.yaml']:
 for p in sorted(N.glob(ext)):
  if p.suffix!='.py' and (p.name.startswith(('node','controller')) or p.name in ['capacity.json','submission_intent.json','public_trigger.preflight_attempt.json']):continue
  if re.match(r'(repair|mopd)_checkpoint\d+_verification\.json$',p.name):continue
  copy(p,'training/'+p.name)
for p in sorted((C/'final_delivery_balanced_v1').glob('*.py')):copy(p,'delivery/'+p.name)
for name in ['cpu_validation.json','rotation128_interface_validation.json']:
 copy(C/'final_delivery_balanced_v1/interface_validation'/name,'evidence/'+name)
for name in ['balanced_continuation_controller.py','balanced_final_handoff.py','backup_balanced_checkpoints.py','campaign_monitor_balanced.py','prepare_sat_replay_20261102.py','cleanup_rejected_focus_inference_20261102.py']:
 copy(C/name,'operations/'+name)
for name in ['verify_balanced_checkpoint.py','verify_entire_balanced_hub.py','validate_balanced_release_contract.py','balanced_release_contract_validation.json',
 'preupload_balanced_merged_blobs.py','cleanup_verified_balanced_weights.py','test_cleanup_verified_balanced_weights.py',
 'snapshot_balanced_source.py','prepare_balanced_operations.py','update_balanced_delivery_pointer.py',
 'failure_resume_storage_review_balanced.md']:
 copy(V/name,'operations/'+name)
for source,relative in [
 (C/'sat_replay_20261102/manifest.json','evidence/sat_manifest.json'),
 (C/'refspatial_grounding_20261101/manifest.json','evidence/grounding_manifest.json'),
 (C/'delivery_runs/balanced_repair_20261102/policy.json','delivery_policy.json'),
 (C/'delivery_runs/targeted_repair_20261031/evaluation/final_report.json','evidence/parent_final_report.json'),
 (C/'delivery_runs/targeted_repair_20261031/training_lineage.json','evidence/parent_training_lineage.json'),
 (C/'targeted_repair_20261031/mopd_checkpoint768_verification.json','evidence/parent_mopd768_integrity.json'),
 (C/'delivery_runs/focus_repair_20261101/evaluation/final_report.json','evidence/rejected_focus_final_report.json'),
 (C/'delivery_runs/focus_repair_20261101/quality_review.json','evidence/rejected_focus_quality_review.json'),
 (C/'focus_checkpoint_backup_20261101/mopd_512/verification_receipt.json','evidence/fourteen_checkpoint_backup_receipt.json'),
 (C/'focus_checkpoint_backup_20261101/mopd_512/manifest.json','evidence/fourteen_checkpoint_backup_manifest.json'),
 (N/'repair_checkpoint64_verification.json','evidence/first_production_checkpoint_integrity.json'),
 (C/'storage_cleanup_rejected_focus_20261102/deletion_receipt.json','operations/rejected_inference_cleanup_receipt.json'),
 (C/'storage_cleanup_rejected_focus_20261102/preflight.json','operations/rejected_inference_cleanup_preflight.json')]:copy(source,relative)
(OUT/'README.md').write_text('''# Balanced continuation from the stronger MOPD768 parent

Task `t-20261102021323-bhkpt` uses four nodes / 32 A800 80GB GPUs. It branches from the fully evaluated MOPD768 parent: repair2560 at5e-6, then genuine on-policy MOPD512 at5e-7, batch64. The original5,220,000-record SFT is retained. New stages intentionally reset their optimizers; sealed within-stage checkpoints retain optimizer, scheduler,32-rank RNG and committed log/exposure prefixes for recovery.

The preceding focus experiment completed normally but regressed from the parent's70.1171% to67.6957% seven-benchmark macro (paper74.27%). It was rejected. All complete metrics and its rejection are archived here. Answer-format/domain transfer and higher repair LR are possible contributors, not established unique causes. The new model does not inherit that rejected branch or its exposure counts. Its five focus_* checkpoint paths remain experiment backups, clearly marked as nonancestors.

SAT contributes51,322 source-original, single-image training examples and13,699 distinct decoded images. Exact, decoded and perceptual-image exclusions protect native/public/general validation and source-test images.9,755 multi-image rows are excluded because the encoder supports one image, and18,923 rows fail protected-image exclusion. No validation answers enter training. Source train/test annotations are inherited, not newly certified scene splits or fully audited semantic labels. Only the exact source-added reasoning-format suffix is removed. Repair plans61,440 SAT exposures, enough for one full admitted-pool permutation; inventories, unique coverage and repeated exposures remain distinct. Existing FSD, native, grounding, depth/mask, vacancy and general QA replay continue. The rejected branch's new RefSpatial spatialqa source is not sampled.

Original5.8M corpus /160K-image inventory and4,756,000 reachable native training records remain disclosed. Actual original source composition differs from the paper. The historical native65.35 aggregate lacks a uniquely reconstructible published normalization; distinct-unit native metrics remain separate. Validation has informed repairs, so final results are not a new blind test. No paper reproduction or quality success is asserted.

Measured prior role throughput estimates8.33 training hours, with2h reserved for full validation and upload before2026-11-02 05:48 UTC. Queueing, loading and generation variability remain. Full580,000 native,7,366 public and10,105 general evaluation follows training automatically. No small GPU diagnostic arms are launched. Acceptance remains an explicit evidence-based review of that exact candidate.

Repair saves every64 updates and MOPD every32. The first repair64 production checkpoint passed CPU verification of38 payload hashes,496 finite updated adapter tensors, optimizer/scheduler64,32 RNG files and4,096 committed exposure records. This is saved-state integrity, not a GPU resume experiment or quality result.

Fourteen previous complete key checkpoints are remotely verified at`3463ded9ba5fc24939206f986c6a3bafc1e776a0` (563 files). Sequential backup adds balanced_repair1280/2560 and balanced_mopd128/256/512, for19 total. After all backups, a worker can preupload exact merged LFS blobs without committing a final release. Final publication requires the full evaluation, quality acceptance, complete19-key package and pinned whole-repository verification. Server HF credentials are unchanged; credentials are supplied transiently.

A further17.98GiB of reconstructible rejected-focus merged inference shards was removed after all evaluators/writers ended and428 current frozen dependencies were checked; total cleanup is approximately413.55GiB, mainly shared storage. Adapters, checkpoints, base/teachers, data, logs, evidence and Codex history are retained. Final payload cleanup is prepared with explicit file identities,19-key verification and idle-job/writer gates; it remains unexecuted. Its10 CPU temporary-file safety tests passed.

`source_manifest.json` maps this archival source to immutable runtime files. Runtime layout dependencies remain explicit; this is not a standalone relocatable launcher. Sources, policies and evidence are copied only after freezing; running training sources are not edited.
''')
(OUT/'source_manifest.json').write_text(json.dumps(dict(schema='anvil-balanced-source-snapshot-v1',created=time.time(),runtime_root=str(C),files=files,live_sources_modified=False,includes_weights_or_source_data=False),indent=2)+'\n')
shutil.copyfile(V/'verify_entire_balanced_hub.py',REPO/'tools/verify_anvil_final_hub.py')
for name in ['cleanup_verified_balanced_weights.py','test_cleanup_verified_balanced_weights.py']:shutil.copyfile(V/name,REPO/'tools'/name)
print(json.dumps(dict(files=len(files),source_snapshot=str(OUT),required_checkpoint_count=19)))
