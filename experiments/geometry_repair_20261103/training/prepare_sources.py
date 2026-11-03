"""Create isolated geometry-continuation sources; never mutate existing runs."""
import ast,hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent;C=O.parent;OLD=C/'format_repair_20261103';H=C/'final_delivery_geometry_v1';PREV=C/'final_delivery_format_v1'
def save(path,source):
 ast.parse(source,filename=str(path))
 with path.open('x') as f:f.write(source)
H.mkdir(exist_ok=False)
for old in sorted(PREV.glob('*.py')):
 source=old.read_text()
 if old.name=='common.py':source=source.replace("pipeline=C/'format_repair_20261103'", "pipeline=C/'geometry_repair_candidate_20261103'")
 if old.name=='lineage.py':
  source=source.replace("stage='format_repair'","stage='geometry_repair'").replace("stage='format_mopd'","stage='geometry_mopd'")
  source=source.replace('The parent candidate failed the public benchmark review. Its weights are an ancestor, not an accepted release.','The parent improved public macro to 73.481429 but retains weak native geometry; this continuation targets those weaknesses. Parent weights are not an accepted final release.')
 if old.name=='package.py':source=source.replace('the format correction/MOPD stages, and the rejected','the format and geometry correction/MOPD stages, and the rejected')
 if old.name=='node.py':
  source=(C/'format_eval_quota_recovery_20261103/eval_node.py').read_text().replace('final_delivery_format_v1','final_delivery_geometry_v1').replace('/tmp/anvil-format-eval-quota','/tmp/anvil-geometry-eval')
 save(H/old.name,source)
for name in ['train.py','node.py','resume_state.py','test_resume_state.py']:
 source=(OLD/name).read_text()
 if name=='node.py':
  source=source.replace("'format_continuation_'","'geometry_continuation_'")
  marker="env['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'"
  assert source.count(marker)==1
  source=source.replace(marker,marker+"\n    import shutil\n    assert shutil.disk_usage('/tmp').free >= 16*1024**3\n    cache=Path('/tmp/anvil-geometry-train')/str(node)\n    for key,leaf in [('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg'),('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:\n        path=cache/leaf;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)\n    probe=O/('storage_preflight_node_'+str(node)+'.pending')\n    with probe.open('xb') as f:\n        for _ in range(64):f.write(bytes(1024**2))\n        f.flush();os.fsync(f.fileno())\n    assert probe.stat().st_size==64*1024**2\n    probe.unlink()\n    save(O/('storage_preflight_node_'+str(node)+'.json'),dict(state='passed',shared_write_bytes=64*1024**2,local_free_bytes=shutil.disk_usage('/tmp').free,cache=str(cache),checked=time.time()))")
 save(O/name,source)
source=(C/'format_continuation_controller.py').read_text().replace('final_delivery_format_v1','final_delivery_geometry_v1').replace('format_repair8192_MOPD256','geometry_repair4096_MOPD256')
save(C/'geometry_continuation_controller.py',source)
source=(C/'format_final_handoff.py').read_text().replace('final_delivery_format_v1','final_delivery_geometry_v1').replace('format_full_final_evaluation_node','geometry_full_final_evaluation_node')
save(C/'geometry_final_handoff.py',source)
# Explicitly verify that model objectives, prompts, scoring, and budgets did not change.
assert (O/'train.py').read_bytes()==(OLD/'train.py').read_bytes()
for name in ['native_worker.py','public_worker.py','score.py','merge.py','prepare.py','upload.py']:
 assert (H/name).read_bytes()==(PREV/name).read_bytes(),name
savefile=O/'source_preparation.json'
files=[*H.glob('*.py'),*(O/n for n in ['train.py','node.py','resume_state.py','test_resume_state.py']),C/'geometry_continuation_controller.py',C/'geometry_final_handoff.py']
savefile.write_text(json.dumps(dict(state='isolated_sources_prepared_not_frozen',training_objective_unchanged=True,heldout_protocol_and_scoring_unchanged=True,training_and_eval_compiler_caches_node_local=True,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},submitted=False),indent=2)+'\n')
print(json.dumps(dict(sources=len(files),submitted=False,training_objective_unchanged=True)))
