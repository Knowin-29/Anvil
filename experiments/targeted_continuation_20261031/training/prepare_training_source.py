"""Create an isolated continuation trainer; no cloud submission or live edits."""
import ast
import hashlib
import json
from pathlib import Path

O = Path(__file__).resolve().parent
C = O.parent
OLD = C / 'direct_recovery_20261030'


def write_new(path, text):
    if path.exists():
        assert path.read_text() == text, 'Refuse to overwrite different prepared source'
    else:
        with path.open('x') as stream:
            stream.write(text)
    ast.parse(text, filename=str(path))


def main():
    source = (OLD / 'train.py').read_text()
    replacements = [
        ('from pool import Pool,SCHEDULES',
         'from pool import Pool,SCHEDULES,FORMS\nfrom resume_state import restore_exposures,restore_logs'),
        ("distributed_timeout_seconds=spec['distributed_timeout_seconds'],recovery=spec['recovery'])",
         "distributed_timeout_seconds=spec['distributed_timeout_seconds'],recovery=spec['recovery'],\n"
         "        sampling_seed=spec['sampling_seed'],source_offsets=spec['source_offsets'],\n"
         "        balanced_native_forms=list(FORMS),continuation_parent=spec['continuation_parent'])"),
        ("        random.setstate(rng['python']);np.random.set_state(rng['numpy'])",
         "        random.setstate(rng['python']);np.random.set_state(rng['numpy'])\n"
         "        assert 0 < start < cfg['steps'],'Resume requires an incomplete stage'\n"
         "        if rank==0:restore_logs(out/'logging.jsonl',logs,start)"),
        ("    begin=time.monotonic();qualified=dict.fromkeys('GSA',True)",
         "    begin=time.monotonic();elapsed_before_resume=logs[-1]['elapsed'] if logs else 0.\n"
         "    qualified=dict.fromkeys('GSA',True)"),
        ("        dest=out/f'checkpoint-{step}';dest.mkdir(exist_ok=True)",
         "        dest=out/f'checkpoint-{step}'\n"
         "        if rank==0:\n"
         "            assert not (dest/'checkpoint_complete.json').exists(),'Never overwrite a sealed checkpoint'\n"
         "            if dest.exists():dest.rename(out/f'partial_checkpoint_{step}_before_resume_{time.time_ns()}')\n"
         "            dest.mkdir()\n"
         "        dist.barrier()"),
        ("elapsed=time.monotonic()-begin,examples_seen=(step+1)*spec['global_batch']",
         "elapsed=elapsed_before_resume+time.monotonic()-begin,examples_seen=(step+1)*spec['global_batch']"),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    start = source.index('        # A declared resume preserves the uncommitted suffix before trimming it.')
    stop = source.index("    with exposure_path.open('a') as stream:", start)
    source = source[:start] + '        restore_exposures(exposure_path,start,rank,accum)\n' + source[stop:]
    write_new(O / 'train.py', source)
    node = (OLD / 'node.py').read_text()
    node = node.replace('Four-worker MOPD recovery from the completed corrective checkpoint.',
                        'Four-worker targeted repair and MOPD continuation from the completed parent.')
    node = node.replace("    for stage in ['mopd']:", "    for stage in ['repair','mopd']:")
    node = node.replace("env=environment(range(8),'direct_completion_'+str(node))",
                        "env=environment(range(8),'targeted_continuation_'+str(node))")
    write_new(O / 'node.py', node)
    print(json.dumps({'training_source':str(O/'train.py'),'node_source':str(O/'node.py'),
                      'live_sources_changed':False,'cloud_task_submitted':False}))


if __name__ == '__main__':
    main()
