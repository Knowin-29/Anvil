"""Compare real8/16GPU execution on the same768 source exposures.

Global batch32 is retained. Assignment to rank/RNG and per-update auxiliary
composition differ; this is measured system throughput, not model-quality proof.
"""
import argparse
from collections import Counter
import hashlib,json,math,statistics
from pathlib import Path
C=Path(__file__).resolve().parent;D=C/'diagnosis_32gpu'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pctl(values,fraction):
    values=sorted(values);return values[min(len(values)-1,math.ceil(len(values)*fraction)-1)]


def inspect(root,world):
    runtime_path=root/'resume/runtime_verified.json';probe_path=root/'probe/probe_verified.json'
    runtime=json.loads(runtime_path.read_text());probe=json.loads(probe_path.read_text())
    assert runtime['state']=='passed_execution_only' and runtime['world']==world
    assert probe['state']=='passed_full_budget_execution_probe' and probe['formal_mopd'] is False
    protocol=json.loads((root/'probe/protocol.json').read_text())
    assert protocol['world_size']==world and protocol['global_batch']==32 and protocol['smoke'] and protocol['full_budget_probe']
    endpoint=32 if world==8 else 24
    state=json.loads((root/f'probe/checkpoint-{endpoint}/trainer_state.json').read_text())
    logs=state['log_history'][:24];assert [r['step'] for r in logs]==list(range(1,25))
    seconds=[r['seconds'] for r in logs];assert all(math.isfinite(s) and s>0 for s in seconds)
    ids=Counter();counts=Counter();tokens=[];stop_reasons=Counter();files={}
    for rank in range(world):
        path=root/f'probe/exposure_rank{rank}.jsonl';files[str(path)]=sha(path)
        for row in map(json.loads,path.open()):
            if row['step']>24:continue
            assert row['rank']==rank and row['smoke'];ids[row['canonical_uid']]+=1;counts[row['source_category']]+=1
            if row['source_category']=='native_distillation':
                tokens.append(row['rollout_tokens']);stop_reasons[row['rollout_stop_reason']]+=1
    assert counts==dict(native_distillation=384,general_qa=192,spatial_reference=192)
    wall=logs[-1]['elapsed']
    result=dict(world=world,global_batch=32,measured_updates=24,exposures=768,
        mean_update_seconds=statistics.mean(seconds),median_update_seconds=statistics.median(seconds),
        p90_update_seconds=pctl(seconds,.9),p95_update_seconds=pctl(seconds,.95),max_update_seconds=max(seconds),
        elapsed_including_intermediate_saves_seconds=wall,samples_per_second_with_saves=768/wall,
        GPU_seconds_per_sample_with_saves=world*wall/768,seconds_per_update=seconds,
        rank0_reported_peak_gpu_gib=max(r['peak_gpu_gib'] for r in logs),
        mean_rollout_tokens=statistics.mean(tokens),max_rollout_tokens=max(tokens),rollout_stop_reasons=dict(stop_reasons),
        source_exposures=dict(counts),actual_source_UID_multiset_sha256=hashlib.sha256(json.dumps(sorted(ids.items())).encode()).hexdigest(),
        evidence={str(runtime_path):sha(runtime_path),str(probe_path):sha(probe_path),
                  str(root/'probe/protocol.json'):sha(root/'probe/protocol.json'),**files})
    return result,ids,runtime


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();assert not args.output.exists()
    a,ai,ar=inspect(D/'mopd_v6_candidate1',8)
    b,bi,br=inspect(D/'mopd_v6_candidate1_multinode/actual',16)
    assert ai==bi
    for key in ['student_adapter_sha256','teacher_adapter_sha256','trainer_code_sha256','representation_sha256']:assert ar[key]==br[key],key
    report=dict(state='complete_equal_source_throughput_comparison_requires_budget_review',world8=a,world16=b,
        wall_time_speedup_16_over8=a['elapsed_including_intermediate_saves_seconds']/b['elapsed_including_intermediate_saves_seconds'],
        same_source_UID_multiset=True,same_global_batch=True,same_exact_initial_student_and_teachers=True,
        stochastic_rollouts_bit_identical=False,per_update_auxiliary_composition_identical=False,
        no_guarantee_of_unobserved_long_tail_throughput=True,production_budget_frozen=False,model_quality_qualified=False,
        formal_mopd=False,code_sha256=sha(Path(__file__)))
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(args.output)


if __name__=='__main__':main()
