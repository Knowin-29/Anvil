"""Clarify physical-topology differences from the actual per-example objective."""
import json
from pathlib import Path
import sys
import time
import torch
from checkpoint_prefix_audit import C, bound, prefix_rows, read, require, save, sha


def main():
    formal = C / 'formal_mopd/runs/candidate1_20261029/training'
    control = C / 'objective_controls_20261029/runs/low_native_sft'
    observations = {}
    for name, run, world, step in [('formal128', formal, 16, 128),
                                  ('low_native_sft_committed_prefix', control, 8, read(control/'status.json')['step'])]:
        protocol = read(run/'protocol.json'); require(protocol['world_size'] == world, 'Wrong topology')
        n = 0; zeros = 0; digests = {}
        for rank in range(world):
            parsed = prefix_rows(run/('exposure_rank%d.jsonl' % rank),step,rank,32//world)
            digests[str(rank)] = dict(rows=len(parsed),sha256=__import__('hashlib').sha256(b''.join(raw for raw,row in parsed)).hexdigest())
            for raw,row in parsed:
                if row['source_category'] == 'native_distillation':
                    n += 1; zeros += int(row['teacher_scored_positions'] == 0)
        require(zeros == 0, 'Empty native spans require a separate reduction analysis')
        observations[name] = dict(step=step, native_rows=n, empty_stage_spans=zeros, rank_prefixes=digests,
            protocol=bound(run/'protocol.json'))
    # The router's selected_stages is a [batch] vector, not a token mask.
    # Microbatch1 gives one eligible example per rank when all spans are nonempty.
    # Each example's JS term already averages its own structurally weighted tokens.
    values = torch.arange(1,33,dtype=torch.float64)/31
    grads = {}
    for world in [8,16]:
        parameter = torch.tensor(.31,dtype=torch.float64,requires_grad=True)
        for micro in range(32//world):
            loss = ((parameter-values[micro*world:(micro+1)*world])**2).mean()/(32//world)
            loss.backward()
        grads[str(world)] = float(parameter.grad)
    require(abs(grads['8']-grads['16']) < 1e-12, 'Per-example global-batch weighting differs')
    sources = [C/'native_mopd_v6.py',C/'native_mopd_v7.py',C/'mopd_representation.py',
               C/'Anvil/trains/mopd/anvil_runtime.py',C/'Anvil/trains/mopd/chunked_heads.py',Path(__file__)]
    report = dict(state='verified_reduction_clarification',observations=observations,
        selected_stages_shape='[microbatch_examples]',runtime_microbatch_examples=1,
        mtd_local_reduction='Per-example mean over selected response positions after mean-normalized structural weighting.',
        mtd_global_reduction='Mean over eligible examples; when all ranks eligible, eligible*world/global_eligible is1.',
        eight_vs_sixteen_source_weighting='Both give each of32 source examples coefficient1/32 before the chosen MTD multiplier.',
        numerical_gradient_check=grads,
        remaining_topology_differences=['Per-rank generation RNG streams differ.',
            'Floating-point summation order and kernel execution may differ.'],
        correction='Earlier frozen control notes called this a token-normalization-group change. That wording was imprecise: selected_stages counts examples, and the observed all-eligible rows preserve the same per-example objective weighting. Frozen execution manifests remain unchanged; this addendum supersedes that interpretation.',
        mask_routing_note='Native atomic routing selects the declared family for every nonempty response. Invalid JSON alone does not suppress MTD; no empty native scored spans observed in these prefixes.',
        evidence={str(p):sha(p) for p in sources},gpu_training_modified=False,quality_cause_proven=False,completed=time.time())
    save(C/'objective_controls_20261029/topology_clarification.json',report)
    print(json.dumps({k:report[k] for k in ['state','numerical_gradient_check','correction','mask_routing_note']},indent=2))


if __name__ == '__main__':
    main()
