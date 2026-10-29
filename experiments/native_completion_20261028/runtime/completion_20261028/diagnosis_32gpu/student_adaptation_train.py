"""Eight-GPU student interface adaptation with original/native/QA/spatial replay.
Not formal MOPD. The exact initial checkpoint and pose objective require a sealed spec.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
C, R = ROOT.parent, ROOT.parent.parent
sys.path[:0] = [str(C), str(R), str(R / 'cloud_a800')]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import fla.ops.gated_delta_rule
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration
from training_data import BASE, encode, processor
from memory_loss import chunked_ce
from native_mopd_v2 import FullPool, install_hidden_forward, move, forward_inputs, sha
from checkpoints import save, seal, verify

from student_adaptation_pool import Pool, load_spec
from dof_codebook_interface import training_row, VERSION as CODEC_VERSION
from dof_pose_loss import PoseLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec=load_spec(args.spec);STEPS=spec['steps'];learning_rate=spec['learning_rate']
    rank, world, local = [int(os.environ.get(k, default)) for k, default in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')]]
    assert world == 8
    assert args.stop_after in (1, 2) if args.smoke else args.stop_after is None
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    torch.set_num_threads(4)
    torch.manual_seed(17 + rank)
    random.seed(17 + rank)
    np.random.seed(17 + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    pool = Pool()
    initial = spec['initial']
    protocol = dict(schema='anvil-student-representation-adaptation-v1', initial=initial,
        model_representation=CODEC_VERSION,decode_required_for_native_inference=True,
        spec_sha256=sha(args.spec),source_export_manifest_sha256=pool.index['export_manifest_sha256'],
        source_index_sha256=sha(R/'training_repair/full_training_index/manifest.json'),
        source_pool=5220000,full_original_sft_completed=True,world=world,global_batch=32,steps=STEPS,seed=17,
        mixture=dict(dof_union=.5,original_native=.25,broad_qa=.125,spatial_supplement=.125),
        dof_original_records=3008,dof_supplemental_records=13536,source_manifests=pool.source_manifests,
        data_inventory_sha256=sha(ROOT/'augmented_data_scale_reconciliation_v2.json'),
        available_training_records=5316931,available_training_images=204792,inventory_is_actual_exposure=False,
        pose_loss=dict(mode=spec['pose_mode'],token_weight=32.,sigma_degrees=15.),
        lr=learning_rate,warmup_fraction=.03,schedule='cosine',trainable_parameters=173113344,
        context=16384,image_tokens=1536,source_data_modified=False,heldout_training=False,
        supplemental_separately_counted=True,final_mopd=False,smoke=args.smoke,
        code_sha256={n:sha(ROOT/n) for n in ['student_adaptation_train.py','student_adaptation_pool.py','dof_pose_loss.py','dof_codebook_interface.py','dof_rotation_codebook.json','dof_codec_prototype.py','dof_codec_interface.py','prepare_dof_rotation_codebook.py']})
    if rank == 0:
        path = args.output / 'protocol.json'
        if path.exists():
            assert json.loads(path.read_text()) == protocol
        else:
            save(path, protocol)
        save(args.output / 'status.json', dict(stage='loading', arm='student_adaptation', smoke=args.smoke, started=time.time()))
    dist.barrier()
    install_hidden_forward()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map={'':device}, attn_implementation='sdpa', local_files_only=True)
    model = PeftModel.from_pretrained(model, str(args.resume or initial['checkpoint']), is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == 173113344
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.01, fused=True)
    start, logs = 0, []
    if args.resume:
        verify(args.resume, world)
        optimizer.load_state_dict(torch.load(args.resume / 'optimizer.pt', map_location=device, weights_only=True))
        state = json.loads((args.resume / 'trainer_state.json').read_text())
        start, logs = state['global_step'], state['log_history']
        rng = torch.load(args.resume / f'rng_state_{rank}.pth', map_location='cpu', weights_only=False)
        torch.set_rng_state(rng['cpu'])
        torch.cuda.set_rng_state(rng['cuda'], device)
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
    p, head = processor(), model.get_output_embeddings()
    pose_loss=PoseLoss(p.tokenizer,spec['pose_mode'])
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    accum, began = 32//world, time.monotonic()

    def checkpoint(step):
        dest = args.output / f'checkpoint-{step}'
        dest.mkdir(exist_ok=True)
        if rank == 0:
            model.save_pretrained(dest, safe_serialization=True)
            torch.save(optimizer.state_dict(), dest / 'optimizer.pt')
            torch.save(dict(step=step, schedule='cosine', steps=STEPS), dest / 'scheduler.pt')
            save(dest / 'trainer_state.json', dict(global_step=step, max_steps=STEPS, examples_seen=step*32, log_history=logs, smoke=args.smoke, final_mopd=False))
        torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device), python=random.getstate(), numpy=np.random.get_state()), dest / f'rng_state_{rank}.pth')
        dist.barrier()
        if rank == 0:
            seal(dest, world)
            save(args.output / 'latest.json', dict(path=str(dest), step=step))
        dist.barrier()

    end = args.stop_after or STEPS
    with (args.output / f'exposure_rank{rank}.jsonl').open('a') as exposure:
        for step in range(start, end):
            t0 = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            stats = torch.zeros(1, device=device)
            warmup = math.ceil(STEPS * .03)
            ratio = min(1., (step+1)/warmup) if step < warmup else .5 * (1 + math.cos(math.pi * (step-warmup)/(STEPS-warmup)))
            for group in optimizer.param_groups:
                group['lr'] = learning_rate * ratio
            for micro in range(accum):
                row, source = pool.sample(step * accum + micro, rank, world)
                assert row['gsa_meta']['split'] == 'train'
                original_target=row['messages'][1]['content'];original_prompt=row['messages'][0]['content']
                if row['gsa_form']=='dof':row=training_row(row)
                data = move(encode(p, row), device)
                model.train()
                with ddp.no_sync() if micro < accum-1 else nullcontext():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        hidden = ddp(**forward_inputs(data), anvil_hidden_only=True, use_cache=False)
                        ce = pose_loss(hidden,head,data['labels'],row,original_target)[0] if row['gsa_form']=='dof' else chunked_ce(hidden, head, data['labels'])
                        assert torch.isfinite(ce)
                        (ce/accum).backward()
                        stats[0] += ce.detach()/accum
                exposure.write(json.dumps(dict(step=step+1, micro=micro, rank=rank, canonical_uid=row['canonical_uid'],
                    source_category=source, form=row['gsa_form'], source_binding=row['gsa_meta']['binding'],
                    original_target_sha256=hashlib.sha256(original_target.encode()).hexdigest(),
                    original_prompt_sha256=hashlib.sha256(original_prompt.encode()).hexdigest(),
                    derived_target_sha256=hashlib.sha256(row['messages'][1]['content'].encode()).hexdigest(),
                    derived_prompt_sha256=hashlib.sha256(row['messages'][0]['content'].encode()).hexdigest(),
                    tokens=data['input_ids'].shape[1], smoke=args.smoke)) + '\n')
                exposure.flush()
                del hidden, ce, data
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            dist.all_reduce(stats)
            stats /= world
            entry = dict(step=step+1, max_steps=STEPS, ce=float(stats[0]), grad_norm=float(norm), lr=optimizer.param_groups[0]['lr'],
                seconds=time.monotonic()-t0, elapsed=time.monotonic()-began, examples_seen=(step+1)*32,
                peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
            logs.append(entry)
            if rank == 0:
                with (args.output / 'logging.jsonl').open('a') as log:
                    log.write(json.dumps(entry) + '\n')
                save(args.output / 'status.json', dict(stage='training', arm='student_adaptation', smoke=args.smoke, updated=time.time(), **entry))
                print(json.dumps(entry), flush=True)
            if (step+1) % 256 == 0 or step+1 == end:
                checkpoint(step+1)
    if rank == 0:
        dest = args.output / f'checkpoint-{end}'
        save(args.output / 'complete.json', dict(state='smoke_complete' if args.smoke else 'repair_complete_requires_evaluation',
            arm='student_adaptation', checkpoint=str(dest), adapter_sha256=sha(dest/'adapter_model.safetensors'), step=end, smoke=args.smoke,
            final_mopd=False, teacher_qualified=False, completed=time.time()))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
