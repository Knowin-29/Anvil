"""Matched source-coverage repair, with original native and general QA replay.

All arms start from byte-bound completed checkpoints, not from a copied merged
model. Each includes a real optimizer resume prerequisite in spatial_node.py.
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

ARMS = ['joint_control', 'joint_spatial', 'G_control', 'G_spatial']
STEPS = 1024


class Pool:
    def __init__(self, arm):
        self.arm = arm
        self.specialist = arm.startswith('G_')
        self.intervention = arm.endswith('_spatial')
        self.native = FullPool()
        self.index = json.loads((R / 'training_repair/full_training_index/manifest.json').read_text())
        self.rows = {}
        self.source_manifests = {}
        for folder in ('spatial_replay', 'retention_replay'):
            manifest_path = ROOT / folder / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            assert manifest['state'] == 'verified_train_only'
            self.source_manifests[folder] = sha(manifest_path)
            for name, digest in manifest['files_sha256'].items():
                path = ROOT / folder / (name + '.jsonl')
                assert sha(path) == digest
                self.rows[name] = [json.loads(l) for l in path.open()]
                assert len(self.rows[name]) == manifest['additional_training_records'][name]
        point = self.index['files']['train_point.jsonl']
        self.points = np.load(point['index_file'], mmap_mode='r')
        assert sha(point['index_file']) == point['index_sha256']
        self.point_handle = (Path(self.index['export']) / 'train_point.jsonl').open('rb')

    @staticmethod
    def permute(ordinal, count):
        stride = 1000003
        while math.gcd(stride, count) != 1:
            stride += 2
        return (ordinal * stride + 17017) % count

    def sample(self, micro, rank, world):
        slot = micro % 4
        if slot == 1:
            ordinal = micro // 4
            # G candidates retain their specialist scope; joint retains all GSA.
            row = self.native.sample(ordinal * 3 if self.specialist else ordinal, rank, world)
            return row, 'original_native'
        if slot == 3:
            ordinal = micro // 4
            name = ['ai2d', 'scienceqa', 'textvqa'][ordinal % 3]
            index = (ordinal // 3) * world + rank
            return self.rows[name][self.permute(index, len(self.rows[name]))], 'general_qa'
        ordinal = micro // 2
        if not self.intervention:
            index = self.permute(ordinal * world + rank, len(self.points))
            offset, size = self.points[index]
            self.point_handle.seek(int(offset))
            row = json.loads(self.point_handle.read(int(size)))
            assert row['gsa_form'] == 'point'
            return row, 'original_point_control'
        # Half the microbatches are source coverage; inside them FSD:vacancy=3:1.
        subslot = ordinal % 4
        name = 'refspatial_vacancy' if subslot == 3 else 'fsd_goal'
        within = ordinal // 4 if subslot == 3 else (ordinal // 4) * 3 + subslot
        index = within * world + rank
        return self.rows[name][self.permute(index, len(self.rows[name]))], name


def initial_checkpoint(arm):
    directory = ROOT / ('repairs/G_unlikelihood' if arm.startswith('G_') else 'retention_repairs/joint_broad')
    receipt = json.loads((directory / 'complete.json').read_text())
    adapter = Path(receipt['checkpoint'])
    assert receipt['step'] == 512 and not receipt['smoke']
    assert sha(adapter / 'adapter_model.safetensors') == receipt['adapter_sha256']
    return dict(checkpoint=str(adapter), adapter_sha256=receipt['adapter_sha256'], completion_receipt_sha256=sha(directory / 'complete.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    rank, world, local = [int(os.environ.get(k, default)) for k, default in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')]]
    assert world == 4
    assert args.stop_after in (1, 2) if args.smoke else args.stop_after is None
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    torch.set_num_threads(4)
    torch.manual_seed(17 + rank)
    random.seed(17 + rank)
    np.random.seed(17 + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.arm)
    initial = initial_checkpoint(args.arm)
    protocol = dict(schema='anvil-spatial-source-coverage-v1', arm=args.arm, initial=initial,
        source_export_manifest_sha256=pool.index['export_manifest_sha256'],
        source_index_sha256=sha(R / 'training_repair/full_training_index/manifest.json'),
        source_pool=5220000, full_original_sft_completed=True, world=4, global_batch=32, steps=STEPS, seed=17,
        mixture=dict(native=.25, broad_qa=.25, fsd_goal=.375, refspatial_vacancy=.125) if pool.intervention else
                dict(native=.25, broad_qa=.25, original_point=.5),
        native_scope='G' if pool.specialist else 'GSA', source_manifests=pool.source_manifests,
        lr=2e-5, warmup_fraction=.03, schedule='cosine', trainable_parameters=173113344,
        context=16384, image_tokens=1536, source_data_modified=False, heldout_training=False,
        supplemental_separately_counted=True, final_mopd=False, smoke=args.smoke,
        code_sha256={n:sha(ROOT / n) for n in ['spatial_train.py']})
    if rank == 0:
        path = args.output / 'protocol.json'
        if path.exists():
            assert json.loads(path.read_text()) == protocol
        else:
            save(path, protocol)
        save(args.output / 'status.json', dict(stage='loading', arm=args.arm, smoke=args.smoke, started=time.time()))
    dist.barrier()
    install_hidden_forward()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map={'':device}, attn_implementation='sdpa', local_files_only=True)
    model = PeftModel.from_pretrained(model, str(args.resume or initial['checkpoint']), is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == 173113344
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=.01, fused=True)
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
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    accum, began = 8, time.monotonic()

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
                group['lr'] = 2e-5 * ratio
            for micro in range(accum):
                row, source = pool.sample(step * accum + micro, rank, world)
                assert row['gsa_meta']['split'] == 'train'
                data = move(encode(p, row), device)
                model.train()
                with ddp.no_sync() if micro < accum-1 else nullcontext():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        hidden = ddp(**forward_inputs(data), anvil_hidden_only=True, use_cache=False)
                        ce = chunked_ce(hidden, head, data['labels'])
                        assert torch.isfinite(ce)
                        (ce/accum).backward()
                        stats[0] += ce.detach()/accum
                exposure.write(json.dumps(dict(step=step+1, micro=micro, rank=rank, canonical_uid=row['canonical_uid'],
                    source=source, form=row['gsa_form'], source_binding=row['gsa_meta']['binding'],
                    target_sha256=hashlib.sha256(row['messages'][1]['content'].encode()).hexdigest(),
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
                save(args.output / 'status.json', dict(stage='training', arm=args.arm, smoke=args.smoke, updated=time.time(), **entry))
                print(json.dumps(entry), flush=True)
            if (step+1) % 256 == 0 or step+1 == end:
                checkpoint(step+1)
    if rank == 0:
        dest = args.output / f'checkpoint-{end}'
        save(args.output / 'complete.json', dict(state='smoke_complete' if args.smoke else 'repair_complete_requires_evaluation',
            arm=args.arm, checkpoint=str(dest), adapter_sha256=sha(dest/'adapter_model.safetensors'), step=end, smoke=args.smoke,
            final_mopd=False, teacher_qualified=False, completed=time.time()))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
