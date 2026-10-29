"""Native on-policy distillation with exact bounded-head losses and resume receipts.

The original full-corpus SFT is preserved; supplemental QA is counted separately.
Update count and every sampled row are recorded. Intermediate SFT retention is
diagnostic. Smoke runs never qualify teachers or count as final MOPD.
"""
import argparse
import bisect
from contextlib import nullcontext
from copy import deepcopy
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT.parent
sys.path[:0] = [str(ROOT / "Anvil"), str(RECOVERY), str(RECOVERY / "cloud_a800")]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import fla.ops.gated_delta_rule
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration, StoppingCriteriaList
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast
from checkpoints import save, seal, verify
from training_data import BASE, EMPTY_THINK, encode, processor
from trains.mopd.anvil_runtime import AnvilBatchRouter, validate_teacher_eligibility
from trains.mopd.chunked_heads import stage_sft_from_hidden, selected_mtd_from_hidden
from eval_core import BUDGETS
from memory_loss import chunked_ce
from completion_policy import student_diagnostics, POLICY
from mopd_lineage import inventory, bind_lineage, validate_runtime
from mopd_representation import Representation, RepresentationRouter, ExactCycleStop, install_linear_bytelevel_spans

EXPOSURE_CATEGORIES=('native_distillation','general_qa','spatial_reference')


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def install_hidden_forward():
    cls = Qwen3_5ForConditionalGeneration
    original = cls.forward
    @functools.wraps(original)
    def forward(self, *args, anvil_hidden_only=False, **kwargs):
        if not anvil_hidden_only:
            return original(self, *args, **kwargs)
        assert not args and kwargs.pop("labels", None) is None
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("return_dict", None)
        result = self.model(return_dict=True, **kwargs)
        return result.last_hidden_state
    cls.forward = forward


class FullPool:
    def __init__(self):
        index = json.loads((RECOVERY / "training_repair/full_training_index/manifest.json").read_text())
        self.export = Path(index["export"])
        assert sha(self.export / "manifest.json") == index["export_manifest_sha256"]
        self.groups = {f: [] for f in "GSA"}
        self.counts = {f: 0 for f in "GSA"}
        for name, info in sorted(index["files"].items()):
            family = info["family"]
            if family == "QA":
                continue
            assert sha(info["index_file"]) == info["index_sha256"]
            offsets = np.load(info["index_file"], mmap_mode="r")
            assert len(offsets) == info["records"]
            self.counts[family] += len(offsets)
            self.groups[family].append((self.counts[family], name, offsets))
        assert sum(self.counts.values()) == 4756000
        self.handles = {}

    def sample(self, micro, rank, world):
        family = "GSA"[micro % 3]
        ordinal = (micro // 3) * world + rank
        count = self.counts[family]
        multiplier = 1000003
        while math.gcd(multiplier, count) != 1:
            multiplier += 2
        selected = (ordinal * multiplier + 17) % count
        groups = self.groups[family]
        group = bisect.bisect_right([v[0] for v in groups], selected)
        prior = groups[group - 1][0] if group else 0
        _, name, offsets = groups[group]
        offset, size = offsets[selected - prior]
        if name not in self.handles:
            self.handles[name] = (self.export / name).open("rb")
        handle = self.handles[name]
        handle.seek(int(offset))
        row = json.loads(handle.read(int(size)))
        assert row["gsa_family"] == family and row["gsa_meta"]["split"] == "train"
        return row


class ReplayPool(FullPool):
    def __init__(self, spatial_replay=False):
        super().__init__()
        path=ROOT/'diagnosis_32gpu/retention_replay'
        self.replay_manifest=path/'manifest.json';manifest=json.loads(self.replay_manifest.read_text())
        assert manifest['state']=='verified_train_only'
        self.qa={}
        for name in ['ai2d','scienceqa','textvqa']:
            file=path/(name+'.jsonl');assert sha(file)==manifest['files_sha256'][name]
            self.qa[name]=[json.loads(line) for line in file.open()]
            assert len(self.qa[name])==manifest['additional_training_records'][name]
        self.spatial_replay = spatial_replay
        self.spatial = {}
        self.spatial_manifest = None
        if spatial_replay:
            path=ROOT/'diagnosis_32gpu/spatial_replay'
            self.spatial_manifest=path/'manifest.json'
            manifest=json.loads(self.spatial_manifest.read_text())
            assert manifest['state']=='verified_train_only'
            for name in ['fsd_goal','refspatial_vacancy']:
                file=path/(name+'.jsonl');assert sha(file)==manifest['files_sha256'][name]
                self.spatial[name]=[json.loads(line) for line in file.open()]
                assert len(self.spatial[name])==manifest['additional_training_records'][name]
                for row in self.spatial[name]:
                    assert row['gsa_meta']['split']=='train' and row['gsa_form']=='point'
                    row['anvil_replay_source']=name

    @staticmethod
    def pick(rows, occurrence):
        multiplier=1000003
        while math.gcd(multiplier,len(rows))!=1:multiplier+=2
        return rows[(occurrence*multiplier+17)%len(rows)]

    def sample(self,micro,rank,world):
        if self.spatial_replay:
            if micro%2:
                return super().sample(micro//2,rank,world)
            ordinal=micro//4
            if micro%4==2:
                slot=ordinal%4;name='refspatial_vacancy' if slot==3 else 'fsd_goal'
                occurrence=ordinal//4 if slot==3 else (ordinal//4)*3+slot
                return self.pick(self.spatial[name],occurrence*world+rank)
            name=['ai2d','scienceqa','textvqa'][ordinal%3]
            return self.pick(self.qa[name],(ordinal//3)*world+rank)
        if micro%4:
            return super().sample(micro-(micro//4+1),rank,world)
        ordinal=micro//4;name=['ai2d','scienceqa','textvqa'][ordinal%3];rows=self.qa[name]
        occurrence=(ordinal//3)*world+rank;multiplier=1000003
        while math.gcd(multiplier,len(rows))!=1:multiplier+=2
        row=rows[(occurrence*multiplier+17)%len(rows)]
        assert row['gsa_meta']['split']=='train' and row['gsa_family']=='QA'
        return row


def move(data, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}


def forward_inputs(data):
    return {k: v for k, v in data.items() if k not in {"labels", "token_type_ids"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--eligibility", type=Path)
    ap.add_argument("--student-diagnostics", type=Path, help="Optional diagnostic evidence; SFT retention scores do not gate MOPD")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-forms", default="bbox,direction,affordance")
    ap.add_argument("--smoke-token-cap", type=int, default=64)
    ap.add_argument("--smoke-dataset", type=Path)
    ap.add_argument("--smoke-reference-rollout", action="store_true")
    ap.add_argument("--teacher-device", default=None)
    ap.add_argument('--dof-representation',choices=['native','rotation6d','rotation128'],default='native')
    ap.add_argument('--spatial-replay',action='store_true')
    ap.add_argument('--runtime-bindings',type=Path,help='Exact adapter overrides for execution-only smoke/probe; never production eligibility')
    ap.add_argument('--full-budget-probe',action='store_true',help='Execution-only sampling of the real pool at full rollout budgets; never formal MOPD')
    ap.add_argument("--initialization-lineage",type=Path,help="Audited ancestry of the exact initial student, required for formal training")
    ap.add_argument("--save-every", type=int, default=64)
    args = ap.parse_args()
    if args.full_budget_probe:
        assert args.smoke and not args.smoke_dataset and not args.smoke_reference_rollout
    assert not args.runtime_bindings or args.smoke
    assert args.smoke or not (args.smoke_dataset or args.smoke_reference_rollout)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    teacher_device = torch.device(args.teacher_device or device)
    if world > 1:
        dist.init_process_group("nccl")
    assert args.steps > 0 and args.global_batch % world == 0
    accumulation = args.global_batch // world
    torch.set_num_threads(4)
    torch.manual_seed(17 + rank)
    random.seed(17 + rank)
    np.random.seed(17 + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    status = json.loads((RECOVERY.parent / "a800_repair_20261020/pipeline_status.json").read_text())
    adapters = {f: status["checkpoints"][f]["checkpoint"] for f in "GSA"}
    initial = status["checkpoints"]["joint"]["checkpoint"]
    for f, c in status["checkpoints"].items():
        assert sha(Path(c["checkpoint"]) / "adapter_model.safetensors") == c["adapter_sha256"]
    install_linear_bytelevel_spans()
    representation = Representation(args.dof_representation)
    runtime_binding_sha256 = None
    if args.runtime_bindings:
        binding=json.loads(args.runtime_bindings.read_text())
        assert binding['schema']=='anvil-execution-only-checkpoint-binding-v1'
        assert binding['teachers_qualified'] is False and binding['formal_mopd'] is False
        assert set(binding['teachers'])==set('GSA')
        for item in [binding['student'],*binding['teachers'].values()]:
            assert sha(Path(item['adapter'])/'adapter_model.safetensors')==item['adapter_sha256']
        adapters={family:binding['teachers'][family]['adapter'] for family in 'GSA'}
        initial=binding['student']['adapter']
        runtime_binding_sha256=sha(args.runtime_bindings)
    if not args.smoke:
        assert args.eligibility, "Exact teacher eligibility remains required"
        eligibility=json.loads(args.eligibility.read_text())
        adapters={family:eligibility['teachers'][family]['adapter'] for family in 'GSA'}
        initial=eligibility['joint_adapter']
        assert eligibility.get('model_representation_sha256')==representation.digest, 'Eligibility must bind the chosen model interface'
        runtime_evidence=eligibility.get('runtime_evidence',{})
        evidence_path=Path(runtime_evidence.get('path',''))
        assert evidence_path.is_file() and sha(evidence_path)==runtime_evidence.get('sha256'), 'Missing versioned real-GPU runtime evidence'
        runtime_report=json.loads(evidence_path.read_text())
        assert runtime_report['state']=='passed_execution_only' and runtime_report['representation_sha256']==representation.digest
        validate_runtime(runtime_report,Path(__file__),initial,adapters)
        retention=student_diagnostics(args.student_diagnostics,initial)
    else:
        retention=student_diagnostics(args.student_diagnostics,initial)
    adapter_interfaces=representation.validate_adapters(adapters,initial)
    if not args.smoke:
        assert not adapter_interfaces['student']['student_requires_representation_adaptation'],'Adapt student representation before formal MOPD'
    lineage=bind_lineage(args.initialization_lineage,initial,required=not args.smoke)
    available=inventory()
    pool = ReplayPool(args.spatial_replay)
    os.environ.update(ANVIL_DATA_CONTRACT="anvil-source-native-train-v1", TRAIN=str(pool.export))
    if args.smoke:
        assert args.eligibility is None and args.steps <= 32
        qualified = dict.fromkeys("GSA", True)  # Execution-only; never an eligibility report.
    else:
        assert args.eligibility and not args.teacher_device
        qualified = validate_teacher_eligibility(args.eligibility, adapters, initial)
    protocol = dict(schema="anvil-native-on-policy-final-outcome-v6", base=BASE,
                    initial=initial, teachers=adapters, scope="all_11_native_forms",
                    spatial_pool=4756000, sft_pool_already_covered=5220000,
                    train_export_manifest_sha256=sha(pool.export / "manifest.json"),
                    world_size=world, global_batch=args.global_batch,
                    updates=args.steps if not args.smoke else "smoke_resume_variable",
                    reference_context=16384, rollout_budgets=BUDGETS,
                    beta=.5, temperature=.9, topk=128, structural_alpha=2,
                    lr=5e-6, seed=17, native_router="declared family on atomic native rows",
                    sampling="family-balanced affine permutation per full family pool; repeats recorded",
                    smoke=args.smoke, quality_qualified=not args.smoke,
                    general_replay_fraction=.25,additional_replay_pool=50563,
                    spatial_reference_replay_fraction=.25 if args.spatial_replay else 0.,
                    additional_spatial_replay_pool=32832 if args.spatial_replay else 0,
                    spatial_replay_manifest_sha256=sha(pool.spatial_manifest) if pool.spatial_manifest else None,
                    representation=representation.descriptor,model_representation_sha256=representation.digest,
                    model_representation=representation.descriptor['version'],decode_required_for_native_inference=args.dof_representation!='native',
                    adapter_interfaces=adapter_interfaces,runtime_bindings_sha256=runtime_binding_sha256,
                    full_budget_probe=args.full_budget_probe,
                    cycle_stop='Exact token cycle: >=512 generated tokens, check every32; period1..32, >=16 repeats and >=256 identical-span tokens. Keep original sampled prefix and record reason.',
                    token_spans='Exact linear ByteLevel/UTF8 mapping of actual sampled IDs; original prefix-decoding fallback for unsupported streams',
                    original_full_corpus_records=5800000,original_full_corpus_images=160000,
                    supplemental_qa_images=28665,available_union_images=available["counts"]["available_union_images"],
                    original_plus_supplement_available_training_records=available["counts"]["available_training_records"],
                    available_inventory_binding=available,student_initialization_lineage=lineage,
                    supplementary_PRISM_available_records=13536,supplementary_PRISM_direct_replay_in_this_stage=False,
                    inventory_is_actual_training_coverage=False,
                    active_mopd_unique_pool_capacity=4839395 if args.spatial_replay else 4806563,
                    report_qa_source_mixture_matched=False,
                    replay_manifest_sha256=sha(pool.replay_manifest),student_retention_diagnostic=retention,
                    completion_policy=POLICY,completion_policy_code_sha256=sha(ROOT/"completion_policy.py"),
                    exposure_rule=('50%native from full4756000 pool;25%reference-only broadQA;25%reference-only FSD/vacancy replay' if args.spatial_replay else '75%native from full4756000 pool;25%reference-only broadQA'),
                    eligibility_sha256=sha(args.eligibility) if args.eligibility else None,
                    smoke_dataset_sha256=sha(args.smoke_dataset) if args.smoke_dataset else None,
                    smoke_reference_rollout=args.smoke_reference_rollout,
                    smoke_token_cap=args.smoke_token_cap if args.smoke and not args.full_budget_probe else None,
                    smoke_forms=args.smoke_forms if args.smoke and not args.full_budget_probe else None,
                    code_sha256=sha(__file__),lineage_runtime_code_sha256=sha(ROOT/'mopd_lineage.py'),representation_runtime_code_sha256=sha(ROOT/'mopd_representation.py'))
    protocol_path = args.output / "protocol.json"
    if rank == 0:
        if protocol_path.exists():
            assert json.loads(protocol_path.read_text()) == protocol, "Resume protocol changed"
        else:
            save(protocol_path, protocol)
    if world > 1:
        dist.barrier()
    install_hidden_forward()
    student = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="sdpa", local_files_only=True)
    student = PeftModel.from_pretrained(student, str(args.resume or initial), is_trainable=True)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    trainable = [p for p in student.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == 173113344
    teacher = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map={"": teacher_device}, attn_implementation="sdpa", local_files_only=True)
    teacher = PeftModel.from_pretrained(teacher, adapters["G"], adapter_name="G", is_trainable=False)
    for family in "SA":
        teacher.load_adapter(adapters[family], adapter_name=family, is_trainable=False)
    teacher.requires_grad_(False)
    teacher.eval()
    p = processor()
    router = RepresentationRouter(representation)
    optimizer = torch.optim.AdamW(trainable, lr=5e-6, weight_decay=.01, fused=True)
    start = 0
    exposure_totals=dict.fromkeys(EXPOSURE_CATEGORIES,0)
    if args.resume:
        verify(args.resume, world)
        optimizer.load_state_dict(torch.load(args.resume / "optimizer.pt", map_location=device, weights_only=True))
        start = json.loads((args.resume / "trainer_state.json").read_text())["global_step"]
        exposure_totals=json.loads((args.resume/'trainer_state.json').read_text())['source_exposures']
        saved = torch.load(args.resume / (f"rng_state_{rank}.pth" if world > 1 else "rng_state.pth"),
                           map_location="cpu", weights_only=False)
        torch.set_rng_state(saved["cpu"])
        torch.cuda.set_rng_state(saved["cuda"], device)
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
    model = DistributedDataParallel(student, device_ids=[local], broadcast_buffers=False,
                                   find_unused_parameters=False) if world > 1 else student
    head = student.get_output_embeddings()
    t_head = teacher.get_output_embeddings()
    assert not any(x.requires_grad for x in head.parameters())
    forms = args.smoke_forms.split(",")
    fixtures = {}
    if args.smoke and not args.full_budget_probe:
        if args.smoke_dataset:
            for line in args.smoke_dataset.open():
                row = json.loads(line)
                form = row['gsa_form']
                if form in forms and (form not in fixtures or len(row['messages'][1]['content']) > len(fixtures[form]['messages'][1]['content'])):
                    fixtures[form] = row
            assert set(fixtures) == set(forms)
        else:
            for form in forms:
                if form=='QA':
                    fixtures[form]=pool.sample(0,0,1)
                    continue
                fixtures[form] = json.loads((RECOVERY / "training_repair/encoding_probe" / (form + ".json")).read_text())
    begin = time.monotonic()
    logs = json.loads((args.resume / 'trainer_state.json').read_text())['log_history'] if args.resume else []
    def checkpoint(step):
        dest = args.output / f"checkpoint-{step}"
        dest.mkdir(exist_ok=True)
        if rank == 0:
            student.save_pretrained(dest, safe_serialization=True)
            torch.save(optimizer.state_dict(), dest / "optimizer.pt")
            torch.save({"schedule": "linear-warmup3%-cosine", "step": step}, dest / "scheduler.pt")
            save(dest / "trainer_state.json", dict(global_step=step, max_steps=args.steps,
                 epoch=exposure_totals['native_distillation']/4756000, log_history=logs, smoke=args.smoke,
                 examples_seen=step * args.global_batch, source_exposures=exposure_totals,
                 qa_examples_seen=exposure_totals['general_qa'], spatial_examples_seen=exposure_totals['native_distillation']+exposure_totals['spatial_reference'], final_mopd=not args.smoke))
        torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
                        python=random.getstate(), numpy=np.random.get_state()),
                   dest / (f"rng_state_{rank}.pth" if world > 1 else "rng_state.pth"))
        if world > 1:
            dist.barrier()
        if rank == 0:
            seal(dest, world)
            save(args.output / "latest.json", dict(path=str(dest), step=step, smoke=args.smoke))
        if world > 1:
            dist.barrier()
        return dest
    with (args.output / f"exposure_rank{rank}.jsonl").open("a") as exposure:
        for step in range(start, args.steps):
            optimizer.zero_grad(set_to_none=True)
            t0 = time.monotonic()
            scores = []
            step_counts=torch.zeros(3,device=device,dtype=torch.long)
            cycle_count=torch.zeros(1,device=device,dtype=torch.long)
            warmup = max(1, math.ceil(args.steps * .03))
            ratio = min(1., (step + 1) / warmup)
            if step >= warmup:
                ratio = .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = 5e-6 if args.smoke else 5e-6 * ratio
            for micro in range(accumulation):
                ordinal = step * accumulation + micro
                source_row = deepcopy(fixtures[forms[ordinal % len(forms)]]) if args.smoke and not args.full_budget_probe else pool.sample(ordinal, rank, world)
                category='general_qa' if source_row['gsa_family']=='QA' else 'spatial_reference' if source_row.get('anvil_replay_source') else 'native_distillation'
                step_counts[EXPOSURE_CATEGORIES.index(category)]+=1
                row = representation.row(source_row) if category=='native_distillation' else source_row
                data = move(encode(p, row), device)
                if category!='native_distillation':
                    # All ranks use the same form schedule; QA is supervised replay only.
                    student.train()
                    sync=model.no_sync() if world>1 and micro<accumulation-1 else nullcontext()
                    with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                        hidden=model(**forward_inputs(data),anvil_hidden_only=True,use_cache=False)
                        loss=chunked_ce(hidden,head,data['labels'])
                        assert torch.isfinite(loss)
                        (loss/accumulation).backward()
                    scores.append([float(loss.detach()),0.])
                    exposure.write(json.dumps(dict(step=step+1,micro=micro,rank=rank,canonical_uid=row['canonical_uid'],family=row['gsa_family'],form=row['gsa_form'],source_category=category,replay_source=row.get('anvil_replay_source'),source_binding=row['gsa_meta']['binding'],reference_tokens=int((data['labels']!=-100).sum()),rollout_tokens=0,teacher_scored_positions=0,smoke=args.smoke))+'\n');exposure.flush()
                    del data,hidden,loss
                    continue
                reference_plan = router.plan([row], data, p.tokenizer, reference=True,
                                             qualified=qualified, reference_prefix=EMPTY_THINK)
                target_start = int((data["labels"][0] != -100).nonzero()[0])
                prompt_length = target_start + len(p.tokenizer.encode(EMPTY_THINK, add_special_tokens=False))
                prompt = forward_inputs(data)
                prompt["input_ids"] = prompt["input_ids"][:, :prompt_length]
                prompt["attention_mask"] = prompt["attention_mask"][:, :prompt_length]
                if "mm_token_type_ids" in prompt:
                    prompt["mm_token_type_ids"] = prompt["mm_token_type_ids"][:, :prompt_length]
                student.eval()
                budget = args.smoke_token_cap if args.smoke and not args.full_budget_probe else BUDGETS[row["gsa_form"]]
                cycle_stop=ExactCycleStop(prompt_length)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.smoke_reference_rollout:
                        # Explicit memory stress only: use the unchanged long training
                        # response, with all its tokens retained. Never production MOPD.
                        ids = data['input_ids'].clone()
                    else:
                        ids = student.generate(**prompt, max_new_tokens=budget, do_sample=True,
                            temperature=.9, top_p=1., top_k=0, use_cache=True,
                            logits_to_keep=1, eos_token_id=p.tokenizer.eos_token_id,
                            pad_token_id=p.tokenizer.pad_token_id,stopping_criteria=StoppingCriteriaList([cycle_stop]))
                assert torch.equal(ids[:, :prompt_length], prompt["input_ids"])
                cycle_count[0]+=int(cycle_stop.cycle is not None)
                sampled = dict(prompt, input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
                if "mm_token_type_ids" in prompt:
                    sampled["mm_token_type_ids"] = torch.cat((prompt["mm_token_type_ids"],
                        torch.zeros_like(ids[:, prompt_length:])), dim=1)
                sampled["labels"][:, :prompt_length] = -100
                plan = router.plan([row], sampled, p.tokenizer, reference=False, qualified=qualified)
                teacher.set_adapter(row["gsa_family"])
                teacher.requires_grad_(False)
                teacher.eval()
                student.train()
                sync = model.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model(**forward_inputs(data), anvil_hidden_only=True, use_cache=False)
                    sft = stage_sft_from_hidden(hidden[:, :-1], head, reference_plan.token_ids,
                                                reference_plan.stage_ids, chunk_size=128)
                    (sft / accumulation).backward()
                    del hidden
                    student_hidden = model(**forward_inputs(sampled), anvil_hidden_only=True, use_cache=False)
                    with torch.no_grad():
                        teacher_hidden = teacher(**move(forward_inputs(sampled), teacher_device),
                                                 anvil_hidden_only=True, use_cache=False)
                    def teacher_head(x):
                        return t_head(x.to(teacher_device)).to(device)
                    mtd = selected_mtd_from_hidden(student_hidden[:, :-1], teacher_hidden[:, :-1].to(device),
                        head, teacher_head, plan.token_ids, plan.stage_ids, plan.selected_stages,
                        plan.structural_mask, topk=128, beta=.5, structural_alpha=2., temperature=.9, chunk_size=16)
                    eligible = (plan.selected_stages >= 0).sum().to(torch.float32)
                    global_eligible = eligible.clone()
                    if world > 1:
                        dist.all_reduce(global_eligible)
                    weighted_mtd = mtd * (eligible * world / global_eligible.clamp_min(1))
                    (weighted_mtd / accumulation).backward()
                    del student_hidden, teacher_hidden
                assert torch.isfinite(sft) and torch.isfinite(mtd)
                scores.append([float(sft.detach()), float(mtd.detach())])
                exposure.write(json.dumps(dict(step=step + 1, micro=micro, rank=rank,
                    canonical_uid=row["canonical_uid"], family=row["gsa_family"], form=row["gsa_form"],
                    source_category=category,source_binding=source_row['gsa_meta']['binding'],
                    model_representation_sha256=representation.digest,
                    reference_tokens=int((reference_plan.stage_ids >= 0).sum()),
                    rollout_tokens=ids.shape[1] - prompt_length,
                    rollout_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                    rollout_stop_reason='reference_memory_stress' if args.smoke_reference_rollout else 'exact_cycle' if cycle_stop.cycle else 'eos' if int(ids[0,-1])==p.tokenizer.eos_token_id else 'budget',
                    exact_cycle=cycle_stop.cycle,
                    teacher_scored_positions=int((plan.stage_ids >= 0).sum()), smoke=args.smoke)) + "\n")
                exposure.flush()
                del data, sampled, prompt, ids, plan, reference_plan, sft, mtd, weighted_mtd
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            assert torch.isfinite(norm), "Non-finite gradient"
            optimizer.step()
            score_tensor=torch.tensor(scores,device=device,dtype=torch.float64).mean(0)
            if world>1:
                dist.all_reduce(step_counts);dist.all_reduce(cycle_count);dist.all_reduce(score_tensor);score_tensor/=world
            for key,count in zip(EXPOSURE_CATEGORIES,step_counts.tolist()):exposure_totals[key]+=count
            assert sum(exposure_totals.values())==(step+1)*args.global_batch
            entry = dict(step=step + 1, max_steps=args.steps, sft=float(score_tensor[0]),
                         mtd=float(score_tensor[1]), grad_norm=float(norm),
                         seconds=time.monotonic() - t0, elapsed=time.monotonic() - begin,
                         examples_seen=(step + 1) * args.global_batch, source_exposures=dict(exposure_totals),
                         exact_cycle_stops=int(cycle_count[0]),
                         qa_examples_seen=exposure_totals['general_qa'], spatial_examples_seen=exposure_totals['native_distillation']+exposure_totals['spatial_reference'],
                         peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30)
            logs.append(entry)
            if rank == 0:
                with (args.output / "logging.jsonl").open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                save(args.output / "status.json", dict(stage="training", **entry, smoke=args.smoke))
                print(json.dumps(entry), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                checkpoint(step + 1)
    if rank == 0:
        save(args.output / "complete.json", dict(stage="smoke_complete" if args.smoke else "mopd_training_complete",
            checkpoint=str(args.output / f"checkpoint-{args.steps}"), steps=args.steps,
            examples_seen=args.steps * args.global_batch, source_exposures=exposure_totals,
            qa_examples_seen=exposure_totals['general_qa'], spatial_examples_seen=exposure_totals['native_distillation']+exposure_totals['spatial_reference'], smoke=args.smoke,
            validation_complete=False, weights_uploaded=False, completed_at=time.time()))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
