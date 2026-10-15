"""ms-swift plugin for the MOPD baseline and Anvil on-policy optimization.

The plugin loads three frozen LoRA adapters into one shared teacher backbone.
Each batch is balanced and homogeneous in declared ``gsa_family``; the adapter
is selected before the student's rollout and the teacher forward pass. This is
vanilla routed MOPD by default. ANVIL_MTD=1 selects the separately implemented
sampled-stage route, reference SFT and residual-preserving selected-span loss.
"""

from __future__ import annotations

import os
import importlib
import json
import math
import sys
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peft import PeftModel
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import swift.rlhf_trainers as swift_rlhf_trainers
from swift.dataloader import DataLoaderShard
from swift.rlhf_trainers import GKDTrainer
from swift.rlhf_trainers.gkd_trainer import HFSFTTrainer
from swift.trainers import disable_gradient_checkpointing
from swift.trainers import TrainerFactory
from swift.utils import seed_worker, to_device, unwrap_model_for_generation

from trains.mopd.routing import FamilyBalancedBatchSampler, dataset_families
from trains.mopd.anvil_runtime import AnvilBatchRouter, FAMILIES, validate_teacher_eligibility
from trains.mopd.firstfault_core import anvil_selected_span_mtd_loss, anvil_stage_sft_loss


def _adapter_paths() -> dict[str, str]:
    paths = {
        family: os.environ.get(f"GSA_TEACHER_{family}_ADAPTER", "")
        for family in (("G", "S") if os.environ.get('ANVIL_EXPERIMENT_SCOPE')=='GS_MECHANISM' else ("G", "S", "A"))
    }
    missing = [family for family, path in paths.items() if not path]
    if missing:
        raise ValueError(f"missing teacher adapter environment variables for {missing}")
    invalid = [f"{family}:{path}" for family, path in paths.items() if not Path(path, "adapter_config.json").is_file()]
    if invalid:
        raise ValueError(f"teacher adapter_config.json not found: {invalid}")
    return paths


def _find_adapter_model(model):
    current = model
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "set_adapter") and hasattr(current, "peft_config"):
            return current
        next_model = getattr(current, "module", None) or getattr(current, "model", None)
        if next_model is None:
            break
        current = next_model
    raise TypeError(f"cannot find a PEFT teacher inside {type(model).__name__}")


class GSARoutedMOPDTrainer(GKDTrainer):
    """GKD with declared-family routing over G/S/A teacher adapters."""

    def __init__(self, model=None, *_args, **kwargs):
        teacher_model = kwargs.get("teacher_model")
        if teacher_model is None:
            raise ValueError("routed MOPD requires a separate teacher backbone")
        adapters = _adapter_paths()
        teacher_model = PeftModel.from_pretrained(
            teacher_model,
            adapters["G"],
            adapter_name="G",
            is_trainable=False,
        )
        for family in adapters:
            if family == 'G':continue
            teacher_model.load_adapter(
                adapters[family],
                adapter_name=family,
                is_trainable=False,
            )
        teacher_model.set_adapter("G")
        teacher_model.requires_grad_(False)
        kwargs["teacher_model"] = teacher_model
        super().__init__(model, *_args, **kwargs)
        self._gsa_active_family = "G"

    def get_train_dataloader(self, skip_batches=0):
        if self.train_dataset is None or not hasattr(self.train_dataset, "__len__"):
            raise ValueError("routed MOPD requires a finite map-style dataset")
        args = self.args
        families = dataset_families(self.train_dataset)
        scoped = os.environ.get('ANVIL_EXPERIMENT_SCOPE')=='GS_MECHANISM'
        if scoped and set(families)!={'S'}:
            raise ValueError('GS mechanism study requires only linked G-to-S terminal rows')
        sampler = FamilyBalancedBatchSampler(
            families,
            batch_size=self._train_batch_size,
            seed=args.data_seed or 0,
            required_families=('S',) if scoped else FAMILIES,
        )
        if skip_batches:
            from accelerate.data_loader import SkipBatchSampler
            sampler = SkipBatchSampler(sampler, skip_batches=skip_batches)
        params = {
            "batch_sampler": sampler,
            "collate_fn": self.data_collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
            "persistent_workers": args.dataloader_persistent_workers if args.dataloader_num_workers else False,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=args.dataloader_num_workers,
                rank=args.process_index,
            ),
        }
        if args.dataloader_num_workers:
            params["prefetch_factor"] = args.dataloader_prefetch_factor
        return DataLoaderShard(self.train_dataset, device=self.accelerator.device, **params)

    @staticmethod
    def _batch_family(inputs: list[Mapping[str, object]]) -> str:
        families = {str(row.get("gsa_family", "")) for row in inputs}
        if len(families) != 1 or next(iter(families)) not in {"G", "S", "A"}:
            raise ValueError(f"routed MOPD batch is not family-homogeneous: {sorted(families)}")
        return next(iter(families))

    def _activate_teacher(self, family: str) -> None:
        teacher = _find_adapter_model(self.accelerator.unwrap_model(self.teacher_model))
        teacher.set_adapter(family)
        # PEFT set_adapter can re-enable trainability even after loading frozen
        # adapters; re-freeze the entire teacher after every route switch.
        teacher.requires_grad_(False)
        self._gsa_active_family = family

    def _prepare_batch_inputs(self, inputs, encode_prompt_only=False):
        from collections.abc import Mapping as BatchMapping
        if isinstance(inputs, BatchMapping):
            if inputs.get("gsa_contract_version") == "anvil-source-native-train-v1":
                raise ValueError("Native MTD requires a list of source records, not a single mapping")
            native_batch = False
        else:
            native_batch = any(row.get("gsa_contract_version") == "anvil-source-native-train-v1" for row in inputs)
        if native_batch:
            # Swift's CLI "delete" becomes template "raise". Anvil's custom
            # training_step calls this encoder directly, without GKD resampling.
            # Never silently clip native prompts or supervised references.
            if getattr(self.template, "truncation_strategy", None) != "raise":
                raise ValueError("Native MTD requires non-truncating template encoding (CLI --truncation_strategy delete)")
            limit = getattr(self.template, "max_length", None)
            if type(limit) is not int or limit <= 0:
                raise ValueError("Native MTD requires a finite positive template context limit")
            # Upstream GKD calls _template_context without max_length, which
            # temporarily clears the configured limit. Preserve the same SDK
            # encoding/collation path but pass the native limit explicitly.
            from swift.rlhf_trainers.utils import replace_assistant_response_with_ids
            rows = []
            mode = "transformers" if encode_prompt_only else "train"
            with self._template_context(self.template, max_length=limit, mode=mode):
                for original in inputs:
                    data = deepcopy(original)
                    if data.get("response_token_ids"):
                        data["messages"] = replace_assistant_response_with_ids(data["messages"], data["response_token_ids"])
                    messages = data.get("messages", [])
                    if encode_prompt_only and messages and messages[-1].get("role") == "assistant":
                        messages[-1]["content"] = None
                    rows.append(self.template.encode(data, return_length=True))
                encoded = to_device(self.template.data_collator(rows), self.model.device)
        else:
            encoded = super()._prepare_batch_inputs(inputs, encode_prompt_only=encode_prompt_only)
        # Keep routing/lineage on the raw rows. Swift's remove_unused_columns=False
        # otherwise forwards it through the template into model.generate/forward.
        return {key: value for key, value in encoded.items()
                if key not in {"id", "canonical_uid", "sample_id", "record_id", "group_id", "source", "split",
                               "teacher_role", "spatial_task"} and not key.startswith("gsa_")}

    def training_step(self, model, inputs, num_items_in_batch=None):
        family = self._batch_family(inputs)
        self._activate_teacher(family)
        return super().training_step(model, inputs, num_items_in_batch)


class AnvilOnPolicyTrainer(GSARoutedMOPDTrainer):
    """Two student forwards: reference SFT plus routed student-rollout MTD.

    Every rank visits the same active teacher families in G/S/A order, including
    a dummy scoring position on ranks with no local route for an active family.
    This keeps ZeRO-3 teacher parameter gathers aligned across ranks. Actual
    teacher scoring positions, including padding/dummy overhead, are logged.
    """

    def __init__(self, model=None, *_args, **kwargs):
        router = kwargs.pop("anvil_router", None)
        selector = os.environ.get("ANVIL_ROUTING_MODE", "semantic")
        if selector not in {"semantic", "structural", "declared", "flat"}:
            raise ValueError(f"unknown Anvil selector: {selector}")
        alpha = float(os.environ.get("ANVIL_STRUCTURAL_ALPHA", "0" if selector == "flat" else "2"))
        if not math.isfinite(alpha) or alpha < 0 or (selector == "flat" and alpha != 0):
            raise ValueError("structural alpha must be nonnegative; the flat control requires alpha=0")
        if router is not None and router.selector != selector:
            raise ValueError("supplied router disagrees with ANVIL_ROUTING_MODE")
        if kwargs.get("eval_dataset") is not None:
            raise ValueError("Anvil runtime currently supports training only; evaluate completed checkpoints with the frozen benchmark harness")
        # Reject a missing/nonpassing binding before GSARoutedMOPDTrainer loads
        # three adapters or initializes any teacher/student trainer state.
        qualified = validate_teacher_eligibility(os.environ.get("TEACHER_ELIGIBILITY_JSON"),
                                                 _adapter_paths(), os.environ.get("STUDENT_ADAPTER"))
        super().__init__(model, *_args, **kwargs)
        self.model_accepts_loss_kwargs = False
        if (self.lmbda != 1 or self.seq_kd or self.args.use_vllm or self.use_teacher_api
                or self.use_liger_gkd_loss or self.template.padding_free or self.gkd_logits_topk is not None):
            raise ValueError("Anvil requires local on-policy generation, unpacked sequences and full local teacher logits")
        if self.template.enable_thinking is not False:
            raise ValueError("Anvil structured-output routing requires enable_thinking=false")
        evaluator = None
        if os.environ.get("ANVIL_STAGE_EVALUATOR"):
            module, name = os.environ["ANVIL_STAGE_EVALUATOR"].rsplit(":", 1)
            evaluator = getattr(importlib.import_module(module), name)
            if not callable(evaluator):
                raise ValueError("ANVIL_STAGE_EVALUATOR must name a callable")
        self.anvil_router = router or AnvilBatchRouter(evaluator, selector=selector)
        self.anvil_qualified = qualified
        self.anvil_data_contract = os.environ.get("ANVIL_DATA_CONTRACT", "")
        self.anvil_options = {"topk": int(os.environ.get("ANVIL_TOPK", "128")), "beta": float(self.args.beta),
                              "structural_alpha": alpha,
                              "temperature": float(self.temperature),
                              "chunk_size": int(os.environ.get("ANVIL_LOGIT_CHUNK", "32"))}
        self.anvil_sft_weight = float(os.environ.get("ANVIL_LAMBDA_SFT", "1"))
        self.anvil_mtd_weight = float(os.environ.get("ANVIL_LAMBDA_MTD", "1"))
        if any(not math.isfinite(value) or value <= 0 for value in (self.anvil_sft_weight, self.anvil_mtd_weight)):
            raise ValueError("Anvil SFT/MTD weights must be finite and positive")
        self._anvil_rows = self._anvil_reference = None
        self.anvil_route_log = Path(self.args.output_dir) / f"anvil_routes_rank{self.args.process_index}.jsonl"

    def training_step(self, model, inputs, num_items_in_batch=None):
        # GKD's stock pure on-policy branch omits reference SFT and mutates the
        # assistant message during prompt encoding. Preserve references first.
        rows = deepcopy(inputs)
        if (any(row.get("gsa_contract_version") == "anvil-source-native-train-v1" for row in rows)
                and getattr(self, "anvil_data_contract", "") != "anvil-source-native-train-v1"):
            raise ValueError("Native rows require the native-qualified trainer initialization")
        if any(row.get("teacher_prompt") for row in rows):
            raise ValueError("Anvil teacher and student must use identical sampled prefixes")
        references = self._prepare_batch_inputs(deepcopy(rows), encode_prompt_only=False)
        prompt = self._prepare_batch_inputs(deepcopy(rows), encode_prompt_only=True)
        with unwrap_model_for_generation(model, self.accelerator,
                                         gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped:
            was_training = unwrapped.training
            unwrapped.eval()
            try:
                with torch.no_grad():
                    ids, attention, labels = self.generate_on_policy_outputs(
                        unwrapped, prompt, self.generation_config, self.processing_class.pad_token_id)
            finally:
                unwrapped.train(was_training)
        prompt.update(input_ids=ids, attention_mask=attention, labels=labels)
        self._anvil_rows, self._anvil_reference = rows, references
        try:
            return HFSFTTrainer.training_step(self, model, prompt, num_items_in_batch)
        finally:
            self._anvil_rows = self._anvil_reference = None

    @staticmethod
    def _forward_inputs(inputs):
        return {key: value for key, value in inputs.items()
                if key not in {"labels", "prompt", "loss_scale", "logits_to_keep"} and not key.startswith("_")}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self._anvil_rows is None or self._anvil_reference is None:
            raise ValueError("Anvil compute_loss requires fresh rollout and reference batch context")
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        reference = self._anvil_reference
        reference_plan = self.anvil_router.plan(self._anvil_rows, reference, tokenizer,
                                                 reference=True, qualified=self.anvil_qualified,
                                                 reference_prefix=getattr(self.template, "response_prefix", "") or "")
        plan = self.anvil_router.plan(self._anvil_rows, inputs, tokenizer,
                                      reference=False, qualified=self.anvil_qualified)
        if not torch.equal(plan.token_ids, inputs["input_ids"][:, 1:]):
            raise ValueError("route plan is not bound to the current sampled tokens")
        with self.template.forward_context(model, reference):
            reference_outputs = model(**self._forward_inputs(reference), use_cache=False)
        sft = anvil_stage_sft_loss(reference_outputs.logits[:, :-1], reference_plan.token_ids, reference_plan.stage_ids)
        del reference_outputs
        with self.template.forward_context(model, inputs):
            sampled_outputs = model(**self._forward_inputs(inputs), use_cache=False)
        student_logits = sampled_outputs.logits[:, :-1]
        if student_logits.shape[:2] != plan.stage_ids.shape:
            raise ValueError("student logits do not align with next-token rollout positions")
        local_counts = torch.stack([(plan.selected_stages == index).sum() for index in range(3)])
        global_counts = local_counts.clone()
        world = dist.get_world_size() if dist.is_initialized() else 1
        if dist.is_initialized():
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        global_eligible = global_counts.sum().clamp_min(1)
        mtd = student_logits[..., :0].sum()
        teacher_positions = 0
        teacher_forwards = []
        for index, family in enumerate(FAMILIES):
            if not bool(global_counts[index]):
                continue
            self._activate_teacher(family)
            family_selected = torch.where(plan.selected_stages == index, plan.selected_stages, -1)
            active = (plan.stage_ids == index) & (family_selected[:, None] == index)
            positions = active.any(0).nonzero().flatten()
            if positions.numel() == 0:
                positions = torch.zeros(1, dtype=torch.long, device=student_logits.device)
            teacher_inputs = self._forward_inputs(inputs)
            teacher_inputs["logits_to_keep"] = positions
            load = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with torch.no_grad(), load, disable_gradient_checkpointing(self.teacher_model, self.args.gradient_checkpointing_kwargs):
                with self.template.forward_context(self.teacher_model, inputs):
                    teacher_outputs = self.teacher_model(**teacher_inputs, use_cache=False)
            teacher_logits = teacher_outputs.logits
            selected_student = student_logits[:, positions]
            if teacher_logits.shape != selected_student.shape:
                raise ValueError("teacher must implement logits_to_keep without altering prefix/token alignment")
            teacher_positions += teacher_logits.shape[0] * teacher_logits.shape[1]
            teacher_forwards.append({"family": family, "rows": teacher_logits.shape[0],
                                     "positions_per_row": teacher_logits.shape[1],
                                     "useful_positions": int(active.sum()),
                                     "selected_examples": int(local_counts[index])})
            loss = anvil_selected_span_mtd_loss(selected_student, teacher_logits,
                                               plan.token_ids[:, positions], plan.stage_ids[:, positions],
                                               family_selected, plan.structural_mask[:, positions], **self.anvil_options)
            # One global conditional sample mean across all families. Averaging
            # per-family means would overweight a rare routed family.
            mtd = mtd + loss * (local_counts[index] * world / global_eligible)
            del teacher_outputs, teacher_logits
        loss = self.anvil_sft_weight * sft + self.anvil_mtd_weight * mtd
        self._metrics["train"]["anvil/sft"].append(float(sft.detach()))
        self._metrics["train"]["anvil/mtd"].append(float(mtd.detach()))
        self._metrics["train"]["anvil/eligible_local"].append(int(local_counts.sum()))
        self._metrics["train"]["anvil/teacher_positions_local"].append(teacher_positions)
        response_positions = inputs["labels"][:, 1:] != -100
        response_positions &= inputs["attention_mask"][:, 1:].bool()
        budget = {"schema": "anvil-local-forward-budget-v1", "examples": len(self._anvil_rows),
                  "sampled_attended_response_positions": int(response_positions.sum()),
                  "reference_loss_positions": int((reference_plan.stage_ids >= 0).sum()),
                  "student_reference_attended_positions": int(reference["attention_mask"].sum()),
                  "student_sampled_attended_positions": int(inputs["attention_mask"].sum()),
                  "mtd_loss_positions": int(((plan.stage_ids == plan.selected_stages[:, None])
                                              & (plan.selected_stages[:, None] >= 0)).sum()),
                  "teacher_forward_calls": len(teacher_forwards),
                  "teacher_scored_rows": sum(item["rows"] for item in teacher_forwards),
                  "teacher_scored_positions": teacher_positions,
                  "teacher_attended_prefix_positions": len(teacher_forwards) * int(inputs["attention_mask"].sum()),
                  "teacher_forwards": teacher_forwards}
        useful = sum(item["useful_positions"] for item in teacher_forwards)
        if useful != budget["mtd_loss_positions"] or useful > teacher_positions:
            raise ValueError("teacher budget accounting does not cover exactly the selected MTD positions")
        budget["teacher_padding_or_dummy_positions"] = teacher_positions - useful
        self._anvil_last_budget = budget
        for name in ("sampled_attended_response_positions", "reference_loss_positions", "mtd_loss_positions",
                     "teacher_forward_calls", "teacher_padding_or_dummy_positions"):
            self._metrics["train"][f"anvil/{name}_local"].append(budget[name])
        if self.anvil_route_log is not None:
            self.anvil_route_log.parent.mkdir(parents=True, exist_ok=True)
            with self.anvil_route_log.open("a") as handle:
                handle.write(json.dumps({"step": self.state.global_step, "routes": plan.decisions,
                                         "eligible_local": int(local_counts.sum()), "teacher_positions_local": teacher_positions,
                                         "loss_options": self.anvil_options, "budget": budget}) + "\n")
        return (loss, sampled_outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, *args, **kwargs):
        raise ValueError("Anvil training-only runtime does not define a teacher-based validation loss; use the frozen checkpoint evaluation harness")


swift_rlhf_trainers.GSARoutedMOPDTrainer = GSARoutedMOPDTrainer
swift_rlhf_trainers.AnvilOnPolicyTrainer = AnvilOnPolicyTrainer
TrainerFactory.TRAINER_MAPPING["gkd"] = ("swift.rlhf_trainers.AnvilOnPolicyTrainer" if os.environ.get("ANVIL_MTD") == "1"
                                       else "swift.rlhf_trainers.GSARoutedMOPDTrainer")
