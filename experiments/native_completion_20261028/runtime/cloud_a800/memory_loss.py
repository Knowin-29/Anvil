"""Exact causal CE without materializing sequence-length by vocabulary logits."""
import functools
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_ce(hidden, head, labels, *, denominator=None, chunk_size=128):
    targets = labels[:, 1:]
    active = targets != -100
    states = hidden[:, :-1][active]
    targets = targets[active]
    assert targets.numel() > 0
    total = states.sum() * 0

    def project_and_score(x, y):
        return F.cross_entropy(head(x).float(), y, reduction="sum")

    for offset in range(0, len(targets), chunk_size):
        total = total + checkpoint(project_and_score, states[offset:offset + chunk_size],
                                   targets[offset:offset + chunk_size], use_reentrant=False)
    return total / (targets.numel() if denominator is None else denominator)


def install():
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration, Qwen3_5CausalLMOutputWithPast)
    cls = Qwen3_5ForConditionalGeneration
    original = cls.forward
    if getattr(original, "_anvil_chunked", False):
        return

    @functools.wraps(original)
    def forward(self, input_ids=None, *args, labels=None, **kwargs):
        if labels is None:
            return original(self, input_ids, *args, **kwargs)
        if args:
            raise ValueError("Chunked training requires keyword model inputs")
        keep = kwargs.pop("logits_to_keep", 0)
        if not isinstance(keep, int) or keep != 0:
            raise ValueError("Set --use_logits_to_keep false; the chunked loss selects labels itself")
        denominator = kwargs.pop("num_items_in_batch", None)
        kwargs.pop("return_dict", None)
        result = self.model(input_ids=input_ids, return_dict=True, **kwargs)
        loss = chunked_ce(result.last_hidden_state, self.lm_head, labels, denominator=denominator)
        return Qwen3_5CausalLMOutputWithPast(loss=loss, logits=None,
                                            past_key_values=result.past_key_values,
                                            rope_deltas=result.rope_deltas)
    forward._anvil_chunked = True
    cls.forward = forward
