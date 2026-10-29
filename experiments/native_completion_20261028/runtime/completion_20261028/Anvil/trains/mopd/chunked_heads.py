"""The released stage objectives, projected one bounded head chunk at a time.

These helpers preserve the full-vocabulary normalizer, shared top-k support,
structural-token union, residual mass and per-stage/per-example reductions.
They do not modify routing, target tokens or teacher eligibility.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .firstfault_core import _anvil_compressed_log_probs


def stage_sft_from_hidden(hidden, head, targets, stages, *, chunk_size=128):
    assert hidden.shape[:2] == targets.shape == stages.shape
    assert chunk_size > 0
    total = hidden[..., :0].sum()
    for row in range(hidden.shape[0]):
        present = [stage for stage in range(3) if bool((stages[row] == stage).any())]
        assert present, "Every SFT sample requires at least one reference stage"
        for stage in present:
            positions = (stages[row] == stage).nonzero().flatten()

            def score(states, ids):
                logits = head(states)
                dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
                assert torch.isfinite(logits).all()
                return F.cross_entropy(logits.to(dtype), ids.long(), reduction="sum")

            for indices in positions.split(chunk_size):
                piece = checkpoint(score, hidden[row, indices], targets[row, indices],
                                   use_reentrant=False)
                total = total + piece / (len(positions) * len(present) * len(hidden))
    return total


def selected_mtd_from_hidden(student_hidden, teacher_hidden, student_head, teacher_head,
                             token_ids, stages, selected_stages, structural_mask, *,
                             topk=128, beta=.5, structural_alpha=2., temperature=.9,
                             chunk_size=32):
    assert student_hidden.shape == teacher_hidden.shape
    assert student_hidden.shape[:2] == token_ids.shape == stages.shape == structural_mask.shape
    assert topk > 0 and chunk_size > 0 and 0 < beta < 1
    assert math.isfinite(temperature) and temperature > 0
    assert math.isfinite(structural_alpha) and structural_alpha >= 0
    selected = (stages == selected_stages[:, None]) & (selected_stages[:, None] >= 0)
    total = student_hidden[..., :0].sum()
    count = 0
    for row in range(len(student_hidden)):
        positions = selected[row].nonzero().flatten()
        if not len(positions):
            continue
        count += 1
        structural_ids = token_ids[row, selected[row] & structural_mask[row]].unique().long()
        dtype = torch.float64 if student_hidden.dtype == torch.float64 else torch.float32
        weights = 1 + structural_alpha * structural_mask[row, positions].to(dtype)
        weights = weights / weights.mean()

        # Pass row-specific support and weights as explicit checkpoint inputs:
        # a closure over loop variables would read the final row during backward.
        def score(s, t, support, weight):
            student = student_head(s).to(dtype) / temperature
            with torch.no_grad():
                teacher = teacher_head(t.detach()).to(dtype) / temperature
            assert torch.isfinite(student).all() and torch.isfinite(teacher).all()
            log_p, log_q = _anvil_compressed_log_probs(student, teacher, support, topk)
            occupied = torch.isfinite(log_p) | torch.isfinite(log_q)
            safe_p, safe_q = log_p.masked_fill(~occupied, 0), log_q.masked_fill(~occupied, 0)
            mixture = torch.logaddexp(safe_q + math.log(beta), safe_p + math.log1p(-beta))
            divergence = (beta * log_q.exp() * (safe_q - mixture)
                          + (1 - beta) * log_p.exp() * (safe_p - mixture)).sum(-1)
            return (divergence * weight).sum()

        for offset in range(0, len(positions), chunk_size):
            indices = positions[offset:offset + chunk_size]
            piece = checkpoint(score, student_hidden[row, indices], teacher_hidden[row, indices],
                               structural_ids, weights[offset:offset + chunk_size],
                               use_reentrant=False)
            total = total + piece / len(positions)
    return total / count if count else total
