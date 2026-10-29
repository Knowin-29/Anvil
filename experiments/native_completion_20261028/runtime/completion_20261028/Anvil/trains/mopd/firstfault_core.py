"""Pure routing and distillation-loss primitives, including the Anvil objective.

This module has no trainer, model, or launcher dependency.  It turns complete
paired-intervention evidence into one of the three publication-level decisions
and provides token losses for a future trainer integration. Historical FirstFault
token-sum helpers are retained; the Anvil helpers implement the current
stage-normalized, selected-span manuscript objective separately.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F

from firstfault.routing import (
    FirstFaultRoute,
    RouteKind,
    confirm_state_use,
    response_scores_consistent,
    route_first_fault,
)


def _positive_scalar(value: object, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite scalar")
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return result


def _token_mask(reference: Tensor, token_mask: Tensor | None) -> Tensor:
    if token_mask is None:
        mask = torch.ones(reference.shape, dtype=torch.bool, device=reference.device)
    else:
        if not isinstance(token_mask, Tensor):
            raise TypeError("token_mask must be a torch.Tensor")
        if token_mask.dtype is not torch.bool:
            raise ValueError("token_mask must have bool dtype")
        if token_mask.shape != reference.shape:
            raise ValueError(
                f"token_mask shape {tuple(token_mask.shape)} != token shape "
                f"{tuple(reference.shape)}"
            )
        if token_mask.device != reference.device:
            raise ValueError("token_mask must be on the same device as the inputs")
        mask = token_mask
    if not bool(mask.any().item()):
        raise ValueError("token_mask selects no tokens")
    return mask


def _require_floating(value: Tensor, label: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise ValueError(f"{label} must have floating dtype")


def _require_active_finite(value: Tensor, mask: Tensor, label: str) -> None:
    if not bool(torch.isfinite(value[mask]).all().item()):
        raise ValueError(f"{label} contains a non-finite active value")


def dependency_advantage(
    repaired_log_probs: Tensor,
    predicted_log_probs: Tensor,
    token_mask: Tensor | None = None,
    *,
    detach: bool = True,
) -> Tensor:
    """Return ``[log p_T(y|c+) - log p_T(y|c-)]_+`` per token."""

    _require_floating(repaired_log_probs, "repaired_log_probs")
    _require_floating(predicted_log_probs, "predicted_log_probs")
    if repaired_log_probs.shape != predicted_log_probs.shape:
        raise ValueError("repaired and predicted log-prob shapes must match")
    if repaired_log_probs.device != predicted_log_probs.device:
        raise ValueError("repaired and predicted log-probs must share a device")
    if repaired_log_probs.ndim < 1:
        raise ValueError("dependency advantage requires at least one token dimension")
    mask = _token_mask(repaired_log_probs, token_mask)
    _require_active_finite(repaired_log_probs, mask, "repaired_log_probs")
    _require_active_finite(predicted_log_probs, mask, "predicted_log_probs")
    repaired = torch.where(
        mask, repaired_log_probs, torch.zeros_like(repaired_log_probs)
    )
    predicted = torch.where(
        mask, predicted_log_probs, torch.zeros_like(predicted_log_probs)
    )
    advantage = torch.where(
        mask, (repaired - predicted).clamp_min(0.0), torch.zeros_like(repaired)
    )
    return advantage.detach() if detach else advantage


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_ids(
    value: Tensor, shape: torch.Size, device: torch.device, label: str
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{label} must have integer dtype")
    if value.shape != shape:
        raise ValueError(
            f"{label} shape {tuple(value.shape)} != token shape {tuple(shape)}"
        )
    if value.device != device:
        raise ValueError(f"{label} must be on the same device as the inputs")


def within_stage_weights(
    advantages: Tensor,
    stage_ids: Tensor,
    token_mask: Tensor | None = None,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Normalize advantages independently in every sequence-stage segment.

    Active weights have mean one within each stage.  An all-zero stage therefore
    receives unit weights instead of being silently removed.
    """

    _require_floating(advantages, "advantages")
    if advantages.ndim < 1:
        raise ValueError("advantages require at least one token dimension")
    _validate_ids(stage_ids, advantages.shape, advantages.device, "stage_ids")
    mask = _token_mask(advantages, token_mask)
    _require_active_finite(advantages, mask, "advantages")
    if bool((advantages[mask] < 0).any().item()):
        raise ValueError("advantages must be non-negative on active tokens")
    eps = _positive_scalar(epsilon, "epsilon")

    safe = torch.where(mask, advantages, torch.zeros_like(advantages))
    flat_adv = safe.reshape(-1, safe.shape[-1])
    flat_stage = stage_ids.reshape(-1, stage_ids.shape[-1])
    flat_mask = mask.reshape(-1, mask.shape[-1])
    rows: list[Tensor] = []
    for row_adv, row_stage, row_mask in zip(flat_adv, flat_stage, flat_mask):
        row_weights = torch.zeros_like(row_adv)
        for stage in torch.unique(row_stage[row_mask]):
            group = row_mask & (row_stage == stage)
            denominator = row_adv[group].mean() + eps
            normalized = (row_adv + eps) / denominator
            row_weights = row_weights + torch.where(
                group, normalized, torch.zeros_like(normalized)
            )
        rows.append(row_weights)
    return torch.stack(rows).reshape_as(advantages)


def advantage_group_ids(
    advantages: Tensor,
    threshold: float,
    token_mask: Tensor | None = None,
) -> Tensor:
    """Assign low/high IDs (0/1); masked tokens receive -1."""

    _require_floating(advantages, "advantages")
    if advantages.ndim < 1:
        raise ValueError("advantages require at least one token dimension")
    mask = _token_mask(advantages, token_mask)
    _require_active_finite(advantages, mask, "advantages")
    if bool((advantages[mask] < 0).any().item()):
        raise ValueError("advantages must be non-negative on active tokens")
    cutoff = _positive_scalar(threshold, "threshold", allow_zero=True)
    safe = torch.where(mask, advantages, torch.zeros_like(advantages))
    groups = (safe >= cutoff).to(torch.long)
    return torch.where(mask, groups, torch.full_like(groups, -1))


def _prepare_logits(
    student_logits: Tensor,
    teacher_logits: Tensor,
    token_mask: Tensor | None,
    *,
    detach_teacher: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    _require_floating(student_logits, "student_logits")
    _require_floating(teacher_logits, "teacher_logits")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logit shapes must match")
    if student_logits.device != teacher_logits.device:
        raise ValueError("student and teacher logits must share a device")
    if student_logits.ndim < 2 or student_logits.shape[-1] < 2:
        raise ValueError("logits must have shape [..., tokens, vocabulary>=2]")
    token_reference = student_logits[..., 0]
    mask = _token_mask(token_reference, token_mask)
    _require_active_finite(student_logits, mask, "student_logits")
    _require_active_finite(teacher_logits, mask, "teacher_logits")
    expanded = mask.unsqueeze(-1)
    student = torch.where(expanded, student_logits, torch.zeros_like(student_logits))
    teacher_source = teacher_logits.detach() if detach_teacher else teacher_logits
    teacher = torch.where(expanded, teacher_source, torch.zeros_like(teacher_source))
    return student, teacher, mask


def token_reverse_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    token_mask: Tensor | None = None,
    *,
    temperature: float = 1.0,
    detach_teacher: bool = True,
) -> Tensor:
    """Compute tokenwise reverse KL, ``KL(student || teacher)``."""

    student, teacher, mask = _prepare_logits(
        student_logits,
        teacher_logits,
        token_mask,
        detach_teacher=detach_teacher,
    )
    scale = _positive_scalar(temperature, "temperature")
    log_student = F.log_softmax(student / scale, dim=-1)
    log_teacher = F.log_softmax(teacher / scale, dim=-1)
    loss = (log_student.exp() * (log_student - log_teacher)).sum(dim=-1).clamp_min(0.0)
    return torch.where(mask, loss, torch.zeros_like(loss))


def token_generalized_jsd(
    student_logits: Tensor,
    teacher_logits: Tensor,
    token_mask: Tensor | None = None,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
    detach_teacher: bool = True,
) -> Tensor:
    """Compute generalized Jensen-Shannon divergence per token.

    This follows the repository's ms-swift/GKD convention: ``beta`` is the
    teacher weight.  The endpoints are forward KL at zero and reverse KL at
    one; interior values use ``beta * teacher + (1-beta) * student``.
    """

    student, teacher, mask = _prepare_logits(
        student_logits,
        teacher_logits,
        token_mask,
        detach_teacher=detach_teacher,
    )
    mixing = _positive_scalar(beta, "beta", allow_zero=True)
    if mixing > 1.0:
        raise ValueError("beta must be between zero and one")
    scale = _positive_scalar(temperature, "temperature")
    log_student = F.log_softmax(student / scale, dim=-1)
    log_teacher = F.log_softmax(teacher / scale, dim=-1)
    if mixing == 0.0:
        loss = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=-1)
    elif mixing == 1.0:
        loss = (log_student.exp() * (log_student - log_teacher)).sum(dim=-1)
    else:
        log_mixture = torch.logaddexp(
            log_teacher + math.log(mixing),
            log_student + math.log1p(-mixing),
        )
        teacher_kl = (log_teacher.exp() * (log_teacher - log_mixture)).sum(dim=-1)
        student_kl = (log_student.exp() * (log_student - log_mixture)).sum(dim=-1)
        loss = mixing * teacher_kl + (1.0 - mixing) * student_kl
    loss = loss.clamp_min(0.0)
    return torch.where(mask, loss, torch.zeros_like(loss))


def grouped_token_mean(
    token_losses: Tensor,
    group_ids: Tensor,
    token_mask: Tensor | None = None,
    *,
    stage_ids: Tensor | None = None,
    token_weights: Tensor | None = None,
    detach_weights: bool = True,
) -> Tensor:
    """Average per-sample weighted token sums for the FirstFault objective.

    ``group_ids`` and ``stage_ids`` remain explicit audit metadata, but they do not
    alter the reduction. Each flattened sample contributes exactly once, matching
    the historical FirstFault token-sum experiment contract. The current Anvil
    stage-normalized objective uses the separately named helpers below.
    """

    _require_floating(token_losses, "token_losses")
    if token_losses.ndim < 1:
        raise ValueError("token_losses require at least one token dimension")
    _validate_ids(group_ids, token_losses.shape, token_losses.device, "group_ids")
    if stage_ids is not None:
        _validate_ids(stage_ids, token_losses.shape, token_losses.device, "stage_ids")
    mask = _token_mask(token_losses, token_mask)
    _require_active_finite(token_losses, mask, "token_losses")
    if bool((group_ids[mask] < 0).any().item()):
        raise ValueError("group_ids must be non-negative on active tokens")
    if stage_ids is not None and bool((stage_ids[mask] < 0).any().item()):
        raise ValueError("stage_ids must be non-negative on active tokens")

    if token_weights is None:
        weights = torch.ones_like(token_losses)
    else:
        _require_floating(token_weights, "token_weights")
        if token_weights.shape != token_losses.shape:
            raise ValueError("token_weights shape must match token_losses")
        if token_weights.device != token_losses.device:
            raise ValueError("token_weights must share the token-loss device")
        _require_active_finite(token_weights, mask, "token_weights")
        if bool((token_weights[mask] < 0).any().item()):
            raise ValueError("token_weights must be non-negative on active tokens")
        weights = token_weights.detach() if detach_weights else token_weights

    safe_losses = torch.where(mask, token_losses, torch.zeros_like(token_losses))
    safe_weights = torch.where(mask, weights, torch.zeros_like(weights))
    flat_losses = safe_losses.reshape(-1, safe_losses.shape[-1])
    flat_mask = mask.reshape(-1, mask.shape[-1])
    flat_weights = safe_weights.reshape(-1, safe_weights.shape[-1])
    if not bool(flat_mask.any(dim=-1).all().item()):
        raise ValueError("every sample must select at least one token")
    if not bool((flat_weights.sum(dim=-1) > 0).all().item()):
        raise ValueError("every sample must have positive total token weight")
    per_sample = (flat_losses * flat_weights).sum(dim=-1)
    return per_sample.mean()


def grouped_reverse_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    group_ids: Tensor,
    token_mask: Tensor | None = None,
    *,
    stage_ids: Tensor | None = None,
    token_weights: Tensor | None = None,
    temperature: float = 1.0,
    detach_teacher: bool = True,
) -> Tensor:
    """Grouped reverse-KL loss for a student rollout."""

    token_losses = token_reverse_kl(
        student_logits,
        teacher_logits,
        token_mask,
        temperature=temperature,
        detach_teacher=detach_teacher,
    )
    return grouped_token_mean(
        token_losses,
        group_ids,
        token_mask,
        stage_ids=stage_ids,
        token_weights=token_weights,
    )


def grouped_generalized_jsd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    group_ids: Tensor,
    token_mask: Tensor | None = None,
    *,
    stage_ids: Tensor | None = None,
    token_weights: Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    detach_teacher: bool = True,
) -> Tensor:
    """Grouped generalized-JSD loss for a student rollout."""

    token_losses = token_generalized_jsd(
        student_logits,
        teacher_logits,
        token_mask,
        beta=beta,
        temperature=temperature,
        detach_teacher=detach_teacher,
    )
    return grouped_token_mean(
        token_losses,
        group_ids,
        token_mask,
        stage_ids=stage_ids,
        token_weights=token_weights,
    )


def _anvil_layout(logits: Tensor, ids: Tensor, stage_ids: Tensor) -> None:
    """Validate already next-token-aligned tensors; -1 marks non-stage tokens."""
    _require_floating(logits, "logits")
    if logits.ndim != 3 or logits.shape[-1] < 2 or logits.shape[0] == 0:
        raise ValueError("Anvil logits require [nonempty batch, tokens, vocabulary>=2]")
    for name, value in (("token_ids", ids), ("stage_ids", stage_ids)):
        _validate_ids(value, logits.shape[:2], logits.device, name)
    if bool(((stage_ids < -1) | (stage_ids > 2)).any()):
        raise ValueError("stage_ids must use -1=excluded, 0=G, 1=S, 2=A")
    active = stage_ids >= 0
    if bool(((ids[active] < 0) | (ids[active] >= logits.shape[-1])).any()):
        raise ValueError("active token IDs are outside the model vocabulary")


def anvil_stage_sft_loss(
    logits: Tensor,
    target_ids: Tensor,
    stage_ids: Tensor,
    *,
    stage_weights: Tensor | None = None,
) -> Tensor:
    """Reference-stage means, then the declared stage mixture, then sample mean.

    Inputs are next-token aligned, not raw causal-model positions. Stage weights
    have shape [batch, 3], sum to one per sample and are zero for absent stages.
    The default weights represented stages equally. This is distinct from the
    historical FirstFault token-sum objective, which remains unchanged above.
    """
    _anvil_layout(logits, target_ids, stage_ids)
    active = stage_ids >= 0
    present = torch.stack([(stage_ids == stage).any(-1) for stage in range(3)], -1)
    if not bool(present.any(-1).all()):
        raise ValueError("every SFT sample needs a nonempty reference stage")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    if stage_weights is None:
        weights = present.to(dtype) / present.sum(-1, keepdim=True)
    else:
        _require_floating(stage_weights, "stage_weights")
        if stage_weights.shape != present.shape or stage_weights.device != logits.device:
            raise ValueError("stage_weights require [batch, 3] on the logits device")
        if (not bool(torch.isfinite(stage_weights).all())
                or bool((stage_weights < 0).any())
                or bool((stage_weights[~present] != 0).any())
                or not torch.allclose(stage_weights.sum(-1), torch.ones_like(stage_weights[:, 0]))):
            raise ValueError("stage_weights must be nonnegative, sum to one and exclude absent stages")
        weights = stage_weights.detach().to(dtype)
    _require_active_finite(logits, active, "SFT logits")
    token_nll = F.cross_entropy(logits[active].to(dtype), target_ids[active].long(), reduction="none")
    row_ids = torch.arange(logits.shape[0], device=logits.device)[:, None].expand_as(stage_ids)
    result = logits[..., :0].sum().to(dtype)
    for row in range(logits.shape[0]):
        for stage in range(3):
            if present[row, stage]:
                positions = (row_ids[active] == row) & (stage_ids[active] == stage)
                result = result + weights[row, stage] * token_nll[positions].mean()
    return result / logits.shape[0]


def _anvil_compressed_log_probs(
    student: Tensor, teacher: Tensor, structural_ids: Tensor, topk: int,
) -> tuple[Tensor, Tensor]:
    """Exact shared-support probabilities plus the complement's probability mass."""
    if topk >= student.shape[-1]:
        return F.log_softmax(student, -1), F.log_softmax(teacher, -1)
    top_ids = teacher.topk(topk, dim=-1).indices
    support = torch.cat((top_ids, structural_ids.expand(student.shape[0], -1)), -1)
    support = support.sort(-1).values
    unique = torch.ones_like(support, dtype=torch.bool)
    unique[:, 1:] = support[:, 1:] != support[:, :-1]
    remaining = torch.ones_like(student, dtype=torch.bool)
    remaining.scatter_(1, support, False)
    has_remaining = remaining.any(-1)

    def compress(values: Tensor) -> Tensor:
        normalizer = torch.logsumexp(values, -1, keepdim=True)
        kept = (values.gather(-1, support) - normalizer).masked_fill(~unique, -torch.inf)
        complement = values.masked_fill(~remaining, -torch.inf)
        # logsumexp of an all -inf row has undefined backward. Use a dummy row
        # for fully covered vocabularies, then explicitly assign zero tail mass.
        complement = torch.where(has_remaining[:, None], complement, torch.zeros_like(complement))
        tail = torch.logsumexp(complement, -1, keepdim=True) - normalizer
        tail = tail.masked_fill(~has_remaining[:, None], -torch.inf)
        return torch.cat((kept, tail), -1)

    return compress(student), compress(teacher)


def anvil_selected_span_mtd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    sampled_token_ids: Tensor,
    stage_ids: Tensor,
    selected_stages: Tensor,
    structural_mask: Tensor,
    *,
    topk: int = 128,
    beta: float = 0.5,
    structural_alpha: float = 2.0,
    temperature: float = 0.9,
    chunk_size: int = 32,
    distributed_mean: bool = False,
) -> Tensor:
    """Anvil's selected-span, residual-preserving JSD objective.

    Stage IDs are 0=G, 1=S, 2=A; -1 excludes a token or abstains for a sample.
    ``selected_stages`` must come from qualified, freshly evaluated student
    rollouts. This function implements the loss, not route inference or teacher
    qualification. The teacher must score those same student prefixes. Structural
    positions must come from decoded field/coordinate/delimiter spans, not an
    arbitrary vocabulary-wide list. All tensors are next-token aligned.

    Each nonempty selected span contributes one normalized mean. Abstaining
    examples do not dilute the conditional MTD mean; an empty batch contributes
    differentiable zero. With ``distributed_mean=True``, all ranks participate
    in an eligible-count reduction, including ranks with zero eligible samples.
    The local gradient is scaled to offset DDP/ZeRO's subsequent rank average.
    """
    _anvil_layout(student_logits, sampled_token_ids, stage_ids)
    _require_floating(teacher_logits, "teacher_logits")
    if teacher_logits.shape != student_logits.shape or teacher_logits.device != student_logits.device:
        raise ValueError("teacher/student logits must have identical shapes and devices")
    _validate_ids(selected_stages, student_logits.shape[:1], student_logits.device, "selected_stages")
    if bool(((selected_stages < -1) | (selected_stages > 2)).any()):
        raise ValueError("selected_stages must use -1, 0=G, 1=S, 2=A")
    if (not isinstance(structural_mask, Tensor) or structural_mask.dtype != torch.bool
            or structural_mask.shape != stage_ids.shape or structural_mask.device != stage_ids.device):
        raise ValueError("structural_mask requires boolean [batch, tokens] on the logits device")
    for name, value in (("topk", topk), ("chunk_size", chunk_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(distributed_mean) is not bool:
        raise ValueError("distributed_mean must be a boolean")
    mixing = _positive_scalar(beta, "beta")
    if mixing >= 1:
        raise ValueError("Anvil JSD requires 0 < beta < 1")
    alpha = _positive_scalar(structural_alpha, "structural_alpha", allow_zero=True)
    scale = _positive_scalar(temperature, "temperature")
    selected = (stage_ids == selected_stages[:, None]) & (selected_stages[:, None] >= 0)
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    zero = student_logits[..., :0].sum().to(dtype)
    per_sample = []
    for row in range(student_logits.shape[0]):
        positions = selected[row].nonzero().flatten()
        if positions.numel() == 0:
            continue
        structural_ids = sampled_token_ids[row, selected[row] & structural_mask[row]].unique().long()
        weights = 1 + alpha * structural_mask[row, positions].to(dtype)
        weights = weights / weights.mean()
        total = zero
        for begin in range(0, positions.numel(), chunk_size):
            indices = positions[begin:begin + chunk_size]
            student = student_logits[row, indices].to(dtype) / scale
            teacher = teacher_logits[row, indices].detach().to(dtype) / scale
            if not bool(torch.isfinite(student).all() & torch.isfinite(teacher).all()):
                raise ValueError("selected logits must be finite")
            log_p, log_q = _anvil_compressed_log_probs(student, teacher, structural_ids, topk)
            occupied = torch.isfinite(log_p) | torch.isfinite(log_q)
            # A duplicate support slot has zero mass in both distributions.
            # Remove its -inf values before differentiable mixture arithmetic.
            safe_p = log_p.masked_fill(~occupied, 0)
            safe_q = log_q.masked_fill(~occupied, 0)
            mixture = torch.logaddexp(safe_q + math.log(mixing), safe_p + math.log1p(-mixing))
            divergence = (mixing * log_q.exp() * (safe_q - mixture)
                          + (1 - mixing) * log_p.exp() * (safe_p - mixture)).sum(-1)
            total = total + (divergence * weights[begin:begin + chunk_size]).sum()
        per_sample.append(total / positions.numel())
    total = torch.stack(per_sample).sum() if per_sample else zero
    if distributed_mean:
        import torch.distributed as dist
        if not dist.is_initialized():
            raise ValueError("distributed_mean requires an initialized process group")
        count = torch.tensor(len(per_sample), dtype=torch.long, device=student_logits.device)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return total * (dist.get_world_size() / count.clamp_min(1))
    return total / len(per_sample) if per_sample else zero


__all__ = [
    "FirstFaultRoute",
    "RouteKind",
    "advantage_group_ids",
    "anvil_selected_span_mtd_loss",
    "anvil_stage_sft_loss",
    "confirm_state_use",
    "dependency_advantage",
    "grouped_generalized_jsd_loss",
    "grouped_reverse_kl_loss",
    "grouped_token_mean",
    "route_first_fault",
    "response_scores_consistent",
    "token_generalized_jsd",
    "token_reverse_kl",
    "within_stage_weights",
]
