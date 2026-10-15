from __future__ import annotations

import hashlib
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from trains.mopd.firstfault_core import (
    RouteKind,
    advantage_group_ids,
    anvil_selected_span_mtd_loss,
    anvil_stage_sft_loss,
    confirm_state_use,
    dependency_advantage,
    grouped_generalized_jsd_loss,
    grouped_reverse_kl_loss,
    grouped_token_mean,
    route_first_fault,
    token_generalized_jsd,
    token_reverse_kl,
    within_stage_weights,
)


def _anvil_loss(student, teacher, ids, stages, selected, structural=None, **kwargs):
    kwargs.setdefault("temperature", 1.0)
    return anvil_selected_span_mtd_loss(
        student, teacher, ids, stages, selected,
        torch.zeros_like(stages, dtype=torch.bool) if structural is None else structural,
        **kwargs,
    )


def _probability_jsd(p, q, beta=0.5):
    mixture = beta * q + (1 - beta) * p
    return beta * (q * (q / mixture).log()).sum() + (1 - beta) * (p * (p / mixture).log()).sum()


def test_anvil_topk_retains_tail_mass_that_renormalized_topk_would_erase():
    p = torch.tensor([0.8, 0.2], dtype=torch.float64)
    q = torch.tensor([0.9, 0.1], dtype=torch.float64)
    loss = _anvil_loss(p.log()[None, None], q.log()[None, None],
                       torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([0]), topk=1)
    torch.testing.assert_close(loss, _probability_jsd(p, q))
    assert loss > 0.009


@pytest.mark.parametrize("topk", [1, 2, 4, 128])
def test_anvil_structural_support_matches_explicit_probability_partition(topk):
    p = torch.tensor([0.15, 0.25, 0.35, 0.25], dtype=torch.float64)
    q = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float64)
    # Token 3 is structural but outside the teacher's top 1/2. Its exact mass
    # must stay separate from the residual rather than being dropped or doubled.
    support = sorted(set(range(min(topk, 4))) | {3})
    other = [index for index in range(4) if index not in support]
    pooled_p, pooled_q = p[support], q[support]
    if other:
        pooled_p = torch.cat((pooled_p, p[other].sum().reshape(1)))
        pooled_q = torch.cat((pooled_q, q[other].sum().reshape(1)))
    loss = _anvil_loss(p.log()[None, None], q.log()[None, None],
                       torch.tensor([[3]]), torch.tensor([[1]]), torch.tensor([1]),
                       torch.tensor([[True]]), topk=topk, beta=0.3)
    torch.testing.assert_close(loss, _probability_jsd(pooled_p, pooled_q, 0.3))


def test_anvil_span_normalization_excludes_abstentions_and_unselected_gradients():
    torch.manual_seed(91)
    student = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    stages = torch.tensor([[0, 1, 1, 2], [0, 0, 0, 0], [0, 1, 1, 2]])
    ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    selected = torch.tensor([1, 0, -1])
    structural = torch.tensor([[False, True, False, False], [False] * 4, [True] * 4])
    loss = _anvil_loss(student, teacher, ids, stages, selected, structural,
                       topk=5, structural_alpha=2, chunk_size=1)
    d0 = torch.stack([_probability_jsd(student[0, j].softmax(-1), teacher[0, j].detach().softmax(-1))
                      for j in (1, 2)])
    d1 = torch.stack([_probability_jsd(student[1, j].softmax(-1), teacher[1, j].detach().softmax(-1))
                      for j in range(4)])
    torch.testing.assert_close(loss, ((d0 * torch.tensor([1.5, 0.5])).mean() + d1.mean()) / 2)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[0, [0, 3]].count_nonzero() == 0
    assert student.grad[2].count_nonzero() == 0
    assert student.grad[0, 1:3].abs().sum() > 0
    # Repeating a selected sequence must not increase its weight.
    repeated = _anvil_loss(student.detach().repeat_interleave(2, 1), teacher.detach().repeat_interleave(2, 1),
                           ids.repeat_interleave(2, 1), stages.repeat_interleave(2, 1), selected,
                           structural.repeat_interleave(2, 1), topk=5, structural_alpha=2)
    torch.testing.assert_close(loss, repeated)


def test_anvil_empty_mtd_is_differentiable_zero_even_with_inactive_nan_logits():
    student = torch.full((2, 3, 4), float("nan"), requires_grad=True)
    loss = _anvil_loss(student, student.detach(), torch.zeros(2, 3, dtype=torch.long),
                       torch.zeros(2, 3, dtype=torch.long), torch.tensor([-1, -1]))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(student.grad).all() and student.grad.count_nonzero() == 0


def test_anvil_compressed_loss_gradcheck_including_fully_covered_union():
    teacher = torch.tensor([[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]], dtype=torch.float64,
                           requires_grad=True)
    student = torch.tensor([[[0.3, 0.2, -0.2], [0.7, 0.1, -0.4]]], dtype=torch.float64,
                           requires_grad=True)
    ids, stages = torch.tensor([[1, 2]]), torch.tensor([[0, 0]])
    for structural in (torch.tensor([[False, False]]), torch.tensor([[True, True]])):
        fn = lambda values: _anvil_loss(values, teacher, ids, stages, torch.tensor([0]), structural, topk=1)
        assert torch.autograd.gradcheck(fn, (student,), eps=1e-6, atol=1e-5)


def test_anvil_sft_weights_stages_and_examples_instead_of_serialized_length():
    torch.manual_seed(72)
    logits = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    stages = torch.tensor([[0, 1, 1, 1, 2], [-1, 0, 0, 0, 0]])
    weights = torch.tensor([[0.2, 0.3, 0.5], [1., 0., 0.]], dtype=torch.float64, requires_grad=True)
    loss = anvil_stage_sft_loss(logits, ids, stages, stage_weights=weights)
    nll = -logits.log_softmax(-1).gather(-1, ids[..., None]).squeeze(-1)
    expected = (0.2 * nll[0, 0] + 0.3 * nll[0, 1:4].mean() + 0.5 * nll[0, 4] + nll[1, 1:].mean()) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert weights.grad is None and logits.grad[1, 0].count_nonzero() == 0
    with pytest.raises(ValueError, match="exclude absent"):
        anvil_stage_sft_loss(logits, ids, stages, stage_weights=torch.full((2, 3), 1 / 3))


@pytest.mark.parametrize("value", [0., 1., float("nan")])
def test_anvil_jsd_does_not_silently_switch_to_kl_at_beta_endpoints(value):
    with pytest.raises(ValueError, match="beta"):
        _anvil_loss(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), torch.tensor([[0]]),
                    torch.tensor([[0]]), torch.tensor([0]), beta=value)


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_route_attributes_the_single_monotonic_upstream_fault() -> None:
    route = route_first_fault(
        stage_order=("G", "S", "A"),
        local_correct={"G": True, "S": False, "A": True},
        predicted_target_score=0.0,
        repair_target_scores={"G": 0.0, "S": 1.0},
        predicted_target_response_identity=_identity("predicted"),
        repair_target_response_identities={
            "G": _identity("repair-g"),
            "S": _identity("repair-s"),
        },
        state_use_confirmed={"G": True, "S": True},
        probes_valid=True,
        hit_token_cap=False,
        correct_threshold=1.0,
        monotonic_tolerance=0.0,
    )
    assert route.kind is RouteKind.ATTRIBUTED
    assert route.stage == "S"
    assert route.eligible is True


def test_route_rejects_a_string_stage_order_instead_of_guessing() -> None:
    route = route_first_fault(
        stage_order="GS",
        local_correct={"G": False, "S": True},
        predicted_target_score=0.0,
        repair_target_scores={"G": 1.0},
        predicted_target_response_identity=_identity("predicted"),
        repair_target_response_identities={"G": _identity("repair-g")},
        state_use_confirmed={"G": True},
        probes_valid=True,
        hit_token_cap=False,
        correct_threshold=1.0,
        monotonic_tolerance=0.0,
    )
    assert route.kind is RouteKind.INCONCLUSIVE
    assert route.reason == "unsupported_or_invalid_stage_order"


@pytest.mark.parametrize(
    ("stages", "local", "repairs", "expected"),
    (
        (("G", "S"), {"G": True, "S": False}, {"G": 0.0}, "S"),
        (
            ("G", "S", "A"),
            {"G": True, "S": True, "A": False},
            {"G": 0.0, "S": 0.0},
            "A",
        ),
    ),
)
def test_route_assigns_local_only_with_correct_oracle_parents(
    stages: tuple[str, ...],
    local: dict[str, bool],
    repairs: dict[str, float],
    expected: str,
) -> None:
    route = route_first_fault(
        stage_order=stages,
        local_correct=local,
        predicted_target_score=0.0,
        repair_target_scores=repairs,
        predicted_target_response_identity=_identity("predicted"),
        repair_target_response_identities={
            stage: _identity(f"repair-{stage}") for stage in stages[:-1]
        },
        state_use_confirmed={stage: True for stage in stages[:-1]},
        probes_valid=True,
        hit_token_cap=False,
        correct_threshold=1.0,
        monotonic_tolerance=0.0,
    )
    assert route.kind is RouteKind.LOCAL
    assert route.stage == expected


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"probes_valid": False}, "invalid_or_incomplete_probes"),
        ({"hit_token_cap": True}, "capped_probe"),
        (
            {"local_correct": {"G": False, "S": False, "A": True}},
            "multiple_unresolved_faults",
        ),
        (
            {"state_use_confirmed": {"G": False, "S": True}},
            "context_insensitive_repair",
        ),
        (
            {"repair_target_scores": {"G": 1.0, "S": 0.0}},
            "oracle_degradation",
        ),
        (
            {
                "predicted_target_score": 1.0,
                "repair_target_scores": {"G": 1.0, "S": 1.0},
            },
            "predicted_target_correct",
        ),
    ),
)
def test_route_fails_closed_for_ambiguous_or_unhealthy_evidence(
    overrides: dict[str, object], reason: str
) -> None:
    evidence: dict[str, object] = {
        "stage_order": ("G", "S", "A"),
        "local_correct": {"G": False, "S": True, "A": True},
        "predicted_target_score": 0.0,
        "repair_target_scores": {"G": 1.0, "S": 1.0},
        "predicted_target_response_identity": _identity("predicted"),
        "repair_target_response_identities": {
            "G": _identity("repair-g"),
            "S": _identity("repair-s"),
        },
        "state_use_confirmed": {"G": True, "S": True},
        "probes_valid": True,
        "hit_token_cap": False,
        "correct_threshold": 1.0,
        "monotonic_tolerance": 0.0,
    }
    evidence.update(overrides)
    route = route_first_fault(**evidence)  # type: ignore[arg-type]
    assert route.kind is RouteKind.INCONCLUSIVE
    assert route.stage is None
    assert route.reason == reason
    assert route.eligible is False


def test_dependency_advantage_and_stage_normalization_match_equations() -> None:
    repaired = torch.tensor([[-0.1, -2.0, -0.2, -0.2]], dtype=torch.float64)
    predicted = torch.tensor([[-1.0, -1.0, -0.2, -0.2]], dtype=torch.float64)
    advantage = dependency_advantage(repaired, predicted)
    torch.testing.assert_close(
        advantage, torch.tensor([[0.9, 0.0, 0.0, 0.0]], dtype=torch.float64)
    )
    stages = torch.tensor([[0, 0, 1, 1]])
    weights = within_stage_weights(advantage, stages, epsilon=1e-6)
    torch.testing.assert_close(
        weights[0, :2].mean(), torch.tensor(1.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        weights[0, 2:].mean(), torch.tensor(1.0, dtype=torch.float64)
    )
    torch.testing.assert_close(weights[0, 2:], torch.ones(2, dtype=torch.float64))


def test_advantage_mask_allows_nonfinite_padding_but_not_active_values() -> None:
    repaired = torch.tensor([[-0.1, float("nan")]])
    predicted = torch.tensor([[-1.1, float("inf")]])
    mask = torch.tensor([[True, False]])
    result = dependency_advantage(repaired, predicted, mask)
    torch.testing.assert_close(result, torch.tensor([[1.0, 0.0]]))
    groups = advantage_group_ids(result, threshold=0.5, token_mask=mask)
    assert groups.tolist() == [[1, -1]]
    with pytest.raises(ValueError, match="non-finite active"):
        dependency_advantage(repaired, predicted, torch.tensor([[True, True]]))


def test_token_divergences_match_direct_probability_math() -> None:
    student_probability = torch.tensor([0.8, 0.2], dtype=torch.float64)
    teacher_probability = torch.tensor([0.3, 0.7], dtype=torch.float64)
    student = student_probability.log().reshape(1, 1, 2)
    teacher = teacher_probability.log().reshape(1, 1, 2)

    reverse = token_reverse_kl(student, teacher)[0, 0]
    expected_reverse = (
        student_probability * (student_probability.log() - teacher_probability.log())
    ).sum()
    torch.testing.assert_close(reverse, expected_reverse)

    beta = 0.25
    mixture = beta * teacher_probability + (1.0 - beta) * student_probability
    expected_jsd = (
        beta * (teacher_probability * (teacher_probability.log() - mixture.log())).sum()
        + (1.0 - beta)
        * (student_probability * (student_probability.log() - mixture.log())).sum()
    )
    jsd = token_generalized_jsd(student, teacher, beta=beta)[0, 0]
    torch.testing.assert_close(jsd, expected_jsd)

    forward = (
        teacher_probability * (teacher_probability.log() - student_probability.log())
    ).sum()
    torch.testing.assert_close(
        token_generalized_jsd(student, teacher, beta=0.0)[0, 0], forward
    )
    torch.testing.assert_close(
        token_generalized_jsd(student, teacher, beta=1.0)[0, 0], reverse
    )


@pytest.mark.parametrize(
    "loss_fn",
    (grouped_reverse_kl_loss, grouped_generalized_jsd_loss),
)
def test_grouped_loss_is_not_diluted_by_formatting_tokens(
    loss_fn: Callable[..., Tensor],
) -> None:
    decision_student = torch.tensor([[4.0, -4.0]])
    decision_teacher = torch.tensor([[-4.0, 4.0]])
    formatting = torch.tensor([[0.0, 0.0]])

    def build(formatting_count: int) -> tuple[Tensor, Tensor, Tensor]:
        student = torch.cat((decision_student, formatting.repeat(formatting_count, 1)))
        teacher = torch.cat((decision_teacher, formatting.repeat(formatting_count, 1)))
        groups = torch.tensor([1] + [0] * formatting_count)
        return student.unsqueeze(0), teacher.unsqueeze(0), groups.unsqueeze(0)

    short = build(1)
    long = build(100)
    short_loss = loss_fn(short[0], short[1], short[2])
    long_loss = loss_fn(long[0], long[1], long[2])
    torch.testing.assert_close(short_loss, long_loss)
    assert short_loss.item() > 0.0


@pytest.mark.parametrize(
    "loss_fn",
    (grouped_reverse_kl_loss, grouped_generalized_jsd_loss),
)
def test_grouped_losses_backpropagate_only_to_student_by_default(
    loss_fn: Callable[..., Tensor],
) -> None:
    student = torch.tensor(
        [[[2.0, -1.0], [0.2, -0.2]]], dtype=torch.float64, requires_grad=True
    )
    teacher = torch.tensor(
        [[[-1.0, 2.0], [-0.3, 0.3]]], dtype=torch.float64, requires_grad=True
    )
    groups = torch.tensor([[1, 0]])
    weights = torch.tensor([[1.5, 0.5]], dtype=torch.float64)
    loss = loss_fn(student, teacher, groups, token_weights=weights)
    loss.backward()
    assert student.grad is not None
    assert bool(torch.isfinite(student.grad).all())
    assert student.grad.abs().sum().item() > 0.0
    assert teacher.grad is None


def test_masks_shapes_and_nonfinite_logits_fail_closed() -> None:
    student = torch.tensor([[[2.0, -1.0], [float("nan"), float("nan")]]])
    teacher = torch.tensor([[[-1.0, 2.0], [float("inf"), 0.0]]])
    groups = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, False]])
    loss = grouped_reverse_kl_loss(student, teacher, groups, mask)
    assert torch.isfinite(loss)

    with pytest.raises(ValueError, match="non-finite active"):
        grouped_reverse_kl_loss(student, teacher, groups, torch.tensor([[True, True]]))
    with pytest.raises(ValueError, match="bool dtype"):
        grouped_reverse_kl_loss(student, teacher, groups, mask.to(torch.int64))
    with pytest.raises(ValueError, match="group_ids shape"):
        grouped_reverse_kl_loss(student, teacher, torch.tensor([1]), mask)
    with pytest.raises(ValueError, match="logit shapes"):
        token_generalized_jsd(student, teacher[..., :1])
    with pytest.raises(ValueError, match="non-negative"):
        grouped_reverse_kl_loss(
            student,
            teacher,
            groups,
            mask,
            token_weights=torch.tensor([[-1.0, 0.0]]),
        )


def test_loss_reduction_is_a_per_sample_weighted_token_sum() -> None:
    losses = torch.tensor([[2.0, 0.0, 8.0, 0.0]])
    groups = torch.tensor([[1, 0, 1, 0]])
    stages = torch.tensor([[0, 0, 1, 1]])
    # Group metadata does not change the publication objective's token sum.
    torch.testing.assert_close(
        grouped_token_mean(losses, groups, stage_ids=stages),
        torch.tensor(10.0),
    )


def test_loss_reduction_weights_samples_equally_not_nonempty_groups() -> None:
    losses = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
    groups = torch.tensor([[0, 0], [0, 1]])
    torch.testing.assert_close(grouped_token_mean(losses, groups), torch.tensor(3.0))


def test_loss_reduction_rejects_an_empty_sample() -> None:
    losses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    groups = torch.zeros_like(losses, dtype=torch.long)
    mask = torch.tensor([[True, True], [False, False]])
    with pytest.raises(ValueError, match="every sample"):
        grouped_token_mean(losses, groups, mask)


def test_route_uses_nonunit_bound_threshold() -> None:
    route = route_first_fault(
        stage_order=("G", "S"),
        local_correct={"G": False, "S": True},
        predicted_target_score=0.4,
        repair_target_scores={"G": 0.6},
        predicted_target_response_identity=_identity("predicted-nonunit"),
        repair_target_response_identities={"G": _identity("repair-nonunit")},
        state_use_confirmed={"G": True},
        probes_valid=True,
        hit_token_cap=False,
        correct_threshold=0.5,
        monotonic_tolerance=0.0,
    )
    assert route.kind is RouteKind.ATTRIBUTED
    assert route.stage == "G"


def test_route_rejects_same_response_with_different_scores() -> None:
    identity = _identity("same-response")
    route = route_first_fault(
        stage_order=("G", "S"),
        local_correct={"G": False, "S": True},
        predicted_target_score=0.0,
        repair_target_scores={"G": 1.0},
        predicted_target_response_identity=identity,
        repair_target_response_identities={"G": identity},
        state_use_confirmed={"G": True},
        probes_valid=True,
        hit_token_cap=False,
        correct_threshold=1.0,
        monotonic_tolerance=0.0,
    )
    assert route.kind is RouteKind.INCONCLUSIVE
    assert route.reason == "response_score_inconsistency"


def test_state_use_requires_a_changed_repair_response() -> None:
    identity = _identity("unchanged-repair")
    assert not confirm_state_use(
        repaired_target_score=1.0,
        absent_target_score=0.0,
        corrupted_target_score=0.0,
        minimum_effect=0.0,
        previous_response_identity=identity,
        repaired_response_identity=identity,
    )
    assert confirm_state_use(
        repaired_target_score=1.0,
        absent_target_score=0.0,
        corrupted_target_score=0.0,
        minimum_effect=0.0,
        previous_response_identity=identity,
        repaired_response_identity=_identity("changed-repair"),
    )


def test_monotonic_tolerance_is_applied_to_continuous_scores() -> None:
    evidence = {
        "stage_order": ("G", "S", "A"),
        "local_correct": {"G": True, "S": False, "A": True},
        "predicted_target_score": 0.20,
        "repair_target_scores": {"G": 0.19, "S": 0.60},
        "predicted_target_response_identity": _identity("predicted"),
        "repair_target_response_identities": {
            "G": _identity("repair-g"),
            "S": _identity("repair-s"),
        },
        "state_use_confirmed": {"G": True, "S": True},
        "probes_valid": True,
        "hit_token_cap": False,
        "correct_threshold": 0.50,
    }
    strict = route_first_fault(**evidence, monotonic_tolerance=0.0)
    tolerant = route_first_fault(**evidence, monotonic_tolerance=0.01)
    assert strict.kind is RouteKind.INCONCLUSIVE
    assert strict.reason == "oracle_degradation"
    assert tolerant.kind is RouteKind.ATTRIBUTED
    assert tolerant.stage == "S"
