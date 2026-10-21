import torch
from trains.mopd.chunked_heads import stage_sft_from_hidden, selected_mtd_from_hidden
from trains.mopd.firstfault_core import anvil_stage_sft_loss, anvil_selected_span_mtd_loss


def test_stage_mixture_loss_and_hidden_and_head_gradients_match():
    torch.manual_seed(19)
    hidden = torch.randn(2, 9, 7, dtype=torch.float64, requires_grad=True)
    head = torch.randn(17, 7, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(0, 17, (2, 9))
    stages = torch.tensor([[-1, 0, 0, 1, 1, 1, 2, -1, -1], [0, 0, 0, -1, 2, 2, 2, 2, -1]])
    project = lambda x: x @ head.T
    reference = anvil_stage_sft_loss(project(hidden), ids, stages)
    expected = torch.autograd.grad(reference, (hidden, head))
    actual = stage_sft_from_hidden(hidden, project, ids, stages, chunk_size=2)
    gradients = torch.autograd.grad(actual, (hidden, head))
    torch.testing.assert_close(reference, actual)
    for a, b in zip(expected, gradients):
        torch.testing.assert_close(a, b)


def test_mtd_preserves_support_tail_structural_weights_and_conditional_mean():
    torch.manual_seed(29)
    student = torch.randn(3, 9, 7, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 9, 7, dtype=torch.float64, requires_grad=True)
    s_head = torch.randn(17, 7, dtype=torch.float64, requires_grad=True)
    t_head = torch.randn(17, 7, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(0, 17, (3, 9))
    stages = torch.tensor([[0, 0, 0, 0, 0, 1, 1, -1, -1],
                           [-1, -1, 1, 1, 1, 1, 2, 2, 2],
                           [2, 2, 2, 2, 2, 2, 2, 2, 2]])
    selected = torch.tensor([0, 1, -1])
    structural = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    options = dict(topk=3, beta=.35, structural_alpha=2., temperature=.9, chunk_size=2)
    reference = anvil_selected_span_mtd_loss(
        student @ s_head.T, teacher @ t_head.T, ids, stages, selected, structural, **options)
    expected = torch.autograd.grad(reference, (student, s_head, teacher, t_head), allow_unused=True)
    actual = selected_mtd_from_hidden(student, teacher, lambda x: x @ s_head.T,
                                      lambda x: x @ t_head.T, ids, stages, selected, structural, **options)
    gradients = torch.autograd.grad(actual, (student, s_head, teacher, t_head), allow_unused=True)
    torch.testing.assert_close(reference, actual)
    for a, b in zip(expected[:2], gradients[:2]):
        torch.testing.assert_close(a, b)
    assert expected[2:] == gradients[2:] == (None, None)


def test_abstaining_batch_has_differentiable_zero_without_teacher_work():
    states = torch.randn(2, 4, 3, requires_grad=True)
    def unused_head(_):
        raise AssertionError("No head forward is needed for an abstaining batch")
    loss = selected_mtd_from_hidden(
        states, torch.zeros_like(states), unused_head, unused_head,
        torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long),
        torch.full((2,), -1), torch.zeros(2, 4, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and torch.equal(states.grad, torch.zeros_like(states))
