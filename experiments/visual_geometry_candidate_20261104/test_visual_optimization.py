import copy
import unittest

import torch

from visual_optimization import (gradient_norms, parameter_groups,
                                 set_learning_rates, visual_state,
                                 visual_update_receipt)


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Linear(3, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 3, bias=False)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = Adapter()
        self.model.visual = Adapter()

    def forward(self, x):
        return self.model.language_model(x) + self.model.visual(x)


EXPECTED = dict(language_parameters=12, visual_parameters=12)
CONFIG = dict(lr=.001, targeted_geometry_lr=.002, visual_lr=.01)


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = Model()
        groups, self.named = parameter_groups(self.model, EXPECTED)
        self.optimizer = torch.optim.AdamW(groups, lr=.001)
        self.x = torch.randn(4, 3)

    def step(self, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        (model(self.x) - 1).square().mean().backward()
        optimizer.step()

    def test_partition_and_different_rates(self):
        self.assertEqual([g['family'] for g in self.optimizer.param_groups], ['language', 'visual'])
        set_learning_rates(self.optimizer, CONFIG, .5, True)
        self.assertEqual([g['lr'] for g in self.optimizer.param_groups], [.001, .005])
        set_learning_rates(self.optimizer, CONFIG, .5, False)
        self.assertEqual([g['lr'] for g in self.optimizer.param_groups], [.0005, .005])

    def test_base_parameter_and_count_fail_closed(self):
        self.model.extra = torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            parameter_groups(self.model, EXPECTED)
        del self.model.extra
        with self.assertRaises(ValueError):
            parameter_groups(self.model, dict(language_parameters=12, visual_parameters=13))

    def test_visual_branch_must_receive_gradient(self):
        self.model.model.language_model(self.x).sum().backward()
        with self.assertRaisesRegex(ValueError, 'Missing adapter gradient'):
            gradient_norms(self.named)

    def test_nonfinite_gradient_fails(self):
        self.model(self.x).sum().backward()
        self.model.model.visual.lora_B.weight.grad[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            gradient_norms(self.named)

    def test_real_update_required_for_checkpoint(self):
        before = visual_state(self.named)
        with self.assertRaisesRegex(ValueError, 'No visual B'):
            visual_update_receipt(self.named, before, 0, 1)
        (self.model(self.x) - 1).square().mean().backward()
        norms = gradient_norms(self.named)
        self.assertGreater(norms['visual'], 0)
        self.optimizer.step()
        receipt = visual_update_receipt(self.named, before, 0, 1)
        self.assertEqual(receipt['changed_b_tensors'], 1)
        self.assertTrue(receipt['real_training_update_verified'])

    def test_optimizer_resume_preserves_groups_and_exact_next_update(self):
        set_learning_rates(self.optimizer, CONFIG, .8, True)
        self.step(self.model, self.optimizer)
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        resumed = Model()
        resumed.load_state_dict(model_state)
        groups, _ = parameter_groups(resumed, EXPECTED)
        restored = torch.optim.AdamW(groups, lr=99.)
        restored.load_state_dict(optimizer_state)
        for optimizer in [self.optimizer, restored]:
            set_learning_rates(optimizer, CONFIG, .7, False)
        self.step(self.model, self.optimizer)
        self.step(resumed, restored)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
        self.assertEqual({int(s['step']) for s in restored.state.values()}, {2})

    def test_invalid_rates_or_group_order_rejected(self):
        for config, ratio in [(dict(CONFIG, visual_lr=0), 1),
                              (dict(CONFIG, visual_lr=float('nan')), 1), (CONFIG, 1.1)]:
            with self.assertRaises(ValueError):
                set_learning_rates(self.optimizer, config, ratio, False)
        self.optimizer.param_groups.reverse()
        with self.assertRaises(ValueError):
            set_learning_rates(self.optimizer, CONFIG, 1, False)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
