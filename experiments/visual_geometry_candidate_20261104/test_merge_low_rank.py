import unittest

import torch

from merge_low_rank import collect_pairs, merge_tensor


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.modules = ['model.language_model.layer.q_proj', 'model.visual.merger.linear_fc2']
        self.weights = {}
        for name, rank in zip(self.modules, [2, 1]):
            self.weights['base_model.model.' + name + '.lora_A.weight'] = torch.ones(rank, 3)
            self.weights['base_model.model.' + name + '.lora_B.weight'] = torch.ones(4, rank)
        self.index = {name + '.weight': 'shard' for name in self.modules}
        self.config = dict(r=2, lora_alpha=4, rank_pattern={self.modules[1]: 1},
                           alpha_pattern={self.modules[1]: 3})

    def test_each_projection_uses_its_own_rank_and_alpha(self):
        pairs = collect_pairs(self.weights, self.config, self.index)
        base = torch.full((4, 3), 2., dtype=torch.bfloat16)
        language = merge_tensor(base, pairs[self.modules[0] + '.weight'])
        visual = merge_tensor(base, pairs[self.modules[1] + '.weight'])
        self.assertTrue(torch.equal(language, torch.full_like(base, 6.)))
        self.assertTrue(torch.equal(visual, torch.full_like(base, 5.)))
        self.assertTrue(torch.equal(base, torch.full_like(base, 2.)))

    def test_zero_delta_retains_base_exactly(self):
        for key, value in self.weights.items():
            if '.lora_B.' in key:
                value.zero_()
        base = torch.randn(4, 3).bfloat16()
        for pair in collect_pairs(self.weights, self.config, self.index).values():
            self.assertTrue(torch.equal(merge_tensor(base, pair), base))

    def test_rank_pattern_must_resolve_and_match_payload(self):
        config = dict(self.config, rank_pattern={self.modules[1]: 2})
        with self.assertRaises(ValueError):
            collect_pairs(self.weights, config, self.index)
        config = dict(self.config, alpha_pattern={'model.visual.*': 3})
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            collect_pairs(self.weights, config, self.index)

    def test_missing_extra_nonfinite_and_wrong_shape_fail(self):
        weights = dict(self.weights)
        weights.pop(next(key for key in weights if '.lora_B.' in key))
        with self.assertRaises(ValueError):
            collect_pairs(weights, self.config, self.index)
        weights = dict(self.weights, unexpected=torch.ones(1))
        with self.assertRaises(ValueError):
            collect_pairs(weights, self.config, self.index)
        pairs = collect_pairs(self.weights, self.config, self.index)
        with self.assertRaises(ValueError):
            merge_tensor(torch.ones(5, 3), next(iter(pairs.values())))
        next(iter(self.weights.values()))[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            collect_pairs(self.weights, self.config, self.index)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
