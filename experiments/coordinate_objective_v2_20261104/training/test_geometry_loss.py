import json
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from geometry_loss import GeometryLoss, coordinate_spans, digit_weights, geometry_eligible

BASE = '/data/models/Qwen/Qwen3.5-9B'
PREFIX = '<think>\n\n</think>\n\n'


class GeometryLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        cls.objective = GeometryLoss(cls.tokenizer, PREFIX)

    def test_supplemental_point_formats_keep_original_ce(self):
        self.assertTrue(geometry_eligible('point','native'))
        for kind in ['grounding','vacancy','fsd','qa']:
            self.assertFalse(geometry_eligible('point',kind))
        for form in ['mask','affordance','bbox','detection','dof']:
            self.assertTrue(geometry_eligible(form,'native'))

    def test_json_semantics_not_numeric_strings(self):
        text = json.dumps({'label': 'cup #4 123.45', 'rotation_bin': 51,
                           'instruction': 'say "bbox_2d": [1,2,3,4]',
                           'region_bbox_2d': [12.345, 56.789, 100, 900],
                           'unrelated': [11, 22], 'position': [-0.12, 0.345]}, ensure_ascii=False)
        values = [float(text[a:b]) for a, b, _ in coordinate_spans(text)]
        self.assertEqual(values, [12.345, 56.789, 100, 900, -0.12, 0.345])

    def test_nested_polygons_and_unicode(self):
        text = '[{"label":"杯子 №2", "polygons":[[[1.234,5.678],[10,20],[30,40]],[[50,60],[70,80],[90,100]]]}]'
        plan = self.objective.plan(text)
        self.assertEqual(plan['scalars'], 12)
        self.assertAlmostEqual(sum(plan['weights']), 12)
        for index in plan['positions']:
            self.assertTrue(any(c.isdigit() or c in '.-' for c in self.tokenizer.decode([plan['ids'][index]])))

    def test_precision_and_camera_units(self):
        self.assertEqual(digit_weights('123.456', 'polygons'), [8,8,8,1,.5,.5,.5])
        self.assertEqual(digit_weights('-0.123456', 'position'), [8,8,1,8,8,8,.5,.5,.5])
        self.assertEqual(digit_weights('1e-4', 'position'), [1,1,1,1])

    def test_each_scalar_has_equal_mass(self):
        text = '{"region_bbox_2d":[1,123.456789]}'
        plan = self.objective.plan(text)
        self.assertAlmostEqual(sum(plan['weights']), 2)
        first = plan['weights'][0]
        self.assertAlmostEqual(first, 1)
        self.assertAlmostEqual(sum(plan['weights'][1:]), 1)

    def tensors(self, content):
        plan = self.objective.plan(content)
        # A small head uses remapped actual tokenizer IDs while alignment is
        # checked against the original IDs by the wrapper below.
        unique = sorted(set(plan['ids']))
        remap = {v:i for i,v in enumerate(unique)}
        class Head(torch.nn.Linear):
            def forward(head, x):
                small = super(Head, head).forward(x)
                # Real token IDs require the real output vocabulary dimension.
                full = x.new_full((*x.shape[:-1], max(unique)+1), -30.0)
                full[..., unique] = small
                return full
        torch.manual_seed(31)
        hidden = torch.randn(1, len(plan['ids'])+3, 5, requires_grad=True)
        labels = torch.tensor([[-100]*3 + plan['ids']])
        head = Head(5, len(remap), bias=False)
        return plan, hidden, labels, head

    def test_causal_chunk_loss_and_gradients_match_dense(self):
        content = '{"region_bbox_2d":[12.3,45.6,78.9,100]}'
        plan, hidden, labels, head = self.tensors(content)
        value, count = self.objective(hidden, head, labels, content, chunk_size=3)
        positions = torch.tensor(plan['positions']) + 3
        weights = torch.tensor(plan['weights'])
        expected = (F.cross_entropy(head(hidden[0,positions-1]), labels[0,positions], reduction='none')*weights).sum()/weights.sum()
        self.assertEqual(count, 4)
        torch.testing.assert_close(value, expected)
        actual_grad = torch.autograd.grad(value, (hidden, head.weight), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (hidden, head.weight))
        for a,b in zip(actual_grad,expected_grad):
            torch.testing.assert_close(a,b)
            self.assertTrue(torch.isfinite(a).all())
            self.assertGreater(float(a.abs().sum()),0)

    def test_alignment_fails_closed(self):
        content = '{"point_2d":[12,34]}'
        _, hidden, labels, head = self.tensors(content)
        labels[0,-1] += 1
        with self.assertRaisesRegex(AssertionError, 'alignment'):
            self.objective(hidden,head,labels,content)

    def test_no_geometry_has_zero_auxiliary_gradient(self):
        content = '{"label":"3 cups","answer":"B"}'
        _, hidden, labels, head = self.tensors(content)
        loss,n = self.objective(hidden,head,labels,content)
        self.assertEqual((float(loss),n),(0,0))
        loss.backward()
        self.assertEqual(float(hidden.grad.abs().sum()),0)

    def test_nonfinite_and_broken_json_rejected(self):
        for text in ['{"point_2d":[NaN,2]}','{"point_2d":[1,2]']:
            with self.assertRaises(ValueError):
                coordinate_spans(text)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
