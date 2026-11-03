"""Coordinate-focused auxiliary CE without changing targets or native scoring.

The ordinary full-answer loss remains in the trainer. This auxiliary gives
coarse spatial digits more weight than sub-pixel decimal precision. It is an
optimization hypothesis, not a claim of geometric validation improvement.
"""
import json
import math
import re

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FIELDS = frozenset({'polygons', 'bbox_2d', 'region_bbox_2d', 'point_2d', 'position'})
NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
DECODER = json.JSONDecoder()


def coordinate_spans(text):
    """Return only numeric leaves inside actual geometry JSON fields.

    Numeric labels, copied instructions, rotation code IDs and string contents
    are deliberately excluded. Every character is checked as valid JSON.
    """
    json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    result = []

    def space(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i

    def value(i, field=None):
        i = space(i)
        if text[i] == '{':
            i = space(i + 1)
            if text[i] == '}':
                return i + 1
            while True:
                key, end = DECODER.raw_decode(text, i)
                assert isinstance(key, str)
                i = space(end)
                assert text[i] == ':'
                i = space(value(i + 1, key if key in FIELDS else None))
                if text[i] == '}':
                    return i + 1
                assert text[i] == ','
                i = space(i + 1)
        if text[i] == '[':
            i = space(i + 1)
            if text[i] == ']':
                return i + 1
            while True:
                i = space(value(i, field))
                if text[i] == ']':
                    return i + 1
                assert text[i] == ','
                i = space(i + 1)
        obj, end = DECODER.raw_decode(text, i)
        if type(obj) in (int, float) and field is not None:
            assert math.isfinite(obj)
            assert NUMBER.fullmatch(text[i:end])
            result.append((i, end, field))
        return end

    assert space(value(0)) == len(text)
    return result


def digit_weights(number, field):
    """Weights retain all exact digits, including fine precision and signs."""
    if 'e' in number.lower():
        # Rare scientific notation gets uniform supervision; never infer an
        # incorrect decimal place from its exponent's text.
        return [1.0] * len(number)
    dot = number.find('.')
    if dot < 0:
        dot = len(number)
    weights = []
    for i, char in enumerate(number):
        if char in '+-':
            weight = 8.0
        elif char == '.':
            weight = 1.0
        elif field == 'position':
            # Camera translations and transformed contact coordinates need
            # their leading fractional digits, unlike 0..1000 image pixels.
            weight = 8.0 if i < dot + 4 else 0.5
        else:
            weight = 8.0 if i < dot else 0.5
        weights.append(weight)
    return weights


class GeometryLoss:
    def __init__(self, tokenizer, prefix, suffix='<|im_end|>\n'):
        self.tokenizer, self.prefix, self.suffix = tokenizer, prefix, suffix

    def plan(self, content):
        text = self.prefix + content + self.suffix
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        spans = coordinate_spans(content)
        token_weights = {}
        # Normalize each scalar before combining: a 6-decimal coordinate must
        # not outweigh another coordinate merely because its spelling is long.
        cursor = 0
        for left, right, field in spans:
            chars = digit_weights(content[left:right], field)
            left += len(self.prefix)
            right += len(self.prefix)
            while cursor < len(offsets) and offsets[cursor][1] <= left:
                cursor += 1
            selected = []
            j = cursor
            while j < len(offsets) and offsets[j][0] < right:
                lo, hi = offsets[j]
                a, b = max(lo, left), min(hi, right)
                if b > a:
                    selected.append((j, sum(chars[a-left:b-left])))
                j += 1
            assert selected, 'A coordinate was not covered by target tokens'
            total = sum(w for _, w in selected)
            for index, weight in selected:
                token_weights[index] = token_weights.get(index, 0.0) + weight / total
        positions = sorted(token_weights)
        return dict(ids=encoded['input_ids'], positions=positions,
                    weights=[token_weights[i] for i in positions], scalars=len(spans))

    def __call__(self, hidden, head, labels, content, chunk_size=128):
        meta = self.plan(content)
        assert labels.shape[0] == hidden.shape[0] == 1
        start = labels.shape[1] - len(meta['ids'])
        assert start >= 0 and labels[0, start:].tolist() == meta['ids'], 'Target-token alignment mismatch'
        if not meta['positions']:
            return hidden.sum() * 0.0, 0
        positions = torch.tensor(meta['positions'], device=hidden.device) + start
        assert int(positions.min()) > 0
        states = hidden[0, positions-1]
        targets = labels[0, positions]
        weights = torch.tensor(meta['weights'], device=hidden.device, dtype=torch.float32)
        total = states.sum() * 0.0

        def score(x, y, w):
            return (F.cross_entropy(head(x).float(), y, reduction='none') * w).sum()

        for i in range(0, len(positions), chunk_size):
            total = total + checkpoint(score, states[i:i+chunk_size], targets[i:i+chunk_size],
                                       weights[i:i+chunk_size], use_reentrant=False)
        return total / weights.sum(), meta['scalars']
