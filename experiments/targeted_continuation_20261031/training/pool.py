"""Training-only corrective replay with explicit coverage of rare native forms.

All 4,756,000 native training records remain reachable. Native forms receive
equal update slots; inventories and actual repeated exposures remain separate.
This candidate module is not imported by the running final evaluation.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

O = Path(__file__).resolve().parent
C = O.parent
R = C.parent
FORMS = ('depth_order', 'point', 'direction', 'bbox', 'metric_comparison',
         'affordance', 'relation_graph', 'detection', 'dof', 'mask', 'action_cot')
SCHEDULES = {
    'repair': ['depth', 'fsd', 'vacancy', 'point', 'native', 'qa', 'depth', 'fsd',
               'vacancy', 'mask', 'native', 'qa', 'point', 'fsd', 'vacancy', 'prism'],
    'mopd': ['native', 'fsd', 'native', 'depth', 'native', 'vacancy', 'native', 'qa',
             'native', 'point', 'native', 'vacancy', 'native', 'mask', 'native', 'prism'],
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def permute(ordinal, count, seed):
    stride = 1000003
    while math.gcd(stride, count) != 1:
        stride += 2
    return (ordinal * stride + seed) % count


class Pool:
    def __init__(self, spec):
        self.spec = spec
        self.seed = spec['sampling_seed']
        self.source_offsets = spec['source_offsets']
        index = json.loads((R / 'training_repair/full_training_index/manifest.json').read_text())
        self.export = Path(index['export'])
        assert sha(self.export / 'manifest.json') == index['export_manifest_sha256']
        self.index = index
        self.offsets, self.handles, self.rows = {}, {}, {}
        for form in FORMS:
            info = index['files'][f'train_{form}.jsonl']
            assert sha(info['index_file']) == info['index_sha256']
            offsets = np.load(info['index_file'], mmap_mode='r')
            assert len(offsets) == info['records']
            self.offsets[form] = offsets
            self.handles[form] = (self.export / f'train_{form}.jsonl').open('rb')
        assert sum(map(len, self.offsets.values())) == 4756000
        for name, binding in spec['datasets'].items():
            assert sha(binding['path']) == binding['sha256'], name
            with open(binding['path']) as stream:
                rows = [json.loads(line) for line in stream]
            assert len(rows) == binding['records']
            assert all(row['gsa_meta']['split'] == 'train' for row in rows)
            self.rows[name] = rows
        with (self.export / 'train_dof.jsonl').open() as stream:
            self.rows['prism'] += [json.loads(line) for line in stream]
        assert len(self.rows['prism']) == 16544

    def offset(self, stage, source):
        value = self.source_offsets[stage][source]
        assert isinstance(value, int) and value >= 0
        return value

    def native_row(self, stage, form, ordinal, source):
        offsets = self.offsets[form]
        selected = permute(ordinal + self.offset(stage, source), len(offsets), self.seed)
        offset, length = offsets[selected]
        handle = self.handles[form]
        handle.seek(int(offset))
        row = json.loads(handle.read(int(length)))
        assert row['gsa_form'] == form
        return row

    def sample(self, stage, step, micro, rank, world, global_batch):
        assert global_batch % world == 0 and 0 <= rank < world
        assert 0 <= micro < global_batch // world
        schedule = SCHEDULES[stage]
        cycle, slot = divmod(step, len(schedule))
        kind = schedule[slot]
        occurrence = cycle * schedule.count(kind) + schedule[:slot].count(kind)
        ordinal = occurrence * global_batch + micro * world + rank
        if kind == 'native':
            form = FORMS[occurrence % len(FORMS)]
            ordinal = (occurrence // len(FORMS)) * global_batch + micro * world + rank
            row = self.native_row(stage, form, ordinal, 'native:' + form)
            source = 'native_distillation' if stage == 'mopd' else 'original_native'
        elif kind in {'depth', 'point', 'mask'}:
            form = {'depth': 'depth_order', 'point': 'point', 'mask': 'mask'}[kind]
            row = self.native_row(stage, form, ordinal, kind)
            source = {'depth': 'depth_replay', 'point': 'point_replay', 'mask': 'mask_replay'}[kind]
        else:
            name = kind
            if kind == 'qa':
                name = ('ai2d', 'scienceqa', 'textvqa')[occurrence % 3]
                ordinal = (occurrence // 3) * global_batch + micro * world + rank
            rows = self.rows[name]
            row = rows[permute(ordinal + self.offset(stage, name), len(rows), self.seed)]
            source = name
        assert row['gsa_meta']['split'] == 'train'
        return deepcopy(row), kind, source
