"""CPU-only regression checks for exact geometry checkpoint preupload binding."""
import copy
import runpy
import unittest
from pathlib import Path


MODULE = runpy.run_path(str(Path(__file__).with_name('preupload_geometry_merged_blobs_v2.py')))


class GeometryPreuploadBindingTests(unittest.TestCase):
    def setUp(self):
        checkpoint = MODULE['PIPELINE'] / 'runs/mopd/checkpoint-256'
        keys = [f'ancestor/checkpoint-{i}' for i in range(30)] + ['geometry_mopd/checkpoint-256']
        self.spec = dict(schema='anvil-final-evaluation-spec-v1', training_run=str(checkpoint.parent),
                         checkpoint=str(checkpoint), model=str(MODULE['SPEC'].parent / 'merged_model'),
                         native_records=580000, general_records=10105,
                         model_files={f'model-{i}.safetensors': str(i) for i in range(4)})
        self.pipeline = dict(key_checkpoints=keys)
        self.backup = dict(state='complete_remote_verified_intermediate_checkpoints',
                           repository=MODULE['REPOSITORY'], key_checkpoints=[dict(path=k) for k in keys],
                           files={
                               'checkpoints/geometry_mopd/checkpoint-256/adapter_model.safetensors': dict(sha256='geometry'),
                               'checkpoints/format_mopd/checkpoint-256/adapter_model.safetensors': dict(sha256='old-format'),
                           })
        self.complete = dict(checkpoint=str(checkpoint), adapter_sha256='geometry')

    def validate(self):
        MODULE['validate_identity'](self.spec, self.pipeline, self.backup)
        MODULE['validate_checkpoint_binding'](self.spec, self.backup, self.complete)

    def test_exact_geometry_checkpoint_with_distinct_parent_is_accepted(self):
        self.validate()

    def test_stale_parent_adapter_is_rejected(self):
        self.backup['files']['checkpoints/geometry_mopd/checkpoint-256/adapter_model.safetensors']['sha256'] = 'old-format'
        with self.assertRaises(ValueError):
            self.validate()

    def test_different_completed_checkpoint_is_rejected(self):
        self.complete['checkpoint'] = str(MODULE['PIPELINE'] / 'runs/mopd/checkpoint-128')
        with self.assertRaises(ValueError):
            self.validate()

    def test_different_evaluation_checkpoint_is_rejected(self):
        self.spec['checkpoint'] = str(MODULE['PIPELINE'] / 'runs/mopd/checkpoint-128')
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_backup_is_rejected(self):
        self.backup['key_checkpoints'].pop()
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_backup_does_not_satisfy_count(self):
        self.backup['key_checkpoints'][-1] = copy.deepcopy(self.backup['key_checkpoints'][0])
        with self.assertRaises(ValueError):
            self.validate()

    def test_incomplete_native_scope_is_rejected(self):
        self.spec['native_records'] -= 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_unsafe_shard_path_is_rejected(self):
        self.spec['model_files']['../model-0.safetensors'] = self.spec['model_files'].pop('model-0.safetensors')
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
