"""CPU-only checks for the destructive cleanup boundary; temporary files only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).with_name('cleanup_verified_weights.py')
spec = importlib.util.spec_from_file_location('cleanup_format', SOURCE)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class CleanupBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='anvil-cleanup-boundary-')
        self.root = Path(self.temp.name)
        self.original_control = cleanup.CONTROL
        cleanup.CONTROL = self.root / "control"
        self.original_root = cleanup.C
        self.original_verification_root = cleanup.V
        cleanup.C = self.root
        cleanup.V = self.root / 'verification'
        self.run = self.root / 'campaign/runs/mopd'
        self.run.mkdir(parents=True)
        self.package = self.root / 'delivery/package'
        self.package.mkdir(parents=True)
        self.merged = self.root / 'delivery/merged_model'
        self.merged.mkdir()
        self.files = {}
        for step in (32, 64):
            folder = self.run / f'checkpoint-{step}'
            folder.mkdir()
            names = ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt']
            names += [f'rng_state_{rank}.pth' for rank in range(32)]
            payloads = {}
            for name in names:
                data = f'{step}:{name}'.encode()
                (folder / name).write_bytes(data)
                payloads[name] = self.binding(data)
                if step == 64:
                    dest = self.package / 'checkpoints/mopd/checkpoint-64' / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    self.files['checkpoints/mopd/checkpoint-64/' + name] = self.binding(data)
            seal = dict(schema='anvil-complete-checkpoint-v1', world_size=32, global_step=step, files=payloads)
            path = folder / 'checkpoint_complete.json'
            path.write_text(json.dumps(seal))
            if step == 64:
                self.files['checkpoints/mopd/checkpoint-64/checkpoint_complete.json'] = self.binding(path.read_bytes())
        (self.run / 'complete.json').write_text(json.dumps(dict(stage='mopd_training_complete', smoke=False,
            steps=64, checkpoint=str(self.run / 'checkpoint-64'),
            adapter_sha256=self.files['checkpoints/mopd/checkpoint-64/adapter_model.safetensors']['sha256'])))
        data = b'merged weights'
        for folder in (self.package, self.merged):
            (folder / 'model.safetensors').write_bytes(data)
        self.files['model.safetensors'] = self.binding(data)
        self.manifest = dict(files=self.files)
        self.pipeline = dict(key_checkpoint_sources={'mopd/checkpoint-64': str(self.run / 'checkpoint-64')})
        self.evaluation = dict(model=str(self.merged), model_files={'model.safetensors': self.binding(data)['sha256']})
        self.sentinel = self.root / 'training_data.jsonl'
        self.sentinel.write_text('keep data and logs')
        self.prior = cleanup.V / 'grounded_superseded_cleanup_20261103'
        self.prior.mkdir(parents=True)
        self.pruned = [dict(path=str(self.root / 'historical' / f'{i}.pt')) for i in range(4410)]
        self.write_prior_ledger()
        self.second_prior = cleanup.V / 'format_superseded_cleanup_quota_20261103'
        self.second_prior.mkdir()
        self.second_pruned = [dict(path=str(self.root / 'historical_format' / f'{i}.pt')) for i in range(4410)]
        self.write_second_ledger()
        self.original_resume = cleanup.RESUME
        self.original_pipeline = cleanup.PIPELINE
        cleanup.RESUME = self.root / 'resume'
        cleanup.PIPELINE = self.root / 'campaign'
        cleanup.RESUME.mkdir()
        self.quota_entries = [dict(path=str(cleanup.C / 'geometry_objective_v2_20261104/runs/repair' / f'checkpoint-{64*(i+1)}' / 'optimizer.pt'), bytes=1, sha256='a'*64, inode=i+1, mtime_ns=1) for i in range(126)]
        self.write_current_quota()

    def write_current_quota(self, protected=None):
        preview=cleanup.RESUME/'optimizer_cleanup_preview.json'
        preview.write_text(json.dumps(dict(candidates=self.quota_entries, protected=protected or [])))
        (cleanup.RESUME/'optimizer_cleanup_deleted.jsonl').write_text(''.join(json.dumps(dict(e,deleted=1))+'\n' for e in self.quota_entries))
        (cleanup.RESUME/'optimizer_cleanup_complete.json').write_text(json.dumps(dict(state='complete',files_removed=126,bytes_freed=126,preview_sha256=hashlib.sha256(preview.read_bytes()).hexdigest())))

    def write_second_ledger(self):
        intent = self.second_prior / 'intent.json'
        intent.write_text(json.dumps(dict(files=self.second_pruned)))
        (self.second_prior / 'complete.json').write_text(json.dumps(dict(state='completed',
            files=4410, intent_sha256=hashlib.sha256(intent.read_bytes()).hexdigest())))

    def write_prior_ledger(self):
        intent = self.prior / 'intent.json'
        intent.write_text(json.dumps(dict(files=self.pruned)))
        (self.prior / 'complete.json').write_text(json.dumps(dict(state='completed',
            files=4410, intent_sha256=hashlib.sha256(intent.read_bytes()).hexdigest())))


    def tearDown(self):
        cleanup.RESUME = self.original_resume
        cleanup.PIPELINE = self.original_pipeline
        cleanup.CONTROL = self.original_control
        cleanup.C = self.original_root
        cleanup.V = self.original_verification_root
        self.temp.cleanup()

    @staticmethod
    def binding(data):
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def candidates(self):
        return cleanup.candidate_sources(self.manifest, self.pipeline, self.evaluation, self.package)

    def test_exact_uploaded_and_superseded_payload_scope(self):
        items = self.candidates()
        self.assertEqual(len(items), 107)
        self.assertEqual(sum(x['reason'] == 'superseded_nonkey_save' for x in items.values()), 35)
        self.assertNotIn(str(self.sentinel), items)
        self.assertTrue(all(Path(p).suffix in {'.safetensors', '.pt', '.pth'} for p in items))
        self.assertTrue((self.run / 'checkpoint-64/checkpoint_complete.json').exists())

    def test_requires_completed_stage(self):
        (self.run / 'complete.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.candidates()
        self.assertTrue(self.sentinel.exists())

    def test_requires_backed_up_last_checkpoint(self):
        self.pipeline['key_checkpoint_sources'] = {'mopd/checkpoint-32': str(self.run / 'checkpoint-32')}
        with self.assertRaisesRegex(ValueError, 'Last stage checkpoint'):
            self.candidates()

    def test_rejects_missing_remote_payload(self):
        del self.files['checkpoints/mopd/checkpoint-64/optimizer.pt']
        with self.assertRaises(KeyError):
            self.candidates()

    def test_rejects_incomplete_rng_inventory(self):
        path = self.run / 'checkpoint-32/checkpoint_complete.json'
        seal = json.loads(path.read_text())
        del seal['files']['rng_state_31.pth']
        path.write_text(json.dumps(seal))
        with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint payload'):
            self.candidates()

    def test_rejects_changed_local_payload(self):
        path = self.run / 'checkpoint-64/optimizer.pt'
        expected = self.files['checkpoints/mopd/checkpoint-64/optimizer.pt']
        path.write_bytes(b'x' * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'Local payload differs'):
            cleanup.identity(path, expected, self.root)

    def test_rejects_symlink_and_outside_paths(self):
        link = self.root / 'alias'
        link.symlink_to(self.run)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            cleanup.within(link / 'checkpoint-64/optimizer.pt', self.root)
        with self.assertRaisesRegex(ValueError, 'outside allowed'):
            cleanup.within(Path('/etc/passwd'), self.root)

    def test_rejects_mutation_after_preflight(self):
        path = self.run / 'checkpoint-64/optimizer.pt'
        item = cleanup.identity(path, self.files['checkpoints/mopd/checkpoint-64/optimizer.pt'], self.root)
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed since preflight'):
            cleanup.unchanged(item)

    def test_rejects_live_local_payload_reader(self):
        path = self.run / 'checkpoint-64/optimizer.pt'
        process = subprocess.Popen([sys.executable, '-u', '-c',
            'import sys,time;f=open(sys.argv[1],"rb");print("ready",flush=True);time.sleep(20)', str(path)],
            stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), 'ready')
            with self.assertRaisesRegex(ValueError, 'payload open'):
                cleanup.no_local_readers([dict(path=str(path))])
        finally:
            process.terminate(); process.wait(timeout=5)
            process.stdout.close()

    def test_rejects_live_campaign_publisher(self):
        process = subprocess.Popen([sys.executable, '-u', '-c',
            'import time;print("ready",flush=True);time.sleep(20)', str(cleanup.CONTROL / 'finish_delivery.py')],
            stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), 'ready')
            with self.assertRaisesRegex(ValueError, 'campaign writer still running'):
                cleanup.no_local_readers([])
        finally:
            process.terminate();process.wait(timeout=5);process.stdout.close()

    def test_known_previously_pruned_nonkey_is_skipped(self):
        path = self.run / 'checkpoint-32/optimizer.pt'
        self.pruned[0] = dict(path=str(path))
        self.write_prior_ledger()
        path.unlink()
        items = self.candidates()
        self.assertEqual(len(items), 106)
        self.assertNotIn(str(path), items)
        self.assertTrue(self.sentinel.exists())

    def test_unexplained_missing_nonkey_is_rejected(self):
        (self.run / 'checkpoint-32/optimizer.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing payload'):
            self.candidates()

    def test_missing_key_is_never_exempted_by_prior_ledger(self):
        path = self.run / 'checkpoint-64/optimizer.pt'
        self.pruned[0] = dict(path=str(path))
        self.write_prior_ledger()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing payload'):
            self.candidates()

    def test_changed_prior_ledger_is_rejected(self):
        (self.prior / 'intent.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Changed prior cleanup ledger'):
            self.candidates()

    def test_quota_pruned_nonkey_is_skipped(self):
        path = self.run / 'checkpoint-32/optimizer.pt'
        self.second_pruned[0] = dict(path=str(path))
        self.write_second_ledger()
        path.unlink()
        self.assertEqual(len(self.candidates()), 106)

    def test_changed_quota_ledger_is_rejected(self):
        (self.second_prior / 'intent.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Changed prior cleanup ledger'):
            self.candidates()

    def test_quota_ledger_cannot_exempt_missing_key(self):
        path = self.run / 'checkpoint-64/optimizer.pt'
        self.second_pruned[0] = dict(path=str(path))
        self.write_second_ledger()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing payload'):
            self.candidates()

    def test_current_quota_ledger_tamper_rejected(self):
        (cleanup.RESUME/'optimizer_cleanup_deleted.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Incomplete quota cleanup ledger'):
            self.candidates()

    def test_current_quota_scope_rejected(self):
        self.quota_entries[0]['path']=str(cleanup.C/'geometry_objective_v2_20261104/runs/repair/checkpoint-64/adapter_model.safetensors')
        self.write_current_quota()
        with self.assertRaisesRegex(ValueError,'Unexpected quota cleanup scope'):
            self.candidates()

    def test_current_quota_protected_rejected(self):
        self.write_current_quota([str(Path(self.quota_entries[0]['path']).parent)])
        with self.assertRaisesRegex(ValueError,'Protected or recreated'):
            self.candidates()

    def test_current_quota_recreated_rejected(self):
        path=Path(self.quota_entries[0]['path']);path.parent.mkdir(parents=True);path.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'Protected or recreated'):
            self.candidates()

    def test_current_quota_entry_hash_tamper_rejected(self):
        lines=[dict(e,deleted=1) for e in self.quota_entries];lines[0]['sha256']='b'*64
        (cleanup.RESUME/'optimizer_cleanup_deleted.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in lines))
        with self.assertRaisesRegex(ValueError,'Quota deletion differs'):
            self.candidates()

    def test_requires_accepted_complete_publication(self):
        good=dict(state='accepted_final_model_and39_checkpoints_remotely_verified',revision='a'*40,quality_review_sha256='b'*64)
        cleanup.require_release_complete(good,'a'*40,'b'*64)
        for field,value in [('state','quality_rejected_no_publication'),('revision','c'*40),('quality_review_sha256','d'*64)]:
            bad=dict(good);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                cleanup.require_release_complete(bad,'a'*40,'b'*64)

    def test_current_unfinished_release_cannot_enter_cleanup(self):
        args = SimpleNamespace(manifest=self.root / 'missing/release_manifest.json',
            final_receipt=self.root / 'missing/final.json', backup_receipt=self.root / 'missing/backup.json',
            verification_receipt=self.root / 'missing/verified.json')
        with self.assertRaises(FileNotFoundError):
            cleanup.load_release(args)
        self.assertEqual(self.sentinel.read_text(), 'keep data and logs')


if __name__ == '__main__':
    unittest.main(verbosity=2)
