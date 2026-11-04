import copy
import json
from pathlib import Path
import unittest
from model_presentation import build_plan, destination, digest, encode, FEATURED

class PresentationTests(unittest.TestCase):
    def setUp(self):
        pipeline=json.loads((Path(__file__).resolve().parent.parent/'visual_geometry_candidate_20261104/spec.json').read_text())
        self.keys=pipeline['key_checkpoints'];self.nonancestors=set(pipeline['retained_nonancestor_checkpoints'])
        self.files={'README.md':digest(b'original card'),'release_manifest.json':digest(b'original manifest'),
                    'evidence/evaluation_report.json':digest(b'original report'),'model-00001.safetensors':digest(b'merged')}
        self.seals={};self.originals={}
        for key in self.keys:
            values={'README.md':digest(b'original stage card'),'adapter_model.safetensors':digest(key.encode()),
                    'optimizer.pt':digest(b'optimizer'),'scheduler.pt':digest(b'scheduler')}
            values.update({f'rng_state_{i}.pth':digest(f'rng{i}'.encode()) for i in range(32)})
            seal={'schema':'anvil-complete-checkpoint-v1','files':values}
            self.seals[key]=seal
            for name,item in values.items(): self.files['checkpoints/'+key+'/'+name]=item
            origin='checkpoints/'+key+'/checkpoint_complete.json'
            self.originals[origin]=encode(seal);self.files[origin]=digest(encode(seal))
        self.report={'paper_aligned_public':{'cv_bench':80.1234},'paper_target':{'cv_bench':88.21},
                     'paper_aligned_macro':80.1234,'native':{'groups':[]},'general':{}}

    def plan(self):
        return build_plan(self.files,self.report,'test-representation','a'*40,self.seals,self.originals,self.nonancestors)

    def test_five_featured_and_thirty_four_archived(self):
        mapping,added,expected=self.plan()
        adapters=[n for n in expected if n.startswith('checkpoints/') and n.endswith('/adapter_model.safetensors')]
        self.assertEqual(len(adapters),39)
        self.assertEqual(sum('/archive/' in n for n in adapters),34)
        for slug,_ in FEATURED.values(): self.assertIn('checkpoints/'+slug+'/adapter_model.safetensors',expected)

    def test_all_weight_optimizer_and_rng_bytes_preserved(self):
        mapping,added,expected=self.plan()
        for name,item in self.files.items():
            if name.endswith(('.safetensors','.pt','.pth')): self.assertEqual(expected[mapping[name]],item)

    def test_checkpoint_seals_change_documentation_only(self):
        mapping,added,expected=self.plan()
        for key,old in self.seals.items():
            base=destination('checkpoints/'+key+'/README.md').rsplit('/',1)[0]
            updated=json.loads(added[base+'/checkpoint_complete.json'])
            self.assertEqual(updated['files']['README.md'],digest(added[base+'/README.md']))
            for name,item in old['files'].items():
                if name!='README.md': self.assertEqual(updated['files'][name],item)

    def test_manifest_includes_all_preserved_and_added_files(self):
        mapping,added,expected=self.plan()
        manifest=json.loads(added['release_manifest.json'])
        self.assertEqual(set(manifest['files']),set(expected)-{'release_manifest.json'})
        self.assertIn('docs/provenance/original_release_manifest.json',expected)
        for name,content in added.items():self.assertEqual(expected[name],digest(content))
        self.assertFalse(any(n.startswith(('backup/','evidence/')) for n in expected))

    def test_ancestry_is_preserved(self):
        _,added,_=self.plan()
        index=json.loads(added['docs/provenance/checkpoint_index.json'])
        for item in index['checkpoints']:
            key=item['original'].removeprefix('checkpoints/')
            self.assertEqual(item['in_final_optimizer_ancestry'],key not in self.nonancestors)

    def test_current_measurements_used_in_card(self):
        _,added,_=self.plan()
        self.assertIn('| CV-Bench | 80.12 | 88.21 |',added['README.md'].decode())
        self.assertNotIn('[More Information Needed]',added['README.md'].decode())

    def test_incomplete_release_rejected(self):
        del self.files['checkpoints/coordinate_mopd/checkpoint-256/adapter_model.safetensors']
        with self.assertRaises(ValueError):self.plan()

    def test_changed_original_seal_rejected(self):
        self.originals[next(iter(self.originals))]=b'changed'
        with self.assertRaises(ValueError):self.plan()

    def test_ambiguous_destination_rejected(self):
        self.files['docs/evaluation/evaluation_report.json']=digest(b'other')
        with self.assertRaises(ValueError):self.plan()

if __name__=='__main__':unittest.main()
