"""Exercise restoration against real saved state and real discarded log suffixes."""
import importlib.util,json,tempfile,unittest,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;V=W.parent/'visual_geometry_candidate_20261104'
spec=importlib.util.spec_from_file_location('restore_actual',V/'resume_state.py');restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
class ResumeTests(unittest.TestCase):
 def test_actual_all32_exposure_prefixes(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for rank in range(32):
    source=V/f'runs/repair/exposure_rank{rank}.jsonl';target=root/source.name;shutil.copyfile(source,target);before=target.read_bytes()
    saved=restore.restore_exposures(target,768,rank,2);rows=[json.loads(l) for l in target.read_text().splitlines()]
    self.assertEqual(len(rows),1536);self.assertEqual(rows[-1]['step'],768);self.assertEqual(Path(saved).read_bytes(),before)
 def test_actual_committed_logging(self):
  state=json.loads((V/'runs/repair/checkpoint-768/trainer_state.json').read_text())
  with tempfile.TemporaryDirectory() as td:
   target=Path(td)/'logging.jsonl';shutil.copyfile(V/'runs/repair/logging.jsonl',target);before=target.read_bytes()
   saved=restore.restore_logs(target,state['log_history'],768)
   self.assertEqual([json.loads(l) for l in target.read_text().splitlines()],state['log_history']);self.assertEqual(Path(saved).read_bytes(),before)
if __name__=='__main__':unittest.main()
