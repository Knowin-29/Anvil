"""Acceptance policy tests in temporary fixtures; never accept a real candidate."""
import copy,hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from quality_gate import validate_tolerance
C=Path(__file__).resolve().parent.parent
class ToleranceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.p=self.root/'hub_presentation_visual_20261104/user_acceptance_policy.json';self.p.parent.mkdir()
  self.s=self.root/'visual_native_score_20261104/final_native_score.json';self.s.parent.mkdir()
  orig=C/'anvil_eval_score_v1_1_20261104';shutil.copytree(orig,self.root/orig.name,ignore=shutil.ignore_patterns('*.log','__pycache__','final_native_score.json','previous_geometry_native_score.json'))
  self.policy=json.loads((C/'hub_presentation_visual_20261104/user_acceptance_policy.json').read_text());self.p.write_text(json.dumps(self.policy))
  self.score=dict(state='complete',n=580000,schema=self.policy['native_aggregate_protocol'],report_sha256='a'*64,final_spec_sha256='b'*64,score=58.815,scorer_sha256=self.sha(orig/'score_records.py'),protocol_sha256=self.sha(orig/'PROTOCOL.md'))
  self.report=dict(paper_aligned_public={'benchmark':90.},paper_target={'benchmark':100.})
  self.review=dict(user_acceptance_policy_sha256=self.sha(self.p),paper_native_tolerance_assessment=dict(accepted=True,measured_score=58.815,score_artifact=str(self.s)))
  self.write_score()
 def tearDown(self):self.tmp.cleanup()
 @staticmethod
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def write_score(self):
  self.s.write_text(json.dumps(self.score));self.review['paper_native_tolerance_assessment'].update(measured_score=self.score['score'],score_artifact_sha256=self.sha(self.s))
 def check(self):return validate_tolerance(self.review,self.report,'a'*64,'b'*64,root=self.root)
 def test_exact_threshold_and_improvement_pass(self):
  self.check();self.score['score']=80.;self.write_score();self.report['paper_aligned_public']['benchmark']=110.;self.check()
 def test_native_below_threshold_rejected(self):
  self.score['score']=50.476384441895995;self.write_score()
  with self.assertRaisesRegex(ValueError,'Native score below'):self.check()
 def test_public_below_threshold_rejected(self):
  self.report['paper_aligned_public']['benchmark']=89.999
  with self.assertRaisesRegex(ValueError,'Public benchmark'):self.check()
 def test_wrong_model_and_incomplete_native_rejected(self):
  for key,value in [('report_sha256','c'*64),('final_spec_sha256','c'*64),('n',579999)]:
   old=self.score[key];self.score[key]=value;self.write_score()
   with self.subTest(key=key),self.assertRaises(ValueError):self.check()
   self.score[key]=old
 def test_changed_protocol_rejected(self):
  (self.root/'anvil_eval_score_v1_1_20261104/PROTOCOL.md').write_text('changed')
  with self.assertRaisesRegex(ValueError,'Scoring implementation'):self.check()
 def test_review_cannot_substitute_score(self):
  self.review['paper_native_tolerance_assessment']['measured_score']=80.
  with self.assertRaises(ValueError):self.check()
 def test_unaccepted_native_rejected(self):
  self.review['paper_native_tolerance_assessment']['accepted']=False
  with self.assertRaises(ValueError):self.check()
if __name__=='__main__':unittest.main()
