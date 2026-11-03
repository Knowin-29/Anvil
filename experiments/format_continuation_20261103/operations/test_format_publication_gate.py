"""CPU-only acceptance-binding checks; never create an actual acceptance file."""
from pathlib import Path
import copy,hashlib,json,runpy,time
V=Path(__file__).resolve().parent;C=V.parent
helper=runpy.run_path(str(V/'finish_format_delivery.py'));decide=helper['quality_decision'];sha=helper['sha']
D=C/'delivery_runs/focus_repair_20261101'
review=json.loads((D/'quality_review.json').read_text());report=json.loads((D/'evaluation/final_report.json').read_text())
rsha=sha(D/'evaluation/final_report.json');ssha=sha(D/'evaluation_spec.json')
checks=[]
assert decide(review,report,rsha,ssha) is False;checks.append('real_rejected_focus_stays_rejected')
accepted=copy.deepcopy(review);accepted['accepted']=True
assert decide(accepted,report,rsha,ssha) is True;checks.append('synthetic_explicit_bound_acceptance_recognized_not_persisted')
for name,mutation in [
 ('wrong_report_rejected',lambda v,r:v.update(report_sha256='0'*64)),
 ('wrong_spec_rejected',lambda v,r:v.update(final_spec_sha256='0'*64)),
 ('missing_benchmark_rejected',lambda v,r:v['benchmark_assessments'].pop('embspatial')),
 ('changed_score_rejected',lambda v,r:v['benchmark_assessments']['embspatial'].update(score=99.9)),
 ('changed_paper_target_rejected',lambda v,r:v['benchmark_assessments']['embspatial'].update(paper=0)),
 ('partial_native_rejected',lambda v,r:r['native'].update(n=100)),
 ('missing_rationale_rejected',lambda v,r:v.update(rationale='')),
 ('nonbool_acceptance_rejected',lambda v,r:v.update(accepted='yes')),
]:
 v,r=copy.deepcopy(accepted),copy.deepcopy(report);mutation(v,r)
 try:decide(v,r,rsha,ssha)
 except (ValueError,KeyError):checks.append(name)
 else:raise AssertionError(name)
out=dict(state='passed_cpu_only',checks=checks,synthetic_fixtures=True,real_accepted_review_written=False,
 gpu_work=False,hub_calls=False,source_sha256=sha(V/'finish_format_delivery.py'),validator_sha256=sha(__file__),completed=time.time())
(V/'format_publication_gate_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
