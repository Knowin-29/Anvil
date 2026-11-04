"""Compute the complete versioned Anvil-Eval score from bound native predictions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

VERSION='anvil-eval-record-score-v1.1'
COUNTS={('action_cot','ALFRED'):3327,('affordance','ShareRobot-Bench'):193,
    ('bbox','coco_train2017'):81280,('depth_order','PRISM'):3601,
    ('detection','coco_train2017'):3738,('direction','coco_train2017'):89045,
    ('dof','PRISM'):580,('mask','coco_train2017'):81250,
    ('metric_comparison','coco_train2017'):88984,('point','coco_train2017'):81244,
    ('point','pixmo_points'):143016,('relation_graph','coco_train2017'):3742}
DIRECT={'bbox':'matched_box_iou','detection':'box_f1_iou50','mask':'matched_mask_iou',
    'direction':'reference_relation_f1','depth_order':'reference_relation_f1',
    'relation_graph':'reference_relation_f1','affordance':'region_iou',
    'action_cot':'expert_continuation_exact'}

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def measurement(metrics,name,maximum=1.,tolerance=1e-12):
    value=metrics[name]
    require(type(value) in (int,float,bool) and math.isfinite(value) and 0<=value<=maximum+tolerance,
            'Invalid metric: '+name)
    return min(float(value),maximum)

def record_score(form,source,metrics,finish_reason):
    require((form,source) in COUNTS,'Unexpected form/source')
    require(all(type(v) in (int,float,bool) and math.isfinite(v) for v in metrics.values()),'Nonfinite metric')
    # Validate required metrics even for invalid outputs; missing measurements are corruption.
    valid=measurement(metrics,'contract_valid')==1 and measurement(metrics,'json_valid')==1
    truncated=measurement(metrics,'truncated')!=0
    if form in DIRECT:score=measurement(metrics,DIRECT[form])
    elif form=='point':score=measurement(metrics,'interior_point_f1' if source=='coco_train2017' else 'annotated_point_f1_r50')
    elif form=='metric_comparison':
        score=measurement(metrics,'relation_identity_exact')*(1-measurement(metrics,'normalized_image_distance_mae',math.sqrt(2),tolerance=1e-6)/math.sqrt(2))
    else:
        score=(1-measurement(metrics,'rotation_error_degrees',180)/180)*(1-measurement(metrics,'normalized_projected_contact_l2',math.sqrt(2))/math.sqrt(2))
    return max(0.,score) if valid and not truncated and finish_reason=='stop' else 0.

def aggregate(report_path,spec_path,output):
    report_path=Path(report_path).resolve();spec_path=Path(spec_path).resolve();output=Path(output)
    report=json.loads(report_path.read_text());spec=json.loads(spec_path.read_text())
    require(report['state']=='complete_requires_quality_review' and report['final_spec_sha256']==sha(spec_path),'Incomplete/mismatched report')
    require(report['native']['n']==580000,'Wrong native scope')
    report_groups={(g['form'],g['source']):g for g in report['native']['groups']}
    require(len(report_groups)==len(report['native']['groups']) and {k:g['n'] for k,g in report_groups.items()}==COUNTS,'Incomplete form/source coverage')
    root=report_path.parent/'native'
    paths={Path(p):h for p,h in report['artifacts_sha256'].items()
           if Path(p).name.startswith('predictions_') and Path(p).suffix=='.jsonl' and Path(p).parent.parent==root}
    require(paths,'Missing native prediction artifacts')
    seen=set();counts=Counter();sums=Counter();metric_sums=defaultdict(Counter);bindings={};protocols={}
    for path,expected_sha in sorted(paths.items()):
        require(path.is_file() and sha(path)==expected_sha,'Prediction artifact changed')
        shard=int(path.parent.name);protocol_path=path.parent/'protocol.json'
        if shard not in protocols:
            protocol=json.loads(protocol_path.read_text())
            require(protocol['shard']==shard and protocol['shards']==spec['native_shards'],'Wrong shard protocol')
            protocols[shard]=sha(protocol_path)
        for line in path.open():
            row=json.loads(line);key=(row['native_form'],row['ordinal']);group=(row['native_form'],row['source'])
            require(key not in seen and row['ordinal']%spec['native_shards']==shard,'Duplicate/misassigned record')
            require(row['protocol_sha256']==protocols[shard],'Mixed prediction protocol')
            require(row['record_sha256']==digest({k:v for k,v in row.items() if k!='record_sha256'}),'Changed row binding')
            require(hashlib.sha256(row['response'].encode()).hexdigest()==row['response_sha256'],'Changed response')
            score=record_score(*group,row['metrics'],row['finish_reason'])
            seen.add(key);counts[group]+=1;sums[group]+=score;metric_sums[group].update(row['metrics'])
        require(sha(path)==expected_sha,'Prediction changed during scoring')
        bindings[str(path)]=expected_sha
    require(dict(counts)==COUNTS and len(seen)==580000,'Incomplete native records')
    form_counts=Counter()
    for (form,_),n in COUNTS.items():form_counts[form]+=n
    require(seen=={(form,i) for form,n in form_counts.items() for i in range(n)},'Native ordinal coverage differs')
    groups=[]
    for group,n in sorted(counts.items()):
        expected=report_groups[group]['mean'];actual=metric_sums[group]
        require(set(actual)==set(expected),'Metric field coverage differs')
        for name,value in expected.items():
            require(math.isclose(actual[name]/n,value,rel_tol=1e-10,abs_tol=1e-10),'Metrics differ from verified full report')
        groups.append({'form':group[0],'source':group[1],'n':n,'score':100*sums[group]/n,'native_metrics':expected})
    result={'schema':VERSION,'state':'complete','n':580000,'score':100*math.fsum(sums.values())/580000,
        'aggregation':'record_weighted','groups':groups,'report_sha256':sha(report_path),
        'final_spec_sha256':sha(spec_path),'prediction_artifacts_sha256':bindings,
        'scorer_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md')),
        'protocol_origin':'user_authorized_explicit_mapping_20261104',
        'historical_scorer_equivalence_established':False,'paper_reference':65.35,
        'user_reference_minimum':58.815,'final_quality_accepted':False}
    require(not output.exists(),'Use a new score artifact; never overwrite evaluated results')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.pending');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(output)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=aggregate(args.report,args.spec,args.output)
    print(json.dumps({k:result[k] for k in ['schema','state','n','score','final_quality_accepted']}))
