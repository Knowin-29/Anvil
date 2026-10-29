"""Pair fixed student G/S samples with complete teacher evaluations by exact ID."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;C=D.parent


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(directory,evidence):
    receipt=read(directory/'complete.json');out=directory/'evaluation'
    metrics=read(out/'metrics.json');assert sha(out/'metrics.json')==receipt['metrics_sha256']
    protocol=read(out/'protocol.json');protocol_sha=sha(out/'protocol.json')
    assert metrics['state']=='complete_scope_shard' and protocol['smoke_rows']==0
    evidence[str(out/'protocol.json')]=protocol_sha;evidence[str(out/'metrics.json')]=sha(out/'metrics.json')
    for name,digest in metrics['prediction_sha256'].items():
        path=out/name;assert sha(path)==digest;evidence[str(path)]=digest
        for row in map(json.loads,path.open()):
            assert row['protocol_sha256']==protocol_sha
            yield row,protocol


def main():
    target=D/'student_matched_teachers_comparison.json';assert not target.exists()
    spec=read(D/'student_matched_native_spec.json');report=read(D/'student_matched_native_comparison.json')
    assert report['state']=='complete_sample_requires_matched_teacher_review'
    assert report['spec_sha256']==sha(D/'student_matched_native_spec.json')
    student={};evidence={str(D/'student_matched_native_spec.json'):sha(D/'student_matched_native_spec.json')}
    for task in spec['selection']['tasks']:
        for row,protocol in rows(D/'student_matched_native'/str(task['shard']),evidence):
            key=(row['native_form'],row['id']);assert key not in student;student[key]=(row,protocol)
    assert len(student)==spec['selection']['expected_records']
    # Every selected row must have an exact matching teacher reference and protocol.
    teachers={'G':[D/'G_spatial_priority_native'/str(i) for i in range(16)],
              'S':[C/'qualification_v2'/task['id'] for task in read(C/'evaluation_tasks.json') if task['model']=='S']}
    fields=['export_manifest_sha256','selection_sha256','context_length','image_max_tokens','dtype','seed',
            'temperature','thinking','token_budgets','concurrency','max_num_batched_tokens','multimodal_max_pixels',
            'repetition_penalty','cycle_guard','response_policy','malformed_and_aborted_examples_in_denominator']
    groups=defaultdict(lambda:dict(n=0,student=defaultdict(float),teacher=defaultdict(float)));matched=set()
    teacher_models={}
    for family,directories in teachers.items():
        models=set()
        for directory in directories:
            for row,protocol in rows(directory,evidence):
                models.add(protocol['model']);key=(row['native_form'],row['id'])
                if key not in student:continue
                assert key not in matched;matched.add(key)
                baseline,student_protocol=student[key]
                for field in fields:assert protocol[field]==student_protocol[field],(family,field)
                assert row['evaluation_content_sha256']==baseline['evaluation_content_sha256']
                assert row['source']==baseline['source'] and set(row['metrics'])==set(baseline['metrics'])
                group=groups[(family,row['native_form'],row['source'])];group['n']+=1
                for metric,value in row['metrics'].items():
                    group['teacher'][metric]+=float(value);group['student'][metric]+=float(baseline['metrics'][metric])
        assert len(models)==1;teacher_models[family]=next(iter(models))
    assert matched==set(student)
    values=[]
    for (family,form,source),group in sorted(groups.items()):
        n=group['n'];values.append(dict(family=family,form=form,source=source,n=n,
            metrics={key:dict(student=group['student'][key]/n,teacher=value/n,
                             teacher_minus_student=(value-group['student'][key])/n)
                     for key,value in group['teacher'].items()}))
    target.write_text(json.dumps(dict(state='complete_matched_sample_requires_quality_review',n=len(matched),
        student_checkpoint=spec['checkpoint'],student_adapter_sha256=spec['adapter_sha256'],teacher_models=teacher_models,
        groups=values,full_validation=False,teacher_qualified=False,final_mopd=False,
        uncertainty='Paired descriptive estimates on8randomordinalstrata; no independence or statistical significance claim.',
        lower_is_better=['truncated','normalized_image_distance_mae'],
        sample_selection=spec['selection'],evidence=evidence,code_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(target)


if __name__=='__main__':main()
