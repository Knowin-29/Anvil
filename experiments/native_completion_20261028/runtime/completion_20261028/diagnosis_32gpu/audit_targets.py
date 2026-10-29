"""Re-read source-bound targets and audit geometry/cycles without changing input data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent
CODE=ROOT.parent/'Anvil'
sys.path.insert(0,str(CODE))
from trains.flywheel.native_training import validate_target
EXPORT=ROOT.parent.parent.parent/'arxiv/native_training_export_20261012_v4_r2'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def compact(value):
    if isinstance(value,float):return round(value,6)
    if isinstance(value,list):return [compact(x) for x in value]
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
    return value

def audit(form):
    manifest=json.loads((EXPORT/'manifest.json').read_text());sources={x['expected_sha256']:x['path'] for x in manifest['source_inputs']}
    handles={};counts=Counter();lengths=[];bad=[];started=time.time();file=EXPORT/('train_'+form+'.jsonl')
    for raw in file.open('rb'):
        row=json.loads(raw);binding=row['gsa_meta']['binding'];source_sha=binding['source_sha256']
        if source_sha not in handles:
            assert sha(sources[source_sha])==source_sha
            handles[source_sha]=open(sources[source_sha],'rb')
        stream=handles[source_sha];stream.seek(binding['source_offset']);original=stream.read(binding['source_size'])
        assert hashlib.sha256(original).hexdigest()==binding['source_line_sha256']
        source=json.loads(original)
        assert source['messages']==row['messages'],row['id']
        target=json.loads(row['messages'][1]['content'])
        validate_target(form,row['gsa_native_variant'],target,row['messages'][0]['content'])
        counts['rows']+=1;counts['source_equal']+=1;counts['valid_targets']+=1
        if form=='dof':
            new=compact(target)
            validate_target(form,row['gsa_native_variant'],new,row['messages'][0]['content'])
            text=json.dumps(new,sort_keys=True,separators=(',',':'),ensure_ascii=False)
            counts['six_decimal_still_geometry_valid']+=1
            counts['original_target_characters']+=len(row['messages'][1]['content']);counts['compact_target_characters']+=len(text)
        else:
            for obj in target:
                for polygon in obj['polygons']:
                    run=1
                    for i in range(1,len(polygon)):
                        run=run+1 if polygon[i]==polygon[i-1] else 1
                        if run==16:counts['polygons_with16_identical_consecutive_points']+=1
        if counts['rows']%50000==0:print(form,counts['rows'],flush=True)
    assert counts['rows']==manifest['files'][file.name]['records']
    result=dict(form=form,counts=dict(counts),source_file_sha256=sha(file),elapsed=time.time()-started,
        source_data_modified=False,scope='Exact export/source-message equality and structural/geometry validation, not a universal human semantic-quality claim')
    (ROOT/('audit_'+form+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':audit(sys.argv[1])
