"""Freeze complete public and VQA retention inputs for matched base/joint comparison."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
os.environ['ANVIL_BENCHMARK_ROOT']='/Knowin/foundation/data/datasets/transferred_datasets/EmbodiedAI_benchmark'
sys.path.insert(0,str(ROOT/'code'))
from gsa_eval.registry import BENCHMARKS

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    out=ROOT/'inputs';out.mkdir(exist_ok=True)
    manifest={'public':{},'retention':{},'images':{},'code':{},'source_root':os.environ['ANVIL_BENCHMARK_ROOT']}
    for name in ['cv_bench','embspatial','refspatial_bench','where2place','blink_relative_depth','erqa','vabench_point']:
        spec=BENCHMARKS[name];rows=spec.loader(ROOT/'assets')
        assert len(rows)==spec.n_expected,(name,len(rows),spec.n_expected)
        assert len({str(r.uid) for r in rows})==len(rows)
        dest=out/(name+'.jsonl')
        with dest.open('x') as f:
            for row in rows:
                x=asdict(row);assert all(Path(q).is_file() for q in row.image_paths)
                x.update(max_tokens=spec.max_tokens)
                f.write(json.dumps(x,ensure_ascii=False)+'\n')
                for path in row.image_paths:
                    if path not in manifest['images']:manifest['images'][path]=sha(path)
        manifest['public'][name]={'n':len(rows),'sha256':sha(dest),'max_tokens':spec.max_tokens}
        print(name,len(rows),flush=True)
    src=Path('/Knowin/demo/weilinruan/Anvil/04_Experiments/iclr_evidence/engineering/data/p5_retention_clean_run_seed17_v2/retention/inputs/runtime_index.jsonl')
    rows=[json.loads(l) for l in src.open()]
    counts=Counter(r['dataset'] for r in rows)
    assert counts=={'ai2d_test':3088,'scienceqa_test':2017,'textvqa_val':5000},counts
    assert len({r['global_uid'] for r in rows})==len(rows)
    for r in rows:
        path=r['image_path'];assert Path(path).is_file(),path
        if path not in manifest['images']:manifest['images'][path]=sha(path)
        assert manifest['images'][path]==r['image_sha256']
    dest=out/'retention.jsonl';dest.write_bytes(src.read_bytes())
    ref=Path('/Knowin/demo/weilinruan/GSA_collect/data/general_retention_v1/sources/vlmevalkit/vqa_eval.py')
    assert sha(ref)=='c04e0a1cad29cd8caa786ea1a96d2d945630304d756d1bba234b36e66c83f06d'
    manifest['retention']={'counts':dict(counts),'source':str(src),'sha256':sha(dest),'vqa_reference':str(ref),'vqa_reference_sha256':sha(ref)}
    manifest['code']={str(p.relative_to(ROOT/'code')):sha(p) for p in (ROOT/'code').rglob('*.py')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Complete inputs frozen',dict(counts),flush=True)

if __name__=='__main__':main()
