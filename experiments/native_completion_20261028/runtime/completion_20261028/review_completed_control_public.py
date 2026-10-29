"""Score completed public subsets while independent general inference continues."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
C=Path(__file__).resolve().parent;D=C/'diagnosis_32gpu';O=C/'objective_controls_20261029'
sys.path[:0]=[str(D/'code'),str(O)]
from objective_control_common import bound,control_checkpoint,read,require,save,sha
from gsa_eval.registry import BENCHMARKS
from gsa_eval.datasets import Sample


def main():
    root=O/'evaluation/low_native_sft';spec_path=root/'spec.json';spec=read(spec_path)
    require(spec['schema']=='anvil-objective-control-evaluation-v1' and spec['final_mopd'] is False,
            'Not an objective control')
    control_checkpoint(spec['checkpoint'],full=False)
    for filename,expected in spec['frozen_files'].items():require(sha(filename)==expected,'Changed inference/scoring dependency')
    for item in spec['evidence'].values():require(sha(item['path'])==item['sha256'],'Changed control evidence')
    manifest=read(D/'inputs/manifest.json');evidence={str(spec_path):sha(spec_path)};protocols=[]
    for shard in range(8):
        folder=root/'public'/str(shard);state=read(folder/'status.json');protocol=read(folder/'protocol.json')
        require((folder/'complete.json').exists() or state.get('benchmark')=='retention',
                'Public benchmarks have not all finished')
        require(protocol['checkpoint_spec_sha256']==sha(spec_path) and protocol['shard']==shard and protocol['shards']==8,
                'Mixed prediction protocols')
        require(protocol['model']==spec['model'] and protocol['inputs_sha256']==sha(D/'inputs/manifest.json')
                and protocol['smoke'] is False,'Mixed model/input binding')
        for filename,digest in protocol['weights'].items():require(spec['model_files'][filename]==digest,'Mixed weights')
        protocols.append(protocol);evidence[str(folder/'protocol.json')]=sha(folder/'protocol.json')
    scores={}
    for name,info in manifest['public'].items():
        path=D/'inputs'/(name+'.jsonl');require(sha(path)==info['sha256'],'Changed benchmark input')
        rows=[json.loads(line) for line in path.open()];by_id={}
        for shard in range(8):
            path=root/'public'/str(shard)/(name+'.jsonl');digest=sha(path)
            for line in path.open():
                row=json.loads(line);require(row['uid'] not in by_id,'Duplicate public UID')
                require(hashlib.sha256(row['response'].encode()).hexdigest()==row['response_sha256'],'Changed response')
                by_id[row['uid']]=row
            require(sha(path)==digest,'Public file changed during review');evidence[str(path)]=digest
        require(set(by_id)=={str(row['uid']) for row in rows} and len(rows)==spec['public_records'][name],
                'Incomplete public denominator')
        samples=[Sample(**{k:v for k,v in row.items() if k!='max_tokens'}) for row in rows]
        scorer=BENCHMARKS[name];kwargs={'spec':scorer} if 'spec' in inspect.signature(scorer.scorer).parameters else {}
        score,detail,fails=scorer.scorer(samples,[by_id[str(row.uid)]['response'] for row in samples],**kwargs)
        scores[name]=dict(n=len(samples),score=score,detail=detail,parse_failures=fails,
                         capped=sum(row['hit_token_cap'] for row in by_id.values()))
    aligned={k:(v['detail']['per_group']['location']['mean_in_mask'] if k=='refspatial_bench' else v['score']) for k,v in scores.items()}
    initial=read(D/'student_adaptation_public_comparison.json')['results']['student_adaptation']['paper_aligned_public']
    result=dict(state='complete_seven_public_only_general_pending',arm='low_native_sft',
        public=scores,paper_aligned_public=aligned,paper_aligned_macro=sum(aligned.values())/7,
        minus_initial={k:v-initial[k] for k,v in aligned.items()},prediction_evidence=evidence,
        general_scored_here=False,full_native_validation=False,final_quality_accepted=False,weights_uploaded=False,
        adaptive_validation=True,code=bound(Path(__file__)),completed=time.time())
    save(O/'low_native_sft_public_interim.json',result)
    print(json.dumps({k:result[k] for k in ['state','paper_aligned_public','paper_aligned_macro','minus_initial']},indent=2))


if __name__=='__main__':main()
