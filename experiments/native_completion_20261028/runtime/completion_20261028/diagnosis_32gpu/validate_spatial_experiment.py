"""Check complete deterministic schedules and representative real encodings."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from spatial_train import ARMS, Pool, STEPS, initial_checkpoint, sha, ROOT
from training_data import encode, processor


def main():
    started = time.time()
    outputs = {}
    anchors = {}
    examples = {}
    for arm in ARMS:
        pool = Pool(arm)
        counts, forms = Counter(), Counter()
        uids, image_ids = defaultdict(set), defaultdict(set)
        common = []
        for micro in range(STEPS*8):
            for rank in range(4):
                row, source = pool.sample(micro, rank, 4)
                assert row['gsa_meta']['split'] == 'train'
                assert len(row['messages']) == 2
                assert row['messages'][0]['content'].count('<image>') == 1
                assert Path(row['images'][0]).is_file()
                counts[source] += 1
                forms[row['gsa_form']] += 1
                uids[source].add(row['canonical_uid'])
                image_ids[source].add(row['images'][0])
                if source in ('original_native','general_qa'):
                    common.append(row['canonical_uid'])
                key = source + ':' + row['gsa_form']
                if key not in examples or sum(len(m['content']) for m in row['messages']) > sum(len(m['content']) for m in examples[key]['messages']):
                    examples[key] = row
        family = arm.split('_')[0]
        if family in anchors:
            assert common == anchors[family], 'Original native/QA controls must be exactly matched'
        else:
            anchors[family] = common
        assert sum(counts.values()) == STEPS*32
        assert counts['original_native'] == 8192 and counts['general_qa'] == 8192
        if arm.endswith('spatial'):
            assert counts['fsd_goal'] == 12288 and counts['refspatial_vacancy'] == 4096
        else:
            assert counts['original_point_control'] == 16384
        outputs[arm] = dict(initial=initial_checkpoint(arm), planned_exposures=dict(counts),
            planned_unique_records={k:len(v) for k,v in uids.items()}, planned_image_paths={k:len(v) for k,v in image_ids.items()}, forms=dict(forms))
        print(arm, counts, flush=True)
    p = processor()
    encodings = {}
    for key, row in examples.items():
        data = encode(p, row)
        encodings[key] = dict(tokens=data['input_ids'].shape[1], supervised_tokens=int((data['labels']!=-100).sum()), image_grid=data['image_grid_thw'].tolist())
        assert encodings[key]['supervised_tokens'] > 0
    report = dict(state='passed_cpu_preflight', schedules=outputs, real_encoding_extremes=encodings,
        planned_not_actual_exposure=True, actual_gpu_runtime_verified=False, elapsed=time.time()-started,
        bindings={str(ROOT/n):sha(ROOT/n) for n in ['spatial_train.py','spatial_node.py','merge_spatial.py','spatial_benchmark_worker.py','score_spatial_benchmarks.py','score_spatial_native.py','spatial_replay/manifest.json']})
    (ROOT/'spatial_cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CPU preflight PASS', flush=True)


if __name__ == '__main__':
    main()
