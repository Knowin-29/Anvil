"""CPU-only integration check of new source pairs with the production encoder."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
O = Path(__file__).resolve().parent
C = O.parent
sys.path.insert(0, str(C.parent))
from training_data import encode, processor


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


manifest_path = C / 'refspatial_grounding_20261101/manifest.json'
manifest = json.loads(manifest_path.read_text())
p = processor()
results = {}
for kind, info in manifest['files'].items():
    assert sha(info['path']) == info['sha256']
    first = largest = None
    count = 0
    max_bytes = 0
    for line in Path(info['path']).open():
        row = json.loads(line)
        content = row['messages'][0]['content'] + row['messages'][1]['content']
        size = len(content.encode('utf-8'))
        assert size + 1536 + 2048 < 16384, 'Refuse a potentially clipped training record'
        assert row['gsa_meta']['split'] == 'train' and len(row['images']) == 1
        assert row['messages'][0]['content'].count('<image>') == 1
        if first is None:
            first = row
        if largest is None or size > max_bytes:
            largest, max_bytes = row, size
        count += 1
    assert count == info['records']
    encoded = []
    for row in [first, largest]:
        data = encode(p, row)
        assert data['input_ids'].device.type == 'cpu'
        encoded.append(dict(uid=row['canonical_uid'], tokens=int(data['input_ids'].shape[-1]),
                            target_tokens=int((data['labels'] != -100).sum())))
    results[kind] = dict(records_checked=count, file_sha256=info['sha256'],
                         max_text_utf8_bytes=max_bytes, encoded_first_and_longest=encoded)
out = O / 'supplement_encoding_validation.json'
with out.open('x') as stream:
    json.dump(dict(state='passed_cpu_only', sources=results, manifest_sha256=sha(manifest_path),
                   validator_sha256=sha(__file__), gpu_jobs=0), stream, indent=2)
    stream.write('\n')
print(out)
