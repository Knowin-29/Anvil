"""Admit source SAT train examples, excluding heldout images and close copies.

Targets remain exactly the source answers. Remove only the exact injected
reasoning-format suffix from the source train prompt. No benchmark predictions
or answers are read by this builder; evaluation image identities are protected.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

C = Path(__file__).resolve().parent
D = C / 'diagnosis_32gpu'
sys.path.insert(0, str(D))
from prepare_spatial_replay import Protected, fingerprint, sha, EXPORT

SOURCE = Path('/data/sources/originDatasets_transfer_from_hpc3/SAT_rft_spatial_qa_train_80000_test_4000_0428')
OUT = C / 'sat_replay_20261102'
SUFFIX = '\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. The overall format being: <think> reasoning process here </think><answer> answer here </answer>'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def image_paths(row):
    values = [row['image']] if 'image' in row else row['images']
    assert isinstance(values, list) and values
    result = []
    for value in values:
        relative = Path(value)
        assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'images'
        result.append(str(SOURCE / relative))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    assert not list(OUT.iterdir()), 'Never overwrite a preparation output'
    started = time.time()
    train_path = SOURCE / 'data_train_internvl.jsonl'
    test_path = SOURCE / 'data_test_internvl.jsonl'
    bindings = {str(p): sha(p) for p in [train_path, test_path]}
    train = [json.loads(line) for line in train_path.open()]
    test = [json.loads(line) for line in test_path.open()]
    assert len(train) == 80000 and len(test) == 4000
    assert {r['id'] for r in train}.isdisjoint(r['id'] for r in test)
    protected = Protected()
    original = json.loads((EXPORT / 'manifest.json').read_text())
    db = sqlite3.connect('file:' + original['pool_index'] + '?mode=ro', uri=True)
    native_count = 0
    for decoded, file_sha, path, split, ph, dh in db.execute('SELECT decoded_sha256,file_sha256,path,split,phash64,dhash64 FROM images'):
        if split == 'benchmark':
            protected.add(dict(sha256=file_sha, decoded_rgb_sha256=decoded, phash64=ph, dhash64=dh), path)
            native_count += 1
    db.close()
    inputs = json.loads((D / 'inputs/manifest.json').read_text())
    test_images = {path for row in test for path in image_paths(row)}
    blocked_paths = sorted(set(inputs['images']) | test_images)

    def inspect(path):
        return path, fingerprint(path)

    with ThreadPoolExecutor(max_workers=16) as pool:
        for path, fp in pool.map(inspect, blocked_paths):
            if path in inputs['images']:
                assert fp['sha256'] == inputs['images'][path]
            protected.add(fp, path)
    print('heldout image protection ready', native_count, len(inputs['images']), len(test_images), flush=True)
    single_image_train = [row for row in train if len(image_paths(row)) == 1]
    paths = sorted({image_paths(row)[0] for row in single_image_train})
    fingerprints = {}
    with (OUT / 'image_fingerprints.jsonl').open('x') as cache, ThreadPoolExecutor(max_workers=16) as pool:
        for i, (path, fp) in enumerate(pool.map(inspect, paths), 1):
            fingerprints[path] = fp
            cache.write(json.dumps(dict(path=path, fingerprint=fp)) + '\n')
            if i % 5000 == 0:
                cache.flush(); print('fingerprinted', i, '/', len(paths), flush=True)
    matches = {}
    for i, path in enumerate(paths, 1):
        matches[path] = protected.match(fingerprints[path])
        if i % 10000 == 0:
            print('heldout exclusion checked', i, '/', len(paths), flush=True)
    counts = Counter(multi_image_not_supported=len(train)-len(single_image_train))
    seen, images = set(), set()
    output = OUT / 'sat.jsonl'
    with output.open('x') as stream, (OUT / 'exclusions.jsonl').open('x') as excluded:
        for ordinal, raw in enumerate(train):
            if len(image_paths(raw)) != 1:
                continue
            path = image_paths(raw)[0]
            if matches[path]:
                counts['heldout_or_similar'] += 1
                excluded.write(json.dumps(dict(source_id=raw['id'], image=path, match=matches[path])) + '\n')
                continue
            turns = raw['conversations']
            assert len(turns) == 2 and [x['from'] for x in turns] == ['human', 'gpt']
            prompt, target = [x['value'] for x in turns]
            assert isinstance(prompt, str) and isinstance(target, str) and target.strip()
            assert prompt.count('<image>') == 1 and prompt.startswith('<image>')
            # Skip unknown wrappers rather than silently removing question text.
            if not prompt.endswith(SUFFIX):
                counts['unsupported_prompt_wrapper'] += 1
                continue
            prompt = prompt[:-len(SUFFIX)].strip()
            assert prompt and '<think>' not in prompt and '<answer>' not in prompt
            fp = fingerprints[path]
            assert [raw['width'], raw['height']] == [fp['width'], fp['height']]
            key = digest([fp['decoded_rgb_sha256'], prompt, target])
            if key in seen:
                counts['duplicate'] += 1
                continue
            seen.add(key); images.add(fp['decoded_rgb_sha256'])
            uid = 'sat_source_train:' + key
            binding = dict(source_dataset='SAT', source_id=raw['id'], source_ordinal=ordinal,
                           source_file=str(train_path), source_file_sha256=bindings[str(train_path)],
                           source_record_sha256=digest(raw), source_split='existing_source_train_annotation',
                           image_path=path, **fp)
            row = dict(id=uid, canonical_uid=uid, gsa_form='QA', gsa_family='QA',
                       gsa_meta=dict(split='train', binding=binding), images=[path],
                       messages=[dict(role='user', content=prompt), dict(role='assistant', content=target)])
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts['admitted'] += 1
    assert counts['admitted'] >= 10000, 'Insufficient source SAT examples after exclusions'
    assert {str(p): sha(p) for p in [train_path, test_path]} == bindings
    manifest = dict(schema='anvil-source-SAT-replay-v1', state='verified_train_only',
                    files=dict(sat=dict(path=str(output), sha256=sha(output), records=counts['admitted'],
                                       bytes=output.stat().st_size, unique_images=len(images))),
                    source_files=bindings, source_train_records=len(train), source_test_records=len(test),
                    counts=dict(counts), protected_native_images=native_count,
                    protected_public_general_images=len(inputs['images']), protected_source_test_images=len(test_images),
                    inputs_manifest_sha256=sha(D / 'inputs/manifest.json'),
                    original_export_manifest_sha256=sha(EXPORT / 'manifest.json'),
                    image_fingerprints_sha256=sha(OUT / 'image_fingerprints.jsonl'),
                    exclusions_sha256=sha(OUT / 'exclusions.jsonl'),
                    source_targets_regenerated=False, source_answers_unchanged=True,
                    removed_prompt_suffix=SUFFIX, evaluation_answers_consumed=False,
                    original_sft_records=5220000, supplemental_not_original_scale=True,
                    builder_sha256=sha(__file__),
                    limitations=['SAT source train/test annotation split is inherited, not newly certified as an official scene split.',
                                 'Exact/decoded/perceptual image exclusion does not establish all semantic or scene-level disjointness.',
                                 'Source annotation correctness is not independently established for every example.'],
                    completed=time.time(), elapsed=time.time()-started)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
