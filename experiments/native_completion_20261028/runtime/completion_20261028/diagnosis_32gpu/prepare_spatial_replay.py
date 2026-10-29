"""Audit and deduplicate supplemental training sources before any GPU use.

No evaluation answers enter training. Both encoded and decoded image identities,
and conservative perceptual-neighbor exclusions protect all frozen evaluation
images. A train-only development partition is additionally separated by image.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'Anvil/tools'))
from arxiv_source_scale import fingerprint, sha, save

OUT = ROOT / 'spatial_replay'
FSD = Path('/Knowin/foundation/data/datasets/originDatasets_transfer_from_hpc3/FSD_points_rft_fsd_free_point_train_32790_test_300_0425')
VAC = Path('/Knowin/demo/weilinruan/Anvil/04_Experiments/iclr_evidence/engineering/data/source_controls_20260915/vacancy/train.jsonl')
EXPORT = ROOT.parent.parent.parent / 'arxiv/native_training_export_20261012_v4_r2'


class Protected:
    """Eight blocks guarantee retrieval of every radius <= 7 hash neighbor."""
    def __init__(self):
        self.exact = {}
        self.buckets = defaultdict(list)

    def add(self, fp, identity):
        for k in ('sha256', 'decoded_rgb_sha256'):
            self.exact.setdefault(fp[k], identity)
        for k in ('phash64', 'dhash64'):
            h = int(fp[k], 16)
            for b in range(8):
                self.buckets[(k, b, (h >> (b * 8)) & 255)].append((h, identity))

    def match(self, fp):
        for k in ('sha256', 'decoded_rgb_sha256'):
            if fp[k] in self.exact:
                return dict(reason=k, protected=self.exact[fp[k]])
        # Conservative OR, not AND: visually similar records are excluded even
        # if only one hash suggests overlap. False positives reduce pool size.
        for k in ('phash64', 'dhash64'):
            h = int(fp[k], 16)
            for b in range(8):
                for other, identity in self.buckets.get((k, b, (h >> (8 * b)) & 255), ()):
                    distance = (h ^ other).bit_count()
                    if distance <= 6:
                        return dict(reason=k + '_radius6', distance=distance, protected=identity)
        return None


def rows(p):
    return [json.loads(l) for l in Path(p).open()]


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'manifest.json').exists(), 'Never overwrite a sealed training pool'
    started = time.time()
    original = json.loads((EXPORT / 'manifest.json').read_text())
    db = sqlite3.connect('file:' + original['pool_index'] + '?mode=ro', uri=True)
    protected = Protected()
    original_rgb = set()
    native_count = 0
    for decoded, file_sha, path, split, ph, dh in db.execute('SELECT decoded_sha256,file_sha256,path,split,phash64,dhash64 FROM images'):
        original_rgb.add(decoded)
        if split == 'benchmark':
            protected.add(dict(sha256=file_sha, decoded_rgb_sha256=decoded, phash64=ph, dhash64=dh), path)
            native_count += 1
    inputs = json.loads((ROOT / 'inputs/manifest.json').read_text())
    # Hash and bind the actual immutable public/general evaluation images.
    def frozen(path):
        fp = fingerprint(path)
        assert fp['sha256'] == inputs['images'][path]
        return path, fp
    with ThreadPoolExecutor(max_workers=16) as executor:
        for path, fp in executor.map(frozen, inputs['images']):
            protected.add(fp, path)
    print('protected native/public/general', native_count, len(inputs['images']), flush=True)
    train_path = FSD / 'data_train_internvl_no_think.jsonl'
    test_path = FSD / 'data_test_internvl_no_think.jsonl'
    source_bindings = {str(p): sha(p) for p in (train_path, test_path, VAC)}
    train, test = rows(train_path), rows(test_path)
    assert len(train) == 32790 and len(test) == 300
    test_ids = {r['id'] for r in test}
    assert test_ids.isdisjoint(r['id'] for r in train)
    for row in test:
        path = str(FSD / row['image'])
        protected.add(fingerprint(path), path)
    # Reserve existing historical vacancy development images as well. These
    # labels are never consumed by the training builder.
    dev_path = VAC.parent / 'dev.jsonl'
    historical_dev_images = set()
    if dev_path.exists():
        source_bindings[str(dev_path)] = sha(dev_path)
        historical_dev_images = {p for row in rows(dev_path) for p in row['images']}
        for path in historical_dev_images:
            protected.add(fingerprint(path), path)
    candidates, duplicate_counts, seen = [], Counter(), set()
    source_counts = Counter()
    for source, raw_rows in [('fsd_goal', train), ('refspatial_vacancy', rows(VAC))]:
        for ordinal, row in enumerate(raw_rows):
            source_counts[source] += 1
            if source == 'fsd_goal':
                assert [x['from'] for x in row['conversations']] == ['human', 'gpt']
                path = str(FSD / row['image'])
                prompt, target = [x['value'] for x in row['conversations']]
                source_id = row['id']
            else:
                assert row['gsa_meta']['source'] == 'refspatial_3d_vacant'
                assert len(row['images']) == 1
                path = row['images'][0]
                prompt, target = [x['content'] for x in row['messages']]
                source_id = str(ordinal)
            assert prompt.startswith('<image>') and prompt.count('<image>') == 1
            match = re.fullmatch(r'\s*<point>(\[.*\])</point>\s*', target)
            assert match, (source, source_id)
            points = json.loads(match[1])
            assert points and all(len(p) == 2 and all(type(v) in (int, float) and 0 <= v <= 1000 for v in p) for p in points)
            key = hashlib.sha256(json.dumps([str(Path(path).resolve()), prompt, points], sort_keys=True).encode()).hexdigest()
            if key in seen:
                duplicate_counts[source] += 1
                continue
            seen.add(key)
            candidates.append(dict(source=source, source_id=source_id, uid=source + ':' + key,
                path=path, prompt=prompt, target=target, record_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()))
    paths = sorted({r['path'] for r in candidates})
    cache = OUT / 'image_fingerprints.jsonl'
    fingerprints = {r['path']:r['fingerprint'] for r in rows(cache)} if cache.exists() else {}
    def inspect(path):
        fp = fingerprint(path)
        return dict(path=path, fingerprint=fp)
    with cache.open('a') as output, ThreadPoolExecutor(max_workers=16) as executor:
        for i, row in enumerate(executor.map(inspect, [p for p in paths if p not in fingerprints])):
            fingerprints[row['path']] = row['fingerprint']
            output.write(json.dumps(row) + '\n')
            if i % 2000 == 0:
                output.flush()
                print('fingerprinted', i, 'of', len(paths), flush=True)
    matches = {}
    for i, path in enumerate(paths):
        matches[path] = protected.match(fingerprints[path])
        if i % 2000 == 0:
            print('leakage checked', i, 'of', len(paths), flush=True)
    # Exclude an entire RefSpatial scan if any frame is flagged, to avoid
    # cross-view transfer of the heldout geometry. FSD has no trustworthy
    # trajectory ID in these annotations, so that stronger claim is not made.
    def scan(path):
        return Path(path).name.split('_')[0]
    blocked_scans = {scan(r['path']) for r in candidates if r['source'] == 'refspatial_vacancy' and matches[r['path']]}
    counts, development, excluded, image_sets = Counter(), Counter(), Counter(), defaultdict(set)
    streams = {s:(OUT / (s + '.jsonl')).open('x') for s in source_counts}
    development_stream = (OUT / 'development.jsonl').open('x')
    exclusions = (OUT / 'exclusions.jsonl').open('x')
    semantic_seen = set()
    for row in candidates:
        source, path = row['source'], row['path']
        fp = fingerprints[path]
        collision = matches[path]
        if source == 'refspatial_vacancy' and scan(path) in blocked_scans:
            collision = collision or dict(reason='same_refspatial_scan_as_flagged_frame', scan=scan(path))
        if collision:
            excluded[source] += 1
            exclusions.write(json.dumps(dict(uid=row['uid'], path=path, **collision)) + '\n')
            continue
        identity = (fp['decoded_rgb_sha256'], row['prompt'], row['target'])
        if identity in semantic_seen:
            duplicate_counts[source] += 1
            continue
        semantic_seen.add(identity)
        # RefSpatial scan-level dev split; FSD image-level. Neither uses scores.
        group = 'scan:' + scan(path) if source == 'refspatial_vacancy' else fp['decoded_rgb_sha256']
        isdev = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20 == 0
        binding = dict(source_dataset=source, source_id=row['source_id'], source_record_sha256=row['record_sha256'],
                       source_split='existing_train_annotation', image_path=path, **fp)
        item = dict(id=row['uid'], canonical_uid=row['uid'], gsa_form='point', gsa_family='G',
                    gsa_meta=dict(split='development' if isdev else 'train', binding=binding), images=[path],
                    messages=[dict(role='user', content=row['prompt']), dict(role='assistant', content=row['target'])])
        (development_stream if isdev else streams[source]).write(json.dumps(item, ensure_ascii=False) + '\n')
        (development if isdev else counts)[source] += 1
        if not isdev:
            image_sets[source].add(fp['decoded_rgb_sha256'])
    for stream in [*streams.values(), development_stream, exclusions]:
        stream.close()
    assert all(counts[s] >= 256 for s in source_counts), dict(counts)
    report = dict(state='verified_train_only', source_records=dict(source_counts), duplicate_records_removed=dict(duplicate_counts),
        additional_training_records=dict(counts), development=dict(development), excluded_heldout_or_similar=dict(excluded),
        source_sha256=source_bindings, files_sha256={s:sha(OUT / (s + '.jsonl')) for s in counts},
        additional_distinct_images={s:len(v) for s,v in image_sets.items()},
        new_to_original_images={s:len(v - original_rgb) for s,v in image_sets.items()},
        protected_native_images=native_count, protected_public_general_images=len(inputs['images']),
        fsd_test_ids_disjoint=True, fsd_test_images_protected=300, vacancy_blocked_scans=sorted(blocked_scans),
        heldout_inputs_manifest_sha256=sha(ROOT / 'inputs/manifest.json'), original_export_manifest_sha256=sha(EXPORT / 'manifest.json'),
        original_selection_manifest_sha256=sha(original['pool_manifest']),
        image_fingerprints_sha256=sha(cache), exclusions_sha256=sha(OUT / 'exclusions.jsonl'),
        leakage_rule='exclude file SHA or decoded RGB SHA; OR pHash64/dHash64 Hamming <= 6; RefSpatial any-hit entire scan excluded',
        limitation='Perceptual exclusion is conservative, not proof of absence of every semantic or trajectory-level overlap; FSD source has no trusted trajectory ID.',
        targets='Unchanged source point coordinates in thousandths and source prompts; deduplicated only. No benchmark target used.',
        original_sft_records=5220000, supplemental_not_original_scale=True, source_data_modified=False,
        code_sha256=sha(__file__), fingerprint_code_sha256=sha(ROOT.parent / 'Anvil/tools/arxiv_source_scale.py'),
        elapsed=time.time()-started)
    save(OUT / 'manifest.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
