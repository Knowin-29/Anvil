"""Read-only comparison of active vacancy targets to upstream annotations.

This verifies conversion fidelity, not semantic label correctness, suitability
for a benchmark, or permission to expand the current sealed training pool.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

C = Path(__file__).resolve().parent
D = C / 'diagnosis_32gpu'
UPSTREAM = Path('/Knowin/foundation/data/datasets/originDatasets_transfer_from_hpc3/refspatial/3D')
HISTORICAL = Path('/Knowin/demo/weilinruan/Anvil/04_Experiments/iclr_evidence/engineering/data/source_controls_20260915/vacancy/train.jsonl')
RAW_SUFFIX = (' Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], '
              'where each tuple contains the x and y coordinates of a point satisfying '
              'the conditions above. The coordinates should be between 0 and 1, '
              'indicating the normalized pixel locations of the points in the image.')


def bound(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def point_target(text):
    m = re.fullmatch(r'\s*<point>(\[.*\])</point>\s*', text)
    assert m, 'Not a point target'
    points = json.loads(m[1])
    assert points and all(len(p) == 2 and all(type(v) in (int, float) and 0 <= v <= 1000 for v in p) for p in points)
    return tuple(tuple(p) for p in points)


def signature(row):
    return (Path(row['images'][0]).name, row['messages'][0]['content'],
            point_target(row['messages'][1]['content']))


def main():
    active = D / 'spatial_replay/refspatial_vacancy.jsonl'
    processed = UPSTREAM / 'vacant_qa_filtered_3D_groupLength.jsonl'
    raw = UPSTREAM / 'vacant_qa_qwenvl.jsonl'
    paths = [active, HISTORICAL, processed, raw, D / 'spatial_replay/manifest.json', Path(__file__)]
    before = {str(p):bound(p) for p in paths}
    active_rows = list(read_rows(active))
    desired = {signature(r):r for r in active_rows}
    assert len(desired) == len(active_rows) == 3384
    historical = Counter(signature(r) for r in read_rows(HISTORICAL))
    assert set(desired).issubset(historical)
    processed_seen = Counter(); raw_seen = Counter(); counts = Counter()
    mismatches = []; raw_deltas = []; removed_suffix = 0; invalid_raw_examples = []
    images = {key[0] for key in desired}
    processed_keys = set(); raw_keys = set(); source_scans = set()
    upstream_by_query = defaultdict(set)
    active_by_query = defaultdict(list)
    for key in desired:
        active_by_query[key[:2]].append(key)
    for row in read_rows(processed):
        counts['processed_rows'] += 1
        basename = Path(row['image']).name
        prompt, answer = [x['value'] for x in row['conversations']]
        points = point_target(answer)
        key = (basename, prompt, points)
        processed_keys.add(key)
        source_scans.add(basename.split('_')[0])
        if key in desired:
            processed_seen[key] += 1
            binding = desired[key]['gsa_meta']['binding']
            if (row['width'], row['height']) != (binding['width'], binding['height']):
                mismatches.append(dict(kind='dimensions', uid=desired[key]['canonical_uid']))
        if basename in images:
            upstream_by_query[(basename,prompt)].add(points)
    for row in read_rows(raw):
        counts['raw_rows'] += 1
        basename = Path(row['image']).name
        prompt, answer = [x['value'] for x in row['conversations']]
        if prompt.endswith(RAW_SUFFIX):
            prompt = prompt[:-len(RAW_SUFFIX)]
            removed_suffix += 1
        try:
            points = ast.literal_eval(answer)
            valid = isinstance(points, (list, tuple)) and bool(points) and all(
                isinstance(p, (list, tuple)) and len(p) == 2 and
                all(type(v) in (int,float) and 0 <= v <= 1 for v in p) for p in points)
        except (ValueError, SyntaxError):
            valid = False
        if not valid:
            counts['raw_invalid_or_empty_normalized_targets'] += 1
            if len(invalid_raw_examples) < 10:
                invalid_raw_examples.append(dict(id=row.get('id'), image=basename, target=answer))
            continue
        raw_keys.add((basename, prompt, tuple(tuple(p) for p in points)))
        for key in active_by_query.get((basename,prompt), []):
            target = key[2]
            if len(points) != len(target):
                continue
            delta = max(abs(1000*a-b) for p,q in zip(points,target) for a,b in zip(p,q))
            if delta <= .50000001:
                raw_seen[key] += 1
                raw_deltas.append(delta)
    missing_processed = set(desired)-set(processed_seen)
    missing_raw = set(desired)-set(raw_seen)
    for key in sorted(missing_processed | missing_raw):
        mismatches.append(dict(uid=desired[key]['canonical_uid'], missing_processed=key in missing_processed,
                               missing_raw=key in missing_raw))
    after = {str(p):bound(p) for p in paths}
    assert before == after, 'Input changed during read-only audit'
    report = dict(schema='anvil-vacancy-upstream-fidelity-v1',
        state='conversion_fidelity_verified' if not mismatches else 'requires_source_review',
        active_records=len(active_rows), active_unique_images=len(images),
        historical_rows=sum(historical.values()), historical_unique_records=len(historical),
        historical_repeated_rows=sum(historical.values())-len(historical),
        upstream=dict(**counts, processed_unique_records=len(processed_keys), raw_unique_records=len(raw_keys),
                      processed_scans=len(source_scans), known_raw_format_suffix_removed=removed_suffix),
        exact_processed_matches=len(processed_seen), raw_normalized_coordinate_matches=len(raw_seen),
        raw_coordinate_max_absolute_error_on_1000_scale=max(raw_deltas,default=None),
        dimension_mismatches=sum(m['kind']=='dimensions' for m in mismatches if 'kind' in m),
        active_queries_with_multiple_upstream_point_targets=sum(len(upstream_by_query[k[:2]])>1 for k in desired),
        mismatches=mismatches, raw_invalid_examples=invalid_raw_examples, evidence=before,
        limitations=['Equality to upstream is not independent semantic validation of free space.',
                     'Full upstream inventory is not leak-screened or admitted for training.',
                     'Multiple point targets for a query may represent valid alternative positions.',
                     'No live dataset, labels, teachers, training plan or scorer was changed.'],
        training_changed=False, full_upstream_admitted=False, created=time.time())
    output=C/'vacancy_upstream_fidelity_20261029.json'
    assert not output.exists(), 'Do not overwrite an audit receipt'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('evidence','mismatches')},indent=2))


if __name__ == '__main__':
    main()
