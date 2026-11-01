"""Extract original grounding/depth pairs only on already-admitted train images."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

C = Path(__file__).resolve().parent
OLD = C / 'vacancy_expansion_candidate_20261029'
OUT = C / 'refspatial_grounding_20261101'
SOURCE = Path('/Knowin/foundation/data/datasets/originDatasets_transfer_from_hpc3/refspatial/3D')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    temporary = Path(path).with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    OUT.mkdir(exist_ok=False)
    started = time.time()
    def status(stage, **fields):
        save(OUT / 'status.json', dict(stage=stage, pid=os.getpid(), updated=time.time(), **fields))
    old = json.loads((OLD / 'manifest.json').read_text())
    for name in ['train.jsonl', 'policy.json', 'image_fingerprints.jsonl', 'exclusions.jsonl']:
        assert sha(OLD / name) == old['files'][name]['sha256'], name
    allowed = {}
    for line in (OLD / 'train.jsonl').open():
        row = json.loads(line)
        assert row['gsa_meta']['split'] == 'train' and len(row['images']) == 1
        binding = row['gsa_meta']['binding']
        path = row['images'][0]
        previous = allowed.setdefault(path, binding)
        for key in ['sha256', 'decoded_rgb_sha256', 'width', 'height', 'phash64', 'dhash64']:
            assert previous[key] == binding[key]
    assert len(allowed) == old['candidate_unique_images'] == 35736
    inputs = {
        'grounding': SOURCE / 'reasoning_template_qa_filtered_3D_groupLength.jsonl',
        'spatialqa': SOURCE / 'choice_qa_filtered_3D_groupLength.jsonl',
    }
    bindings = {str(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in inputs.values()}
    selected = {kind: defaultdict(dict) for kind in inputs}
    counts = Counter()
    depth = re.compile(r'\b(?:closer|farther|further|nearest|farthest|behind|distance)\b|in front of', re.I)
    ambiguous = re.compile(r'\b(?:former|latter|they|them)\b', re.I)
    for kind, path in inputs.items():
        status('extracting_original_pairs', source=kind, counts=dict(counts))
        for ordinal, line in enumerate(path.open()):
            row = json.loads(line)
            counts[kind + '_source_groups'] += 1
            relative = Path(row['image'])
            assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'image'
            image = str(SOURCE / relative)
            if image not in allowed:
                counts[kind + '_unadmitted_image_groups'] += 1
                continue
            fp = allowed[image]
            assert [row['width'], row['height']] == [fp['width'], fp['height']]
            turns = row['conversations']
            assert len(turns) % 2 == 0
            for pair in range(0, len(turns), 2):
                question, answer = turns[pair:pair + 2]
                assert question['from'] == 'human' and answer['from'] == 'gpt'
                question, answer = question['value'], answer['value']
                assert isinstance(question, str) and isinstance(answer, str)
                if question.startswith('<image>'):
                    question = question.removeprefix('<image>').lstrip('\n')
                assert '<image>' not in question and '<image>' not in answer
                # Each selected turn is self-contained; retain its exact original text.
                if kind == 'grounding':
                    matched = re.fullmatch(r'\s*<point>(\[.*\])</point>\s*', answer)
                    if not matched or '<point>' in question or not re.search(r'\b(?:locate|select|point|where|find)\b', question, re.I):
                        continue
                    points = json.loads(matched.group(1))
                    if not points or not all(isinstance(p, list) and len(p) == 2 and
                            all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in p) for p in points):
                        counts['invalid_point_targets'] += 1
                        continue
                    semantic_key = digest(points)
                else:
                    if not depth.search(question) or ambiguous.search(question):
                        continue
                    if not all(tag in question for tag in ('(A)', '(B)')) or not re.match(r'^\s*\([AB]\)(?:\s|$|[.,])', answer):
                        continue
                    semantic_key = digest([question, answer])
                counts[kind + '_eligible_pairs'] += 1
                uid = 'refspatial_' + kind + ':' + digest([image, question, answer])
                item = dict(uid=uid, question=question, answer=answer, image=image,
                            source_id=str(row['id']), source_group_ordinal=ordinal,
                            source_pair_index=pair // 2, source_record_sha256=digest(row),
                            source_file=str(path), source_file_sha256=bindings[str(path)]['sha256'])
                candidates = selected[kind][image]
                if semantic_key not in candidates or uid < candidates[semantic_key]['uid']:
                    candidates[semantic_key] = item
                # Bounded, deterministic source-only selection, independent of evaluation scores.
                if len(candidates) > 4:
                    del candidates[max(candidates, key=lambda key: candidates[key]['uid'])]
            if ordinal % 20000 == 0:
                status('extracting_original_pairs', source=kind, counts=dict(counts))
    images = sorted(set().union(*(set(groups) for groups in selected.values())))
    status('verifying_current_train_images', images=len(images), counts=dict(counts))
    def verify_image(path):
        assert sha(path) == allowed[path]['sha256'], 'Admitted train image changed'
        return path
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, _ in enumerate(pool.map(verify_image, images), 1):
            if i % 2000 == 0:
                status('verifying_current_train_images', images=len(images), complete=i)
    output_files = {}
    all_uids = set()
    for kind, groups in selected.items():
        output = OUT / (kind + '.jsonl')
        emitted = 0
        with output.open('x') as stream:
            for image in sorted(groups):
                fp = allowed[image]
                for item in sorted(groups[image].values(), key=lambda row: row['uid']):
                    uid = item.pop('uid')
                    assert uid not in all_uids
                    all_uids.add(uid)
                    binding = {key: fp[key] for key in ['sha256', 'decoded_rgb_sha256', 'width', 'height', 'phash64', 'dhash64']}
                    binding.update(source_dataset='refspatial_' + kind, image_path=image,
                                   source_split='subset_of_previously_admitted_image_and_scan_partition',
                                   source_id=item['source_id'], source_group_ordinal=item['source_group_ordinal'],
                                   source_pair_index=item['source_pair_index'], source_record_sha256=item['source_record_sha256'],
                                   source_file=item['source_file'], source_file_sha256=item['source_file_sha256'])
                    row = dict(id=uid, canonical_uid=uid, gsa_form='point' if kind == 'grounding' else 'QA',
                               gsa_family='G' if kind == 'grounding' else 'QA',
                               gsa_meta=dict(split='train', binding=binding), images=[image],
                               messages=[dict(role='user', content='<image>\n' + item['question']),
                                         dict(role='assistant', content=item['answer'])])
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    emitted += 1
        assert emitted >= 1024, 'Insufficient eligible source pairs'
        output_files[kind] = dict(path=str(output), bytes=output.stat().st_size, sha256=sha(output),
                                  records=emitted, unique_images=len(groups))
    assert {str(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in inputs.values()} == bindings
    report = dict(schema='anvil-existing-train-image-refspatial-supplement-v1', state='verified_train_only',
                  files=output_files, counts=dict(counts), source_files=bindings,
                  inherited_partition_manifest=dict(path=str(OLD / 'manifest.json'), sha256=sha(OLD / 'manifest.json')),
                  admitted_train_images=len(allowed), used_train_images=len(images), new_images_vs_admitted_pool=0,
                  max_pairs_per_image_per_source=4, all_current_image_hashes_verified=True,
                  source_targets_regenerated=False, evaluation_answers_consumed=False,
                  selection_uses_evaluation_predictions=False, actual_training_exposures=0,
                  limitations=['These are the existing source annotations; semantic correctness of every answer is not independently established.',
                               'Original 3D source has no claimed official split; the pre-existing conservative campaign image/scan partition is inherited.',
                               'Independent pairs retain source questions and answers, with the image prefix restored for each selected turn.',
                               'Grounding keeps at most one paraphrase per distinct coordinate target per image.'],
                  builder_sha256=sha(__file__), completed=time.time(), elapsed=time.time() - started)
    save(OUT / 'manifest.json', report)
    status('verified_train_only', files=output_files)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            save(OUT / 'failure.json', dict(error=repr(exc), time=time.time()))
        raise
