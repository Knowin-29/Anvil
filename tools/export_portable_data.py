#!/usr/bin/env python3
"""Package complete local native exports without changing image bytes or targets.

This builds a local release candidate, not redistribution permission. Original
images are byte-preserved in deterministic TAR shards; only container paths and
filesystem metadata change. Known private terms in image metadata or retained
record text stop the export rather than altering training/evaluation content.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import time
import zipfile

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.native_metrics import CocoPointReference
from tools.run_native_evaluation import validate_row as validate_evaluation
from tools.train_native_specialist import save, sha
from trains.flywheel.native_training import FAMILIES, SHA256, canonical, digest, require, strict_json, validate_target, validate_training_record

VERSION = 'anvil-portable-native-data-v1'
SOURCES = {
    'coco_train2017': {'url': 'https://cocodataset.org/#termsofuse', 'declared_license': 'annotations cc-by-4.0; images retain individual upstream licenses'},
    'pixmo_points': {'url': 'https://huggingface.co/datasets/allenai/pixmo-points', 'declared_license': 'odc-by-1.0; linked image rights separate'},
    'CLEVR': {'url': 'https://cs.stanford.edu/people/jcjohns/clevr/', 'declared_license': 'cc-by-4.0'},
    'PRISM': {'url': 'https://huggingface.co/datasets/allenai/PRISM', 'declared_license': 'mit; underlying asset terms require review'},
    'ALFRED': {'url': 'https://github.com/askforalfred/alfred', 'declared_license': 'repository mit; rendered asset terms require review'},
    'ShareRobot': {'url': 'https://huggingface.co/datasets/BAAI/ShareRobot', 'declared_license': 'not established for this aggregate'},
    'ShareRobot-Bench': {'url': 'https://huggingface.co/datasets/BAAI/ShareRobot-Bench', 'declared_license': 'apache-2.0; component rights require review'},
}


def audit_coco_licenses(package, evaluation, source_manifest, output):
    """Join selected image bytes to ALL official license IDs; not legal clearance.

    No pixels, target records or immutable package manifests are changed. The
    image index is already part of the separately verified portable package.
    The annotation archive alone does not supply complete creator attribution.
    """
    package, evaluation, source_manifest, output = map(Path, (package, evaluation, source_manifest, output))
    require(not output.exists(), 'Use a new license inventory directory')
    pm, em = package/'manifest.json', evaluation/'manifest.json'
    m, e, source = (strict_json(p.read_bytes()) for p in (pm, em, source_manifest))
    bindings = {str(p): sha(p) for p in (pm, em, source_manifest)}
    require(m['state'] == 'complete_local_portable_candidate'
            and e['state'] == 'all_selected_native_evaluation_rows_exported'
            and sha(em) == m['parent_manifest_sha256']['benchmark'], 'Unbound package/evaluation manifests')
    coco_inputs = [i for i in e['source_inputs'] if i['source'] == 'coco_train2017']
    require(coco_inputs and source['source'] == 'coco_train2017'
            and all(i['manifest_sha256'] == sha(source_manifest) for i in coco_inputs), 'Unbound COCO source manifest')
    index, source_index = package/'image_index.jsonl', source_manifest.parent/'images.jsonl'
    annotation = source['input_binding']['annotations']
    for p, expected in ((index, m['file_inventory']['image_index.jsonl']['sha256']),
                        (source_index, source['files']['images.jsonl']['sha256']),
                        (Path(annotation['path']), annotation['sha256'])):
        require(sha(p) == expected, 'License inventory input bytes changed')
        bindings[str(p)] = expected
    selected, image_counts, total = {}, Counter(), 0
    with index.open(encoding='utf-8') as stream:
        for raw in stream:
            row = strict_json(raw); total += 1
            image_counts.update(row['sources'])
            if 'coco_train2017' in row['sources']:
                require(row['sha256'] not in selected, 'Duplicate selected image bytes')
                selected[row['sha256']] = row
    require(total == m['image_files'] and selected, 'Incomplete image index or no COCO images')
    with zipfile.ZipFile(annotation['path']) as archive:
        with archive.open('annotations/instances_train2017.json') as stream:
            official = json.load(stream)
    licenses = {i['id']: i for i in official['licenses']}
    images = {str(i['id']): i for i in official['images']}
    require(len(licenses) == len(official['licenses']) and len(images) == len(official['images']),
            'Duplicate official image/license identities')
    matches = {key: {} for key in selected}
    with source_index.open(encoding='utf-8') as stream:
        for raw in stream:
            row = strict_json(raw); key = row['sha256']
            if key not in selected: continue
            image_id = str(row['source_image_id']); image = images[image_id]
            require(row['source'] == 'coco_train2017'
                    and row['decoded_rgb_sha256'] == selected[key]['canonical_rgb_sha256']
                    and [image['width'], image['height']] == selected[key]['image_size_wh'], 'Image identity/dimensions disagree')
            require(image_id not in matches[key] and image['license'] in licenses, 'Duplicate or unknown official license mapping')
            matches[key][image_id] = {'image_id': image['id'], 'license_id': image['license'],
                'flickr_url': image.get('flickr_url'), 'coco_url': image.get('coco_url')}
    require(all(matches.values()), 'Selected COCO image has no bound license mapping')
    output.mkdir(parents=True, exist_ok=False); target = output/'coco_image_licenses.jsonl'
    counts = Counter()
    with target.open('x', encoding='utf-8') as stream:
        for key, matched in sorted(matches.items()):
            ids = sorted({i['license_id'] for i in matched.values()}); counts[','.join(map(str, ids))] += 1
            stream.write(canonical({'image_sha256': key, 'canonical_rgb_sha256': selected[key]['canonical_rgb_sha256'],
                'split': selected[key]['split'], 'official_images': [matched[k] for k in sorted(matched, key=int)],
                'license_ids': ids}) + '\n')
    require(all(sha(p) == expected for p, expected in bindings.items()), 'License inventory inputs changed during join')
    result = {'schema': 'anvil-coco-image-license-inventory-v1', 'state': 'complete_bound_image_license_join',
        'package_manifest_sha256': sha(pm), 'evaluation_manifest_sha256': sha(em),
        'source_manifest_sha256': sha(source_manifest), 'image_index_sha256': sha(index),
        'source_image_index_sha256': sha(source_index), 'annotation_archive_sha256': annotation['sha256'],
        'tool_sha256': sha(Path(__file__)), 'image_files_by_source': dict(sorted(image_counts.items())),
        'coco_image_files': len(selected), 'coco_multi_id_images': sum(len(v) > 1 for v in matches.values()),
        'coco_license_sets': dict(sorted(counts.items())), 'license_definitions': licenses,
        'files': {target.name: {'sha256': sha(target), 'records': len(selected)}},
        'creator_attribution_complete': False, 'redistribution_review_complete': False,
        'public_release_eligible': False, 'images_or_targets_changed': False,
        'scope': 'Official license IDs and source URLs only; retain every ambiguous identity, not a permission or attribution-completeness decision.'}
    save(output/'summary.json', result)
    return result


def private_pattern(terms, *, generic_paths=True):
    require(all(isinstance(t, str) and len(t.strip()) >= 3 for t in terms), 'Private terms must be explicit nonempty names')
    # Specific owner identities are provided privately, never embedded in source.
    expressions = [rb'gh[pousr]_[A-Za-z0-9]{20,}', rb'hf_[A-Za-z0-9]{20,}',
                   rb'-----BEGIN [A-Z ]*PRIVATE KEY-----']
    if generic_paths: expressions.append(rb'/(?:root|home|mnt|workspace)/')
    expressions.extend((rb'(?<![A-Za-z0-9])' + re.escape(t.encode()) + rb'(?![A-Za-z0-9])')
                       if t.isalnum() else re.escape(t.encode()) for t in terms)
    return re.compile(b'|'.join(expressions), re.I)


def check_image_metadata(original, payload, pattern):
    # Source images may retain their upstream renderer's paths (e.g. CLEVR's
    # public Blender File field). Explicit owner/server terms still apply, but
    # a generic /mnt/ prefix alone does not identify this project's private data.
    # Never search short names in compressed pixel bytes: those bytes are not
    # text and can contain coincidental matches. Pillow decodes PNG text fields,
    # including compressed zTXt/iTXt, and exposes JPEG/EXIF metadata separately.
    metadata = [original.info, dict(original.getexif())]
    exif = original.getexif()
    for tag in (34665, 34853, 40965):
        if tag in exif: metadata.append(exif.get_ifd(tag))
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items(): inspect(key); inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value: inspect(item)
        elif isinstance(value, (str, bytes)):
            raw = value.encode() if isinstance(value, str) else value
            require(pattern.search(raw) is None, 'Private term in source image metadata; image left unchanged')
    inspect(metadata)
    # Long credential/private-key signatures remain prohibited anywhere,
    # including unparsed metadata and bytes appended after the image payload.
    require(private_pattern([], generic_paths=False).search(payload) is None,
            'Credential signature in source image bytes; image left unchanged')


def image_path(image_sha):
    require(isinstance(image_sha, str) and SHA256.fullmatch(image_sha), 'Image byte hash absent')
    # Pillow identifies format from bytes; an opaque .img suffix avoids guessing
    # and never requires transcoding a benchmark image.
    return f'images/{image_sha[:2]}/{image_sha}.img'


def convert(row, split, form, selection_sha, pattern):
    if split == 'train':
        validate_training_record(row)
        meta = row['gsa_meta']; binding = meta['binding']
        require(row['gsa_native_form'] == form and binding['selection_sha256'] == selection_sha,
                'Training export form/selection changed')
        variant = row['gsa_native_variant']; original_content_sha = meta['training_content_sha256']
        group = binding.get('selected_group_id', binding['group_id'])
    else:
        validate_evaluation(row, form, selection_sha)
        binding = row['evaluation_meta']; variant = row['native_variant']
        original_content_sha = binding['evaluation_content_sha256']; group = binding['selected_group_id']
    require(binding['source'] in SOURCES, 'Unregistered source; add verified source attribution first')
    original_image = row['images'][0]; encoded = binding['image_sha256']; decoded = binding['decoded_rgb_sha256']
    require(SHA256.fullmatch(decoded), 'Canonical image identity absent')
    result = {'schema': VERSION, 'id': row['id'], 'split': split, 'native_form': form,
        'native_variant': variant, 'family': FAMILIES.get(form, 'QA'),
        'messages': row['messages'], 'images': [image_path(encoded)],
        'provenance': {'source': binding['source'], 'source_group_id': group,
            'selection_sha256': selection_sha, 'source_sha256': binding['source_sha256'],
            'source_line_sha256': binding['source_line_sha256'], 'original_content_sha256': original_content_sha,
            'image_sha256': encoded, 'canonical_rgb_sha256': decoded,
            'human_reviewed': False, 'strong_model_verified': False, 'public_release_eligible': False}}
    if form == 'QA':
        result['source_reference'] = {'type': 'clevr_official_question', 'question_index': meta['official_question_id'],
                                      'questions_sha256': meta['official_questions_sha256']}
    require(pattern.search(canonical(result).encode()) is None,
            'Known private text in retained record; no silent redaction of scientific content')
    return result, {'path': original_image, 'encoded': encoded, 'decoded': decoded, 'split': split, 'source': binding['source']}


def coco_reference(row, loader):
    """Keep an upstream annotation identity, not private source-file offsets."""
    meta = row['evaluation_meta']; key = meta['source_sha256']; item = loader.inputs[key]
    if key not in loader.handles: loader.handles[key] = Path(item['path']).open('rb', buffering=8 << 20)
    stream = loader.handles[key]; stream.seek(meta['source_offset']); raw = stream.read(meta['source_size'])
    require(hashlib.sha256(raw).hexdigest() == meta['source_line_sha256'], 'COCO reference row changed')
    source = strict_json(raw); metadata = source['anvil_meta']; ids = metadata['source_annotation_ids']
    require(source['id'] == row['id'] and digest(metadata) == meta['source_metadata_sha256'] and len(ids) == 1,
            'COCO reference identity mismatch')
    annotation = loader.annotations[ids[0]]
    require(str(annotation['image_id']) == metadata['source_image_id'], 'COCO reference image mismatch')
    image = loader.images[annotation['image_id']]
    require([image['width'], image['height']] == row['image_size_wh'], 'COCO reference dimensions changed')
    return {'type': 'coco_instance_interior', 'annotation_id': ids[0], 'image_id': annotation['image_id'],
            'annotation_archive_sha256': loader.annotation_binding['sha256'],
            'annotation_archive_url': 'https://images.cocodataset.org/annotations/annotations_trainval2017.zip'}


def read_image(item, pattern):
    encoded, info = item
    payload = None
    for path in sorted(info['paths']):
        raw = Path(path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == encoded, 'Source image byte identity changed')
        if payload is None: payload = raw
    require(payload is not None, 'Source image payload absent')
    with Image.open(io.BytesIO(payload)) as original:
        check_image_metadata(original, payload, pattern)
        size = list(original.size); image_format = original.format
        rgb = ImageOps.exif_transpose(original).convert('RGB')
        fingerprint = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode() + rgb.tobytes()).hexdigest()
    require(fingerprint == info['decoded'], 'Source decoded image identity changed')
    record = {'path': image_path(encoded), 'sha256': encoded, 'bytes': len(payload),
        'canonical_rgb_sha256': fingerprint, 'image_size_wh': size, 'format': image_format,
        'split': info['split'], 'sources': sorted(info['sources'])}
    return record, payload


def validate_portable(row, pattern):
    require(row.get('schema') == VERSION and row.get('split') in {'train', 'benchmark'}, 'Invalid portable schema/split')
    form = row['native_form']; require(form in FAMILIES or form == 'QA' and row['split'] == 'train', 'Invalid portable form')
    require(row['family'] == FAMILIES.get(form, 'QA') and isinstance(row['id'], str) and row['id'], 'Portable identity/family absent')
    provenance = row['provenance']; require(provenance['source'] in SOURCES, 'Unknown portable source')
    for key in ('selection_sha256', 'source_sha256', 'source_line_sha256', 'original_content_sha256', 'image_sha256', 'canonical_rgb_sha256'):
        require(SHA256.fullmatch(provenance[key]) is not None, 'Invalid portable provenance hash')
    require(row['images'] == [image_path(provenance['image_sha256'])], 'Unsafe portable image path')
    require([m['role'] for m in row['messages']] == ['user', 'assistant']
            and all(isinstance(m['content'], str) and m['content'].strip() for m in row['messages']), 'Portable messages malformed')
    if form != 'QA': validate_target(form, row['native_variant'], strict_json(row['messages'][1]['content']), row['messages'][0]['content'])
    require(pattern.search(canonical(row).encode()) is None, 'Known private text in portable record')


def verify(root, terms=()):
    """Read all shards independently; no extraction, model inference or upload."""
    root = root.resolve(); require(not (root / 'manifest.json').is_symlink(), 'Manifest must not be a symlink')
    m = strict_json((root / 'manifest.json').read_bytes()); pattern = private_pattern(terms)
    image_pattern = private_pattern(terms, generic_paths=False)
    require(m['schema'] == VERSION and m['state'] == 'complete_local_portable_candidate', 'Incomplete portable candidate')
    expected = set(m['file_inventory']) | {'manifest.json'}
    require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == expected, 'Unexpected or missing package file')
    for name, info in m['file_inventory'].items():
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root)
                and sha(path) == info['sha256'] and path.stat().st_size == info['bytes'], 'Package file identity mismatch')
    require(pattern.search(canonical(m).encode()) is None, 'Private term in package manifest')
    images = {}; decoded = {}; media_seen = set()
    with (root / 'image_index.jsonl').open() as stream:
        for raw in stream:
            row = strict_json(raw); require(row['path'] not in images and row['path'] == image_path(row['sha256']), 'Duplicate/unsafe image identity')
            require(pattern.search(canonical(row).encode()) is None, 'Private term in image index')
            images[row['path']] = row
            require(decoded.setdefault(row['canonical_rgb_sha256'], row['split']) == row['split'], 'Decoded image crosses splits')
    require(len(images) == m['image_files'] and len(decoded) == m['unique_images'], 'Image inventory counts differ')
    for name, info in m['file_inventory'].items():
        if not name.endswith('.tar'): continue
        count = 0
        with tarfile.open(root / name, 'r:') as archive:
            for member in archive:
                require(member.isfile() and member.name in images and member.name not in media_seen,
                        'Unexpected/duplicate media archive member')
                reference = images[member.name]; require(reference['archive'] == name, 'Image archive binding differs')
                require(member.uid == member.gid == member.mtime == 0 and member.uname == member.gname == '', 'Nonneutral archive metadata')
                payload = archive.extractfile(member).read()
                require(hashlib.sha256(payload).hexdigest() == reference['sha256'] and len(payload) == reference['bytes'], 'Image payload differs')
                with Image.open(io.BytesIO(payload)) as original:
                    check_image_metadata(original, payload, image_pattern)
                    require(list(original.size) == reference['image_size_wh'], 'Image dimensions differ')
                    rgb = ImageOps.exif_transpose(original).convert('RGB')
                    fp = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode() + rgb.tobytes()).hexdigest()
                require(fp == reference['canonical_rgb_sha256'], 'Packed decoded pixels differ')
                media_seen.add(member.name); count += 1
        require(count == info['images'], 'Media shard count mismatch')
    require(media_seen == set(images), 'Unpacked image reference')
    seen = set(); counts = Counter(); used_images = set()
    for name, info in m['file_inventory'].items():
        if not name.endswith('.jsonl.gz'): continue
        count = 0
        with gzip.open(root / name, 'rb') as stream:
            for raw in stream:
                row = strict_json(raw); validate_portable(row, pattern)
                require(row['id'] not in seen and row['split'] == info['split'] and row['native_form'] == info['form'], 'Record identity/scope mismatch')
                seen.add(row['id']); image = images[row['images'][0]]; p = row['provenance']
                require(image['split'] == row['split'] and image['canonical_rgb_sha256'] == p['canonical_rgb_sha256']
                        and p['selection_sha256'] == m['selection_sha256'] and p['source'] in image['sources'], 'Record image/provenance mismatch')
                used_images.add(row['images'][0]); count += 1; counts[row['split'] + '/' + row['native_form']] += 1
        require(count == info['records'], 'Record shard count mismatch')
    require(len(seen) == m['records'] and dict(counts) == m['by_split_form'] and used_images == set(images), 'Full package conservation failed')
    return {'state': 'all_portable_bytes_records_images_verified', 'records': len(seen),
            'unique_images': len(decoded), 'image_files': len(images), 'manifest_sha256': sha(root / 'manifest.json'),
            'public_release_eligible': False, 'uploaded': False}


def build(args):
    start = time.monotonic(); pattern = private_pattern(args.private_term)
    image_pattern = private_pattern(args.private_term, generic_paths=False)
    require(args.output.resolve() != args.private_work.resolve()
            and not args.private_work.resolve().is_relative_to(args.output.resolve())
            and not args.output.resolve().is_relative_to(args.private_work.resolve()), 'Private work and package must not overlap')
    require(not args.output.exists() and not args.private_work.exists(), 'Use new output and private work directories')
    require(args.rows_per_shard > 0 and args.images_per_shard > 0 and args.workers > 0, 'Invalid shard/worker counts')
    roots = {'train': args.training, 'benchmark': args.evaluation}
    manifests = {split: strict_json((root / 'manifest.json').read_bytes()) for split, root in roots.items()}
    expected_states = {'train': 'all_selected_training_rows_exported', 'benchmark': 'all_selected_native_evaluation_rows_exported'}
    for split, m in manifests.items():
        require(m['state'] == expected_states[split] and m['dropped_records'] == m['targets_repaired'] == 0, 'Incomplete parent export')
        expected_forms = set(FAMILIES) | ({'QA'} if split == 'train' else set())
        require(set(m['by_form']) == expected_forms and {v['form'] for v in m['files'].values()} == expected_forms,
                'Missing forms in complete parent export')
        require(sum(v['records'] for v in m['files'].values()) == m['records'], 'Parent record counts disagree')
    selection = manifests['train']['selection_sha256']
    require(selection == manifests['benchmark']['selection_sha256'], 'Training/evaluation selection mismatch')
    parent_hashes = {s: sha(r / 'manifest.json') for s, r in roots.items()}
    args.output.mkdir(parents=True); args.private_work.mkdir(parents=True)
    code_sha = sha(__file__)
    code_hashes = {name: sha(Path(__file__).parents[1] / name) for name in
                  ('tools/export_portable_data.py', 'tools/run_native_evaluation.py', 'tools/train_native_specialist.py',
                   'tools/arxiv_source_scale.py', 'tools/map_prism_native.py', 'eval/native_metrics.py', 'trains/flywheel/native_training.py')}
    counts = Counter(); source_counts = Counter(); images = {}; decoded_splits = {}; seen = set(); files = {}
    def progress(stage, **extra):
        value = {'stage': stage, 'records': sum(counts.values()), 'unique_images': len(decoded_splits),
                 'image_files': len(images), 'seconds': time.monotonic() - start, **extra}
        save(args.private_work / 'progress.json', value); print(json.dumps(value), flush=True)
    save(args.private_work / 'input_binding.json', {'training': str(args.training), 'evaluation': str(args.evaluation),
        'parent_manifest_sha256': parent_hashes, 'exporter_sha256': code_sha, 'code_sha256': code_hashes,
        'private_term_count': len(args.private_term)})
    try:
        loader = (CocoPointReference(manifests['benchmark']) if any(x['source'] == 'coco_train2017'
                  for x in manifests['benchmark']['source_inputs']) else None)
        for split, root in roots.items():
            for name, info in sorted(manifests[split]['files'].items()):
                require(name == f"{'train' if split == 'train' else 'eval'}_{info['form']}.jsonl", 'Unsafe or inconsistent source filename')
                require(sha(root / name) == info['sha256'], 'Parent source file changed')
                count = 0; form = info['form']; part = -1; stack = ExitStack(); shard_count = 0; shard_name = None
                def close_shard():
                    if shard_name is not None:
                        stack.close(); path = args.output / shard_name
                        files[shard_name] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                                            'records': shard_count, 'split': split, 'form': form}
                with (root / name).open('rb') as stream:
                    try:
                        for raw in stream:
                            row = strict_json(raw); portable, binding = convert(row, split, form, selection, pattern)
                            require(row['id'] not in seen, 'Duplicate record identity across exported splits/forms'); seen.add(row['id'])
                            if split == 'benchmark' and form == 'point' and binding['source'] == 'coco_train2017':
                                require(loader is not None, 'COCO source reference absent')
                                portable['evaluation_reference'] = coco_reference(row, loader)
                            image = images.setdefault(binding['encoded'], {'paths': set(), 'decoded': binding['decoded'],
                                'sources': set(), 'split': split})
                            require(image['decoded'] == binding['decoded'] and image['split'] == split, 'Conflicting image binding/split')
                            image['paths'].add(binding['path']); image['sources'].add(binding['source'])
                            prior_split = decoded_splits.setdefault(binding['decoded'], split)
                            require(prior_split == split, 'Decoded image crosses training/evaluation split')
                            if count % args.rows_per_shard == 0:
                                close_shard(); stack = ExitStack(); part += 1; shard_count = 0
                                shard_name = f'records/{split}/{form}-{part:05d}.jsonl.gz'
                                path = args.output / shard_name; path.parent.mkdir(parents=True, exist_ok=True)
                                destination = stack.enter_context(path.open('xb'))
                                out = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=destination, compresslevel=3, mtime=0))
                            out.write((canonical(portable) + '\n').encode()); count += 1; shard_count += 1
                            counts[(split, form)] += 1; source_counts[(split, binding['source'])] += 1
                            if sum(counts.values()) % 100000 == 0: progress('records')
                    finally: close_shard()
                require(count == info['records'] and sha(root / name) == info['sha256'], 'Parent row count/bytes changed during packing')
        if loader: loader.close()
        require(sum(counts.values()) == sum(m['records'] for m in manifests.values()), 'Record conservation failed')
        observed_split_images = Counter(decoded_splits.values())
        require(all(observed_split_images[s] == m['unique_images'] for s, m in manifests.items()), 'Image conservation failed')
        progress('records_complete_images_pending')
        with (args.output / 'image_index.jsonl').open('x') as index, ThreadPoolExecutor(max_workers=args.workers) as pool:
            ordered = sorted(images.items()); completed = 0
            for part, start_index in enumerate(range(0, len(ordered), args.images_per_shard)):
                name = f'media/images-{part:05d}.tar'; path = args.output / name; path.parent.mkdir(exist_ok=True)
                # Bound pending image payloads to 2*workers, not all 160K images.
                with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as archive:
                    subset = ordered[start_index:start_index + args.images_per_shard]
                    for offset in range(0, len(subset), args.workers * 2):
                        tasks = [pool.submit(read_image, item, image_pattern) for item in subset[offset:offset + args.workers * 2]]
                        for task in tasks:
                            record, payload = task.result(); member = tarfile.TarInfo(record['path'])
                            member.size = len(payload); member.mtime = 0; member.mode = 0o644
                            archive.addfile(member, io.BytesIO(payload)); record['archive'] = name
                            index.write(canonical(record) + '\n'); completed += 1
                files[name] = {'sha256': sha(path), 'bytes': path.stat().st_size, 'images': len(subset)}
                index.flush(); progress('images', images_packed=completed)
        index_path = args.output / 'image_index.jsonl'
        files['image_index.jsonl'] = {'sha256': sha(index_path), 'bytes': index_path.stat().st_size, 'images': len(images)}
        require(all(sha(root / 'manifest.json') == parent_hashes[s] for s, root in roots.items())
                and all(sha(Path(__file__).parents[1] / name) == value for name, value in code_hashes.items()),
                'Input manifest or exporter dependencies changed during build')
        result = {'schema': VERSION, 'state': 'complete_local_portable_candidate', 'selection_sha256': selection,
            'parent_manifest_sha256': parent_hashes, 'records': sum(counts.values()), 'unique_images': len(decoded_splits),
            'image_files': len(images), 'by_split': {s: sum(n for (sp, _), n in counts.items() if sp == s) for s in roots},
            'by_split_form': {f'{s}/{f}': n for (s, f), n in sorted(counts.items())},
            'by_split_source': {f'{s}/{source}': n for (s, source), n in sorted(source_counts.items())},
            'image_policy': 'Original encoded bytes and upstream metadata preserved; opaque relative paths; neutral archive ownership/timestamps; explicit private terms rejected in decoded metadata; credentials rejected anywhere',
            'messages_changed': 0, 'records_dropped': 0, 'file_inventory': files,
            'sources': {source: {**SOURCES[source], 'redistribution_review_complete': False}
                        for source in sorted({s for _, s in source_counts})},
            'public_release_eligible': False, 'uploaded': False, 'model_evaluation_complete': False,
            'exporter_sha256': code_sha, 'code_sha256': code_hashes, 'seconds': time.monotonic() - start}
        save(args.output / 'manifest.json', result); progress('complete'); return result
    except BaseException as error:
        save(args.private_work / 'failure.json', {'type': type(error).__name__, 'reason': str(error),
             'public_release_eligible': False, 'partial_candidate_preserved': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'evaluation', 'output', 'private-work'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--verify', type=Path, help='Independently read every record and image in an existing candidate')
    parser.add_argument('--audit-coco-licenses', type=Path, help='Join a portable package image index to bound official COCO license IDs')
    parser.add_argument('--coco-source-manifest', type=Path)
    parser.add_argument('--private-term', action='append', default=[])
    parser.add_argument('--rows-per-shard', type=int, default=100000)
    parser.add_argument('--images-per-shard', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.audit_coco_licenses:
        require(all((args.evaluation, args.coco_source_manifest, args.output))
                and not any((args.training, args.private_work, args.verify, args.private_term)),
                'License inventory needs evaluation, coco-source-manifest and new output only')
        print(json.dumps(audit_coco_licenses(args.audit_coco_licenses, args.evaluation,
                                            args.coco_source_manifest, args.output))); return
    require(args.coco_source_manifest is None, 'coco-source-manifest requires audit-coco-licenses')
    if args.verify:
        require(not any((args.training, args.evaluation, args.output, args.private_work)), 'Verify does not accept build paths')
        print(json.dumps(verify(args.verify, args.private_term))); return
    require(all((args.training, args.evaluation, args.output, args.private_work)), 'All four build paths are required')
    build(args)


if __name__ == '__main__': main()
