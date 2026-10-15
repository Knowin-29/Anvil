#!/usr/bin/env python3
"""Bootstrap an ms-swift native dataset_info.json from a GSA registry YAML.

Registration / usage separation:
  - Registration (this output): name -> absolute path (+ tags). Edit it to change.
  - Usage (training): `--custom_dataset_info gsa_datasets.json --dataset name#N`.

ms-swift contract (swift/dataset/register.py):
  - Each entry: {"dataset_name", "dataset_path", "tags"}. Extra keys would break
    DatasetMeta(**d_info), so ALL registry metadata (dimensions/license/scene/coord)
    is folded into `tags`, never top-level.
  - Duplicate dataset_name raises at registration -> we enforce uniqueness here.
  - ratio is NOT registered (it is a usage-time concern via #N).

Usage:
    python trains/build_dataset_info.py \
        --registry data/sources_registry_fullmix.yaml \
        --out data/gsa_datasets.json
"""
import argparse
import json
import os
import re
from collections import Counter

import yaml

NORM_RE = re.compile(r'0-?1000|0~1000|norm1000|千分位|归一化')
PIXEL_RE = re.compile(r'像素|pixel|绝对坐标')
SUPERSEDE_RE = re.compile(r'替代旧\s*([A-Za-z0-9_]+)|replaces?\s+([A-Za-z0-9_]+)')


def license_tags(lic):
    """'open (PRISM)' -> ['license:open', 'license_src:prism']; 'internal (verifier)' -> ...."""
    if not lic:
        return []
    m = re.match(r'\s*([A-Za-z_]+)\s*(?:\(([^)]*)\))?', lic)
    base = (m.group(1) if m else lic).strip().lower()
    tags = [f'license:{base}']
    if m and m.group(2):
        slug = re.sub(r'[^a-z0-9]+', '_', m.group(2).strip().lower()).strip('_')
        if slug:
            tags.append(f'license_src:{slug}')
    return tags


def coord_tag(notes):
    n = notes or ''
    if NORM_RE.search(n):
        return ['coord:norm1000']
    if PIXEL_RE.search(n):
        return ['coord:pixel']
    return []


def build_tags(src):
    tags = list(src.get('dimensions') or [])
    tags += license_tags(src.get('license'))
    if src.get('scene_type'):
        tags.append(f"scene:{src['scene_type']}")
    tags.append('modality:video' if src.get('video_root') else 'modality:image')
    tags += coord_tag(src.get('notes'))
    if src.get('general'):
        tags.append('general')
    if src.get('dedup_key'):
        tags.append(f"dedup:{src['dedup_key']}")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default='data/sources_registry_fullmix.yaml')
    ap.add_argument('--out', default='data/gsa_datasets.json')
    ap.add_argument('--report', default=None)
    args = ap.parse_args()

    reg = yaml.safe_load(open(args.registry))
    sources = reg.get('sources', []) if isinstance(reg, dict) else reg

    # detect supersede declarations across all notes (high-confidence signal)
    superseded = {}  # old_name -> declaring source name
    for s in sources:
        m = SUPERSEDE_RE.search(s.get('notes') or '')
        if m:
            old = m.group(1) or m.group(2)
            if old:
                superseded[old] = s.get('name')

    entries = []
    report = {'registry': args.registry, 'registered': 0, 'skipped_raw': [],
              'renamed': [], 'deprecated': [], 'missing_file': []}
    seen = {}

    for s in sources:
        name = s.get('name')
        path = s.get('path')
        if not name or not path:
            continue
        if s.get('format') == 'raw':
            report['skipped_raw'].append({'name': name, 'path': path})
            continue

        abspath = os.path.abspath(path)
        if not os.path.isfile(abspath):
            report['missing_file'].append({'name': name, 'path': abspath})

        ds_name = name
        if ds_name in seen:  # uniqueness guard (ms-swift would crash on dup)
            i = 2
            while f'{ds_name}_{i}' in seen:
                i += 1
            ds_name = f'{ds_name}_{i}'
            report['renamed'].append({'from': name, 'to': ds_name, 'path': abspath})
        seen[ds_name] = abspath

        tags = build_tags(s)
        if name in superseded:
            tags.append('status:deprecated')
            tags.append(f'superseded_by:{superseded[name]}')
            report['deprecated'].append({'name': name, 'by': superseded[name]})

        entries.append({'dataset_name': ds_name, 'dataset_path': abspath, 'tags': tags})

    report['registered'] = len(entries)

    json.dump(entries, open(args.out, 'w'), indent=2, ensure_ascii=False)
    report_path = args.report or (os.path.splitext(args.out)[0] + '.report.json')
    json.dump(report, open(report_path, 'w'), indent=2, ensure_ascii=False)

    print(f'registered: {len(entries)} datasets -> {args.out}')
    print(f'  skipped_raw={len(report["skipped_raw"])} renamed={len(report["renamed"])} '
          f'deprecated={len(report["deprecated"])} missing_file={len(report["missing_file"])}')
    if report['renamed']:
        print('  renamed:', report['renamed'])
    if report['missing_file']:
        print('  MISSING FILES:', [m['name'] for m in report['missing_file']])
    print(f'  report -> {report_path}')


if __name__ == '__main__':
    main()
