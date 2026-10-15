#!/usr/bin/env python3
"""Validate ms-swift compliance of every source in a registry.

For each source: file existence, JSON validity, messages structure,
<image>/<video> placeholder-vs-media-count alignment (the #1 ms-swift crash),
media path absoluteness + existence (after image_root/video_root resolution),
objects/grounding sanity, and format detection.

Streams files and samples; safe on multi-GB jsonl.

Usage:
    python trains/validate_registry.py --registry data/sources_registry_fullmix.yaml \
        --deep 500 --media-sample 30 --out trains/validate_report.json
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import yaml

IMAGE_TAG = re.compile(r'<image>')
VIDEO_TAG = re.compile(r'<video>')
AUDIO_TAG = re.compile(r'<audio>')


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def resolve(p, root):
    if not p or os.path.isabs(str(p)) or not root:
        return str(p)
    return os.path.normpath(os.path.join(root, str(p)))


def content_text(msgs):
    parts = []
    for m in msgs:
        c = m.get('content')
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):  # multimodal content list form
            for seg in c:
                if isinstance(seg, dict) and seg.get('type') == 'text':
                    parts.append(seg.get('text', ''))
    return '\n'.join(parts)


def check_record(rec, src, deep_media_paths):
    """Return list of issue codes for one record."""
    issues = []
    msgs = rec.get('messages')
    if not isinstance(msgs, list) or len(msgs) < 1:
        # maybe sharegpt/alpaca form
        if any(k in rec for k in ('conversation', 'conversations')):
            return ['fmt_sharegpt_not_messages']
        if any(k in rec for k in ('instruction', 'query', 'prompt', 'question')):
            return ['fmt_alpaca_or_qr_not_messages']
        return ['no_messages']

    roles = [m.get('role') for m in msgs]
    if 'assistant' not in roles:
        issues.append('no_assistant')
    text = content_text(msgs)

    # placeholder vs media alignment
    n_img_tag = len(IMAGE_TAG.findall(text))
    n_vid_tag = len(VIDEO_TAG.findall(text))
    images = as_list(rec.get('images'))
    videos = as_list(rec.get('videos'))
    # flatten frame-list videos (each entry may itself be a list of frames)
    n_images = len(images)
    n_videos = len(videos)
    if n_img_tag != n_images:
        issues.append(f'img_tag_mismatch({n_img_tag}tag/{n_images}img)')
    if n_vid_tag != n_videos:
        issues.append(f'vid_tag_mismatch({n_vid_tag}tag/{n_videos}vid)')

    # objects/grounding sanity
    objs = rec.get('objects')
    if objs is not None and isinstance(objs, dict):
        ref = objs.get('ref', [])
        bbox = objs.get('bbox', [])
        n_refobj = text.count('<ref-object>')
        n_bboxtag = text.count('<bbox>')
        if isinstance(ref, list) and n_refobj and len(ref) != n_refobj:
            issues.append(f'ref_mismatch({n_refobj}tag/{len(ref)}ref)')
        if isinstance(bbox, list) and n_bboxtag and len(bbox) != n_bboxtag:
            issues.append(f'bbox_mismatch({n_bboxtag}tag/{len(bbox)}bbox)')

    # collect media paths (resolved) for later existence sampling
    for p in images:
        if p:
            deep_media_paths.append(resolve(p, src.get('image_root')))
    for v in videos:
        if isinstance(v, list):
            for fr in v:
                if fr:
                    deep_media_paths.append(resolve(fr, src.get('video_root')))
        elif v:
            deep_media_paths.append(resolve(v, src.get('video_root')))
    return issues


def validate_source(src, deep_n, media_sample):
    name = src.get('name', '?')
    path = src.get('path')
    res = {'name': name, 'path': path, 'status': 'ok', 'issues': Counter(),
           'total_lines': 0, 'checked': 0, 'bad_json': 0,
           'media_checked': 0, 'media_missing': 0, 'sample_missing': []}
    if not path or not os.path.isfile(path):
        res['status'] = 'FILE_NOT_FOUND'
        return res

    media_paths = []
    with open(path) as f:
        for i, line in enumerate(f):
            res['total_lines'] += 1
            if i < deep_n:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    res['bad_json'] += 1
                    continue
                res['checked'] += 1
                for code in check_record(rec, src, media_paths):
                    res['issues'][code] += 1
    # sample media existence
    if media_paths:
        step = max(1, len(media_paths) // media_sample)
        for p in media_paths[::step][:media_sample]:
            res['media_checked'] += 1
            if not os.path.isfile(p):
                res['media_missing'] += 1
                if len(res['sample_missing']) < 3:
                    res['sample_missing'].append(p)
    res['issues'] = dict(res['issues'])
    if res['issues'] or res['media_missing'] or res['bad_json']:
        res['status'] = 'WARN'
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', required=True)
    ap.add_argument('--deep', type=int, default=500, help='records per source for deep checks')
    ap.add_argument('--media-sample', type=int, default=30)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    reg = yaml.safe_load(open(args.registry))
    sources = reg.get('sources', []) if isinstance(reg, dict) else reg
    print(f'Registry: {args.registry}  sources={len(sources)}\n')

    results = []
    for src in sources:
        if not src.get('name'):
            continue
        r = validate_source(src, args.deep, args.media_sample)
        results.append(r)
        flag = {'ok': 'OK  ', 'WARN': 'WARN', 'FILE_NOT_FOUND': 'MISS'}[r['status']]
        issue_str = ''
        if r['issues']:
            issue_str = '  ' + '; '.join(f'{k}={v}' for k, v in r['issues'].items())
        media_str = ''
        if r['media_checked']:
            media_str = f"  media={r['media_checked']-r['media_missing']}/{r['media_checked']}"
        print(f"[{flag}] {r['name']:32s} lines={r['total_lines']:>8} "
              f"json_ok={r['checked']-r['bad_json']}/{r['checked']}{media_str}{issue_str}")
        if r['sample_missing']:
            print(f"        missing e.g.: {r['sample_missing'][0]}")

    # summary
    n_miss = sum(1 for r in results if r['status'] == 'FILE_NOT_FOUND')
    n_warn = sum(1 for r in results if r['status'] == 'WARN')
    n_ok = sum(1 for r in results if r['status'] == 'ok')
    print(f"\n{'='*70}")
    print(f"  OK={n_ok}  WARN={n_warn}  FILE_NOT_FOUND={n_miss}  total={len(results)}")
    # aggregate issue codes
    agg = Counter()
    for r in results:
        for k, v in r['issues'].items():
            agg[k.split('(')[0]] += v
    if agg:
        print('  issue codes:', dict(agg))
    print('='*70)

    if args.out:
        json.dump({'registry': args.registry, 'results': results}, open(args.out, 'w'),
                  indent=2, ensure_ascii=False)
        print(f'  report -> {args.out}')


if __name__ == '__main__':
    main()
