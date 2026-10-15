#!/usr/bin/env python3
"""Retrieve datasets from the ms-swift native registry by tag.

Registration lives in gsa_datasets.json (name -> path + tags). This is the
'retrieval' side: filter by tags, list names, count.

Examples:
    python trains/query_datasets.py --tag G_grounding
    python trains/query_datasets.py --tag S_spatial --tag license:open   # AND
    python trains/query_datasets.py --any G_grounding S_spatial          # OR
    python trains/query_datasets.py --exclude status:deprecated --names
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default='data/gsa_datasets.json')
    ap.add_argument('--tag', action='append', default=[], help='must have ALL of these (AND)')
    ap.add_argument('--any', nargs='+', default=[], help='must have AT LEAST ONE (OR)')
    ap.add_argument('--exclude', action='append', default=[], help='must NOT have any of these')
    ap.add_argument('--names', action='store_true', help='print only dataset names (one per line)')
    args = ap.parse_args()

    entries = json.load(open(args.registry))
    out = []
    for e in entries:
        tags = set(e.get('tags', []))
        if args.tag and not all(t in tags for t in args.tag):
            continue
        if args.any and not any(t in tags for t in args.any):
            continue
        if args.exclude and any(t in tags for t in args.exclude):
            continue
        out.append(e)

    if args.names:
        for e in out:
            print(e['dataset_name'])
        return

    tagfreq = Counter(t for e in out for t in e.get('tags', []))
    print(f"matched {len(out)}/{len(entries)} datasets")
    for e in out:
        print(f"  {e['dataset_name']:32s} {sorted(e.get('tags', []))}")
    print(f"\ntag frequency: {dict(tagfreq.most_common())}")


if __name__ == '__main__':
    main()
