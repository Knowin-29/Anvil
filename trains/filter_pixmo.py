#!/usr/bin/env python3
"""Filter pixmo_points: remove samples with anomalous/invalid points."""
import json, re, os

INPUT = "${DATA_ROOT}/datasets/dataset_jsons_qwen3_swift/pixmo_points/point_points_both_to_swift_onlypoint.jsonl"
OUTPUT = "${GSA_FORGE_ROOT}/trains/pixmo_filtered.jsonl"
MAX_POINTS = 15
total = 0; kept = 0; too_many = 0; out_of_bound = 0; dup_points = 0

with open(INPUT) as fin, open(OUTPUT, 'w') as fout:
    for line in fin:
        total += 1
        d = json.loads(line)
        msgs = d.get('messages', [])
        asst = next((m['content'] for m in msgs if m.get('role') == 'assistant'), '')
        
        # Extract points from response
        points = re.findall(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', asst)
        
        # Rule 1: too many points
        if len(points) > MAX_POINTS:
            too_many += 1
            continue
        
        # Rule 2: out of bounds (Qwen 1000-scale)
        bad = False
        for px, py in points:
            x, y = float(px), float(py)
            if x < 0 or x > 1000 or y < 0 or y > 1000:
                bad = True
                break
        if bad:
            out_of_bound += 1
            continue
        
        # Rule 3: duplicate points in same sample
        if len(points) != len(set(points)):
            dup_points += 1
            continue
        
        fout.write(line)
        kept += 1

print(f"pixmo filter: total={total} kept={kept} ({kept/total*100:.1f}%)")
print(f"  too_many_points(>{MAX_POINTS}): {too_many}")
print(f"  out_of_bounds: {out_of_bound}")
print(f"  duplicate_points: {dup_points}")
