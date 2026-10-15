#!/usr/bin/env python3
"""Print a CUDA_VISIBLE_DEVICES list of GPUs with enough free memory, right now.

Written because the node is shared and the free set moves: the sweep's first smoke
run reserved 5,6,7 at launch, two unrelated jobs took 6 and 7 during the ~2 min
model load, and every rank died with OOM after paying the load cost. Picking at
launch from measured free memory (and never killing anyone) is the only safe
protocol here.

`--exclude` defaults to the four cards the General Retention evaluation was pinned
to, so a stale allocation cannot be reclaimed by accident.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def query() -> list[tuple[int, int, int, int]]:
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in out.strip().splitlines():
        idx, total, used, util = (int(v.strip()) for v in line.split(","))
        rows.append((idx, total, used, util))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-mib", type=int, default=38000,
                    help="free memory a card must have to be usable")
    ap.add_argument("--max-gpus", type=int, default=3)
    ap.add_argument("--min-gpus", type=int, default=1)
    ap.add_argument("--exclude", default="0,1,2,3")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    excluded = {int(v) for v in a.exclude.split(",") if v.strip() != ""}
    free = []
    for idx, total, used, util in query():
        avail = total - used
        ok = idx not in excluded and avail >= a.need_mib
        if a.verbose:
            print(f"gpu{idx} total={total} used={used} free={avail} util={util}% "
                  f"{'USABLE' if ok else 'skip'}", file=sys.stderr)
        if ok:
            free.append((avail, idx))
    free.sort(reverse=True)
    picked = sorted(i for _, i in free[:a.max_gpus])
    if len(picked) < a.min_gpus:
        print(f"FATAL: only {len(picked)} GPU(s) with >={a.need_mib} MiB free "
              f"(need {a.min_gpus})", file=sys.stderr)
        return 1
    print(",".join(str(i) for i in picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
