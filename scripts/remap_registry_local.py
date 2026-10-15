#!/usr/bin/env python3
"""Remap the aliyun sources_registry.yaml (/mnt/*) to local /path/to/workspace paths.

The registry was authored for aliyun; every source path points to /path/to/storage or /mnt/data,
none of which exist on this box, so the flywheel silently fell back to ~3 mirrored sources.
41/48 sources ARE mirrored locally under /path/to/datasets/. This produces
`sources_registry_local.yaml` with remapped paths and an `available_local` flag, plus a report.

Remap rules (first existing wins):
  /mnt/data/datasets   -> /path/to/datasets/datasets
  /mnt/data            -> /path/to/datasets
  /path/to/storage             -> /path/to/workspace/foundation

Usage: python scripts/remap_registry_local.py [--in data/sources_registry.yaml]
"""
import argparse
import os
import yaml

REMAPS = [
    ("/mnt/data/datasets", "/path/to/datasets/datasets"),
    ("/mnt/data", "/path/to/datasets"),
    ("/path/to/storage", "/path/to/workspace/foundation"),
]


def remap(path):
    if not path:
        return path, False
    for a, b in REMAPS:
        if path.startswith(a):
            cand = path.replace(a, b, 1)
            if os.path.exists(cand):
                return cand, True
            return cand, False  # remapped but still missing
    return path, os.path.exists(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/sources_registry.yaml")
    ap.add_argument("--out", default="data/sources_registry_local.yaml")
    args = ap.parse_args()

    with open(args.inp) as f:
        reg = yaml.safe_load(f)

    # remap image_roots
    if isinstance(reg.get("image_roots"), dict):
        for k, v in list(reg["image_roots"].items()):
            reg["image_roots"][k], _ = remap(v)

    avail = 0
    report = []
    for s in reg.get("sources", []):
        p = s.get("path", "")
        newp, ok = remap(p)
        s["path"] = newp
        s["available_local"] = ok
        s.pop("available_aliyun", None)
        if ok:
            avail += 1
        report.append((s.get("name", "?"), ok, newp))

    reg["_note"] = ("Auto-remapped from aliyun registry to local /path/to/workspace paths by "
                    "remap_registry_local.py. available_local=True means the path exists here.")

    with open(args.out, "w") as f:
        yaml.safe_dump(reg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"remapped {len(report)} sources -> {avail} available locally")
    print(f"wrote {args.out}")
    print("\nMISSING (remapped but not found):")
    for n, ok, p in report:
        if not ok:
            print(f"  {n}: {p}")


if __name__ == "__main__":
    main()
