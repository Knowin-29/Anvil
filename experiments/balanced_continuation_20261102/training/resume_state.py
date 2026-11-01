"""Restore only the committed checkpoint prefix, preserving discarded suffixes."""
import json
from pathlib import Path
import time


def replace_preserving_original(path, content):
    path = Path(path)
    old = path.read_bytes() if path.exists() else b''
    if old == content:
        return None
    backup = path.with_name(path.name + f'.before_resume_{time.time_ns()}')
    with backup.open('xb') as stream:
        stream.write(old)
    pending = path.with_name(path.name + '.resume_pending')
    with pending.open('xb') as stream:
        stream.write(content)
    pending.replace(path)
    return str(backup)


def restore_exposures(path, step, rank, accumulation):
    path = Path(path)
    raw = path.read_bytes()
    keep = []
    expected = [(s, m) for s in range(1, step + 1) for m in range(accumulation)]
    observed = []
    for line in raw.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except ValueError:
            if observed != expected:
                raise ValueError('Malformed exposure before the committed prefix is complete')
            continue
        if row['step'] <= step:
            if row['rank'] != rank:
                raise ValueError('Exposure belongs to another rank')
            observed.append((row['step'], row['micro']))
            keep.append(line if line.endswith(b'\n') else line + b'\n')
    if observed != expected:
        raise ValueError('Missing, duplicated or reordered committed exposures')
    return replace_preserving_original(path, b''.join(keep))


def restore_logs(path, history, step):
    if [row['step'] for row in history] != list(range(1, step + 1)):
        raise ValueError('Checkpoint log history does not cover its committed steps')
    content = b''.join((json.dumps(row, allow_nan=False) + '\n').encode() for row in history)
    return replace_preserving_original(path, content)
