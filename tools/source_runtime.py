"""Small runtime primitives for source conversion and optional local verification."""
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, values):
    with Path(path).open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def environment(run):
    # A user-selected short scratch path also avoids Unix socket length limits.
    scratch = Path(os.environ.get('ANVIL_TMP_ROOT', tempfile.gettempdir()))
    scratch.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix='anvil-', dir=scratch))
    env = dict(os.environ)
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', DS_IGNORE_CUDA_DETECTION='1',
               WANDB_MODE='disabled', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               HF_HOME=str(run/'cache/hf'), XDG_CACHE_HOME=str(run/'cache'),
               TMPDIR=str(runtime), TRITON_CACHE_DIR=str(run/'cache/triton'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR',
                'MASTER_PORT', 'NPROC_PER_NODE', 'PYTHONPATH'):
        env.pop(key, None)
    save(run/'runtime.json', {'tmpdir': str(runtime)})
    return env


def acquire(run, selectors, count=1):
    """Cooperative GPU locks, never displacing existing compute processes."""
    configured = os.environ.get('ANVIL_GPU_LOCK_DIR')
    if not configured:
        raise ValueError('Set ANVIL_GPU_LOCK_DIR to one shared lock directory for all local workers')
    devices = [value.strip() for value in selectors.split(',') if value.strip()]
    if len(set(devices)) != len(devices) or len(devices) < count or count < 1:
        raise ValueError('Distinct GPU selectors must cover the requested device count')
    locks = Path(configured)
    locks.mkdir(parents=True, exist_ok=True)
    while True:
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader'], text=True)
        busy = {row[0].strip() for row in csv.reader(active.splitlines()) if row}
        chosen = []
        try:
            for gpu in devices:
                uuid, memory = map(str.strip, subprocess.check_output(
                    ['nvidia-smi', '-i', gpu, '--query-gpu=uuid,memory.used',
                     '--format=csv,noheader,nounits'], text=True).split(','))
                if uuid in busy or int(memory) > 4096 or uuid in {u for u, _ in chosen}:
                    continue
                handle = (locks/(uuid+'.lock')).open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                chosen.append((uuid, handle))
                if len(chosen) == count:
                    save(run/'gpu.json', {'devices': [u for u, _ in chosen], 'allocated': stamp()})
                    return chosen
        except BaseException:
            for _, handle in chosen:
                handle.close()
            raise
        for _, handle in chosen:
            handle.close()
        save(run/'waiting.json', {'state': 'waiting_for_idle_GPU', 'selectors': selectors, 'at': stamp()})
        time.sleep(30)
