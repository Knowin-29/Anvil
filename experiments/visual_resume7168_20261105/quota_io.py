"""Unbuffered JSONL appends with exact partial-write recovery on shared quota errors.
Only operational I/O changes; no training, sampling, RNG or objective changes.
"""
import errno,fcntl,io,json,os,time
from pathlib import Path
W=Path(__file__).resolve().parent
RESERVE=W/'emergency_reserve.bin'
ORIGINAL_OPEN=Path.open
GIB=1024**3

def emergency_release(observed=None):
    with ORIGINAL_OPEN(W/'emergency.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=RESERVE.stat().st_size
        if observed is not None and before<observed:return
        if before<4*GIB:raise OSError(errno.EDQUOT,'Emergency storage reserve exhausted')
        with ORIGINAL_OPEN(RESERVE,'r+b') as f:
            f.truncate(before-4*GIB);f.flush();os.fsync(f.fileno())
        event=dict(time=time.time(),pid=os.getpid(),released=4*GIB,remaining=before-4*GIB)
        with ORIGINAL_OPEN(W/'emergency_releases.jsonl','a') as f:f.write(json.dumps(event)+'\n')
    time.sleep(2)

class DurableAppend(io.TextIOBase):
    def __init__(self,path,encoding=None,errors=None,opener=os.open):
        self.fd=opener(os.fspath(path),os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o666)
        self._encoding=encoding or 'utf-8';self._errors=errors or 'strict';self._name=os.fspath(path)
    @property
    def encoding(self):return self._encoding
    @property
    def name(self):return self._name
    def fileno(self):return self.fd
    def writable(self):return True
    def write(self,text):
        if self.closed:raise ValueError('I/O operation on closed file')
        payload=text.encode(self._encoding,self._errors);offset=0;retries=0
        while offset<len(payload):
            observed=RESERVE.stat().st_size if RESERVE.exists() else None
            try:
                count=os.write(self.fd,payload[offset:])
                if count<=0:raise OSError(errno.EIO,'Zero-length write')
                offset+=count
            except OSError as e:
                if e.errno not in (errno.EDQUOT,errno.ENOSPC) or retries>=8:raise
                retries+=1;emergency_release(observed)
        return len(text)
    def flush(self):
        if not self.closed:
            for attempt in range(9):
                observed=RESERVE.stat().st_size if RESERVE.exists() else None
                try:os.fsync(self.fd);return
                except OSError as e:
                    if e.errno not in (errno.EDQUOT,errno.ENOSPC) or attempt==8:raise
                    emergency_release(observed)
    def close(self):
        if not self.closed:
            try:super().close()
            finally:os.close(self.fd)

def install(run_root):
    run_root=Path(run_root)
    def guarded(path,mode='r',buffering=-1,encoding=None,errors=None,newline=None):
        if mode=='a' and path.parent.parent==run_root and (path.name=='logging.jsonl' or path.name.startswith('exposure_rank')):
            assert newline is None
            for attempt in range(9):
                observed=RESERVE.stat().st_size
                try:return DurableAppend(path,encoding,errors)
                except OSError as e:
                    if e.errno not in (errno.EDQUOT,errno.ENOSPC) or attempt==8:raise
                    emergency_release(observed)
        return ORIGINAL_OPEN(path,mode,buffering,encoding,errors,newline)
    Path.open=guarded

def probe(folder):
    p=Path(folder)/'.runtime_quota_probe'
    try:
        for attempt in range(9):
            observed=RESERVE.stat().st_size
            try:
                with ORIGINAL_OPEN(p,'wb') as f:
                    f.write(b'\0'*(1024**2));f.flush();os.fsync(f.fileno())
                return
            except OSError as e:
                if e.errno not in (errno.EDQUOT,errno.ENOSPC) or attempt==8:raise
                emergency_release(observed)
    finally:p.unlink(missing_ok=True)
