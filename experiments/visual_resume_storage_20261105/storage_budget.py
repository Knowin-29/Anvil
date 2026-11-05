"""Allocate real filesystem blocks and consume only this run's reserved capacity."""
from pathlib import Path
import fcntl,json,os,time
W=Path(__file__).resolve().parent;RESERVE=W/'storage_reserve.bin';STATE=W/'storage_reserve.json';GIB=1024**3

def save(p,v):
 t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def initialize(gib=256):
 assert not RESERVE.exists() and not STATE.exists()
 with RESERVE.open('xb') as f:
  os.posix_fallocate(f.fileno(),0,gib*GIB);os.fsync(f.fileno());s=os.fstat(f.fileno());assert s.st_size==gib*GIB and s.st_blocks*512>=gib*GIB
 save(STATE,dict(initial_bytes=gib*GIB,remaining_bytes=gib*GIB,inode=s.st_ino,created=time.time(),releases={}))
 return s.st_size

def release(label,gib):
 with (W/'storage_reserve.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  state=json.loads(STATE.read_text());amount=gib*GIB
  if label in state['releases']:
   assert state['releases'][label]['bytes']==amount;return
  s=RESERVE.stat();assert s.st_ino==state['inode'] and s.st_size==state['remaining_bytes'] and amount<=s.st_size
  with RESERVE.open('r+b') as f:f.truncate(s.st_size-amount);f.flush();os.fsync(f.fileno())
  state['remaining_bytes']-=amount;state['releases'][label]=dict(bytes=amount,time=time.time());save(STATE,state)

def before_checkpoint(stage,current,spec):
 cfg=spec['stages'][stage]
 for target in range(cfg['save_every'],cfg['steps']+1,cfg['save_every']):
  if stage=='repair' and target<=768:continue
  if current>=target-32:release(stage+'_checkpoint_'+str(target),3)

if __name__=='__main__':print(initialize())
