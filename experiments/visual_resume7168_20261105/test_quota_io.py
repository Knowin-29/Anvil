import errno,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import quota_io as q
class QuotaTest(unittest.TestCase):
 def test_partial_write_quota_no_duplicate_or_missing_unicode(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'test';p.write_bytes(b'prefix\n');real=q.os.write;calls=[0]
   def write(fd,data):
    calls[0]+=1
    if calls[0]==1:return real(fd,data[:3])
    if calls[0]==2:raise OSError(errno.EDQUOT,'injected quota')
    return real(fd,data)
   with patch.object(q.os,'write',write),patch.object(q,'emergency_release') as release:
    with q.DurableAppend(p) as f:self.assertEqual(f.write('中文 record\n'),10)
    release.assert_called_once()
   self.assertEqual(p.read_bytes(),b'prefix\n'+'中文 record\n'.encode())
 def test_flush_quota_retries_without_rewriting(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'test'
   with q.DurableAppend(p) as f:
    f.write('once\n')
    with patch.object(q.os,'fsync',side_effect=[OSError(errno.EDQUOT,'injected'),None]),patch.object(q,'emergency_release') as release:
     f.flush();release.assert_called_once()
   self.assertEqual(p.read_text(),'once\n')
 def test_nonquota_error_propagates(self):
  with tempfile.TemporaryDirectory() as tmp:
   with q.DurableAppend(Path(tmp)/'test') as f:
    with patch.object(q.os,'write',side_effect=OSError(errno.EIO,'injected')):
     with self.assertRaises(OSError):f.write('line')
 def test_scope_and_exact_content(self):
  original=Path.open;old_reserve=q.RESERVE
  try:
   with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)/'runs';stage=root/'repair';stage.mkdir(parents=True)
    q.install(root)
    q.RESERVE=Path(tmp)/'reserve';q.RESERVE.write_bytes(b'0'*64)
    with (stage/'logging.jsonl').open('a') as f:
     self.assertIsInstance(f,q.DurableAppend);f.write('{"step":1}\n')
    self.assertEqual((stage/'logging.jsonl').read_text(),'{"step":1}\n')
    with (stage/'other').open('a') as f:self.assertNotIsInstance(f,q.DurableAppend)
  finally:Path.open=original;q.RESERVE=old_reserve
 def test_shared_release_consumed_once_for_same_observation(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);reserve=root/'reserve';reserve.write_bytes(b'0'*8192)
   with patch.object(q,'W',root),patch.object(q,'RESERVE',reserve),patch.object(q,'GIB',1024),patch.object(q.time,'sleep'):
    q.emergency_release(8192);q.emergency_release(8192)
   self.assertEqual(reserve.stat().st_size,4096)
if __name__=='__main__':unittest.main()
