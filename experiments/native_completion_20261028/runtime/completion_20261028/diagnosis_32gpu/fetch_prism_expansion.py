"""Download four pinned official training archives without using HF credentials.

Downloaded source is not admitted to training. Scene/image exclusions, calibrated
mapping, source scale accounting and a new training-only split must follow.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'prism_expansion'
sys.path.insert(0,str(ROOT.parent.parent/'cloud_a800'))
from checkpoints import save


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    manifest=json.loads((ROOT/'prism_expansion_inventory.json').read_text())
    assert manifest['repo']=='allenai/PRISM' and manifest['revision']=='34cfb7f6d98e019a2251cdc40c8079fbf0cccba1'
    assert set(manifest['files'])=={f'PRISM-train/chunk_{i}_of_100.tar' for i in range(2,6)}
    OUT.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        name,info=item;target=OUT/'downloads'/name;target.parent.mkdir(parents=True,exist_ok=True)
        status=OUT/(Path(name).stem+'_download.json');began=time.time()
        save(status,dict(stage='downloading',source=name,expected_bytes=info['bytes'],started=began))
        if not target.exists():
            partial=target.with_suffix('.tar.partial')
            url=f"https://huggingface.co/datasets/{manifest['repo']}/resolve/{manifest['revision']}/{name}"
            cmd=['curl','--disable','--fail','--location','--silent','--show-error','--retry','4',
                 '--retry-all-errors','--connect-timeout','30','--max-time','14400','--continue-at','-',
                 '--output',str(partial),url]
            subprocess.run(cmd,check=True)
            assert partial.stat().st_size==info['bytes'] and sha(partial)==info['sha256'],name
            partial.rename(target)
        assert target.stat().st_size==info['bytes'] and sha(target)==info['sha256'],name
        receipt=dict(state='download_integrity_verified',source=name,path=str(target),bytes=info['bytes'],
            sha256=info['sha256'],seconds=time.time()-began,training_admitted=False)
        save(status,receipt);return receipt
    receipts=[];failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,item):item[0] for item in manifest['files'].items()}
        for future in as_completed(futures):
            try:
                result=future.result();receipts.append(result);print(json.dumps(result),flush=True)
            except BaseException as exc:
                failures.append(dict(source=futures[future],error=str(exc)))
    save(OUT/'download_complete.json',dict(state='failed' if failures else 'all_archives_verified',
        files=receipts,failures=failures,training_admitted=False,inventory_sha256=sha(ROOT/'prism_expansion_inventory.json'),
        server_hf_credentials_modified=False,completed=time.time()))
    if failures:raise RuntimeError('Some pinned downloads failed; partial files retained')


if __name__=='__main__':main()
