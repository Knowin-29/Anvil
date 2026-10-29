"""Prepare a separate CPU-only vacancy pool; never change active training.

Same source targets, conservative image/scan exclusion, deterministic scan
reservation. The resulting inventory is not an admitted training recipe.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

C = Path(__file__).resolve().parent
D = C / 'diagnosis_32gpu'
sys.path.insert(0, str(D))
from prepare_spatial_replay import Protected, fingerprint, sha, EXPORT, FSD
from audit_vacancy_source_lineage import UPSTREAM, bound, point_target, read_rows

OUT = C / 'vacancy_expansion_candidate_20261029'


def save(path, item):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(item, indent=2)+'\n')
    temp.replace(path)


def scan(path):
    return Path(path).name.split('_')[0]


def reserved(path):
    return int(hashlib.sha256(('scan:'+scan(path)).encode()).hexdigest()[:8],16)%20 == 0


def main():
    started=time.time()
    OUT.mkdir(exist_ok=False)
    source=UPSTREAM/'vacant_qa_filtered_3D_groupLength.jsonl'
    historical_dev=Path('/Knowin/demo/weilinruan/Anvil/04_Experiments/iclr_evidence/engineering/data/vacancy_repair_20260914/split/dev.jsonl')
    development_paths=[historical_dev,D/'spatial_replay/development.jsonl',D/'retention_replay/development.jsonl',
                       D/'prism_expansion/sft_source/evaluation.jsonl']
    paths=[source, C/'vacancy_upstream_fidelity_20261029.json',D/'inputs/manifest.json',EXPORT/'manifest.json',
           D/'spatial_replay/manifest.json',D/'spatial_replay/refspatial_vacancy.jsonl',
           FSD/'data_test_internvl_no_think.jsonl',*development_paths,Path(__file__),
           D/'prepare_spatial_replay.py',C/'audit_vacancy_source_lineage.py',C/'Anvil/tools/arxiv_source_scale.py']
    bindings={str(p):bound(p) for p in paths}
    policy=dict(schema='anvil-vacancy-expansion-preparation-v1',created=time.time(),
        source_records=231496,rule='File/RGB identity or pHash/dHash Hamming<=6; any-hit entire RefSpatial scan excluded.',
        reserve_rule='sha256(scan:<id>)[:8] modulo20 ==0; no reserved scan enters training.',
        other_reservations='All campaign native/public/general evaluation; FSD test; historical and current development.',
        targets='Exact processed source prompts and point targets; no label regeneration or coordinate modification.',
        evidence=bindings,training_admitted=False,active_training_modified=False)
    save(OUT/'policy.json',policy)
    save(OUT/'status.json',dict(stage='reading_annotations',updated=time.time(),pid=os.getpid()))
    original=json.loads((EXPORT/'manifest.json').read_text())
    inputs=json.loads((D/'inputs/manifest.json').read_text())
    active=list(read_rows(D/'spatial_replay/refspatial_vacancy.jsonl'))
    active_by_uid={r['canonical_uid']:r for r in active}
    candidates=[];seen=set();image_paths=set();existing_matched=set()
    for row in read_rows(source):
        relative=Path(row['image'])
        assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='image'
        path=str((UPSTREAM/relative).resolve())
        prompt,target=[x['value'] for x in row['conversations']]
        assert prompt.startswith('<image>') and prompt.count('<image>')==1
        points=[list(p) for p in point_target(target)]
        key=hashlib.sha256(json.dumps([path,prompt,points],sort_keys=True).encode()).hexdigest()
        uid='refspatial_vacancy:'+key
        assert uid not in seen,'Duplicate upstream record'
        seen.add(uid);image_paths.add(path)
        if uid in active_by_uid:
            old=active_by_uid[uid]
            assert old['images']==[path] and [m['content'] for m in old['messages']]==[prompt,target]
            existing_matched.add(uid)
        candidates.append(dict(uid=uid,path=path,prompt=prompt,target=target,source_id=row['id'],
            width=row['width'],height=row['height'],
            record_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()))
    assert len(candidates)==231496 and existing_matched==set(active_by_uid)
    protected=Protected();original_rgb=set();native_count=0;database_rows=hashlib.sha256()
    db=sqlite3.connect('file:'+original['pool_index']+'?mode=ro',uri=True)
    for decoded,file_sha,path,split,ph,dh in db.execute('SELECT decoded_sha256,file_sha256,path,split,phash64,dhash64 FROM images ORDER BY decoded_sha256,path'):
        database_rows.update(json.dumps([decoded,file_sha,path,split,ph,dh],separators=(',',':')).encode()+b'\n')
        original_rgb.add(decoded)
        if split=='benchmark':
            protected.add(dict(sha256=file_sha,decoded_rgb_sha256=decoded,phash64=ph,dhash64=dh),path)
            native_count+=1
    db.close()
    protection_paths=set(inputs['images']);protected_ref_scans=set()
    for path in development_paths:
        for row in read_rows(path):
            for image_path in row['images']:
                protection_paths.add(image_path)
                if '/refspatial/3D/' in image_path:
                    protected_ref_scans.add(scan(image_path))
    for row in read_rows(FSD/'data_test_internvl_no_think.jsonl'):
        protection_paths.add(str(FSD/row['image']))
    all_paths=sorted(image_paths|protection_paths)
    fingerprints={}
    save(OUT/'status.json',dict(stage='fingerprinting',images=len(all_paths),candidates=len(candidates),updated=time.time(),pid=os.getpid()))
    def inspect(path):
        return path,fingerprint(path)
    with (OUT/'image_fingerprints.jsonl').open('x') as out, ThreadPoolExecutor(max_workers=16) as pool:
        for i,(path,fp) in enumerate(pool.map(inspect,all_paths),1):
            if path in inputs['images']:
                assert fp['sha256']==inputs['images'][path],'Changed public/general evaluation image'
            fingerprints[path]=fp
            out.write(json.dumps(dict(path=path,fingerprint=fp))+'\n')
            if i%1000==0 or i==len(all_paths):
                out.flush()
                save(OUT/'status.json',dict(stage='fingerprinting',images=len(all_paths),complete=i,updated=time.time(),pid=os.getpid()))
                print('fingerprints',i,len(all_paths),flush=True)
    for path in protection_paths:
        protected.add(fingerprints[path],path)
    # The fixed scan reservation is protected before examining any scores.
    reserved_paths={p for p in image_paths if reserved(p)}
    for path in reserved_paths:
        protected.add(fingerprints[path],path)
    matches={};blocked_scans=set(protected_ref_scans)
    for i,path in enumerate(sorted(image_paths),1):
        match=protected.match(fingerprints[path])
        if reserved(path):
            match=dict(reason='deterministic_reserved_scan',scan=scan(path))
        elif scan(path) in protected_ref_scans:
            match=dict(reason='historical_or_campaign_development_scan',scan=scan(path))
        matches[path]=match
        if match:blocked_scans.add(scan(path))
        if i%1000==0:
            save(OUT/'status.json',dict(stage='excluding_overlap',complete=i,images=len(image_paths),updated=time.time(),pid=os.getpid()))
    counts=Counter();train_uids=set();train_images=set();train_scans=set();semantic_seen=set();kept_old=0
    with (OUT/'train.jsonl').open('x') as train,(OUT/'exclusions.jsonl').open('x') as excluded:
        for row in candidates:
            path=row['path'];fp=fingerprints[path]
            assert (row['width'],row['height'])==(fp['width'],fp['height']),'Source/image dimension mismatch'
            match=matches[path]
            if scan(path) in blocked_scans:
                match=match or dict(reason='same_scan_as_excluded_image',scan=scan(path))
            identity=(fp['decoded_rgb_sha256'],row['prompt'],row['target'])
            if not match and identity in semantic_seen:
                match=dict(reason='duplicate_decoded_image_prompt_target')
            if match:
                counts['excluded_'+match['reason']]+=1
                excluded.write(json.dumps(dict(uid=row['uid'],path=path,**match))+'\n')
                continue
            semantic_seen.add(identity)
            if row['uid'] in active_by_uid:
                item=deepcopy(active_by_uid[row['uid']]);kept_old+=1
                assert item['gsa_meta']['binding']['sha256']==fp['sha256']
            else:
                binding=dict(source_dataset='refspatial_vacancy',source_id=row['source_id'],
                    source_record_sha256=row['record_sha256'],source_split='source_annotation_after_image_and_scan_exclusion',
                    image_path=path,**fp)
                item=dict(id=row['uid'],canonical_uid=row['uid'],gsa_form='point',gsa_family='G',
                    gsa_meta=dict(split='train',binding=binding),images=[path],
                    messages=[dict(role='user',content=row['prompt']),dict(role='assistant',content=row['target'])])
            train.write(json.dumps(item,ensure_ascii=False)+'\n')
            assert row['uid'] not in train_uids
            train_uids.add(row['uid']);train_images.add(fp['decoded_rgb_sha256']);train_scans.add(scan(path));counts['train']+=1
    assert train_scans.isdisjoint(blocked_scans)
    assert sum(counts.values())==231496 and len(train_uids)==counts['train']
    assert {str(p):bound(p) for p in paths}==bindings,'Sealed input changed'
    report=dict(schema='anvil-vacancy-expansion-candidate-v1',state='prepared_requires_training_recipe_review',
        counts=dict(counts),source_records=231496,source_unique_images=len(image_paths),
        candidate_train_records=counts['train'],candidate_unique_images=len(train_images),candidate_train_scans=len(train_scans),
        unchanged_active_rows_retained=kept_old,active_rows_excluded_by_stronger_full_pool_reservation=len(active)-kept_old,
        additional_records_vs_active=len(train_uids-set(active_by_uid)),new_to_original_images=len(train_images-original_rgb),
        protected_native_images=native_count,protected_external_images=len(protection_paths),
        fixed_reserved_source_images=len(reserved_paths),blocked_scans=len(blocked_scans),
        database_image_rows_sha256=database_rows.hexdigest(),
        files={n:bound(OUT/n) for n in ['policy.json','train.jsonl','exclusions.jsonl','image_fingerprints.jsonl']},
        original_corpus_records=5800000,original_sft_records=5220000,actual_training_exposures=0,
        training_admitted=False,active_training_modified=False,benchmark_labels_modified=False,
        limitations=['Source annotations have no claimed official train/test partition; conservative campaign reservations are applied.',
                     'Image and scan exclusions do not establish absence of every semantic relationship.',
                     'This pool does not establish target semantic correctness or improvement on a benchmark.',
                     'Expanded inventory must not be counted as observed training or added twice to existing supplements.'],
        elapsed=time.time()-started,completed=time.time())
    save(OUT/'manifest.json',report)
    save(OUT/'status.json',dict(stage=report['state'],training_admitted=False,updated=time.time(),pid=os.getpid()))
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():save(OUT/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
