"""Materialize pinned PRISM train scenes and build a separate, audited supplement.

Original native inputs, splits, targets, scores and inventory counts are immutable.
Raw calibration/annotation mapping uses the existing mapper unchanged. New scenes
are assigned before training and compared against all frozen evaluation images.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent;OUT=ROOT/'prism_expansion'
ORIGINAL=R.parent/'arxiv/reconstruction_20260924/arxiv_eval_v2.cjcKvZ0N'
TOOLS=C/'Anvil/tools'
sys.path[:0]=[str(ORIGINAL/'inspect_deps'),str(TOOLS),str(C/'Anvil'),str(ROOT),str(R/'cloud_a800')]
from checkpoints import save
from arxiv_source_scale import fingerprint
from prepare_spatial_replay import Protected
import arxiv_eval_v2_sources as raw_mapper
import map_prism_native as native
from trains.flywheel.native_training import bind_training_record,validate_training_record


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def records(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]


def verify_sources():
    bound=json.loads((OUT/'preparation_sources.json').read_text())
    for path,digest in bound['files_sha256'].items():assert sha(path)==digest,path


def chunk(number):
    verify_sources()
    inventory=json.loads((ROOT/'prism_expansion_inventory.json').read_text())
    name=f'PRISM-train/chunk_{number}_of_100.tar';info=inventory['files'][name]
    staged=OUT/'source_prepare/allenai/PRISM'/name
    assert staged.stat().st_size==info['bytes'] and sha(staged)==info['sha256']
    # Per-process source parameterization; the original module/file is unchanged.
    original=list(raw_mapper.SOURCES)
    original[1]=(inventory['repo'],inventory['revision'],name,info['bytes'],info['sha256'])
    raw_mapper.SOURCES=tuple(original)
    destination=OUT/'materialized'/f'chunk_{number}';destination.mkdir(parents=True,exist_ok=True)
    raw=destination/'prism_raw'
    if not (raw/'manifest.json').exists():
        assert not raw.exists(),'Partial extraction retained; explicit recovery required'
        raw_mapper.materialize_prism(OUT,destination)
    else:
        receipt=json.loads((raw/'manifest.json').read_text());assert receipt['archive_sha256']==info['sha256']
    mapped=destination/'native'
    if not (mapped/'manifest.json').exists():
        assert not mapped.exists(),'Partial native mapping retained; explicit recovery required'
        native.convert(raw,mapped)
    return str(mapped)


def main(wait_download):
    verify_sources()
    while not (OUT/'download_complete.json').exists():
        assert wait_download,'Pinned downloads are not complete'
        save(OUT/'preparation_status.json',dict(stage='waiting_for_downloads',updated=time.time()))
        time.sleep(15)
    downloaded=json.loads((OUT/'download_complete.json').read_text());assert downloaded['state']=='all_archives_verified'
    assert not (OUT/'manifest.json').exists(),'Never overwrite an admitted supplemental pool'
    metadata=ORIGINAL/'source_prepare/allenai/PRISM/train.csv'
    assert sha(metadata)==raw_mapper.SOURCES[0][4]
    staging=OUT/'source_prepare/allenai/PRISM';staging.mkdir(parents=True,exist_ok=True)
    links={staging/'train.csv':metadata}
    links.update({staging/item['source']:Path(item['path']) for item in downloaded['files']})
    for destination,source in links.items():
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert destination.resolve()==source.resolve()
        else:destination.symlink_to(source.resolve())
    save(OUT/'preparation_status.json',dict(stage='raw_mapping',updated=time.time()))
    def launch(number):
        with (OUT/f'chunk_{number}_preparation.log').open('a') as log:
            subprocess.run([sys.executable,__file__,'--chunk',str(number)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,range(2,6)))
    save(OUT/'preparation_status.json',dict(stage='scene_and_image_exclusions',updated=time.time()))
    all_records=[];seen=set();raw_images={};raw_proofs={}
    for number in range(2,6):
        base=OUT/'materialized'/f'chunk_{number}';file=base/'native/records.jsonl'
        receipt=json.loads((base/'native/manifest.json').read_text())
        assert sha(file)==receipt['files']['records.jsonl']
        raw_proofs[str(base/'native/manifest.json')]=sha(base/'native/manifest.json')
        for row in records(file):
            native.validate_source_native_record(row)
            assert row['id'] not in seen;seen.add(row['id']);all_records.append(row)
            relative=row['images'][0];image=base/'prism_raw'/relative
            if relative in raw_images:assert raw_images[relative]==image
            raw_images[relative]=image
    train_export=R.parent/'arxiv/native_training_export_20261012_v4_r2'
    native_manifest=json.loads((train_export/'manifest.json').read_text())
    protected=Protected();original_rgb=set();protected_sha=set();protected_scenes=set()
    for file in [train_export/'train_dof.jsonl',R.parent/'arxiv/native_evaluation_export_20261012_v4/eval_dof.jsonl']:
        for row in records(file):protected_scenes.add('PRISM:'+Path(row['images'][0]).parent.name)
    with sqlite3.connect('file:'+native_manifest['pool_index']+'?mode=ro',uri=True) as db:
        for rgb,file_sha,path,split,ph,dh in db.execute('SELECT decoded_sha256,file_sha256,path,split,phash64,dhash64 FROM images'):
            original_rgb.add(rgb)
            if split=='benchmark':
                protected.add(dict(sha256=file_sha,decoded_rgb_sha256=rgb,phash64=ph,dhash64=dh),path);protected_sha.add(file_sha)
    public=json.loads((ROOT/'inputs/manifest.json').read_text())
    def inspect(item):
        path,digest=item;fp=fingerprint(path);assert fp['sha256']==digest;return path,fp
    with ThreadPoolExecutor(max_workers=16) as pool:
        for path,fp in pool.map(inspect,public['images'].items()):protected.add(fp,path);protected_sha.add(fp['sha256'])
    # Protect supplementary development images too; only their image identities
    # enter this exclusion pass, never their reference answers.
    dev_paths=set()
    for row in records(ROOT/'spatial_replay/development.jsonl'):dev_paths.update(row['images'])
    fsd=Path('/Knowin/foundation/data/datasets/originDatasets_transfer_from_hpc3/FSD_points_rft_fsd_free_point_train_32790_test_300_0425')
    for row in records(fsd/'data_test_internvl_no_think.jsonl'):dev_paths.add(str(fsd/row['image']))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for path,fp in zip(sorted(dev_paths),pool.map(fingerprint,sorted(dev_paths))):protected.add(fp,path);protected_sha.add(fp['sha256'])
    fingerprints={}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for relative,fp in zip(raw_images,pool.map(fingerprint,raw_images.values())):fingerprints[relative]=fp
    scenes={r['group_id'] for r in all_records};blocked={s:'existing_original_PRISM_scene' for s in scenes&protected_scenes}
    for relative,fp in fingerprints.items():
        hit=protected.match(fp)
        if hit:blocked['PRISM:'+Path(relative).parts[0]]=hit
    # Frozen deterministic development partition; all scenes are assigned before
    # constructing training rows. A near-hit excludes the entire training scene.
    development={s for s in scenes if s not in blocked and int(hashlib.sha256(('Anvil-PRISM-expansion-dev-v1:'+s).encode()).hexdigest()[:8],16)%10==0}
    for relative,fp in fingerprints.items():
        if 'PRISM:'+Path(relative).parts[0] in development:protected.add(fp,'new_development:'+relative)
    for relative,fp in fingerprints.items():
        scene='PRISM:'+Path(relative).parts[0]
        if scene in development or scene in blocked:continue
        hit=protected.match(fp)
        if hit:blocked[scene]=hit
    assignments={scene:dict(split='exclude' if scene in blocked else 'validation' if scene in development else 'train',
                           training_approved=scene not in blocked and scene not in development) for scene in sorted(scenes)}
    combined=OUT/'combined_native';combined.mkdir(exist_ok=False)
    file=combined/'records.jsonl'
    with file.open('x') as f:
        for row in all_records:f.write(native.canonical(row)+'\n')
    source_sha=sha(file)
    save(combined/'manifest.json',dict(state='source_native_extension_materialized',contract=native.CONTRACT,
        files={'records.jsonl':source_sha},raw_mapping_manifests=raw_proofs,supplemental=True,records=len(all_records)))
    save(OUT/'scene_assignments.json',dict(schema=native.SPLIT_CONTRACT,contract=native.CONTRACT,
        source_records_sha256=source_sha,assignments=assignments,
        authority='User-authorized project repair and supplemental train-only data preparation',development_seed='Anvil-PRISM-expansion-dev-v1'))
    save(OUT/'exclusions.json',dict(schema=native.EXCLUSION_CONTRACT,contract=native.CONTRACT,source_records_sha256=source_sha,
        protected_group_ids=sorted(protected_scenes|set(blocked)),protected_image_sha256=sorted(protected_sha),
        evidence_sha256=[sha(train_export/'manifest.json'),sha(ROOT/'inputs/manifest.json'),sha(ROOT/'spatial_replay/manifest.json')],
        excluded_scenes=blocked,perceptual_rule='OR pHash/dHash Hamming<=6; exclude fullscene',original_scenes_excluded=True))
    image_root=OUT/'images'
    for relative,image in raw_images.items():
        dest=image_root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(image,dest)
        assert sha(dest)==fingerprints[relative]['sha256']
    export=OUT/'sft_source'
    native.export_sft(combined,export,image_root,OUT/'scene_assignments.json',OUT/'exclusions.json',native.CONTRACT,include_heldout=True)
    source=export/'train.jsonl';source_sha=sha(source);selection_sha=sha(OUT/'scene_assignments.json')
    count=0;images=set();training_scenes=set();target=OUT/'train_bound.jsonl';offset=0
    with source.open('rb') as f,target.open('x') as out:
        for line in f:
            row=json.loads(line);meta=row['anvil_meta'];relative=str(Path(row['images'][0]).relative_to(image_root));fp=fingerprints[relative]
            binding=dict(split='train',source_uid=row['id'],source='PRISM',native_form='dof',native_variant=native.CONTRACT,
                group_id=meta['group_id'],selected_group_id=meta['group_id'],image_path=row['images'][0],image_sha256=fp['sha256'],
                decoded_rgb_sha256=fp['decoded_rgb_sha256'],selection_sha256=selection_sha,source_sha256=source_sha,
                source_line_sha256=hashlib.sha256(line).hexdigest(),source_offset=offset,source_size=len(line))
            bound=bind_training_record(line,binding);validate_training_record(bound)
            out.write(native.canonical(bound)+'\n');offset+=len(line);count+=1;images.add(fp['decoded_rgb_sha256']);training_scenes.add(meta['group_id'])
    assert training_scenes.isdisjoint(development|protected_scenes|set(blocked)) and count>0
    save(OUT/'manifest.json',dict(state='verified_train_only_supplement',additional_training_records=count,
        training_scenes=len(training_scenes),training_unique_images=len(images),new_images_vs_original_inventory=len(images-original_rgb),
        original_corpus_records_unchanged=5800000,original_sft_records_unchanged=5220000,
        total_source_records=len(all_records),source_scenes=len(scenes),development_scenes=len(development),excluded_scenes=len(blocked),
        files_sha256={'train_bound.jsonl':sha(target),'scene_assignments.json':sha(OUT/'scene_assignments.json'),
            'exclusions.json':sha(OUT/'exclusions.json'),'sft_source/evaluation.jsonl':sha(export/'evaluation.jsonl')},
        official_inventory_sha256=sha(ROOT/'prism_expansion_inventory.json'),metadata_sha256=sha(metadata),
        code_sha256={str(p):sha(p) for p in [Path(__file__),TOOLS/'map_prism_native.py',TOOLS/'arxiv_eval_v2_sources.py',TOOLS/'arxiv_source_scale.py']},
        source_pose_annotations_unchanged=True,benchmark_targets_or_scores_changed=False,actual_training_exposures=0,
        no_claim_of_all_semantic_overlap_exclusion=True,completed=time.time()))
    save(OUT/'preparation_status.json',dict(stage='complete_verified_supplement_not_yet_trained',records=count,completed=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--chunk',type=int,choices=range(2,6));ap.add_argument('--wait-download',action='store_true');args=ap.parse_args()
    try:
        if args.chunk:chunk(args.chunk)
        else:main(args.wait_download)
    except BaseException as exc:
        save(OUT/(f'chunk_{args.chunk}_failure.json' if args.chunk else 'preparation_failure.json'),dict(error=str(exc),time=time.time()))
        raise
