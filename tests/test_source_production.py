"""Tiny genuine source-shaped fixtures; not evidence of production-scale quality."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

from tools import arxiv_source_scale as scale
from tools import arxiv_source_semantic as semantic
from tools import arxiv_eval_v2_sources as sources
from tools import source_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['arxiv_source_scale', 'arxiv_source_semantic', 'arxiv_eval_v2_sources'])
def test_public_source_commands_help(name):
    result = subprocess.run([sys.executable, '-B', str(ROOT/'tools'/f'{name}.py'), '--help'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--root' in result.stdout


def test_verify_requires_explicit_execute(tmp_path):
    result = subprocess.run([sys.executable, '-B', str(ROOT/'tools/arxiv_source_semantic.py'),
                            'verify', '--root', str(tmp_path/'audit')], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'requires --execute' in result.stderr
    assert not (tmp_path/'audit').exists()


def test_pixmo_cli_actual_conversion_and_summary(tmp_path):
    raw = tmp_path/'source'
    raw.mkdir()
    Image.fromarray(np.random.default_rng(17).integers(0,256,(32,48,3),dtype=np.uint8)).save(raw/'image.png')
    row = {'id':'fixture-1','image':'image.png','width':48,'height':32,
           'conversations':[{'value':'<image>Where is the cup?'},
                            {'value':'<point>[[400,500],[400,500]]</point>'}]}
    source=raw/'points.jsonl'
    source.write_text(json.dumps(row)+'\n')
    empty=tmp_path/'empty.jsonl'
    empty.write_text('')
    output=tmp_path/'production'
    command=[sys.executable,'-B',str(ROOT/'tools/arxiv_source_scale.py'),'pixmo',
             '--root',str(output),'--pixmo-source',str(source)]
    for flag in ('eval-fingerprints','source-fingerprints','benchg-data','p5-dev','retention-panel'):
        command+=['--'+flag,str(empty)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    records=[r for split in ('train','evaluation_candidate') for r in scale.rows(output/'pixmo_points'/f'{split}.jsonl')]
    assert len(records)==1
    answer=json.loads(records[0]['messages'][1]['content'])
    assert answer==[{'label':'cup','point_2d':[400,500]}]
    assert records[0]['anvil_meta']['strong_model_verified'] is False
    manifest=json.loads((output/'pixmo_points/manifest.json').read_text())
    assert manifest['records']==1 and manifest['model_calls']==0
    for name,identity in manifest['files'].items():
        assert scale.sha(output/'pixmo_points'/name)==identity['sha256']
    scale.summary(output)
    summary=json.loads((output/'summary.json').read_text())
    assert summary['unique_records']==1 and summary['complete_11_form_release'] is False
    # Completed products are immutable; repeating conversion cannot append duplicates.
    retry=subprocess.run(command,capture_output=True,text=True)
    assert retry.returncode!=0
    assert scale.sha(output/'pixmo_points/train.jsonl')==manifest['files']['train.jsonl']['sha256']


def test_coco_all_seven_forms_have_source_geometry(tmp_path):
    from pycocotools import mask as mask_api
    image=tmp_path/'image.png'
    Image.fromarray(np.random.default_rng(23).integers(0,256,(64,64,3),dtype=np.uint8)).save(image)
    output=scale.Output(tmp_path,'coco_fixture',scale.Fingerprints(),scale.Fingerprints())
    info=output.image(image,64,64,'fixture-1')
    annotations=[]
    for i,(x,y) in enumerate([(4,4),(30,4),(16,34)],1):
        annotations.append({'id':i,'category_id':i,'area':100,'iscrowd':0,
                            'bbox':[x,y,10,10],
                            'segmentation':[[x,y,x+10,y,x+10,y+10,x,y+10]]})
    scale.coco_targets(output,info,annotations,{1:'cup',2:'bottle',3:'plate'},mask_api,24)
    output.finish({'scope':'synthetic source fixture'})
    records=[r for split in ('train','evaluation_candidate') for r in scale.rows(tmp_path/'coco_fixture'/f'{split}.jsonl')]
    assert len(records)==17 and len({r['id'] for r in records})==17
    assert {r['anvil_meta']['native_form'] for r in records}=={
        'bbox','point','mask','detection','direction','metric_comparison','relation_graph'}
    for record in records:
        meta=record['anvil_meta']
        assert meta['source_annotation_ids']
        assert meta['candidate_origin']=='public_source_annotation_conversion'
        assert meta['human_reviewed'] is False and meta['public_release_eligible'] is False
        if meta['native_form']=='metric_comparison':
            answer=json.loads(record['messages'][1]['content'])
            assert answer['unit']=='normalized_image'


def test_source_exclusions_and_duplicate_record_identity(tmp_path):
    image=tmp_path/'image.png'
    Image.new('RGB',(32,32),'red').save(image)
    protected=scale.Fingerprints()
    protected.add(scale.fingerprint(image))
    output=scale.Output(tmp_path,'blocked',protected,scale.Fingerprints())
    assert output.image(image,32,32,'same') is None
    assert output.skipped['protected_eval_overlap']==1
    output.finish({})
    assert json.loads((tmp_path/'blocked/manifest.json').read_text())['records']==0


def test_download_resume_verifies_existing_extracted_bytes(tmp_path,monkeypatch):
    cache=tmp_path/'coco_cache'
    cache.mkdir()
    with zipfile.ZipFile(cache/'train2017.zip','w') as archive:
        archive.writestr('train2017/000000000001.jpg',b'fixture-image')
    monkeypatch.setattr(scale,'COCO_IMAGE_BYTES',(cache/'train2017.zip').stat().st_size)
    monkeypatch.setattr(scale,'COCO_IMAGE_SHA256',scale.sha(cache/'train2017.zip'))
    scale.download_coco(tmp_path)
    scale.download_coco(tmp_path)
    (cache/'train2017/000000000001.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs'):
        scale.download_coco(tmp_path)


def test_unverified_existing_coco_archive_is_not_accepted(tmp_path):
    cache=tmp_path/'coco_cache'
    cache.mkdir()
    (cache/'train2017.zip').write_bytes(b'not-the-pinned-archive')
    with pytest.raises(ValueError,match='checksum'):
        scale.download_coco(tmp_path)
    assert not (cache/'download.json').exists()


def test_semantic_canonical_summary_preserves_votes_and_counts(tmp_path):
    run=tmp_path/'source_semantic'
    run.mkdir()
    candidates=[{'id':identity,'anvil_meta':{'split':'train'}} for identity in ('a','b','c')]
    runtime.write_rows(run/'candidates.jsonl',candidates)
    receipts=[]
    for identity,decision in zip(('a','b','c'),('accept','reject','uncertain')):
        answer=json.dumps({'decision':decision,'reason':'fixture evidence'})
        receipts.append({'id':identity,'decision':'uncertain','response':{'choices':[{
            'finish_reason':'stop','message':{'content':'```json\n'+answer+'\n```'}}]}})
    runtime.write_rows(run/'receipts.jsonl',receipts)
    original=runtime.sha(run/'receipts.jsonl')
    runtime.save(run/'manifest.json',{'n':3})
    semantic.summarize(tmp_path)
    manifest=json.loads((run/'canonical_manifest.json').read_text())
    assert manifest['counts']=={'accept':1,'reject':1,'uncertain':1}
    assert manifest['new_model_calls']==manifest['new_unique_records_added']==0
    assert runtime.sha(run/'receipts.jsonl')==original
    assert [r['id'] for r in runtime.rows(run/'canonical_accepted.jsonl')]==['a']


@pytest.mark.parametrize('text,finish,decision',[
    ('{"decision":"accept","reason":"visible"}','stop','accept'),
    ('{"decision":"reject","reason":"wrong object"}','stop','reject'),
    ('{"decision":"uncertain","reason":"occluded"}','stop','uncertain'),
    ('{"decision":"accept","reason":"visible"}','length','uncertain'),
    ('```json\n{"decision":"accept","reason":"visible"}\n```','stop','uncertain'),
    ('{"decision":"accept","reason":""}','stop','uncertain'),
    ('[]','stop','uncertain')])
def test_strict_semantic_votes(text,finish,decision):
    assert semantic.parse_vote(text,finish)[0]==decision


def test_gpu_selection_requires_shared_lock_path_before_hardware(tmp_path,monkeypatch):
    monkeypatch.delenv('ANVIL_GPU_LOCK_DIR',raising=False)
    with pytest.raises(ValueError,match='ANVIL_GPU_LOCK_DIR'):
        runtime.acquire(tmp_path,'0,1',2)
    monkeypatch.setenv('ANVIL_GPU_LOCK_DIR',str(tmp_path/'locks'))
    with pytest.raises(ValueError,match='Distinct'):
        runtime.acquire(tmp_path,'0,0',2)


def test_prism_depth_fixture_checks_reference_and_negative(tmp_path):
    import h5py
    raw=tmp_path/'prism_raw'
    raw.mkdir()
    image=raw/'view.png'
    Image.new('RGB',(24,16),'white').save(image)
    hdf5=raw/'scene.hdf5'
    segmentation=np.ones((16,24),dtype=np.int32)
    segmentation[:,12:]=2
    xyz=np.zeros((16,24,3),dtype=np.float32)
    xyz[:,:,2]=np.where(segmentation==1,1.,2.)
    with h5py.File(hdf5,'w') as file:
        view=file.create_group('view_0')
        view.create_dataset('xyz',data=xyz)
        view.create_dataset('seg',data=segmentation)
    item={'scene_id':'fixture-scene','view_id':'view_0','raw_hdf5':str(hdf5),
          'raw_sha256':runtime.sha(hdf5),'image':str(image),'image_sha256':runtime.sha(image),
          'object_names':['cup','bottle']}
    runtime.write_rows(raw/'views.jsonl',[item])
    runtime.save(raw/'manifest.json',{'files':{'views.jsonl':runtime.sha(raw/'views.jsonl')}})
    sources.prism_depth(tmp_path,tmp_path)
    manifest=json.loads((tmp_path/'prism_depth/manifest.json').read_text())
    assert manifest['counts']['records']==1
    assert manifest['counts']['reversed_reference_rejected']==1
    assert manifest['training_records_added']==manifest['evaluation_records_added']==0
    record=next(runtime.rows(tmp_path/'prism_depth/records.jsonl'))
    assert record['partition']=='unassigned_source_only'
    assert 'depth_quantiles' not in record['student_prompt']
    assert record['human_reviewed'] is False


def test_new_production_sources_pass_release_privacy_rules():
    from tools.export_anvil_release import SECRET,KEY_UUID,PRIVATE_IDENTITY,has_deployed_endpoint
    for name in ('arxiv_source_scale.py','arxiv_source_semantic.py','arxiv_eval_v2_sources.py','source_runtime.py'):
        raw=(ROOT/'tools'/name).read_bytes()
        assert not SECRET.search(raw),name
        assert not KEY_UUID.search(raw),name
        assert not PRIVATE_IDENTITY.search(raw),name
        assert not has_deployed_endpoint(raw),name
