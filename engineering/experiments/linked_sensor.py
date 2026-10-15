"""Recover linked G→S supervision from cached, official-training RGB-D frames.

No teacher output, test score or language-model call selects targets. Unique
visible names or explicit interior clicks identify objects without hidden IDs.
"""
from __future__ import annotations

import collections
from copy import deepcopy
from dataclasses import asdict
import fcntl
from functools import lru_cache
import hashlib
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

from native_completion import A, W, D, sha, save, commit_bytes, rows, read_binding

ROOT = D / 'linked_sensor_gs_v1'
SCANNET = Path('/path/to/source-archives/nas-backup/datasets/public_sources/downloads/ScanNet-official')
LABEL_MAP = SCANNET / 'tasks/scannetv2-labels.combined.tsv'
FRAMES_PER_SCAN = 8


def sensor_module():
    sys.path[:0] = [str(W / 'tools'), str(W)]
    import build_depth_v50_scannet_sensor as sensor
    return sensor


@lru_cache(maxsize=1)
def label_maps():
    return sensor_module().load_label_map(LABEL_MAP)


def source_tasks():
    sensor = sensor_module()
    original_path = D / 'scannet_flywheel_v1/manifest.json'
    original = read_binding(original_path)
    _, official = sensor.load_split_manifests(W / 'configs/scannet_splits')
    salt = original['selection']['salt']
    tasks, seen = [], set()
    for binding in original['source_manifests']:
        path = Path(binding['path'])
        assert sha(path) == binding['sha256']
        document = read_binding(path)
        assert document['verdict'] == 'SCALEOUT_VALIDATED_TRAIN_SPLIT_ONLY'
        assert document['use_policy']['training_materialization_eligible'] is True
        assert document['use_policy']['official_train_only'] is True
        for scene in document['scenes']:
            name = scene['scene_id']
            assert name not in seen and official[name] == scene['official_split'] == 'train'
            assert scene['depth_shift_raw_units_per_meter'] == 1000.0
            seen.add(name)
            group = name.split('_')[0]
            bucket = int(hashlib.sha256(f'{salt}\0{group}'.encode()).hexdigest()[:16], 16) % 1000
            split = 'dev' if bucket < original['split_rule']['dev_permille'] else 'train'
            selected = sorted(scene['frames'], key=lambda f: hashlib.sha256(
                f"{salt}\0{name}\0{f['frame_index']}".encode()).hexdigest())[:FRAMES_PER_SCAN]
            for frame in selected:
                tasks.append({'id':f"{name}_{frame['frame_index']:06d}", 'scene_id':name,
                    'scene_group':group, 'split':split, 'scene_root':str(path.parent / name),
                    'frame':frame, 'extraction_sha256':binding['sha256']})
    identity = {'schema':'anvil-linked-sensor-gs-v1', 'original_index_manifest_sha256':sha(original_path),
        'builder_sha256':sha(Path(__file__)), 'sensor_builder_sha256':sha(Path(sensor.__file__)),
        'validator_sha256':sha(W/'trains/flywheel/validators.py'), 'label_map_sha256':sha(LABEL_MAP),
        'native_scorer_sha256':sha(W/'eval/g_family_qualification.py'),
        'frames_per_scan':FRAMES_PER_SCAN, 'scans':len(seen), 'frames':len(tasks),
        'selection':'First eight frames in the original source hash order; no target/model-score selection',
        'physical_scene_split':original['split_rule'], 'salt':salt,
        'source_manifests':original['source_manifests'],
        'referent_policy':'Prefer uniquely visible category names; otherwise give explicit interior click coordinates and task-local target IDs. Never ask the model to infer an internal source instance ID.',
        'depth_policy':'Existing registered-depth robust gap and disjoint depth-quartile criteria',
        'model_calls':0, 'new_annotation_calls':0}
    return tasks, identity


def frame_candidate(task):
    import cv2
    import numpy as np
    from trains.flywheel.contracts import context_for, grounding, task_columns
    from trains.flywheel.validators import validate, validate_structural
    cv2.setNumThreads(1)
    sensor = sensor_module()
    scene_root = Path(task['scene_root']).resolve()
    files = {}
    for name in ['color.png', 'depth_native.png', 'instance.png', 'semantic.png',
                 'intrinsic_color.txt', 'intrinsic_depth.txt', 'extrinsic_color.txt', 'extrinsic_depth.txt']:
        binding = task['frame']['outputs'][name]
        path = (scene_root / binding['path']).resolve()
        assert path.is_relative_to(scene_root) and path.stat().st_size == binding['size_bytes']
        assert sha(path) == binding['sha256'], f'Source bytes changed: {path}'
        files[name] = {'path':str(path), 'sha256':binding['sha256']}
    arrays = {name:sensor.load_png(Path(files[name]['path']))[0]
              for name in ['color.png', 'depth_native.png', 'instance.png', 'semantic.png']}
    rgb, depth = arrays['color.png'], arrays['depth_native.png']
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    assert depth.ndim == 2 and depth.dtype == np.uint16
    assert arrays['instance.png'].shape == arrays['semantic.png'].shape == rgb.shape[:2]
    matrices = {name:sensor.read_matrix(Path(files[name+'.txt']['path']))
                for name in ['intrinsic_color','intrinsic_depth','extrinsic_color','extrinsic_depth']}
    registered = sensor.register_depth_to_color(depth, matrices['intrinsic_depth'], matrices['extrinsic_depth'],
        matrices['intrinsic_color'], matrices['extrinsic_color'], rgb.shape[:2])
    aggregation_path = SCANNET/'scans'/task['scene_id']/(task['scene_id']+'.aggregation.json')
    aggregation, aggregation_sha = sensor.load_aggregation(aggregation_path, task['scene_id'], allow_bare_scene_id=True)
    accepted, rejected = sensor.instance_statistics(arrays['instance.png'], arrays['semantic.png'], registered,
        depth.shape, label_maps(), aggregation)
    visible = collections.Counter(aggregation.get(int(i), '').casefold() for i in np.unique(arrays['instance.png']) if i > 0)
    legal = [item for item in accepted if validate('bbox',
        [{'label':item.aggregation_label,'bbox_2d':list(item.bbox_2d)}])[0]]
    unique = [item for item in legal if visible[item.aggregation_label.casefold()] == 1]
    image_sha = files['color.png']['sha256']
    pixel_sha = hashlib.sha256(str(rgb.shape).encode()+b'\0'+rgb.tobytes()).hexdigest()
    result = {k:task[k] for k in ['id','scene_id','scene_group','split','extraction_sha256']}
    result.update(frame_index=task['frame']['frame_index'], files=files,
        aggregation={'path':str(aggregation_path),'sha256':aggregation_sha},
        source_rejections=dict(rejected), accepted_instances=len(accepted), unique_referents=len(unique),
        image_sha256=image_sha,decoded_rgb_sha256=pixel_sha)
    pair = sensor.choose_pair(unique)
    interface = 'unique_visible_labels'
    if pair is None:
        pair = sensor.choose_pair(legal)
        interface = 'point_referred_entities'
    if pair is None:
        return {**result, 'accepted':False, 'reason':'no_native_separated_pair'}
    ordered = sorted([pair.nearer, pair.farther], key=lambda item:hashlib.sha256(
        f'{image_sha}\0{item.instance_id}'.encode()).hexdigest())
    names = {}; referents = []
    for number,item in enumerate(ordered,1):
        name = item.aggregation_label
        spec = {'label':name}
        if interface == 'point_referred_entities':
            mask = (arrays['instance.png'] == item.instance_id).astype(np.uint8)
            distance = cv2.distanceTransform(mask,cv2.DIST_L2,5)
            y,x = np.unravel_index(int(np.argmax(distance)),distance.shape)
            point = [round(1000*x/mask.shape[1]),round(1000*y/mask.shape[0])]
            qx,qy = round(point[0]*mask.shape[1]/1000),round(point[1]*mask.shape[0]/1000)
            if distance[y,x]<3 or not (0<=qy<mask.shape[0] and 0<=qx<mask.shape[1] and mask[qy,qx]):
                return {**result,'accepted':False,'reason':'no_robust_point_reference'}
            name += f' target {number}'
            spec = {'label':name,'interior_click_2d':point}
        names[item.instance_id] = name; referents.append(spec)
    ground = [{'label':names[item.instance_id], 'bbox_2d':list(item.bbox_2d)} for item in ordered]
    relation = [{'label':names[item.instance_id], 'rank':rank, 'relative_depth':float(rank-1)}
                for rank,item in enumerate([pair.nearer,pair.farther],1)]
    gs = [grounding(item['label'],item['bbox_2d']) for item in ground]
    contexts = [context_for('detection',gs),context_for('depth',gs,{'relation_mode':'pairwise'})]
    for form,payload,context in zip(['detection','depth'],[ground,relation],contexts):
        for check in [validate_structural(form,payload), validate(form,payload,context)]:
            if not check[0]:return {**result, 'accepted':False,'reason':form+':'+str(check[1])}
    g_text = '<GROUND>'+json.dumps(ground)+'</GROUND>'
    s_text = '<RELATE>'+json.dumps(relation)+'</RELATE>'
    text = g_text+s_text
    query = ('<image>Localize the specified visible objects '+json.dumps(referents)+
        ' with labelled bounding boxes on the 0–1000 image coordinate scale. '
        'An interior click, when given, identifies an object; predict its full visible extent. '
        'Then use exactly those grounded entities to rank their camera depths from nearest to farthest. '
        'Write <GROUND>[{"label":...,"bbox_2d":[...]}]</GROUND>'
        '<RELATE>[{"label":...,"rank":...,"relative_depth":...}]</RELATE>. '
        'Use ranks 1 and 2 and ordinal relative_depth values 0.0 and 1.0. Return only these two stages.')
    row = {'sample_id':'scannet-linked-gs-'+task['id'], 'images':[files['color.png']['path']],
        'messages':[{'role':'user','content':query},{'role':'assistant','content':text}],
        **task_columns('depth',contexts[1],source='scannet_unique_rgbd_linked',scene=task['scene_id'],image_id=image_sha),
        'gsa_payload':relation, 'gsa_stages':['G','S'], 'gsa_stage_forms':['detection','depth'],
        'gsa_edges':[{'upstream':'G','downstream':'S','consumed_field':'label'}],
        'gsa_stage_contexts':contexts, 'gsa_references':[ground,relation],
        'gsa_stage_span_unit':'reference_characters',
        'gsa_stage_spans':[{'start':0,'end':len(g_text)},{'start':len(g_text),'end':len(text)}]}
    row['gsa_meta'].update(split=task['split'],scene_group=task['scene_group'],
        label_provenance='official_instance_projection_and_registered_depth',
        teacher_eligible=False,linked_causal_trace=True,source_frame_id=task['id'])
    row['gsa_meta'].update(referent_interface=interface,referent_spec=referents)
    row['gsa_meta']['decoded_rgb_sha256']=pixel_sha
    return {**result,'accepted':True,'row':row,'pair':asdict(pair),
        'visible_name_counts':{item.aggregation_label:visible[item.aggregation_label.casefold()] for item in ordered}}


def controlled_calibration(candidates):
    """Fit a band on dev-only exact references and known semantic counterfactuals.

    This calibrates the finite controlled family, not general judge reliability.
    It never uses qualification/test targets or model predictions.
    """
    from trains.mopd.anvil_runtime import reference_geometry_evaluator
    from trains.flywheel.validators import validate_structural
    measurements=[]
    for candidate in candidates:
        row=candidate['row']
        if row['gsa_meta']['split']!='dev':continue
        g,s=row['gsa_references']
        for stage,form,positive,upstream in [('G','detection',g,{}),('S','depth',s,{'G':g})]:
            negative=deepcopy(positive)
            if stage=='G':
                # One hallucinated identity is a known semantic error while
                # preserving valid JSON, cardinality and source geometry.
                negative[0]['label']='unrequested_object'
            else:
                for item in negative:item['rank']=3-item['rank'];item['relative_depth']=1-item['relative_depth']
            assert validate_structural(form,negative)[0]
            scores=[reference_geometry_evaluator(row=row,stage=stage,form=form,payload=value,upstream=upstream,reference=positive)
                    for value in [positive,negative]]
            assert scores[0]==1.0 and scores[1] is not None
            measurements.append({'sample_id':row['sample_id'],'stage':stage,'positive':scores[0],'counterfactual':scores[1]})
    settings={}
    for stage in ['G','S']:
        population=[v for v in measurements if v['stage']==stage]
        assert population, 'No development calibration cases'
        low=min(v['positive'] for v in population);high=max(v['counterfactual'] for v in population)
        assert high<low, 'Controlled positive/negative scores are not separated'
        settings[stage]={'threshold':(low+high)/2,'uncertainty_band':(low-high)/4,
            'minimum_positive':low,'maximum_controlled_negative':high,'dev_examples':len(population)}
    return {'schema':'anvil-dev-controlled-routing-band-v1','settings':settings,'measurements':measurements,
        'scorer_sha256':hashlib.sha256(inspect.getsource(reference_geometry_evaluator).encode()).hexdigest(),
        'scope':'Exact references vs one unrequested G identity or reversed S order on dev frames only; not human or general semantic-judge precision.'}


def timed_frame_candidate(task):
    started=time.monotonic()
    result=frame_candidate(task)
    result['worker_elapsed_seconds']=time.monotonic()-started
    return result


def finalize(tasks):
    from trains.mopd.anvil_runtime import ContractV2BatchRouter
    from trains.flywheel.contracts import training_instruction
    results=[read_binding(ROOT/'frames'/(t['id']+'.json')) for t in tasks]
    candidates=[r for r in results if r['accepted']]
    # Pixel-identical frames are one example, even when separate scan records exist.
    unique={};duplicates=[]
    for r in sorted(candidates,key=lambda x:x['id']):
        key=r['decoded_rgb_sha256']
        if key in unique:
            assert unique[key]['split']==r['split'], 'Image bytes cross physical-scene splits'
            duplicates.append(r['id']);continue
        unique[key]=r
    candidates=list(unique.values())
    calibration=controlled_calibration(candidates);save(ROOT/'calibration.json',calibration)
    linked=[];atomic=collections.defaultdict(list);router=ContractV2BatchRouter()
    for candidate in candidates:
        row=deepcopy(candidate['row'])
        row['gsa_thresholds']=[calibration['settings'][s]['threshold'] for s in ['G','S']]
        row['gsa_uncertainty_bands']=[calibration['settings'][s]['uncertainty_band'] for s in ['G','S']]
        row['gsa_meta']['routing_calibration_sha256']=sha(ROOT/'calibration.json')
        if row['gsa_meta']['split']=='train':router._record(row)
        linked.append(row)
        for stage,form,payload,context in zip(['G','S'],['detection','depth'],row['gsa_references'],row['gsa_stage_contexts']):
            item=deepcopy(row)
            for key in ['gsa_stages','gsa_stage_forms','gsa_edges','gsa_stage_contexts','gsa_references',
                        'gsa_stage_span_unit','gsa_stage_spans','gsa_thresholds','gsa_uncertainty_bands']:
                item.pop(key)
            item.update(sample_id=row['sample_id']+'-'+stage,gsa_form=form,gsa_family=stage,gsa_payload=payload,gsa_context=context)
            item['gsa_meta'].update(form=form,family=stage,linked_causal_trace=False)
            instruction=(('Localize the specified visible objects '+json.dumps(row['gsa_meta']['referent_spec'])+
                ' with labelled bounding boxes on the 0–1000 image coordinate scale. '
                'An interior click, when given, identifies an object; predict its full visible extent. Return only a JSON array.')
                if stage=='G' else training_instruction(form,context))
            item['messages']=[{'role':'user','content':'<image>'+instruction},
                              {'role':'assistant','content':json.dumps(payload)}]
            atomic[(row['gsa_meta']['split'],stage)].append(item)
    files={}
    for split in ['train','dev']:
        selected=[r for r in linked if r['gsa_meta']['split']==split]
        groups={r['gsa_meta']['scene_group'] for r in selected}
        other={r['gsa_meta']['scene_group'] for r in linked if r['gsa_meta']['split']!=split}
        assert not groups&other
        for name,data in [(split+'.jsonl',selected)]+[(f'{split}_{stage}.jsonl',atomic[(split,stage)]) for stage in ['G','S']]:
            path=ROOT/name;commit_bytes(path,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data).encode())
            files[name]={'sha256':sha(path),'records':len(data)}
    report={'schema':'anvil-linked-sensor-gs-v1','binding_sha256':sha(ROOT/'binding.json'),
        'attempted_frames':len(tasks),'accepted_before_image_dedup':sum(r['accepted'] for r in results),
        'duplicate_source_frames':duplicates,'linked_records':len(linked),'files':files,
        'physical_scene_groups':len({r['gsa_meta']['scene_group'] for r in linked}),
        'referent_interfaces':dict(collections.Counter(r['gsa_meta']['referent_interface'] for r in linked)),
        'rejections':dict(collections.Counter(r['reason'] for r in results if not r['accepted'])),
        'calibration_sha256':sha(ROOT/'calibration.json'),'model_calls':0,'new_annotation_calls':0,
        'use_policy':{'local_programmatic_training':True,'benchmark_ground_truth':False,'teacher_eligible':False,
            'human_semantic_precision_established':False,'redistribution_eligible':False},
        'scope':'Two-stage G→S targets plus atomic views over the same images; no new Action targets or proof of teacher superiority.'}
    save(ROOT/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ['attempted_frames','linked_records','physical_scene_groups','files','rejections']}),flush=True)


def prepare():
    started=time.monotonic()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks,binding=source_tasks()
        if (ROOT/'binding.json').exists():assert read_binding(ROOT/'binding.json')==binding, 'Linked sensor source/code binding changed'
        else:save(ROOT/'binding.json',binding)
        if (ROOT/'manifest.json').exists():
            manifest=read_binding(ROOT/'manifest.json')
            for name,value in manifest['files'].items():assert sha(ROOT/name)==value['sha256']
            assert sha(ROOT/'calibration.json')==manifest['calibration_sha256']
            print('Linked sensor corpus already complete; all output hashes match.',flush=True)
            return
        (ROOT/'frames').mkdir(exist_ok=True)
        missing=[t for t in tasks if not (ROOT/'frames'/(t['id']+'.json')).exists()]
        with mp.get_context('spawn').Pool(8,maxtasksperchild=64) as pool:
            for n,result in enumerate(pool.imap_unordered(timed_frame_candidate,missing,chunksize=1),1):
                save(ROOT/'frames'/(result['id']+'.json'),result)
                if n%25==0:
                    save(ROOT/'progress.json',{'processed':len(tasks)-len(missing)+n,'total':len(tasks),'updated':time.time()})
                    print('LINKED_SENSOR',len(tasks)-len(missing)+n,'/',len(tasks),flush=True)
        finalize(tasks)
        manifest=read_binding(ROOT/'manifest.json')
        manifest['construction_timing']={'current_invocation_wall_seconds':time.monotonic()-started,
            'frames_processed_this_invocation':len(missing),'frames_recovered':len(tasks)-len(missing),
            'cpu_workers':8,'aggregate_frame_worker_seconds':sum(read_binding(ROOT/'frames'/(t['id']+'.json'))['worker_elapsed_seconds'] for t in tasks),
            'gpu_seconds':0,'model_calls':0,'new_annotation_calls':0}
        save(ROOT/'manifest.json',manifest)


if __name__=='__main__':
    prepare()
