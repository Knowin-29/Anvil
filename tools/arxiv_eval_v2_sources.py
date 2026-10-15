#!/usr/bin/env python3
"""Fetch pinned source evidence; never export evaluation labels or score models."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time


SOURCES = (
    ("allenai/PRISM", "34cfb7f6d98e019a2251cdc40c8079fbf0cccba1", "train.csv",
     168211184, "84a7d84709977a54df1c69f1d53f70d6bcb0ed4711a9afc387040e7e122692d3"),
    ("allenai/PRISM", "34cfb7f6d98e019a2251cdc40c8079fbf0cccba1", "PRISM-train/chunk_1_of_100.tar",
     2606090240, "3e0b53d4e73e28f0fcd6a05597ea22eeb109bf3deaed2d3fe0f2713b23521541"),
    ("tyb197/RoboAfford", "8355115e4d326a436360c5e0055e6bb724b8fc81", "object_affordance_prediction_absxy_561K.json",
     304896352, "6f809b7604169d4f90cb98e322b56e7b05f758fe8ba60244b4932416ddd6ab85"),
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root, spec):
    repo, revision, name, size, digest = spec
    target = root / repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
    start = time.monotonic()
    if not target.exists():
        subprocess.run(["curl", "--http1.1", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "4", "--retry-all-errors", "--connect-timeout", "30", "--continue-at", "-",
                        "--output", str(partial), url], check=True)
        if partial.stat().st_size != size or sha(partial) != digest:
            raise ValueError(f"Pinned source size/hash mismatch: {repo}/{name}")
        partial.rename(target)
    elif target.stat().st_size != size or sha(target) != digest:
        raise ValueError(f"Existing source differs from pinned metadata: {target}")
    receipt = {"repo": repo, "revision": revision, "file": name, "path": str(target),
               "bytes": size, "sha256": digest, "state": "download_integrity_verified",
               "wall_seconds": time.monotonic() - start, "benchmark_ready": False}
    print(json.dumps(receipt), flush=True)
    return receipt


def audit_source_fields(root, eligible_index=None):
    """Inspect available fields without converting them into evaluation targets."""
    run = root / "source_prepare"
    raw = json.loads((run / "tyb197/RoboAfford/object_affordance_prediction_absxy_561K.json").read_text())
    eligible_coco = None
    if eligible_index is not None:
        with sqlite3.connect(f'file:{Path(eligible_index).resolve()}?mode=ro', uri=True) as db:
            eligible_coco = {Path(r[0]).name for r in db.execute("SELECT path FROM images WHERE source='coco_train2017'")}
    affordance = {"source_image_rows": len(raw),
                  "source_qa_pairs": sum(len(r['conversations']) // 2 for r in raw),
                  "indexed_eligible_image_rows": (sum(Path(r['image']).name in eligible_coco for r in raw)
                                                  if eligible_coco is not None else None),
                  "native_schema_complete": False,
                  "remaining": "Source answers localize functional objects. Do not treat object points as verified interaction/contact points; native contact-field mapping and independent checking are still required."}
    with (run / "allenai/PRISM/train.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        scene_counts = Counter(r['scene_path'] for r in reader)
    prism = {"metadata_columns": columns, "source_task_rows": sum(scene_counts.values()),
             "physical_scene_references": len(scene_counts), "native_schema_complete": False,
             "remaining": "Freeze held-out physical scenes, jaw-axis projection and camera/world approach convention, then validate the native scorer; raw grasp poses alone are not the complete existing DoF schema."}
    sys.path.insert(0, str(root / "inspect_deps"))
    import h5py
    import numpy as np
    archive = run / "allenai/PRISM/PRISM-train/chunk_1_of_100.tar"
    with tarfile.open(archive) as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith('.hdf5')), key=lambda m: m.name)
        if not members or members[0].size > 512 * 1024 * 1024:
            raise ValueError('No bounded representative HDF5 source member')
        with tar.extractfile(members[0]) as stream:
            data = stream.read()
        with h5py.File(io.BytesIO(data), 'r') as h5:
            view_key = sorted(h5.keys())[0]
            view = h5[view_key]
            obs_key = sorted(k for k in view if k.startswith('obs_'))[0]
            obs = view[obs_key]
            pose = obs['grasp_pose'][:]
            camera = view['cam_params'][:]
            pose_valid = bool(pose.shape == (4, 4) and np.isfinite(pose).all()
                and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
                and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                and np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4))
            prism['raw_probe'] = {
                'archive_scene_members': len(members), 'member': members[0].name,
                'view': view_key, 'observation': obs_key,
                'view_fields': list(view.keys()), 'observation_fields': list(obs.keys()),
                'rgb_shape': list(view['rgb'].shape), 'xyz_shape': list(view['xyz'].shape),
                'seg_shape': list(view['seg'].shape), 'intrinsics_shape': list(camera.shape),
                'grasp_pose_shape': list(pose.shape), 'rigid_grasp_pose_valid': pose_valid,
                'h5py_version': h5py.__version__, 'numpy_version': np.__version__,
                'scope': 'One fixed first-member probe, not full-source target validation.'}
    return {'RoboAfford': affordance, 'PRISM': prism, 'benchmark_ready': False}


def main(root, resume=False, eligible_index=None):
    run = root / "source_prepare"
    if resume:
        original = json.loads((run / "manifest.json").read_text())
        if not original.get("errors"):
            raise ValueError("No failed input fetch to resume")
        if (run / "retry_manifest.json").exists():
            raise FileExistsError("Retry receipt already exists; preserve it")
    else:
        run.mkdir()
    receipts, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as workers:
        pending = {workers.submit(fetch, run, spec): spec for spec in SOURCES}
        for future in as_completed(pending):
            spec = pending[future]
            try:
                receipts.append(future.result())
            except Exception as exc:
                errors.append({"repo": spec[0], "file": spec[2], "error": str(exc)})
                print(json.dumps(errors[-1]), flush=True)
    audit = None
    if not errors:
        try:
            audit = audit_source_fields(root, eligible_index)
        except Exception as exc:
            errors.append({'stage': 'source_field_audit', 'error': str(exc)})
    result = {
        "state": "source_files_ready_for_field_audit" if not errors else "source_fetch_incomplete",
        "files": receipts, "errors": errors, "field_audit": audit, "benchmark_ready": False,
        "model_calls": 0, "new_evaluation_records": 0,
        "requirements": {
            "depth_order": "Sensor depth and instance masks; exclude all previously trained/development physical scenes.",
            "affordance": "Verify functional labels, region and contact fields; object interiors alone are not affordance truth.",
            "dof": "Inspect PRISM raw grasp_pose and camera calibration; document jaw/approach conventions before conversion.",
            "action_cot": "ALFRED expert plans and PDDL state transitions; old binary validity labels are not complete native action targets."
        },
        "policy": "All eleven supported forms and frozen scene/image partitions and native scorers are required before unified evaluation; no score-based selection."
    }
    with (run / ("retry_manifest.json" if resume else "manifest.json")).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if errors:
        raise SystemExit(1)


def materialize_prism(root, output):
    """Recover calibrated source assets, retaining raw poses rather than guessed DoF labels."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(root / 'inspect_deps'))
    import h5py
    run = output / 'prism_raw'
    run.mkdir()
    archive = root / 'source_prepare' / SOURCES[1][0] / SOURCES[1][2]
    if sha(archive) != SOURCES[1][4]:
        raise ValueError('PRISM source archive changed')
    metadata = root / 'source_prepare' / SOURCES[0][0] / SOURCES[0][2]
    if sha(metadata) != SOURCES[0][4]:
        raise ValueError('PRISM source metadata changed')
    counts = Counter()
    scene_ids = set()
    with (run / 'views.jsonl').open('x') as view_out, (run / 'grasps.jsonl').open('x') as grasp_out:
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.hdf5'):
                    continue
                if not re.fullmatch(r'[a-f0-9]{32}[.]hdf5', member.name) or member.size > 512 * 1024 * 1024:
                    raise ValueError(f'Unexpected PRISM member: {member.name}')
                scene = Path(member.name).stem
                if scene in scene_ids:
                    raise ValueError('Duplicate physical scene')
                scene_ids.add(scene)
                scene_dir = run / scene
                scene_dir.mkdir()
                target = scene_dir / member.name
                with tar.extractfile(member) as src, target.open('xb') as dest:
                    shutil.copyfileobj(src, dest)
                digest = sha(target)
                with h5py.File(target, 'r') as h5:
                    for key in sorted(h5):
                        if not re.fullmatch(r'view_[0-9]+', key):
                            raise ValueError('Unexpected PRISM view')
                        view = h5[key]
                        rgb = view['rgb'][:]
                        h, w = rgb.shape[:2]
                        if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
                            raise ValueError('Invalid RGB source')
                        if view['xyz'].shape != (h, w, 3) or view['seg'].shape != (h, w):
                            raise ValueError('Unaligned PRISM source arrays')
                        camera = view['cam_params'][:]
                        if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
                            raise ValueError('Invalid intrinsics')
                        image = scene_dir / (key + '.png')
                        Image.fromarray(rgb).save(image)
                        names = [v.decode() if isinstance(v, bytes) else str(v) for v in view['object_names'][:]]
                        view_out.write(json.dumps({'scene_id': scene, 'view_id': key, 'source_split': 'train',
                            'image': str(image), 'image_sha256': sha(image), 'raw_hdf5': str(target),
                            'raw_sha256': digest, 'object_names': names, 'intrinsics': camera.tolist(),
                            'view_pose': view['view_pose'][:].tolist(), 'width': w, 'height': h,
                            'partition': 'unassigned_source_only'}) + '\n')
                        counts['views'] += 1
                        for obs_key in sorted(k for k in view if k.startswith('obs_')):
                            obs = view[obs_key]
                            pose, point, px = obs['grasp_pose'][:], obs['grasp_point'][:], obs['grasp_point_px'][:]
                            rigid = bool(pose.shape == (4, 4) and np.isfinite(pose).all()
                                and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
                                and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                                and np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4))
                            projection_ok = False
                            if point.shape == (3,) and px.shape == (2,) and np.isfinite(point).all() and np.isfinite(px).all() and point[2] > 0:
                                projected = camera @ point
                                projection_ok = bool(np.allclose(projected[:2] / projected[2], px, atol=0.1, rtol=0))
                            annotation = obs['annot'][()]
                            if isinstance(annotation, bytes): annotation = annotation.decode()
                            grasp_out.write(json.dumps({'scene_id': scene, 'view_id': key, 'obs_id': obs_key,
                                'raw_hdf5': str(target), 'raw_sha256': digest, 'image': str(image),
                                'grasp_pose_camera': pose.tolist(), 'grasp_point_camera': point.tolist(),
                                'grasp_point_px': px.tolist(), 'annotation_yaml': str(annotation),
                                'rigid_pose_valid': rigid, 'projection_consistent': projection_ok,
                                'native_schema_complete': False}) + '\n')
                            counts['observations'] += 1
                            counts['rigid_pose_valid'] += rigid
                            counts['projection_consistent'] += projection_ok
                counts['physical_scenes'] += 1
                print(json.dumps({'PRISM': dict(counts)}), flush=True)
    with metadata.open(newline='') as stream, (run / 'tasks.jsonl').open('x') as out:
        for task in csv.DictReader(stream):
            if task['scene_id'] in scene_ids:
                out.write(json.dumps(task) + '\n')
                counts['source_tasks'] += 1
    result = {'state': 'raw_source_materialized', 'counts': dict(counts), 'archive_sha256': SOURCES[1][4],
              'metadata_sha256': SOURCES[0][4], 'benchmark_ready': False, 'new_model_calls': 0,
              'native_records_exported': 0, 'partition': 'unassigned; no training or evaluation use',
              'remaining': 'Native contact sub-part, jaw-axis and approach mapping; physical-scene exclusion and scoring freeze.',
              'files': {n: sha(run / n) for n in ['views.jsonl', 'grasps.jsonl', 'tasks.jsonl']}}
    (run / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')


def materialize_alfred(output, sources, archive, bsdtar):
    """Recover every frame linked to interaction steps; choose no temporal target yet."""
    from PIL import Image
    run = output / 'alfred_interactions'
    run.mkdir()
    catalog = json.loads((sources / 'archive_members.json').read_text())
    st = archive.stat()
    identity = {'path': str(archive), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
    if catalog['archive'] != identity:
        raise ValueError('ALFRED archive binding changed')
    packets = [json.loads(s) for s in (sources / 'expert_sources.jsonl').read_text().splitlines() if s]
    members, interactions = set(), []
    for packet in packets:
        trajectory = Path(packet['trajectory']['path'])
        if sha(trajectory) != packet['trajectory']['sha256']:
            raise ValueError('ALFRED trajectory changed')
        relative = trajectory.relative_to(sources / 'source').parent
        doc = json.loads(trajectory.read_text())
        for low, action in enumerate(doc['plan']['low_actions']):
            args = action.get('discrete_action', {}).get('args', {})
            if not args.get('mask') or not args.get('bbox'):
                continue
            frames = []
            for item in doc['images']:
                if item['low_idx'] != low:
                    continue
                member = str(relative / 'raw_images' / Path(item['image_name']).with_suffix('.jpg').name)
                if not re.fullmatch(r'full_2[.]1[.]0/(valid_seen|valid_unseen)/[^/]+/[^/]+/raw_images/[0-9]+[.]jpg', member):
                    raise ValueError('Invalid ALFRED member path')
                frames.append({'member': member, 'metadata': item})
                members.add(member)
            interactions.append({'task_id': packet['task_id'], 'scene_id': packet['scene'], 'split': packet['split'],
                'low_idx': low, 'high_idx': action['high_idx'], 'source_action': action,
                'trajectory': packet['trajectory'], 'frames': frames, 'temporal_target_verified': False})
    if not members:
        raise ValueError('No interaction frames')
    targets = run / 'members.txt'
    targets.write_text('\n'.join(sorted(members)) + '\n')
    raw = run / 'frames'
    raw.mkdir()
    print(json.dumps({'ALFRED': 'extracting', 'frames': len(members), 'interactions': len(interactions)}), flush=True)
    with (run / 'extract.log').open('x') as log:
        subprocess.run([str(bsdtar), '-x', '-q', '-m', '-f', str(archive), '-C', str(raw), '-T', str(targets)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    with (run / 'images.jsonl').open('x') as out:
        for name in sorted(members):
            image = raw / name
            if not image.is_file() or image.is_symlink() or not image.resolve().is_relative_to(raw.resolve()):
                raise ValueError(f'Invalid extracted source frame: {name}')
            with Image.open(image) as im:
                size = im.size
                im.verify()
            out.write(json.dumps({'member': name, 'path': str(image), 'sha256': sha(image), 'size': size}) + '\n')
    with (run / 'interactions.jsonl').open('x') as out:
        for record in interactions:
            out.write(json.dumps(record) + '\n')
    result = {'state': 'interaction_frames_materialized', 'archive': identity, 'frames': len(members),
              'interactions': len(interactions), 'trajectories': len(packets),
              'sources_sha256': sha(sources / 'expert_sources.jsonl'), 'benchmark_ready': False,
              'new_model_calls': 0, 'native_records_exported': 0,
              'remaining': 'Verify before/after timing, native action target and contact semantics; exclude previously used physical scenes.',
              'files': {n: sha(run / n) for n in ['members.txt', 'images.jsonl', 'interactions.jsonl']}}
    (run / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')


def prism_depth(root, output):
    """Construct unassigned camera-depth references, not a scored benchmark."""
    import itertools
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(root / 'inspect_deps'))
    import h5py
    from trains.flywheel.alignment_schema import Candidate, Scene, student_prompt, validate_candidate
    source = output / 'prism_raw'
    binding = json.loads((source / 'manifest.json').read_text())
    if sha(source / 'views.jsonl') != binding['files']['views.jsonl']:
        raise ValueError('PRISM view manifest changed')
    run = output / 'prism_depth'
    run.mkdir()
    design = {'source_manifest': str(source / 'manifest.json'), 'source_manifest_sha256': sha(source / 'manifest.json'),
        'source_definition': 'https://github.com/abhaybd/GraspMolmo/blob/73990b5c5121ac5ab5fd402453bfeeefd270d153/graspmolmo/datagen/generate_obs.py',
        'definition_lines': '84-97 camera-frame XYZ; 116-142 one-based instance names; 179-184 aligned outputs',
        'min_valid_pixels': 64, 'depth_quantiles': [.1, .5, .9], 'relative_interval_gap': .02,
        'measurement': 'visible-surface camera-Z; not object-centroid depth or Euclidean camera distance',
        'selection': 'one record per unordered visible-instance pair, every eligible source view; no predictions',
        'partition': 'unassigned_source_only', 'benchmark_ready': False, 'model_calls': 0}
    (run / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
    counts = Counter(); seen_raw = set(); ids = set(); used_images = set(); used_scenes = set()
    start = time.monotonic()
    with (run / 'records.jsonl').open('x') as out:
        for line in (source / 'views.jsonl').open():
            view = json.loads(line)
            raw = Path(view['raw_hdf5'])
            if str(raw) not in seen_raw:
                if sha(raw) != view['raw_sha256']: raise ValueError('PRISM raw HDF5 changed')
                seen_raw.add(str(raw))
            if sha(Path(view['image'])) != view['image_sha256']: raise ValueError('PRISM RGB changed')
            with h5py.File(raw, 'r') as h5:
                data = h5[view['view_id']]
                xyz, seg = data['xyz'][:], data['seg'][:]
            h, w = seg.shape
            if xyz.shape != (h, w, 3): raise ValueError('RGB/XYZ/instance shapes differ')
            entities = []
            for instance_id, name in enumerate(view['object_names'], 1):
                if name.lower() in {'table', 'floor', 'background', 'ground'}: continue
                mask = (seg == instance_id) & np.isfinite(xyz).all(axis=-1) & (xyz[:, :, 2] > 0)
                yy, xx = np.nonzero(mask)
                if len(xx) < 64:
                    counts['small_or_invalid_instances'] += 1; continue
                if xx.max() <= xx.min() or yy.max() <= yy.min(): continue
                q = np.quantile(xyz[:, :, 2][mask], [.1, .5, .9]).tolist()
                entities.append({'entity_id': f'instance_{instance_id}', 'label': f'visible_object_{instance_id}',
                    'bbox': [1000.*xx.min()/w, 1000.*yy.min()/h, 1000.*(xx.max()+1)/w, 1000.*(yy.max()+1)/h],
                    'source_name': name, 'depth_quantiles': q, 'valid_pixels': len(xx)})
            for left, right in itertools.combinations(entities, 2):
                counts['pairs_considered'] += 1
                near, far = sorted([left, right], key=lambda e: e['depth_quantiles'][1])
                gap = far['depth_quantiles'][0] - near['depth_quantiles'][2]
                if gap <= .02 * near['depth_quantiles'][1]:
                    counts['overlapping_or_close_depth_intervals'] += 1; continue
                identity = ':'.join([view['scene_id'], view['view_id'], left['entity_id'], right['entity_id'], 'depth_order'])
                rid = hashlib.sha256(identity.encode()).hexdigest()
                if rid in ids: raise ValueError('Duplicate depth target')
                subject, obj = (near, far) if int(rid[-1], 16) % 2 else (far, near)
                predicate = 'nearer' if subject is near else 'farther'
                requested = [subject['entity_id'], obj['entity_id']]
                relation = {'task': 'relative_depth', 'subject_id': requested[0], 'object_id': requested[1],
                            'predicate': predicate, 'frame': 'camera', 'truth': True}
                scene = Scene.model_validate({'source': 'PRISM', 'scene_id': view['scene_id']+':'+view['view_id'],
                    'group_id': 'PRISM:'+view['scene_id'], 'images': [view['image']], 'source_split': 'train',
                    'training_approved': False,
                    'entities': [{k:e[k] for k in ['entity_id','label','bbox']} for e in [subject,obj]],
                    'evidence': [{'evidence_id': 'camera_depth', 'kind': 'derived_geometry',
                        'description': 'Aligned rendered camera-Z and visible instance masks with disjoint central depth intervals.',
                        'entity_ids': requested, 'relation': relation}],
                    'tasks': [{'task_id': 'ground', 'form': 'detection', 'native_form': 'detection',
                        'question': 'Locate both identified visible instances.', 'entity_ids': requested},
                        {'task_id': 'depth', 'form': 'spatial', 'native_form': 'depth_order', 'spatial_task': 'relative_depth',
                         'question': f'Is {requested[0]} nearer or farther than {requested[1]} in visible-surface depth along the camera axis?',
                         'entity_ids': requested}]})
                candidate = Candidate.model_validate({'task_id': 'depth', 'form': 'spatial',
                    'answer': [{**relation, 'evidence_ids': ['camera_depth']}],
                    'reasoning_trace': [{'evidence_ids': ['camera_depth'], 'rule': 'Compare disjoint central camera-depth intervals.',
                                         'conclusion': f'{requested[0]} is {predicate} than {requested[1]}.'}]})
                task = scene.tasks[1]
                validate_candidate(candidate, scene, task, [])
                reversed_candidate = candidate.model_dump()
                reversed_candidate['answer'][0]['predicate'] = 'farther' if predicate == 'nearer' else 'nearer'
                try: validate_candidate(Candidate.model_validate(reversed_candidate), scene, task, [])
                except ValueError: counts['reversed_reference_rejected'] += 1
                else: raise ValueError('Source-reversed reference incorrectly admitted')
                prompt = student_prompt(scene, task)
                if 'depth_quantiles' in prompt or 'camera_depth' in prompt: raise ValueError('Reference evidence entered prompt')
                packet = {'id': rid, 'native_form': 'depth_order', 'partition': 'unassigned_source_only',
                    'image_sha256': view['image_sha256'], 'scene': scene.model_dump(), 'reference': candidate.model_dump(),
                    'student_prompt': prompt, 'source_binding': {'hdf5': str(raw), 'sha256': view['raw_sha256'],
                        'view_id': view['view_id'], 'instances': [subject, obj], 'interval_gap': gap},
                    'verification': 'source_rendered_depth_and_deterministic_checks',
                    'strong_verifier_reviewed': False, 'human_reviewed': False}
                out.write(json.dumps(packet)+'\n')
                ids.add(rid); used_images.add(view['image']); used_scenes.add(view['scene_id'])
                counts['records'] += 1; counts[predicate] += 1
            counts['views_checked'] += 1
            if counts['views_checked'] % 100 == 0: print(json.dumps(dict(counts)), flush=True)
    result = {'state': 'native_source_candidates_validated_partition_pending', 'counts': dict(counts),
        'unique_images': len(used_images), 'physical_scenes': len(used_scenes), 'wall_seconds': time.monotonic()-start,
        'records_sha256': sha(run/'records.jsonl'), 'design_sha256': sha(run/'design.json'),
        'benchmark_ready': False, 'training_records_added': 0, 'evaluation_records_added': 0, 'model_calls': 0,
        'remaining': 'Physical-scene exclusion against all training/history, 11-form split and scoring freeze; no model evaluated.'}
    (run/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eligible-index", type=Path, help="Optional frozen candidate SQLite image index for coverage audit")
    parser.add_argument('--mode', choices=['fetch', 'prism', 'alfred', 'prism-depth'], default='fetch')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--alfred-sources', type=Path)
    parser.add_argument('--alfred-archive', type=Path)
    parser.add_argument('--bsdtar', type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'fetch':
        main(args.root, args.resume, args.eligible_index)
    else:
        if args.output is None: parser.error('--output is required')
        args.output.mkdir(parents=True, exist_ok=True)
        try:
            if args.mode == 'prism': materialize_prism(args.root, args.output)
            elif args.mode == 'prism-depth': prism_depth(args.root, args.output)
            else:
                if not all([args.alfred_sources, args.alfred_archive, args.bsdtar]):
                    parser.error('ALFRED sources, archive and bsdtar are required')
                materialize_alfred(args.output, args.alfred_sources, args.alfred_archive, args.bsdtar)
        except Exception as exc:
            with (args.output / (args.mode + '.failed.json')).open('x') as out:
                json.dump({'error': str(exc), 'benchmark_ready': False}, out)
            raise
