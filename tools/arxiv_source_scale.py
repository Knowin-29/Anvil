#!/usr/bin/env python3
"""Source-backed spatial records. No model votes, synthetic truth, or human-review claims."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps

cv2.setNumThreads(1)
SEED = 'Anvil-Eval-v2-source-split-20260924'
FORMS = ('bbox','detection','mask','point','depth_order','metric_comparison',
         'direction','relation_graph','affordance','dof','action_cot')
COCO_IMAGE_BYTES = 19336861798
COCO_IMAGE_SHA256 = '69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'))


def save(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def fingerprint(path):
    raw = Path(path).read_bytes()
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert('RGB')
    gray = image.convert('L')
    small = np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
    low = cv2.dct(small)[:8,:8].reshape(-1)
    ph = sum(int(v > np.median(low[1:])) << i for i,v in enumerate(low))
    small = np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS))
    dh = sum(int(v) << i for i,v in enumerate((small[:,:-1]>small[:,1:]).reshape(-1)))
    return {'sha256':digest(raw), 'decoded_rgb_sha256':digest(
            f'{image.width}x{image.height}:RGB:'.encode()+image.tobytes()),
            'width':image.width,'height':image.height,'phash64':f'{ph:016x}','dhash64':f'{dh:016x}'}


class Fingerprints:
    """Five-block index: every Hamming-distance <=4 pair shares one block."""
    def __init__(self):
        self.exact=set(); self.buckets=defaultdict(list)

    def add(self, r):
        self.exact.update(r[k] for k in ('sha256','decoded_rgb_sha256') if r.get(k))
        if not r.get('phash64') or not r.get('dhash64'): return
        p,d=int(r['phash64'],16),int(r['dhash64'],16)
        for i in range(5): self.buckets[(i,(p>>(13*i))&8191)].append((p,d))

    def matches(self, r):
        if r['sha256'] in self.exact or r['decoded_rgb_sha256'] in self.exact: return True
        p,d=int(r['phash64'],16),int(r['dhash64'],16)
        return any((p^a).bit_count()<=4 and (d^b).bit_count()<=4
                   for i in range(5) for a,b in self.buckets.get((i,(p>>(13*i))&8191),()))


def load_exclusions():
    protected, trained = Fingerprints(), Fingerprints()
    bindings={}
    for key,index in [('ANVIL_EVAL_FINGERPRINTS',protected),('ANVIL_SOURCE_FINGERPRINTS',trained)]:
        path=Path(os.environ[key]); bindings[key]={'path':str(path),'sha256':sha(path)}
        for r in rows(path): index.add(r)
    seen=set()
    for key in ('ANVIL_BENCHG_DATA','ANVIL_P5_DEV','ANVIL_RETENTION_PANEL'):
        path=Path(os.environ[key]); bindings[key]={'path':str(path),'sha256':sha(path)}
        for r in rows(path):
            for image in r.get('images',[]) or [r.get('image_path')]:
                if image and image not in seen:
                    protected.add(fingerprint(image)); seen.add(image)
    return protected,trained,bindings


class Output:
    def __init__(self, root, source, protected, trained):
        self.root=root; self.source=source; self.protected=protected; self.trained=trained
        self.run=root/source; self.run.mkdir()
        self.db=sqlite3.connect(root/'source_index.sqlite',timeout=60)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS images(pixel TEXT PRIMARY KEY, payload TEXT NOT NULL, split TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, source TEXT, form TEXT, split TEXT);
        ''')
        # All new-source train images also exclude subsequent v2 candidates.
        # Existing v2 images conversely exclude subsequent train images.
        self.new_eval=Fingerprints()
        for payload,split in self.db.execute('SELECT payload,split FROM images'):
            (self.new_eval if split=='evaluation_candidate' else trained).add(json.loads(payload))
        self.files={s:(self.run/(s+'.jsonl')).open('x') for s in ('train','evaluation_candidate')}
        self.images=(self.run/'images.jsonl').open('x')
        self.counts=Counter(); self.skipped=Counter(); self.image_cache={}; self.start=time.monotonic()

    def image(self, path, width, height, identity):
        key=str(path)
        if key in self.image_cache:return self.image_cache[key]
        try: f=fingerprint(path)
        except (OSError,ValueError) as e:
            self.skipped['image_unreadable']+=1; self.image_cache[key]=None; return None
        if (f['width'],f['height'])!=(width,height):
            self.skipped['source_dimension_mismatch']+=1; self.image_cache[key]=None; return None
        if self.protected.matches(f):
            self.skipped['protected_eval_overlap']+=1; self.image_cache[key]=None; return None
        group=f['decoded_rgb_sha256']
        old=self.db.execute('SELECT split FROM images WHERE pixel=?',(group,)).fetchone()
        if old: split=old[0]
        else:
            proposed=int(digest((SEED+group).encode())[:16],16)/2**64 < .1
            split='evaluation_candidate' if proposed and not self.trained.matches(f) else 'train'
            if split=='train' and self.new_eval.matches(f):
                self.skipped['near_new_eval_overlap']+=1; self.image_cache[key]=None; return None
            self.db.execute('INSERT INTO images VALUES (?,?,?)',(group,encode(f),split))
            (self.new_eval if split=='evaluation_candidate' else self.trained).add(f)
        info={**f,'path':key,'source_image_id':str(identity),'source':self.source,
              'group_id':group,'group_unit':'decoded_image','split':split}
        self.images.write(encode(info)+'\n'); self.image_cache[key]=info
        return info

    def add(self, info, form, question, target, annotation_ids, rule, scope='visual'):
        assert form in FORMS and info['split'] in self.files
        # Equivalent targets/questions are deduplicated even across source IDs.
        identity=digest(encode([info['group_id'],form,question,target]).encode())
        changed=self.db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?)',
                      (identity,self.source,form,info['split'])).rowcount
        if not changed: self.skipped['duplicate_record']+=1; return
        record={'id':identity,'images':[info['path']],
          'messages':[{'role':'user','content':'<image>\n'+question},
                      {'role':'assistant','content':encode(target)}],
          'anvil_meta':{'schema':'anvil-source-native-v1','native_form':form,'source':self.source,
            'source_image_id':info['source_image_id'],'source_annotation_ids':annotation_ids,
            'group_id':info['group_id'],'split':info['split'],'coordinate_scale':1000,
            'candidate_origin':'public_source_annotation_conversion',
            'verification':'source_annotation_and_deterministic_geometry',
            'verification_rule':rule,'task_scope':scope,'strong_model_verified':False,
            'human_reviewed':False,'public_release_eligible':False}}
        self.files[info['split']].write(encode(record)+'\n')
        self.counts[(info['split'],form)]+=1
        if sum(self.counts.values())%5000==0:self.progress()

    def progress(self, complete=False):
        for f in self.files.values():f.flush()
        self.images.flush();self.db.commit()
        value={'source':self.source,'state':'complete' if complete else 'running',
            'records':sum(self.counts.values()),'images_seen':len(self.image_cache),
            'images_usable':sum(x is not None for x in self.image_cache.values()),
            'by_split_form':{s:{f:self.counts[(s,f)] for f in FORMS} for s in self.files},
            'skipped':dict(self.skipped),'wall_seconds':time.monotonic()-self.start,
            'candidate_evaluation_not_human_confirmed':True,'target_pool':5800000,
            'target_scene_images':160000,'model_calls':0}
        save(self.run/'progress.json',value)
        print(encode(value),flush=True)
        return value

    def finish(self, binding):
        stats=self.progress(True)
        for f in self.files.values():f.close()
        self.images.close();self.db.close()
        save(self.run/'manifest.json',{**stats,'input_binding':binding,
            'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in self.run.glob('*.jsonl')},
            'split_seed':SEED,'evaluation_fraction_before_training_exclusions':.1,
            'split_note':'Image grouped, existing train and public-eval images excluded from new evaluation; no model predictions used.'})


def pixmo(root, limit):
    protected,trained,bindings=load_exclusions()
    source=Path(os.environ['ANVIL_PIXMO_SOURCE'])
    bindings['annotations']={'path':str(source),'sha256':sha(source)}
    out=Output(root,'pixmo_points',protected,trained)
    save(out.run/'inputs.json',bindings)
    for i,r in enumerate(rows(source)):
        if limit and i>=limit:break
        query=r['conversations'][0]['value'].replace('<image>','').strip()
        raw=r['conversations'][1]['value']
        match=re.fullmatch(r'\s*<point>\s*(.*?)\s*</point>\s*',raw,re.S)
        if not match:out.skipped['source_point_parse']+=1;continue
        try: points=json.loads(match.group(1))
        except ValueError:out.skipped['source_point_parse']+=1;continue
        if not points or any(not isinstance(p,list) or len(p)!=2 or
             any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1000 for v in p) for p in points):
            out.skipped['invalid_source_points']+=1;continue
        points=[list(p) for p in sorted(set(tuple(p) for p in points))]
        path=(source.parent/r['image']).resolve()
        if not path.is_relative_to(source.parent.resolve()):raise ValueError('Source path escapes image directory')
        info=out.image(path,int(r['width']),int(r['height']),r['image'])
        if not info:continue
        label=re.search(r'^Where (?:are|is) (?:the )?(.+?)\?',query)
        label=label.group(1) if label else 'queried entity'
        target=[{'label':label,'point_2d':p} for p in points]
        prompt=query+' Return the annotated points as a JSON array of label and point_2d; coordinates use a 0-1000 image-relative scale.'
        out.add(info,'point',prompt,target,[r['id']],
                'Preserve original source point coordinates; finite bounds and decoded image dimensions checked.')
    out.finish(bindings)


def coco_targets(out, info, annotations, categories, mask_api, max_pairs):
    w,h=info['width'],info['height']
    valid=[]
    for a in annotations:
        if a.get('iscrowd') or a.get('area',0)<=0:continue
        x,y,bw,bh=a['bbox']
        if bw<=0 or bh<=0 or x<0 or y<0 or x+bw>w+.01 or y+bh>h+.01:continue
        category=categories[a['category_id']]
        box=[round(1000*x/w,3),round(1000*y/h,3),round(1000*(x+bw)/w,3),round(1000*(y+bh)/h,3)]
        valid.append({'ann':a,'category':category,'bbox':box,'cx':x+bw/2,'cy':y+bh/2})
    by_cat=defaultdict(list)
    for e in valid:by_cat[e['category']].append(e)
    for category,entities in by_cat.items():
        entities.sort(key=lambda e:(e['cx'],e['cy'],e['ann']['id']))
        for rank,e in enumerate(entities,1):
            e['label']=category if len(entities)==1 else f'{category} #{rank} in left-to-right box-center order'
    for e in valid:
        a=e['ann']; label=e['label'];aid=a['id']
        out.add(info,'bbox',f'Locate {label}. Return label and bbox_2d in a JSON array, on the 0-1000 image-relative scale.',
                [{'label':label,'bbox_2d':e['bbox']}],[aid],'Official non-crowd instance box, dimensions and positive extent checked.')
        seg=a.get('segmentation')
        if not isinstance(seg,list) or not seg:continue
        try:
            rs=mask_api.frPyObjects(seg,h,w);binary=mask_api.decode(mask_api.merge(rs))
        except (ValueError,TypeError):out.skipped['mask_decode_failed']+=1;continue
        if not binary.any():out.skipped['empty_mask']+=1;continue
        # A source mask interior, not an unverified box center or contact label.
        dist=cv2.distanceTransform(np.pad(binary.astype(np.uint8),1),cv2.DIST_L2,3)[1:-1,1:-1]
        py,px=np.unravel_index(np.argmax(dist),dist.shape)
        point=[round(1000*(px+.5)/w,3),round(1000*(py+.5)/h,3)]
        out.add(info,'point',f'Indicate an interior point of {label}. Return label and point_2d in a JSON array, on the 0-1000 image-relative scale.',
                [{'label':label,'point_2d':point}],[aid],'Maximum mask-interior distance; positive membership checked. Not a contact or affordance target.')
        polygons=[]
        for polygon in seg:
            if len(polygon)<6 or len(polygon)%2:continue
            p=np.asarray(polygon,dtype=float).reshape(-1,2)
            if not np.isfinite(p).all() or (p<0).any() or (p[:,0]>w).any() or (p[:,1]>h).any():continue
            # Collinear edge subdivision retains the exact annotated geometry.
            while len(p)<6:p=np.array([v for j in range(len(p)) for v in (p[j],(p[j]+p[(j+1)%len(p)])/2)])
            polygons.append([[round(1000*x/w,3),round(1000*y/h,3)] for x,y in p])
        if len(polygons)==len(seg):
            out.add(info,'mask',f'Segment {label}. Return label and polygons (all visible components) on the 0-1000 image-relative scale.',
                    [{'label':label,'polygons':polygons}],[aid],'Original instance polygons; raster decoding, positive area, coordinates and image bounds checked.')
    if valid:
        out.add(info,'detection','Return all annotated non-crowd COCO-category instances as a JSON array of label and bbox_2d, on the 0-1000 image-relative scale.',
                [{'label':e['label'],'bbox_2d':e['bbox']} for e in valid],
                [e['ann']['id'] for e in valid],'Official instance inventory; not exhaustive objects outside COCO categories.')
    # Deterministic pair sampling independent of any model scores.
    pairs=list(itertools.combinations(valid,2))
    pairs.sort(key=lambda pair:digest(encode([SEED,pair[0]['ann']['id'],pair[1]['ann']['id']]).encode()))
    edges=[]
    for a,b in pairs[:max_pairs]:
        dx=(a['cx']-b['cx'])/w;dy=(a['cy']-b['cy'])/h
        if abs(dx)<.02 and abs(dy)<.02:continue
        predicate=('left_of' if dx<0 else 'right_of') if abs(dx)>=abs(dy) else ('above' if dy<0 else 'below')
        edge={'subject':a['label'],'predicate':predicate,'object':b['label'],'frame':'image_box_centers'}
        prompt=f'Using image-frame bounding-box centers, where is {a["label"]} relative to {b["label"]}? Return subject, predicate, object, and frame as JSON.'
        ids=[a['ann']['id'],b['ann']['id']]
        out.add(info,'direction',prompt,edge,ids,'Dominant signed box-center axis with 0.02 normalized-image uncertainty band.')
        target={'subject':a['label'],'predicate':'distance','object':b['label'],
                'value':round(math.hypot(dx,dy),6),'unit':'normalized_image','frame':'image_box_centers'}
        out.add(info,'metric_comparison',f'What is the image-plane Euclidean distance between the box centers of {a["label"]} and {b["label"]}? Normalize x by image width and y by height; return a JSON distance relation.',
                target,ids,'Euclidean distance from official 2D boxes. NOT physical metric distance.')
        edges.append(edge)
    if len(edges)>=2:
        labels=sorted({e[k] for e in edges for k in ('subject','object')})
        out.add(info,'relation_graph','Return the following image-frame box-center relations as a JSON edge list: '+
                encode([{'subject':e['subject'],'object':e['object']} for e in edges])+'. Use the dominant displacement axis.',
                edges,[e['ann']['id'] for e in valid if e['label'] in labels],
                'Graph edges agree with independently computed source-box direction targets.')


def coco(root, limit, max_pairs, image_dir=None):
    archive=Path(os.environ['ANVIL_COCO_ANNOTATIONS'])
    if sha(archive)!='113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268':
        raise ValueError('Official annotation archive checksum changed')
    if image_dir is None:
        download=root/'coco_cache/download.json'
        if not download.is_file():
            raise FileNotFoundError('Run download first, or pass --coco-images with an existing train2017 directory')
        acquired=json.loads(download.read_text())
        if acquired['state']!='complete':raise RuntimeError('Incomplete image archive')
        image_dir=root/'coco_cache/train2017'
    else:
        image_dir=Path(image_dir).resolve()
        if not image_dir.is_dir():raise NotADirectoryError(image_dir)
        acquired={'state':'user_supplied_directory','path':str(image_dir),
                  'verification':'Each used image is decoded and hashed; archive integrity not asserted.'}
    if os.environ.get('ANVIL_COCO_DEPS'):
        sys.path.insert(0,os.environ['ANVIL_COCO_DEPS'])
    from pycocotools import mask as mask_api
    with zipfile.ZipFile(archive) as z:
        data=json.loads(z.read('annotations/instances_train2017.json'))
    by_image=defaultdict(list)
    for a in data['annotations']:by_image[a['image_id']].append(a)
    categories={r['id']:r['name'] for r in data['categories']}
    protected,trained,bindings=load_exclusions()
    bindings.update(annotations={'path':str(archive),'sha256':sha(archive)},images=acquired)
    out=Output(root,'coco_train2017',protected,trained);save(out.run/'inputs.json',bindings)
    for i,r in enumerate(sorted(data['images'],key=lambda r:r['id'])):
        if limit and i>=limit:break
        name=r['file_name']
        if not re.fullmatch(r'\d{12}\.jpg',name):raise ValueError('Unexpected COCO image filename')
        info=out.image(image_dir/name,r['width'],r['height'],r['id'])
        if info:coco_targets(out,info,by_image[r['id']],categories,mask_api,max_pairs)
    out.finish(bindings)


def download_coco(root):
    cache=root/'coco_cache';cache.mkdir(exist_ok=True)
    archive=cache/'train2017.zip';part=cache/'train2017.zip.part'
    url='https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip'
    if not archive.exists():
        subprocess.run(['curl','--fail','--location','--retry','5','--connect-timeout','30',
                        '--continue-at','-','--output',str(part),url],check=True)
        if part.stat().st_size!=COCO_IMAGE_BYTES or sha(part)!=COCO_IMAGE_SHA256:
            raise ValueError('COCO archive size/checksum mismatch')
        part.rename(archive)
    if archive.is_symlink() or archive.stat().st_size!=COCO_IMAGE_BYTES or sha(archive)!=COCO_IMAGE_SHA256:
        raise ValueError('Existing COCO archive size/checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():continue
            if not re.fullmatch(r'train2017/\d{12}\.jpg',member.filename):raise ValueError('Unexpected ZIP member')
            target=cache/member.filename
            if not target.resolve().is_relative_to(cache.resolve()):
                raise ValueError('Extracted image path escapes cache')
            target.parent.mkdir(exist_ok=True)
            # Reading all bytes also verifies each ZIP CRC; extracted files are private new cache only.
            content=z.read(member)
            if target.exists():
                if target.is_symlink() or target.read_bytes()!=content:
                    raise ValueError('Existing extracted image differs; choose a fresh output directory')
            else:
                with target.open('xb') as f:f.write(content)
    save(cache/'download.json',{'state':'complete','url':url,'bytes':archive.stat().st_size,
         'sha256':COCO_IMAGE_SHA256,'zip_crc_verified':True,'image_rights':'Per-image COCO license; source conversion does not grant redistribution rights.'})


SHAREROBOT_REVISION = '3266d92902b038ce40e7fb8aac5bfd9287eb3e45'
SHAREROBOT_ANNOTATION_SHA256 = 'c45041ad2c25c3ed1da018e6aa8f7b7b2e3ffcdbf96c46a09ca4a9a113d67b90'
AFFORDANCE_REGION_CONTRACT = 'anvil-affordance-region-v1'


def sharerobot_region_reference(row, official, image_root):
    """Preserve an official instruction-conditioned region; never invent a contact point."""
    meta = row['metadata']; source_meta = meta['src_meta']
    if (meta.get('source'), meta.get('split'), meta.get('output_type')) != (
            'sharerobot', 'train', 'affordance_box'):
        raise ValueError('not_a_training_affordance_region')
    if source_meta.get('label_source') != 'human_annotated':
        raise ValueError('source_annotation_provenance_missing')
    if meta.get('src_id') != 'sharerobot_affordance_' + str(official['id']):
        raise ValueError('official_record_identity_mismatch')
    raw_meta = official['meta_data']; region = official['affordance']
    size = [raw_meta['original_width'], raw_meta['original_height']]
    if (source_meta.get('original_dataset') != raw_meta['original_dataset']
            or source_meta.get('original_size') != size
            or source_meta.get('instruction') != official['instruction']
            or source_meta.get('affordance_px') != region):
        raise ValueError('official_annotation_binding_mismatch')
    instruction = official['instruction']
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('empty_instruction')
    values = [*size, *(region.get(k) for k in ('x', 'y', 'width', 'height'))]
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool)
               and math.isfinite(v) for v in values):
        raise ValueError('nonfinite_or_nonnumeric_geometry')
    w, h = size; x, y, bw, bh = (region[k] for k in ('x', 'y', 'width', 'height'))
    if not (w > 0 and h > 0 and bw > 0 and bh > 0 and x >= 0 and y >= 0
            and x + bw <= w + 1e-6 and y + bh <= h + 1e-6):
        raise ValueError('out_of_bounds_source_region')
    relative = Path(official['image_path'])
    image_root = Path(image_root).resolve()
    image = (image_root / relative).resolve()
    if relative.is_absolute() or not image.is_relative_to(image_root):
        raise ValueError('source_image_path_escape')
    old_images = row.get('images')
    if (not isinstance(old_images, list) or len(old_images) != 1
            or not isinstance(old_images[0], str)
            or not old_images[0].endswith('/' + relative.as_posix())
            or meta.get('scene_id') != str(Path(old_images[0]).parent)):
        raise ValueError('source_image_or_episode_binding_mismatch')
    episode = relative.parent.name
    if '#episode_' not in episode or not episode.split('#episode_', 1)[1]:
        raise ValueError('physical_episode_identity_missing')
    box = [round(1000*x/w, 6), round(1000*y/h, 6),
           round(1000*(x+bw)/w, 6), round(1000*(y+bh)/h, 6)]
    messages = row.get('messages', [])
    if len(messages) != 2 or [m.get('role') for m in messages] != ['user', 'assistant']:
        raise ValueError('unexpected_source_message_shape')
    match = re.fullmatch(r'\s*<bbox>(.*?)</bbox>\s*', messages[1]['content'], re.S)
    try:
        legacy = json.loads(match.group(1)) if match else None
        if (not isinstance(legacy, list) or len(legacy) != 2
                or any(not isinstance(v, list) or len(v) != 2 for v in legacy)):
            raise ValueError()
        flat = legacy[0] + legacy[1]
        if any(not isinstance(a, (float, int)) or isinstance(a, bool) or not math.isfinite(a)
               or abs(a-b) > 0.501 for a, b in zip(flat, box)):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError('cached_target_disagrees_with_official_region') from None
    # Episode, not image or frame, is the partition unit. Source-root names are not identities.
    group = 'ShareRobot:' + raw_meta['original_dataset'] + ':' + episode
    target = {'affordance': instruction, 'region_bbox_2d': box}
    return {'image': image, 'size': size, 'group_id': group, 'target': target,
            'official_id': str(official['id']), 'original_dataset': raw_meta['original_dataset'],
            'source_image_path': relative.as_posix()}


def sharerobot_affordance(root, source, official_source, image_root, *,
                         expected_source_sha256, expected_official_sha256,
                         protected_fingerprints=(), protected_records=(), export=False):
    """Validate/train-only export of an additive native affordance-region variant."""
    start = time.monotonic()
    if not protected_fingerprints and not protected_records:
        raise ValueError('Explicit protected-evaluation inputs are required')
    if export and Path(root).exists():
        raise FileExistsError('Use a new immutable output directory')
    source, official_source = Path(source), Path(official_source)
    source_raw, official_raw = source.read_bytes(), official_source.read_bytes()
    if digest(source_raw) != expected_source_sha256 or digest(official_raw) != expected_official_sha256:
        raise ValueError('Pinned annotation SHA256 mismatch')
    raw_entries = json.loads(official_raw)
    official = {str(r['id']): r for r in raw_entries}
    if len(official) != len(raw_entries):
        raise ValueError('Duplicate official annotation ID')
    protected = Fingerprints(); bindings = []; blocked_groups = set(); known_paths = set()
    for path in protected_fingerprints:
        path = Path(path); bindings.append({'kind': 'fingerprints', 'path': str(path), 'sha256': sha(path)})
        for item in rows(path):
            protected.add(item)
            if str(item.get('group_id', '')).startswith('ShareRobot:'):
                blocked_groups.add(item['group_id'])
    for path in protected_records:
        path = Path(path); bindings.append({'kind': 'evaluation_records', 'path': str(path), 'sha256': sha(path)})
        for item in rows(path):
            metadata = item.get('anvil_meta', {})
            if str(metadata.get('group_id', '')).startswith('ShareRobot:'):
                blocked_groups.add(metadata['group_id'])
            for image in item.get('images', []) or [item.get('image_path')]:
                if image and image not in known_paths:
                    protected.add(fingerprint(image)); known_paths.add(image)
    candidates = []; skipped = Counter(); seen_source = set(); image_cache = {}; source_rows = 0
    for line_number, line in enumerate(source_raw.splitlines(), 1):
        if not line.strip(): continue
        source_rows += 1
        try:
            row = json.loads(line); source_id = row['metadata']['src_id']
            if source_id in seen_source:
                raise ValueError('duplicate_source_id')
            seen_source.add(source_id)
            raw_id = source_id.removeprefix('sharerobot_affordance_')
            if raw_id not in official: raise ValueError('official_source_record_missing')
            value = sharerobot_region_reference(row, official[raw_id], image_root)
            image = str(value['image'])
            if image not in image_cache: image_cache[image] = fingerprint(image)
            fp = image_cache[image]
            if [fp['width'], fp['height']] != value['size']:
                raise ValueError('decoded_image_size_mismatch')
            if protected.matches(fp): blocked_groups.add(value['group_id'])
            value.update(image=image, fingerprint=fp, source_line=line_number,
                         source_line_sha256=digest(line))
            candidates.append(value)
        except (ValueError, TypeError, KeyError, OSError) as error:
            # Reject unchanged; never repair, clip, or synthesize a missing source field.
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
            skipped[reason] += 1
    selected = []; record_ids = set()
    for value in candidates:
        if value['group_id'] in blocked_groups:
            skipped['protected_evaluation_episode'] += 1; continue
        uid = digest(encode([AFFORDANCE_REGION_CONTRACT,
                            value['fingerprint']['decoded_rgb_sha256'], value['target']]).encode())
        if uid in record_ids:
            skipped['duplicate_image_instruction_region'] += 1; continue
        record_ids.add(uid); instruction = value['target']['affordance']
        selected.append({'id': uid, 'images': [value['image']],
            'messages': [{'role': 'user', 'content': '<image>\nFor instruction '+json.dumps(instruction)+
                ', localize the annotated instruction-conditioned affordance region. Return one JSON object '
                'with affordance (the instruction) and region_bbox_2d ([x1,y1,x2,y2], image-relative 0-1000). '
                'Predict a region, not an object box or a contact point.'},
                {'role': 'assistant', 'content': encode(value['target'])}],
            'anvil_meta': {'schema': 'anvil-source-native-v2', 'native_form': 'affordance',
                'native_variant': AFFORDANCE_REGION_CONTRACT, 'source': 'ShareRobot',
                'original_dataset': value['original_dataset'], 'source_annotation_id': value['official_id'],
                'source_image_path': value['source_image_path'], 'source_line': value['source_line'],
                'source_line_sha256': value['source_line_sha256'], 'source_sha256': expected_source_sha256,
                'official_annotations_sha256': expected_official_sha256,
                'group_id': value['group_id'], 'group_unit': 'physical_episode',
                'image_sha256': value['fingerprint']['sha256'],
                'decoded_rgb_sha256': value['fingerprint']['decoded_rgb_sha256'],
                'split': 'train', 'previous_split': 'train', 'training_approved': True,
                'benchmark_eligible': False, 'coordinate_scale': 1000,
                'verification': 'official_region_binding_image_geometry_and_protected_eval_exclusion',
                'source_claims_human_annotated': True, 'new_human_reviewed': False,
                'strong_model_verified': False, 'legacy_full_contact_schema': False,
                'public_release_eligible': False}})
    selected.sort(key=lambda r: r['id'])
    result = {'state': 'train_only_native_region_exported' if export else 'validated_no_write',
        'contract': AFFORDANCE_REGION_CONTRACT, 'native_form': 'affordance', 'source_rows': source_rows,
        'official_rows': len(raw_entries), 'records': len(selected), 'skipped': dict(skipped),
        'unique_images': len({r['anvil_meta']['decoded_rgb_sha256'] for r in selected}),
        'physical_episodes': len({r['anvil_meta']['group_id'] for r in selected}),
        'by_original_dataset': dict(Counter(r['anvil_meta']['original_dataset'] for r in selected)),
        'source': {'path': str(source), 'sha256': expected_source_sha256},
        'official_annotations': {'path': str(official_source), 'sha256': expected_official_sha256},
        'protected_inputs': bindings, 'protected_episodes_excluded': len(blocked_groups),
        'split_policy': 'Original train-only membership retained; exclude entire episodes when any image matches protected evaluation; never create a new test split.',
        'model_calls': 0, 'new_human_reviews': 0, 'benchmark_ready': False,
        'legacy_schema_unchanged': True, 'public_release_eligible': False,
        'rights': 'Upstream dataset and component image rights remain separate. No dataset-wide grant verified; do not redistribute these records/images under the code license.',
        'scope': 'Native instruction-conditioned region target, not an invented contact point, not a semantic-quality rate or an eleven-form release.',
        'wall_seconds': time.monotonic()-start}
    if export:
        root = Path(root); root.mkdir(parents=True, exist_ok=False)
        with (root/'train.jsonl').open('x') as output:
            for row in selected: output.write(encode(row)+'\n')
        result['train_sha256'] = sha(root/'train.jsonl')
        result['exporter_sha256'] = sha(Path(__file__))
        save(root/'manifest.json', result)
    return result


def add_sharerobot_arguments(parser):
    parser.add_argument('--sharerobot-source', type=Path)
    parser.add_argument('--sharerobot-official', type=Path)
    parser.add_argument('--sharerobot-images', type=Path)
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--expected-official-sha256', default=SHAREROBOT_ANNOTATION_SHA256)
    parser.add_argument('--protected-fingerprints', type=Path, action='append', default=[])
    parser.add_argument('--protected-records', type=Path, action='append', default=[])
    parser.add_argument('--export', action='store_true', help='Write a new train-only native region snapshot')


def run_sharerobot_arguments(args, parser):
    if not all((args.sharerobot_source, args.sharerobot_official, args.sharerobot_images,
                args.expected_source_sha256)):
        parser.error('sharerobot-affordance requires --sharerobot-source, --sharerobot-official, '
                     '--sharerobot-images and --expected-source-sha256')
    result = sharerobot_affordance(args.root, args.sharerobot_source, args.sharerobot_official,
        args.sharerobot_images, expected_source_sha256=args.expected_source_sha256,
        expected_official_sha256=args.expected_official_sha256,
        protected_fingerprints=args.protected_fingerprints,
        protected_records=args.protected_records, export=args.export)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def summary(root):
    db=sqlite3.connect(f'file:{(root/"source_index.sqlite").resolve()}?mode=ro',uri=True)
    values=list(db.execute('SELECT split,form,count(*) FROM records GROUP BY split,form'))
    images=dict(db.execute('SELECT split,count(*) FROM images GROUP BY split'))
    db.close()
    total=sum(n for _,_,n in values)
    save(root/'summary.json',{'unique_records':total,'target_records':5800000,
        'remaining_to_target':max(0,5800000-total),'images_by_split':images,
        'by_split_form':{s:{f:next((n for a,b,n in values if a==s and b==f),0) for f in FORMS}
                        for s in ('train','evaluation_candidate')},
        'human_confirmed_evaluation_records':0,'strong_model_verified_in_this_source_pool':0,
        'complete_11_form_release':False,'public_release_eligible':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['download','pixmo','coco','summary','sharerobot-affordance'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--pairs-per-image',type=int,default=24)
    p.add_argument('--coco-images',type=Path,help='Existing train2017 images; no download is performed')
    for name in ('eval-fingerprints','source-fingerprints','benchg-data','p5-dev','retention-panel','pixmo-source','coco-annotations'):
        p.add_argument('--'+name,type=Path,help='Override ANVIL_'+name.replace('-','_').upper())
    add_sharerobot_arguments(p);a=p.parse_args()
    if a.mode=='sharerobot-affordance':
        run_sharerobot_arguments(a,p);sys.exit(0)
    if a.limit<0 or a.pairs_per_image<0:p.error('Counts must be nonnegative')
    for name in ('eval_fingerprints','source_fingerprints','benchg_data','p5_dev','retention_panel','pixmo_source','coco_annotations'):
        value=getattr(a,name)
        if value is not None:os.environ['ANVIL_'+name.upper()]=str(value.resolve())
    a.root.mkdir(parents=True,exist_ok=True)
    try:
        if a.mode=='download':download_coco(a.root)
        elif a.mode=='pixmo':pixmo(a.root,a.limit)
        elif a.mode=='coco':coco(a.root,a.limit,a.pairs_per_image,a.coco_images)
        else:summary(a.root)
    except Exception as e:
        save(a.root/(a.mode+'.failed.json'),{'error':type(e).__name__,'reason':str(e),'at':time.time()})
        raise
