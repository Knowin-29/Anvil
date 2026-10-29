"""Official train-only general replay with exact file/RGB heldout exclusion.
The original5.22M training corpus is immutable; additions are explicitly counted.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib,io,json,zipfile
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'retention_replay';OUT.mkdir(exist_ok=True)
SOURCE=Path('/Knowin/demo/weilinruan/GSA_collect/data/general_retention_v1/sources')
NATIVE=ROOT.parent.parent.parent/'arxiv/native_evaluation_export_20261012_v4'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rgb(data):
 with Image.open(io.BytesIO(data)) as im:
  im=im.convert('RGB');return hashlib.sha256(f'{im.width}x{im.height}:RGB:'.encode()+im.tobytes()).hexdigest()
def mc(q,choices,hint=''):
 return ('Context: '+hint+'\n' if hint else '')+'Question: '+q+'\nOptions:\n'+'\n'.join(chr(65+i)+'. '+s for i,s in enumerate(choices))+'\nSelect the correct answer. Respond with exactly one option letter (for example, A).'
held_files=set();held_rgb=set()
for path in NATIVE.glob('eval_*.jsonl'):
 for row in map(json.loads,path.open()):
  meta=row['evaluation_meta'];held_files.add(meta['image_sha256']);held_rgb.add(meta['decoded_rgb_sha256'])
manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
def held(path):
 data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==manifest['images'][path];return rgb(data)
with ThreadPoolExecutor(max_workers=16) as pool:held_rgb.update(pool.map(held,manifest['images']))
held_files.update(manifest['images'].values())
print('bound heldout files/RGB',len(held_files),len(held_rgb),flush=True)
files={'ai2d':SOURCE/'ai2d/ai2d-all.zip','scienceqa':ROOT/'retention_sources/scienceqa_train.zip','textvqa':SOURCE/'textvqa/train_val_images.zip'}
zips={k:zipfile.ZipFile(v) for k,v in files.items()};rows=[];source_bindings={str(p):sha(p) for p in files.values()}
ai_test=SOURCE/'ai2d/ai2d_test_ids.csv';test_ids=set(ai_test.read_text().split());source_bindings[str(ai_test)]=sha(ai_test)
for name in zips['ai2d'].namelist():
 if not name.startswith('ai2d/questions/') or not name.endswith('.json'):continue
 doc=json.loads(zips['ai2d'].read(name));image=doc['imageName']
 if Path(image).stem in test_ids:continue
 for q,item in doc['questions'].items():
  choices=item['answerTexts'];answer=int(item['correctAnswer']);assert 0<=answer<len(choices)
  rows.append(dict(dataset='ai2d',uid='ai2d-train:'+item['questionId'],zipname='ai2d/images/'+image,prompt=mc(q,choices),target=chr(65+answer),source_record_sha256=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()))
problems=SOURCE/'scienceqa/repo/data/scienceqa/problems.json';splits=SOURCE/'scienceqa/repo/data/scienceqa/pid_splits.json'
source_bindings.update({str(p):sha(p) for p in [problems,splits]});docs=json.loads(problems.read_text());partition=json.loads(splits.read_text())
assert set(partition['train']).isdisjoint(partition['test']) and set(partition['train']).isdisjoint(partition['val'])
for uid in partition['train']:
 doc=docs[uid];assert doc['split']=='train'
 if doc['image'] is None:continue
 rows.append(dict(dataset='scienceqa',uid='scienceqa-train:'+uid,zipname='train/'+uid+'/'+doc['image'],prompt=mc(doc['question'],doc['choices'],doc['hint']),target=chr(65+doc['answer']),source_record_sha256=hashlib.sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()))
textfile=ROOT/'retention_sources/TextVQA_0.5.1_train.json';valfile=SOURCE/'textvqa/TextVQA_0.5.1_val.json'
source_bindings.update({str(p):sha(p) for p in [textfile,valfile]});val=json.loads(valfile.read_text())['data'];val_images={x['image_id'] for x in val};val_questions={x['question_id'] for x in val}
for doc in json.loads(textfile.read_text())['data']:
 assert doc['set_name']=='train' and doc['question_id'] not in val_questions and doc['image_id'] not in val_images
 counts=Counter(doc['answers']);target=sorted(counts,key=lambda x:(-counts[x],x))[0]
 rows.append(dict(dataset='textvqa',uid='textvqa-train:'+str(doc['question_id']),zipname='train_images/'+doc['image_id']+'.jpg',prompt=doc['question']+'\nAnswer the question using a single word or phrase.',target=target,source_record_sha256=hashlib.sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()))
images={}
for row in rows:images[(row['dataset'],row['zipname'])]=None
# Read each archive entry once; verify decoded image identity before admission.
def extract(key):
 dataset,name=key;data=zips[dataset].read(name);file_sha=hashlib.sha256(data).hexdigest();rgb_sha=rgb(data)
 overlap=file_sha in held_files or rgb_sha in held_rgb
 path=OUT/'images'/dataset/Path(name).name
 if dataset=='scienceqa':path=OUT/'images'/dataset/(name.split('/')[1]+Path(name).suffix)
 if not overlap:
  path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():assert sha(path)==file_sha
  else:path.write_bytes(data)
 return key,dict(path=str(path),image_sha256=file_sha,decoded_rgb_sha256=rgb_sha,excluded=overlap)
with ThreadPoolExecutor(max_workers=16) as pool:
 for i,(key,value) in enumerate(pool.map(extract,images)):
  images[key]=value
  if i%5000==0:print('images',i,len(images),flush=True)
counts=Counter();excluded=Counter();dev=Counter();streams={k:(OUT/(k+'.jsonl')).open('w') for k in files};devfile=(OUT/'development.jsonl').open('w')
for row in rows:
 image=images[(row['dataset'],row['zipname'])]
 if image['excluded']:excluded[row['dataset']]+=1;continue
 isdev=int(image['decoded_rgb_sha256'][:8],16)%20==0
 binding=dict(source_dataset=row['dataset'],official_split='train',source_record_sha256=row['source_record_sha256'],archive_member=row['zipname'],**image)
 item=dict(id=row['uid'],canonical_uid=row['uid'],gsa_form='QA',gsa_family='QA',gsa_meta=dict(split='development' if isdev else 'train',binding=binding),
  images=[image['path']],messages=[dict(role='user',content='<image>\n'+row['prompt']),dict(role='assistant',content=row['target'])])
 (devfile if isdev else streams[row['dataset']]).write(json.dumps(item,ensure_ascii=False)+'\n');(dev if isdev else counts)[row['dataset']]+=1
for f in streams.values():f.close()
devfile.close()
report=dict(state='verified_train_only',original_sft_records=5220000,additional_training_records=dict(counts),development=dict(dev),excluded_heldout_image_overlap=dict(excluded),
 source_sha256=source_bindings,files_sha256={k:sha(OUT/(k+'.jsonl')) for k in files},heldout_native_manifest_sha256=sha(NATIVE/'manifest.json'),heldout_public_retention_manifest_sha256=sha(ROOT/'inputs/manifest.json'),
 split_rule='official train only, exclude exact image file/RGB overlap with native/public/retention heldout; reserve5% by image hash for development',
 textvqa_target_rule='most frequent official annotator answer; lexicographic tie break',code_sha256=sha(__file__),source_data_modified=False)
(OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
