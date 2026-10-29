"""Count distinct image-bound pose annotations separately from task paraphrases."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
from dof_codec_interface import context

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 base=ROOT.parent.parent.parent/'arxiv'
 sources={'original_train':base/'native_training_export_20261012_v4_r2/train_dof.jsonl','supplemental_train':ROOT/'prism_expansion/train_bound.jsonl','heldout':base/'native_evaluation_export_20261012_v4/eval_dof.jsonl'}
 results={}
 for split,path in sources.items():
  observations=defaultdict(list);entities=defaultdict(set);exact_inputs=defaultdict(set);poses=set();n=0
  for line in path.open():
   row=json.loads(line);c=context(row['messages'][0]['content']);y=json.loads(row['messages'][1]['content'])
   image=row['images'][0]
   pose=canonical(dict(pose=y['camera_from_gripper'],contact=y['contact_point_camera'],pixel=y['contact_point_px']))
   geometry=hashlib.sha256(pose.encode()).hexdigest();entity=canonical([image,c['entity']['label'],c['entity']['bbox_2d']])
   observations[(entity,geometry)].append(c['task']);entities[entity].add(geometry);poses.add(geometry)
   exact_inputs[(entity,c['task'])].add(geometry);n+=1
  results[split]=dict(records=n,image_entity_groups=len(entities),distinct_image_entity_pose_annotations=len(observations),
   task_rows_per_pose_histogram=dict(Counter(len(tasks) for tasks in observations.values())),
   entities_with_multiple_source_poses=sum(len(values)>1 for values in entities.values()),
   maximum_source_poses_per_image_entity=max(map(len,entities.values())),
   exact_image_entity_task_with_conflicting_targets=sum(len(values)>1 for values in exact_inputs.values()),
   unique_pose_contact_geometry=len(poses),
   interpretation='Repeated geometry with distinct task text is not counted as a new independent pose observation. This is a diversity audit, not an assertion that paraphrases are invalid or source semantics are wrong.')
 result=dict(state='complete_source_diversity_diagnostic',splits=results,training_or_evaluation_changed=False,
  heldout_used_for_training=False,bindings={str(p):sha(p) for p in [*sources.values(),Path(__file__)]})
 (ROOT/'prism_observation_diversity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
