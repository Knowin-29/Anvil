"""Train-derived DoF priors, evaluated without changing any model prediction.

This diagnostic asks whether the codec's valid poses are semantically better
than simple training-only priors. Baselines use prompt context, never heldout
targets; heldout answers are accessed only by the existing metric evaluator.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'eval_code/score_code'))
from eval.native_metrics import score_row
from dof_codec_interface import context
from dof_codec_prototype import encode, decode
sys.path.insert(0, str(ROOT.parent / 'Anvil/tools'))
from arxiv_source_scale import sha, save


def medoid(rotations):
    # Training-only medoid under the original pi-symmetric rotation metric.
    r = np.asarray(rotations)
    diagonal = np.einsum('aij,bij->abi', r.transpose(0, 2, 1), r.transpose(0, 2, 1))
    trace = diagonal.sum(-1)
    symmetric_trace = (diagonal * [-1, -1, 1]).sum(-1)
    distance = np.minimum(np.arccos(np.clip((trace-1)/2, -1, 1)),
                          np.arccos(np.clip((symmetric_trace-1)/2, -1, 1)))
    return r[np.argmin(distance.mean(1))]


def main():
    base = ROOT.parent.parent.parent / 'arxiv'
    train_path = base / 'native_training_export_20261012_v4_r2/train_dof.jsonl'
    eval_path = base / 'native_evaluation_export_20261012_v4/eval_dof.jsonl'
    train = [json.loads(l) for l in train_path.open()]
    heldout = [json.loads(l) for l in eval_path.open()]
    assert len(train) == 3008 and len(heldout) == 580
    groups = defaultdict(list)
    by_image = defaultdict(list)
    inputs = []
    for row in train:
        c = context(row['messages'][0]['content'])
        y = json.loads(row['messages'][1]['content'])
        latent = encode(y, c)
        groups[c['entity']['label']].append((latent, np.array(y['camera_from_gripper'])[:3,:3]))
        groups['ALL'].append((latent, np.array(y['camera_from_gripper'])[:3,:3]))
        by_image[row['images'][0]].append(row)
        inputs.append(c)
    priors = {}
    for name, values in groups.items():
        latent = np.median([v[0] for v in values], axis=0)
        rotation = medoid([v[1] for v in values])
        latent[:3], latent[3:6] = rotation[:,0], rotation[:,1]
        priors[name] = latent.tolist()
    label_counts = Counter(c['entity']['label'] for c in inputs)
    eval_labels = Counter(context(r['messages'][0]['content'])['entity']['label'] for r in heldout)
    results = {}
    predictions = ROOT / 'dof_prior_predictions.jsonl'
    with predictions.open('x') as f:
        for variant in ('global_prior', 'category_prior', 'category_rotation_bbox_contact'):
            sums = Counter()
            for row in heldout:
                prompt = row['messages'][0]['content']
                c = context(prompt)
                group = 'ALL' if variant == 'global_prior' else c['entity']['label']
                latent = list(priors.get(group, priors['ALL']))
                if variant == 'category_rotation_bbox_contact':
                    bbox = c['entity']['bbox_2d']
                    uv = np.clip((np.array(bbox[:2]) + bbox[2:]) / 2000, 1e-8, 1-1e-8)
                    latent[9:11] = np.log(uv/(1-uv)).tolist()
                response = json.dumps(decode(latent, c))
                metrics = score_row(row, response)
                sums.update(metrics)
                f.write(json.dumps(dict(variant=variant, id=row['id'], response=response, metrics=metrics))+'\n')
            results[variant] = {k:v/len(heldout) for k,v in sums.items()}
    codec = json.loads((ROOT / 'dof_codec_comparison.json').read_text())
    for name, report in codec['results'].items():
        results[name] = next(g['mean'] for g in report['groups'] if g['form'] == 'dof')
    # Audit conventions and visible-contact support separately from accuracy.
    diagnostics = {}
    for split, rows in [('train',train),('heldout',heldout)]:
        counts = Counter()
        for row in rows:
            c = context(row['messages'][0]['content'])
            gold = json.loads(row['messages'][1]['content'])
            w,h = c['image_size_wh']
            uv = np.array(gold['contact_point_px']) / [w,h] * 1000
            b = c['entity']['bbox_2d']
            counts['rows'] += 1
            counts['bbox_exceeds_pixel_dimensions'] += b[2] > w or b[3] > h
            counts['gold_contact_inside_thousandth_bbox'] += int(b[0] <= uv[0] <= b[2] and b[1] <= uv[1] <= b[3])
            counts['bbox_unit_explicitly_named_in_prompt'] += 'thousandth' in row['messages'][0]['content'] or '[0,1000]' in row['messages'][0]['content']
        diagnostics[split] = dict(counts)
    report = dict(state='complete_diagnostic_only', n_train=len(train), n_heldout=len(heldout),
        train_images=len(by_image), train_category_counts=dict(label_counts), heldout_category_counts=dict(eval_labels),
        heldout_unseen_category_rows=sum(n for c,n in eval_labels.items() if c not in label_counts),
        results=results, input_convention_diagnostic=diagnostics,
        scope='All 580 heldout; unchanged scorer. Train-only pi-symmetric rotation medoids and median latent priors. No pixels used by these simple baselines.',
        teacher_qualified=False, model_predictions_modified=False,
        bindings={str(p):sha(p) for p in [train_path,eval_path,ROOT/'dof_codec_comparison.json',Path(__file__),predictions]})
    save(ROOT / 'dof_generalization_audit.json', report)
    print(json.dumps(dict(results=results,conventions=diagnostics),indent=2))


if __name__ == '__main__':
    main()
