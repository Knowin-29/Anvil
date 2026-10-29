"""Source-native metrics; not replacements for official public leaderboards.

All predictions remain in the denominator. No answer repair, best-of parsing,
coordinate-scale guessing, or conditional-on-success headline. Geometry errors
have explicit failure penalties. Report each form/source separately; do not
average angles, distances and overlap scores into a manufactured overall score.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from trains.flywheel.native_training import canonical, digest, require, strict_json, validate_target

VERSION = 'anvil-native-reference-metrics-v1'


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def label(value):
    # Case/whitespace normalization only; retain instance IDs and rank qualifiers.
    return ' '.join(value.casefold().split())


def assignment_sum(matrix):
    if not matrix.size: return 0.
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return float(matrix[rows, columns].sum())


def polygon_mask(polygons, size):
    from pycocotools import mask as mask_api
    width, height = size
    coordinates = [[coordinate * (width if axis % 2 == 0 else height) / 1000
                    for axis, coordinate in enumerate(np.asarray(polygon).reshape(-1))] for polygon in polygons]
    return mask_api.decode(mask_api.merge(mask_api.frPyObjects(coordinates, height, width))).astype(bool)


def multiset_f1(prediction, reference):
    pred = Counter(canonical(item) for item in prediction); gold = Counter(canonical(item) for item in reference)
    overlap = sum((pred & gold).values())
    return 2 * overlap / (sum(pred.values()) + sum(gold.values()))


def metric_names(row):
    form = row['native_form']
    if form == 'bbox': return {'matched_box_iou': 0., 'box_f1_iou50': 0.}
    if form == 'detection': return {'box_f1_iou50': 0., 'matched_box_iou': 0.}
    if form == 'mask': return {'matched_mask_iou': 0.}
    if form == 'point':
        source = row['evaluation_meta']['source']
        require(source in {'coco_train2017', 'pixmo_points'}, 'Point scoring requires an explicit source protocol')
        return {'interior_point_f1' if source == 'coco_train2017' else 'annotated_point_f1_r50': 0.}
    if form in {'direction', 'depth_order', 'relation_graph'}: return {'reference_relation_f1': 0., 'reference_exact': 0.}
    if form == 'metric_comparison': return {'normalized_image_distance_mae': math.sqrt(2), 'relation_identity_exact': 0.}
    if form == 'affordance': return {'region_iou': 0., 'region_iou50': 0., 'region_iou75': 0., 'region_iou90': 0., 'instruction_copy_exact': 0.}
    if form == 'action_cot': return {'expert_continuation_exact': 0., 'next_action_exact': 0., 'target_object_exact': 0., 'target_role_exact': 0.}
    if form == 'dof': return {'rotation_error_degrees': 180., 'pi_symmetric_rotation_error_degrees': 180.,
                               'normalized_projected_contact_l2': math.sqrt(2)}
    raise ValueError('Unknown native evaluation form')


def score_row(row, response, *, finish_reason='stop', point_region=None):
    form, variant = row['native_form'], row['native_variant']
    prompt = row['messages'][0]['content']; target = strict_json(row['messages'][1]['content'])
    validate_target(form, variant, target, prompt)  # Invalid references are fatal, never silently excluded.
    scores = metric_names(row)
    if form == 'point' and row['evaluation_meta']['source'] == 'coco_train2017':
        require(isinstance(point_region, np.ndarray) and point_region.dtype == bool
                and point_region.shape == tuple(reversed(row['image_size_wh'])) and point_region.any()
                and len(target) == 1, 'COCO interior-point reference requires its source instance mask')
    result = {'json_valid': False, 'contract_valid': False, 'truncated': finish_reason == 'length', **scores}
    try:
        prediction = strict_json(response)
        result['json_valid'] = True
        validate_target(form, variant, prediction, prompt)
        result['contract_valid'] = True
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return result
    if finish_reason != 'stop': return result
    if form in {'bbox', 'detection', 'mask', 'point'}:
        matrix = np.zeros((len(prediction), len(target)), dtype=float)
        if form == 'point' and row['evaluation_meta']['source'] == 'pixmo_points':
            pred_xy = np.asarray([item['point_2d'] for item in prediction])
            gold_xy = np.asarray([item['point_2d'] for item in target])
            distances2 = np.sum((pred_xy[:, None, :] - gold_xy[None, :, :]) ** 2, axis=-1)
            labels_match = np.asarray([label(item['label']) for item in prediction])[:, None] == np.asarray([label(item['label']) for item in target])[None, :]
            matrix = ((distances2 <= 2500.) & labels_match).astype(float)
            result['annotated_point_f1_r50'] = 2 * assignment_sum(matrix) / (len(prediction) + len(target))
            return result
        if form == 'mask':
            pred_masks = [polygon_mask(item['polygons'], row['image_size_wh']) for item in prediction]
            gold_masks = [polygon_mask(item['polygons'], row['image_size_wh']) for item in target]
            require(all(mask.any() for mask in gold_masks), 'Empty rasterized reference mask')
        for i, pred in enumerate(prediction):
            for j, gold in enumerate(target):
                if label(pred['label']) != label(gold['label']): continue
                if form in {'bbox', 'detection'}:
                    matrix[i, j] = box_iou(pred['bbox_2d'], gold['bbox_2d'])
                elif form == 'mask':
                    union = np.logical_or(pred_masks[i], gold_masks[j]).sum()
                    matrix[i, j] = np.logical_and(pred_masks[i], gold_masks[j]).sum() / union if union else 0
                elif point_region is not None and row['evaluation_meta']['source'] == 'coco_train2017':
                    x, y = pred['point_2d']; width, height = row['image_size_wh']
                    matrix[i, j] = float(0 <= x < 1000 and 0 <= y < 1000
                                         and point_region[int(y * height / 1000), int(x * width / 1000)])
                else:
                    matrix[i, j] = float(np.linalg.norm(np.asarray(pred['point_2d']) - gold['point_2d']) <= 50.)
        if form in {'bbox', 'detection'}:
            result['matched_box_iou'] = assignment_sum(matrix) / max(len(prediction), len(target))
            # Maximum-cardinality threshold matching is separate from max-IoU matching.
            result['box_f1_iou50'] = 2 * assignment_sum((matrix >= .5).astype(float)) / (len(prediction) + len(target))
        elif form == 'mask': result['matched_mask_iou'] = assignment_sum(matrix) / max(len(prediction), len(target))
        else: result[next(iter(scores))] = 2 * assignment_sum(matrix) / (len(prediction) + len(target))
    elif form in {'direction', 'depth_order', 'relation_graph'}:
        pred, gold = ([prediction], [target]) if form == 'direction' else (prediction, target)
        result['reference_relation_f1'] = multiset_f1(pred, gold)
        result['reference_exact'] = float(Counter(map(canonical, pred)) == Counter(map(canonical, gold)))
    elif form == 'metric_comparison':
        keys = ('subject', 'object', 'predicate', 'frame', 'unit')
        same = all(prediction[k] == target[k] for k in keys)
        result['relation_identity_exact'] = float(same)
        if same: result['normalized_image_distance_mae'] = abs(prediction['value'] - target['value'])
    elif form == 'affordance':
        overlap = box_iou(prediction['region_bbox_2d'], target['region_bbox_2d'])
        result.update(region_iou=overlap, region_iou50=float(overlap >= .5), region_iou75=float(overlap >= .75),
                      region_iou90=float(overlap >= .9), instruction_copy_exact=float(label(prediction['affordance']) == label(target['affordance'])))
    elif form == 'action_cot':
        result.update(expert_continuation_exact=float(prediction == target),
            next_action_exact=float(prediction['next_expert_action'] == target['next_expert_action']),
            target_object_exact=float(prediction['target_object_type'] == target['target_object_type']),
            target_role_exact=float(prediction['target_role'] == target['target_role']))
    elif form == 'dof':
        pred_pose = np.asarray(prediction['camera_from_gripper']); gold_pose = np.asarray(target['camera_from_gripper'])
        # Reference validation supplies the coordinate system and physical units.
        u, _, vt = np.linalg.svd(pred_pose[:3, :3]); rotation = u @ vt
        gu, _, gvt = np.linalg.svd(gold_pose[:3, :3]); gold_rotation = gu @ gvt
        distance = lambda other: float(np.degrees(np.arccos(np.clip((np.trace(rotation.T @ other) - 1) / 2, -1, 1))))
        canonical_angle = distance(gold_rotation)
        result['rotation_error_degrees'] = canonical_angle
        result['pi_symmetric_rotation_error_degrees'] = min(canonical_angle, distance(gold_rotation @ np.diag([-1., -1., 1.])))
        context = strict_json(prompt.removeprefix('<image>').strip())['input']
        projected = np.asarray(context['intrinsics']) @ np.asarray(prediction['contact_point_camera'])
        error = np.linalg.norm((projected[:2] / projected[2] - np.asarray(target['contact_point_px'])) / np.asarray(context['image_size_wh']))
        result['normalized_projected_contact_l2'] = min(math.sqrt(2), float(error))
    return result


class CocoPointReference:
    """Resolve interior regions from the exact official annotation source.

    A valid alternative point anywhere in the target mask must be credited;
    closeness to the source's single distance-transform anchor is not the task.
    Paths come only from the frozen local export manifest, never model output.
    """
    def __init__(self, export_manifest):
        from tools.arxiv_source_scale import sha
        inputs = [item for item in export_manifest['source_inputs'] if item['source'] == 'coco_train2017']
        require(inputs, 'COCO source binding absent')
        self.inputs = {}; self.handles = {}
        source_manifests = set()
        for item in inputs:
            require(sha(item['path']) == item['expected_sha256'] and sha(item['manifest']) == item['manifest_sha256'], 'COCO source changed')
            self.inputs[item['expected_sha256']] = item
            source_manifests.add(item['manifest'])
        annotation_bindings = [strict_json(Path(path).read_bytes())['input_binding']['annotations'] for path in source_manifests]
        require(len({(v['path'], v['sha256']) for v in annotation_bindings}) == 1, 'Inconsistent COCO annotations')
        annotation = annotation_bindings[0]
        require(sha(annotation['path']) == annotation['sha256'], 'COCO official annotation archive changed')
        with zipfile.ZipFile(annotation['path']) as archive:
            with archive.open('annotations/instances_train2017.json') as stream: data = json.load(stream)
        self.annotations = {item['id']: item for item in data['annotations']}
        require(len(self.annotations) == len(data['annotations']), 'Duplicate official annotation IDs')
        self.images = {item['id']: item for item in data['images']}
        self.annotation_binding = annotation

    def __call__(self, row):
        from pycocotools import mask as mask_api
        meta = row['evaluation_meta']; source_sha = meta['source_sha256']
        require(row['native_form'] == 'point' and meta['source'] == 'coco_train2017', 'Wrong region-reference task')
        item = self.inputs[source_sha]
        if source_sha not in self.handles: self.handles[source_sha] = Path(item['path']).open('rb', buffering=8 << 20)
        stream = self.handles[source_sha]; stream.seek(meta['source_offset']); raw = stream.read(meta['source_size'])
        require(hashlib.sha256(raw).hexdigest() == meta['source_line_sha256'], 'COCO source row changed')
        source = strict_json(raw); metadata = source['anvil_meta']
        require(source['id'] == row['id'] and digest(metadata) == meta['source_metadata_sha256'], 'COCO source identity changed')
        ids = metadata['source_annotation_ids']; require(len(ids) == 1, 'Interior point requires one source instance')
        annotation = self.annotations[ids[0]]; image = self.images[annotation['image_id']]
        require(str(annotation['image_id']) == metadata['source_image_id']
                and [image['width'], image['height']] == row['image_size_wh'], 'COCO image/annotation mismatch')
        segmentation = annotation['segmentation']; require(isinstance(segmentation, list) and segmentation, 'COCO polygon reference absent')
        rle = mask_api.merge(mask_api.frPyObjects(segmentation, image['height'], image['width']))
        return mask_api.decode(rle).astype(bool)

    def close(self):
        for stream in self.handles.values(): stream.close()


def summarize(scored_rows):
    groups = {}; seen = set()
    for row in scored_rows:
        require(isinstance(row.get('id'), str) and row['id'] and row['id'] not in seen, 'Missing or duplicated scored identity')
        seen.add(row['id'])
        key = (row['native_form'], row['source'])
        group = groups.setdefault(key, {'n': 0, 'sums': Counter(), 'keys': set(row['metrics'])})
        require(set(row['metrics']) == group['keys'], 'Inconsistent metric fields within a form/source')
        group['n'] += 1
        for name, value in row['metrics'].items():
            require(type(value) in (int, float, bool) and math.isfinite(value), 'Invalid scored value')
            group['sums'][name] += value
    return {'protocol': VERSION, 'n': len(seen), 'grouping': 'native_form/source', 'global_composite_score': None,
        'groups': [{'form': form, 'source': source, 'n': group['n'],
                    'mean': {name: value / group['n'] for name, value in sorted(group['sums'].items())}}
                   for (form, source), group in sorted(groups.items())]}
