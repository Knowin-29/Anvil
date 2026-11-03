"""Require explicit review of public, native and general ability before release.

This gate checks review completeness and exact evidence binding. It does not
invent a native paper aggregate or decide that any numerical score is adequate.
"""
import math

PUBLIC_COUNTS = {'cv_bench':2638, 'embspatial':3640, 'refspatial_bench':277,
                 'where2place':100, 'blink_relative_depth':124, 'erqa':287,
                 'vabench_point':300}
NATIVE_COUNTS = {
    'action_cot::ALFRED':3327, 'affordance::ShareRobot-Bench':193,
    'bbox::coco_train2017':81280, 'depth_order::PRISM':3601,
    'detection::coco_train2017':3738, 'direction::coco_train2017':89045,
    'dof::PRISM':580, 'mask::coco_train2017':81250,
    'metric_comparison::coco_train2017':88984, 'point::coco_train2017':81244,
    'point::pixmo_points':143016, 'relation_graph::coco_train2017':3742,
}
GENERAL_COUNTS = {'ai2d_test':3088, 'scienceqa_test':2017, 'textvqa_val':5000}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, float) else True


def approved(item):
    return item.get('accepted') is True and isinstance(item.get('rationale'),str) and bool(item['rationale'].strip())


def quality_decision(review, report, report_sha, spec_sha):
    require(review['schema']=='anvil-final-quality-review-v1','Unknown review schema')
    require(review['report_sha256']==report_sha and review['final_spec_sha256']==spec_sha,
            'Review binds a different model/report')
    require(report['state']=='complete_requires_quality_review' and report['final_spec_sha256']==spec_sha,
            'Incomplete or different report')
    require(finite(report) and finite(review),'Nonfinite evaluation or review')
    require(type(review['accepted']) is bool and bool(review.get('rationale')),'Missing quality decision/rationale')
    require({k:v['n'] for k,v in report['public'].items()}==PUBLIC_COUNTS,'Incomplete public evaluation')
    require({k:v['n'] for k,v in report['general'].items()}==GENERAL_COUNTS,'Incomplete general evaluation')
    groups=report['native']['groups']
    by_group={g['form']+'::'+g['source']:g for g in groups}
    require(len(groups)==len(by_group)==len(NATIVE_COUNTS) and
            {k:v['n'] for k,v in by_group.items()}==NATIVE_COUNTS and
            report['native']['n']==sum(NATIVE_COUNTS.values())==580000,'Incomplete native evaluation')
    require(set(review['benchmark_assessments'])==set(report['paper_aligned_public'])==set(PUBLIC_COUNTS),
            'Every public benchmark must be reviewed')
    for name,score in report['paper_aligned_public'].items():
        item=review['benchmark_assessments'][name]
        require(item['score']==score and item['paper']==report['paper_target'][name],
                'Review changes measured public scores')
    if not review['accepted']:
        return False
    require(review.get('native_quality_required') is True and
            review.get('final_model_publication_authorized_by_this_review') is True,
            'Acceptance must explicitly include native geometry and final publication')
    require(all(approved(item) for item in review['benchmark_assessments'].values()),
            'Missing public benchmark acceptance or rationale')
    native=review.get('native_group_assessments',{})
    require(set(native)==set(by_group),'Every native form/source group must be reviewed')
    for name,group in by_group.items():
        item=native[name]
        require(item['n']==group['n'] and item['metrics']==group['mean'],
                'Review changes or omits native measurements: '+name)
        require(approved(item),'Native capability not accepted: '+name)
    general=review.get('general_assessments',{})
    require(set(general)==set(GENERAL_COUNTS),'General capability review missing')
    for name,measurements in report['general'].items():
        item=general[name]
        require(item['measurements']==measurements and approved(item),
                'General capability not accepted or measurements changed: '+name)
    return True
