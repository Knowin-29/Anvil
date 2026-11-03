"""CPU-only negative tests; no persisted acceptance and no Hub calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

from quality_gate import quality_decision

J=Path(__file__).resolve().parent;C=J.parent;V=C/'verification_20261101'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=C/'delivery_runs/geometry_repair_candidate_20261103'
        cls.report=json.loads((old/'evaluation/final_report.json').read_text())
        cls.rejected=json.loads((C/'geometry_delivery_control_20261103/quality_review.json').read_text())
        cls.report_sha=cls.rejected['report_sha256'];cls.spec_sha=cls.rejected['final_spec_sha256']
        cls.pre=module('coordinate_preupload_cpu',V/'preupload_coordinate_merged_blobs_v2.py')
        cls.whole=module('coordinate_whole_cpu',V/'verify_entire_coordinate_hub_v2.py')
        cls.pipeline=json.loads((C/'geometry_objective_v2_20261104/spec.json').read_text())

    def decision(self,review,report=None):
        return quality_decision(review,report or self.report,self.report_sha,self.spec_sha)

    def synthetic_review(self):
        # Completeness fixture only; never written to a quality_review.json.
        r=deepcopy(self.rejected);r.update(accepted=True,native_quality_required=True,
            final_model_publication_authorized_by_this_review=True)
        for item in r['benchmark_assessments'].values():
            item.update(accepted=True,rationale='Synthetic completeness fixture, not real acceptance')
        r['native_group_assessments']={g['form']+'::'+g['source']:dict(n=g['n'],metrics=deepcopy(g['mean']),
            accepted=True,rationale='Synthetic completeness fixture, not a quality claim') for g in self.report['native']['groups']}
        r['general_assessments']={name:dict(measurements=deepcopy(value),accepted=True,
            rationale='Synthetic completeness fixture') for name,value in self.report['general'].items()}
        return r

    def test_real_rejected_model_remains_rejected(self):
        self.assertIs(self.decision(self.rejected),False)

    def test_global_acceptance_cannot_override_missing_native_review(self):
        r=deepcopy(self.rejected);r['accepted']=True
        with self.assertRaises(ValueError):self.decision(r)

    def test_synthetic_complete_review_schema(self):
        self.assertIs(self.decision(self.synthetic_review()),True)

    def test_native_regressions_cannot_be_skipped(self):
        for group in ['mask::coco_train2017','affordance::ShareRobot-Bench','dof::PRISM']:
            with self.subTest(group=group):
                r=self.synthetic_review();r['native_group_assessments'][group]['accepted']=False
                with self.assertRaises(ValueError):self.decision(r)

    def test_missing_or_changed_native_metrics_rejected(self):
        for mutation in ['remove_group','remove_rotation','change_iou','wrong_count']:
            r=self.synthetic_review();items=r['native_group_assessments']
            if mutation=='remove_group':items.pop('point::pixmo_points')
            elif mutation=='remove_rotation':items['dof::PRISM']['metrics'].pop('rotation_error_degrees')
            elif mutation=='change_iou':items['mask::coco_train2017']['metrics']['matched_mask_iou']=1.
            else:items['affordance::ShareRobot-Bench']['n']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.decision(r)

    def test_wrong_identity_or_scope_rejected(self):
        for mutation in ['report','spec','native','public','general','duplicate_group']:
            r=self.synthetic_review();report=deepcopy(self.report)
            if mutation=='report':r['report_sha256']='0'*64
            elif mutation=='spec':r['final_spec_sha256']='0'*64
            elif mutation=='native':report['native']['n']=100
            elif mutation=='public':report['public']['cv_bench']['n']=100
            elif mutation=='general':report['general']['ai2d_test']['n']=100
            else:report['native']['groups'].append(deepcopy(report['native']['groups'][0]))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.decision(r,report)

    def test_general_review_required_and_finite(self):
        r=self.synthetic_review();r['general_assessments'].pop('textvqa_val')
        with self.assertRaises(ValueError):self.decision(r)
        r=self.synthetic_review();r['general_assessments']['textvqa_val']['measurements']['score']=float('nan')
        with self.assertRaises(ValueError):self.decision(r)

    def preupload_fixture(self):
        expected=self.pre.PIPELINE/'runs/mopd/checkpoint-256'
        spec=dict(schema='anvil-final-evaluation-spec-v1',training_run=str(expected.parent),
            checkpoint=str(expected),model=str(self.pre.SPEC.parent/'merged_model'),native_records=580000,
            general_records=10105,model_files={f'model-{i}.safetensors':'a'*64 for i in range(4)})
        backup=dict(state='complete_remote_verified_intermediate_checkpoints',repository=self.pre.REPOSITORY,
            key_checkpoints=[dict(path=k) for k in self.pipeline['key_checkpoints']],
            files={'checkpoints/coordinate_mopd/checkpoint-256/adapter_model.safetensors':dict(sha256='b'*64)})
        complete=dict(checkpoint=str(expected),adapter_sha256='b'*64)
        return spec,backup,complete

    def test_current35_keys_and_exact_adapter_binding(self):
        spec,backup,complete=self.preupload_fixture()
        self.assertEqual(len(self.pre.validate_identity(spec,self.pipeline,backup)),4)
        self.pre.validate_checkpoint_binding(spec,backup,complete)
        self.assertEqual(self.whole.KEYS,set(self.pipeline['key_checkpoints']))
        self.assertEqual(len(self.whole.KEYS),35)
        self.assertEqual(self.whole.PIPELINE_SPEC_SHA256,self.pre.PIPELINE_SHA256)

    def test_stale_parent_adapter_rejected(self):
        spec,backup,complete=self.preupload_fixture();complete['adapter_sha256']='c'*64
        with self.assertRaises(ValueError):self.pre.validate_checkpoint_binding(spec,backup,complete)
        complete['checkpoint']=str(C/'geometry_repair_candidate_20261103/runs/mopd/checkpoint-256')
        with self.assertRaises(ValueError):self.pre.validate_checkpoint_binding(spec,backup,complete)

    def test_missing_duplicate_keys_and_unsafe_shards_rejected(self):
        for mutation in ['missing','duplicate','shard']:
            spec,backup,_=self.preupload_fixture()
            if mutation=='missing':backup['key_checkpoints'].pop()
            elif mutation=='duplicate':backup['key_checkpoints'][-1]=backup['key_checkpoints'][0]
            else:spec['model_files']['../escape.safetensors']=spec['model_files'].pop('model-0.safetensors')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.pre.validate_identity(spec,self.pipeline,backup)


if __name__=='__main__':unittest.main()
