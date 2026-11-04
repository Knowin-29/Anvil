import math
import unittest
from score_records import record_score,COUNTS

class Scores(unittest.TestCase):
    def metrics(self,**kw):return dict(json_valid=True,contract_valid=True,truncated=False,**kw)
    def test_all_forms_perfect(self):
        keys={'bbox':('matched_box_iou',1),'detection':('box_f1_iou50',1),'mask':('matched_mask_iou',1),
              'direction':('reference_relation_f1',1),'depth_order':('reference_relation_f1',1),
              'relation_graph':('reference_relation_f1',1),'affordance':('region_iou',1),
              'action_cot':('expert_continuation_exact',1)}
        for form,source in COUNTS:
            if form in keys:m={keys[form][0]:1}
            elif form=='point':m={'interior_point_f1' if source=='coco_train2017' else 'annotated_point_f1_r50':1}
            elif form=='metric_comparison':m={'relation_identity_exact':1,'normalized_image_distance_mae':0}
            else:m={'rotation_error_degrees':0,'normalized_projected_contact_l2':0}
            self.assertEqual(record_score(form,source,self.metrics(**m),'stop'),1)
    def test_invalid_and_truncated_zero(self):
        for key in ['json_valid','contract_valid','truncated']:
            m=self.metrics(matched_mask_iou=1);m[key]=(key=='truncated')
            self.assertEqual(record_score('mask','coco_train2017',m,'stop'),0)
        self.assertEqual(record_score('mask','coco_train2017',self.metrics(matched_mask_iou=1),'length'),0)
    def test_distance_endpoints_and_identity(self):
        for identity,error,want in [(1,0,1),(1,math.sqrt(2),0),(0,0,0),(1,math.sqrt(2)/2,.5)]:
            self.assertAlmostEqual(record_score('metric_comparison','coco_train2017',self.metrics(
                relation_identity_exact=identity,normalized_image_distance_mae=error),'stop'),want)
    def test_native_distance_floating_margin(self):
        m=self.metrics(relation_identity_exact=1,normalized_image_distance_mae=math.sqrt(2)+5e-7)
        self.assertEqual(record_score('metric_comparison','coco_train2017',m,'stop'),0)
        m['normalized_image_distance_mae']=math.sqrt(2)+2e-6
        with self.assertRaises(ValueError):record_score('metric_comparison','coco_train2017',m,'stop')
    def test_pose_requires_both_factors(self):
        for angle,error,want in [(0,0,1),(180,0,0),(0,math.sqrt(2),0),(90,math.sqrt(2)/2,.25)]:
            self.assertAlmostEqual(record_score('dof','PRISM',self.metrics(
                rotation_error_degrees=angle,normalized_projected_contact_l2=error),'stop'),want)
    def test_pose_score_before_averaging(self):
        a=record_score('dof','PRISM',self.metrics(rotation_error_degrees=0,normalized_projected_contact_l2=math.sqrt(2)),'stop')
        b=record_score('dof','PRISM',self.metrics(rotation_error_degrees=180,normalized_projected_contact_l2=0),'stop')
        self.assertEqual((a+b)/2,0)  # Product of averaged factors would incorrectly give .25.
    def test_nonfinite_out_of_range_and_missing_rejected(self):
        for value in [float('nan'),float('inf'),-.1,1.1]:
            with self.assertRaises(ValueError):record_score('mask','coco_train2017',self.metrics(matched_mask_iou=value),'stop')
        with self.assertRaises(KeyError):record_score('mask','coco_train2017',self.metrics(),'stop')
    def test_unknown_source_and_fixed_scope(self):
        with self.assertRaises(ValueError):record_score('point','unknown',self.metrics(interior_point_f1=1),'stop')
        self.assertEqual(sum(COUNTS.values()),580000);self.assertEqual(len({x[0] for x in COUNTS}),11)

if __name__=='__main__':unittest.main()
