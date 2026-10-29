"""Prepare an audited alternative A teacher without altering or launching training."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
C = Path(__file__).resolve().parent
D = C / 'diagnosis_32gpu'
sys.path[:0] = [str(C), str(C / 'Anvil'), str(C / 'final_delivery'), str(C.parent), str(C.parent / 'cloud_a800')]
from common import bound, read, require, save, sha, verify
from lineage import teacher_trace
from mopd_representation import Representation


def main():
    out = C / 'next_candidate_prism_20261029'
    require(not out.exists(), 'Use a new candidate preparation directory')
    review_path = D / 'prism_learning_matched_review.json'; review = read(review_path)
    require(review['state'] == 'complete_matched_A_learning_control_review'
            and review['n_per_model'] == 4100 and review['rows_and_references_exactly_matched']
            and review['inference_protocols_matched'], 'Missing exact paired A evidence')
    for filename, expected in review['evidence'].items():
        require(sha(filename) == expected, 'Changed A comparison evidence')
    run = D / 'prism_learning_repairs/A_prism_lr1e4'
    complete, protocol = read(run / 'complete.json'), read(run / 'protocol.json')
    checkpoint = Path(complete['checkpoint'])
    require(complete['step'] == protocol['steps'] == 2048 and complete['smoke'] is False,
            'Incomplete new A training')
    seal = verify(checkpoint, protocol['world'])
    require(seal['files']['adapter_model.safetensors']['sha256'] == complete['adapter_sha256'], 'Changed A weights')
    binding = deepcopy(read(D / 'mopd_v6_candidate1/bindings.json'))
    original_A = binding['teachers']['A']
    binding['teachers']['A'] = dict(adapter=str(checkpoint), adapter_sha256=complete['adapter_sha256'],
        completion_receipt=str(run / 'complete.json'), completion_sha256=sha(run / 'complete.json'))
    representation = Representation('rotation128')
    interfaces = representation.validate_adapters({k:v['adapter'] for k,v in binding['teachers'].items()},
                                                 binding['student']['adapter'])
    require(not interfaces['student']['student_requires_representation_adaptation'], 'Student interface not ready')
    require(binding['representation_sha256'] == representation.digest, 'Representation changed')
    trace = teacher_trace('A', checkpoint)
    require(trace['adapter_sha256'] == complete['adapter_sha256'], 'Teacher lineage tip mismatch')
    comparison = review['comparisons']['A_prism_lr1e4']
    against_student = {x['form']:x for x in comparison['initial_student']}
    require(against_student['action_cot']['metrics']['next_action_exact']['candidate_minus_baseline'] > 0
            and against_student['dof']['metrics']['pi_symmetric_rotation_error_degrees']['candidate_minus_baseline'] < 0
            and against_student['dof']['metrics']['normalized_projected_contact_l2']['candidate_minus_baseline'] < 0
            and against_student['affordance']['metrics']['region_iou']['candidate_minus_baseline'] > 0,
            'Evidence no longer supports the proposed complementary candidate')
    out.mkdir()
    binding['note'] = 'Prospective new A teacher with matched geometry/contact/affordance improvements and a small action loss versus current A. No formal admission or execution proof for this tuple yet.'
    save(out / 'bindings.json', binding)
    shutil.copy2(D / 'mopd_v6_candidate1/initialization_lineage.json', out / 'initialization_lineage.json')
    save(out / 'A_teacher_lineage.json', trace)
    summary = dict(state='prepared_for_future_decision_only', source_comparison=bound(review_path),
        binding=bound(out / 'bindings.json'), teacher_lineage=bound(out / 'A_teacher_lineage.json'),
        initial_student_lineage=bound(out / 'initialization_lineage.json'),
        previous_A=original_A, proposed_A=binding['teachers']['A'], interfaces=interfaces,
        comparisons=comparison, category_prior_rotation_degrees=review['category_prior_rotation_error_degrees'],
        limitations=['Pose76.365deg remains worse than the70.88deg train-only category prior.',
            'Next-action accuracy falls0.961827pp versus current A, although it is4.628795pp above the initial student.',
            'Teacher comparisons and objective controls used held-out evaluation adaptively.',
            'The best native-reference loss coefficient and any future run budget remain undecided.'],
        remaining_before_training=['Review complete low-native-SFT and formal512 public results.',
            'Make an explicit new teacher/run decision; freeze chosen loss weights and budget.',
            'Prove actual GPU optimizer resume and throughput with this exact student/teacher/trainer tuple.'],
        teacher_qualified=False, formal_run_admitted=False, submitted=False,
        current_training_modified=False, final_quality_accepted=False, weights_uploaded=False,
        code=bound(Path(__file__)), completed=time.time())
    save(out / 'preparation.json',summary)
    print(json.dumps(dict(state=summary['state'],adapter_sha256=complete['adapter_sha256'],
        teacher_post_sft_exposures=trace['post_sft_actual']['exposures'],current_training_modified=False),indent=2))


if __name__ == '__main__':
    main()
