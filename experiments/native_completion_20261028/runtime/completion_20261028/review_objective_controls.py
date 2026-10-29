"""Report exact complete objective controls without promoting a diagnostic model."""
import json
from pathlib import Path
import sys
import time
C = Path(__file__).resolve().parent
O = C / 'objective_controls_20261029'
sys.path.insert(0, str(O))
from objective_control_common import bound, control_checkpoint, load_experiment, read, require, save, sha


def main():
    experiment = load_experiment()
    baseline_path = C / 'diagnosis_32gpu/student_adaptation_public_comparison.json'
    baseline = read(baseline_path)['results']['student_adaptation']['paper_aligned_public']
    formal_path = C / 'formal_mopd/runs/candidate1_20261029/evaluation/128/report.json'
    formal = read(formal_path)
    results = {}; pending = []; evidence = dict(initial=bound(baseline_path), formal128=bound(formal_path), experiment=bound(O / 'experiment.json'))
    for arm, weights in experiment['arms'].items():
        out = O / 'runs' / arm; path = O / 'evaluation' / arm / 'report.json'
        if not path.exists():
            pending.append(arm)
            continue
        report = read(path); spec = read(path.parent / 'spec.json')
        require(report['state'] == 'complete_objective_control_requires_review' and report['final_mopd'] is False,
                'Incomplete/noncontrol report')
        require(report['checkpoint_spec_sha256'] == sha(path.parent / 'spec.json'), 'Changed evaluation spec')
        require(spec['evidence']['experiment'] == bound(O / 'experiment.json'), 'Changed experiment')
        complete, protocol, receipt = control_checkpoint(out / 'checkpoint-128', full=False)
        require(protocol['native_sft_weight'] == weights['native_sft_weight'] and protocol['mtd_weight'] == weights['mtd_weight'],
                'Wrong objective coefficients')
        for filename, expected in report['artifacts_sha256'].items():
            require(sha(filename) == expected, 'Changed evaluated predictions')
        for shard in range(8):
            prediction_protocol = read(path.parent / 'public' / str(shard) / 'protocol.json')
            require(prediction_protocol['checkpoint_spec_sha256'] == sha(path.parent / 'spec.json'), 'Mixed predictions')
            for key, value in dict(image_max_tokens=1536, thinking=False, temperature=0, batch=8, smoke=False).items():
                require(prediction_protocol[key] == value, 'Changed inference protocol')
        audit = read(out / 'exposure_audit.json')
        require(audit['records'] == 4096 and audit['matched_source_rows_per_update'] is True, 'Unmatched source exposure')
        scores = report['paper_aligned_public']
        require(set(scores) == set(baseline) and report['student_initial_public'] == baseline, 'Wrong benchmark baseline')
        results[arm] = dict(loss_weights=weights, checkpoint=report['checkpoint'],
            adapter_sha256=receipt['files']['adapter_model.safetensors']['sha256'],
            public=scores, macro=report['paper_aligned_macro'], general=report['general'],
            minus_initial={k:scores[k]-baseline[k] for k in scores},
            minus_formal128={k:scores[k]-formal['paper_aligned_public'][k] for k in scores},
            source_exposures=audit['source_exposures'], unique_by_category=audit['unique_by_category'])
        evidence[arm] = bound(path); evidence[arm + '_exposures'] = bound(out / 'exposure_audit.json')
        evidence[arm + '_gpu_resume'] = bound(O / 'runtime' / arm / 'runtime_verified.json')
    observations = []
    if 'no_mtd' in results:
        delta = results['no_mtd']['macro'] - sum(baseline.values()) / 7
        observations.append('Zero-MTD supervision also changes the public macro by %.6fpp. Distillation is not required for this observed regression; this does not exonerate teachers or prove one sole cause.' % delta)
    if 'low_native_sft' in results:
        delta = results['low_native_sft']['macro'] - sum(baseline.values()) / 7
        observations.append('Low-native-reference-SFT plus positive MTD changes public macro by %.6fpp versus initial. Inspect all seven task scores and formal512 before choosing a new training run.' % delta)
    report = dict(state='complete_requires_training_decision' if not pending else 'partial_waiting_controls',
        initial_public=baseline, initial_macro=sum(baseline.values()) / 7,
        formal128_public=formal['paper_aligned_public'], formal128_macro=formal['paper_aligned_macro'],
        results=results, pending=pending, observations=observations, evidence=evidence,
        limitations=['Eight physical GPUs versus16 changes RNG and MTD normalization groups.',
                    'One128-update run per arm, no replicate confidence interval.',
                    'Evaluation sets used adaptively; no new blind-test claim.',
                    'Objective controls are nonfinal; no automatic final acceptance or task restart.'],
        final_mopd=False, final_quality_accepted=False, weights_uploaded=False, completed=time.time())
    save(O / 'comparison.json', report)
    print(json.dumps(dict(state=report['state'], pending=pending, macros={k:v['macro'] for k,v in results.items()}, observations=observations), indent=2))


if __name__ == '__main__':
    main()
