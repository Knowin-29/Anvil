"""Fault injection against real formal rows; never edits live training artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from checkpoint_prefix_audit import C, bound, prefix_rows, read, require, save, sealed_checkpoint, sha, validate


def rejected(call):
    try:
        call()
    except (ValueError, AssertionError, KeyError):
        return True
    raise AssertionError('Fault was not rejected')


def main():
    report_path = C / 'checkpoint_prefixes/candidate1_step128/report.json'
    report = validate(report_path)
    require(report['actual']['exposures'] == report['actual']['unique_records'] == 4096,
            'Real first128 source coverage differs')
    require(report['actual']['source_exposures'] == dict(native_distillation=2048, general_qa=1024, spatial_reference=1024),
            'Wrong real-source mixture')
    source = Path(report['rank_prefixes'][0]['snapshot']); raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    tests = dict(real4096_prefix_and_live_append_validation=True)
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp); trial = temp / 'rows.jsonl'
        trial.write_bytes(raw + b'{partial future update')
        require(b''.join(x for x,row in prefix_rows(trial,128,0,2)) == raw, 'Uncommitted suffix was consumed')
        tests['partial_future_suffix_ignored'] = True
        trial.write_bytes(b''.join(lines[:-1]))
        tests['missing_committed_row_rejected'] = rejected(lambda:prefix_rows(trial,128,0,2))
        trial.write_bytes(lines[0] + lines[0] + b''.join(lines[2:]))
        tests['duplicate_committed_row_rejected'] = rejected(lambda:prefix_rows(trial,128,0,2))
        trial.write_bytes(lines[1] + lines[0] + b''.join(lines[2:]))
        tests['reordered_committed_rows_rejected'] = rejected(lambda:prefix_rows(trial,128,0,2))
        tests['wrong_rank_rejected'] = rejected(lambda:prefix_rows(source,128,1,2))
        trial.write_bytes(raw[:-1])
        tests['partial_committed_row_rejected'] = rejected(lambda:prefix_rows(trial,128,0,2))
        modified = deepcopy(report); modified['final_quality_accepted'] = True
        save(temp / 'report.json', modified)
        tests['quality_promotion_rejected'] = rejected(lambda:validate(temp / 'report.json'))
        modified = deepcopy(report); modified['evidence']['trainer_state']['sha256'] = '0'*64
        save(temp / 'report.json', modified)
        tests['changed_checkpoint_evidence_rejected'] = rejected(lambda:validate(temp / 'report.json'))
    control = C / 'objective_controls_20261029/runs/no_mtd/checkpoint-128'
    tests['actual_zero_mtd_control_rejected'] = rejected(lambda:sealed_checkpoint(control,full=False))
    receipt = dict(state='passed', tests=tests, actual_records=4096,
        report=bound(report_path), code=bound(C / 'checkpoint_prefix_audit.py'),
        validation_code=bound(Path(__file__)), frozen_final_handoff_modified=False,
        delivery_selection_created=False, final_quality_accepted=False, completed=time.time())
    save(C / 'checkpoint_prefix_validation.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
