"""Check every admitted source row without loading heldout labels or GPUs."""
import ast
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from point_contracts import adapt, points_from_row

O = Path(__file__).resolve().parent
C = O.parent


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    spec = json.loads((C/'grounded_repair_20261102/spec.json').read_text())
    results = {}
    for source in ('grounding', 'vacancy', 'fsd'):
        binding = spec['datasets'][source]
        path = Path(binding['path'])
        assert sha(path) == binding['sha256']
        count = 0
        variants = Counter()
        with path.open() as stream:
            for ordinal, line in enumerate(stream):
                original = json.loads(line)
                before = json.dumps(original, sort_keys=True)
                points = points_from_row(original, source)
                for variant in (range(4) if ordinal < 8 else (ordinal % 4,)):
                    row = adapt(original, source, variant)
                    assert json.dumps(original, sort_keys=True) == before
                    assert row['canonical_uid'] == original['canonical_uid']
                    assert row['gsa_meta']['binding'] == original['gsa_meta']['binding']
                    assert row['images'] == original['images']
                    target = row['messages'][1]['content']
                    if source != 'fsd' and variant:
                        parsed = ast.literal_eval(target)
                        recovered = [[int(Decimal(str(x))*1000), int(Decimal(str(y))*1000)] for x,y in parsed]
                        assert recovered == points
                        assert len(parsed) == len(points) == 1
                        assert '<point>' not in target and '<answer>' not in target
                    elif source == 'fsd' and variant:
                        assert target == '<answer>' + original['messages'][1]['content'] + '</answer>'
                    else:
                        assert row['messages'] == original['messages']
                    variants[row['gsa_meta']['training_derivation']['contract']] += 1
                count += 1
        assert count == binding['records']
        results[source] = dict(records=count, source_sha256=binding['sha256'], contracts_checked=dict(variants),
                               label_coordinates_preserved_exactly=True, uids_and_images_preserved=True)
    result = dict(state='passed_cpu_only', datasets=results, total_unique_source_rows=sum(r['records'] for r in results.values()),
                  created=time.time(), code_sha256=sha(O/'point_contracts.py'), test_sha256=sha(Path(__file__)),
                  benchmark_answers_read=False, new_unique_records_claimed=0, gpu_jobs=0,
                  measured_quality_improvement=False)
    with (O/'point_contract_validation.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
