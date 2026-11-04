"""Training-only output-contract variants of already admitted point labels.

No new labels, image selection, benchmark inputs, scoring or coordinate guesses.
Keep the canonical UID and source binding so variants are never counted as new
training records. Integer thousandths convert exactly to three-place decimals.
"""
from copy import deepcopy
import hashlib
import json
import re

VERSION = 'anvil-admitted-point-contract-v1'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def points_from_row(row, source):
    assert source in {'grounding', 'vacancy', 'fsd'}
    assert row['gsa_meta']['split'] == 'train' and row['gsa_form'] == 'point'
    target = row['messages'][1]['content']
    match = re.fullmatch(r'<point>(.*?)</point>', target, flags=re.S)
    assert match, 'Unexpected admitted point answer'
    points = json.loads(match[1])
    assert len(points) == (3 if source == 'fsd' else 1)
    assert all(isinstance(p, list) and len(p) == 2 and
               all(type(v) is int and 0 <= v <= 1000 for v in p) for p in points)
    return points


def thousandths(value):
    return f'{value // 1000}.{value % 1000:03d}'


def adapt(row, source, variant):
    """Variant0 preserves the original; variants1..3 explicitly specify output."""
    assert type(variant) is int and 0 <= variant <= 3
    points = points_from_row(row, source)
    result = deepcopy(row)
    original_prompt = row['messages'][0]['content']
    original_target = row['messages'][1]['content']
    prompt, target = original_prompt, original_target
    contract = 'original_native_point'
    if source == 'fsd':
        if variant:
            prompt += ('\nReturn the target locations as integer image coordinates on a 0 to 1000 scale. '
                       'Put the point list inside <answer><point>...</point></answer>. '
                       'Keep x before y; the origin is the top-left corner.')
            target = '<answer>' + original_target + '</answer>'
            contract = 'answer_wrapped_native_point'
    elif variant:
        task = ('Return the location of the requested object as one point.' if source == 'grounding'
                else 'Return a valid placement point inside the requested vacant space.')
        normalized = [[thousandths(x), thousandths(y)] for x, y in points]
        prompt += ('\n' + task + ' Express x and y as fractions of the image width and height, '
                   'respectively, each in [0, 1], with the origin at the top-left corner. ')
        if variant == 3:
            prompt += 'Respond with a JSON list of coordinate pairs, [[x, y]], without tags or other text.'
            target = '[' + ', '.join('[' + ', '.join(p) + ']' for p in normalized) + ']'
            contract = 'normalized_json_points'
        else:
            prompt += 'Respond with a list of coordinate tuples, [(x, y)], without tags or other text.'
            target = '[' + ', '.join('(' + ', '.join(p) + ')' for p in normalized) + ']'
            contract = 'normalized_tuple_points'
    result['messages'][0]['content'] = prompt
    result['messages'][1]['content'] = target
    result['gsa_meta']['training_derivation'] = dict(
        version=VERSION, source=source, variant=variant, contract=contract,
        original_prompt_sha256=digest(original_prompt), original_target_sha256=digest(original_target),
        derived_prompt_sha256=digest(prompt), derived_target_sha256=digest(target),
        canonical_uid_unchanged=True, point_count=len(points), coordinate_mapping='exact_divide_1000' if
        contract.startswith('normalized_') else 'identity', annotations_added=False)
    assert result['canonical_uid'] == row['canonical_uid']
    assert result['images'] == row['images'] and result['gsa_meta']['binding'] == row['gsa_meta']['binding']
    return result
