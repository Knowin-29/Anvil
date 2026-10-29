"""Freeze a rotation representation using training poses only.

No heldout poses influence centers, K, iterations, or code ordering. This is a
representation study, not a trained teacher or a repair of existing predictions.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'Anvil/tools'))
from arxiv_source_scale import sha, save


def distances(a, b):
    # Diagonal of A^T B, for each pair. Jaw-axis sign has pi symmetry.
    diagonal = np.einsum('aji,bji->abi', a, b)
    trace = diagonal.sum(-1)
    flipped = (diagonal * [-1, -1, 1]).sum(-1)
    return np.minimum(np.arccos(np.clip((trace-1)/2, -1, 1)),
                      np.arccos(np.clip((flipped-1)/2, -1, 1)))


def main():
    source = ROOT.parent.parent.parent / 'arxiv/native_training_export_20261012_v4_r2/train_dof.jsonl'
    rows = [json.loads(l) for l in source.open()]
    assert len(rows) == 3008 and all(r['gsa_meta']['split']=='train' for r in rows)
    rotations = np.array([json.loads(r['messages'][1]['content'])['camera_from_gripper'] for r in rows])[:,:3,:3]
    # Normalize numerical roundoff exactly as in the existing metric.
    u, _, vt = np.linalg.svd(rotations)
    rotations = u @ vt
    matrix = distances(rotations, rotations)
    np.fill_diagonal(matrix, 0.)
    assert np.max(np.abs(matrix-matrix.T)) < 1e-6
    centers = [int(matrix.mean(1).argmin())]
    for _ in range(127):
        centers.append(int(matrix[:,centers].min(1).argmax()))
    assert len(set(centers)) == 128
    for _ in range(5):
        assignment = matrix[:,centers].argmin(1)
        updated = []
        for k, old in enumerate(centers):
            members = np.flatnonzero(assignment == k)
            updated.append(int(members[matrix[np.ix_(members,members)].mean(1).argmin()]) if len(members) else old)
        if updated == centers:
            break
        centers = updated
    error = np.degrees(matrix[:,centers].min(1))
    assignments = matrix[:,centers].argmin(1)
    report = dict(schema='anvil-train-only-rotation-codebook-v1', state='representation_prepared_not_trained',
        codebook_size=128, rotations=rotations[centers].tolist(),
        source_uids=[rows[i]['canonical_uid'] for i in centers],
        source_train_rows=3008, source_sha256=sha(source), code_sha256=sha(__file__),
        algorithm='Fixed K128, pi-symmetric geodesic medoid; train-only farthest-point initialization and five Lloyd medoid iterations',
        train_quantization_degrees=dict(mean=float(error.mean()), median=float(np.median(error)), max=float(error.max())),
        train_bin_counts=np.bincount(assignments, minlength=128).tolist(),
        heldout_used_to_fit=False, scoring_changed=False, actual_model_performance=None,
        requires_explicit_model_decoder=True, production_mopd_integrated=False)
    path = ROOT/'dof_rotation_codebook.json'
    assert not path.exists()
    save(path, report)
    print(json.dumps({k:report[k] for k in ['state','train_quantization_degrees','train_bin_counts']}, indent=2))


if __name__ == '__main__':
    main()
