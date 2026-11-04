# Anvil-Eval record score v1.1

Anvil-Eval measures native spatial prediction across 580,000 held-out records and 11 forms. Each record contributes one score in [0, 1]. The headline score is `100 × sum(record_score) / 580000`; all records remain in the denominator.

This version fixes the task-to-score mapping before evaluating the coordinate-objective MOPD candidate. It implements the paper's record-weighted aggregation with the following explicit native score definitions. The version identifier accompanies the released score. The manuscript's 65.35 is the user-designated reference target; historical scorer equivalence is a separate provenance property, not inferred from matching the aggregate formula.

| Native form | Record score |
|---|---|
| bbox | Label-matched box IoU, assignment sum divided by the larger object count |
| detection | Label-matched box F1 at IoU ≥ 0.5 |
| mask | Label-matched rasterized polygon IoU, assignment sum divided by the larger object count |
| point / COCO | Label-matched interior-point F1 against the source instance mask |
| point / PixMo | Label-matched annotated-point F1 with radius 50 on the 0–1000 coordinate grid |
| direction | Reference relation F1 |
| depth_order | Reference relation F1 |
| relation_graph | Reference relation F1 |
| metric_comparison | Relation-identity match × `max(0, 1 − normalized_image_distance_mae / √2)` |
| affordance | Region-box IoU |
| action_cot | Exact expert-continuation agreement |
| dof | `max(0, 1 − rotation_error_degrees / 180) × max(0, 1 − normalized_projected_contact_l2 / √2)` |

The distance normalizer √2 is the diagonal of the unit image square. The angle normalizer 180° is the maximum SO(3) geodesic error. DoF multiplies orientation and projected-contact quality so both contribute to the record score; it uses canonical rotation, preserving the separately reported symmetry-aware diagnostic.

Malformed, contract-invalid, missing, non-stop or truncated outputs receive zero. Missing records are detected as an incomplete evaluation and prevent generation of a complete report. Duplicate identities, changed prediction hashes, nonfinite metrics and out-of-range measurements are errors. Scores are computed per record before averaging, including the two-factor DoF score.

The implementation consumes the full evaluation's existing native metrics and original prediction records. It retains image tokens, generation budgets, task references, point matching, polygon rasterization and evaluation splits. The report includes each form/source score and every underlying native measurement; public and general evaluations remain separate.

## Acceptance

The user's 10% tolerance corresponds to 58.815 against the reference target 65.35. Every public benchmark is also assessed individually at 90% of its paper reference or better. Native geometry and general ability retain their explicit per-group review: a passing record-weighted aggregate does not automatically accept an individual capability.

## Version provenance

Mapping supplied on November 4, 2026 under the user's instruction to complete missing Anvil-Eval scoring components. The protocol is fixed before the current candidate's final predictions and scores are available. Original task metrics come from `anvil-native-reference-metrics-v1`; the aggregation layer is `anvil-eval-record-score-v1.1`.

Version 1.1 preserves the native target validator’s 1e-6 tolerance at the image-distance upper bound and clamps this floating-point margin to the geometric endpoint. All mappings and other metric checks remain unchanged.
