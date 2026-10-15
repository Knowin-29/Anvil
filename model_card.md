---
base_model: Qwen/Qwen3.5-9B
tags:
  - spatial-reasoning
  - multimodal
  - knowledge-distillation
  - research
---

# Anvil-MOPD

**A spatial generalist learned through scene-aligned supervision and selective
multi-teacher distillation.**

## Release identity

Model-card preview updated on **2026-10-12**. This card does not yet
accompany an uploaded or verified final checkpoint. The manuscript's final model
must be matched to weight hashes, SFT and distillation configurations, and its
evaluation predictions before release. Historical SFT and small Ground–Relation
control adapters are separate artifacts.

| Field | Status |
|---|---|
| Model family | Qwen3.5-9B-based multimodal generalist |
| Training approach | Native spatial supervision, then specialist distillation |
| Final checkpoint / weight SHA256 | Not yet verified for release |
| Final base revision / training recipe | Must be bound to the final checkpoint |
| Weight upload date | Not uploaded |
| Intended visibility | Private; public release requires owner approval |

## Method

Shared scene and entity references connect native spatial targets.
Semantic-first-fault routing selects a specialist and correction span on a
student-generated response. Reference supervision covers active stages;
normalized distillation concentrates teacher guidance on eligible spans.
Inference does not require teachers or reference-aware routing.

The schema interface covers Ground (box, detection, mask, point), Relation
(depth order, metric comparison, direction, relation graph), and Act
(affordance, gripper configuration, action plan). Schema coverage is not itself
an independently validated result for each task of a released model.

## Manuscript-reported results

These are Table 1 values from the final manuscript, **not a newly reproduced
evaluation of downloadable weights**. Scores are percentages. Protocols and
checkpoint identity must be bound before publishing model evaluation metadata.

| Benchmark | Anvil-MOPD | Base model |
|---|---:|---:|
| CV-Bench | 88.21 | 87.15 |
| EmbSpatial | 82.34 | 75.58 |
| RefSpatial | 59.00 | 61.67 |
| Where2Place | 68.00 | 47.97 |
| BLINK Depth | 91.13 | 90.32 |
| ERQA | 50.52 | 49.83 |
| VABench-Point | 80.67 | 42.33 |
| Seven-public-benchmark mean | **74.27** | **64.98** |

The displayed mean gain is **9.29 percentage points**, recomputed from the
submitted package's active `tables/eval_results.tex`. Earlier release preparation
used an inactive older table; this update corrects that source selection.
VABench-Point uses
any-point success, not first-point or execution success. The manuscript also
lists Anvil-Eval at 65.35 versus 56.87 for the base; its version, form coverage,
denominators and aggregation remain to be bound to the release. The historical
1,463-record Ground panel must not be relabeled as the full 580K partition.

## Data

The manuscript describes 5.8M records over 160K scene images: 5.22M training
records including 464K QA replay records, and 580K held-out spatial records.
Downloads remain withheld pending complete production and release checks.
Source-label conversion and teacher generation retain separate provenance;
automatic verification does not mean every record was human-reviewed.

## Intended use and limitations

Research on static-image spatial understanding, structured localization,
relational reasoning and state-conditioned plan verification. Outputs are not
certified robot commands. Geometry, reference frames and action constraints
require checks before downstream use. Some mechanism studies use separate
checkpoints; their scores are not additional results for the final model.
Multi-round production improvement requires its own evaluation.

## Loading and reproducibility

Loading instructions will be bound to the actual release format, base revision,
processor/chat template and verified dependency versions. No `from_pretrained`
example is supplied for a nonexistent release. The code evaluation entry point
is `python -m anvil evaluate`; preserve raw predictions and scorer identity.

Only one qualified final checkpoint will be uploaded after selection and
verification. Scores from different checkpoints are not combined into a fictitious
model. No private images, annotations, credentials or server paths are in this card.

## License

Project-owned code is MIT licensed. Weight licensing must be checked against the
actual base-model and data terms before upload. The code license does not
automatically license derived weights or source data.
