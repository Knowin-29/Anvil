# Anvil

### Scaling Embodied Spatial Intelligence through Knowledge Recirculation

**Scene-aligned supervision. Selective specialist transfer. Evidence-backed feedback.**

Anvil connects spatial data production, native-output evaluation and a 9B
generalist through a shared scene interface. Entities, geometry and task
dependencies remain explicit from source conversion to verification and learning.

| Component | Purpose | Implementation |
|---|---|---|
| **Anvil-Gen** | Convert source annotations, generate native targets, verify and repair | `trains/flywheel/`, `apps/data_flywheel/` |
| **Anvil-Ben** | Organize Anvil-Train and held-out Anvil-Eval; preserve native scoring | `trains/flywheel/alignment.py`, `eval/` |
| **Anvil-MOPD** | Transfer specialist knowledge on student rollouts | `trains/mopd/`, `firstfault/` |

## Release status

This is the **reviewed research source distribution** prepared on October 15,
2026, released at [Knowin-29/Anvil](https://github.com/Knowin-29/Anvil).
The repository starts from a clean, source-only snapshot; it does not include
private development history, credentials, server configuration, weights or images.
Project-owned code is MIT-licensed; inherited copyright attribution and third-party
terms are retained in `NOTICE.md`. See the [project page](https://knowin-29.github.io/Anvil.github.io/)
and [data collection](https://huggingface.co/collections/Knowin-29/anvil-6ad0934f9fe868151f1ccb9a).
This author-account repository is not a double-anonymous review URL. Use the
separately redacted review package for anonymous review.
The [staged data release](https://huggingface.co/datasets/Knowin-29/Anvil-Ben)
provides 5,766,133 annotations from the frozen 5.8M package, including 572,299
held-out spatial records. The remaining 33,867 records and original image
bundle are not public. Download metadata and source-specific terms are indexed
in `registry/datasets.json`; the subset is not the full eleven-form benchmark.
Final-model upload is paused until complete matched evaluation and checkpoint
binding. Existing private SFT preparation backups are not public final weights.

This release also preserves legacy mapping-batch metadata filtering in the MOPD
plugin while retaining strict native-row encoding budgets. The selected CPU
release suite passes 267 tests and 13 subtests; no GPU training was run for this
packaging change. It does not claim acceptance of the entire legacy test suite.

The submitted manuscript's active main table yields a **74.27% seven-public-benchmark mean**. The matching
final checkpoint has not yet been bound to that result in this worktree. The
registered 256-update Ground–Relation adapter is a mechanism-study artifact,
not a substitute for the final Anvil-MOPD weights. See the
[model-card preview](model_card.md). No default download silently selects a
historical checkpoint.
The table yields +9.29 percentage points over the 64.98 base mean. The October 12
alignment follows the submitted package's included `tables/eval_results.tex`,
not its unused older table. These are manuscript-reported values, not new inference.
The model registries explicitly mark historical adapters as ineligible for
final-model release. Their hashes remain available for reproducing the separate
mechanism studies; they do not identify the final main-table model.

## Native supervision

| Family | Native formats |
|---|---|
| Ground | Box, detection, mask, point |
| Relation | Depth order, metric comparison, direction, relation graph |
| Act | Affordance, gripper configuration, action plan |

QA replay is auxiliary training data, not a twelfth spatial format. Record and
image counts are distinct. Source conversion, specialist generation, deterministic
checks and model verification retain separate provenance. Model agreement alone
does not establish correctness. Evaluation targets never enter training or feedback.

## Source-backed production

The release includes the actual COCO/PixMo conversion path used for the large
source pool, plus PRISM/ALFRED source materialization and optional local semantic
verification. Conversion preserves source identities, image hashes, exclusions,
native targets and receipts; it does not grant automatic training or release approval.

```bash
python -m pip install -e '.[source]'
export ANVIL_EVAL_FINGERPRINTS=/data/exclusions/public_eval_images.jsonl
export ANVIL_SOURCE_FINGERPRINTS=/data/exclusions/previous_train_images.jsonl
export ANVIL_BENCHG_DATA=/data/exclusions/benchmark_records.jsonl
export ANVIL_P5_DEV=/data/exclusions/development_records.jsonl
export ANVIL_RETENTION_PANEL=/data/exclusions/retention_records.jsonl

python tools/arxiv_source_scale.py pixmo \
  --root /data/anvil_run/source_scale \
  --pixmo-source /data/pixmo_points/point_points_both.jsonl
python tools/arxiv_source_scale.py coco \
  --root /data/anvil_run/source_scale \
  --coco-annotations /data/coco/annotations_trainval2017.zip \
  --coco-images /data/coco/train2017 --pairs-per-image 24
python tools/arxiv_source_scale.py summary --root /data/anvil_run/source_scale
```

All five exclusion inputs must be real inventories; do not silently replace
missing inventories with empty files. Existing converted source outputs are
immutable; use a new output root for a changed conversion recipe. Image-group
hashing does not establish physical-scene isolation across different views.
COCO image-space distances are not metric 3D distances, and points inside masks
are not verified contact affordances.

```bash
python tools/arxiv_eval_v2_sources.py --mode fetch --root /data/anvil_sources
python tools/arxiv_eval_v2_sources.py --mode prism \
  --root /data/anvil_sources --output /data/anvil_native
python tools/arxiv_eval_v2_sources.py --mode prism-depth \
  --root /data/anvil_sources --output /data/anvil_native
python tools/map_prism_native.py \
  --source /data/anvil_native/prism_raw --output /data/anvil_native/prism_gripper
```

PRISM downloads pin source revisions and hashes. The gripper mapper emits the
explicit `anvil-prism-gripper-se3-v1` extension: calibrated contact, full pose,
approach and jaw axes bound to original annotations. It preserves the source's
unscaled length unit and never invents categorical approach or execution success.
This extension is separate from the legacy 2D schema. The explicit SFT adapter
below exports full-pose `messages`/`images` records with checked scene assignments;
it does not replace the original contract with categorical DoF labels.

### Native gripper and action entry points

Use actual source materializations and nonempty protection inventories, not
placeholder files. PRISM's `--partition-sft` reads the protected evaluation and
historical training inventories, groups shared scenes/images, and assigns a
fixed-seed scene partition. ALFRED accepts only official validation records by
default; its separate training path excludes every physical `floor_plan`/scene ID
present in the supplied validation inventory before extracting first-BEFORE frames.

```bash
export ANVIL_NATIVE_ROOT=/data/anvil_native
export ANVIL_ALFRED_ARCHIVE=/data/sources/full_2.1.0.7z
export ANVIL_ALFRED_INTERACTIONS=/data/alfred_materialized/alfred_interactions

python tools/map_prism_native.py --partition-sft \
  --contract anvil-prism-gripper-se3-v1 \
  --source "$ANVIL_NATIVE_ROOT/prism_gripper" \
  --output "$ANVIL_NATIVE_ROOT/prism_sft" --image-root "$ANVIL_NATIVE_ROOT/prism_raw" \
  --protected-records "$ANVIL_BENCHG_DATA" --protected-records "$ANVIL_P5_DEV" \
  --protected-records "$ANVIL_RETENTION_PANEL" \
  --protected-fingerprints "$ANVIL_EVAL_FINGERPRINTS" \
  --training-fingerprints "$ANVIL_SOURCE_FINGERPRINTS"
python tools/map_alfred_native.py --interactions-root "$ANVIL_ALFRED_INTERACTIONS" \
  --output "$ANVIL_NATIVE_ROOT/alfred"
python tools/map_alfred_native.py --train-archive "$ANVIL_ALFRED_ARCHIVE" \
  --heldout-records "$ANVIL_NATIVE_ROOT/alfred/records.jsonl" \
  --output "$ANVIL_NATIVE_ROOT/alfred/train_sources"
```

The verified PRISM source export contains 3,588 task-conditioned rows: 3,008
training rows in 80 scenes and 580 held-out rows in 19 scenes, with no shared
scenes or images. This is data/encoding acceptance, not completed model training.
The ALFRED validation conversion contains 3,327 interactions in 506 trajectories
and 92 scenes, all reserved for evaluation. Its training extraction currently
targets approximately 714 trajectories in 20 other scenes; this is a production
target, not an additional completed dataset count.

Calibrated PRISM depth reuses exactly the same frozen physical-scene assignments:

```bash
python tools/arxiv_source_semantic.py export-prism-sft \
  --source /data/prism/depth/records.jsonl \
  --source-manifest /data/prism/depth/manifest.json \
  --scene-assignments /data/prism/sft_policy/scene_assignments.json \
  --partition-audit /data/prism/sft_policy/partition_audit.json \
  --training-exclusions /data/prism/sft_policy/training_exclusions.json \
  --image-root /data/prism/raw --root /data/prism/depth_sft
```

The current export contains 14,849 training and 3,601 held-out depth records;
83 records from a scene absent from the frozen policy are excluded. This checks
source intervals and image identity; it does not add a human-quality estimate.

Instruction-conditioned ShareRobot contact regions have a distinct native variant:

```bash
python tools/arxiv_source_scale.py sharerobot-affordance \
  --root /data/anvil/sharerobot-native \
  --sharerobot-source /data/sharerobot/train-cache.jsonl \
  --sharerobot-official /data/sharerobot/affordance.json \
  --sharerobot-images /data/sharerobot/images \
  --expected-source-sha256 "$SHAREROBOT_SOURCE_SHA256" \
  --protected-fingerprints /data/protected/eval_image_fingerprints.jsonl \
  --protected-records /data/protected/evaluation.jsonl --export
```

Without `--export`, this is validation only. The current export has 5,907 records
on 4,994 unique images, all training-only, with matching official region labels.
Regions are not fabricated point contacts. Upstream data/image redistribution
rights are unresolved; this code release does not license or upload those data.

ALFRED emits `anvil-alfred-source-native-action-v1`: the goal, next expert action,
remaining expert plan and visible target mask/bbox. Preconditions, failure modes
and physical contact points remain absent, rather than invented to satisfy the
legacy Action schema. It is not generated CoT or an execution-success benchmark.
The training job has an exclusive lock, per-file hashes and restart journals;
rerun the same command to resume unchanged inputs. `train_sources/status.json`
reports progress; completion writes sibling `train_records.jsonl` and a manifest
with scene exclusions. No automatic download, GPU use or forced timeout is added.

The expert-continuation SFT export is a separate, source-bound interface:

```bash
python tools/map_alfred_native.py \
  --sft-train-records /data/alfred/train_records.jsonl \
  --sft-train-manifest /data/alfred/train_sources/manifest.json \
  --sft-evaluation-records /data/alfred/records.jsonl \
  --sft-evaluation-manifest /data/alfred/manifest.json \
  --sft-contract anvil-alfred-expert-continuation-sft-v1 \
  --output /data/alfred/sft
```

The current export has 2,402 train and 3,327 evaluation records, with 20/92
disjoint physical scenes. Input contains the before-action image and original
goal; source object IDs/coordinates and future actions are not in the prompt.
The reference preserves expert action strings, including empty source arguments
(25 train / 11 evaluation records); it is not a complete executable plan.

`trains/flywheel/native_training.py` provides a distinct atomic admission adapter
for source-native supervision. It preserves prompt/reference bytes and binds
source/selection/image identities without relabelling records as legacy v2/v3,
strong-model adjudication, new human review, or redistribution approval.
Callers must first verify complete sources and the frozen selection. The MTD
runtime accepts these eleven atomic native formats without converting them to
legacy v2/v3. Native and legacy linked rows retain separate route validation;
QA remains reference-only replay in the initial joint SFT. Code integration does
not establish qualified native teachers or a completed distillation run.

### Officially matched QA replay

CLEVR's complete local question/answer groups can be matched to official training
annotations even when conversion lost official IDs. The mapper removes only
known formatting wrappers, rejects ambiguous groups, excludes the frozen pool
benchmark and all supplied historical holdouts, and fingerprints local images.
It does not assert that JPEG bytes equal original official PNGs. QA is auxiliary
replay, not a twelfth native format or newly model-generated supervision.

```bash
python tools/map_clevr_replay.py \
  --source "$ANVIL_CLEVR_SOURCE" --source-sha256 "$ANVIL_CLEVR_SOURCE_SHA256" \
  --official-zip /data/sources/CLEVR_v1.0_no_images.zip \
  --official-sha256 d4435dceab30022b64f7d34b3ec217162bfc1c2950c79999784750ced22674c4 \
  --image-root /data/clevr/images \
  --pool-index /data/anvil/pool/selection.sqlite --pool-manifest /data/anvil/pool/manifest.json \
  --protected-fingerprints /data/protected/eval_image_fingerprints.jsonl \
  --protected-data /data/protected/evaluation.jsonl \
  --protected-data /data/protected/clevr_holdout.jsonl \
  --workers 8 --output /data/anvil/clevr_candidates
```

Obtain the no-image annotation archive from the [official CLEVR page](https://cs.stanford.edu/people/jcjohns/clevr/).
Repeat `--protected-data` for every historical panel; missing images fail closed.
An explicit `--protected-image-prefix OLD_ROOT LOCAL_ROOT` relocates legacy paths
without skipping protection. The manifest separates official QA matching, local
image identity, source rights and final-pool completion. Exporting candidates
does not itself satisfy the final 464K QA / 5.8M-record / 160K-image recipe.

Optional local verification requires deliberate execution and selected idle GPUs:

```bash
python tools/arxiv_source_semantic.py prepare-prism \
  --root /data/anvil_audit --prism-depth-root /data/anvil_native
python tools/arxiv_source_semantic.py verify --root /data/anvil_audit --execute \
  --strong-model-path /data/models/local-verifier --gpus 0,1 \
  --gpu-lock-dir /data/anvil_gpu_locks --tmp-root /data/tmp
python tools/arxiv_source_semantic.py summarize --root /data/anvil_audit
```

Use one shared lock directory across workers. Keep accepted, rejected and
uncertain receipts with model identity and costs. This runner uses new output
directories; it is not an arbitrary-interruption resume scheduler. Source-data
licenses and image redistribution rights remain separate from the code's MIT
license. Runtime manifests contain local paths and require separate dataset
packaging before upload. CPU source-regression checks:

```bash
python -m pytest tests/test_source_production.py tests/test_prism_native.py tests/test_alfred_native.py -q
```

### Complete the indexed recipe and native evaluation coverage

Merge QA plus source-approved native training rows into a **new** index. This
retains every frozen benchmark record and replaces dense training rows to meet
the existing record/image budgets; counts are not inflated by appending QA.

```bash
python tools/build_anvil_pool.py complete-with-qa \
  --base-index /data/anvil/pool/selection.sqlite --base-manifest /data/anvil/pool/manifest.json \
  --qa-root /data/anvil/clevr_candidates --qa-records 464000 \
  --native-source alfred_continuation train /data/anvil/alfred_sft/train.jsonl /data/anvil/alfred_sft/manifest.json \
  --output /data/anvil/pool_recipe_v3
```

An official affordance **test** source is available separately from ShareRobot
training data: [ShareRobot-Bench](https://huggingface.co/datasets/BAAI/ShareRobot-Bench),
revision `68e9c15600f85f601b9cf492e4d994b50ed4cd4d`. Download `README.md` and
`affordance/*` at that revision using `huggingface_hub.snapshot_download`, with
`repo_type="dataset"` and `local_dir="/data/sources/ShareRobot-Bench"`.
The converter verifies all 201 annotation/image Git or LFS objects against the
pinned upstream tree. It preserves pixel regions as normalized region boxes,
rejects current/historical training-image matches, and never invents contact points.

```bash
python tools/map_sharerobot_benchmark.py \
  --source-root /data/sources/ShareRobot-Bench \
  --pool-index /data/anvil/pool_recipe_v3/selection.sqlite \
  --pool-manifest /data/anvil/pool_recipe_v3/manifest.json \
  --workers 8 --output /data/anvil/sharerobot_official_test
python tools/build_anvil_pool.py extend-benchmark \
  --base-index /data/anvil/pool_recipe_v3/selection.sqlite \
  --base-manifest /data/anvil/pool_recipe_v3/manifest.json \
  --native-records /data/anvil/sharerobot_official_test/evaluation.jsonl \
  --native-manifest /data/anvil/sharerobot_official_test/manifest.json \
  --rebalance-dense-training \
  --output /data/anvil/pool_native_v4
python -m pytest tests/test_build_anvil_pool.py tests/test_source_native_regions.py -q
```

The second command is an explicit **new benchmark version**, not an in-place
update. Without `--rebalance-dense-training`, all training stays exact and insufficient
capacity is rejected. The explicit flag frees image slots by reselecting dense train
rows while retaining native training, QA, every old benchmark image and all mandatory
native targets. Both modes reallocate dense benchmark rows without using model scores
and replay every selected source offset. Previous benchmark scores are not transferable. The selected subset
is not the complete official ShareRobot-Bench protocol, and missing scene IDs do
not become a claim of scene-level separation. The upstream card declares Apache-2.0;
this does not relicense other sources. Counts, native-form coverage, full manuscript
alignment, model evaluation and redistribution remain distinct completion states.

## Install and inspect

Use Python 3.10 or newer in an isolated environment, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m anvil doctor
python -m anvil align --help
python -m anvil evaluate --list
python -m pytest tests/test_alignment.py -q
```

CPU contract tests do not download data or reproduce model scores. GPU training
additionally requires run-compatible PyTorch, Transformers, PEFT and Swift.
Broad package constraints are not a frozen final-model environment.

Official ScanNet split metadata is bundled and hash-bound by
`configs/scannet_splits/PROVENANCE.json`; scene data is not included.
The full legacy suite is not yet release-qualified: inherited grounding-verifier
and consistency tests still require interface/fixture reconciliation. The small
offline suite above is a contract preflight, not evidence that every inherited
pipeline or the final model has been reproduced.

### Offline scene preflight

The bundled synthetic fixture exercises image resolution, entity references,
native-schema validation and development partition reservation without model
calls, credentials or data downloads:

```bash
ANVIL_SMOKE_DIR=$(mktemp -d)
python -m anvil align run --manifest configs/alignment_smoke.jsonl \
  --config configs/alignment.yaml --ledger "$ANVIL_SMOKE_DIR/run.sqlite"
python -m anvil align status --ledger "$ANVIL_SMOKE_DIR/run.sqlite"
python -m anvil align export --ledger "$ANVIL_SMOKE_DIR/run.sqlite" \
  --split dev --out "$ANVIL_SMOKE_DIR/dev.jsonl"
```

Expected: one `dev` scene remains `pending`, `execute` is false, and the export
is empty because no model verification has run. The fixture is not approved
training data. Add `--execute` only to a separately configured production run,
not to this smoke example. See the eleven native interfaces with
`python -m anvil align schema`; use `--native-form relation_graph` for one schema.

## Generation and evaluation

Inspect `configs/alignment.yaml` and explicitly configure approved sources,
immutable model identities and endpoints before generation. Keep credentials
outside Git. Use `python -m anvil align --help` for production commands.
Legacy verifier templates use localhost or explicit environment placeholders;
configure `VERIFIER_SEED_KEY` and any enabled `VOLC_API_URL`/`VOLC_API_KEY`
outside the repository. Guarded depth launchers additionally require an explicit
`AUTHORIZED_ENDPOINT` matching the approved `GSA_VLM_URL`; configuring a URL
does not authorize image uploads. The independent grounding tuning prompt is
not identical to the current production prompt and is not a reproduction preset.
Accepted records from an existing verified ledger can be exported with:

```bash
python -m anvil align export --ledger /data/anvil/run.sqlite \
  --split train --out /data/anvil/train.jsonl
python -m anvil align export --ledger /data/anvil/run.sqlite \
  --split benchmark --out /data/anvil/eval.jsonl
```

Pass model inputs to inference, never the held-out `target` field. The v3
exact-match scorer is not interchangeable with calibrated spatial metrics or the
historical linked-task protocol. For an explicitly selected merged checkpoint
and a locally installed benchmark:

```bash
python -m anvil evaluate --model /models/anvil-selected \
  --benchmark where2place --seed 0 --batch-size 4 \
  --out /data/anvil/results/where2place.json \
  --dump-predictions /data/anvil/results/where2place_predictions.jsonl
```

Preserve predictions, scorer version, sample count, decoding settings and weight
hashes together. Failed parsing and truncation remain in the denominator.

For a merged training checkpoint with its `SHA256SUMS`, trainer metadata and
processor files, queue all seven public tasks and BenchG in one process:

```bash
python tools/evaluate_release_candidate.py \
  --model /models/selected-checkpoint --output /data/anvil/eval-new-run \
  --benchmark-root /data/benchmarks --benchg-data /data/benchg/test.jsonl \
  --lock-root /data/anvil_gpu_locks --runtime-parent /data/tmp \
  --gpus 0,1,2,3 --workers 4 \
  --candidate-label selected-checkpoint \
  --selection-description 'Terminal checkpoint fixed before evaluation'
```

Run inside your persistent scheduler/session. The output directory must be new.
The runner checks idle GPUs using shared cooperative locks, performs a four-item
compatibility check, then loads each worker once for its shards of every task.
It has no wall-clock kill limit and preserves completed shards after a failure;
it does not automatically retry failed inference. After inference it runs the
unchanged CPU scorers, rejects duplicate or missing sample identities, and
rechecks the weight bytes. Inspect `status.json`, `summary.json` and per-worker
logs. A finished queue is not final-model qualification or permission to publish.

## Training entry points

- `scripts/run_teacher_dim_full_dlc.sh`: specialist training.
- `scripts/run_mopd_9b_lora_l20.sh`: on-policy specialist distillation.
- `scripts/validate_grpo_dataset.py`: training-contract preflight.
- `trains/mopd/anvil_runtime.py`: routed-teacher qualification and runtime.

Set explicit `MODEL`, `TRAIN`, `OUT`, `STUDENT_ADAPTER` and G/S/A teacher-adapter
paths; first-fault runs also require qualified teacher metadata. Do not rely on
historical automatic adapter discovery for a release experiment. The v3
declared-family path and frozen v2 first-fault protocol are distinct; the
compatibility guard deliberately prevents treating one as the other.

For the source-native export, validate all twelve input streams while reserving
QA for initial SFT; all eleven spatial streams are passed to MTD unchanged:

```bash
python scripts/validate_grpo_dataset.py /data/anvil/native_training \
  --source-native --out /data/anvil/native_mtd_preflight.json

# Explicitly select idle GPUs and bind the measured native teacher-quality report.
export ANVIL_DATA_CONTRACT=anvil-source-native-train-v1 ANVIL_MTD=1
export MODEL=/models/Qwen3.5-9B TRAIN=/data/anvil/native_training
export STUDENT_ADAPTER=/models/anvil-joint/checkpoint-1024
export GSA_TEACHER_G_ADAPTER=/models/anvil-G/checkpoint-1024
export GSA_TEACHER_S_ADAPTER=/models/anvil-S/checkpoint-1024
export GSA_TEACHER_A_ADAPTER=/models/anvil-A/checkpoint-1024
export TEACHER_ELIGIBILITY_JSON=/data/anvil/native_teacher_eligibility.json
# Explicit example budgets; validate memory/encoding for the selected hardware.
export MAX_LENGTH=16384 MAX_COMPLETION_LENGTH=16384 IMAGE_MAX_TOKEN_NUM=1536
export REPORT_TO=none
# Set OUT and CUDA_VISIBLE_DEVICES before running. Never displace active jobs.
bash scripts/run_mopd_9b_lora_l20.sh
```

Native MTD requires explicit positive `MAX_LENGTH` and `MAX_COMPLETION_LENGTH`;
it does not inherit the legacy pilot's 3,072/512 defaults. These control separate
limits: the encoded prompt/reference sequence and the new student rollout.
Shorter declared generation budgets are allowed. Completion must be established
from the actual termination and native output, not from the budget alone.
The launcher uses strict preprocessing and Swift's `delete` CLI policy, which
maps to a non-truncating `raise` encoder. Anvil's custom training step propagates
encoding errors rather than resampling another record. The plugin also refuses
native rows under a left/right/split truncation policy, including direct calls.
It explicitly retains the finite context limit inside the SDK encoding context;
upstream GKD's default context otherwise temporarily sets that limit to `None`.
The preparation compatibility panel already contains 9,446-token mask and
13,511-token point sequences; it checks first/longest representatives, not every
training row. The example budgets are not GPU-memory or training-quality approval.

In addition to the existing passing, path-bound teacher report, native MTD
requires `data_contract`, `training_export_manifest_sha256`, each teacher's
`adapter_sha256`, and `joint_adapter_sha256`. These bind an independently
established quality decision, not a license to manufacture one from completed
SFT or final-benchmark tuning. Native rows route by declared G/S/A family; they
do not provide evidence for linked-stage semantic fault localization. Fresh
student token IDs determine the correction span, and the frozen teacher scores
those same prefixes. Source references are checked before reference SFT; neither
the reference nor teacher labels are inserted into the generation prompt.

The calibrated PRISM SE(3) specialist has a separate reproducible entry point:

```bash
python tools/train_prism_specialist.py \
  --model /models/Qwen3.5-9B \
  --train /data/prism/sft/train.jsonl \
  --evaluation /data/prism/sft/evaluation.jsonl \
  --data-manifest /data/prism/sft/manifest.json \
  --output /data/anvil/prism-encoding-check --gpu 0 \
  --lock-root /data/anvil_gpu_locks --runtime-parent /data/tmp \
  --cpu-preflight-only
```

For actual training, remove `--cpu-preflight-only` and choose a new output
directory inside a persistent session. The fixed chain is a separate two-step
smoke run, fresh-base 512-step LoRA, terminal-checkpoint evaluation, then the same
held-out evaluation of the base model. It never selects a checkpoint on held-out
scores. Both arms construct the same non-thinking inference template explicitly,
with a 32K context limit and 2,048 output tokens. Setting only the thinking flag
after template construction is insufficient because it leaves the old response
prefix in place; outputs from that legacy configuration are not interchangeable.
Code/data hashes, commands, environment versions, checkpoints and raw
predictions stay together. CPU encoding of all 3,008 current training records
passed without truncation; this is not a claim that the GPU chain has completed.
The tested encoding environment uses ms-swift 4.2.0.dev0, PyTorch 2.5.1+cu121,
Transformers 5.5.0, PEFT 0.18.1, datasets 3.6.0 and accelerate 1.13.0.
This specialist targets native gripper geometry, not final MOPD or action success.

### Full native training export and specialist preparation

Export the complete training side of a pinned pool before launching specialists:

```bash
python tools/export_native_training.py \
  --index /data/anvil/pool/selection.sqlite \
  --manifest /data/anvil/pool/manifest.json \
  --output /data/anvil/native-training --workers 8

python tools/train_native_specialist.py \
  --export /data/anvil/native-training --model /models/Qwen3.5-9B \
  --output /data/anvil/native-A-run --scope A --sampling family-form \
  --steps 1024 --gradient-accumulation 8 --seed 17 \
  --learning-rate 1e-4 --max-length 8192 --image-tokens 1536 --gpu 0 \
  --lock-root /data/anvil_gpu_locks --runtime-parent /data/tmp
```

The October 12 export contains all 5,220,000 selected training rows across eleven
native forms and 464,000 additional QA replay rows **within that total**, referencing
142,616 training images. The held-out side is not exported. No records are dropped
or targets repaired. Source bytes, selection offsets and image identity are bound;
native/QA protocols remain separate. These local-path JSONL files are not the
portable, anonymized dataset release.

The runner supports `--scope G|S|A|joint`. `--sampling record` uses row-proportional
mixtures; `family-form` balances spatial families and their forms, keeping joint QA
at its corpus fraction (464K/5.22M). Specialists exclude QA. Full map-style shuffling
avoids sequential-prefix bias in bounded runs. File hashes and counts are checked
in full; CPU encoding checks the first 16 and 16 longest serialized rows per form,
not every record. Training uses strict encoding: oversized or malformed rows stop
the run instead of being silently dropped, replaced, or truncated.

Add `--preflight-only` for the CPU compatibility check without acquiring a GPU.
Otherwise the runner waits for an idle GPU under a shared lock, runs a separate
two-update smoke, restarts from the base for the declared main budget, and retains
the terminal checkpoint. It has no wall-clock kill limit or automatic restart.
Use a new output directory and a persistent scheduler/session. Protocol, package
versions, base-weight hashes, commands, logs and terminal-adapter hash remain
together. A bounded run is not a complete epoch. This is source-native SFT
preparation, **not** MOPD, teacher qualification or final-model release; matched
held-out comparison is still required.

### Complete native evaluation and checkpoint queues

Private preparation handoffs can be built separately from the final-model
publisher. `python -m tools.stage_anvil_adapters` takes the model registry, a
verified immutable Hub base binding, the upstream license, a new output folder,
and four `--run FAMILY=/path/to/terminal-run` arguments (`G`, `S`, `A`, `joint`).
It verifies terminal steps, training/export/weight hashes and CPU PEFT topology,
checks every adapter tensor for finite values, and replaces only private base
paths/revisions in the distributed configs. Original tensor bytes are preserved.
The bundle excludes runtime arguments, optimizers, private logs and dataset
media. Its card states unfinished evaluation and known limitations; preparation
staging cannot pass the `final_mopd` publisher. Do not make this bundle public
or substitute it for final trained-and-evaluated weights.

Export all held-out records from the same immutable pool, then queue inference
against a training run's declared terminal checkpoint:

```bash
python tools/export_native_evaluation.py \
  --index /data/anvil/pool/selection.sqlite \
  --manifest /data/anvil/pool/manifest.json \
  --alfred-sft /data/anvil/alfred/sft --output /data/anvil/native-evaluation

python tools/run_native_evaluation.py \
  --export /data/anvil/native-evaluation --model /models/Qwen3.5-9B \
  --training-run /data/anvil/native-A-run --scope A \
  --output /data/anvil/native-A-eval --gpu 0 --batch-size 4 \
  --lock-root /data/anvil_gpu_locks
```

Use a persistent scheduler/session. The evaluation process freezes its own source
before waiting, verifies a live training process (PID plus start identity), checks
successful terminal-step weights, then acquires the same GPU lock without eviction.
For a matched base arm, replace `--training-run` with
`--base-protocol /data/anvil/native-A-run/protocol.json`; no adapter is loaded.
`--scope joint` covers all eleven forms. `--shard K --shards N` partitions every
form by frozen file ordinal, without sampling; all N shards are required for a
complete scope result. Never count one shard as the whole benchmark.
Evaluation also accepts an explicit idle-card pool such as `--gpu 0,1,5,6`;
it acquires one card under the same locks, without interrupting occupied cards.

Check the **entire reference set** with the production scorer on CPU before or
alongside inference, including rasterized masks and COCO interior-point regions:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python tools/export_native_evaluation.py check-references \
  --export /data/anvil/native-evaluation \
  --output /data/anvil/reference-self-check --workers 4
```

The check preserves every reference and denominator, verifies file/content hashes
and unique IDs, and reports each form/source separately. It writes no model
predictions and never grants teacher qualification. A reference scoring ideally
against itself is only scorer/data consistency, **not** independently measured
semantic accuracy, human approval, or benchmark performance. Unexpected reference
geometry, source-region mismatch, or changed inputs remain explicit failures.
The complete native-v4 export passed this self-check on October 14: 580,000
records, eleven forms and twelve form/source groups, with no nonideal reference
scores at the documented floating-point tolerances. This used CPU only and did
not modify the frozen evaluation export or any running inference protocol.

For an existing public Qwen3-VL checkpoint such as RoboBrain2.5-8B-NV, first
bind its complete local inference files to an immutable public Hub revision:

```bash
python tools/run_native_evaluation.py bind-pretrained \
  --model /models/RoboBrain2.5-8B-NV --repository BAAI/RoboBrain2.5-8B-NV \
  --revision 7ae187cc62e45cfa1c5904903a8f017f11b2037b \
  --output /data/anvil/robobrain25-binding.json
python tools/run_native_evaluation.py \
  --export /data/anvil/native-evaluation --model /models/RoboBrain2.5-8B-NV \
  --pretrained-binding /data/anvil/robobrain25-binding.json --scope joint \
  --output /data/anvil/native-robobrain25-eval --gpu 0,1,2,3 \
  --batch-size 4 --lock-root /data/anvil_gpu_locks
```

Binding reads public Hub metadata without credentials, downloads no model
files, and checks LFS SHA256 or Git blob identities for every inference file.
Hidden SDK cache metadata is excluded. The v4 runner records the pretrained
repository/revision separately and never invents an Anvil training history.
It uses the Qwen3-VL template, whose response prefix is empty, and preserves
the entire generated response, including any model-written reasoning or
Markdown. Anvil/Qwen3.5 keeps its verified SDK-prefix policy. Images, task
prompts, references, generation budgets and metric formulas remain unchanged;
this is the Anvil native protocol, not an official RoboBrain leaderboard run.
Existing frozen v1-v3 inference jobs are not migrated or restarted.

The current local RoboBrain run uses three complete ordinal shards
(`--shard 0/1/2 --shards 3`) containing 193,339 / 193,331 / 193,330 records.
Each process acquires one idle GPU under the shared lock and preserves its
own frozen protocol and raw responses. This is an in-progress 580K comparison,
not a completed baseline score; all shards must finish before reporting a
whole-benchmark result. Checkpoint binding and CPU preflight identities are
listed in `registry/models.json`. Use the CPU watcher below with each run;
never merge averages without checking the disjoint IDs and complete denominators.

Action prompt clarification is opt-in at export time:
`export_native_evaluation.py --action-prompt explicit-schema-v1` adds a fixed
JSON output schema, role definitions and the distinction between low-level
interactions and high-level plan actions. Its action vocabulary follows the
[pinned ALFRED renderer](https://github.com/askforalfred/alfred/blob/f91f4c0c96c7a29f33d0557f86b0a21035379b3b/gen/utils/game_util.py),
not held-out answers. IDs, goals, images and reference answers stay unchanged;
the new prompt and policy are bound in export hashes. The default `source`
preserves original prompts. Use a new export and separately declared matched
model runs; this is not a score repair, a resume option, or a completed experiment.
Existing predictions must not be relabelled as schema-explicit results.

Optional Qwen3.5 acceleration is explicit: `--require-fast-kernels` checks that
the causal-convolution, gated-delta and gated-normalization operators are all
available before loading weights; it fails rather than silently using the slow
fallback. Install compatible optional dependencies in a separate environment,
not into a running job's environment. The frozen protocol records Triton, FLA
and causal-conv1d versions (including absence). Changing those versions or this
flag requires a separately declared run, not a resume of old predictions.
Kernel microbenchmarks do not establish end-to-end speed or model accuracy.
Mask/point scopes check the `pycocotools` binary dependency before acquiring a
GPU and record its version; preserve the dependency path when resuming a job.

The complete October 12 local evaluation export contains 580,000 records and
17,384 images, complementary to the 5.22M training export. Form sizes differ:
bbox 81,280; point 224,260; mask 81,250; detection 3,738; direction 89,045;
metric comparison 88,984; relation graph 3,742; depth order 3,601; DoF 580;
expert continuation 3,327; affordance 193. Export completion is not inference
completion or permission to redistribute upstream media.

The Anvil/base arms use a matched non-thinking Qwen3.5 template, 1,536 image tokens,
32K context, bf16/SDPA, greedy decoding and seed 17. Output budgets are fixed by
form in `TOKEN_BUDGETS`; long-output forms use batch 1. Records, actual finish
reasons and usage are saved and fsynced per batch. After verifying no old worker
is live, explicit `--resume` checks code/data/weight identity and re-scores the
saved exact prefix before continuing; it does not silently change the protocol
or repair a damaged tail. There is no hard wall-clock kill or automatic restart.

Metrics are grouped by form/source, not averaged across incompatible units.
COCO interior-point predictions are checked against the original instance mask;
Pixmo annotated points use the separately named normalized-coordinate r50 score.
Boxes/masks use instance matching, relations use exact reference relations,
affordances use region IoU, and ALFRED uses expert-continuation agreement.
DoF reports source-frame rotation and projected-contact error, not robot success.
Invalid or truncated model outputs retain their full-denominator penalties;
infrastructure failures stop the run and preserve completed predictions.
CPU tests and source-reference identity checks do not establish model performance.
Matched comparisons, MOPD and release qualification remain separate steps.

For inexpensive **training-side** teacher calibration, freeze a deterministic
source/form-stratified panel before scoring the terminal specialist and joint
adapters. The selection scans and hashes all twelve training streams; auxiliary
QA is excluded. It never reads the final benchmark or substitutes a small panel
for the full evaluation queue:

```bash
python tools/calibrate_native_teachers.py select \
  --export /data/anvil/native-training --output /data/anvil/teacher-panel \
  --per-group 64 --seed 17
python tools/calibrate_native_teachers.py score \
  --panel /data/anvil/teacher-panel --model /models/Qwen3.5-9B \
  --training-run G=/data/anvil/native-G-run \
  --training-run S=/data/anvil/native-S-run \
  --training-run A=/data/anvil/native-A-run \
  --training-run joint=/data/anvil/native-joint-run \
  --output /data/anvil/teacher-calibration --gpu 0,1 \
  --lock-root /data/anvil_gpu_locks
```

Scoring uses frozen bf16/SDPA adapters on one shared backbone, batch 1, a matched
non-thinking template, 1,536 image tokens and a strict 16K context. Paired
next-token NLL uses all template-supervised assistant tokens, record means and
equal source/form weighting. It does not generate long responses. Completed
rows are fsynced; errors stop without dropping rows, truncating or restarting.
After confirming the old process has exited, `score --resume` verifies the
same code, panel, weights and saved prefix. These rows can have been seen during
SFT: lower NLL measures **training-reference fit**, not spatial accuracy or
unseen generalization. The report deliberately does **not** grant teacher
eligibility or public-release approval.

The completed native-v4 calibration contains 768 paired records in 12 strata.
Specialist training-reference NLL is lower in 11 strata. The exception is
COCO image-plane metric comparison, with specialist-minus-joint NLL +0.00006146.
The equal-stratum family differences are G -0.032941, S -0.001975 and A -0.051311.
All saved rows, paired aggregates, frozen code and adapter hashes were checked.
`registry/models.json` records the panel, score, protocol and acceptance hashes;
the raw local calibration evidence has not been uploaded. The four preparatory
adapters have a checksum-verified private Hub handoff (14 files); its immutable
revision and manifest digest are indexed in `registry/models.json`. This is not
a public download, final-model release, or a change to the qualification verdict.
These results neither establish held-out superiority nor qualify the final model.

From v2 onward, the Anvil native runner retains the raw SDK response and scores only generated text:
Swift's verified fixed empty-think scaffold is removed exactly once. Markdown
fences, nonempty reasoning, repeated scaffolds and malformed answers are **not**
repaired or extracted. The transport policy and model-text hash are recorded.
Existing v1 predictions from the two verified explicit-template runners can be
re-scored on CPU without repeating generation:

```bash
python tools/rescore_native_evaluation.py \
  --run /data/anvil/native-base-run --export /data/anvil/native-evaluation \
  --output /data/anvil/native-base-derived

# Queue a frozen CPU follower while that exact generation worker is still live.
python tools/rescore_native_evaluation.py \
  --run /data/anvil/native-base-run --export /data/anvil/native-evaluation \
  --output /data/anvil/native-base-rescore-queue --wait

# In a persistent session, accept each requested whole form as soon as it ends.
python tools/rescore_native_evaluation.py \
  --run /data/anvil/native-base-run --export /data/anvil/native-evaluation \
  --output /data/anvil/native-base-form-queue --wait \
  --form detection --form relation_graph
```

Without `--wait`, the default requires the complete scope/shard. `--form bbox`
instead processes an already completed whole form and labels scope completion
false. Every source prediction, ID, ordinal, dataset hash, completion receipt and
old score is checked before writing a new derived directory. Raw files are never
overwritten; answer failures stay in the denominator. The follower pins its own
code and the producer's PID/start identity, fails if generation fails, and never
starts or restarts a GPU task. Its final result is under `derived/metrics.json`.
Old thinking-enabled PRISM runs are not eligible for this native v1 migration.

To accept complete forms across **all ordinal shards of the same checkpoint**,
queue one CPU merger alongside inference:

```bash
python tools/rescore_native_evaluation.py merge \
  --run /data/anvil/native-baseline-shard0 \
  --run /data/anvil/native-baseline-shard1 \
  --run /data/anvil/native-baseline-shard2 \
  --export /data/anvil/native-evaluation \
  --output /data/anvil/native-baseline-merged --wait
```

The merger freezes its code and producer identities. Only `files` (per-shard
counts) and `shard` may differ between generation protocols; model weights,
templates, inference settings and scoring code must match. It waits for every
shard of a form, traverses the original export in ordinal order, re-scores each
unchanged response and checks unique IDs and exact denominators. It never
averages shard percentages or writes another copy of raw predictions.
Each accepted form appears in `forms/<form>.json` and `index.json`. Full
`metrics.json` is emitted only after every form and every generation worker's
terminal checks complete. Source failure, PID change or changed bindings stop
the merger while preserving accepted forms; no GPU task is started or restarted.
Use repeated `--form` for a deliberately partial merge, always labelled
`scope_complete=false`. Omit `--wait` only when all requested work is already
complete. Cross-model comparisons are separate from shard merging.

With both `--wait` and repeated `--form`, the follower instead writes each complete
form to `forms/<form>/metrics.json` immediately and binds its checksum in
`index.json`. Select forms not already accepted to avoid duplicate CPU work.
The index always marks `scope_complete=false`, including when every requested
form is done; a shard is not the combined benchmark. Incomplete forms are never
scored. If the producer later fails or its process identity changes, the follower
stops without discarding previously accepted forms or restarting inference.
The default full-scope follower remains available for the final aggregate.

### Portable data candidate (no automatic upload)

The October 12 reconstructed package has now passed the independent full pass:
5,800,000 records, 160,000 unique canonical RGB images and 160,001 encoded image
files. Its 153 inventory files total 28,069,628,956 bytes. The manifest and
verification receipt hashes are indexed in `registry/datasets.json`. This is
byte/representation/split conservation for the source-grounded pool, not a claim
that all records were generated or judged by a strong model, human-reviewed, or
approved for blanket public redistribution. The full portable package is verified
in private Hub staging; the public annotation subset is described above.

Source declarations and remaining media conditions are separated in
[NOTICE.md](NOTICE.md) and `registry/datasets.json`. The bound COCO license-ID
inventory covers all 81,897 selected encoded COCO images, retaining every
upstream identity when identical bytes have multiple IDs. It is not complete
creator attribution or an approval to mirror images. Reproduce this CPU-only
inventory without changing the package, targets or pixels:

```bash
python tools/export_portable_data.py \
  --audit-coco-licenses /data/anvil/portable-candidate \
  --evaluation /data/anvil/native-evaluation \
  --coco-source-manifest /data/anvil/source-scale/coco_train2017/manifest.json \
  --output /data/anvil/coco-license-inventory
```

The command binds the package, parent evaluation, source image index and
official annotation archive by hash. It writes a separate per-image license
index and summary; it neither grants permissions nor uploads any data.

```bash
python tools/export_portable_data.py \
  --training /data/anvil/native-training \
  --evaluation /data/anvil/native-evaluation \
  --output /data/anvil/portable-candidate \
  --private-work /data/anvil/private-packing-state \
  --workers 8 --rows-per-shard 100000 --images-per-shard 2000 \
  --private-term /internal/server/ --private-term OWNER_NAME

python tools/export_portable_data.py --verify /data/anvil/portable-candidate \
  --private-term /internal/server/ --private-term OWNER_NAME
```

Provide the actual owner/team identifiers privately; do not commit them. Both
parent exports must be complete and belong to the same immutable selection.
Packing preserves every record and prompt/answer, keeps QA training-only, and
rejects record/image overlap between splits. Compressed JSONL shards reference
opaque relative image paths. TAR members keep the **original encoded image bytes**
(including source image metadata), with neutral container ownership/timestamps;
`.img` members are decoded by file signature, not by a guessed file extension.
No image recompression, target redaction, or quiet removal is performed. Explicit
owner/server terms in decoded image metadata or retained record text stop the
build. Credential signatures are rejected anywhere in the original image bytes.
Short identity strings are not searched inside compressed pixel data. Generic
upstream renderer paths in original source metadata (such as CLEVR's Blender
File field) are retained; they are not this project's private server paths.
Generic absolute paths are still forbidden in the portable records and manifest.

`image_index.jsonl` records the exact byte and canonical RGB hashes, dimensions,
split, upstream source and media archive for every image file. Unique decoded
images and encoded file variants are counted separately. COCO benchmark interior
points retain the official image/annotation IDs and annotation-archive hash, so
their reference is recoverable without a private server path. QA retains its
official question ID. Extract only the verified TAR members into the package
root and resolve the relative `images` paths against that root before standard
SFT loading; the serialized messages are unchanged. The existing local-export
training/evaluation queue still consumes its bound local manifests, not this
separate package schema.

The independent `--verify` pass reads every compressed record and every archived
image, checks all declared file hashes, decodes the pixels and rechecks split,
form, reference and image conservation. It does not call a model. Private work
logs/failure receipts remain outside the package. A successful pack is still a
**local candidate**: it does not grant redistribution rights, certify all possible
personal information absent from images, or claim final model evaluation.

Upstream dataset terms and component image/asset rights remain separate.
See [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points),
[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/),
[PRISM](https://huggingface.co/datasets/allenai/PRISM),
[ALFRED](https://github.com/askforalfred/alfred),
[COCO](https://cocodataset.org/#termsofuse),
[ShareRobot](https://huggingface.co/datasets/BAAI/ShareRobot), and
[ShareRobot-Bench](https://huggingface.co/datasets/BAAI/ShareRobot-Bench).
The package records these sources and leaves public-release approval unset;
the MIT code license does not apply to all bundled source assets.

## Repository and release boundary

`anvil/` provides the CLI; `configs/` contains templates; `trains/` implements
generation and learning; `eval/` contains scorers; `scripts/` contains launchers;
`registry/` indexes artifacts. `tools/export_anvil_release.py` checks and exports
a reviewed, committed snapshot without Git history.

For double-anonymous review, the exporter also accepts a `.zip` output and
`--anonymous-review --redact-term IDENTITY`, repeated for each owner/author term.
Supply identity values privately, not in this repository. The review rendering
masks those text terms, keeps license clauses, omits source-commit metadata and
recomputes file checksums; it never changes the licensed public source. Check
the rendered archive and run its CPU tests before submitting. Do not redact
module names or the project's scientific task vocabulary.

Logs, credentials, private delivery notes, manuscript drafts, weights and dataset
payloads are not release inputs. A clean Git status alone does not establish
anonymization. Public uploads require separate approval and a verified destination.

## Model publication and data roadmap

`python -m tools.publish_anvil_model --folder /data/anvil/model-release` is an
offline preflight. It does not choose a checkpoint or contact the Hub. Prepare
the folder only after final-model identity has been established; do not export
the historical `registry/anvil-9b` adapter as the main-table model.

The folder's `MODEL_RELEASE.json` uses schema `anvil-model-release-v1` with
`artifact_role=final_mopd`, `status=verified`, `base_model`, immutable
`base_revision`, `training_manifest_sha256`, and a `files` object mapping every
upload filename to SHA256. Its `evaluation` object contains `weights_sha256`
(the exact uploaded weight map), `protocol_id` and `evidence_manifest_sha256`.
These identities must come from actual accepted runs, not this document.
Include a model card, weight license/notice, safetensors and the required model,
adapter, tokenizer and processor files. Adapter configs must pin the same base
repository and revision. Server paths, credentials, runtime files and unrelated
data are rejected. Hash checks do not replace loading/inference acceptance.

After offline checks and authorization, set credentials through the Hub login or
`HF_TOKEN`, then explicitly upload to the approved destination:

```bash
python -m tools.publish_anvil_model --folder /data/anvil/model-release \
  --upload --repo-id OWNER/Anvil-MOPD --expected-account ACCOUNT
```

Replace OWNER and ACCOUNT with the approved namespace and authenticated user.
New repositories are private by default; `--public` is a separate explicit choice.
Existing visibility is never changed automatically, and unrelated remote files
are never deleted. Successful upload verifies remote SHA256 values at the exact
returned revision. Rerun the same command to resume through the Hub client.
See the [official upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).

The staged data release supplies licensed CLEVR/COCO/PixMo annotations, the full
inventory and image reconstruction scripts. Remaining source components and
original media require separate rights review. The full 5.8M package is not
publicly downloadable. No data or held-out answers are uploaded by the model
publication command above.

The `CPU release preflight` GitHub workflow runs selected schema/release tests
and scans the tracked source. It needs no dataset, model credentials or GPU;
it does not publish anything and does not claim full legacy-suite acceptance.

## License

Project-owned code is [MIT licensed](LICENSE); see [NOTICE.md](NOTICE.md) for
inherited components. Source datasets, images, base models and derived weights
retain their applicable terms. The code license does not relicense those assets.
