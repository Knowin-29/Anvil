# Native MOPD completion campaign (2026-10-28/29)

This source snapshot records the current native trainer, data repair scripts, unchanged evaluation code, formal launch/checkpoint diagnostics, and final delivery tools. It does not report a completed formal MOPD model or uploaded final weights.

The selected student has completed the original5,220,000-record SFT and separately audited adaptation stages. Its seven-public benchmark macro is69.982857 against the report's74.27 target. Full G validation and35,989 paired G/S examples support a teacher-usefulness admission decision; A pose quality remains weak and some teacher subforms trail the student. See the evidence files for exact metrics and limitations. SFT general retention is diagnostic, not an entry gate.

`runtime/` preserves paths relative to the shared recovery root listed in `snapshot_manifest.json`. The deployment scripts depend on the corresponding original train/evaluation exports, manifests, cached base model and checkpoints on shared storage. Those large artifacts and authentication credentials are not stored here. Source SHA256 values bind this snapshot to the actual cloud files; live files are not moved or rewritten by archiving.

The selected implementation is `runtime/completion_20261028/native_mopd_v6.py`. `formal_mopd/` prepares evidence-bound formal runs and immutable intermediate public/general evaluations under the32GPU ceiling. `final_delivery/` requires full580,000 native evaluation, all seven public benchmarks, general diagnostics, lineage and an explicit evidence-backed quality review before a complete HF upload with commit-pinned hashes. CPU/preflight success is not a final quality claim.

The campaign used held-out scores adaptively to choose repairs. These evaluation sets must not be described as a new blind test. The original5.8million-record/160,000-image inventory is retained; supplementary96,931 records and extra image coverage are disclosed separately. Actual exposures and unique records must come from the selected student lineage, not inventory or all experiments combined.

Active jobs and timestamps continue to live under the recovery root. Refresh this snapshot after final training and delivery; no static status in this directory should be treated as live monitoring.

Formal task `t-20261029095820-px5rk` was submitted on2026-10-29 at09:58CST: two8-A800 workers,1664updates,globalbatch32. Matched full-budget execution measured1.4702x wall-clock speedup over8GPUs. The budget estimates19.24hours at the measured mean,46.85hours conservatively, while retaining at least24hours for final delivery. These are forecasts; actual final quality and weights remain pending. The checkpoint evaluation schedule is128/512/1024/1664.

The original controller encountered transient cloud visibility immediately after submission. `supervise_formal_v2.py` resumes monitoring the same task with observation retries and conservatively counts unresolved jobs against capacity. No replacement training job was submitted and the frozen training plan/code were not modified.

`final_delivery_handoff_v1.py` now waits for the completed formal training and checkpoint diagnostics, then prepares the exact final checkpoint, audits its lineage and submits four disjoint eight-GPU final-evaluation nodes within the32GPU campaign cap. It stops at a complete report requiring quality review; it cannot accept results or upload weights. CPU fault injection verifies capacity, observation retries and ambiguous-submission protection. The final-model GPU evaluation remains pending.

The project entry and historical GSA source preservation are documented in `../../docs/project_layout_20261029.md`; GSA branch `RWLinno/GSA_Forge/ICLR27` was remotely verified at48f9d4b90473f5427bbca07879ed2a2a355120ba. This is source synchronization, not a claim that all GSA data or models were uploaded.

A separately declared step128 native diagnostic reuses the initial student's fixed35,989 G/S rows and unchanged scorer to detect negative transfer early. It waits for the A controls to finish and always reserves8GPU slots for the formal controller's public validation, keeping the campaign within32GPUs. Paired comparisons require identical sample IDs, reference hashes and inference protocols. CPU validation covers the complete real baseline, mismatch rejection and resource accounting. This diagnostic does not replace full final validation or alter the formal training plan.

## First formal checkpoint and objective controls

The verified formal step128 checkpoint has seven-public macro68.614286 versus
the exact initial student's69.982857 (−1.368571 percentage points). Its general
scores remain essentially stable. On35,989 exact paired native G/S examples,
gains are small and mask validity is essentially unchanged. This is unresolved
public spatial regression, not an accepted final result or proof of collapse.

Two nonfinal128-update controls test native-reference SFT0.1/MTD1 and SFT1/MTD0.
They preserve the1664-update learning-rate horizon and exactly match all4096
training rows in the original first128 updates. Eight physical GPUs with a
virtual16 sampler differ in rank RNG and floating-point reduction order, so
these are diagnostic comparisons rather than exact stochastic replications.
The reduction audit corrects earlier frozen notes: MTD normalizes over eligible
examples; all observed nonempty native rows keep the same per-example weight.
The zero-MTD arm cannot be released as MOPD. Each task requires a genuine GPU
optimizer1→3 resume check before training, then unchanged seven-public and
10,105-example general evaluation. Current task receipts are archived;
source/CPU checks alone do not establish GPU success or quality improvement.

Both phase1 controls have now completed full public/general evaluation. Native
SFT0.1/MTD1 reaches69.278571 public macro; SFT1/MTD0 reaches68.682857. Both remain
below the initial69.982857, while general performance remains stable. The lower
native SFT weight mitigates part of the regression; no sole cause is established.
Further128-update controls test SFT0/MTD1 and SFT0.1/MTD0 with identical initial
student, teachers, data and schedule. The first reserves eight public-evaluation
slots; the second waits for formal512 public success before using spare capacity.
All controls remain excluded from final delivery.

Formal step512 has now completed full seven-public/general evaluation. Its
public macro69.088571 recovers0.474286pp versus step128 but remains0.894286pp
below initialization. RefSpatial Location100 is51.30 and Where2Place60.70;
ERQA49.83 improves1.05pp over initialization. This is not accepted final quality.
The explicit decision is to preserve the frozen run until phase2 objective
results are reviewed (or the next declared checkpoint arrives first).

A read-only upstream audit matched all3,384 active vacancy records exactly to
source coordinates and image dimensions. The raw source has231,554 records,
including58 out-of-range targets already excluded from the231,496 processed
records. The active conversion is faithful, but semantic correctness is not
independently established. A separate CPU preparation screens the full source
against native/public/general/test/development images and fixed scan reserves.
This candidate inventory is not admitted to active training or counted as
actual coverage; its manifest is archived only when preparation completes.
Preparation retained182,878 records/35,736 images, including180,360 records
absent from the existing3,384-record pool. Stronger scan/development exclusions
remove866 existing rows from this separate candidate. Eight flagged public or
general image pairs affecting28 such rows were visually inspected and found
to show distinct content; conservative exclusions remain in place. These new
reservations do not erase prior student exposure or establish a blind test.

The expanded A learning control improves pose80.65°→76.37°, contact error and
affordance IoU, while action accuracy falls about0.96pp. Pose remains worse
than the train-only category prior. It has not replaced the live formal teacher.
