# Native MOPD completion campaign

The active task is `t-20261030102801-c2p47`, submitted on 2026-10-30 at
10:28 Asia/Shanghai. It uses four workers with eight A800 80GB GPUs each.
Final MOPD training, full model-quality review and HF weight delivery remain
incomplete. Runtime status, rather than this source snapshot, is authoritative.

## Active implementation

`runtime/completion_20261028/direct_recovery_20261030/` contains the active
trainer, unchanged source sampler and launch code. The completed corrective
stage is reused from `direct_completion_20261029/runs/repair/checkpoint-2048`.
It completed 2,048 updates and 131,072 exposures, with 126,285 unique records
and 92,857 unique images in its audited logs.

The new task performs genuine on-policy MOPD for 1,536 updates, global batch 64,
with 98,304 planned exposures. It preserves the full 4,756,000-record native
spatial sampling pool, exact reviewed G/S/A teachers, native SFT weight 0.1,
MTD weight 1, greedy mask-cycle unlikelihood, and the existing spatial, QA and
PRISM replay mixture. Full per-form generation budgets are preserved.

The prior task `t-20261029174537-bvfnb` completed the corrective stage but
failed during MOPD update 127 on a 600-second scalar ALLREDUCE timeout.
Other ranks had reached the collective while rank 1 had not enqueued it;
no earlier OOM or numerical exception was found. Long generation is the
likely cause; new per-rank phase telemetry distinguishes generation from
other precollective work. Its 126 completed but uncheckpointed MOPD updates
(8,064 exposures), plus 29 uncommitted microbatch records, are preserved as
failed-attempt evidence and excluded from the delivered model's ancestry.

The recovery increases the collective timeout to 3,600 seconds and saves a
full checkpoint every 32 updates. It starts MOPD again from the verified
corrective checkpoint; the corrective stage is not retrained. A reversible
source comparison verifies that sampling, loss and optimizer calculations
are unchanged. CPU imports, frozen file hashes and unfinished-run rejection
passed. Runtime success still requires actual cloud execution and artifacts.

## Final evaluation and delivery

`direct_recovery_handoff.py` uses `final_delivery_recovery_v1/` and waits for
actual cloud Success and all 1,536 MOPD updates. It then merges one exact final
checkpoint and schedules at most four eight-GPU evaluation nodes within the
32-GPU campaign cap. The full scope is 580,000 native examples, all seven
public benchmarks and 10,105 general examples. General retention is diagnostic.
The handoff stops for evidence-based quality review and cannot auto-accept a
model or upload weights. Evaluation/scoring and decoder helpers are preserved.

Delivery includes the merged model, original adapter, decoder/interface,
evaluation and ancestry evidence, and full checkpoints for repair 2048 and
MOPD 512/1024/1536, including optimizer, scheduler and all rank RNG states.
Only the exact final checkpoint is claimed to have complete final evaluation.
HF delivery requires complete uploads and verification of every remote file
at a pinned commit. Server HF credentials must remain unchanged.

## Data accounting and limitations

Original inventory is 5,800,000 task records / 160,000 images, including
5,220,000 SFT records and 580,000 held-out records. Original QA is CLEVR;
the original spatial source mix is 96.43% COCO and differs from the report's
mixture. Inventory, unique records, exposures and repetitions are separate.
Historical warm-start unique coverage is unavailable and is not invented.

The original supplemental inventory was 96,931 records. Current training
admits the screened 182,878-record vacancy pool instead of the old 3,384-row
pool: 180,360 newly available records, 2,518 retained old rows and 866 old
rows excluded by stronger reservations. PRISM replay combines 13,536 expanded
records with 3,008 original PRISM DoF records; the latter are not new inventory.
Actual final coverage comes from the selected student's exposure logs, not
from unrelated teacher/control runs or the total available pool.

Earlier experiments exposed mask decoding regression and weak A pose geometry.
No new claim of universal teacher superiority or final benchmark success is
made. The initial student's public macro was 69.982857 versus the report's
74.27; the current recovered model has not yet completed final evaluation.
Held-out results informed previous repairs and must not be described as a
new blind test. Historical controls, stopped runs and their sources remain
under `runtime/` and `evidence/` for traceability; they are not active jobs.

`snapshot_manifest.json` binds the source/evidence copies to shared-storage
files. Large weights, training data and credentials are excluded from Git.
The maintained repository is Knowin-29/Anvil, branch main. GSA historical
source remains RWLinno/GSA_Forge, branch ICLR27, remotely verified at
48f9d4b90473f5427bbca07879ed2a2a355120ba. Source sync is separate from weight sync.
