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
