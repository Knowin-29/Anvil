# Explicit selected-checkpoint delivery

This independent workflow supports an evaluated, sealed formal MOPD checkpoint
whose step can differ from the original planned training endpoint. It does not
modify the active `final_delivery` workflow or its frozen automatic handoff.
No checkpoint has been selected for delivery by creating or testing this code.

1. Make an explicit checkpoint and budget decision from actual model results.
   The original cloud training task must be `Success`, `Stopped`, or `Killed`.
   Running, unknown, failed, or stale cloud observations are rejected. A stopped
   run is never represented as having completed all planned updates.
2. Freeze the chosen committed exposure prefix with `checkpoint_prefix_audit.py`.
   It must refer to a declared checkpoint of the original immutable formal plan.
3. Run `selection.py --plan PLAN --prefix-audit PREFIX_REPORT --output SELECTION
   --disposition DISPOSITION --rationale REASON`. The disposition is either
   `planned_run_completed_select_recorded_checkpoint` or
   `planned_run_stopped_select_committed_prefix`.
4. Use `merge.py --adapter CHECKPOINT --selection SELECTION --output MODEL`.
   An existing exact midrun merge may also be used: `prepare.py` verifies its
   adapter identity, every model shard and inference asset.
5. Use `prepare.py --training-run RUN --selection SELECTION --model MODEL
   --output SPEC`. This produces a selected-checkpoint v2 spec for the full
   580,000 native, seven public and 10,105 general examples of one exact model.
6. `configs.py`, `node.py` and `score.py` operate as in the original workflow.
   Their scorers, model inference and native decoder functions remain unchanged.
   Before switching workflows, explicitly retire the old CPU handoff and account
   for any evaluation jobs it already submitted. Enforce the campaign's 32-GPU
   cap using fresh cloud observations before every submission.
7. `lineage.py --spec SPEC --eligibility ELIGIBILITY --output LINEAGE` uses only
   the sealed exposure snapshots of the selected step. Later training records,
   other controls, and teacher exposures cannot inflate the student's coverage.
8. Review every benchmark and native quality. Full evaluation and an accepted,
   hash-bound quality review remain mandatory; checkpoint selection is not
   quality acceptance. Re-run `validate_cpu.py` after builder changes, then use
   `package.py` with the same arguments as the original workflow. The package
   discloses selected/planned updates, the original run's actual disposition,
   and a fresh terminal cloud recheck. It contains no fabricated `complete.json`.
9. The unchanged `upload.py` uses only transient `ANVIL_HF_TOKEN`, verifies the
   Knowin-29 identity, and hashes every remote file at the exact Hub commit.
   It never logs in or modifies the server's HF credential stores.

`validate_selection.py` checks the real step512 checkpoint, 16,384 committed
rows, and its actual prior student ancestry. Terminal cloud states are simulated
only inside deleted temporary test fixtures; the actual running cloud task is
observed and rejected for production selection. These CPU tests do not establish
GPU quality, select a delivery model, stop training, package weights, or upload.

The existing adaptive use of the evaluation sets remains disclosed. Selecting a
checkpoint does not create a new blind test or change a scorer or denominator.
