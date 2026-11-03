# Correct geometry checkpoint binding during merged-weight preupload

After geometry MOPD completed all 256 updates and all 31 key checkpoints were verified on HF, merged-blob preupload stopped because its adapter comparison still referenced `format_mopd/checkpoint-256`. The verified geometry adapter correctly differed from that parent. No final model was published by the failed worker.

The immutable v2 worker binds the evaluation spec, completed training checkpoint and remote `geometry_mopd/checkpoint-256` adapter. It records 31 verified backups and its source hash. The v2 publisher verifies that corrected worker identity before publication. The original source and policy are retained unchanged; the new release policy also verifies their hashes.

Validation: 8 CPU regression tests pass, including stale parent weights, a different checkpoint, missing or duplicate backup keys, incomplete native scope and unsafe shard paths. Nine checks against actual campaign metadata and policy hashes also pass. No GPU diagnostic jobs or training/evaluation changes were made.

Training succeeded and all 31 key checkpoints are verified at HF revision `6bca12d523756a32ac20e9eaf4943adb57a5975a`. Full native/public/general evaluation and report-bound quality approval remain required. Native mask, affordance and DoF capabilities are explicitly part of acceptance. Blob staging alone is not a final release.
