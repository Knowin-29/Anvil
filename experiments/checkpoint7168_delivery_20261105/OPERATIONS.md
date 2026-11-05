# Anvil checkpoint7168 final delivery

## Final selection and acceptance

The latest user instruction accepts the final model on seven public benchmarks, with mean above 68 and scores relatively close to the paper. `public_acceptance_policy.json` interprets the latter using the previously requested 10% relative tolerance. Checkpoint7168 meets both conditions: mean 72.272857, seven of seven within tolerance. No additional DLC submission or training is required.

`public_quality_review.json` binds this decision to the full measured report, checkpoint specification and revised policy. The earlier `quality_review.json` remains the historical rejection under the former native-inclusive gate. Native Anvil-Eval v1.1 remains 50.623671; mask IoU 0.146931, affordance IoU 0.123545 and rotation error 113.631 degrees. Native metrics are not a gate under the revised public-benchmark acceptance scope. No scorer, prediction or score was altered.

## Completed evaluation and training ancestry

DLC `t-20261105200557-zxq4k` succeeded on one 8×A800 node. All 580,000 native, 7,366 public and 10,105 general examples completed. The evaluator and exact BF16 merged model are sealed by `evaluation_spec.json`. The immutable inference-time report retains its original pre-publication status flags; the release review and final verification receipt record subsequent acceptance and publication.

The selected visual repair checkpoint7168 inherits completed coordinate MOPD256 and earlier MOPD stages. It contains 458,752 visual-repair sample exposures. The planned repair8192 and subsequent visual MOPD stage did not complete. Earlier training stopped on storage quota; all logged losses and gradient norms were finite, and the language and visual adapters changed. See `training_update_audit.json`.

## HF publication

`checkpoint_collection/complete.json` verifies 37 complete checkpoints at revision `8bbfbd28ccc17fde47f852570bf23e9ec73b9791`: five featured stages plus 32 archive stages. `publish_public_final.py` adds the evaluated root merged BF16 model and final adapter while retaining this layout. It validates local model hashes, binds the revised gate, locks the Hub writer, uses an explicit parent revision, and verifies the complete remote inventory. `final_release/verification_receipt.json` pins the accepted final revision and every file hash.

Model: https://huggingface.co/Knowin-29/Anvil-MOPD-9B

Data: https://huggingface.co/datasets/Knowin-29/Anvil-Ben at `a2bb5d1c94321de3bab02278344d41a2f8912d2f` (71 files; 63 annotation shards).

The root model card provides highlights, actual public scores and usage. Full native/general metrics and historical decisions live under `docs/evaluation/`. Root `backup/` and `evidence/` directories have been replaced by the organized checkpoint and documentation layout.

`finalize_selected.py` is the superseded native-gated publisher and must not be used for this release. The unused continuation preparer is retained only in `unused_drafts/`; it was not run or submitted.

## Cleanup and source delivery

`cleanup_selected.py` validates the new acceptance and the revision chain from original upload through checkpoint organization to final publication. It removes only exact remotely verified local weight and training-state copies. It requires all campaign jobs idle, matching remote hashes, no local readers and unchanged inode identities. Dry-run and deletion receipts record each file. Logs, datasets, metadata, source, base models, teachers and Codex conversations are preserved.

Unused storage reservations were released separately: 147,102,629,888 allocated bytes (137 GiB); `unused_reserve_release.json` records this operation. Existing optimizer cleanup ledgers remain historical and must not be replayed.

`sync_source.py` publishes final scripts, policies, reports and receipts to GitHub main, verifies the branch head, and creates the source ZIP linked from `03_Delivery/Anvil_Source_Clean.zip`. `source_git_sync.json` and `source_bundle.json` record the final source delivery.

Never change server HF login or credential stores. All authentication uses the session-authorized transient credential.
