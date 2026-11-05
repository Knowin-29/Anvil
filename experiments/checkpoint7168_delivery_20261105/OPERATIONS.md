# Selected checkpoint 7168 delivery

The user selected repair checkpoint 7168, immediate checkpoint upload, and a full evaluation on one 8×A800 node. No further 32-GPU training is authorized by this delivery plan.

## Current outcome

Full DLC evaluation `t-20261105200557-zxq4k` succeeded. All 580,000 native, 7,366 public and 10,105 general records were verified. Seven public scores meet the user tolerance. Native v1.1 score is 50.62367136038148, below 58.815; `quality_review.json` rejects final acceptance. The original scorer has not been changed.

`organize_checkpoints.py --apply` completed the separately authorized checkpoint collection presentation. HF head is `8bbfbd28ccc17fde47f852570bf23e9ec73b9791`, with 37 checkpoints, five featured stages, actual measured results and the standalone pose interface. The original uploaded revision remains pinned in `checkpoint_upload/verification_receipt.json`; `checkpoint_collection/complete.json` verifies the current layout. No root merged model was published as an accepted final model.

The earlier `finalize_selected.py` and `cleanup_selected.py` remain gated and refer to the earlier collection parent. Do not invoke them by manufacturing a passing review; a future accepted candidate needs its own bound evaluation and a release plan against the current HF head.

The user was asked whether the 8-GPU resource remains available for corrective training and for how long, because their latest allocation was for this now-completed evaluation. No follow-up training has been submitted. All original checkpoint payloads remain available for continuation.

## Runtime

- Controller: `evaluate_selected.py`; inspect `status.json` and `controller_failure.json`.
- DLC task: `t-20261105200557-zxq4k`.
- Scope: 580,000 native + 7,366 public + 10,105 general examples.
- Complete checkpoint: `checkpoint_upload/verification_receipt.json` pins all 37 remotely verified checkpoints.
- Inference blob transfer: `preupload_selected.py`; `preupload_complete.json` confirms upload without changing the model repository head.
- Source evaluator files are sealed by `evaluation_spec.json`. Do not modify them.

## Final acceptance and publication

After the controller writes `tolerance_assessment.json`, review the complete results. Each public benchmark must reach at least 90% of its paper reference, and Anvil-Eval v1.1 must reach 58.815. Review all native groups, especially mask, affordance and DoF, together with general retention. The v1.1 mapping was sealed before inference; historical equivalence to the paper scorer is not established. Keep this provenance in technical evaluation materials.

Write `quality_review.json` using schema `anvil-final-quality-review-v1`, with `accepted`, a concrete rationale, `report_sha256`, `final_spec_sha256`, `native_score_sha256`, and `user_acceptance_policy_sha256`. Include `benchmark_assessments` keyed like the report, `native_group_assessments` keyed as `form::source`, and `general_assessments`. Each assessment must include `accepted` and its actual-results rationale. Do not manufacture passing scores or fabricate completion of the planned 8192-step repair or subsequent visual MOPD stage.

Then run the existing train-environment Python on `finalize_selected.py --prepare`, inspect `final_release/preview`, and run `finalize_selected.py --apply`. User authorization already covers publication when acceptance passes. The script refuses missing, failed or mismatched reviews and publishes the selected inference model and five featured stage directories in one atomic commit. Other checkpoints move to `checkpoints/archive/`; root `backup/` and `evidence/` paths are removed through byte-preserving server-side copies.

## Cleanup

Do not remove model/adapter/training-state or evaluator inputs while evaluation is active: protocol validation reads these sources. After `final_release/verification_receipt.json` verifies the exact release remotely, reconcile existing cleanup ledgers (411 + 25 historical optimizer removals), remove only remotely verified obsolete weight payloads without active readers, and release the remaining W/W2 reservations. Preserve logs, source provenance, checksums and all Codex conversations. After final verification, `cleanup_selected.py --output <new-direct-child-of-this-directory>` previews exact remotely verified local weight copies; add `--apply` only after reviewing that plan. It requires successful cloud evaluation, idle campaign locks, unchanged remote hashes and no local readers. This script retains non-key saves for separate superseded-stage cleanup and preserves all metadata and conversations. Final release code itself does not delete local files.

Never change HF login or server credential stores. Authentication uses only the session-authorized transient token.
