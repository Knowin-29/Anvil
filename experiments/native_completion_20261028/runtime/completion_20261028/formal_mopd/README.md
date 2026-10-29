# Formal MOPD launch and intermediate diagnostics

These scripts prepare a new formal run only after the exact candidate's real optimizer-resume tests and equal-input 8/16-GPU throughput measurements complete. Presence of this directory does not imply formal training or final delivery has completed.

`candidate1_quality_review.json` records the evidence-backed admission decision: complementary gains on masks, spatial relations and action prediction, with weaker box/point/affordance results and unsolved pose accuracy disclosed. This is not universal teacher superiority or final model acceptance.

1. Review `diagnosis_32gpu/mopd_scaling_comparison.json` after both probes finish. Choose the measured topology and a multiple-of-128 update budget, retaining at least 24 hours before 2026-11-02 13:48 CST for final validation, correction and upload.
2. Run `prepare_plan.py --output NEW_RUN --nodes 1_OR_2 --steps UPDATES`. It freezes actual runtime, quality, lineage and throughput evidence, writes eligibility, the formal training YAML, and intermediate evaluation YAMLs. It does not submit.
3. Run `controller.py --plan NEW_RUN/plan.json`. It submits formal training and checkpoint128/512/1024/final public diagnostics as sealed checkpoints appear, with at most one8GPU evaluation task and at most32 campaign GPUs. Ambiguous cloud submissions require inspection, never blind retry.
4. Review each `evaluation/STEP/report.json` while training continues. Reports use all seven public benchmarks and10105general examples, unchanged scorers and the paper's RefSpatialLocation100 comparison. These adaptively used sets are not a new blind test. None of these reports substitutes for full580000native final validation.
5. After formal completion and review, use `../final_delivery/` for full native/public/general verification, lineage, accepted-quality packaging, complete HF upload and every-file remote hashes.

`cpu_validation.json` verifies smoke-checkpoint rejection, frozen-code change detection, unchanged public scoring and CLI imports. Actual formal GPU execution has not yet occurred. The trainer is the unchanged native_mopd_v6.py, already separately tested with the exact candidate. Server HF credentials are never edited.
