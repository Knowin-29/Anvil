# Coordinate-objective continuation v2

The initial coordinate implementation was cancelled during model loading with zero committed training updates because supplemental point formats needed explicit routing. This immutable v2 fixes that routing; no interrupted-attempt weights are inherited.

The previous geometry replay failed full native quality: mask IoU0.151372, affordance IoU0.127772 and rotation111.199886 degrees; public macro72.304286. It is rejected, not an accepted final model. The original five-day deadline was missed.

Task t-20261104074431-vgjvs branches from the better evaluated format MOPD256 parent. It preserves the original5.22M-record SFT ancestry and all4.756M spatial training records. The four rejected geometry_* checkpoints remain nonancestor backups and their exposures are excluded from the new model totals.

The new loss retains every original answer token and adds scalar-normalized coordinate CE, emphasizing integer image coordinates and the leading fractional digits of camera positions. Only dedicated mask, affordance and PRISM steps use peak LR2e-5; other repair steps retain5e-6. This is an unproven optimization hypothesis, not a confirmed root cause or quality improvement. Nine CPU loss/gradient tests,768 native source/target-plan checks and12 supplemental point-format encoding checks passed. Supplemental XML/tuple/normalized point replay retains its original full-answer CE and is excluded from the new native-JSON coordinate auxiliary. No small GPU diagnostic jobs were submitted.

Repair8192 plans524288 exposures, then genuine on-policy MOPD256 plans16384 exposures including8192 teacher-scored native rows across all11 forms. Teachers, generation budgets, target annotations and benchmark scoring remain unchanged. Full580000 native,7366 public and10105 general evaluation is required before any acceptance. Key backups extend31 to35 and include full adapters, optimizers, schedulers and32 rank RNG states. The new24h operational window is not a promise of model quality.

Source and receipts only: no weights, datasets or credentials. Absolute paths identify the mounted execution environment.
