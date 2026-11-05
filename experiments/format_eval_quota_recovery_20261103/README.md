# Full evaluation recovery after shared-storage quota failure

The format continuation completed repair 8192 and genuine MOPD 256. All 27 declared key checkpoints are backed up at Hugging Face revision `e5b8da0e1c9c5346d28359405b98c21713b81511`. The first full evaluation failed with `Errno 122: Disk quota exceeded` while writing compiler caches and prediction records. It did not invalidate or restart training.

The recovery uses the same trained checkpoint, merged model, frozen evaluation workers and scoring, and evaluation spec SHA256 `3b6a3c060f6bf9b06641692667d0a8747ddae0f4553f7f76274d1805adab5eca`. Four eight-GPU jobs cover 580,000 native, 7,366 public and 10,105 general records. The failed outputs remain in their original directory; this attempt writes fresh full outputs.

Compiler caches move to each node's `/tmp/anvil-format-eval-quota`. Each node checks 16 GiB of local headroom and a 64 MiB shared-storage write before launching workers. Cleanup removed only 126 superseded non-key repair saves (261,870,029,064 bytes), keeping all key checkpoints, MOPD saves, metadata, logs and data. Both historical cleanup ledgers are required by the final cleanup script.

The supervisor enforces the 32-GPU campaign limit, including queued or unknown tasks. The publisher requires all recovery jobs to succeed and an explicit quality review bound to the exact complete report. Preuploaded model blobs are not a final published release. Final quality was not yet established when this recovery snapshot was prepared.

CPU validation checked unchanged partition/resume logic, frozen workers/scoring and local cache routing. Seventeen cleanup-boundary tests passed. No additional small GPU experiment was submitted. Server credential stores were not changed.

These files are an auditable source snapshot. Runtime paths in policies and YAML refer to the shared production workspace; the authoritative runtime recovery directory is `/data/anvil/recovery_20261020/completion_20261028/format_eval_quota_recovery_20261103`.
