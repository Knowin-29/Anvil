# Anvil 故障、恢复和清理复核

核查时间：2026-10-31T12:46:47.206963+08:00（北京时间）。

截图对应 `t-20261029174537-bvfnb`。云端 Failed；原始四节点日志 SHA256 与故障分析一致。修复阶段完成 2048 步。MOPD 完成 126 步，第 127 步在单元素 ALLREDUCE（序号 5385）等待 600 秒后退出；其他 rank 已入队，rank 1 尚未入队。未发现更早的 OOM 或数值异常。长生成导致进程迟到是较可能解释，旧任务缺乏逐卡阶段记录，无法断定 rank 1 当时具体卡在生成、数据读取还是其他同步前操作。

VPC 入流量由约 1.66 KiB/s 升至 2.24 KiB/s，增加不到 1 KiB/s，不能作为网络拥塞的证据。RDMA 通信降至近零、GPU 利用率高而显存带宽利用率低，与集合通信等待相符。监控曲线本身无法证明网卡故障。

完整 522 万记录 SFT 与修复 checkpoint-2048 保留。旧 MOPD 首次保存设在 128 步，因第 127 步失败，126 次更新没有 checkpoint 可恢复，实际损失约 6718 秒（1 小时 52 分）。保存间隔对长生成任务过粗，是这次恢复成本较高的直接原因。

恢复任务 `t-20261030102801-c2p47` 已确认 Success，完成 1536 步。它复用修复 checkpoint-2048，仅将 MOPD 从第 0 步重新运行。NCCL 等待上限已调整为 3600 秒，并每 32 步保存完整 checkpoint。最终 checkpoint-1536 的 38 个载荷文件（约 2.08 GB）、优化器 step1536、调度器、32 份 RNG、496 个有限适配器张量与完整曝光前缀已校验；这些是完整中间状态，不是只有推理权重。没有额外执行 GPU 断点恢复实验。

目前 32 卡进行该模型的完整验证。公开七项 benchmark 已完成，宏平均 69.3671%，论文目标 74.27%；Where2Place、相对深度等仍有明显差距，尚未通过最终发布验收。正在等待完整 58 万原生验证结束后，控制器将在总量不超过 32 卡条件下提交补训。补训继承上述 checkpoint-1536，计划 repair4096 + MOPD768；因优化目标调整，新阶段重新建立优化器，但保留已有模型权重和全部训练谱系。既有 SFT 与上轮修复/MOPD 不重做。原生形式均衡采样和额外深度/点监督是已固定的修正，效果需以最终验证为准。

最新 HF 核查：`Knowin-29/Anvil-MOPD-9B`，commit `8c2cfe01656473bc2766e2e4fb2af619fb36384a`，只有 `.gitattributes` 与 `README.md`。最终模型和关键 checkpoint 尚未上传，因此不能按“已上传备份”删除现有唯一权重。

此前已删除 20 份可重建的历史合并权重副本，释放 359.6 GiB；本轮没有重复计算这笔释放，也未额外删除原始权重/数据/日志。完成的修复与恢复 MOPD 共保留 56 份 checkpoint（约 108.4 GiB），仍与最终交付、续训谱系有关。失败 MOPD 的全部文件仅约 11.5 MiB，保留用于排错和覆盖量审计。

系统盘现约 69 GiB 可用；共享盘约 2.2 TiB 可用。`/tmp` 本盘占用约 145 MiB。`/root/anvil-storage-relief-20260916` 已是共享盘软链接，不再占原来的约 70 GB 系统盘空间。没有删除 Codex 会话历史。最终权重及关键 checkpoint 完整上传且按固定 commit 逐文件核验后，再清除无运行依赖的项目本地权重副本；训练/验证仍需用的数据现在保留，不能将共享原始数据一并清空。

证据与实时状态：
- `direct_recovery_20261030/failure_analysis.json`
- `direct_recovery_20261030/mopd_checkpoint1536_verification.json`
- `verification_20261031/hf_inventory_latest.json`
- `verification_20261031/cleanup_disposition.json`
- `campaign_targeted_status.json`

源码已同步 Knowin-29/Anvil 的 main 分支，commit 39c8cb7365c5bf6fb0ca9633b41f665866ae2eed；源码同步与模型上传分开核验。服务器 HF 凭证未修改。
