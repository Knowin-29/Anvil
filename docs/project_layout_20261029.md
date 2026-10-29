# Anvil 项目入口

2026-10-21 整合，2026-10-29 更新。后续开发使用 `01_Code` 的
[Knowin-29/Anvil / main](https://github.com/Knowin-29/Anvil/tree/main)。

| 目录 | 当前用途 |
| --- | --- |
| `01_Code/` | 维护源码，包含 native 数据合同、训练、评测和发布检查 |
| `02_Paper/` | 当前论文 PDF、LaTeX 源码及图稿素材 |
| `03_Delivery/` | 网站、源代码交付包、发布证据和远端版本收据 |
| `04_Experiments/` | GSA 历史实验、模型资产、运行依赖和恢复记录 |

## 当前状态（北京时间 2026-10-29）

原始语料为 5,800,000 条记录 / 160,000 张图像，其中完整 SFT 训练记录
5,220,000 条、留出 580,000 条。新增 QA、空间和 PRISM 数据另外统计，
不计作原始语料，也不把实验总采样次数当成唯一训练记录数。

正式 MOPD 任务 `t-20261029095820-px5rk` 已于 09:58 提交并实际开始训练，
使用 16 张 A800，计划 1664 次更新、global batch 32。精确教师与学生候选、
数据合同和训练配置均已固定。中途第 128 / 512 / 1024 / 1664 步自动进行
七项公开基准和通用能力验证。完整最终验证、质量验收和 HF 权重上传仍待完成；
当前不能把 SFT adapter、运行测试或 HF 占位仓库视为最终交付。

实时状态与实际覆盖量：
`/Knowin/sim/weilinruan/Anvil/recovery_20261020/completion_20261028/campaign_status.json`。
正式任务的 `formal_mopd/runs/candidate1_20261029/` 保存冻结计划和训练日志。
最终验收要求同一个 checkpoint 完成评测，再核验 HF 固定 commit 下的所有文件。

## 源码分支与历史资产

GSA 当前源码、配置、实验文档和可编辑图稿已保存在
[RWLinno/GSA_Forge / ICLR27](https://github.com/RWLinno/GSA_Forge/tree/ICLR27)，
2026-10-29 核验 commit `48f9d4b90473f5427bbca07879ed2a2a355120ba`。
GSA 的旧入口全文已无损归档；它是历史实验来源，不另设 Anvil 开发入口。
源代码同步不代表所有原始数据、实验产物或权重已上传，因此这些独有资产仍保留。

最新源码与任务收据见 `03_Delivery/CURRENT_CAMPAIGN.json`。
`03_Delivery/CURRENT_RELEASE.json` 保留原来的发布核验时点，不充当实时训练进度。

## 兼容路径

`01_Anvil` 指向 `01_Code`；`01_Anvil_arxiv` 指向仍有旧进程依赖的
`04_Experiments/runtime_source`；`arxiv` 指向 `03_Delivery/evidence`。
上一级 `GSA_collect`、`GSA_Forge`、`Anvil_ICLR27`、`Anvil.github.io` 为兼容路径。
旧源码和图稿的恢复证据保存在 `04_Experiments/provenance/`。
服务器 HF 凭证配置未修改。
