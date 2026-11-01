# Anvil 项目入口

2026-10-21 整合，2026-11-02 更新。后续开发使用 `01_Code` 的
[Knowin-29/Anvil / main](https://github.com/Knowin-29/Anvil/tree/main)。

| 目录 | 当前用途 |
| --- | --- |
| `01_Code/` | 维护源码，包含 native 数据合同、训练、评测和发布检查 |
| `02_Paper/` | 当前论文 PDF、LaTeX 源码及图稿素材 |
| `03_Delivery/` | 网站、源代码交付包、发布证据和远端版本收据 |
| `04_Experiments/` | GSA 历史实验、模型资产、运行依赖和恢复记录 |

## 当前状态（北京时间 2026-11-02）

原始语料为 5,800,000 条记录 / 160,000 张图像，其中完整 SFT 训练记录
5,220,000 条、留出 580,000 条。新增 QA、空间和 PRISM 数据另外统计，
不计作原始语料，也不把实验总采样次数当成唯一训练记录数。

新任务 `t-20261102021323-bhkpt` 于 02:13 提交，使用32张A800，
从完整评测得分70.1171%的MOPD768父模型继续，执行低学习率repair2560及
正式MOPD512，global batch64。原始SFT不重跑。上一轮focus实验训练和
全量验证均已完成，但七项公开基准均值退步到67.6957%（论文74.27%），
已拒绝作为最终模型；它的权重和曝光量不进入新模型祖先谱系。
新增51322条SAT训练记录，保留原生、FSD和常识replay。训练后自动执行
完整580000条native、7366条公开基准和10105条通用验证。

更早的MOPD127通信超时造成126个未保存更新丢失，其余SFT和repair保留。
当前repair每64步、MOPD每32步保存完整状态；repair64已通过CPU完整性检查。
HF已固定版本校验14个完整关键断点，本轮新增5个，共19个。
最终质量验收、合并模型发布和全仓校验仍未完成。服务器HF凭证未修改。

实时状态与实际覆盖量：
`/Knowin/sim/weilinruan/Anvil/recovery_20261020/completion_20261028/campaign_balanced_status.json`。
当前任务的 `balanced_repair_20261102/` 保存冻结配置与训练日志；历史任务保留。
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
