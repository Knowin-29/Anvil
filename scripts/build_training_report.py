#!/usr/bin/env python3
"""Build GSA Training & Evaluation system report."""
import argparse
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_report")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "training_eval_system.html"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = HTML_TEMPLATE.replace("{{GENERATED_TIME}}", now)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info(f"Report written: {out} ({out.stat().st_size / 1024:.1f} KB)")


# --- FULL HTML TEMPLATE ---
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GSA 训练与测评 · 全流程记录</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,"PingFang SC","Noto Sans SC",sans-serif;line-height:1.7;padding-bottom:60px}
nav{position:sticky;top:0;z-index:100;background:#161b22;border-bottom:1px solid #30363d;padding:12px 40px;display:flex;gap:18px}
nav a{color:#8b949e;text-decoration:none;font-size:13px;padding:4px 12px;border-radius:6px}
nav a:hover{color:#58a6ff}
nav a.active{background:#1f6feb;color:#fff}
.hero{background:linear-gradient(135deg,#1a2332,#0f3460,#16213e);padding:48px 40px;border-bottom:3px solid #d2a8ff}
.hero h1{font-size:30px;color:#f0f6fc;font-weight:700}
.hero .sub{color:#8b949e;font-size:14px;margin-top:8px}
.container{max-width:1280px;margin:0 auto;padding:32px 40px}
h2{font-size:22px;color:#f0f6fc;margin:36px 0 12px;padding-bottom:8px;border-bottom:2px solid #30363d}
h3{font-size:16px;color:#d2a8ff;margin:20px 0 8px}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin:12px 0}
table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #21262d}
th{color:#58a6ff;font-weight:600;background:#161b22}
code{font-family:"JetBrains Mono",monospace;background:#21262d;padding:1px 6px;border-radius:4px;color:#79c0ff;font-size:12px}
pre{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;overflow-x:auto;font-size:12px;color:#c9d1d9;margin:12px 0}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600}
.bg{background:#1b4332;color:#3fb950}.by{background:#2d1f12;color:#f0c674}.br{background:#2d1218;color:#f85149}.bb{background:#0d2137;color:#58a6ff}
.flow{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:16px 0}
.flow-step{background:#1f6feb;color:#fff;padding:6px 14px;border-radius:6px;font-size:12px}
.flow-arrow{color:#58a6ff;font-size:16px}
.grid{display:grid;gap:12px;margin:12px 0}
.grid-4{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.stat-num{font-size:28px;font-weight:700;font-family:monospace;color:#58a6ff}
.stat-label{font-size:11px;color:#8b949e;margin-top:4px}
footer{text-align:center;padding:32px;color:#6e7681;font-size:12px;border-top:1px solid #21262d;margin-top:48px}
</style></head><body>

<nav>
<a href="/data-sources">数据源说明</a>
<a href="/flywheel">数据飞轮说明</a>
<a href="/training-eval" class="active">训练测评说明</a>
</nav>

<div class="hero">
<h1>GSA 训练与测评 · 全流程记录</h1>
<div class="sub">Qwen3.6-27B Teacher → GRPO → OPD/GKD 蒸馏 → Qwen3.5-9B Student | ms-swift 4.2.3 | 2节点×8卡H200</div>
</div>

<div class="container">

<h2>1. 技术路线</h2>
<div class="card">
<pre>
GKD Student ←── 27B Teacher ←── Base Model ──→ 9B Export ──→ GSA MoE
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
              直接 SFT          LLM LoRA          GKD 蒸馏
                  │                 │                 │
                  ▼                 ▼                 ▼
               ckpt.A           ckpt.B            ckpt.C
                  │                 │                 │
                  └────── 统一评测 (业务日志 + benchmarks) ──────┘
                                    │
                         结果分析 → 升级模型 → 改进测评
</pre>
</div>

<h2>2. 运行环境</h2>
<div class="card">
<table>
<tr><th>组件</th><th>版本</th><th>备注</th></tr>
<tr><td>Python</td><td>3.11.11</td><td></td></tr>
<tr><td>PyTorch</td><td>2.8.0+cu128</td><td></td></tr>
<tr><td>CUDA</td><td>12.8 (V12.8.93)</td><td></td></tr>
<tr><td>transformers</td><td>5.2.0</td><td>Qwen3.5 必须</td></tr>
<tr><td>ms-swift</td><td>4.2.3</td><td>源码 /path/to/storage/ms-swift</td></tr>
<tr><td>DeepSpeed</td><td>0.17.6</td><td></td></tr>
<tr><td>flash-attn</td><td>2.8.1</td><td></td></tr>
<tr><td>flash-linear-attention</td><td>≥0.4.2</td><td>Qwen3.5 GatedDeltaNet 必需</td></tr>
<tr><td>peft</td><td>0.19.1</td><td></td></tr>
<tr><td>硬件 (训练)</td><td>2节点 × 8卡 H200</td><td>DLC 分布式</td></tr>
<tr><td>硬件 (调试)</td><td>1卡 L20X-143G</td><td>DSW 单卡</td></tr>
</table>
<p style="margin-top:12px;font-size:12px;color:#f0c674">⚠️ 关键限制: vLLM≥0.17.0 显式排除 transformers 5.2.* — 训练环境和 vLLM 推理环境不能共存</p>
</div>

<h2>3. 数据管线</h2>

<h3>3.1 数据源注册表 (73 活跃源)</h3>
<div class="card">
<table>
<tr><th>类别</th><th>数据源数</th><th>总量</th><th>典型数据集</th></tr>
<tr><td>GSA 自有标注</td><td>3</td><td>~246K</td><td>Desktop bbox (97K), Log sqlite (60K), GSA_volc flywheel (87K)</td></tr>
<tr><td>Point/Affordance</td><td>5</td><td>~2.5M</td><td>Pixmo (119万过滤后), PRISM, Robopoint, Roborefit, FSD</td></tr>
<tr><td>空间关系</td><td>5</td><td>~350K</td><td>RefSpatial (226K), EmbSpatial (25K), SAT, VSR, WhatsUp</td></tr>
<tr><td>目标定位 (bbox)</td><td>3</td><td>~320K</td><td>RefCOCO (120K HF), TallyQA (86K), flywheel G_grounding</td></tr>
<tr><td>通用 VQA</td><td>14+</td><td>~800K</td><td>DVQ, OCRVQA, PlotQA, FigureQA, localized_narratives</td></tr>
</table>
</div>

<h3>3.2 数据质量控制</h3>
<div class="card">
<table>
<tr><th>处理步骤</th><th>具体操作</th><th>效果</th></tr>
<tr><td>数据集级去重</td><td>识别并禁用 pointctl 预配比混合文件</td><td>移除 3M 重复样本</td></tr>
<tr><td>桌面模板去重</td><td>按 (image, response) 分组去重</td><td>48万 → 12万 独立标注</td></tr>
<tr><td>Pixmo 过滤</td><td>规则: 点数&gt;15 / 重复点 / 越界</td><td>127万 → 119万 (93.8%保留)</td></tr>
<tr><td>GSA_volc 路径修复</td><td>三级回退匹配 (直接/basename/源名映射)</td><td>可达率 23.6% → 99.99%</td></tr>
<tr><td>Desktop 隔离</td><td>拉出至 stage2_grpo_eval</td><td>防止 SFT 污染</td></tr>
<tr><td>7 项质量审计</td><td>图片可达/JSON合法/bbox合法/去重/均衡/噪声</td><td>全部通过</td></tr>
<tr><td>RefCOCO 下载</td><td>HF 125 shards → ms-swift grounding format</td><td>新增 120K bbox</td></tr>
</table>
</div>

<h3>3.3 训练集统计</h3>
<div class="grid grid-4">
<div class="card"><div class="stat-num">~4.5M</div><div class="stat-label">SFT 总量</div></div>
<div class="card"><div class="stat-num">73</div><div class="stat-label">活跃数据源</div></div>
<div class="card"><div class="stat-num">99.99%</div><div class="stat-label">图片可达率</div></div>
<div class="card"><div class="stat-num">[0,1000]</div><div class="stat-label">坐标系统</div></div>
</div>
<div class="card">
<table>
<tr><th>成分</th><th>占比</th></tr>
<tr><td>bbox (grounding)</td><td>12-15%</td></tr>
<tr><td>point / affordance</td><td>~40%</td></tr>
<tr><td>general / text VQA</td><td>~45%</td></tr>
<tr><td>choice (选择题)</td><td>~1%</td></tr>
</table>
</div>

<h2>4. 训练历史</h2>

<h3>4.1 27B Teacher (Qwen3.6-27B)</h3>
<div class="card">
<table>
<tr><th>版本</th><th>基底</th><th>数据量</th><th>配置</th><th>结果</th><th>备注</th></tr>
<tr><td><span class="badge bb">v4</span></td><td>base 27B</td><td>3.35M</td><td>6卡 ZeRO-3, lr=1e-5, 2ep</td><td>step5000: loss=0.23, acc=92%</td><td>基线教师模型</td></tr>
<tr><td><span class="badge bg">v5</span></td><td>v4 ckpt-5000</td><td>3.53M (+飞轮)</td><td>lr=5e-6, 1ep</td><td>step7000: loss=0.17, acc=93%</td><td>飞轮数据补齐</td></tr>
<tr><td><span class="badge by">v6</span></td><td>v4 ckpt-5000</td><td>3.12M (重平衡)</td><td>lr=5e-6, 1ep</td><td>loss=0.23, acc=91%</td><td>Desktop 移除</td></tr>
<tr><td><span class="badge bg">v7</span></td><td>base 27B</td><td>4.5M (完整)</td><td>4卡 ZeRO-3, lr=1e-5</td><td>step4500: loss=0.24</td><td>干净训练，无污染</td></tr>
</table>
</div>

<h3>4.2 9B Student (Qwen3.5-9B)</h3>
<div class="card">
<table>
<tr><th>方案</th><th>配置</th><th>状态</th><th>Checkpoints</th></tr>
<tr><td>Full SFT (v1)</td><td>ZeRO-3, 4卡, lr=1e-5, 全参数</td><td><span class="badge bg">已完成</span></td><td>1000/2000/3000</td></tr>
<tr><td>LLM LoRA (v1)</td><td>ZeRO-2, 16卡, lr=1e-4, rank=128</td><td><span class="badge by">训练中</span></td><td>目标: 1000/3000/5000/7000</td></tr>
<tr><td>GKD 蒸馏</td><td>27B teacher → 9B student</td><td><span class="badge bb">待定</span></td><td>TBD</td></tr>
</table>
</div>

<h3>4.3 9B LoRA 训练配置 (2026-06-17)</h3>
<div class="card">
<table>
<tr><th>参数</th><th>值</th><th>说明</th></tr>
<tr><td>模型</td><td>Qwen3.5-9B</td><td>冻结 vision encoder + aligner</td></tr>
<tr><td>训练方式</td><td>LoRA</td><td>rank=128, alpha=256</td></tr>
<tr><td>LoRA 目标层</td><td><code>language_model.*.{q,k,v,o,gate,up,down}_proj</code></td><td>仅训练 LLM 投影矩阵</td></tr>
<tr><td>硬件</td><td>2节点 × 8卡 H200 = 16 GPU</td><td>DLC 分布式</td></tr>
<tr><td>有效批大小</td><td>16 GPU × 4 × accum=2 = 128</td><td></td></tr>
<tr><td>学习率</td><td>1e-4, cosine 衰减, warmup 5%</td><td>LoRA 标准</td></tr>
<tr><td>序列长度</td><td>max_length=4096</td><td>padding_free + packing</td></tr>
<tr><td>并行策略</td><td>DeepSpeed ZeRO-2</td><td>LoRA 参数量小</td></tr>
<tr><td>保存策略</td><td>每200步, 滑动保留最近3个</td><td>+ protect_milestones.sh</td></tr>
<tr><td>监控</td><td>WandB (GSA_Forge_9B_LoRA)</td><td>实时 loss/acc</td></tr>
</table>
</div>

<h2>5. GRPO Reward 设计</h2>
<div class="card">
<p style="margin-bottom:12px;color:#8b949e">6 个维度特化评分器 (gsa_reward_plugin.py)，与 ms-swift 内置 format reward 组合使用:</p>
<table>
<tr><th>评分器</th><th>校验字段</th><th>说明</th></tr>
<tr><td>score_grounding</td><td>object + bbox + confidence</td><td>物体名称+坐标完整性</td></tr>
<tr><td>score_spatial</td><td>subject/predicate/object/description</td><td>空间关系三元组+描述</td></tr>
<tr><td>score_identity_attr</td><td>object + attributes dict</td><td>属性字典非空+合理</td></tr>
<tr><td>score_obj_state</td><td>state_type / state_value</td><td>状态类型-值匹配</td></tr>
<tr><td>score_action_precondition</td><td>can_do_now / blocked_by</td><td>逻辑一致性</td></tr>
<tr><td>score_action_interface</td><td>action_type / grasp_point_2d / jaw_axis_2d</td><td>抓取点+方向合理</td></tr>
</table>
</div>

<h2>6. 评测体系</h2>

<h3>6.1 评测数据</h3>
<div class="card">
<table>
<tr><th>数据集</th><th>规模</th><th>用途</th></tr>
<tr><td>v0616 业务日志</td><td>100K (标准: 前10K)</td><td>主评测集</td></tr>
<tr><td>Embodied Arena</td><td>16 tasks</td><td>内部 GSA v3 benchmark</td></tr>
<tr><td>RoboSpatial-Eval</td><td>-</td><td>空间理解专项</td></tr>
</table>
</div>

<h3>6.2 评分器 (score_vlabel_v4.py)</h3>
<div class="card">
<table>
<tr><th>答案类型</th><th>评分方法</th><th>指标</th></tr>
<tr><td>bbox_grounding</td><td>解析 GT/PRED bbox 计算 IoU</td><td>IoU@0.5 准确率, Mean IoU</td></tr>
<tr><td>point/affordance</td><td>点落在 GT bbox 内 or PCK 距离</td><td>命中率</td></tr>
<tr><td>attribute/state</td><td>大小写不敏感精确匹配 + 归一化</td><td>准确率</td></tr>
<tr><td>closed_choice</td><td>精确匹配 (yes/no/ABCD)</td><td>准确率</td></tr>
<tr><td>free_text</td><td>VLM judge (397B API 语义等价判断)</td><td>语义等价率</td></tr>
</table>
</div>

<h3>6.3 评测结果 (27B v7-3000, 213样本分层抽样)</h3>
<div class="grid grid-4">
<div class="card"><div class="stat-num" style="color:#3fb950">99.5%</div><div class="stat-label">bbox 格式解析率</div></div>
<div class="card"><div class="stat-num">0.838</div><div class="stat-label">Mean IoU</div></div>
<div class="card"><div class="stat-num">0.902</div><div class="stat-label">Median IoU</div></div>
<div class="card"><div class="stat-num" style="color:#3fb950">95.8%</div><div class="stat-label">IoU@0.5</div></div>
</div>
<div class="card">
<table>
<tr><th>指标</th><th>数值</th><th>说明</th></tr>
<tr><td>IoU@0.7</td><td>89.6%</td><td></td></tr>
<tr><td>IoU@0.9</td><td>50.5%</td><td></td></tr>
<tr><td>Exact match</td><td>17.4%</td><td>严格匹配 (非bbox维度)</td></tr>
<tr><td>训练 loss</td><td>0.68 → 0.23</td><td>收敛良好</td></tr>
<tr><td>Token accuracy</td><td>84% → 92%</td><td></td></tr>
</table>
</div>

<h2>7. 已知问题与解决方案</h2>

<h3>7.1 环境问题</h3>
<div class="card">
<table>
<tr><th>问题</th><th>原因</th><th>解决</th></tr>
<tr><td>合成数据 NAS 不可访问</td><td>/path/to/workspace 路径未挂载到阿里云</td><td>从本地可达路径重建 registry</td></tr>
<tr><td>transformers 4.57 不支持 Qwen3.5</td><td>架构 qwen3_5 需要 ≥5.2</td><td>升级到 5.2.0</td></tr>
<tr><td>ms-swift 参数名变更</td><td>4.2.3 用 tuner_type 非 train_type</td><td>更新所有配置文件</td></tr>
</table>
</div>

<h3>7.2 训练稳定性</h3>
<div class="card">
<table>
<tr><th>问题</th><th>原因</th><th>解决</th></tr>
<tr><td>resume_from_checkpoint 失败</td><td>save_only_model=true 无 optimizer state</td><td>用 --model 指向权重目录</td></tr>
<tr><td>关键 checkpoint 被删除</td><td>save_total_limit 滚动删除</td><td>外部 protect_milestones.sh</td></tr>
<tr><td>多实验端口冲突</td><td>共用 MASTER_PORT=29500</td><td>每实验分配独立端口</td></tr>
<tr><td>灾难性遗忘</td><td>续训时移除 desktop 数据</td><td>从 base 全新训练 + RefCOCO</td></tr>
</table>
</div>

<h3>7.3 数据质量</h3>
<div class="card">
<table>
<tr><th>问题</th><th>原因</th><th>解决</th></tr>
<tr><td>pointctl 重复 3M 样本</td><td>预配比混合文件与子集同源</td><td>禁用 pointctl 用独立子集</td></tr>
<tr><td>桌面 bbox 模板膨胀 (4x)</td><td>每个标注有 4 个 prompt 变体</td><td>按 (image, response) 去重</td></tr>
<tr><td>GSA_volc 77% 图片不可达</td><td>HF 下载扁平化目录结构</td><td>三级 basename 回退匹配</td></tr>
<tr><td>Desktop 数据污染 SFT</td><td>模型学入 desktop 特有分布</td><td>隔离至 stage2 从 base 重训</td></tr>
</table>
</div>

<h2>8. DLC 部署 (2026-06-17)</h2>
<div class="card">
<p style="margin-bottom:8px;color:#8b949e">DLC 环境与本地 DSW 差异大，连续解决 8 个错误:</p>
<table>
<tr><th>#</th><th>错误</th><th>解决</th></tr>
<tr><td>1</td><td><code>model: None</code></td><td>CLI --model 显式传递</td></tr>
<tr><td>2</td><td><code>dataset: []</code> 为空</td><td>CLI --dataset 传递</td></tr>
<tr><td>3-4</td><td>remaining_argv / --dtype</td><td>去掉 --config, 用 --torch_dtype</td></tr>
<tr><td>5-6</td><td>No module 'swift' / 错误解释器</td><td>PYTHONPATH=/path/to/storage/ms-swift</td></tr>
<tr><td>7-8</td><td>DeepSpeed local_world_size:1 / No __main__</td><td>_swift_entry.py 调用 cli_main()</td></tr>
</table>
<p style="margin-top:12px;font-size:13px"><b>最终方案:</b> <code>PYTHONPATH=/path/to/storage/ms-swift</code> + <code>_swift_entry.py</code> (4行) + 全部参数 CLI 传递</p>
</div>

<h2>9. 运维命令</h2>
<div class="card">
<pre>
# === 训练 (阿里云 DLC) ===
cd /path/to/storage/anvil/GSA_Forge
bash scripts/run_sft_27b.sh              # 27B SFT
bash scripts/run_grpo_27b.sh             # 27B GRPO
bash scripts/run_opd_9b.sh               # OPD 蒸馏
bash scripts/run_sft_9b_lora.sh          # 9B LoRA

# === 评测 ===
bash eval/run_logs_eval.sh               # 10K 日志评测
bash eval/run_output_benchmark_suite.sh  # Benchmark 套件
bash eval/run_compare_checkpoints_dlc.sh # 多模型对比

# === 数据准备 ===
bash scripts/run_data_prepare.sh         # 从 registry 构建训练数据
python trains/build_msswift_from_registry.py  # 核心数据构建
</pre>
</div>

</div>
<footer>GSA Training & Evaluation | Generated {{GENERATED_TIME}} | GSA_Forge</footer>
</body></html>
"""

if __name__ == "__main__":
    main()
