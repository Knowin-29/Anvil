#!/usr/bin/env python3
"""Build GSA Data Flywheel system report."""
import argparse, json, logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flywheel_report")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "data_flywheel_system.html"
VERIFIER_STATE = PROJECT_ROOT / "data" / "output" / "gsa_flywheel_exp" / "verifier_state.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    metrics = ""
    if VERIFIER_STATE.exists():
        try:
            st = json.loads(VERIFIER_STATE.read_text())
            tv = st.get('total_verified',0); ts = max(st.get('total_scanned',1),1)
            tr = st.get('total_needs_review', 0)
            tg = ts + st.get('total_rejected', 0) + tr
            metrics = f'<div class="grid grid-4"><div class="card"><div class="stat-num">{tg:,}</div><div class="stat-label">Generated (Total Labels)</div></div><div class="card"><div class="stat-num">{tv:,}</div><div class="stat-label">Verified</div></div><div class="card"><div class="stat-num">{ts:,}</div><div class="stat-label">Scanned</div></div><div class="card"><div class="stat-num" style="color:#3fb950">{round(tv/ts*100,1)}%</div><div class="stat-label">Pass Rate</div></div></div>'
        except Exception:
            pass
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = HTML_TEMPLATE.replace("{{TIME}}", now).replace("{{METRICS}}", metrics)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info(f"Report written: {out} ({out.stat().st_size/1024:.1f} KB)")

HTML_TEMPLATE = r"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GSA 数据飞轮</title><style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,"PingFang SC",sans-serif;line-height:1.7;padding-bottom:60px}nav{position:sticky;top:0;z-index:100;background:#161b22;border-bottom:1px solid #30363d;padding:12px 40px;display:flex;gap:18px}nav a{color:#8b949e;text-decoration:none;font-size:13px;padding:4px 12px;border-radius:6px}nav a:hover{color:#58a6ff}nav a.active{background:#1f6feb;color:#fff}.hero{background:linear-gradient(135deg,#1a2332,#0f3460);padding:48px 40px;border-bottom:3px solid #3fb950}.hero h1{font-size:30px;color:#f0f6fc}.hero .sub{color:#8b949e;font-size:14px;margin-top:8px}.container{max-width:1280px;margin:0 auto;padding:32px 40px}h2{font-size:22px;color:#f0f6fc;margin:36px 0 12px;padding-bottom:8px;border-bottom:2px solid #30363d}h3{font-size:16px;color:#3fb950;margin:20px 0 8px}.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin:12px 0}table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #21262d}th{color:#58a6ff;font-weight:600;background:#161b22}pre{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;overflow-x:auto;font-size:12px;margin:12px 0;white-space:pre;color:#c9d1d9}code{font-family:monospace;background:#21262d;padding:1px 6px;border-radius:4px;color:#79c0ff;font-size:12px}.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600}.bg{background:#1b4332;color:#3fb950}.by{background:#2d1f12;color:#f0c674}.br{background:#2d1218;color:#f85149}.bb{background:#0d2137;color:#58a6ff}.grid{display:grid;gap:12px;margin:12px 0}.grid-4{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}.stat-num{font-size:28px;font-weight:700;font-family:monospace;color:#58a6ff}.stat-label{font-size:11px;color:#8b949e;margin-top:4px}.warn{background:#2d1f12;border:1px solid #9e6a03;border-radius:8px;padding:12px 16px;color:#f0c674;font-size:13px;margin:12px 0}footer{text-align:center;padding:32px;color:#6e7681;font-size:12px;border-top:1px solid #21262d;margin-top:48px}</style></head><body><nav><a href="/data-sources">数据源说明</a><a href="/flywheel" class="active">数据飞轮说明</a><a href="/training-eval">训练测评说明</a></nav><div class="hero"><h1>GSA 数据飞轮 · 系统说明</h1><div class="sub">全自动多维能力数据生成与质量演进 | VLM驱动 | 6维度 × 三层校验 × Verified@k</div></div><div class="container"><h2>1. 系统架构</h2><div class="card"><pre>┌──────────────────────────────────────────────────────────────┐
│                     GSA Data Flywheel                          │
├─────────────┬─────────────┬───────────┬─────────────────────┤
│ Generator   │ Verifier    │Reflection │ Viewer              │
│ :10305      │ :10306      │:10308(off)│ :10307              │
├─────────────┼─────────────┼───────────┼─────────────────────┤
│ 24-worker   │ 5维评分      │VLM验证    │ Dashboard           │
│ CoT prompt  │ 双阈值决策   │场景分类   │ Cases+bbox渲染      │
│ 反推验证     │ Verified@k  │降级/升级  │ 人工打标/Export      │
│ VLM场景分类  │ 60s rescan  │周期抽样   │ Statistics          │
│ 速率自适应   │ 严格规则     │          │                     │
└─────────────┴─────────────┴───────────┴─────────────────────┘
      ▼             ▼            ▼            ▼
data/output/gsa_flywheel_exp/{labels,verified,rejected,needs_review}</pre></div>{{METRICS}}<h2>2. Generator (:10305)</h2><div class="card"><table><tr><th>特性</th><th>说明</th></tr><tr><td>并发</td><td>24 workers 调用 Qwen3.5-397B-A17B API</td></tr><tr><td>维度Shuffle</td><td>G/S/A_attr/A_state/A_precond/A_interface 交错避免偏倚</td></tr><tr><td>CoT Prompt</td><td>逐步推理输出结构化JSON</td></tr><tr><td>预取队列</td><td>后台base64编码消除IO</td></tr><tr><td>速率自适应</td><td>429错误指数退避</td></tr><tr><td>持久化</td><td>processed_set重启不重扫</td></tr></table></div><h3>反推验证 (Cross-Check)</h3><div class="card"><pre>生成: bbox=[183,165,246,257], object="white_jug"
反推: "Is there 'white_jug' at [183,165,246,257]?"
VLM: {"match":true} → 通过

生成: bbox=[564,366,602,612], object="damaged_wall_piece"
VLM: {"match":false,"actual":"wall_surface"} → 失败</pre><p style="font-size:12px;color:#8b949e">仅G_grounding/A_identity_attr/A_action_interface | PassRate理想70-90%</p></div><h3>多样性支持</h3><div class="card"><table><tr><th>类型</th><th>数量</th><th>示例</th></tr><tr><td>场景</td><td>12类</td><td>tabletop/kitchen/living_room/bedroom/bathroom/office/warehouse/outdoor</td></tr><tr><td>动作</td><td>24种</td><td>pick/put/open/close/push/pull/slide/toggle/slice/pour/grasp/stack</td></tr><tr><td>属性</td><td>10种</td><td>color/material/shape/size/weight/texture/transparency</td></tr><tr><td>状态</td><td>~20种</td><td>open_close/occupancy/on_off/temperature/liquid_level/damage</td></tr><tr><td>空间谓词</td><td>~20种</td><td>left-of/right-of/above/below/on-top-of/inside/next-to/behind</td></tr></table></div><h2>3. 三层校验</h2><h3>L1: Generator交叉校验(生成时)</h3><div class="card"><p>Schema + bbox合理性 + 物体名 + VLM反推 → 不通过即丢弃</p></div><h3>L2: Scanned Verifier (:10306)</h3><div class="card"><table><tr><th>评分维度</th><th>权重</th><th>内容</th></tr><tr><td>schema_valid</td><td>0.20</td><td>JSON格式/必填字段</td></tr><tr><td>content_quality</td><td>0.30</td><td>语义合理性(最重要)</td></tr><tr><td>geometry_valid</td><td>0.25</td><td>bbox几何(面积/比例)</td></tr><tr><td>consistency</td><td>0.15</td><td>逻辑一致性</td></tr><tr><td>confidence_check</td><td>0.10</td><td>置信度分级</td></tr></table><p style="margin-top:12px"><b>严格规则:</b> jaw_axis模板检测 | bbox&gt;50%降分 | NON_GRASPABLE过滤 | state逻辑匹配</p><p><b>决策:</b> ≥0.85→<span class="badge bg">verified</span> | 0.60-0.85→<span class="badge by">review</span> | &lt;0.60→<span class="badge br">rejected</span></p></div><h3>L3: Reflection (未启用)</h3><div class="card" style="opacity:0.5"><p>VLM逻辑验证 + 多模型投票降级</p></div><h2>4. PASS@k vs Verified@k</h2><div class="warn">⚠️ 含义完全不同</div><div class="card"><table><tr><th></th><th>PASS@k (滑动窗口)</th><th>Verified@k (样本级)</th></tr><tr><td>定义</td><td>最近k样本通过比例</td><td>单样本过k次rescan</td></tr><tr><td>公式</td><td>pass_count/k</td><td>P≈p^k</td></tr><tr><td>用途</td><td>质量趋势</td><td>PRM奖励</td></tr><tr><td>k</td><td>100/1000</td><td>1/5/20</td></tr></table><p style="margin-top:8px">PRM: <code>r_i = 1[survives k] - 1[survives k+1]</code> | 随机性: 规则+多模型+温度采样+覆盖子集</p></div><h2>5. 六维能力</h2><div class="card"><table><tr><th>维度</th><th>说明</th><th>输出字段</th></tr><tr><td style="color:#7ee787">G_grounding</td><td>物体定位</td><td>object_class, bbox, confidence</td></tr><tr><td style="color:#79c0ff">S_spatial</td><td>空间关系</td><td>subject, predicate, object, bboxes</td></tr><tr><td style="color:#f0c674">A_identity_attr</td><td>属性识别</td><td>object_class, bbox, attributes{}</td></tr><tr><td style="color:#f08080">A_obj_state</td><td>物体状态</td><td>state_type, state_value</td></tr><tr><td style="color:#ffa657">A_action_precondition</td><td>动作可行性</td><td>can_do_now, blocked_by, failure_reason</td></tr><tr><td style="color:#d2a8ff">A_action_interface</td><td>动作接口</td><td>grasp_point_2d, jaw_axis_2d</td></tr></table></div><h2>6. 核心指标</h2><div class="card"><table><tr><th>指标</th><th>定义</th><th>目标</th></tr><tr><td>PassRate</td><td>verified/scanned</td><td>&gt;85%</td></tr><tr><td>PASS@100</td><td>近100通过率</td><td>&gt;90%</td></tr><tr><td>PASS@1000</td><td>近1000通过率</td><td>持续上升</td></tr><tr><td>Throughput</td><td>每分钟验证</td><td>&gt;50</td></tr></table><p style="font-size:12px;color:#8b949e;margin-top:8px">日产: 10,000+条 | 6/6维度 | 7+数据源</p></div><h2>7. 坐标统一→[0,1000)</h2><div class="card"><table><tr><th>源坐标</th><th>转换</th></tr><tr><td>pixel_xywh (COCO)</td><td>x*1000/W, y*1000/H</td></tr><tr><td>norm_01 (OIv7)</td><td>v*1000</td></tr><tr><td>percent_100 (PixMo)</td><td>v*10</td></tr><tr><td>thousandth_1000</td><td>直通</td></tr></table></div><h2>8. 迭代历史</h2><div class="card"><table><tr><th>版本</th><th>日期</th><th>关键更新</th></tr><tr><td><span class="badge bg">june03</span></td><td>06-03</td><td>Verifier严格化·Verified@k·反推验证·VLM场景分类·24Action·5数据源</td></tr><tr><td><span class="badge bb">june01</span></td><td>06-02</td><td>PASS@1000·反思系统·多模型投票·人工打标·24workers·速率自适应</td></tr><tr><td><span class="badge by">may31</span></td><td>05-31</td><td>vLLM双实例·三服务架构·CoT prompts·5Tab Viewer</td></tr><tr><td>v2.0</td><td>05-24</td><td>A维度5pipeline·NAS数据源·端到端(50图→1123标签)</td></tr><tr><td>v1.0</td><td>05-22</td><td>基础架构·taxonomy·G/S/A workers·5层校验·首次20图→229标签</td></tr></table></div><h2>9. 运维</h2><div class="card"><pre>cd /path/to/Anvil
bash scripts/start_flywheel.sh            # 启动
bash scripts/start_flywheel.sh --status   # 状态
bash scripts/start_flywheel.sh --stop     # 停止
python scripts/data_export_for_training.py  # 导出到GSA_volc
python scripts/expand_data_sources.py       # 扩展数据源</pre></div></div><footer>GSA Data Flywheel | {{TIME}}</footer></body></html>
"""

if __name__ == "__main__":
    main()
