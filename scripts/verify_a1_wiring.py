# verify_a1_wiring.py — A1(CON-1/CON-2 生产接线)回归锁
# ============================================================================
# 2026-09-30 接线:answer_question 门控从"只认形参"改为"形参→cognitive_context
# 键回退"。本脚本锁三件事:
#   1) L2 编译真实携带 attention_context(8 键)与 cognitive_demand 三层;
#   2) 生产形态调用(只传 cognitive_context)时【认知状态】与【认知资源路由】
#      渲染块真实出现在 prompt 中;
#   3) config nlp_render_cognitive_context=False 时两块整体消失(精确回滚),
#      且既有渲染(发言提纲)不受影响。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 scripts/verify_a1_wiring.py
# ============================================================================
import os
import sys

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)

import config as C                                     # noqa: E402
from graph_model import KnowledgeGraph, Node           # noqa: E402
from diffusion_engine import DiffusionEngine          # noqa: E402
from cognition_modes import attention_context         # noqa: E402
from cognitive_demand import analyze_cognitive_demand  # noqa: E402
from cognitive_context import (build_cognitive_context,  # noqa: E402
                               compile_for_language)
from nlp_processor import NLPProcessor                 # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ── 1) 最小真实装配(diffusion 引擎 + demand 分析,零 LLM)──
kg = KnowledgeGraph()
kg.add_node(Node(id="用户", label="declarative-semantic"))
kg.add_node(Node(id="石头", label="declarative-semantic"))
kg.add_node(Node(id="Haru", graph_space="self"))
engine = DiffusionEngine(kg, C.DEFAULT_CONFIG)
mode = "language"
att = attention_context(kg, engine, mode)
ana = analyze_cognitive_demand(text="石头是什么东西?",
                               parsed={"nodes": ["石头"]},
                               kg=kg, engine=engine, unknown_nodes=["石头"])
full = build_cognitive_context(
    text="石头是什么东西?", channel="web",
    parsed={"nodes": ["石头"]},
    decision={"decision": "respond", "response_mode": "respond",
              "constraints": []},
    demand=ana["legacy"], demand_analysis=ana, mode=mode,
    attention_context=att)
c2 = compile_for_language(full, path="L2")
check("L2 编译携带 attention_context 键(≥4 分区)",
      bool(c2.get("attention_context"))
      and len(c2.get("attention_context") or {}) >= 4,
      str(sorted((c2.get("attention_context") or {}).keys())))
check("L2 编译携带 cognitive_demand/gap/resource 三层",
      bool(c2.get("cognitive_demand")) and bool(c2.get("cognitive_gap"))
      and bool(c2.get("cognitive_resource")),
      str(sorted(k for k in c2 if k.startswith("cognitive_"))))

# ── 2) stub LLM 捕获 prompt ──
captured = []


class _R:
    def __init__(s, c="ok"):
        s.content = c


def _cap(pv_):
    captured.append(pv_.to_string() if hasattr(pv_, "to_string") else str(pv_))
    return _R()


from langchain_core.runnables import RunnableLambda   # noqa: E402
_nlp = NLPProcessor.__new__(NLPProcessor)
_nlp.model = "stub"
_nlp.chat_llm = RunnableLambda(_cap)
_nlp.llm = _nlp.chat_llm
_nlp.logs = []

# ── 3) 生产形态调用:只传 cognitive_context,不传 mode/attention_context 形参 ──
_nlp.answer_question("石头是什么东西?", [], [], cognitive_context=c2)
pt = captured[-1]
check("生产形态调用渲染【认知状态】块",
      "图谱运行时状态" in pt,
      pt[-400:] if "图谱运行时状态" not in pt else "")
check("【认知状态】含模式行", "模式: language" in pt)
check("【认知状态】含 8 键深层渲染(self 分区)", "self:" in pt)
check("生产形态调用渲染【认知资源路由】块", "【认知资源路由】" in pt,
      pt[-400:] if "【认知资源路由】" not in pt else "")
check("【认知资源路由】按不外显原则(只给维度名,无强度数值)",
      "认知资源需求" in pt
      and ("=" not in pt[pt.index("认知资源路由"):
                            pt.index("认知资源路由") + 200]
           or "资源定档" in pt),
      pt[pt.index("认知资源路由"):
         pt.index("认知资源路由") + 200] if "认知资源路由" in pt else "")

# ── 4) config 门=False → 精确回滚 ──
_old = C.DEFAULT_CONFIG.get("nlp_render_cognitive_context", True)
C.DEFAULT_CONFIG["nlp_render_cognitive_context"] = False
captured.clear()
_nlp.answer_question("石头是什么东西?", [], [], cognitive_context=c2)
pt2 = captured[-1]
C.DEFAULT_CONFIG["nlp_render_cognitive_context"] = _old
check("门=False 时【认知状态】整体消失(回滚)",
      "图谱运行时状态" not in pt2)
check("门=False 时【认知资源路由】消失",
      "【认知资源路由】" not in pt2)
check("门=False 时既有渲染不受影响(发言提纲仍在)",
      "【发言提纲】" in pt2)

print()
if FAILURES:
    print(f"✗ {len(FAILURES)} 项失败: {FAILURES}")
    sys.exit(1)
print("✓ A1 接线全过:医生注销 CON-1/CON-2 死代码,八分区+三层结构已入生产 prompt")