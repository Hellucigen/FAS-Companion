# test_world_state_grounding.py — 世界状态接地到语言层
#
# 起因（真实游玩）：用户问"你现在在哪里"，她答"我在Haru这里，和你一起。"——把自己当成了用户、
# 位置靠猜。根因有三层，本测试逐层锁住：
#   1. 状态节点的值只在 extra_attrs.value 里，TopK 渲染**从不带值** → 语言层只看到空洞标签
#   2. 状态节点是**孤岛**（无边）→ 扩散到不了、上下文里也没有"这是 Haru 的状态"这个锚点
#   3. 旧空壳节点 `Fascinator -[位于]-> 当前所在位置`（无值）会抢占"位置"话题
#
# 做法：用占位 LLM 捕获真实 prompt（走真实 answer_question 组装路径），断言坐标确实到达语言层。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_world_state_grounding.py

import os
import sys
import types

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)

import logging
logging.disable(logging.WARNING)

from graph_model import KnowledgeGraph, Node, Edge
from minecraft.perception import update_perception, world_state_snapshot, DYNAMIC_NODES
from prompt_templates import ANSWER_GENERATE

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class _FakeEngine:
    def __init__(self):
        self._active = {}

    def mark_active(self, ids):
        for i in ids:
            self._active[i] = None


MC_STATE = {
    "connected": True,
    "position": {"x": -228.7, "y": 46.0, "z": -376.7},
    "health": 20, "food": 18, "heldItem": "grass_block",
    "playersNearby": [{"name": "Hellucigen", "dist": 10.8,
                       "rel": {"dx": 5.7, "dy": 0.0, "dz": -9.2}}],
    "nearbyBlocks": [{"name": "water", "count": 3}],
    "nearbyEntities": [{"name": "salmon", "displayName": "Salmon", "dist": 4.0}],
    "chat": [], "connectedAt": "t",
}


def build_kg():
    kg = KnowledgeGraph()
    kg.add_node(Node(id="Haru", graph_space="self", label="self"))
    kg.add_node(Node(id="Self", graph_space="self", label="self"))
    kg.add_node(Node(id="Fascinator"))
    # 旧会话残留：空值的位置槽位（曾经抢话题的元凶）
    kg.add_node(Node(id="当前所在位置"))
    kg.add_edge(Edge(src="Fascinator", dst="当前所在位置", relation="位于", weight=1.0))
    return kg


# ── 1. 感知：状态节点必须有值、必须接线（不能再是孤岛） ──
kg = build_kg()
update_perception(kg, _FakeEngine(), MC_STATE)
check("感知写入 7 个状态槽位", all(n in kg.nodes for n in DYNAMIC_NODES))
check("位置槽位带真实坐标",
      kg.nodes["Haru的位置"].extra_attrs.get("value") == "(-228.7,46.0,-376.7)",
      str(kg.nodes["Haru的位置"].extra_attrs.get("value")))
check("状态槽位接到 当前Minecraft状态（不再是孤岛）",
      kg.get_edge("当前Minecraft状态", "Haru的位置", "状态项") is not None
      and kg.get_edge("当前Minecraft状态", "Haru的血量", "状态项") is not None)
check("Haru 经状态中间节点持有状态（Haru 是主体不是 Hub）",
      kg.get_edge("Haru", "当前Minecraft状态", "当前状态") is not None
      and kg.get_edge("Haru", "Haru的位置", "当前状态") is None)
check("接的是 Haru 而不是 Self（避免扰动 Self 出边归一化）",
      kg.get_edge("Self", "Haru的位置", "当前状态") is None)
check("旧空壳槽位被写入真值（不再空着误导）",
      kg.nodes["当前所在位置"].extra_attrs.get("value") == "(-228.7,46.0,-376.7)",
      str(kg.nodes["当前所在位置"].extra_attrs.get("value")))

# ── 2. 快照：语言层的状态来源是图（唯一真源），只给有值的槽位 ──
snap = world_state_snapshot(kg)
check("快照含位置/血量/手持物/附近", {"Haru的位置", "Haru的血量", "Haru的手持物",
                              "附近的玩家"} <= set(snap), str(sorted(snap)))
check("快照只给有值的槽位（未感知的不编造）",
      all(v not in (None, "") for v in snap.values()))
kg_empty = KnowledgeGraph()
check("没感知过 → 快照为空（不造假状态）", world_state_snapshot(kg_empty) == {})

# ── 3. 神经末梢：真实 answer_question 路径必须把坐标带进 prompt ──
from nlp_processor import NLPProcessor
from langchain_core.runnables import RunnableLambda


class _Resp:
    def __init__(self, content="我在 (-228.7,46.0,-376.7) 附近。"):
        self.content = content


captured = []


def _capture(prompt_value):
    """占位 LLM：记录真实 prompt 文本，返回一个带 .content 的响应对象。"""
    captured.append(prompt_value.to_string() if hasattr(prompt_value, "to_string")
                    else str(prompt_value))
    return _Resp("我在 (-228.7,46.0,-376.7) 附近。")


nlp = NLPProcessor.__new__(NLPProcessor)          # 不走构造（避免起 LLM 后端）
nlp.model = "stub"
nlp.chat_llm = RunnableLambda(_capture)
nlp.llm = nlp.chat_llm
nlp.logs = []

topk = [kg.nodes["Haru的位置"], kg.nodes["Haru的血量"],
        kg.nodes["Haru的手持物"], kg.nodes["附近的玩家"], kg.nodes["当前所在位置"]]
ans = nlp.answer_question("你现在在哪里", topk, [], context_type=None,
                          cognitive_context={"world_state": snap})
check("回答生成走通（占位 LLM）", isinstance(ans, str) and ans != "", str(ans)[:80])
prompt_text = captured[-1] if captured else ""
check("prompt 里出现她的真实坐标",
      "(-228.7,46.0,-376.7)" in prompt_text, prompt_text[-300:])
check("prompt 里状态节点渲染成「槽位 = 值」",
      "Haru的位置 = " in prompt_text and "Haru的血量 = 20" in prompt_text,
      prompt_text[-300:])
check("prompt 里有【世界状态】块且带接地要求",
      "【世界状态】" in prompt_text and "不许猜测" in prompt_text)
check("手持物/附近玩家也随之上线（不只有位置）",
      "grass_block" in prompt_text and "Hellucigen" in prompt_text)
check("空值节点不会被渲染成 '= None'", "= None" not in prompt_text)

# ── 4. prompt 规则：人称与身份（她不是用户） ──
check("回答规则含人称/身份约束",
      "人称与身份" in ANSWER_GENERATE and "你的名字叫 Haru" in ANSWER_GENERATE)
check("规则明确禁止把自己说成在别人那里",
      "把自己说成" in ANSWER_GENERATE and "都是错的" in ANSWER_GENERATE)
check("规则要求位置类问题用真实数值回答",
      "世界状态必须接地" in ANSWER_GENERATE and "不许猜测" in ANSWER_GENERATE)
check("规则说明「槽位 = 值」的含义",
      "槽位 = 值" in ANSWER_GENERATE)

# ── 5. 断连时不注入（不发假状态） ──
kg2 = build_kg()
update_perception(kg2, _FakeEngine(), {"connected": False})
check("未连世界 → 快照为空（语言层不会拿到假状态）",
      world_state_snapshot(kg2) == {}, str(world_state_snapshot(kg2)))

print()
if FAILURES:
    print(f"✗ {len(FAILURES)} 项失败: {FAILURES}")
    sys.exit(1)
print("✓ 世界状态接地测试全过（真实 prompt 捕获：坐标确实到达语言层）")
