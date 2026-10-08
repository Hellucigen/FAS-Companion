# test_cognitive_context.py — Cognitive Context 七分区 + Context Compiler + 证据状态测试
# 离线纯函数：不连图、不调 LLM。验证认知上下文重构的核心契约：
#   1. 分区结构（perception/state/memory/motivation/decision/evidence/constraints）
#   2. 证据状态归一化——executing/queued **绝不允许**被渲染成"已完成"
#   3. decision 分区从 dialogue_decide 输出提炼，语言层不许翻案
#   4. L2 编译与旧 _cog_ctx 扁平键完全兼容（nlp_processor 消费侧零破坏）
#   5. L1 编译只带短句路径真正需要的东西
#   6. debug_view 保留 语言→编译→决定→证据→状态 的可追溯链
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_cognitive_context.py

import os
import sys

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)

import logging
logging.disable(logging.WARNING)

from cognitive_context import (
    build_cognitive_context, compile_for_language, debug_view,
    classify_action_evidence, render_action_evidence, pv,
    EV_EXECUTING, EV_QUEUED, EV_DONE, EV_FAILED, EV_REFUSED, EV_NONE,
    EV_CANCELLED)

FAILURES = []


def check(name, cond, detail=""):
    st = "PASS" if cond else "FAIL"
    print(f"[{st}] {name}" + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


PARSED = {
    "nodes": [" minecraft"], "edges": [], "memory_type": "episodic",
    "response_expectation": "high", "suggested_reply_goals": ["acknowledge"],
    "illocutionary_act": "directive", "dialogue_act": "request",
    "intent": "follow_user", "urgency": 0.9,
    "needs_reply": True, "needs_action": True,
}
DD_RESPOND = {"decision": "respond", "desire": 0.66, "winner_behavior": "respond",
              "constraints": ["ack_only"], "tendencies": [],
              "factors": {"竞争胜者": {"behavior": "respond"}},
              "exploration": {"selected": None}}


def full_ctx(**over):
    kw = dict(text="跟着我", channel="web", parsed=PARSED,
              perception_connected=True, environment={"position": {"x": 1, "y": 64, "z": 2}},
              attention_context={"active_core": [{"id": "Minecraft", "act": 1.2}]},
              mode="language", demand={"score": 0.3},
              world_state={"Haru的位置": "(1,64,2)"},
              mood="心情平和", cognitive_events=[],
              recent_dialogue=[{"user_input": "来玩mc", "system_response": "好"}],
              faiss_hits=[{"hit": "Minecraft", "sim": 0.8}],
              tendencies=[{"behavior": "share", "strength": 0.4}],
              drives={"CuriosityDrive": 2.1},
              hormone={"factor": 1.1},
              exploration={"eligible": False, "candidates": []},
              decision=DD_RESPOND,
              action_intent={"action_type": "follow_entity", "target": "Hellucigen",
                             "outcome": "started_pending"},
              action_queue_len=1,
              evidence={})
    kw.update(over)
    return build_cognitive_context(**kw)


# ── 1. 七分区结构 + provenance ───────────────────────────
ctx = full_ctx()
for sec in ("perception", "state", "memory", "motivation",
            "decision", "evidence", "constraints"):
    check(f"分区存在: {sec}", sec in ctx, str(sorted(ctx)))
check("mood 带来源（provenance）",
      (ctx["state"]["mood"] or {}).get("source") == "persona", str(ctx["state"]["mood"]))
check("drives 带激活强度",
      ctx["motivation"]["drives"]["CuriosityDrive"]["strength"] == 2.1)
check("attention 来源=图扩散",
      ctx["state"]["attention"]["source"] == "graph_diffusion")
check("state ≠ decision：mood 不是裁决，response_mode 才是",
      "response_mode" in ctx["decision"] and "mood" not in ctx["decision"],
      str(sorted(ctx["decision"])))


# ── 2. decision 分区（来自 dialogue_decide，不许语言层翻案）──
dec = ctx["decision"]
check("should_respond 来自裁决", dec["should_respond"] is True)
check("ask_question 未被裁决选择时为 False", dec["ask_question"] is False)
check("长度约束来自 constraints(ack_only)",
      dec["length_constraint"] == "一句极短确认（两个字级别）", str(dec))
check("action_intent 带真实 outcome（started_pending）",
      dec["action_intent"]["outcome"] == "started_pending")
dd_silence = {"decision": "silence", "desire": 0.1, "winner_behavior": "silence",
              "constraints": [], "factors": {}}
ctx_s = full_ctx(decision=dd_silence)
check("silence 裁决 → should_respond False",
      ctx_s["decision"]["should_respond"] is False)
ctx_q = full_ctx(decision={"decision": "explore_ask", "winner_behavior": "explore_ask",
                           "constraints": [], "desire": 0.7, "factors": {}})
check("explore_ask 裁决 → ask_question True",
      ctx_q["decision"]["ask_question"] is True)
def pv_list(c):
    v = c.get("constraints")
    return v.get("value") if isinstance(v, dict) else v


check("游戏通道编译含场合约束",
      any("场合约束" in str(x) for x in pv_list(full_ctx(channel="minecraft_chat"))))


# ── 3. 证据状态归一化（计划≠事实）────────────────────────
cases = [
    ({"success": True, "pending": True, "describe": "开始跟随"}, EV_EXECUTING),
    ({"success": True, "queued": True, "describe": "gather_resource"}, EV_QUEUED),
    ({"success": True, "describe": "停下了"}, EV_DONE),
    ({"success": False, "reason": "tool_missing:stone_pickaxe"}, EV_FAILED),
    ({"success": False, "action": "refused", "reason": "没点名目标"}, EV_REFUSED),
    ({"success": False, "cancelled": True}, EV_CANCELLED),
]
for res, want in cases:
    ev = classify_action_evidence(res)
    check(f"证据分类 {want}", ev["status"] == want, str(ev))
check("空结果 → none", classify_action_evidence({})["status"] == EV_NONE)

# 渲染措辞：executing/queued 必须禁止"已完成"表述
line_exec = render_action_evidence(classify_action_evidence(
    {"success": True, "pending": True, "describe": "走向僵尸"}))
check("executing 渲染含'还在进行'", "还在进行" in line_exec, line_exec)
check("executing 渲染禁说完成", "绝不能说做完了" in line_exec, line_exec)
line_queued = render_action_evidence(classify_action_evidence(
    {"success": True, "queued": True, "describe": "挖铁"}))
check("queued 渲染明确'还没开始'", "还没开始" in line_queued, line_queued)
line_fail = render_action_evidence(classify_action_evidence(
    {"success": False, "describe": "挖铁", "reason": "tool_missing:stone_pickaxe"}))
check("failed 渲染带真实原因", "tool_missing:stone_pickaxe" in line_fail, line_fail)


# ── 4. L2 编译：与旧 _cog_ctx 扁平键完全兼容 ──────────────
ctx_ev = full_ctx(evidence={
    "mc_action": {"success": True, "pending": True, "describe": "正在挖石头",
                  "action": "gather_resource"},
    "mc_session": None,
    "web_results": [{"title": "t", "snippet": "s", "url": "u"}],
    "file_result": None, "eye_result": None,
})
c2 = compile_for_language(ctx_ev, path="L2")
legacy_keys = {"mode", "demand", "attention_context", "illocutionary_act",
               "dialogue_act", "response_expectation", "suggested_reply_goals",
               "behavior_tendencies", "web_results", "file_action", "eye_result",
               "response_constraints", "mc_action", "mc_session",
               "recent_dialogue", "world_state", "mood", "cognitive_events"}
check("L2 编译覆盖旧消费键", legacy_keys <= set(c2), str(legacy_keys - set(c2)))
check("L2 旧键值正确迁移（mc_action 原样可读）",
      c2["mc_action"]["describe"] == "正在挖石头", str(c2["mc_action"]))
check("L2 升级键：action_evidence 带状态",
      c2["action_evidence"]["status"] == EV_EXECUTING, str(c2["action_evidence"]))
check("L2 decision 完整（语言层拿到裁决）",
      c2["decision"]["response_mode"] == "respond")
check("L2 约束迁移（ack_only 在）", "ack_only" in c2["response_constraints"])
check("L2 web 证据渲染为原始结果列表（行为不变）",
      isinstance(c2["web_results"], list) and len(c2["web_results"]) == 1)
check("L2 倾向过滤 silence 不外露",
      all(t.get("behavior") != "silence" for t in c2["behavior_tendencies"]))


# ── 5. L1 编译：只带短句路径需要的 ───────────────────────
c1 = compile_for_language(ctx_ev, path="L1")
check("L1 只含精简键（含言外行为投影概念）",
      set(c1) == {"path", "decision", "action_evidence", "session_evidence",
                  "constraints", "mood", "recent_dialogue", "speech_act",
                  "outline"},
      str(sorted(c1)))
check("L1 证据状态在", c1["action_evidence"]["status"] == EV_EXECUTING)
check("L1 不带图谱记忆/世界状态（不膨胀）",
      "world_state" not in c1 and "attention_context" not in c1)
check("L1 决定含长度约束", c1["decision"]["length_constraint"].startswith("一句"))
check("L1 决定继承 ask_question=False", c1["decision"]["ask_question"] is False)


# ── 6. debug_view：追溯链 ─────────────────────────────────
dv = debug_view(ctx_ev)
check("debug 视图含 decision 摘要",
      dv["decision"]["response_mode"] == "respond"
      and dv["decision"]["action_intent"]["outcome"] == "started_pending")
check("debug 视图含证据状态",
      dv["evidence"]["mc_action"]["status"] == EV_EXECUTING, str(dv["evidence"]))
check("debug 视图含动机/注意力（状态→决定→证据链路闭合）",
      dv["motivation"]["drives"].get("CuriosityDrive") == 2.1
      and "Minecraft" in dv["attention_top"])
check("空 context 不炸", debug_view({}) == {} and compile_for_language({}) == {})


# ── 7. 会话/工具证据的兼容细节 ────────────────────────────
sess = {"success": True, "action": "connect_minecraft", "port": 51027,
        "describe": "已进入 Minecraft 世界（端口 51027），并出发走向用户"}
ev_sess = classify_action_evidence(sess, source="turn:mc_session")
check("会话连接成功 → done（可如实告知已进来）", ev_sess["status"] == EV_DONE)
refused = {"success": False, "action": "refused", "reason": "没点名攻击目标"}
ctx_r = full_ctx(evidence={"mc_action": refused})
check("拒绝证据进 evidence 分区且状态 refused",
      ctx_r["evidence"]["mc_action"]["status"] == EV_REFUSED)
check("refused 渲染如实说明没做",
      "拒绝执行" in render_action_evidence(ctx_r["evidence"]["mc_action"]))

# ── 8. 语言层 prompt 捕获（真实组装路径：pending 不得渲染成成功）──
from langchain_core.runnables import RunnableLambda
from nlp_processor import NLPProcessor


class _Resp8:
    def __init__(self, content="收到。"):
        self.content = content


cap8 = []


def _capture8(pv_):
    cap8.append(pv_.to_string() if hasattr(pv_, "to_string") else str(pv_))
    return _Resp8()


nlp8 = NLPProcessor.__new__(NLPProcessor)   # 不起后端
nlp8.model = "stub"
nlp8.chat_llm = RunnableLambda(_capture8)
nlp8.llm = nlp8.chat_llm
nlp8.logs = []

# L2：mc_action 是 pending → prompt 必须出现"还在进行/绝不能说做完了"
ctx8 = full_ctx(
    channel="minecraft_chat",
    evidence={"mc_action": {"success": True, "pending": True,
                            "describe": "正在挖石头",
                            "action": "gather_resource"}})
c2_8 = compile_for_language(ctx8, path="L2")
nlp8.answer_question("挖点石头", [], [], cognitive_context=c2_8)
pt = cap8[-1]
check("L2 prompt 含【交流决定】（裁决到达语言层）", "【交流决定】" in pt, pt[-260:])
check("L2 prompt 里 pending 渲染为进行中而非成功",
      "还在进行" in pt and "你刚刚执行了" not in pt, pt[-320:])
check("L2 prompt 带场合约束（游戏聊天短回复）",
      "场合约束" in pt or "即时聊天场合" in pt)

# L1：queued → "还没开始"；旧扁平键（无 action_evidence）也要被归类
ctx9 = full_ctx(
    evidence={"mc_action": {"success": True, "queued": True,
                            "describe": "gather_resource", "action":
                            "gather_resource"}})
c1_9 = compile_for_language(ctx9, path="L1")
nlp8.answer_short("顺便挖点铁", c1_9.get("action_evidence"), cognitive_context=c1_9)
pt9 = cap8[-1]
check("L1 prompt 中 queued 明确'还没开始'", "还没开始" in pt9, pt9[:300])
check("L1 prompt 带交流决定（ask_question=False 不外问）",
      "不向用户提问" in pt9 or "交流决定" in pt9, pt9[:300])

# 兼容：老式扁平 cognitive_context（只有 mc_action，没有 action_evidence）
nlp8.answer_question("挖点石头", [], [],
                     cognitive_context={"mc_action": {"success": False,
                                                       "reason": "no_path"},
                                        "dialogue_act": "request"})
ptl = cap8[-1]
check("旧扁平键也走归一化（failed 如实）",
      "失败了" in ptl and "no_path" in ptl, ptl[-260:])


# ── §15/§17：三层结构入 OGCTX（Cognitive Context 承载认知状态）────────
from cognitive_demand import analyze_cognitive_demand as _an
_ana = _an(text="什么是XYZ？",
           parsed={"nodes": ["XYZ"], "edges": [], "dialogue_act": "question",
                   "illocutionary_act": "directive"},
           kg=None, engine=None, unknown_nodes=["XYZ"])
_full = build_cognitive_context(
    text="什么是XYZ？", channel="web",
    parsed={"nodes": ["XYZ"], "edges": [], "illocutionary_act": "directive",
            "dialogue_act": "question", "response_expectation": "high"},
    decision={"decision": "respond", "winner_behavior": "respond",
              "constraints": [], "desire": 0.6, "factors": {}},
    demand=_ana["legacy"], demand_analysis=_ana,
    mode=_ana["mode"])
_st = _full["state"]
check("OGCTX: state 含 cognitive_demand/cognitive_gap/cognitive_resource 三分区",
      all(k in _st for k in ("cognitive_demand", "cognitive_gap", "cognitive_resource")),
      str(sorted(_st)))
check("OGCTX: 三分区带 provenance（source=cognitive_demand_analyzer/routing）",
      _st["cognitive_demand"].get("source") == "cognitive_demand_analyzer"
      and _st["cognitive_resource"].get("source") == "cognitive_demand_routing")
_c2 = compile_for_language(_full, path="L2")
check("OGCTX: L2 编译透传三层结构",
      _c2.get("cognitive_demand", {}).get("knowledge", 0) >= 0.3
      and "knowledge" in _c2.get("cognitive_gap", {}), str(_c2.get("cognitive_demand"))[:120])
check("OGCTX: mode 仍是单一 LLM Resource Level（语言层只读档不看维度）",
      _c2.get("mode") in ("graph_only", "language", "interpret", "reason"),
      str(_c2.get("mode")))
_dv = debug_view(_full)
check("OGCTX: debug 视图能回答'为什么用了这个资源'（evidence+why 在）",
      "XYZ" in str(_dv["cognitive_demand"].get("evidence", {}))
      and len(_dv["cognitive_demand"].get("why", [])) >= 1,
      str(_dv.get("cognitive_demand", {}))[:200])


# ── 发言提纲（决定→提纲→措辞 中间层）──────────────────────
_otx = full_ctx(evidence={"mc_action": {"success": True, "pending": True,
                                        "describe": "正在挖石头"}})
_c2o = compile_for_language(_otx, path="L2")
_ot = _c2o.get("outline") or {}
check("提纲: 姿态/意图来自决定", _ot.get("move") == "respond"
      and _ot.get("intent") == "respond", str(_ot)[:150])
check("提纲: pending → 必须交代已开始 + 禁止说成完成",
      any("仍在进行" in m for m in _ot.get("must", []))
      and any("已经完成" in m for m in _ot.get("must_not", [])), str(_ot)[:220])
_otx2 = full_ctx(evidence={"mc_action": {"success": False, "describe": "挖铁",
                                         "reason": "tool_missing:stone_pickaxe"}})
_ot2 = compile_for_language(_otx2, path="L2")["outline"]
check("提纲: failed → 必须如实交代失败与原因",
      any("tool_missing" in m for m in _ot2.get("must", [])), str(_ot2)[:200])
check("提纲: 通用红线在列（不虚构既成事实）",
      any("不虚构" in m for m in _ot2.get("must_not", [])))
# prompt 捕获：提纲确实到达语言层
class _R:
    def __init__(s, c="ok"): s.content = c
_cap = []
def _capp(pv_):
    _cap.append(pv_.to_string() if hasattr(pv_, "to_string") else str(pv_))
    return _R()
from langchain_core.runnables import RunnableLambda as _RL
from nlp_processor import NLPProcessor as _NP
_nlp = _NP.__new__(_NP)
_nlp.model = "stub"; _nlp.chat_llm = _RL(_capp); _nlp.llm = _nlp.chat_llm; _nlp.logs = []
_nlp.answer_question("挖点石头", [], [], cognitive_context=_c2o)
check("L2 prompt 含【发言提纲】与禁止项", "发言提纲" in _cap[-1]
      and "不许把" in _cap[-1], _cap[-1][-260:])
_c1o = compile_for_language(full_ctx(
    evidence={"mc_action": {"success": True, "queued": True,
                            "describe": "gather_resource"}}), path="L1")
_nlp.answer_short("顺便挖点铁", _c1o.get("action_evidence"), cognitive_context=_c1o)
check("L1 prompt 含提纲（排队状态不许说成在做）",
      "发言提纲" in _cap[-1] and "还没开始" in _cap[-1], _cap[-1][:240])


print()
if FAILURES:
    print(f"✗ {len(FAILURES)} 项失败: {FAILURES}")
    sys.exit(1)
print("✓ Cognitive Context 全部通过（七分区/证据状态/双路编译/追溯链）")
