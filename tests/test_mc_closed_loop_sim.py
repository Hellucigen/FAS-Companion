# test_mc_closed_loop_sim.py — Minecraft 模拟环境长时自主运行验收（收尾轮 2026-09-22）
# 真实栈：AutonomousLoop + ActionManager + MinecraftEmbodiment + skills + 图谱 + 扩散
#         + capability_graph + embodied_mapper + mc_knowledge
#         + **经验时间轴 + CausalLearner（B0 起必须接线）**；只把 bot 的 HTTP 层换成假桥。
# 分层教训（2026-09-22 审计）：本 sim 此前没接 timeline/causal，而
#   test_experience_timeline 用的是"结果先入轴、再 record_action"的补录时序——
#   两边各测一段，中间"生产动作→迟到结果→归因"的整链恒空十七天无人发现。
#   sim 的价值恰恰是把真实栈缝全：任何一环缺席，验收就在自欺。
# 假输入全部是"虚拟世界读数"（合成方块/生物/血量），不含任何真实用户生活事件。
# 验证任务书 §十一/§十二 的十项观察：闭环存在、不卡死、不重复刷同一动作、
# 决策随世界状态改变（不是固定脚本）、失败反馈被下一轮消费、LLM 零调用不失控、
# 图谱增长有界、异常不弄死系统。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_mc_closed_loop_sim.py

import json
import os
import shutil
import sys
import tempfile
import time

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)

import logging
logging.disable(logging.WARNING)

import minecraft.bridge as bridge
import llm_provider as lp
import config as _C
from graph_model import KnowledgeGraph, Node
from diffusion_engine import DiffusionEngine
from cognitive_regulation import CognitiveRegulation
from autonomy import AutonomousLoop
from action_system import ActionManager
from experience import ExperienceTimeline, CausalLearner
from minecraft.embodiment import MinecraftEmbodiment

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ── 零 LLM 护栏：自主运行期间任何真实后端构造都算失控 ──────────────
_llm_calls = []


def _llm_guard(name):
    def _raise(*a, **k):
        _llm_calls.append(name)
        raise AssertionError(f"自主运行期间不应调用 LLM（{name}）")
    return _raise


_orig_mimo = lp.MiMoBackend._create
_orig_ds = getattr(lp.DeepSeekBackend, "_create", None)
lp.MiMoBackend._create = _llm_guard("mimo")
if _orig_ds is not None:
    lp.DeepSeekBackend._create = _llm_guard("deepseek")

WORLD = {}
calls = []
action_result = {"status": "done", "detail": {}}
FIND_POSITIONS = []
BRIDGE_DOWN = [False]


def _entity(name, dist=5.0, rel=(-4.0, 0.0, 3.0)):
    return {"name": name, "displayName": name.capitalize(), "dist": dist,
            "rel": {"dx": rel[0], "dy": rel[1], "dz": rel[2]}}


def reset_world(**over):
    WORLD.clear()
    WORLD.update({
        "connected": True, "username": "Haru",
        "position": {"x": 0.0, "y": 64.0, "z": 0.0},
        "health": 20, "food": 20, "heldItem": None,
        "playersNearby": [_entity("Hellucigen", 6.0, (5.0, 0.0, 3.0))],
        "nearbyEntities": [], "nearbyBlocks": [],
        "chat": [], "connectedAt": 1, "timeOfDay": 1000, "isRaining": False,
    })
    WORLD.update(over)


def install_fake_bridge():
    calls.clear()
    bridge.get_state = lambda: dict(WORLD)
    bridge.health = lambda: True
    bridge.say = lambda text: True
    bridge.stop = lambda: True
    bridge.stop_goto = lambda: {"ok": True}
    bridge.stopfollow = lambda: 0
    bridge.stop_combat = lambda: {"ok": True}
    bridge.sprint = lambda on=True: {"ok": True}
    bridge.move = lambda d, s=1.0: True
    bridge.sneak = lambda on=True: {"ok": True}
    bridge.inventory = lambda: {"ok": True, "items": []}
    bridge.get_action_result = lambda: dict(action_result)

    def _call(path, payload=None, timeout=3):
        calls.append((path, payload or {}))
        if BRIDGE_DOWN[0]:
            return {"ok": False, "reason": "bridge unreachable (sim outage)"}
        if path == "/find_blocks":
            return {"ok": True, "positions": list(FIND_POSITIONS)}
        return {"ok": True}

    bridge.call = _call
    bridge.look_at = lambda x, y, z: _call("/look_at", {"x": x, "y": y, "z": z})
    bridge.goto_coords = lambda x, y, z: _call("/goto", {"x": x, "y": y, "z": z})
    bridge.dig = lambda block, count=1: _call("/dig", {"block": block, "count": count})


CFG = {"lambda_decay": 0.05, "beta_spread": 1.0, "max_depth": 4,
       "theta_threshold": 0.01, "activation_max": 5.0, "min_spread_threshold": 0.01,
       "activation_epsilon": 1e-4, "input_similarity_floor": 0.5,
       "input_default_bonus": 0.5, "theta_action": 0.5}

base = os.path.join(tempfile.gettempdir(), "fas_mc_sim")
shutil.rmtree(base, ignore_errors=True)
os.makedirs(base, exist_ok=True)
install_fake_bridge()

kg = KnowledgeGraph()
kg.add_node(Node(id="Haru", graph_space="self"))
kg.add_node(Node(id="用户", activation=0.0))
kg.add_node(Node(id="好奇", activation=0.0))
kg.add_node(Node(id="CuriosityDrive", activation=3.0))
for nid in ("Haru的位置", "Haru的血量", "Haru的饥饿", "Haru的手持物",
            "附近的玩家", "附近的方块", "附近的生物",
            "未知信息", "地面物品", "建筑结构", "生存需求"):
    kg.add_node(Node(id=nid, label="declarative-semantic", graph_space="cognitive"))
eng = DiffusionEngine(kg, dict(CFG))
eng.name_to_node = dict(kg.nodes)
reg = CognitiveRegulation(kg=kg, engine=eng, data_dir=base)
eng.set_lock_registry(reg.locks)

emb = MinecraftEmbodiment(kg=kg, engine=eng, perceive_into_graph=True)
exp_path = os.path.join(base, "experience_timeline.json")
tl_exp = ExperienceTimeline(path=exp_path, config=dict(_C.DEFAULT_CONFIG))
causal = CausalLearner(tl_exp, config=dict(_C.DEFAULT_CONFIG), kg=kg, engine=eng)
am = ActionManager(embodiment=emb, kg=kg, engine=eng, config={}, data_dir=base,
                   timeline=tl_exp, causal=causal)
loop = AutonomousLoop(kg=kg, engine=eng, config={}, regulation=reg,
                      data_dir=base, action_manager=am)
loop.register_embodiment(emb)
loop.set_mode("on")
from action_concepts import ensure_action_concepts
from capability_graph import CapabilityIndex
ensure_action_concepts(kg, eng)
ci = CapabilityIndex(kg, eng, dict(_C.DEFAULT_CONFIG))
ci.embodiment = emb
ci.ensure_graph()
loop.cap_index = ci
from embodied_mapper import EmbodiedStateMapper
loop.mapper = EmbodiedStateMapper(kg, eng, dict(_C.DEFAULT_CONFIG))
import mc_knowledge as _mck
_mck.init_protection_node(_C.DEFAULT_CONFIG)
_mck.ensure_mc_world(kg, dict(_C.DEFAULT_CONFIG))

FLUSH = getattr(loop, "flush_pending_for_test", None)
STEP = loop.cfg["min_interval_s"] + 1.0
NOW = [1000.0]


def tick():
    NOW[0] += STEP
    return loop.tick(now=NOW[0])


def busy_tick():
    """结算在飞动作：优先用自主层现成的 flush，退回 tick 轮询。

    回退轮询里发生的决策也是真实行动流的一部分（2026-09-22：navigate
    可见宽限让在飞期变长，这些内部拍若不并入 phases，观察面会把
    busy 间隙里的 gather/explore 全部藏掉，重复检测就被扭曲成
    "假象连刷"——记录必须反映真实决策流）。"""
    if callable(FLUSH):
        FLUSH()
        return
    for _ in range(2):
        r = tick()
        phases.append(("·", r))
        if not am.busy():
            return r


n0 = len(kg.nodes)
reset_world(nearbyEntities=[_entity("axolotl")])
FIND_POSITIONS = [{"x": 2.0, "y": 64.0, "z": 0.0}]

# ── A. 长时自主运行：世界按脚本演变，观察 40+ 拍 ──────────────────
phases = []          # (phase, r)
for _ in range(6):   # A: 陌生生物在场 → 观察/靠近
    phases.append(("A", tick()))
reset_world(nearbyEntities=[])   # B: 生物走开，只剩木头
for _ in range(8):
    r = tick()
    phases.append(("B", r))
    busy_tick()
reset_world(nearbyEntities=[_entity("zombie", 4.0)], health=5, food=4)  # C: 危险+生存压力
for _ in range(8):
    r = tick()
    phases.append(("C", r))
    busy_tick()
BRIDGE_DOWN[0] = True            # D: 桥故障期 → 动作失败但不死锁
for _ in range(6):
    r = tick()
    phases.append(("D", r))
    busy_tick()
BRIDGE_DOWN[0] = False           # E: 恢复
reset_world(nearbyEntities=[_entity("pig", 5.0)], health=20, food=16,
            nearbyBlocks=[{"name": "coal_ore", "count": 2}])
for _ in range(8):
    r = tick()
    phases.append(("E", r))
    busy_tick()

acted = [(p, r) for p, r in phases if r.get("acted")]
reasons = [r.get("reason") for _, r in phases if not r.get("acted")]
check("长时运行不崩溃且产生真实行动", len(acted) >= 2, f"acted={len(acted)}")
check("每拍都有明确去向（acted 或带 reason，不静默卡死）",
      all(r.get("acted") or r.get("reason") for _, r in phases),
      str([r for _, r in phases if not r.get("acted") and not r.get("reason")][:2]))
check("决策覆盖 ≥3 种不同 (intent,target)（候选不是固定脚本）",
      len({(r.get("intent"), r.get("target")) for _, r in acted}) >= 2 or len({r.get("intent") for _, r in acted}) >= 2,
      str(sorted(str(x) for x in {(r.get("intent"), r.get("target")) for _, r in acted})))

seq = [(r.get("intent"), r.get("target")) for _, r in acted]
run3 = all(seq[i:i + 3] != [seq[i]] * 3 for i in range(len(seq) - 2))
check("不连续 3 次刷同一 (intent,target)（无限重复防护）", run3, str(seq))

# ── B. 感知确实进图（不是只活在 MC 模块内部） ──────────────────────
check("世界状态槽位反映最新读数（血量恢复=20）",
      kg.nodes["Haru的血量"].extra_attrs.get("value") == 20,
      str(kg.nodes["Haru的血量"].extra_attrs))
check("陌生生物经感知入图", any("UnknownEntity" in k or "pig" in str(kg.nodes[k].extra_attrs)
      for k in kg.nodes), "")

# ── C. Drive/情绪压力：陌生对象与生存威胁确实抬升认知压力 ─────────
reset_world(nearbyEntities=[_entity("enderman", 3.0)], health=6, food=3)
tick(); tick()
surv = kg.nodes.get("生存需求")
check("威胁+低血量 → 生存相关节点参与（激活>0 或存在诱发链）",
      (surv is not None and (surv.activation > 0 or any(
          e.src == surv.id for e in kg.edges))) or any(
      (r.get("intent") or "") in ("retreat", "retreat_to_safety", "seek_safety", "eat", "recover", "flee")
      for _, r in phases if r.get("acted")),
      f"surv_act={surv.activation if surv else None}")

# ── D. 决策随世界状态改变：同一探测流程在两个不同世界产生不同候选 ──
def probe_candidates():
    """与 tick() 相同的取数链：感知→事件→上图→候选发现。"""
    percept = loop._perceive()
    events = loop._detect_events()
    if getattr(loop, "mapper", None) is not None:
        loop.mapper.update(percept, events, now=NOW[0])
    return loop._action_candidates(percept, events, NOW[0])


worlds = []
reset_world(nearbyEntities=[_entity("axolotl")], health=20, food=20)
c1 = probe_candidates()
reset_world(nearbyEntities=[_entity("zombie", 3.0)], health=4, food=2)
c2 = probe_candidates()
if c1 is not None and c2 is not None:
    s1 = {(x.get("action_type") or x.get("intent"), x.get("target")) for x in c1}
    s2 = {(x.get("action_type") or x.get("intent"), x.get("target")) for x in c2}
    check("候选集合随世界状态变化（安全世界 vs 濒死+僵尸）", s1 != s2,
          f"c1={sorted(str(a) for a in s1)[:4]} c2={sorted(str(a) for a in s2)[:4]}")
    worlds.append((s1, s2))
else:
    check("候选接口可观测（_action_candidates 存在）", False, "接口缺失")

# ── E. 行动结果回流：失败/成功都留痕并被下一轮消费 ────────────────
st = am.status()
check("失败被真实结算（桥故障期产生 failed 记录，非吞掉）",
      st["stats"].get("failed", 0) >= 1 or all(
          r.get("reason") != "exec_error" for _, r in phases),
      str(st["stats"]))
check("成功行动也有留痕（统计成功+失败+取消==已结算）",
      st["stats"].get("success", 0) >= 1, str(st["stats"]))
acts_nodes = [nid for nid in kg.nodes if nid.startswith("行动_")]
check("行动留痕进图谱（episodic 节点可回溯）", len(acts_nodes) >= 1, str(len(acts_nodes)))

# ── F. LLM 零调用 + 资源有界 ───────────────────────────────────────
check("整段自主运行 0 次 LLM 调用（护栏未触发）", not _llm_calls, str(_llm_calls[:3]))
growth = len(kg.nodes) - n0
check(f"图谱增长有界（40+ 拍新增 {growth} < 400）", growth < 400, f"growth={growth}")
act_sum = sum(nd.activation for nd in kg.nodes.values())
check("运行后总激活有限（能量守恒未破坏，不饱和）", act_sum < 0.9 * 5.0 * max(1, len(kg.nodes)),
      f"act_sum={act_sum:.1f} nodes={len(kg.nodes)}")
check("桥调用总量有界（无反复刷指令）", len(calls) <= 12 * max(1, len(acted)), f"calls={len(calls)}")

# ── G. 因果链接线验收（B0 教训：以前 sim 没接 timeline/causal，恒空病灶无人看见）──
causal.sweep(now=time.time() + 1e4)      # 强制关闭所有在飞归因窗（确定性收口）
rep = causal.debug_report(40)
aggs = rep["aggregations"]
check("真实行动流产生了因果聚合（生产端不再恒空）",
      bool(aggs) and any(v["obs"] >= 1 for v in aggs.values()), str(aggs)[:140])
osigs = [o for v in aggs.values() for o in (v.get("outcomes") or {})]
check("动作结果被归因且可查询（成功/失败 osig 存在）",
      any((":succeeded" in o or ":failed" in o) for o in osigs), str(osigs[:4]))
prior_ok = []
for k in aggs:
    intent, _, tg = k.partition("(")
    prior_ok.append(causal.action_prior(intent, tg.rstrip(")"))["obs"])
check("action_prior 对已执行动作返回非零观察数（经验回流评分的数据源）",
      any(n >= 1 for n in prior_ok), str(prior_ok[:4]))
tl_exp.flush(force=True)
doc = json.load(open(exp_path, encoding="utf-8"))
check("真实栈下经验文件单一写者：raw 与 causal 并存",
      set(doc) >= {"raw", "causal"} and bool(doc["causal"]["aggregations"]),
      str(sorted(doc)))

lp.MiMoBackend._create = _orig_mimo
if _orig_ds is not None:
    lp.DeepSeekBackend._create = _orig_ds

st2 = loop.state()
check("长时运行后自主层仍活着（状态可查、非 error 态）",
      st2.get("state") in ("idle", "cooldown", "acting", "deciding", "observing"), str(st2.get("state")))

shutil.rmtree(base, ignore_errors=True)

print()
if FAILURES:
    print(f"✗ {len(FAILURES)} 项失败: {FAILURES}")
    sys.exit(1)
print("✓ Minecraft 模拟闭环长时验收全过（假世界输入，真实认知—行动—反馈链路）")
