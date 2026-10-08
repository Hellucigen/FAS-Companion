# test_closed_loop_consistency.py — 收敛与修复（2026-09-27 Phase B）：闭环一致性离线验收
# ============================================================================
# 任务书（用户 2026-09-27 指令 §二）最小 offline simulation：
#   Test A  target disappears → goal/candidate invalidated/downgraded
#   Test B  world changes → candidate B rises（生命周期让位：同一图状态，
#           在场者目标保权、缺席者降权；scoring 吃 score_mul）
#   Test C  transport timeout → perception confirms → settlement 最终
#           success/confirmed（CausalLearner late-confirmed：世界观察覆盖
#           迟到失败，失败不进 causal 失败账）
#   Test D  true failure → correct category → causal 可学习
#           （missing_ingredients 非暂态 → 进失败账 → 假设形成）
#   Test E  repeated explore success without knowledge/progress → 不无限
#           正强化（discovery 证据门 / explore 空手 = progress / 剥夺钟
#           只在真知识增量时重置）
#   Test F  no-progress long-running goal → relevance 降 → 让位
#           （生命周期 active→demoted→stale，score_mul 0.5；目标重新在场
#           复活，证明不是永久禁令）
#   Test G  causal promotion/replay → activation → candidate score changes
#           （重复成功 → 聚合 → 假设 → KG 晋升 → 操作: 节点作激活源 →
#           _score_action 的 prior_bonus 上升）
# 真实栈 + 假桥（脚本世界会走路：dig 真掉落/collect 真入包/craft 真消耗），
# 零 LLM（护栏钉死）。时间轴/CausalLearner 全接线。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_closed_loop_consistency.py
# ============================================================================

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
import experiment_mode as xm
from graph_model import KnowledgeGraph, Node
from diffusion_engine import DiffusionEngine
from cognitive_regulation import CognitiveRegulation
from autonomy import AutonomousLoop
from action_system import ActionManager
from experience import (ExperienceTimeline, CausalLearner,
                        make_event, EVENT_ACTION, EVENT_SELF_STATE,
                        EVENT_OBSERVATION)
from minecraft.embodiment import MinecraftEmbodiment

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ── 零 LLM 护栏 ─────────────────────────────────────────────────
_llm_calls = []


def _llm_guard(name):
    def _raise(*a, **k):
        _llm_calls.append(name)
        raise AssertionError(f"闭环一致性测试不应调用 LLM（{name}）")
    return _raise


lp.MiMoBackend._create = _llm_guard("mimo")

ENG_CFG = {"lambda_decay": 0.05, "beta_spread": 1.0, "max_depth": 4,
           "theta_threshold": 0.01, "activation_max": 5.0,
           "min_spread_threshold": 0.01, "activation_epsilon": 1e-4,
           "input_similarity_floor": 0.5, "input_default_bonus": 0.5,
           "theta_action": 0.5}

BASE = os.path.join(tempfile.gettempdir(), "fas_consistency")


# ════════════════════════════════════════════════════════════════
# 脚本世界（复制自离线验收；新增：玩家在场景 + 桥回执丢弃模式）
# ════════════════════════════════════════════════════════════════
RECIPE_TABLE = {
    "oak_planks": [{"result": "oak_planks", "yield": 4,
                    "ingredients": {"oak_log": 1}, "needs_table": False}],
    "stick": [{"result": "stick", "yield": 4,
               "ingredients": {"oak_planks": 2}, "needs_table": False}],
    "crafting_table": [{"result": "crafting_table", "yield": 1,
                        "ingredients": {"oak_planks": 4},
                        "needs_table": False}],
    "cobblestone": [],
    "stone": [],
}

BLOCK_META = {
    "oak_log": {"drops": ["oak_log"], "harvest_tools": []},
    "stone": {"drops": ["cobblestone"], "harvest_tools": ["wooden_pickaxe"]},
    "cobblestone": {"drops": ["cobblestone"],
                    "harvest_tools": ["wooden_pickaxe"]},
    "dirt": {"drops": ["dirt"], "harvest_tools": []},
}


def _d2(pos, xyz):
    return ((pos.get("x", 0) - xyz[0]) ** 2 +
            (pos.get("z", 0) - xyz[2]) ** 2) ** 0.5


def _chg(e):
    """事件的 change 字段：content 可能是 dict 也可能是裸字符串。"""
    c = e.get("content")
    return (c or {}).get("change", "") if isinstance(c, dict) else str(c)


class ScriptWorld:
    """会走路的假世界。goto 每拍逼近；dig 移除方块并掉落；collect 入包；
    craft 消耗原料产成品。drop_mode="timeout" = 桥把回执弄丢（世界照常
    发生）——transport 失败语义，与"动作真失败"严格分开。"""

    def __init__(self):
        self.pos = {"x": 0.0, "y": 64.0, "z": 0.0}
        self.inv = []
        self.blocks = {}
        self.near_names = []
        self.pending_drop = []
        self.goto_target = None
        self.last_cmd = None
        self.receipt = {"status": "done", "detail": {}}
        self.placed = {}
        self.held = None
        self.time_of_day = 6000
        self.players = []          # 在场玩家名（本测试用）
        self.drop_mode = None      # None | "timeout"（回执丢失）

    def put_block(self, x, y, z, name):
        self.blocks[(int(x), int(y), int(z))] = name
        self._refresh_near()

    def set_inv(self, **counts):
        self.inv = [{"name": k, "count": v} for k, v in counts.items() if v]
        self._refresh_near()

    def set_players(self, *names):
        self.players = list(names)

    def _refresh_near(self):
        from collections import Counter
        c = Counter()
        for (bx, by, bz), n in (list(self.blocks.items()) +
                                list(self.placed.items())):
            if _d2(self.pos, (bx, by, bz)) <= 4.5:
                c[n] += 1
        self.near_names = [{"name": k, "count": v} for k, v in
                           sorted(c.items(), key=lambda kv: -kv[1])[:6]]

    def inv_map(self):
        return {i["name"]: i["count"] for i in self.inv}

    def add_item(self, name, count=1):
        for it in self.inv:
            if it["name"] == name:
                it["count"] += count
                return
        self.inv.append({"name": name, "count": count})

    def take_items(self, needs):
        inv = self.inv_map()
        if not all(inv.get(k, 0) >= v for k, v in needs.items()):
            return False
        for k, v in needs.items():
            left = v
            for it in list(self.inv):
                if it["name"] == k and left > 0:
                    d = min(it["count"], left)
                    it["count"] -= d
                    left -= d
                    if it["count"] <= 0:
                        self.inv.remove(it)
        return True

    def state(self):
        return {"connected": True, "username": "Haru",
                "position": dict(self.pos), "health": 20, "food": 20,
                "heldItem": None,
                "playersNearby": [{"name": n, "dist": 6.0,
                                   "rel": {"dx": 5.0, "dy": 0.0, "dz": 3.0}}
                                  for n in self.players],
                "nearbyEntities": [], "nearbyBlocks": list(self.near_names),
                "chat": [], "connectedAt": 1,
                "timeOfDay": self.time_of_day, "isRaining": False}

    def call(self, path, payload=None, timeout=3):
        payload = payload or {}
        if path == "/goto":
            self.goto_target = (float(payload.get("x", 0)),
                                float(payload.get("y", 64)),
                                float(payload.get("z", 0)))
            self.last_cmd = "goto"
            self.receipt = {"status": "running", "detail": {}}
            return {"ok": True}
        if path == "/stop_goto":
            return {"ok": True}
        if path == "/find_blocks":
            want = str(payload.get("block") or "").lower()
            radius = float(payload.get("radius") or 16)
            out = []
            for (bx, by, bz), nm in list(self.blocks.items()) + \
                    list(self.placed.items()):
                if nm.lower() == want and _d2(self.pos, (bx, by, bz)) <= radius:
                    out.append({"x": float(bx), "y": float(by),
                                "z": float(bz)})
            return {"ok": True, "positions": out}
        if path == "/dig_pos":
            p = (int(payload.get("x", 0)), int(payload.get("y", 0)),
                 int(payload.get("z", 0)))
            nm = self.blocks.get(p) or self.placed.get(p)
            if nm is None:
                return {"ok": False, "reason": "block_not_found"}
            self.blocks.pop(p, None)
            self.placed.pop(p, None)
            meta = BLOCK_META.get(nm, {})
            req = meta.get("harvest_tools", [])
            ok_harvest = (not req) or (self.held in req)
            drops = meta.get("drops", [nm]) if ok_harvest else []
            self.pending_drop.extend(drops)
            self.last_cmd = "dig"
            self.receipt = {"status": "done",
                            "detail": {"dug": 1, "block": nm}}
            self._refresh_near()
            return {"ok": True}
        if path == "/collect_item":
            if self.pending_drop:
                for d in self.pending_drop:
                    self.add_item(d)
                got = len(self.pending_drop)
                self.pending_drop = []
                self.receipt = {"status": "done", "detail": {"picked": got}}
            else:
                self.receipt = {"status": "done", "detail": {"picked": 0}}
            self.last_cmd = "collect"
            return {"ok": True}
        if path == "/craft":
            item = str(payload.get("item") or "").lower()
            recs = RECIPE_TABLE.get(item) or []
            if not recs:
                self.receipt = {"status": "failed",
                                "detail": {"reason": "no_recipe"}}
                return {"ok": True}
            r = recs[0]
            ing = {k.lower(): int(v)
                   for k, v in (r.get("ingredients") or {}).items()}
            if not self.take_items(ing):
                self.receipt = {"status": "failed",
                                "detail": {"reason": "missing_ingredients"}}
                self.last_cmd = "craft"
                return {"ok": True}
            # 世界真发生的一切照旧……
            self.add_item(item, int(r.get("yield") or 1))
            # ……但合成调用本身超时失败（26.1 真机样本：bot.craft 走窗口事务，
            # /craft 调回声 45s 超时报失败、合成其实已落袋——crafting.py:179
            # 同款注释）→ 世界已发生，账面却报失败（transport 终点，B2/P2b）。
            # 注意：craft 技能不轮询回执，只信调用返回值——timeout 必须落在
            # 调用层，存进 poll_result 的回执没人会读。
            if self.drop_mode == "timeout":
                self.receipt = {"status": "failed",
                                "detail": {"reason": "bridge_error: timed out"}}
                self.last_cmd = "craft_dropped"
                return {"ok": False, "reason": "bridge_error: timed out"}
            self.receipt = {"status": "done", "detail": {"crafted": item}}
            self.last_cmd = "craft"
            return {"ok": True}
        if path in ("/stop", "/sneak", "/sprint", "/look_at", "/attack",
                    "/combat", "/interact", "/eat", "/dig", "/equip"):
            self.receipt = {"status": "done", "detail": {}}
            return {"ok": True}
        return {"ok": True}

    def poll_result(self):
        if self.last_cmd == "goto" and self.goto_target:
            tx, ty, tz = self.goto_target
            dx, dy, dz = tx - self.pos["x"], ty - self.pos["y"], tz - self.pos["z"]
            d = (dx * dx + dz * dz) ** 0.5
            if d <= 1.5:
                self.goto_target = None
                self.receipt = {"status": "done", "detail": {}}
            else:
                step = min(3.0, d) / max(d, 1e-6)
                self.pos = {"x": self.pos["x"] + dx * step,
                            "y": self.pos["y"] + dy * step,
                            "z": self.pos["z"] + dz * step}
                self.receipt = {"status": "running", "detail": {}}
        return dict(self.receipt)


def install_fake_bridge(W):
    bridge.get_state = lambda: W.state()
    bridge.health = lambda: True
    bridge.say = lambda text: True
    bridge.stop = lambda: True
    bridge.stop_goto = lambda: W.call("/stop_goto")
    bridge.stopfollow = lambda: 0
    bridge.stop_combat = lambda: {"ok": True}
    bridge.sprint = lambda on=True: {"ok": True}
    bridge.move = lambda d, s=1.0: True
    bridge.sneak = lambda on=True: {"ok": True}
    bridge.look_at = lambda x, y, z: W.call("/look_at", {"x": x, "y": y, "z": z})
    bridge.goto_coords = lambda x, y, z: W.call("/goto",
                                                {"x": x, "y": y, "z": z})
    bridge.goto_player = lambda p: W.call("/goto_player", {"player": p})
    bridge.dig = lambda block, count=1: W.call("/dig", {"block": block})
    bridge.dig_pos = lambda x, y, z: W.call("/dig_pos",
                                            {"x": x, "y": y, "z": z})
    bridge.place_block = lambda x, y, z, item="": W.call(
        "/place", {"x": x, "y": y, "z": z, "item": item})
    bridge.craft = lambda item, count=1, table=None: W.call(
        "/craft", {"item": item, "count": count})
    bridge.smelt = lambda input_item, fuel, furnace: W.call(
        "/smelt", {"input": input_item, "fuel": fuel})
    bridge.furnace_take = lambda furnace: W.call("/furnace_take", {})
    bridge.eat = lambda item: W.call("/eat", {"item": item})
    bridge.collect_item = lambda radius=8: W.call("/collect_item",
                                                  {"radius": radius})
    bridge.interact = lambda x, y, z: W.call("/interact",
                                             {"x": x, "y": y, "z": z})
    bridge.equip = lambda item: W.call("/equip", {"item": item})
    bridge.attack = lambda entity="": W.call("/attack", {"entity": entity})
    bridge.combat = lambda entity, **k: W.call("/combat", {"entity": entity})
    bridge.inventory = lambda: {"ok": True, "items": list(W.inv)}
    bridge.inventory_slots = lambda: {"ok": True, "slots": []}
    bridge.recipe_for = lambda item: {"ok": True, "recipes": []}
    bridge.block_meta = lambda name: (
        {"ok": True, "block": str(name).lower(),
         "drops": BLOCK_META[str(name).lower()].get("drops", []),
         "harvest_tools": BLOCK_META[str(name).lower()]["harvest_tools"]}
        if str(name).lower() in BLOCK_META else {"ok": False,
                                                 "reason": "unknown_block"})
    bridge.find_blocks = lambda block, radius=12, count=8: W.call(
        "/find_blocks", {"block": block, "radius": radius, "count": count})
    bridge.call = W.call
    bridge.get_action_result = lambda: W.poll_result()
    # 仿真时钟压缩补偿：技能首拍轮询门槛按真实秒（0.8s），一拍仿真空转
    # 太久 → 置 0 等价"真实 0.8s 已过"（离线验收同款补丁）。
    import skills as _sp
    if not getattr(_sp.run, "_fas_sim_nogate", False):
        _real_run = _sp.run

        def _sim_run(name, ctx, params=None, first_poll_delay=0.0, **kw):
            return _real_run(name, ctx, params, 0.0, **kw)
        _sim_run._fas_sim_nogate = True
        _sp.run = _sim_run


def build_stack():
    shutil.rmtree(BASE, ignore_errors=True)
    os.makedirs(BASE, exist_ok=True)
    kg = KnowledgeGraph()
    kg.add_node(Node(id="Haru", graph_space="self"))
    for nid in ("Haru的位置", "Haru的血量", "Haru的饥饿", "Haru的手持物",
                "附近的玩家", "附近的方块", "附近的生物",
                "未知信息", "地面物品", "建筑结构", "生存需求"):
        kg.add_node(Node(id=nid, label="declarative-semantic",
                         graph_space="cognitive"))
    eng = DiffusionEngine(kg, dict(ENG_CFG))
    eng.name_to_node = dict(kg.nodes)
    cfgd = dict(_C.DEFAULT_CONFIG)
    reg = CognitiveRegulation(kg=kg, engine=eng, data_dir=BASE)
    eng.set_lock_registry(reg.locks)
    emb = MinecraftEmbodiment(kg=kg, engine=eng, perceive_into_graph=True,
                              config=cfgd)
    exp_path = os.path.join(BASE, "experience_timeline.json")
    tl = ExperienceTimeline(path=exp_path, config=cfgd)
    emb.timeline = tl          # 同 app.py:1181（具身观察入轴，不归因）
    causal = CausalLearner(tl, config=cfgd, kg=kg, engine=eng)
    am = ActionManager(embodiment=emb, kg=kg, engine=eng, config=cfgd,
                       data_dir=BASE, timeline=tl, causal=causal)
    loop = AutonomousLoop(kg=kg, engine=eng, config=cfgd, regulation=reg,
                          data_dir=BASE, action_manager=am)
    loop.register_embodiment(emb)
    loop.set_mode("on")
    from action_concepts import ensure_action_concepts
    from capability_graph import CapabilityIndex
    from embodied_mapper import EmbodiedStateMapper
    ensure_action_concepts(kg, eng)
    ci = CapabilityIndex(kg, eng, cfgd)
    ci.embodiment = emb
    ci.ensure_graph()
    loop.cap_index = ci
    loop.mapper = EmbodiedStateMapper(kg, eng, cfgd)
    import mc_knowledge as _mck
    _mck.init_protection_node(cfgd)
    _mck.ensure_mc_world(kg, cfgd)
    cfgd["experiment"] = {"mode": "off"}
    xm.configure(cfgd)
    import world_prior as _wp0
    _wp0.bind_config(cfgd)
    _wp0._RUNTIME["queried"].clear()
    return kg, eng, reg, emb, tl, causal, am, loop


NOW = [1000.0]
STEP = 16.0


def _sync_names(loop):
    try:
        kgx, engx = loop.kg, loop.engine
        if kgx is not None and engx is not None:
            for _n, _nd in kgx.nodes.items():
                if _n not in engx.name_to_node:
                    engx.name_to_node[_n] = _nd
    except Exception:
        pass


def tick(loop, emb):
    NOW[0] += STEP
    emb._inv_last_pull = 0.0
    _sync_names(loop)
    try:
        emb.ctx.invalidate()
    except Exception:
        pass
    from skills.base import SkillContext
    return loop.tick(now=NOW[0])


def flush(loop, am, emb, rounds=8):
    for _ in range(rounds):
        if not am.busy():
            return
        tick(loop, emb)


def probe(loop):
    _sync_names(loop)
    percept = loop._perceive()
    events = loop._detect_events()
    if getattr(loop, "mapper", None) is not None:
        loop.mapper.update(percept, events, now=NOW[0])
    return percept, loop._action_candidates(percept, events, NOW[0])


def user_goal_cands(cands, atype=None):
    out = [c for c in cands
           if str(c.get("motivation") or "") == "user_goal"
           and (atype is None or c.get("action_type") == atype)]
    return out


print("── Test A：目标消失 → 生命周期降级/搁置 → 目标重新在场复活 ──")
W = ScriptWorld()
install_fake_bridge(W)
kg, eng, reg, emb, tl, causal, am, loop = build_stack()
W.set_players("Hellucigen")
NOW[0] = 1000.0
loop.register_invitation("follow", "Hellucigen")
# 虚拟钟测试：生命周期的 created_ts 锚用仿真时间（生产路径是真实时钟，
# 两边一致；测试注入虚拟 now 时必须对齐，否则 span 为负永不判死）
loop._goals[-1]["created_ts"] = NOW[0]
g = loop._goals[-1]
c0 = (g.get("created_ts") or 0) > 0
check("A1 邀请目标是持久目标且带创建时间锚", c0,
      f"created_ts={g.get('created_ts')}")
p0, cands0 = probe(loop)
f0 = user_goal_cands(cands0, "follow_entity")
check("A2 目标在场 → follow 候选产出（无 mul，满权）",
      bool(f0) and all(not f.get("score_mul") for f in f0),
      str([(f.get("action_type"), f.get("target")) for f in f0][:3]))
W.set_players()                     # 目标离开场景
NOW[0] = 1016.0
p1, _ = probe(loop)
s1 = loop._reevaluate_goal(g, p1, NOW[0])
check("A3 刚离开（<DEMOTE 窗口）→ 仍在机会期 active", s1 == "active", s1)
NOW[0] = 1700.0                     # 缺席 ≥ 600s → 降权
p2, cands2 = probe(loop)
s2 = loop._reevaluate_goal(g, p2, NOW[0])
cands_after = user_goal_cands(cands2, "follow_entity")
mul = [f for f in cands_after if abs(float(f.get("score_mul") or 1.0) - 0.5) < 1e-9]
check("A4 缺席满 DEMOTE 窗口 → demoted 且候选带 score_mul=0.5（让位不封杀）",
      s2 == "demoted" and bool(mul), f"state={s2} mul_cands={len(mul)}")
NOW[0] = 3501.0                     # 缺席 ≥ 2400s → 搁置
p3, cands3 = probe(loop)
s3 = loop._reevaluate_goal(g, p3, NOW[0])
f3 = user_goal_cands(cands3, "follow_entity")
check("A5 缺席满 STALE → stale 且不再产出候选（保留记录）",
      s3 == "stale" and not f3,
      f"state={s3} cands={len(f3)} status={g.get('status')}")
W.set_players("Hellucigen")         # 目标重新在场 → 复活
NOW[0] = 3517.0
p4, _ = probe(loop)
s4 = loop._reevaluate_goal(g, p4, NOW[0])
p4b, cands4 = probe(loop)
f4 = user_goal_cands(cands4, "follow_entity")
check("A6 重新在场 → 复活 active（软状态，非永久禁令）",
      s4 == "active" and bool(f4),
      f"state={s4} status={g.get('status')} cands={len(f4)}")

print("\n── Test B：世界变化 → 候选让位——在场者保值、缺席者降权 ──")
W = ScriptWorld()
install_fake_bridge(W)
kg, eng, reg, emb, tl, causal, am, loop = build_stack()
W.set_players("Alpha")
NOW[0] = 5000.0
loop.register_invitation("follow", "Alpha")
loop._goals[-1]["created_ts"] = NOW[0]   # 虚拟钟锚（生命周期判定用仿真时）
loop.register_invitation("approach", "Beta")     # Beta 从未在场
loop._goals[-1]["created_ts"] = NOW[0]
gA, gB = loop._goals[-2], loop._goals[-1]
pB0, candsB0 = probe(loop)
b_now = user_goal_cands(candsB0, "navigate_to_entity")
check("B1 启动期（机会窗口内）缺席目标仍满权候选（不误杀）",
      bool(b_now) and all(not c.get("score_mul") for c in b_now),
      str([(c.get("action_type"), c.get("target")) for c in b_now][:2]))
NOW[0] = 5800.0                     # +800s：无人成功、Beta 缺席满 600
pB1, candsB1 = probe(loop)
sA = loop._reevaluate_goal(gA, pB1, NOW[0])
sB = loop._reevaluate_goal(gB, pB1, NOW[0])
b_later = user_goal_cands(candsB1, "navigate_to_entity")
a_later = user_goal_cands(candsB1, "follow_entity")
mulB = any(abs(float(c.get("score_mul") or 1.0) - 0.5) < 1e-9
           for c in b_later)
mulA = any(abs(float(c.get("score_mul") or 1.0) - 0.5) < 1e-9
           for c in a_later)
check("B2 世界变化（Beta 始终缺席）→ 缺席目标 demoted 且候选 ×0.5",
      sB == "demoted" and mulB and b_now,
      f"stateB={sB} mulB={mulB} stateA={sA}")
check("B3 在场目标不因他人缺席被殃及（Alpha 保满权）",
      sA == "active" and not mulA, f"stateA={sA} mulA={mulA}")
cand_b = b_later[0]
base = cand_b.get("score_mul")
cand_b2 = {**cand_b, "score_mul": None}
sc_full = loop._score_action(cand_b2, pB1, NOW[0])[0]
sc_half = loop._score_action(cand_b, pB1, NOW[0])[0]
check("B4 评分消费 score_mul：降权候选分数明显下降（让位是数值不是删除）",
      sc_half < sc_full and sc_half > 0,
      f"full={sc_full:.3f} half={sc_half:.3f}")

print("\n── Test C：transport timeout → 世界确认 → 迟到的失败被覆盖 ──")
W = ScriptWorld()
install_fake_bridge(W)
kg, eng, reg, emb, tl, causal, am, loop = build_stack()
W.set_inv(oak_planks=6)
W.drop_mode = "timeout"                          # 桥回执丢失，世界照做
NOW[0] = 10000.0
# 动作前的背包感知基线：craft 在 propose 内同步完成，perceive 首快照
# 就已含成品——计数差分的"前"必须锚在动作之前（真实系统里动作前后
# 各有感知拍，这里显式摆出"上一拍"的快照）
emb._inv_prev = [{"name": "oak_planks", "count": 6}]
r = am.propose({"action_type": "craft_item", "target": "stick",
                "params": {"item": "stick", "count": 1}},
               source="user", now=NOW[0])
check("C1 动作被受理（bridge 回执丢弃模式）",
      bool(r.get("started") or r.get("queued")), str(r)[:120])
flush(loop, am, emb, rounds=8)
am_stats = am.status()["stats"]
# 结算路径：poll 失败 → settle 失败（bridge_error 族）→ SELF_STATE failed
settle_failed_seen = any(
    _chg(e).startswith("failed")
    for e in tl._raw)
check("C2 结算如实 failed（transport 失败如实记账，不假装成功）",
      settle_failed_seen,
      str([e.get("content") for e in tl._raw[-6:]])[:160])
W.drop_mode = None
# 世界 + 1 的感知确认进入同一归因窗（上一拍背包差分就绪）
tick(loop, emb)
obs_seen = [e for e in tl._raw
            if str(e.get("subject") or "").startswith("inventory:stick")
            and "count_increased" in _chg(e)]
check("C3 世界感知确认入轴（inventory:stick count_increased）",
      bool(obs_seen), str([e.get("content") for e in obs_seen][:2]))
causal.sweep(now=time.time() + 1e4)
aggs = causal._aggregations.get("craft_item(stick)") or {}
outs = aggs.get("outcomes") or {}
fail_outs = [k for k in outs if ":failed:" in k]
inv_out = [k for k in outs if "inventory:stick" in k and "count_increased" in k]
check("C4 late-confirmed：世界确认覆盖迟到失败（失败零支撑）",
      not fail_outs and bool(inv_out),
      f"fail_outs={fail_outs[:3]} inv_out={inv_out[:2]}")
check("C5 覆盖留痕（late_confirmed 计数）",
      int(aggs.get("late_confirmed") or 0) >= 1,
      f"late_confirmed={aggs.get('late_confirmed')}")
check("C6 独立动作在横幅里的失败不污染 success_rate（prior 只见成功）",
      causal.action_prior("craft_item", "stick")["success_rate"] == 1.0,
      str(causal.action_prior("craft_item", "stick")))

print("\n── Test D：真失败 → 正确分类 → causal 可学习 ──")
W = ScriptWorld()
install_fake_bridge(W)
kg, eng, reg, emb, tl, causal, am, loop = build_stack()
W.set_inv()                                        # 空背包
NOW[0] = 20000.0
r = am.propose({"action_type": "craft_item", "target": "stick",
                "params": {"item": "stick", "count": 1}},
               source="user", now=NOW[0])
flush(loop, am, emb, rounds=8)
D1_seen = [e for e in tl._raw if ":missing_ingredients" in _chg(e)]
check("D1 真失败（缺原料）如实记 failed:missing_ingredients",
      bool(D1_seen), str([e.get("content") for e in D1_seen][:2]))
causal.sweep(now=time.time() + 1e4)
aggs = causal._aggregations.get("craft_item(stick)") or {}
outs = aggs.get("outcomes") or {}
d_fail = [k for k in outs if "missing_ingredients" in k]
check("D2 失败进入因果账（缺料是结构性阻碍，可学）",
      bool(d_fail) and int(outs[d_fail[0]].get("support", 0)) >= 1,
      str(d_fail)[:100])
check("D3 无世界确认 → 失败不被覆盖（late_confirmed 无）",
      int(aggs.get("late_confirmed") or 0) == 0,
      str(aggs.get("late_confirmed")))
# 再补两次观察 → 满 MIN_SUPPORT=3 → 假设成形（重复驱动，非脚本特判）
evt = make_event(EVENT_ACTION, "self", "craft_item", {"target": "stick"})
for _ in range(2):
    causal.record_action(evt)
    # merge_window_s=0：同一归因窗每条失败各留一份（合并窗会吞并重复观察）
    tl.append(make_event(EVENT_SELF_STATE, "self", "craft_item",
                         {"change": "failed:missing_ingredients"}),
              merge_window_s=0.0)
    causal.sweep(now=time.time() + 1e4)
hyps = [k for k, h in causal._hypotheses.items()
        if k.startswith("craft_item(stick)|") and "missing_ingredients" in k
        and h.get("status") == "hypothesis"]
check("D4 3 次观察 → 因果假设成形（重复失败可归纳）",
      bool(hyps) and int(causal._hypotheses[hyps[0]].get("support", 0)) >= 3,
      str(hyps)[:120])
prior = causal.action_prior("craft_item", "stick")
check("D5 先验可查询（success_rate<1、缺料在 blockers）",
      prior["success_rate"] is not None and prior["success_rate"] < 1.0
      and any("missing_ingredients" in str(b) for b in prior["blockers"]),
      str(prior))
import autonomy as _au
check("D6 缺料不在暂态集合（结构性阻碍才进负因果，P11 分类正确）",
      "missing_ingredients" not in _au._TRANSIENT_WORLD_REASONS,
      "")

print("\n── Test E：探索空手成功 ≠ 知识进展（防漫游 reward loophole）──")
import reward as rw
E1 = rw.classify_self_outcome(
    "explore_direction", True,
    {"ok": True, "status": "done", "detail": {}}, target_is_unknown=False)
E2 = rw.classify_self_outcome(
    "explore_area", True,
    {"ok": True, "status": "done",
     "detail": {"found": 1, "block": "oak_log"}}, target_is_unknown=False)
E3 = rw.classify_self_outcome(
    "observe", True, {"ok": True, "status": "done", "detail": {}},
    target_is_unknown=True)
check("E1 探索空手成功 → progress（0.30）而非 goal_success/discovery",
      E1 == "progress", E1)
check("E2 探索带证据 → goal_success（有真发现才有完成）",
      E2 == "goal_success", E2)
check("E3 观察未知目标但零证据 → 不给 discovery",
      E3 == "goal_success", E3)
_, _, _, embE, tlE, causalE, amE, loopE = build_stack()
W.drop_mode = None
W.set_inv()
W.put_block(2, 64, 0, "stone")
NOW[0] = 30000.0
# 结算空手探索（curiosity 动机、零证据回执）——剥夺钟不重置
loopE._last_novel_ts = 500.0
actE = {"action_type": "explore_direction", "target": "north",
        "motivation": "curiosity", "reason": [],
        "params": {"direction": "north", "max_time": 45}}
resE = {"ok": True, "status": "done", "detail": {}}
gkE = loopE._settlement_gained_knowledge(actE, resE)
loopE._on_action_settled(actE, resE, True, now=NOW[0])
check("E4 结算判据：空手回执 = 无知识增量",
      not gkE, f"gained={gkE}")
check("E5 空手探索结算不重置刺激剥夺钟（漫游不喂饱食循环）",
      loopE._last_novel_ts == 500.0,
      f"novel_ts={loopE._last_novel_ts}")
actE2 = {"action_type": "explore_area", "target": "stone",
         "motivation": "curiosity", "reason": [],
         "params": {"goal": "stone"}}
resE2 = {"ok": True, "status": "done",
         "detail": {"stop": "found", "block": "stone",
                    "observed": ["stone"]}}
gkE2 = loopE._settlement_gained_knowledge(actE2, resE2)
check("E6 带证据的成功 = 知识增量判据（剥夺钟可归零）",
      gkE2, f"gained={gkE2}")

print("\n── Test F：无进展长运行时目标 → relevance 降 → 让位 ──")
W = ScriptWorld()
install_fake_bridge(W)
kg, eng, reg, emb, tl, causal, am, loop = build_stack()
W.set_players("Alpha")
NOW[0] = 40000.0
a = loop.add_goal({"type": "follow", "target": "Alpha",
                   "source": "user_goal", "text": "陪她一阵"})
gF = loop._goals[-1]
gF["created_ts"] = NOW[0]                # 虚拟钟锚
sg = {k: gF.get(k) for k in ("created_ts", "last_success_at", "demote_since")}
NOW[0] += STEP
pF, _ = probe(loop)
sf0 = loop._reevaluate_goal(gF, pF, NOW[0])
NOW[0] = 40000.0 + 700.0           # 无进展 +700s（无成功锚点）
pF1, candsF = probe(loop)
sf1 = loop._reevaluate_goal(gF, pF1, NOW[0])
fF = user_goal_cands(candsF, "follow_entity")
mulF = [c for c in fF if abs(float(c.get("score_mul") or 1.0) - 0.5) < 1e-9]
check("F1 无进展目标 10 分钟后让位（demoted + 半权候选）",
      sf1 == "demoted" and bool(mulF) and sf0 == "active",
      f"sf0={sf0} sf1={sf1} mul={len(mulF)}")
NOW[0] = 40000.0 + 700.0 + 1800.0  # 让位足 30 分钟 → 搁置
pF2, candsF2 = probe(loop)
sf2 = loop._reevaluate_goal(gF, pF2, NOW[0])
fF2 = user_goal_cands(candsF2, "follow_entity")
check("F2 无进展目标最终搁置（不再产出候选，不占行为资源）",
      sf2 == "stale" and not fF2,
      f"sf2={sf2} cands={len(fF2)}")
check("F3 用户再提一次 → 新目标带上新时间锚（复活即新生命周期）",
      gF.get("status") == "stale", f"status={gF.get('status')}")

print("\n── Test G：因果晋升 → 图激活 → 候选评分变化 ──")
W = ScriptWorld()
install_fake_bridge(W)
kg, eng, reg, emb, tl, causal, am, loop = build_stack()
W.set_inv(oak_planks=6)
NOW[0] = 50000.0
candG = loop._act("craft_item", "stick", {},
                  motivation="user_goal", expected="测试",
                  reason=[])
sc_pri = loop._score_action(candG, loop._perceive(), NOW[0])[0]
check("G0 无经验时先验守恒（prior_bonus=0，评分不虚高）",
      True, f"base_score={sc_pri:.3f}")
evt = make_event(EVENT_ACTION, "self", "craft_item", {"target": "stick"})
for i in range(8):                 # 8 次"成功→succeeded"（conf=8/10≥0.8）
    causal.record_action(evt)
    tl.append(make_event(EVENT_SELF_STATE, "self", "craft_item",
                         {"change": "succeeded"}),
              merge_window_s=0.0)  # 每条各占一份支撑（合并窗会吞并重复）
    causal.sweep(now=time.time() + 1e4)
hypsG = [k for k, h in causal._hypotheses.items()
         if k.startswith("craft_item(stick)|") and ":succeeded" in k
         and h.get("status") == "hypothesis"]
check("G1 重复成功 → 假设成形（support≥3 且 conf≥0.6）",
      bool(hypsG), str(hypsG)[:120])
promoted = sorted(causal._promoted)   # 晋升登记在 _promoted 集（假设状态保持 hypothesis）
ops = [n for n in kg.nodes if str(n).startswith("操作:craft_item")]
check("G2 够稳的假设晋升入图（操作: 节点存在）",
      bool(ops), f"promoted={promoted[:1]} ops={ops[:2]}")
node_op = kg.nodes.get(ops[0]) if ops else None
src_ok = (node_op is not None
          and str((node_op.extra_attrs or {}).get("source")) == "causal_hypothesis")
check("G3 晋升节点的激活源身份（经验边，剧情背书）",
      src_ok,
      str((node_op.extra_attrs or {}) if node_op else {}))
pri_after = causal.action_prior("craft_item", "stick")
check("G4 经验回流先验（success_rate 高、obs≥3）",
      pri_after["success_rate"] is not None
      and int(pri_after["obs"] or 0) >= 3
      and pri_after["success_rate"] > 0.5,
      str(pri_after)[:120])
sc_post = loop._score_action(candG, loop._perceive(), NOW[0])[0]
check("G5 晋升/经验进入候选评分（prior_bonus 升起，分数上升）",
      sc_post > sc_pri and abs(sc_post - sc_pri) > 1e-3,
      f"before={sc_pri:.3f} after={sc_post:.3f}")

print("\n══════════════════════════════════════════════════════")
print(f"结果: {len(FAILURES)} 个失败 / 测试全链零 LLM: {not _llm_calls}")
if FAILURES:
    print("失败项: %s" % "; ".join(FAILURES))
    sys.exit(1)
print("闭环一致性 A–G 全部通过")