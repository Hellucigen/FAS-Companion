# sandbox_lab.py — 受控环境实验库（§15-29 论文实验套件）
# ============================================================================
# 进程内 headless 仿真：真实认知栈（KG+diffusion+embodiment+action+autonomy
# +timeline+causal），桥函数绑定 ScriptWorld（虚拟世界：对象/关系/隐藏知识/
# 资源/任务目标/环境变化/干扰物全部由部署侧控制）。
# 本文件不向认知系统写任何"任务答案"：世界只提供对象与配方事实，目标只提供
# goal（obtain X），发现/学习/复用全部走真实机制。
# ScriptWorld/栈装配逻辑来源：tests/test_offline_chain_acceptance.py
# （2026-09-26 离线验收 36/36 定版，含其全部修正教训）。
#
# 用法（每场景独立脚本或 import）：
#   from sandbox_lab import scenario_s3, Scenario, run_scenario
#   run_scenario(Scenario(label="S3-..."))
# ============================================================================

import copy
import json
import logging
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
logging.disable(logging.INFO)

import config as _C
from graph_model import KnowledgeGraph, Node
from diffusion_engine import DiffusionEngine
from cognitive_regulation import CognitiveRegulation
from minecraft.embodiment import MinecraftEmbodiment
from action_system import ActionManager
from experience import ExperienceTimeline, CausalLearner
from autonomy import AutonomousLoop
import experiment_mode as xm
import experiment_recorder as er

# 零 LLM 守卫：沙箱事实/因果路径本来零 LLM（经验识别/缺口/因果全确定性），
# 钉死任何意外 LLM 调用会在测试中炸出来。
import llm_provider as lp
_LLM_GUARDED = []


def _guard(name):
    def _raise(*a, **k):
        raise AssertionError(f"sandbox 零 LLM 契约被打破: {name}")
    _raise._fas_llm_guard = name
    return _raise


def install_llm_guard():
    orig = lp.MiMoBackend._create
    if getattr(orig, "_fas_llm_guard", None):
        return
    g = _guard("mimo")
    g._fas_llm_guard = "sandbox"
    lp.MiMoBackend._create = g


install_llm_guard()

# ── 位置记忆隔离（2026-09-28 S3 根因之二）─────────────────────────────
# embodiment/autonomy 各自无参构造 LocationMemory()，默认落真实世界的
# data/mc_locations.json —— 沙箱会读到 Haru 真世界攒下的旧坐标并据此
# seek/dig。这里把默认 path 改为临时目录下每实例一文件：沙箱进程内完全
# 隔离，且绝不读/写真实位置记忆。进程退出即清理（tempfile 自动）。
import itertools as _it
import skills.base as _sb
_LM_ORIG_INIT = _sb.LocationMemory.__init__
_LM_DIR = tempfile.mkdtemp(prefix="fas_sb_locs_")
_LM_COUNTER = _it.count()


def _sandbox_lm_init(self, path=None):
    if path is None:
        path = os.path.join(_LM_DIR, f"loc_{next(_LM_COUNTER)}.json")
    _LM_ORIG_INIT(self, path)


_sb.LocationMemory.__init__ = _sandbox_lm_init

ENG_CFG = {"lambda_decay": 0.05, "beta_spread": 1.0, "max_depth": 4,
           "theta_threshold": 0.01, "activation_max": 5.0,
           "min_spread_threshold": 0.01, "activation_epsilon": 1e-4,
           "input_similarity_floor": 0.5, "input_default_bonus": 0.5,
           "theta_action": 0.5}

# ── 世界事实表（来源：test_offline_chain_acceptance 同款语义）─────────────
RECIPE_TABLE = {
    "oak_planks": [{"result": "oak_planks", "yield": 4,
                    "ingredients": {"oak_log": 1}, "needs_table": False}],
    "stick": [{"result": "stick", "yield": 4,
               "ingredients": {"oak_planks": 2}, "needs_table": False}],
    "crafting_table": [{"result": "crafting_table", "yield": 1,
                        "ingredients": {"oak_planks": 4},
                        "needs_table": False}],
    "wooden_pickaxe": [{"result": "wooden_pickaxe", "yield": 1,
                        "ingredients": {"oak_planks": 3, "stick": 2},
                        "needs_table": True}],
    "stone_pickaxe": [{"result": "stone_pickaxe", "yield": 1,
                       "ingredients": {"cobblestone": 3, "stick": 2},
                       "needs_table": True}],
    "furnace": [{"result": "furnace", "yield": 1,
                 "ingredients": {"cobblestone": 8}, "needs_table": True}],
    "cobblestone": [],
    "iron_ore": [],
    "raw_iron": [],
    "stone": [],
    "oak_log": [],
    "oak_fence": [{"result": "oak_fence", "yield": 3,
                   "ingredients": {"oak_planks": 4, "stick": 2},
                   "needs_table": True}],
    "iron_ingot": [],          # 熔炼产物——仅 prior_extra 人工先验可达
}

BLOCK_META = {
    "oak_log": {"drops": ["oak_log"], "harvest_tools": []},
    "stone": {"drops": ["cobblestone"], "harvest_tools": ["wooden_pickaxe"]},
    "cobblestone": {"drops": ["cobblestone"],
                    "harvest_tools": ["wooden_pickaxe"]},
    "iron_ore": {"drops": ["raw_iron"], "harvest_tools": ["stone_pickaxe"]},
    "deepslate_iron_ore": {"drops": ["raw_iron"],
                           "harvest_tools": ["stone_pickaxe"]},
    "crafting_table": {"drops": ["crafting_table"], "harvest_tools": []},
    "furnace": {"drops": ["furnace"], "harvest_tools": []},
    "dirt": {"drops": ["dirt"], "harvest_tools": []},
    "grass_block": {"drops": ["dirt"], "harvest_tools": []},
    "diamond_ore": {"drops": ["diamond"], "harvest_tools": ["iron_pickaxe"]},
}
from skills.observation import PICK_TIER as _PTIER

SMELT_PRIOR = {"result": "iron_ingot", "kind": "smelt", "tool": "furnace",
               "inputs": {"raw_iron": 1}, "fuel": True,
               "provenance": {"knowledge_type": "prior",
                              "source": "offline_sim",
                              "version_verified": False, "confidence": 0.6}}


def _d2(pos, xyz):
    return ((pos.get("x", 0) - xyz[0]) ** 2 +
            (pos.get("z", 0) - xyz[2]) ** 2) ** 0.5


class SandboxWorld:
    """虚拟世界（同 ScriptWorld 语义 + 场景编辑助手）。"""

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
        self.calls = []
        self.recipe_table = dict(RECIPE_TABLE)
        self.block_meta = dict(BLOCK_META)
        self.smelt_map = {"raw_iron": "iron_ingot"}

    # ── 环境编辑 ──
    def put_block(self, x, y, z, name):
        self.blocks[(int(x), int(y), int(z))] = name
        self._refresh_near()

    def put_tree(self, x, z, log="oak_log"):
        for i in range(3):
            for j in range(3):
                self.put_block(x + i - 1, 65 + j, z, log)
        self.put_block(x, 65, z, "oak_log")

    def put_ore(self, x, z, name="iron_ore", n=1):
        for i in range(n):
            self.put_block(x + i, 60, z, name)

    def remove_block(self, x, y, z):
        self.blocks.pop((int(x), int(y), int(z)), None)
        self.placed.pop((int(x), int(y), int(z)), None)
        self._refresh_near()

    def set_inv(self, **counts):
        self.inv = [{"name": k, "count": v} for k, v in counts.items() if v]
        self._refresh_near()

    def add_item(self, name, count=1):
        for it in self.inv:
            if it["name"] == name:
                it["count"] += count
                return
        self.inv.append({"name": name, "count": count})

    def inv_map(self):
        return {i["name"]: i["count"] for i in self.inv}

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

    def _refresh_near(self):
        from collections import Counter
        c = Counter()
        for (bx, by, bz), n in (list(self.blocks.items()) +
                                list(self.placed.items())):
            if _d2(self.pos, (bx, by, bz)) <= 4.5:
                c[n] += 1
        self.near_names = [{"name": k, "count": v} for k, v in
                           sorted(c.items(), key=lambda kv: -kv[1])[:6]]

    def state(self):
        return {"connected": True, "username": "Haru",
                "position": dict(self.pos), "health": 20, "food": 20,
                "heldItem": None, "playersNearby": [], "nearbyEntities": [],
                "nearbyBlocks": list(self.near_names), "chat": [],
                "connectedAt": 1, "timeOfDay": self.time_of_day,
                "isRaining": False}

    def near_names_raw(self):
        return [b.get("name") for b in self.near_names]

    def call(self, path, payload=None, timeout=3):
        payload = payload or {}
        self.calls.append((path, dict(payload)))
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
                    out.append({"x": float(bx), "y": float(by), "z": float(bz)})
            return {"ok": True, "positions": out}
        if path == "/dig_pos":
            p = (int(payload.get("x", 0)), int(payload.get("y", 0)),
                 int(payload.get("z", 0)))
            nm = self.blocks.get(p) or self.placed.get(p)
            if nm is None:
                return {"ok": False, "reason": "block_not_found"}
            # C20g（真实失败路径探针）：despawn_on_dig 列名的方块在挖取
            # 时已消失（"树在她到达时已被拿走/枯死"）——第一次挖取即把
            # 方块从世界移除并回 not_found，之后 find_blocks 也找不到它。
            # 环境数据层属性，run 脚本装配；非核心逻辑。
            if (getattr(self, "despawn_on_dig", None)
                    and nm in self.despawn_on_dig):
                self.blocks.pop(p, None)
                self.placed.pop(p, None)
                self._refresh_near()
                return {"ok": False, "reason": "block_not_found"}
            self.blocks.pop(p, None)
            self.placed.pop(p, None)
            meta = self.block_meta.get(nm, {})
            req = meta.get("harvest_tools", [])
            ok_harvest = (not req) or (self.held in req) or any(
                t in _PTIER and _PTIER.get(self.held or "", 0) >= _PTIER[t]
                for t in req)
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
            recs = self.recipe_table.get(item) or []
            if not recs:
                return {"ok": False, "reason": "no_recipe"}
            r = recs[0]
            ing = {k.lower(): int(v) for k, v in
                   (r.get("ingredients") or {}).items()}
            if r.get("needs_table") and "crafting_table" not in \
                    self.near_names_raw() and \
                    "crafting_table" not in self.inv_map():
                self.receipt = {"status": "failed",
                                "detail": {"reason": "no_crafting_table"}}
                return {"ok": False, "reason": "no_crafting_table"}
            if not self.take_items(ing):
                self.receipt = {"status": "failed",
                                "detail": {"reason": "missing_ingredients"}}
                return {"ok": False, "reason": "missing_ingredients"}
            self.add_item(item, int(r.get("yield") or 1))
            self.receipt = {"status": "done", "detail": {"crafted": item}}
            self.last_cmd = "craft"
            return {"ok": True}
        if path == "/place":
            item = str(payload.get("item") or "").lower()
            if self.take_items({item: 1}):
                self.placed[(int(payload.get("x", 0)), 65,
                             int(payload.get("z", 0)))] = item
                self._refresh_near()
                self.receipt = {"status": "done", "detail": {"placed": item}}
            else:
                self.receipt = {"status": "failed",
                                "detail": {"reason": "not_in_inventory"}}
            self.last_cmd = "place"
            return {"ok": True}
        if path == "/smelt":
            inp = str(payload.get("input") or "").lower()
            if self.take_items({inp: 1}) and self.inv_map().get("coal"):
                self.take_items({"coal": 1})
                out = self.smelt_map.get(inp, inp)
                self._smelt_done_queue = out
                self.receipt = {"status": "running",
                                "detail": {"smelting": inp}}
            else:
                self._smelt_done_queue = None
                self.receipt = {"status": "failed",
                                "detail": {"reason": "missing_ingredients"}}
            self.last_cmd = "smelt"
            return {"ok": True}
        if path == "/furnace_take":
            out = getattr(self, "_smelt_done_queue", None)
            if out:
                self.add_item(out)
                self._smelt_done_queue = None
                self.receipt = {"status": "done", "detail": {"took": out}}
            else:
                self.receipt = {"status": "failed",
                                "detail": {"reason": "furnace_empty"}}
            self.last_cmd = "furnace_take"
            return {"ok": True}
        if path == "/equip":
            self.held = payload.get("item")
            self.receipt = {"status": "done", "detail": {}}
            return {"ok": True}
        if path in ("/stop", "/sneak", "/sprint", "/look_at", "/attack",
                    "/combat", "/interact", "/eat", "/dig"):
            self.receipt = {"status": "done", "detail": {}}
            return {"ok": True}
        return {"ok": True}

    def poll_result(self):
        if self.last_cmd == "goto" and self.goto_target:
            tx, ty, tz = self.goto_target
            dx = tx - self.pos["x"]
            dz = tz - self.pos["z"]
            d = (dx * dx + dz * dz) ** 0.5
            if d <= 1.5:
                self.goto_target = None
                self.receipt = {"status": "done", "detail": {}}
            else:
                step_min = min(3.0, d) / max(d, 1e-6)
                self.pos = {"x": self.pos["x"] + dx * step_min,
                            "y": 64.0,
                            "z": self.pos["z"] + dz * step_min}
                self.receipt = {"status": "running", "detail": {}}
            self._refresh_near()
        if self.last_cmd == "smelt" and getattr(self, "_smelt_done_queue",
                                                None):
            self.receipt = {"status": "done",
                            "detail": {"smelted": self._smelt_done_queue}}
            self.last_cmd = "smelt_wait"
        return dict(self.receipt)


import minecraft.bridge as _mb                     # noqa: E402


def _install_sim_gate():
    """仿真时钟压缩伪影补偿（同 tests/test_offline_chain_acceptance.py:
    420-440 注释语义）：skills.run 首拍轮询门槛 poll_after 按真实秒 0.8s，
    而一拍仿真真实耗时远小于它 → 技能永远在 gate 上 pending，watchdog（仿真
    秒）先把动作杀掉（run4/5 gather timeout dur=608 根因，2026-09-26 离线
    验收定版）。生产拍间隔以真实秒计，门槛一拍即开，不存在此问题；此补偿
    只存在于沙箱驱动层，不触碰技能代码。"""
    import skills as _sp
    if getattr(_sp.run, "_fas_sim_nogate", False):
        return
    _real_run = _sp.run

    def _sim_run(name, ctx, params=None, first_poll_delay=0.0, **kw):
        return _real_run(name, ctx, params, 0.0, **kw)
    _sim_run._fas_sim_nogate = True
    _sp.run = _sim_run


_install_sim_gate()


def install_fake_bridge(world):
    _mb.get_state = lambda: world.state()
    _mb.health = lambda: True
    _mb.say = lambda text: True
    _mb.stop = lambda: True
    _mb.stop_goto = lambda: world.call("/stop_goto")
    _mb.stopfollow = lambda: 0
    _mb.stop_combat = lambda: {"ok": True}
    _mb.sprint = lambda on=True: {"ok": True}
    _mb.move = lambda d, s=1.0: True
    _mb.sneak = lambda on=True: {"ok": True}
    _mb.look_at = lambda x, y, z: world.call("/look_at", {"x": x, "y": y, "z": z})
    _mb.look = lambda yaw, pitch=0.0: {"ok": True}
    _mb.goto_coords = lambda x, y, z: world.call("/goto", {"x": x, "y": y, "z": z})
    _mb.goto_player = lambda p: world.call("/goto_player", {"player": p})
    _mb.dig = lambda block, count=1: world.call("/dig", {"block": block})
    _mb.dig_pos = lambda x, y, z: world.call("/dig_pos", {"x": x, "y": y, "z": z})
    _mb.find_blocks = lambda block, radius=12, count=8: world.call(
        "/find_blocks", {"block": block, "radius": radius})
    # 通配兜底：技能层走 ctx.bridge.call(...)（包一层），必须也指回世界——
    # 否则真桥（5010 在线时）会被沙箱进程真的打到，去驱动真实 bot
    _mb.call = world.call
    _mb.collect_item = lambda radius=8: world.call("/collect_item",
                                                   {"radius": radius})
    _mb.place_block = lambda x, y, z, item="": world.call(
        "/place", {"x": x, "y": y, "z": z, "item": item})
    _mb.craft = lambda item, count=1, table=None: world.call(
        "/craft", {"item": item, "count": count})
    _mb.smelt = lambda input_item, fuel, furnace, fuel_count=1: world.call(
        "/smelt", {"input": input_item, "fuel": fuel})
    _mb.furnace_take = lambda furnace: world.call("/furnace_take", {})
    _mb.equip = lambda item: world.call("/equip", {"item": item})
    _mb.unequip = lambda: world.call("/unequip", {})
    _mb.inventory = lambda: {"ok": True, "items": world.inv}
    _mb.recipes = lambda: {"ok": True, "recipes": {
        it: [dict(r) for r in recs] for it, recs in world.recipe_table.items()}}
    _mb.recipe_for = lambda item: {"ok": True,
                                   "recipes": [dict(r) for r in
                                               world.recipe_table.get(item, [])],
                                   "mc_version": "26.1", "mcd_version": "26.1"}
    _mb.block_meta = lambda name: {"ok": True,
                                   "block": name,
                                   "drops": world.block_meta.get(name, {}).get(
                                       "drops", [name]),
                                   "harvest_tools": world.block_meta.get(
                                       name, {}).get("harvest_tools", []),
                                   "mc_version": "26.1"}
    _mb.get_action_result = lambda: world.poll_result()   # 回执即世界推进
    _mb.attack = lambda entity="": {"ok": True}
    _mb.interact_entity = lambda entity: {"ok": True}
    _mb.eat = lambda item: {"ok": True}
    _mb.sleep_at = lambda x, y, z: {"ok": True}
    _mb.drop = lambda item, count=1: {"ok": True}
    _mb.sort_inventory = lambda: {"ok": True}
    _mb.inventory_slots = lambda: {"ok": True}
    _mb.chest_store = lambda keep=[], all_items=False: {"ok": True}
    _mb.chest_take = lambda item, count=1: {"ok": True}
    _mb.goto_entity = lambda entity, range_=1.5: {"ok": True}
    _mb.flee = lambda distance=10: {"ok": True}
    _mb.combat = lambda entity, retreat_health=8, max_ms=45000: {"ok": True}
    _mb.recipe_for = lambda item: {"ok": True,
                                   "recipes": world.recipe_table.get(
                                       str(item).lower(), []),
                                   "mc_version": "26.1",
                                   "mcd_version": "26.1"}


# ── 认知栈装配（full 与消融变体通过参数切换）──────────────────────────────
# 消融开关（每个都是参数级/接线级，不引入新代码分支）：
#   diffusion_on     False → beta_spread=0（激活不传播，仅直写+decay）
#   causal_on        False → CausalLearner 不建、timeline 不接动作事件
#   prior_on         False → 不注入 prior_extra（含熔炉先验）
#   goal_on          False → 不注入 obtain 目标（无目标引导）
#   writeback_on     False → CausalLearner 不晋升到 KG（动作知识不写回图谱）
#   self_goal_on     False → 不把当前目标同步进 self_model（相关自我状态缺失）

class Stack:
    def __init__(self, kg, eng, reg, emb, tl, causal, am, loop, world,
                 diffuser=None, efi=None):
        self.kg, self.engine, self.reg = kg, eng, reg
        self.emb, self.tl, self.causal = emb, tl, causal
        self.am, self.loop, self.world = am, loop, world
        self.diffuser = diffuser
        self.efi = efi          # C20 事件框架装配器（失败抑制探针，可 None）


class SandboxDiffuser:
    """生产扩散驱动器的最小环境装配（C18，2026-09-28）。

    生产里 decay+diffuse 由 continuous_cognition.CCLoop.tick_once 每认知
    tick 驱动（busy/回合让位、_running 门控）。沙箱 stack 从未装配 CC 循环
    → **Full 与 -Diffusion 的激活曲线逐点相同**（实验 B pilot 35/35 零分歧
    即是它没跑的指纹）：扩散在沙箱里不是"被闭包遮蔽"，是根本没有执行器。
    此处按 CC.tick_once 的 diffusion 段同语义补装（decay_step +
    diffuse_step，_running 门控），零改动 engine/CC/autonomy；沙箱无对话
    回合故 busy 让位不涉及。cc_diffuse=False 时装配为 None（连衰减都不
    跑 = pre-C18 行为，回归旧脚本用）。"""

    def __init__(self, engine):
        self.engine = engine

    def __call__(self):
        try:
            if not getattr(self.engine, "_running", False):
                self.engine.decay_step()
                self.engine.diffuse_step()
        except Exception:
            pass


class EventFrameInjector:
    """行动侧事件框架装配（C20，实验 D 失败抑制探针，2026-09-28）。

    愿景形态：情景记忆以事件框架存储，事件经槽位边连到客观事物，正/负
    激活调制下一次扩散。行动侧现状：timeline 流水 + 统计聚合进图，无事件
    框架节点、无极性槽位边（对话侧事件框架 2026-08-14 定型，行动侧未接）。

    本装配件是愿景形态的最小沙箱样板（装配层，非核心机制）：
      - 建签名级"行动经验"事件节点 `行动经验:{intent}({obj})`
        （episodic 空间；独立前哨前缀，不碰 promote 的 操作:/变化: 产物）；
      - 接对象槽位边 `-[涉及]→ 实体`（正权 +0.9，未来可接经验回忆正激发）；
      - 接结果槽位边 `-[结果]→ 失败`（**负权 fail_side**：扩散引擎统一抑制
        机制沿负权边发射抑制、不消耗发射预算，压低对象实体激活——
        diffusion_engine.py:24/1015-1045）；
      - 每 tick 低量再点火唤醒经验节点（模拟持续认知 §485 记忆再点火，
        沙箱无 CC 循环的装配替身）：激活后经负权边持续向对象发射抑制。
    失败经验来源为实验注入（数据/装配层，来源标注），非真实失败路径；
    本探针的回答是"失败抑制边经扩散是否改变候选注意选择"（愿景的
    正/负激活调制行为学首问），不是"FAS 真实失败后自愈"。
    """

    OBJ_REL = "涉及"
    FAIL_REL = "结果"

    def __init__(self, kg, engine, obj, fail_side=-0.8, reign_amt=1.2,
                 targets=(), trigger="prearm"):
        self.kg, self.engine = kg, engine
        self.obj = obj
        self.fail_side = float(fail_side)
        self.reign_amt = float(reign_amt)
        self.trigger = trigger            # "prearm"=build 即落图（C20e）；
        # "on_failure"=真实失败回执才落图（C20g：内源成型——失败发生前
        # 图上无任何事件框架节点，_ensure 延迟到 arm() 由结算链触发）
        self._armed = (trigger != "on_failure")
        # 对象槽位的落点：目标实体可多 id（裸块实体 + 物品节点双挂）。
        # C20b：追加缺口锚点 `缺口:用途({obj})` 负调制——实验 D 行为零差
        # 分的根因是 _score_action.attention=max(reason 激活)（autonomy.py
        # :2693）恒由 gap floor 直写的缺口节点供给（缺口活着=她正在想着
        # 这个对象，floor 0.9+mark_active 每 tick 托底），压实体够不到
        # 注意力。负权边直指缺口节点：让"失败经验抑制"能覆写这条
        # 注意力托底（装配层，不动 autonomy/gap 核心直写）。
        self.eid = f"行动经验:gather_resource({obj})"
        self.failed_nid = f"变化:gather_resource({obj}):failed"
        self.targets = tuple(targets) or (obj, f"物品:{obj}", f"缺口:用途({obj})")
        self._wired = False

    def arm(self, action=None, result=None):
        """真实失败回执触发落图（trigger="on_failure" 用，幂等）。"""
        self._armed = True
        if not self._wired:
            self._ensure()
        return True

    def _ensure(self):
        from graph_model import Node, Edge
        with self.kg._lock:
            n = self.kg.nodes.get(self.eid)
            if n is None:
                self.kg.add_node(Node(
                    id=self.eid, weight=0.5, label="declarative-episodic",
                    graph_space="episodic",
                    extra_attrs={"type": "action_experience",
                                 "name": self.eid,
                                 "source": "experiment-injected"}),)
                n = self.kg.nodes.get(self.eid)
            # C20c：失败经验 = 充分接触 = 对象不再"新奇"。候选 reason 里
            # 的 UnknownBlock_{obj} 在真实系统里由感知建档、经验认领后
            # retired（novelty 熄灭）；沙箱 bridge 只消费建档结果、从不
            # 建档（autonomy.py:2708-2711 回退分支在图里无此节点时无条件
            # 顶格 novelty）。注入器补建 retired 标记——让"反复失败过的
            # 东西"不再持续喂满新颖性——零核心改动，装配层。
            ubid = f"UnknownBlock_{self.obj}"
            if ubid not in self.kg.nodes:
                self.kg.add_node(Node(
                    id=ubid, weight=0.3, label="declarative-semantic",
                    graph_space="semantic",
                    extra_attrs={"type": "unknown", "name": self.obj,
                                 "retired": "experiment-injected-failure",
                                 "source": "experiment-injected"}))
            for tgt in self.targets:
                tn = self.kg.nodes.get(tgt)
                if tn is None:
                    self.kg.add_node(Node(
                        id=tgt, weight=0.3, label="declarative-semantic",
                        graph_space="semantic",
                        extra_attrs={"type": "inventory_item",
                                     "name": tgt, "count": 0}))
                # 槽位边极性 = 经验效价（愿景："槽位通过边连客观事物，
                # 正/负激活影响下一次扩散"）。注入失败经验 → 涉及边带
                # fail_side 负权；成功经验则为正权（本探针只注入失败）。
                # 引擎约束（diffusion_engine.py:1021）：负权边的抑制贡献
                # 只在节点 total_w>0（至少一条正权出边）时才被发射——事件
                # 到结果节点的 +0.9 结果边即承担该角色；且抑制直接落在
                # 实体上（C20 初版错误地把 -0.8 连向失败节点这个孤立叶子，
                # 抑制信号永远到不了对象，行为零差分的假阴性根因）。
                if not self.kg.get_edge(self.eid, tgt, self.OBJ_REL):
                    self.kg.add_edge(Edge(src=self.eid, dst=tgt,
                                          relation=self.OBJ_REL,
                                          weight=self.fail_side,
                                          relation_category="cognitive_relation"))
            # 结果槽位边：事件→失败结果节点。权重恒 +0.9（结果边是事件
            # 出边的唯一正权，保证 total_w>0 → 抑制贡献才会被引擎发射，
            # diffusion_engine.py:1021；失败极性已由涉及边负权承担——事件
            # 的"失败"用 -0.8×(涉及边) 压实体，+0.9 结果边只负责搭起发射
            # 循环，不走失败语义）。
            fn = self.kg.nodes.get(self.failed_nid)
            if fn is None:
                self.kg.add_node(Node(
                    id=self.failed_nid, weight=0.3, label="declarative-semantic",
                    graph_space="semantic",
                    extra_attrs={"type": "outcome",
                                 "name": self.failed_nid}))
            if not self.kg.get_edge(self.eid, self.failed_nid, self.FAIL_REL):
                self.kg.add_edge(Edge(src=self.eid, dst=self.failed_nid,
                                      relation=self.FAIL_REL,
                                      weight=0.9,
                                      relation_category="cognitive_relation"))
        self._wired = True

    def __call__(self):
        """每认知 tick：经验节点低量再点火 → 沿负权结果边持续抑制对象。"""
        try:
            if not self._armed:
                return          # C20g：on_failure 模式等真实失败 arm 后再点火
            if not self._wired:
                self._ensure()
            n = self.kg.nodes.get(self.eid)
            if n is None:
                return
            n.activation = min(5.0, float(n.activation or 0.0) + self.reign_amt)
            n.touch()
            self.engine.mark_active([self.eid])
        except Exception:
            pass


def build_stack(goal=None, label="", diffusion_on=True, causal_on=True,
                prior_on=True, goal_on=True, writeback_on=True,
                self_goal_on=True, gap_on=True, seed=None, base_parent=None,
                inherit_graph=None, prior_level="C", mc_gatherable=None,
                cc_diffuse=True, inherit_causal=None, inherit_tl=None,
                eventframe=None, prior_extra=None):
    """每场景一套干净世界+图谱。返回 Stack。
    inherit_graph: 上一 episode 的图谱 JSON 路径（KnowledgeGraph.load
    转录，含其全部节点/边/权重/激活历史）。用于跨 episode 复用验证
    （S3-EpB/M2）：新世界的知识密度来自旧世界经验，而非重新发现。
    mc_gatherable: 环境事实追加表（列表）——自创对象的"可采资源"分类
    归属（C17d，实验 A/B 自制矿石世界）；只影响本 stack 的 config 数据。
    gap_on=False → config["prior"]["enabled"]=False（§7 消融 −Gap）：
    唯一消费点是 autonomy._gap_candidates 的门控（autonomy.py:1469，
    探索缺口→目标/好奇候选），单点关闭，无旁路。
    inherit_causal / inherit_tl（C19，实验 C Transfer 条件）：传入上一
    stack 的 CausalLearner 实例与 ExperienceTimeline 实例 → 跨任务因果
    账本与时间轴连续（同进程通道），装配套件零核心改动。
    eventframe（C20，实验 D 失败抑制探针）：tuple (obj, fail_side,
    reign_amt) 或 (obj, fail_side, reign_amt, trigger) → 装配
    EventFrameInjector（行动侧事件框架签名节点 + 极性槽位边 + 再点火；
    负权结果边经扩散抑制对象实体）。trigger="prearm"（默认）= build 时
    落图 + 恒点火（C20e）；trigger="on_failure"（C20g）= 图状态干净，
    _ensure 延迟到真实失败回执（ActionManager.on_settled 链触发 arm）。"""
    if base_parent is None:
        base_parent = tempfile.mkdtemp(prefix=f"fas_sb_{label or 'run'}_")
    BASE = os.path.join(base_parent, "app")
    shutil.rmtree(BASE, ignore_errors=True)
    os.makedirs(BASE, exist_ok=True)
    world = SandboxWorld()
    install_fake_bridge(world)

    kg = KnowledgeGraph()
    if inherit_graph and os.path.exists(inherit_graph):
        prior = KnowledgeGraph.load(inherit_graph)
        for nid, nd in prior.nodes.items():
            if nid not in kg.nodes:
                kg.nodes[nid] = nd
        for ed in getattr(prior, "edges", []) or []:
            kg.edges.append(ed)
        kg.rebuild_indexes()
    kg.add_node(Node(id="Haru", graph_space="self"))
    for nid in ("Haru的位置", "Haru的血量", "Haru的饥饿", "Haru的手持物",
                "附近的玩家", "附近的方块", "附近的生物",
                "未知信息", "地面物品", "建筑结构", "生存需求"):
        kg.add_node(Node(id=nid, label="declarative-semantic",
                         graph_space="cognitive"))
    eng_cfg = dict(ENG_CFG)
    if not diffusion_on:
        eng_cfg["beta_spread"] = 0.0
    eng = DiffusionEngine(kg, eng_cfg)
    eng.name_to_node = dict(kg.nodes)
    cfgd = dict(_C.DEFAULT_CONFIG)
    if not gap_on:
        cfgd["prior"] = dict(cfgd.get("prior") or {})
        cfgd["prior"]["enabled"] = False   # §7 −Gap：探索缺口通道单点关闭
    # 先验知识条件（§10 A/B/C，2026-09-28 补跑）：
    #   C（complete）= 现状：图谱先验边 + 配方表 + 方块表 + 任务先验。
    #   B（limited）= 图谱先验查询通道关闭（prior.enabled=False，同 −Prior
    #     消融开关；缺口通道随同关闭——本实验所有条件同构，非关注点），
    #     配方表/方块表（世界机制知识）保留。
    #   A（zero/minimal）= B 基础上连配方表/方块表也清空——任何计划知识都
    #     不提供，行为只能从感知/扩散/因果学习累积涌现。
    # 注意：A/B 下 prior_extra 清空（SMELT_PRIOR 等任务专属先验不注入）。
    if prior_level in ("A", "B"):
        cfgd["prior"] = dict(cfgd.get("prior") or {})
        cfgd["prior"]["enabled"] = False
    if prior_level == "A":
        world.recipe_table = {}
        world.block_meta = {}
    # 沙箱探索量程匹配（环境侧伪影修正，技能/认知零改动）：探索技能的路点
    # 选择按足迹罚分在沙箱小世界里很快进入"原点环上转圈"，唯一出口是
    # 壁钟 deadline（explore_max_time_s=300 真实秒）——整场 run 才几十真实
    # 秒，技能永续巡航，把烧着的炉子晾在一边（2026-09-28 iron_ingot 链根因，
    # 与"离线验收三坑"同族的仿真时钟伪影）。缩小距离/步长/时限让一次探索
    # 在数拍内以 time_limit/distance_limit 诚实结算，决策权还给炉前取货。
    cfgd["explore_max_time_s"] = 32.0
    cfgd["explore_step"] = 6.0
    cfgd["explore_max_distance"] = 24.0
    # 环境事实追加（C17d）：自创对象的可采归属经数据表注入，非代码分支
    if mc_gatherable:
        _gl = cfgd.get("mc_world") or {}
        _g = list(_gl.get("gatherable") or [])
        for _nm in mc_gatherable:
            if _nm not in _g:
                _g.append(_nm)
        _gl["gatherable"] = _g
        cfgd["mc_world"] = _gl
    reg = CognitiveRegulation(kg=kg, engine=eng, data_dir=BASE)
    eng.set_lock_registry(reg.locks)
    emb = MinecraftEmbodiment(kg=kg, engine=eng, perceive_into_graph=True,
                              config=cfgd)
    if inherit_tl is not None:
        # 实验 C（Transfer 条件）：同一 timeline 继续追加——跨任务因果账本
        # 通道（CausalLearner._load 恢复同一段 causal section；事件/聚合
        # 均连续），装配套件，非核心机制改动。
        tl = inherit_tl
    else:
        exp_path = os.path.join(BASE, "experience_timeline.json")
        tl = ExperienceTimeline(path=exp_path, config=cfgd)
    emb.timeline = tl
    causal = None
    if causal_on:
        if inherit_causal is not None:
            # 实验 C（Transfer 条件）：同一 CausalLearner 实例跨任务延续
            # （action_prior 成功率记忆 + 归因窗），装配套件。
            causal = inherit_causal
        else:
            # writeback 消融：kg/engine 置 None → promote_to_kg 无目标可写
            # （experience.py:774 kg is None → return []，空安全），聚合照常，
            # 只断"知识写回图谱"这一环。
            causal = CausalLearner(tl, config=cfgd,
                                   kg=kg if writeback_on else None,
                                   engine=eng if writeback_on else None)
    am = ActionManager(embodiment=emb, kg=kg, engine=eng, config=cfgd,
                       data_dir=BASE, timeline=tl, causal=causal)
    loop = AutonomousLoop(kg=kg, engine=eng, config=cfgd, regulation=reg,
                          data_dir=BASE, action_manager=am)
    loop.register_embodiment(emb)
    loop.set_mode("on")
    from action_concepts import ensure_action_concepts
    from capability_graph import CapabilityIndex
    from embodied_mapper import EmbodiedStateMapper
    import mc_knowledge as _mck
    ensure_action_concepts(kg, eng)
    ci = CapabilityIndex(kg, eng, cfgd)
    ci.embodiment = emb
    ci.ensure_graph()
    loop.cap_index = ci
    loop.mapper = EmbodiedStateMapper(kg, eng, cfgd)
    _mck.init_protection_node(cfgd)
    _mck.ensure_mc_world(kg, cfgd)
    exp_cfg = {
        "mode": "learning_closed_loop" if goal_on and goal else "off",
        "goal_obtain": goal or "",
        "attention_floor": 0.9,
        # prior_extra 显式覆盖（C31，实验 H 分解条件 L3b）：None=旧行为
        # （C 注入 SMELT_PRIOR，A/B 清空）；列表=按给定注入（与 prior_level
        # 解耦，供"机制表保留 + 熔炼任务先验"的分解对照）。
        "prior_extra": (prior_extra if prior_extra is not None else
                        ([SMELT_PRIOR] if prior_on
                         and prior_level == "C" else [])),
        "seed": seed,
    }
    cfgd["experiment"] = exp_cfg
    xm.configure(cfgd)
    import world_prior as _wp0
    _wp0.bind_config(cfgd)
    _wp0._RUNTIME["queried"].clear()
    if goal_on and goal:
        loop.add_goal({"type": "obtain", "target": goal,
                       "source": "experiment_obtain",
                       "text": f"实验目标：自主获得 {goal}"})
    if not self_goal_on:
        # 相关自我状态缺失：不走 current_goal 同步（self_graph 照常建，
        # 只是当前目标不落入 self-model 参照面）
        try:
            loop._sync_self_goal = lambda *a, **k: None
        except Exception:
            pass
    efi = (EventFrameInjector(kg, eng, eventframe[0],
                              fail_side=eventframe[1],
                              reign_amt=eventframe[2],
                              trigger=eventframe[3]
                              if len(eventframe) > 3 else "prearm")
           if eventframe else None)
    if efi is not None and efi.trigger == "on_failure":
        # C20g：真实失败回执链（装配层）——ActionManager 结算失败时
        # 内源触发事件框架落图（不是 build 预注入）。包链保原回调：
        # 先让自主层原语义（recency/失败计数/causal）完整跑，再 arm。
        _orig_settled = am.on_settled

        def _chained_settled(action, result, success, now=None):
            try:
                _orig_settled(action, result, success, now)
            except Exception:
                try:
                    _orig_settled(action, result, success)
                except Exception:
                    pass
            if not success and not (result or {}).get("cancelled"):
                efi.arm(action, result)
        am.on_settled = _chained_settled
    return Stack(kg, eng, reg, emb, tl, causal, am, loop, world,
                 diffuser=SandboxDiffuser(eng) if cc_diffuse else None,
                 efi=efi)


class Clock:
    """仿真时钟（16s/拍，与离线验收同款）。"""

    def __init__(self, start=1000.0, step=16.0):
        self.now = start
        self.step = step


def tick(st: Stack, clk: Clock):
    clk.now += clk.step
    try:
        st.emb.ctx.invalidate()
    except Exception:
        pass
    try:
        st.emb._inv_last_pull = 0.0
    except Exception:
        pass
    try:
        st.loop.kg
        for _n, _nd in st.kg.nodes.items():
            if _n not in st.engine.name_to_node:
                st.engine.name_to_node[_n] = _nd
    except Exception:
        pass
    for _ in range(int(clk.step / 4.0)):
        st.world.poll_result()
    res = st.loop.tick(now=clk.now)
    if st.diffuser is not None:
        # C18：生产扩散驱动器（decay+diffuse，见 SandboxDiffuser）
        st.diffuser()
    if st.efi is not None:
        # C20：行动侧事件框架再点火（失败抑制探针，见 EventFrameInjector）
        st.efi()
    return res


def flush_causal(st: Stack, now=None):
    """沙箱收尾把因果归因窗确定性关窗（CausalLearner.sweep 兜底）。

    沙箱加速时钟 vs 壁钟窗口：因果窗（8–120s）挂在动作的**壁钟**时间上，
    而整场 run 只跑几真实秒——窗口永不到期，账本悬浮在 _pending。真实
    MC 运行持续几十分钟，晚到事件自然会到期关窗，无需此步。这里用
    "远未来"一次推进会话内全部 pending 窗关窗记账（sweep 本就是给
    "时间轴安静时/测试的确定性把手"设计的公开 API）。
    """
    if st.causal is None:
        return 0
    import time as _t
    return st.causal.sweep(now=now if now is not None else _t.time() + 2 ** 31)


def goal_state(st: Stack):
    try:
        gs = st.loop.goals() or []
        return [{"type": g.get("type"), "target": g.get("target"),
                 "success": bool(g.get("last_success_at")),
                 "last_success": g.get("last_success_at")} for g in gs]
    except Exception as e:
        return [{"_err": str(e)[:80]}]


def have_item(st: Stack, name):
    return st.world.inv_map().get(name, 0)


def graph_stats(st: Stack):
    return {"nodes": len(st.kg.nodes), "edges": len(st.kg.edges)}


def find_edges(st: Stack, src=None, dst=None):
    out = []
    for e in st.kg.edges.values() if hasattr(st.kg.edges, "values") \
            else st.kg.edges:
        s, d = (e.src, e.dst) if hasattr(e, "src") else \
            (e.get("src"), e.get("dst"))
        if src and s != src:
            continue
        if dst and d != dst:
            continue
        out.append((s, d, getattr(e, "weight", None)))
    return out