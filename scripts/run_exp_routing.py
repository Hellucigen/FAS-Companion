# run_exp_routing.py — 核心机制验证：显式认知资源路由 vs 历史/检索式信息供给
# （Paper-I routing campaign；CHANGE_LOG C33。四条件 LLM 决策头协议。）
# ─────────────────────────────────────────────────────────────────────
# 研究问题（唯一）：在 LLM 能力、环境、任务、先验、预算一致时，FAS 的
#   显式图谱 + 动态激活 + 注意转移 + Cognitive Demand/Gap/Resource Routing
#   是否比 "LLM + 历史 / 普通检索" 更有效地选择与利用当前真正需要的信息？
#
# 公平性设计（§7）：唯一自变量 = 决策前 LLM 获得的信息及其选择方式。
#   - 同一 LLM（MiMo mimo-v2.6-flash，temperature=0，max_tokens=40，1 次
#     调用/决策步）；同一环境（SandboxWorld）；同一任务实例（同 seed 同
#     布局）；同一先验（任务事实在 base prompt 里，四条件相同）；同一
#     动作菜单（由环境推导，与条件无关）；同一经历流（所有条件写入同一
#     格式的 experience store）；同一决策步数上限。
# 条件：
#   A llm_direct     base prompt（任务+事实+环境态+菜单）
#   B llm_history    base + 原始经历流最近 8 条（无精选）
#   C llm_rag        base + BGE 相似度检索 top-5 经历条目（预注册 k=5）
#   D fas_full       base + 真实 FAS 机制选出的上下文：真 KnowledgeGraph +
#                    DiffusionEngine（activate_from_inputs/decay/diffuse/
#                    get_topk）+ 真 analyze_cognitive_demand（三层需求/
#                    缺口/路由）。无原始历史。
#   E random_ctx     base + 与 D 等量的随机经历条目（token 尽量匹配）——
#                    反事实对照（§19）：激活选择是否真有信息选择价值
# 消融（T2/T3，§14）：D-noact（无扩散，图检索序）/ D-nodemand（Top-k 但
#   无 demand/gap/routing 注解）/ D-flat（图节点文本扁平相似度，无传播）
# 判定头执行器：透明沙箱技能原语（gather/craft/place/explore/eat），
#   四条件完全一致；执行层不是研究对象，信息选择才是。
# 指标（§12）：success/decisions、target_utilization、distractor_rate、
#   T_reroute、obsolete_actions、retrieval_precision（程序化相关性标注）、
#   context tokens、efficiency（success/1k input tokens）、逐决策 JSON。
# 统计（§17）：配对 Wilcoxon signed-rank + 效应量 r；n=20 paired seeds。
# ─────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import random
import re
import sys
import time
import zlib
from collections import Counter

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
if os.getcwd() != os.path.dirname(os.path.dirname(os.path.abspath(__file__))):
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox_lab import SandboxWorld, RECIPE_TABLE, BLOCK_META  # noqa

# ── LLM 决策头（同一模型/参数；用量记录）─────────────────────────
class LLMHead:
    def __init__(self):
        import openai
        sec = json.load(open("data/llm_secret.json", encoding="utf-8"))
        from llm_provider import MiMoBackend
        self.model = MiMoBackend._DEFAULT_MODEL
        self.url = MiMoBackend._DEFAULT_BASE_URL
        self.client = openai.OpenAI(api_key=sec["api_key"], base_url=self.url)
        self.temperature = 0
        self.max_tokens = 40
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency = 0.0

    def decide(self, prompt):
        t0 = time.time()
        last = None
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                self.calls += 1
                self.latency += time.time() - t0
                u = getattr(r, "usage", None)
                if u:
                    self.prompt_tokens += u.prompt_tokens or 0
                    self.completion_tokens += u.completion_tokens or 0
                return r.choices[0].message.content or ""
            except Exception as e:                      # 退避重试，不吞
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM failed after retries: {last!r}")


# ── 透明执行器（四条件一致）─────────────────────────────────────
class Executor:
    """沙箱技能原语。gather=find+dig+掉落入包（按 BLOCK_META，含工具需求）；
    craft=配方消耗/产出（needs_table 检查已放置工作台）；explore=移动；
    eat=消耗食物抬饥饿。所有动作返回 (ok, detail)。"""

    def __init__(self, world):
        self.w = world
        self.hunger = 20

    # ---- 菜单构建（环境推导，条件无关）----
    def menu(self, task):
        items = []
        for name in sorted(set(self.w.near_names_raw())):
            if name in ("grass_block", "dirt") and "dirt" not in task.gatherable:
                continue
            items.append(("gather_" + name, name))
        for out, recs in sorted(self.w.recipe_table.items()):
            if not recs:
                continue
            inv = self.w.inv_map()
            for r in recs:
                if all(inv.get(k, 0) >= v for k, v in r["ingredients"].items()):
                    items.append(("craft_" + out, out))
                    break
        for it in self.w.inv_map():
            if it in ("bread", "apple"):
                items.append(("eat_" + it, it))
            if it in ("crafting_table", "furnace"):
                items.append(("place_" + it, it))
        for d in ("north", "south", "east", "west"):
            items.append(("explore_" + d, d))
        items.append(("wait", None))
        return items

    def execute(self, act, obj):
        w = self.w
        if act.startswith("gather_"):
            target = act[len("gather_"):]
            pos = None
            best = 1e9
            for (bx, by, bz), nm in list(w.blocks.items()) + list(w.placed.items()):
                if nm == target:
                    d = ((w.pos["x"] - bx) ** 2 + (w.pos["z"] - bz) ** 2) ** 0.5
                    if d < best:
                        best, pos = d, (bx, by, bz)
            if pos is None or best > 16:
                return False, "not_found"
            meta = w.block_meta.get(target, {})
            tools = meta.get("harvest_tools", [])
            if tools and not any(t in w.inv_map() for t in tools):
                return False, f"tool_missing:{tools[0]}"
            w.remove_block(*pos)
            drops = meta.get("drops", [])
            if not drops:
                return True, "dug_no_drop"
            for dname in drops:
                w.add_item(dname, 1)
            return True, f"got:{drops[0]}"
        if act.startswith("craft_"):
            out = act[len("craft_"):]
            recs = self.w.recipe_table.get(out) or []
            inv = w.inv_map()
            for r in recs:
                if all(inv.get(k, 0) >= v for k, v in r["ingredients"].items()):
                    if r.get("needs_table") and "crafting_table" not in w.near_names_raw():
                        return False, "needs_crafting_table"
                    if not w.take_items(r["ingredients"]):
                        return False, "missing_ingredients"
                    w.add_item(out, r.get("yield", 1))
                    return True, f"crafted:{out}"
            return False, "missing_ingredients"
        if act.startswith("place_"):
            it = act[len("place_"):]
            if w.inv_map().get(it, 0) < 1:
                return False, "not_in_inventory"
            w.take_items({it: 1})
            w.put_block(int(w.pos["x"]), 64, int(w.pos["z"]) - 1, it)
            return True, f"placed:{it}"
        if act.startswith("explore_"):
            d = act[len("explore_"):]
            step = {"north": (0, -4), "south": (0, 4), "east": (4, 0), "west": (-4, 0)}[d]
            w.pos["x"] += step[0]; w.pos["z"] += step[1]
            w._refresh_near()
            return True, f"pos:{int(w.pos['x'])},{int(w.pos['z'])}"
        if act.startswith("eat_"):
            it = act[len("eat_"):]
            if w.inv_map().get(it, 0) < 1:
                return False, "not_in_inventory"
            w.take_items({it: 1})
            self.hunger = min(20, self.hunger + 7)
            return True, f"ate:{it},hunger:{self.hunger}"
        if act == "wait":
            return True, "waited"
        return False, "unknown_action"


# ── FAS 机制（条件 D 真实组件）──────────────────────────────────
class FASContext:
    """真 KnowledgeGraph + DiffusionEngine + analyze_cognitive_demand。
    每决策步：把当步观察/动作/结果 activate_from_inputs 入图 → decay →
    diffuse → get_topk → demand/gap/routing 注解。"""

    def __init__(self, use_diffusion=True, use_demand=True):
        from graph_model import KnowledgeGraph
        from diffusion_engine import DiffusionEngine
        self.use_diffusion = use_diffusion
        self.use_demand = use_demand
        self.kg = KnowledgeGraph()
        cfg = {"beta_spread": 1.0, "activation_epsilon": 1e-4,
               "min_spread_threshold": 0.01, "activation_max": 5.0}
        self.eng = DiffusionEngine(self.kg, cfg)
        self.eng.name_to_node = dict(self.kg.nodes)

    def _ensure(self, nid, label="declarative-semantic"):
        from graph_model import Node
        if nid not in self.kg.nodes:
            self.kg.add_node(Node(id=nid, weight=0.4, label=label,
                                  graph_space="semantic"))
        if nid not in self.eng.name_to_node:
            self.eng.name_to_node[nid] = self.kg.nodes[nid]
        return nid

    def ingest(self, obs_text, action, result, goal_entities):
        """把一步经验入图：观察实体节点 + 动作节点 + 共现边。"""
        node_ids, edge_specs = [], []
        for tok in re.findall(r"[a-zA-Z_]+", obs_text):
            if tok in GENERIC_TOKENS or tok in ("not_found", "disappeared"):
                continue
            if tok not in GOAL_VOCAB:
                continue          # C34：只入图已知实体（生产感知语义）
            nid = self._ensure("实体:" + tok)
            node_ids.append(nid)
        for g in goal_entities:
            node_ids.append(self._ensure("实体:" + g))
        if action:
            an = self._ensure("动作:" + action, label="procedural")
            node_ids.append(an)
            for nid in node_ids[:-1]:
                edge_specs.append({"src": an, "dst": nid, "type": "共现"})
        if result:
            rn = self._ensure("结果:" + result[:24])
            node_ids.append(rn)
        try:
            self.eng.activate_from_inputs(node_ids, edge_specs,
                                          source_type="external_input")
        except Exception as e:
            print(f"[FAS ingest warn] {e!r}", flush=True)

    def step_dynamics(self):
        try:
            self.eng.decay_step()
            if self.use_diffusion:
                self.eng.diffuse_step()
        except Exception as e:
            print(f"[FAS dyn warn] {e!r}", flush=True)

    def context(self, task_text):
        top = self.eng.get_topk(k=8)[0]
        node_lines = [f"- {getattr(n,'id','?')} (act {round(float(getattr(n,'activation',0) or 0),2)})"
                      for n in top]
        demand_line = ""
        if self.use_demand:
            try:
                import cognitive_demand as cd
                out = cd.analyze_cognitive_demand(
                    text=task_text, parsed={}, kg=self.kg, engine=self.eng,
                    exploration={}, should_speak=False)
                demand, gap, route = out[0], out[1], out[2]
                dtop = sorted(demand.items(), key=lambda kv: -kv[1])[:3]
                demand_line = ("Cognitive demand (top dims): "
                               + ", ".join(f"{k}={v}" for k, v in dtop)
                               + "\nResource routing: "
                               + json.dumps(route.get("route", route), ensure_ascii=False)[:160])
            except Exception as e:
                demand_line = f"(demand module unavailable: {type(e).__name__})"
        return "Cognitive-graph focus (top activated):\n" + "\n".join(node_lines) + ("\n" + demand_line if demand_line else "")


# ── 经历存储与检索 ───────────────────────────────────────────────
class MemoryStore:
    def __init__(self):
        self.entries = []          # ["obs: ... | action: ... | result: ..."]
        self._emb = None
        self._matrix = None

    def add(self, text):
        self.entries.append(text)

    def history_block(self, k=8):
        return "\n".join("- " + e for e in self.entries[-k:])

    def _embedder(self):
        if self._emb is None:
            from embedding_manager import EmbeddingProvider
            self._emb = EmbeddingProvider()
        return self._emb

    def rag_block(self, query, k=5):
        emb = self._embedder()
        import numpy as np
        if self._matrix is None or len(self._matrix) != len(self.entries):
            texts = self.entries or ["(empty)"]
            self._matrix = emb.encode(texts)
        if not self.entries:
            return "(no memories yet)"
        q = emb.encode_single(query)
        sims = self._matrix @ (q / (np.linalg.norm(q) + 1e-9))
        idx = sorted(range(len(sims)), key=lambda i: -float(sims[i]))[:k]
        return "\n".join("- " + self.entries[i] for i in idx)

    def random_block(self, n, rng):
        if not self.entries:
            return "(no memories yet)"
        picks = [self.entries[i] for i in rng.sample(
            range(len(self.entries)), min(n, len(self.entries)))]
        return "\n".join("- " + e for e in picks)


GOAL_VOCAB = set()


# ── 任务定义 ─────────────────────────────────────────────────────
class Task:
    name = "?"
    goal_text = ""
    facts = ""
    gatherable = ()
    goal_entities = ()

    def setup_phase(self, w, ex, phase):
        raise NotImplementedError

    def phases(self):
        return 1

    def phase_goal(self, phase):
        return self.goal_text

    def phase_facts(self, phase):
        return self.facts

    def done(self, w):
        raise NotImplementedError

    def max_decisions(self, phase):
        return 8


class T1Reuse(Task):
    name = "T1_knowledge_reuse"
    goal_text = "Obtain 1 oak_planks in inventory."
    facts = "Recipe: 1 oak_log -> 4 oak_planks (craftable by hand). Oak trees can be gathered for oak_log."
    gatherable = ("oak_log",)
    goal_entities = ("oak_log", "oak_planks")

    def phases(self):
        return 2

    def setup_phase(self, w, ex, phase):
        w.__init__()                      # 世界重置（同 seed 同布局）
        w.pos = {"x": 0.0, "y": 64.0, "z": 0.0}
        ex.hunger = 20
        for (x, z) in ((3, 0), (5, 0)):
            for i in range(3):
                for j in range(3):
                    w.put_block(x + i - 1, 65 + j, z, "oak_log")
            w.put_block(x, 65, z, "oak_log")
        w._refresh_near()

    def done(self, w):
        return w.inv_map().get("oak_planks", 0) >= 1

    def max_decisions(self, phase):
        return 6 if phase == 1 else 7


class T2Distractor(Task):
    name = "T2_distractor_suppression"
    goal_text = "Obtain 1 oak_planks in inventory."
    facts = "Recipe: 1 oak_log -> 4 oak_planks (hand). Only oak_log matters for this goal."
    gatherable = ("oak_log", "coal", "iron_ore", "stone", "dirt", "birch_log")
    goal_entities = ("oak_log", "oak_planks")

    def phases(self):
        return 1

    def setup_phase(self, w, ex, phase):
        w.__init__()
        w.pos = {"x": 0.0, "y": 64.0, "z": 0.0}
        ex.hunger = 20
        for i in range(3):
            for j in range(3):
                w.put_block(3 + i - 1, 65 + j, 0, "oak_log")
        w.put_block(3, 65, 0, "oak_log")
        for i in range(3):
            for j in range(3):
                w.put_block(-3 + i - 1, 65 + j, -3, "birch_log")
        w.put_block(-3, 65, -3, "birch_log")
        w.put_block(2, 65, 3, "coal_ore"); w.put_block(3, 65, 3, "coal_ore")
        w.put_block(-2, 65, 3, "iron_ore")
        w.put_block(3, 65, -3, "stone"); w.put_block(4, 65, -3, "stone")
        w.block_meta.setdefault("coal_ore", {"drops": ["coal"], "harvest_tools": []})
        w.block_meta.setdefault("iron_ore", {"drops": ["raw_iron"], "harvest_tools": []})
        w._refresh_near()

    def done(self, w):
        return w.inv_map().get("oak_planks", 0) >= 1

    def max_decisions(self, phase):
        return 9


class T3Reroute(Task):
    """Phase1: 橡木链建立计划；Phase2: 橡木消失，桦木（同构替代资源）
    在同一位置出现且可合成——环境改变后的认知资源重路由。四条件的
    goal/facts/环境信息同权，唯一差异是上下文选择。"""
    name = "T3_environment_reroute"
    goal_text = "Obtain 1 oak_planks in inventory."
    facts = "Recipe: 1 oak_log -> 4 oak_planks (hand). Oak trees provide oak_log."
    gatherable = ("oak_log",)
    goal_entities = ("oak_log", "oak_planks", "birch_log", "birch_planks")

    def phases(self):
        return 2

    def phase_goal(self, phase):
        return self.goal_text if phase == 1 else             "Obtain 1 birch_planks in inventory. (The oak trees are gone.)"

    def phase_facts(self, phase):
        return self.facts if phase == 1 else             "Recipe: 1 birch_log -> 4 birch_planks (hand). Birch trees provide birch_log."

    def setup_phase(self, w, ex, phase):
        w.__init__()
        w.pos = {"x": 0.0, "y": 64.0, "z": 0.0}
        ex.hunger = 20
        if phase == 1:
            for i in range(3):
                for j in range(3):
                    w.put_block(3 + i - 1, 65 + j, 0, "oak_log")
            w.put_block(3, 65, 0, "oak_log")
        else:
            # 环境改变：橡木消失；同位置出现桦木（同构替代，视野内）
            for i in range(3):
                for j in range(3):
                    w.put_block(3 + i - 1, 65 + j, 0, "birch_log")
            w.put_block(3, 65, 0, "birch_log")
            w.block_meta["birch_log"] = {"drops": ["birch_log"], "harvest_tools": []}
            # 同构替代配方（世界侧数据，四条件同等可得）
            w.recipe_table["birch_planks"] = [
                {"result": "birch_planks", "yield": 4,
                 "ingredients": {"birch_log": 1}, "needs_table": False}]
        w._refresh_near()

    def done(self, w):
        return (w.inv_map().get("oak_planks", 0) >= 1
                or w.inv_map().get("birch_planks", 0) >= 1)

    def max_decisions(self, phase):
        return 5 if phase == 1 else 8


class T4Competing(Task):
    name = "T4_competing_goals"
    goal_text = "Obtain 1 oak_planks in inventory. Manage your hunger: if hunger stays 0 you lose."
    facts = "Recipe: 1 oak_log -> 4 oak_planks (hand). Bread restores +7 hunger."
    gatherable = ("oak_log",)
    goal_entities = ("oak_log", "oak_planks", "bread", "hunger")

    def phases(self):
        return 1

    def setup_phase(self, w, ex, phase):
        w.__init__()
        w.pos = {"x": 0.0, "y": 64.0, "z": 0.0}
        ex.hunger = 6                    # 驱动压力：饥饿高
        w.add_item("bread", 2)
        for i in range(3):
            for j in range(3):
                w.put_block(3 + i - 1, 65 + j, 0, "oak_log")
        w.put_block(3, 65, 0, "oak_log")
        w._refresh_near()
        self._feed_done = False

    def env_drift(self, ex, step):
        # 环境改变（§11）：第 4 个决策后饥饿自动恢复到 19（饱食）——
        # 最优行为从"先吃"切换为"专注任务"
        if step == 4:
            ex.hunger = 19
            return "You feel full. (hunger restored to 19)"
        ex.hunger = max(0, ex.hunger - 1)
        return ""

    def done(self, w):
        return w.inv_map().get("oak_planks", 0) >= 1

    def max_decisions(self, phase):
        return 10


TASKS = {"T1": T1Reuse(), "T2": T2Distractor(), "T3": T3Reroute(), "T4": T4Competing()}
GOAL_VOCAB.update({"oak", "log", "planks", "birch", "bread", "hunger", "coal",
                   "iron", "stone", "craft", "table", "stick", "apple", "wood",
                   "furnace", "raw_iron", "wheat", "food", "ore"})
# C34：实体白名单 = 世界/配方已知名词（与 base prompt 中 facts 同源，
# 四条件同等可得，非特权信息）。生产感知将观察结构化映射到已知实体
# 节点；runner 初版用正则全量吸纳英文词（实体:near 等）属于 harness
# 缺陷，违反 FAS 感知设计——修复后重跑全部 D 族实验（见 experiment_audit）。
GENERIC_TOKENS = {"near", "inventory", "none", "ok", "fail", "got", "dug",
                  "no", "drop", "crafted", "placed", "ate", "waited", "pos",
                  "missing", "needs", "tool", "not", "in", "the", "and",
                  "unknown", "action", "result", "obs"}

CONDITIONS = ("llm_direct", "llm_history", "llm_rag", "fas_full", "random_ctx",
              "D-noact", "D-nodemand", "D-flat")
HISTORY_K = 8
RAG_K = 5


# ── 条件上下文构建（唯一自变量）─────────────────────────────────
def build_context(cond, task, store, fas, obs_line, rng):
    if cond == "llm_direct":
        return "No additional memory or context."
    if cond == "llm_history":
        h = store.history_block(HISTORY_K)
        return "Recent raw history (oldest->newest):\n" + (h or "(empty)")
    if cond == "llm_rag":
        q = task.goal_text + " " + obs_line
        return "Retrieved relevant memories (top-%d by similarity):\n" % RAG_K + \
            store.rag_block(q, RAG_K)
    if cond == "fas_full":
        return fas.context(task.goal_text)
    if cond == "random_ctx":
        n = 8
        return "Memory entries (sample):\n" + store.random_block(n, rng)
    if cond == "D-noact":
        fas.use_diffusion = False
        c = fas.context(task.goal_text)
        fas.use_diffusion = True
        return c
    if cond == "D-nodemand":
        fas.use_demand = False
        c = fas.context(task.goal_text)
        fas.use_demand = True
        return c
    if cond == "D-flat":
        # 扁平：图节点文本的相似度排序（无传播/无结构）——近似 RAG-over-graph
        texts = [str(nid) for nid in fas.kg.nodes]
        if not texts:
            return "(graph empty)"
        emb = MemoryStore()._embedder()
        import numpy as np
        M = emb.encode(texts)
        q = emb.encode_single(task.goal_text)
        sims = M @ (q / (np.linalg.norm(q) + 1e-9))
        idx = sorted(range(len(texts)), key=lambda i: -float(sims[i]))[:8]
        return "Graph nodes by text similarity (no propagation):\n" + \
            "\n".join("- " + texts[i] for i in idx)
    raise ValueError(cond)


BASE_PROMPT = """You are the decision module of an agent in a Minecraft-like world.
Task: {goal}
Facts: {facts}
State: {state}
{context}
Available actions (choose exactly one by number):
{menu}
Respond ONLY with JSON: {{"action": <number>, "why": "<=8 words"}}"""


def relevance(tags, text):
    return any(t in text for t in tags)


def run_task(cond, task_key, seed, llm, log):
    task = TASKS[task_key]
    rng = random.Random(seed * 1000 + zlib.crc32(task_key.encode()) % 997)
    store = MemoryStore()
    fas = FASContext() if cond.startswith(("fas", "D-")) else None
    if cond == "D-noact":
        fas.use_diffusion = False
    if cond == "D-nodemand":
        fas.use_demand = False
    w = SandboxWorld()
    ex = Executor(w)
    trace = []
    tok0 = (llm.prompt_tokens, llm.completion_tokens, llm.calls)
    first_acts = []
    for phase in range(1, task.phases() + 1):
        task.setup_phase(w, ex, phase)
        done = False
        change_step = None
        for step in range(task.max_decisions(phase)):
            if task_key == "T4":
                drift = task.env_drift(ex, step)
                if drift and "full" in drift:
                    change_step = step
            near = ", ".join(f"{b['name']}x{b['count']}" for b in w.w.near_names) \
                if hasattr(w, "w") else ", ".join(f"{b['name']}x{b['count']}" for b in w.near_names)
            inv = json.dumps(w.inv_map(), ensure_ascii=False)
            obs_line = f"near: {near or '(none)'}; inventory: {inv}; hunger: {ex.hunger}"
            menu = ex.menu(task)
            menu_text = "\n".join(f"{i}. {a}" for i, (a, _) in enumerate(menu))
            ctx = build_context(cond, task, store, fas, obs_line, rng)
            prompt = BASE_PROMPT.format(goal=task.phase_goal(phase), facts=task.phase_facts(phase),
                                        state=obs_line, context=ctx,
                                        menu=menu_text)
            reply = llm.decide(prompt)
            act, obj, ok, detail = None, None, True, "unparseable_reply"
            m = re.search(r'"action"\s*:\s*(\d+)', reply)
            if m and int(m.group(1)) < len(menu):
                act, obj = menu[int(m.group(1))]
            else:
                low = reply.lower()
                for a, o in menu:                      # 动作名回退匹配
                    key = a.split("_", 1)[-1] if a.startswith("gather_") else a
                    if a.lower() in low or key in low:
                        act, obj = a, o
                        break
            if act is None:
                act, obj = "wait", None
            else:
                ok, detail = ex.execute(act, obj)
            if task_key == "T4" and act.startswith("eat_") and ex.hunger >= 18:
                detail += "|wasted_eat"
            entry = f"obs: {obs_line} | action: {act} | result: {'ok' if ok else 'fail:' + str(detail)}"
            store.add(entry)
            if fas is not None:
                fas.ingest(obs_line, act, str(detail), task.goal_entities)
                fas.step_dynamics()
            trace.append({
                "task": task_key, "phase": phase, "step": step, "seed": seed,
                "condition": cond, "obs": obs_line, "menu": [a for a, _ in menu],
                "context": ctx, "reply": reply[:120], "action": act,
                "ok": ok, "detail": str(detail)[:60],
                "change_step": change_step,
            })
            # decisions = len(trace)
            first_acts.append(act)
            if task.done(w):
                done = True
                break
        if done and task_key in ("T1", "T3") and phase == 1:
            continue                          # T1/T3：phase1 达成后进 phase2
        if done:
            break
    # ---- 任务级指标 ----
    goal_tags = task.goal_entities
    acts = [t["action"] for t in trace]
    goal_acts = [a for a in acts if any(g.replace("_", "") in a.replace("_", "") or g in a for g in goal_tags)]
    distractor = [a for a in acts if a not in goal_acts and a != "wait"]
    ctx_sizes = [len(t["context"]) for t in trace]
    m = {
        "task": task_key, "condition": cond, "seed": seed,
        "success": int(done), "decisions": len(acts),
        "target_utilization": round(len(goal_acts) / max(len(acts), 1), 3),
        "distractor_rate": round(len(distractor) / max(len(acts), 1), 3),
        "first_action": acts[0] if acts else None,
        "acts": acts,
        "obsolete_actions": sum(1 for t in trace if str(t["detail"]).startswith("not_found")),
        "wasted_eats": sum(1 for t in trace if "wasted_eat" in str(t["detail"])),
        "prompt_tokens": llm.prompt_tokens - tok0[0],
        "completion_tokens": llm.completion_tokens - tok0[1],
        "llm_calls": llm.calls - tok0[2],
        "latency_s": round(llm.latency, 1),
        "ctx_chars_mean": round(sum(ctx_sizes) / max(len(ctx_sizes), 1), 1),
    }
    # T3：重路由时延
    if task_key == "T3":
        p2 = [t for t in trace if t["phase"] == 2]
        rr = None
        for i, t in enumerate(p2):
            if "birch" in t["action"] and t["ok"]:
                rr = i
                break
        m["t_reroute"] = rr
        m["obsolete_actions"] = sum(1 for t in p2 if "oak" in t["action"])
    # 检索精度（C/D/E：上下文条目中相关占比；程序化标注）
    if trace and trace[0]["condition"] in ("llm_rag", "fas_full", "random_ctx",
                                           "D-noact", "D-nodemand", "D-flat"):
        rel, tot = 0, 0
        for t in trace:
            for line in t["context"].splitlines():
                if line.strip().startswith(("-", "Cognitive", "Resource")):
                    tot += 1
                    if relevance(goal_tags, line.lower()):
                        rel += 1
        m["retrieval_precision"] = round(rel / max(tot, 1), 3)
        m["retrieval_entries"] = tot
    # 效率
    m["efficiency"] = round(m["success"] / max(m["prompt_tokens"] / 1000.0, 0.001), 4)
    log.write(json.dumps({"type": "task_result", **m}, ensure_ascii=False) + "\n")
    for t in trace:
        log.write(json.dumps({"type": "decision", **t}, ensure_ascii=False) + "\n")
    log.flush()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="llm_direct,llm_history,llm_rag,fas_full,random_ctx")
    ap.add_argument("--tasks", default="T1,T2,T3,T4")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--out", default="experiments/core_routing")
    args = ap.parse_args()
    conds = args.conditions.split(",")
    tasks = args.tasks.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    logpath = os.path.join(args.out, "raw_results.jsonl")
    summary = os.path.join(args.out, "summary.csv")
    log = open(logpath, "a", encoding="utf-8")
    rows = []
    for seed in seeds:
        for cond in conds:
            llm = LLMHead()
            for tk in tasks:
                try:
                    m = run_task(cond, tk, seed, llm, log)
                except Exception as e:
                    m = {"task": tk, "condition": cond, "seed": seed,
                         "success": 0, "error": repr(e)[:200]}
                    log.write(json.dumps({"type": "task_error", **m}) + "\n")
                    log.flush()
                rows.append(m)
                print(f"[routing] {cond} {tk} seed{seed}: "
                      f"succ={m.get('success')} dec={m.get('decisions')} "
                      f"dist={m.get('distractor_rate')}", flush=True)
    # summary.csv（追加）
    header = "condition,task,seed,success,decisions,target_utilization,distractor_rate,t_reroute,obsolete_actions,wasted_eats,retrieval_precision,prompt_tokens,completion_tokens,efficiency\n"
    new = not os.path.exists(summary)
    with open(summary, "a", encoding="utf-8") as f:
        if new:
            f.write(header)
        for m in rows:
            f.write(",".join(str(m.get(k, "")) for k in
                             ("condition", "task", "seed", "success", "decisions",
                              "target_utilization", "distractor_rate", "t_reroute",
                              "obsolete_actions", "wasted_eats", "retrieval_precision",
                              "prompt_tokens", "completion_tokens", "efficiency")) + "\n")
    print(f"done: {len(rows)} task-runs -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
