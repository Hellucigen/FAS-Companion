# run_exp_routing_v2.py — core_routing 实验装配修复版（R1/R2/R3）
# ═══════════════════════════════════════════════════════════════════════
# 只修**实验装配/接口/序列化/校验**，不动 FAS 核心机制：
#   · 不改 diffusion_engine 扩散算法 / activation 公式 / 边权规则 / 抑制竞争
#   · 不改 cognitive_demand / gap / resource routing 核心算法
#   · 不调参（阈值/系数/top-k/reward/任务难度）、不改 prompt、不改任务与统计
#
# v1（run_exp_routing.py）三个装配缺陷（见 activation_audit/）：
#   R1 实验图 0 边：v1 只 append edge_specs 从不 add_edge；而生产 API
#      activate_from_inputs 的语义是"仅激活**已有**边"（不创建边）。
#      → 本次修复：用生产建图路径（mc_knowledge.ensure_mc_world +
#        world_prior.build_recipe_closure，由世界表/配方表驱动，非任务硬编码）
#        + 观察共现边（由本步 observation/action/result 推导）真建边。
#   R2 demand 层零输出：v1 写 `out[0],out[1],out[2]`，而
#      analyze_cognitive_demand 返回 dict → KeyError: 0 → 被 except 吞成占位。
#      → 本次修复：按真实 schema 取键；核心模块异常一律**显式失败**（fail fast）。
#   R3 step-0 焦点恒空（80/80）：v1 先建 context 后 ingest。
#      → 本次修复：每步顺序改为 obs→ingest→dynamics→context→decision→act。
#
# 统一序列化：D / D-noact / D-nodemand / D-flat 四条件共用同一 schema
#   （serialize_context），差别只在被研究的机制开关，不产生不同格式。
# 校验：preflight + 逐决策 telemetry + 硬 invariant（违反即 FAIL FAST）。
# ═══════════════════════════════════════════════════════════════════════

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

# 离线模式（C35 reproducibility fix）：嵌入模型已在本地 HF 缓存，
# 运行中访问 HF Hub 会因网络抖动崩溃（v2 smoke D-flat 5 run 失败根因）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sandbox_lab import SandboxWorld, RECIPE_TABLE, BLOCK_META, \
    install_fake_bridge  # noqa

GOAL_VOCAB = set()
GENERIC_TOKENS = {"near", "inventory", "none", "ok", "fail", "got", "dug",
                  "no", "drop", "crafted", "placed", "ate", "waited", "pos",
                  "missing", "needs", "tool", "not", "in", "the", "and",
                  "unknown", "action", "result", "obs"}


# ── 核心模块异常：不吞、显式失败（R2 修复要求）────────────────────
class CoreModuleError(RuntimeError):
    """graph/diffusion/demand/gap/routing/context 任一核心环节失败 → 实验失败。"""


class InfrastructureError(RuntimeError):
    """纯基础设施失败（网络/模型不可用/进程级故障）。单独立类，
    不与 harness 装配失败混淆（正式规范 §16）。"""


class InvariantViolation(RuntimeError):
    """装配 invariant 被破坏 → FAIL FAST，不产出无效数据。"""


class LLMHead:
    """决策头（与 v1 完全一致：同模型、同 temperature、同 max_tokens、同重试）。"""

    def __init__(self):
        import openai
        sec = json.load(open("data/llm_secret.json", encoding="utf-8"))
        from llm_provider import MiMoBackend
        self.model = MiMoBackend._DEFAULT_MODEL
        self.client = openai.OpenAI(api_key=sec["api_key"],
                                    base_url=MiMoBackend._DEFAULT_BASE_URL)
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
            except Exception as e:
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise InfrastructureError(f"LLM decision head failed: {last!r}")


class Executor:
    """沙箱技能原语（与 v1 一致，未修改）。"""

    def __init__(self, world):
        self.w = world
        self.hunger = 20

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
            step = {"north": (0, -4), "south": (0, 4), "east": (4, 0),
                    "west": (-4, 0)}[d]
            w.pos["x"] += step[0]
            w.pos["z"] += step[1]
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


# ── 经历存储（baseline 条件用；v1 原样）──────────────────────────
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


BASELINE_CONDS = ("llm_direct", "llm_history", "llm_rag", "random_ctx")
HISTORY_K = 8
RAG_K = 5


def baseline_context(cond, store, obs_line, goal_text, rng):
    """v1 原样移植的 baseline 上下文构造（逻辑零改动）。"""
    if cond == "llm_direct":
        return "No additional memory or context."
    if cond == "llm_history":
        h = store.history_block(HISTORY_K)
        return "Recent raw history (oldest->newest):\n" + (h or "(empty)")
    if cond == "llm_rag":
        q = goal_text + " " + obs_line
        return ("Retrieved relevant memories (top-%d by similarity):\n" % RAG_K
                + store.rag_block(q, RAG_K))
    if cond == "random_ctx":
        return "Memory entries (sample):\n" + store.random_block(8, rng)
    raise ValueError(cond)


# ── FAS 条件：真实图 + 真实机制（R1/R2/R3 修复的核心）──────────────
class FASContext:
    """真 KnowledgeGraph + DiffusionEngine + analyze_cognitive_demand。

    与 v1 的差别（全部为装配层）：
      · 图由**生产建图路径**播种（world knowledge + recipe closure）→ 有边
      · ingest 用 `kg.add_edge` 真建观察共现边（v1 只 append spec）
      · name_to_node 在每次图变更后刷新（v1 只在构造时设一次）
      · context 按 analyze_cognitive_demand 的**真实 dict schema** 取键
      · 核心环节异常一律抛出（不吞）
    """

    def __init__(self, world, use_diffusion=True, use_demand=True,
                 use_flat=False):
        from graph_model import KnowledgeGraph
        from diffusion_engine import DiffusionEngine
        import mc_knowledge as mck
        import world_prior as wp
        import config as _C
        import minecraft.bridge as BR

        self.use_diffusion = use_diffusion
        self.use_demand = use_demand
        self.use_flat = use_flat
        self.core_errors = []

        cfgd = dict(_C.DEFAULT_CONFIG)
        self.world = world
        cfgd["mc_world"] = {
            "gatherable": sorted(set(world.recipe_table) |
                                 {d for m in world.block_meta.values()
                                  for d in m.get("drops", [])} |
                                 set(world.block_meta)),
            "block_meta": {r: dict(v) for r, v in world.block_meta.items()},
        }
        wp.bind_config(cfgd)
        self.kg = KnowledgeGraph()
        self.eng = DiffusionEngine(self.kg, {"beta_spread": 1.0,
                                             "activation_epsilon": 1e-4,
                                             "min_spread_threshold": 0.01,
                                             "activation_max": 5.0})
        # ── 生产建图路径（世界表驱动，非任务硬编码）──
        try:
            mck.ensure_mc_world(self.kg, cfgd)
            for tgt in sorted(set(world.recipe_table)):
                wp.build_recipe_closure(self.kg, self.eng, tgt, bridge=BR)
        except Exception as e:
            raise CoreModuleError(f"graph seeding failed: {e!r}") from e
        self._sync_index()

    # ── 图维护 ────────────────────────────────────────────────
    def _sync_index(self):
        """图变更后同步引擎名册（v1 遗漏点：建图后未刷新导致注入打空）。"""
        self.eng.name_to_node = dict(self.kg.nodes)

    def _node(self, nid, label="declarative-semantic"):
        from graph_model import Node
        if nid not in self.kg.nodes:
            self.kg.add_node(Node(id=nid, weight=0.45, label=label,
                                  graph_space="semantic"))
        return nid

    def _edge(self, src, dst, rel="关联", w=0.5):
        """用真实 kg.add_edge 建边（add_edge 端点缺失会静默丢，故先保节点）。"""
        from graph_model import Edge
        self._node(src)
        self._node(dst)
        if self.kg.get_edge(src, dst, rel) is None:
            self.kg.add_edge(Edge(src=src, dst=dst, relation=rel, weight=w))

    def entities_of(self, text):
        return sorted({t for t in re.findall(r"[a-zA-Z_]+", str(text))
                       if t not in GENERIC_TOKENS and t in GOAL_VOCAB})

    def node_id(self, ent):
        """实体 → 图中首选节点 id（配方/掉落图里的 物品:X 若存在则用之）。"""
        for cand in (f"物品:{ent}", ent):
            if cand in self.kg.nodes:
                return cand
        return f"实体:{ent}"

    # ── 经验入图（R1 修复点：真建边）──────────────────────────
    def ingest(self, obs_text, action, result, goal_entities):
        try:
            ents = [self.node_id(e) for e in self.entities_of(obs_text)]
            for g in goal_entities:
                nid = self.node_id(g)
                if nid not in ents:
                    ents.append(nid)
            an = self._node("动作:" + str(action), label="procedural") \
                if action else None
            rn = self._node("结果:" + str(result)[:24]) if result else None
            # 实体 ↔ 实体（同一步共现）
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    self._edge(ents[i], ents[j])
            # 实体 ↔ 动作、动作 ↔ 结果（有向，供 forward 传播）
            if an:
                for nid in ents:
                    self._edge(an, nid)
                    self._edge(nid, an)
            if an and rn:
                self._edge(an, rn)
            self._sync_index()
            # 注入本步观察到的实体（种子激活）
            seeds = [n for n in ents if n in self.eng.name_to_node]
            if seeds:
                self.eng.activate_from_inputs(seeds, [],
                                              source_type="external_input")
        except CoreModuleError:
            raise
        except Exception as e:
            raise CoreModuleError(f"graph ingest failed: {e!r}") from e

    def step_dynamics(self):
        try:
            self.eng.decay_step()
            if self.use_diffusion and not self.use_flat:
                self.eng.diffuse_step()
        except Exception as e:
            raise CoreModuleError(f"diffusion step failed: {e!r}") from e

    # ── 焦点与 demand（R2 修复点：真实 schema + 不吞异常）────
    def focus(self, k=8):
        try:
            top = self.eng.get_topk(k=k)[0]
            return [{"id": n.id,
                     "activation": round(float(getattr(n, "activation", 0)
                                               or 0), 4)}
                    for n in top]
        except Exception as e:
            raise CoreModuleError(f"get_topk failed: {e!r}") from e

    _EMB = None   # 进程级单例（模型加载一次，C35）

    def flat_focus(self, task_text, k=8):
        """D-flat：同一张图上的扁平文本相似度排序（无激活/无传播），
        但**输出同一 schema**（activation 字段为 None）。"""
        try:
            from embedding_manager import EmbeddingProvider
            import numpy as np
            ids = sorted(self.kg.nodes)
            if not ids:
                return []
            if FASContext._EMB is None:
                FASContext._EMB = EmbeddingProvider()
            emb = FASContext._EMB
            M = emb.encode(ids)
            q = emb.encode_single(task_text)
            sims = M @ (q / (np.linalg.norm(q) + 1e-9))
            idx = sorted(range(len(ids)), key=lambda i: -float(sims[i]))[:k]
            return [{"id": ids[i], "activation": None} for i in idx]
        except Exception as e:
            raise CoreModuleError(f"flat retrieval failed: {e!r}") from e

    def demand_block(self, task_text):
        """返回 (demand, gap, routing, mode)。异常一律抛出（不吞）。"""
        import cognitive_demand as cd
        try:
            out = cd.analyze_cognitive_demand(
                text=task_text, parsed={}, kg=self.kg, engine=self.eng,
                exploration={}, should_speak=False)
        except Exception as e:
            raise CoreModuleError(
                f"analyze_cognitive_demand failed: {e!r}") from e
        if not isinstance(out, dict):
            raise CoreModuleError(
                f"analyze_cognitive_demand returned {type(out).__name__}, "
                "expected dict")
        for key in ("demand", "gap", "resource", "mode"):
            if key not in out:
                raise CoreModuleError(
                    f"demand schema missing key {key!r}; got {sorted(out)}")
        return out["demand"], out["gap"], out["resource"], out["mode"]

    def telemetry(self):
        act_nodes = sum(1 for nd in self.kg.nodes.values()
                        if float(getattr(nd, "activation", 0) or 0) > 0)
        act_edges = sum(1 for e in self.kg.edges
                        if float(getattr(e, "activation", 0) or 0) > 0)
        return {"graph_nodes": len(self.kg.nodes),
                "graph_edges": len(self.kg.edges),
                "activated_nodes": act_nodes,
                "activated_edges": act_edges}


# ── 统一序列化（四条件同一 schema）───────────────────────────────
CTX_SCHEMA_KEYS = ("condition", "mode", "selected_nodes", "selected_edges",
                   "activation_summary", "demand", "gap", "routing",
                   "memory_context")


def serialize_context(condition, mode, selected, demand=None, gap=None,
                      routing=None, memory_context=None, edges=None,
                      activation_summary=None):
    """所有 D 族条件共用的序列化。返回 (payload_dict, rendered_text)。

    四条件只在被研究的机制上有差异（mode 字段标明），**格式完全一致**。
    """
    payload = {
        "condition": condition,
        "mode": mode,                    # full | no-diffusion | no-demand | flat
        "selected_nodes": selected,      # [{id, activation|None}]
        "selected_edges": edges or [],
        "activation_summary": activation_summary or {},
        "demand": demand,                # dict | None（禁用时为 None 并标 mode）
        "gap": gap,
        "routing": routing,
        "memory_context": memory_context,
    }
    lines = [f"[cognitive-context mode={mode}]"]
    lines.append("selected nodes:")
    for n in selected:
        a = n.get("activation")
        lines.append(f"- {n['id']} (act {a})" if a is not None
                     else f"- {n['id']}")
    if payload["selected_edges"]:
        lines.append("selected edges:")
        for e in payload["selected_edges"][:8]:
            lines.append(f"- {e}")
    if demand is not None:
        dtop = sorted(demand.items(), key=lambda kv: -float(kv[1]))[:3]
        lines.append("demand (top): " + ", ".join(f"{k}={v}" for k, v in dtop))
    if gap is not None:
        gtop = sorted(((k, v) for k, v in gap.items()
                       if isinstance(v, (int, float))),
                      key=lambda kv: -float(kv[1]))[:2]
        lines.append("gap (top): " + ", ".join(f"{k}" for k, _ in gtop))
    if routing is not None:
        lines.append("routing: " + json.dumps(routing, ensure_ascii=False)[:160])
    if memory_context:
        lines.append("memory: " + memory_context)
    return payload, "\n".join(lines)


# ── 任务（与 v1 完全一致，未修改）────────────────────────────────
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
        w.__init__()
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
    name = "T3_environment_reroute"
    goal_text = "Obtain 1 oak_planks in inventory."
    facts = "Recipe: 1 oak_log -> 4 oak_planks (hand). Oak trees provide oak_log."
    gatherable = ("oak_log",)
    goal_entities = ("oak_log", "oak_planks", "birch_log", "birch_planks")

    def phases(self):
        return 2

    def phase_goal(self, phase):
        return self.goal_text if phase == 1 else \
            "Obtain 1 birch_planks in inventory. (The oak trees are gone.)"

    def phase_facts(self, phase):
        return self.facts if phase == 1 else \
            "Recipe: 1 birch_log -> 4 birch_planks (hand). Birch trees provide birch_log."

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
            for i in range(3):
                for j in range(3):
                    w.put_block(3 + i - 1, 65 + j, 0, "birch_log")
            w.put_block(3, 65, 0, "birch_log")
            w.block_meta["birch_log"] = {"drops": ["birch_log"], "harvest_tools": []}
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
        ex.hunger = 6
        w.add_item("bread", 2)
        for i in range(3):
            for j in range(3):
                w.put_block(3 + i - 1, 65 + j, 0, "oak_log")
        w.put_block(3, 65, 0, "oak_log")
        w._refresh_near()

    def env_drift(self, ex, step):
        if step == 4:
            ex.hunger = 19
            return "You feel full. (hunger restored to 19)"
        ex.hunger = max(0, ex.hunger - 1)
        return ""

    def done(self, w):
        return w.inv_map().get("oak_planks", 0) >= 1

    def max_decisions(self, phase):
        return 10


TASKS = {"T1": T1Reuse(), "T2": T2Distractor(), "T3": T3Reroute(),
         "T4": T4Competing()}
GOAL_VOCAB.update({"oak", "log", "planks", "birch", "bread", "hunger", "coal",
                   "iron", "stone", "craft", "table", "stick", "apple", "wood",
                   "furnace", "raw_iron", "wheat", "food", "ore"})

COND_MODE = {"fas_full": "full", "D-noact": "no-diffusion",
             "D-nodemand": "no-demand", "D-flat": "flat"}

BASE_PROMPT = """You are the decision module of an agent in a Minecraft-like world.
Task: {goal}
Facts: {facts}
State: {state}
{context}
Available actions (choose exactly one by number):
{menu}
Respond ONLY with JSON: {{"action": <number>, "why": "<=8 words"}}"""


# ── preflight（Step 6）───────────────────────────────────────────
def preflight(world):
    """最小装配自检：图有边 + 传播可达 + demand API 可用。失败即 FAIL FAST。"""
    rep = {}
    fc = FASContext(world)
    rep["graph_nodes"] = len(fc.kg.nodes)
    rep["graph_edges"] = len(fc.kg.edges)
    if rep["graph_edges"] <= 0:
        raise InvariantViolation(
            f"preflight: graph has {rep['graph_edges']} edges (must be > 0)")
    # 传播自证：注入一个有多跳结构的节点，跑扩散看是否点亮邻居
    seeds = [n for n in ("物品:oak_log", "oak_log") if n in fc.kg.nodes]
    if not seeds:
        raise InvariantViolation("preflight: no seed node for propagation probe")
    before = {n: float(getattr(nd, "activation", 0) or 0)
              for n, nd in fc.kg.nodes.items()}
    fc.eng.activate_from_inputs([seeds[0]], [],
                                source_type="external_input")
    for _ in range(4):
        fc.eng.diffuse_step()
    lit = [n for n in fc.kg.nodes
           if float(getattr(fc.kg.nodes[n], "activation", 0) or 0) > 0.001
           and before.get(n, 0.0) <= 0.001]
    rep["propagation_probe_seed"] = seeds[0]
    rep["propagation_newly_lit"] = len(lit)
    if not lit:
        raise InvariantViolation(
            "preflight: diffusion produced no newly lit node "
            "(传播不可达 → 装配无效)")
    # demand/gap/routing schema 自检
    d, g, r, mode = fc.demand_block("Obtain 1 oak_planks in inventory.")
    rep["demand_keys"] = sorted(d)[:4]
    rep["gap_type"] = type(g).__name__
    rep["routing_mode"] = str(mode)
    return rep


# ── 主循环（R3 修复：obs→ingest→dynamics→context→decision→act）───
def run_task(cond, task_key, seed, llm, log, fail_fast=True):
    task = TASKS[task_key]
    is_baseline = cond in BASELINE_CONDS
    mode = ("baseline-" + cond.split("_", 1)[1]) if is_baseline else COND_MODE[cond]
    rng = random.Random(seed * 1000 + zlib.crc32(task_key.encode()) % 997)
    world = SandboxWorld()
    ex = Executor(world)
    t_start = time.time()
    fc = None
    store = None
    if not is_baseline:
        fc = FASContext(world,
                        use_diffusion=(mode != "no-diffusion"),
                        use_demand=(mode != "no-demand"),
                        use_flat=(mode == "flat"))
    else:
        store = MemoryStore()
    trace = []
    tok0 = (llm.prompt_tokens, llm.completion_tokens, llm.calls)
    core_exceptions = 0
    done = False
    for phase in range(1, task.phases() + 1):
        task.setup_phase(world, ex, phase)
        for step in range(task.max_decisions(phase)):
            if task_key == "T4":
                task.env_drift(ex, step)
            # 1) observation
            near = ", ".join(f"{b['name']}x{b['count']}"
                             for b in world.near_names)
            inv = json.dumps(world.inv_map(), ensure_ascii=False)
            obs_line = f"near: {near or '(none)'}; inventory: {inv}; hunger: {ex.hunger}"
            # 2-4) ingest → dynamics → context（baseline 分支无图机制）
            if is_baseline:
                store.add(f"obs: {obs_line} | action: "
                          f"{trace[-1]['action'] if trace else '(start)'} | "
                          f"result: {trace[-1]['detail'] if trace else 'init'}")
                ctx_text = baseline_context(cond, store, obs_line,
                                            task.phase_goal(phase), rng)
                tel = {"graph_nodes": None, "graph_edges": None,
                       "activated_nodes": None, "activated_edges": None}
                selected, demand, gap, routing, edges = [], None, None, None, []
            else:
                fc.ingest(obs_line, trace[-1]["action"] if trace else None,
                          trace[-1]["detail"] if trace else "init",
                          task.goal_entities)
                fc.step_dynamics()
                tel = fc.telemetry()
                if mode == "flat":
                    selected = fc.flat_focus(task.phase_goal(phase), k=8)
                    edges = []
                else:
                    selected = fc.focus(k=8)
                    ids = {n["id"] for n in selected}
                    edges = [f"{e.src} -{e.relation}-> {e.dst}"
                             for e in fc.kg.edges
                             if e.src in ids and e.dst in ids][:8]
                if mode == "no-demand":
                    demand = gap = routing = None
                else:
                    demand, gap, routing, _m = fc.demand_block(
                        task.phase_goal(phase))
                payload, ctx_text = serialize_context(
                    cond, mode, selected, demand, gap, routing,
                    memory_context=None, edges=edges,
                    activation_summary={"n_selected": len(selected), **tel})
            if not selected and not is_baseline:
                # 空焦点 invariant 只约束 D 族（baseline 记忆为空是 v1 合法状态）
                raise InvariantViolation(
                    f"{cond} {task_key} phase{phase} step{step}: "
                    "empty context (invariant: context non-empty)")
            menu = ex.menu(task)
            prompt = BASE_PROMPT.format(goal=task.phase_goal(phase),
                                        facts=task.phase_facts(phase),
                                        state=obs_line, context=ctx_text,
                                        menu="\n".join(
                                            f"{i}. {a}" for i, (a, _) in
                                            enumerate(menu)))
            # 5) LLM decision
            reply = llm.decide(prompt)
            idx = -1
            m = re.search(r'"action"\s*:\s*(\d+)', reply)
            if m and int(m.group(1)) < len(menu):
                idx = int(m.group(1))
            if idx < 0:
                low = reply.lower()
                for i, (a, _) in enumerate(menu):
                    key = a.split("_", 1)[-1] if a.startswith("gather_") else a
                    if a.lower() in low or key in low:
                        idx = i
                        break
            if idx < 0:
                act, obj, ok, detail = "wait", None, True, "unparseable_reply"
            else:
                act, obj = menu[idx]
                ok, detail = ex.execute(act, obj)
            # 6) telemetry（逐决策）
            trace.append({
                "task": task_key, "phase": phase, "step": step, "seed": seed,
                "condition": cond, "mode": mode, "obs": obs_line,
                "action": act, "ok": ok, "detail": str(detail)[:60],
                "selected_nodes": [n["id"] for n in selected],
                "selected_acts": [n["activation"] for n in selected],
                "graph_nodes": tel["graph_nodes"],
                "graph_edges": tel["graph_edges"],
                "activated_nodes": tel["activated_nodes"],
                "activated_edges": tel["activated_edges"],
                "demand": demand, "gap": (gap if isinstance(gap, dict) else None),
                "routing": routing,
                "exception_count": core_exceptions,
                "llm_prompt_tokens_step": None,
                "harness_valid": True,
            })
            if task.done(world):
                done = True
                break
        if done:
            break
    acts = [t["action"] for t in trace]
    goal_tags = task.goal_entities
    goal_acts = [a for a in acts
                 if any(g.replace("_", "") in a.replace("_", "")
                        or g in a for g in goal_tags)]
    distractor = [a for a in acts if a not in goal_acts and a != "wait"]
    ctx_sizes = [len(json.dumps(t["selected_nodes"])) for t in trace]
    result = {
        "task": task_key, "condition": cond, "mode": mode, "seed": seed,
        "success": int(done), "decisions": len(acts),
        "target_utilization": round(len(goal_acts) / max(len(acts), 1), 3),
        "distractor_rate": round(len(distractor) / max(len(acts), 1), 3),
        "first_action": acts[0] if acts else None, "acts": acts,
        "prompt_tokens": llm.prompt_tokens - tok0[0],
        "completion_tokens": llm.completion_tokens - tok0[1],
        "llm_calls": llm.calls - tok0[2],
        "ctx_chars_mean": round(sum(ctx_sizes) / max(len(ctx_sizes), 1), 1),
        "latency_s": round(time.time() - t_start, 1),
        "harness_valid": True,
        "failure_class": "none",
        "failure_reason": "",
        "core_exceptions": core_exceptions,
        "step0_n_nodes": len(trace[0]["selected_nodes"]) if trace else 0,
        "graph_edges_final": (trace[-1]["graph_edges"] if trace and trace[-1]["graph_edges"] is not None else None),
        "activated_edges_mean": (
            round(sum(t["activated_edges"] for t in trace
                      if t["activated_edges"] is not None)
                  / max(sum(1 for t in trace
                            if t["activated_edges"] is not None), 1), 2)
            if any(t["activated_edges"] is not None for t in trace) else None),
    }
    # ── invariant 校验（FAIL FAST）──
    if fail_fast:
        reason = None
        if (result["graph_edges_final"] or 0) <= 0 and not is_baseline:
            reason = "graph_edges == 0 at end"
        elif result["core_exceptions"] != 0:
            reason = f"core_exceptions={result['core_exceptions']}"
        elif (result["step0_n_nodes"] or 0) <= 0 and not cond.startswith("llm") \
                and cond != "random_ctx":
            reason = "step-0 context empty"
        if reason:
            result["harness_valid"] = False
            result["failure_class"] = "INVALID_HARNESS_RUN"
            result["failure_reason"] = reason
            log.write(json.dumps({"type": "task_result", **result},
                                 ensure_ascii=False) + "\n")
            log.flush()
            raise InvariantViolation(f"{cond} {task_key} seed{seed}: {reason}")
    log.write(json.dumps({"type": "task_result", **result},
                         ensure_ascii=False) + "\n")
    for t in trace:
        log.write(json.dumps({"type": "decision", **t},
                             ensure_ascii=False) + "\n")
    log.flush()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions",
                    default="fas_full,D-noact,D-nodemand,D-flat")
    ap.add_argument("--tasks", default="T1,T2,T3,T4")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="experiments/core_routing_v2_smoke")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    conds = args.conditions.split(",")
    tasks = args.tasks.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    # ── preflight（无论是否 flag，装配自检必须通过才允许跑）──
    w0 = SandboxWorld()
    rep = preflight(w0)
    print("[preflight] " + json.dumps(rep, ensure_ascii=False), flush=True)
    with open(os.path.join(args.out, "preflight.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    log = open(os.path.join(args.out, "raw_results.jsonl"), "a",
               encoding="utf-8")
    rows = []
    for seed in seeds:
        for cond in conds:
            llm = LLMHead()
            for tk in tasks:
                try:
                    m = run_task(cond, tk, seed, llm, log)
                except InvariantViolation as e:
                    m = {"condition": cond, "task": tk, "seed": seed,
                         "success": 0, "harness_valid": False,
                         "failure_class": "INVALID_HARNESS_RUN",
                         "failure_reason": str(e)[:200]}
                    log.write(json.dumps({"type": "task_result", **m},
                                         ensure_ascii=False)  + "\n")
                    log.flush()
                except InfrastructureError as e:
                    m = {"condition": cond, "task": tk, "seed": seed,
                         "success": 0, "harness_valid": True,
                         "failure_class": "INFRASTRUCTURE_FAILURE",
                         "failure_reason": str(e)[:200]}
                    log.write(json.dumps({"type": "task_result", **m},
                                         ensure_ascii=False)  + "\n")
                    log.flush()
                except CoreModuleError as e:
                    m = {"condition": cond, "task": tk, "seed": seed,
                         "success": 0, "harness_valid": False,
                         "failure_class": "INVALID_HARNESS_RUN",
                         "failure_reason": str(e)[:200]}
                    log.write(json.dumps({"type": "task_result", **m},
                                         ensure_ascii=False)  + "\n")
                    log.flush()
                rows.append(m)
                print(f"[v2] {cond} {tk} seed{seed}: succ={m['success']} "
                      f"dec={m['decisions']} edges={m['graph_edges_final']} "
                      f"aedges={m['activated_edges_mean']} "
                      f"step0={m['step0_n_nodes']} exc={m['core_exceptions']}",
                      flush=True)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        import csv as _csv
        wtr = _csv.DictWriter(f, fieldnames=[
            "condition", "mode", "task", "seed", "success", "decisions",
            "target_utilization", "distractor_rate", "first_action",
            "prompt_tokens", "completion_tokens", "llm_calls",
            "ctx_chars_mean", "core_exceptions", "step0_n_nodes",
            "graph_edges_final", "activated_edges_mean"])
        wtr.writeheader()
        for m in rows:
            wtr.writerow({k: m.get(k) for k in wtr.fieldnames})
    print(f"done: {len(rows)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
