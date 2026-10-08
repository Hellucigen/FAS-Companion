# test_screen_observe_autonomy.py — 自主链上的 SCREEN_OBSERVE（Stage 3 验收）
# 链路：CuriosityDrive 激活 →-[驱动]-> 看屏幕（config 种子边）
#       → capability_graph.discover_candidates 产 spec（executor=screen_observe
#         ∈ router.capabilities，MC 离线也不拦）
#       → ActionManager.propose → EmbodimentRouter → ScreenObserver 执行入图
# 离线：假 capture、零 LLM、不连 MC 桥。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_screen_observe_autonomy.py

import logging
import os
import shutil
import sys
import tempfile

_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _r)
# split_repos bootstrap: FAS-Cognitive sibling
_cog = os.environ.get("FAS_COG_ROOT") or os.path.join(
    os.path.dirname(_r), "FAS-Cognitive")
if os.path.isdir(_cog) and _cog not in sys.path:
    sys.path.insert(0, _cog)
logging.disable(logging.WARNING)

from graph_model import KnowledgeGraph, Node
from diffusion_engine import DiffusionEngine
from action_system import ActionManager
from capability_graph import CapabilityIndex
from action_concepts import ensure_action_concepts, EXECUTOR_TO_CONCEPT
import config as _C
from eye.observer import ScreenObserver, EmbodimentRouter

FAILURES = []


def check(name, cond, detail=""):
    st = "PASS" if cond else "FAIL"
    print(f"[{st}] {name}" + (f" | {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


CFG = {"lambda_decay": 0.05, "beta_spread": 1.0, "max_depth": 4,
       "theta_threshold": 0.01, "activation_max": 5.0,
       "min_spread_threshold": 0.01, "activation_epsilon": 1e-4,
       "input_similarity_floor": 0.5, "input_default_bonus": 0.5,
       "theta_action": 0.5}

base = os.path.join(tempfile.gettempdir(), "fas_test_screen_auto")
shutil.rmtree(base, ignore_errors=True)
os.makedirs(base, exist_ok=True)

kg = KnowledgeGraph()
kg.add_node(Node(id="Haru", graph_space="self"))
kg.add_node(Node(id="CuriosityDrive", activation=3.0,
                 graph_space="cognitive"))
eng = DiffusionEngine(kg, dict(CFG))
eng.name_to_node = dict(kg.nodes)

n_concepts = ensure_action_concepts(kg, eng)
cfg_full = dict(_C.DEFAULT_CONFIG)
cap_index = CapabilityIndex(kg, eng, cfg_full)


class OfflineMC:
    """MC 桥离线：capabilities 空集（真实具身的连接态真相）。"""

    name = "mc_offline"

    def available(self):
        return False

    def capabilities(self):
        return set()

    def execute(self, action):
        raise AssertionError("离线动作不应路由到 MC")

    def poll_action(self):
        return {"status": "done"}

    def cancel(self):
        return {"ok": True}

    def perceive(self):
        return {}

    def raw_state(self):
        return {}


def fake_capture(region=None):
    items = [{"text": "Haru 你在看什么", "score": 0.9, "center": (200, 300),
              "box": [[200, 300], [300, 300], [300, 312], [200, 312]]}]
    return {"items": items, "count": 1, "full_text": "Haru 你在看什么",
            "region": "fake"}


observer = ScreenObserver(kg, engine=eng, embedder_fn=lambda: None)
observer.capture = fake_capture
mc = OfflineMC()
cap_index.embodiment = EmbodimentRouter(default=mc,
                                        routes={"screen_observe": observer})
cap_index.ensure_graph()

# ── 1. 执行接口升级：executor 脱离 inline，进概念反查表 ────────
node = kg.get_node("看屏幕")
ea = node.extra_attrs or {}
check("1 图谱概念节点 executor 已去 inline（boot 自愈生效）",
      ea.get("executor") == "screen_observe"
      and ea.get("type") == "action_concept", str(ea.get("executor")))
check("1' EXECUTOR_TO_CONCEPT 反查收录 screen_observe",
      EXECUTOR_TO_CONCEPT.get("screen_observe") == "SCREEN_OBSERVE")
check("1'' 驱动边已种：CuriosityDrive-[驱动]->看屏幕",
      cap_index.kg.get_edge("CuriosityDrive", "看屏幕", "驱动") is not None)

# ── 2. 候选发现：好奇心点亮 → 屏幕动作进候选（MC 离线不拦）────
specs = cap_index.discover_candidates()
screen_specs = [s for s in specs
                if s.get("action_type") == "screen_observe"]
check("2 discover_candidates 产出 screen_observe spec（图谱驱动，非代码名单）",
      len(screen_specs) == 1, str([s.get("action_type") for s in specs]))
sp = screen_specs[0] if screen_specs else {}
check("2' spec 携带概念归属与诱因节点（可解释）",
      sp.get("concept") == "看屏幕" and "CuriosityDrive" in (sp.get("reason") or []),
      str(sp))

# ── 3. 提议→执行：ActionManager 收到 spec 后眼睛真的工作 ───────
am = ActionManager(embodiment=cap_index.embodiment, kg=kg, engine=eng,
                   config={}, data_dir=base)
out = am.propose(dict(sp), source="autonomy")
check("3 spec 直接可被 ActionManager 执行（autonomy 零改动口径）",
      out.get("started") and out.get("success"), str(out))
eyes = [nid for nid in kg.nodes if str(nid).startswith("eye_text_")]
check("3' 自主观察写入 eye_text_* 记忆（观察结果入认知）", len(eyes) == 1,
      str(eyes))
check("3'' '看屏幕'沿前沿点亮（激活直写点 mark_active 不变量）",
      kg.nodes["看屏幕"].activation >= 3.0 and "看屏幕" in eng._active_nodes)

# ── 4. 决策条件归既有机制：好奇心熄灭 → 候选消失（不设新开关）──
kg.nodes["CuriosityDrive"].activation = 0.0
kg.nodes["看屏幕"].activation = 0.0
specs2 = cap_index.discover_candidates()
check("4 诱因熄灭后不再产出候选（何时行动仍由激活/门槛裁决，无新定时器）",
      not [s for s in specs2 if s.get("action_type") == "screen_observe"],
      str([s.get("action_type") for s in specs2]))

print()
if FAILURES:
    print(f"✗ {len(FAILURES)} 项失败: {FAILURES}")
    sys.exit(1)
print("✓ 自主链 SCREEN_OBSERVE 测试全过（好奇心→图谱发现→真执行→入图）")
shutil.rmtree(base, ignore_errors=True)
