# test_screen_observe_action.py — SCREEN_OBSERVE 真执行链（Stage 2 验收）
# 链路：propose → ActionManager → EmbodimentRouter → ScreenObserver
#       → run_observation（复用 eye/screen_ocr 同一套函数）→ eye_text_* 入图
# 离线：假 capture（不截屏、不加载 RapidOCR）、零 LLM。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_screen_observe_action.py

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
from eye.observer import ScreenObserver, EmbodimentRouter, run_observation

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


def item(text, conf=0.9, cx=100, cy=100):
    return {"text": text, "score": conf, "center": (cx, cy),
            "box": [[cx, cy], [cx + 50, cy], [cx + 50, cy + 12], [cx, cy + 12]]}


def make_capture(items):
    """假截屏：绕开 mss/RapidOCR，只喂识别结果。"""
    def cap(region=None):
        return {"items": [dict(i) for i in items], "count": len(items),
                "full_text": "\n".join(i["text"] for i in items),
                "region": "fake"}
    return cap


class StubMinecraft:
    """默认子具身替身：可配置离线（capabilities 空）或带技能。"""

    def __init__(self, caps=("mine",), pending_next=False):
        self.caps = set(caps)
        self.pending_next = pending_next
        self.executed = []
        self.cancelled = 0
        self.polls = 0
        self.ctx = {"marker": "mc-only"}   # 供 __getattr__ 透传断言

    def available(self):
        return bool(self.caps)

    def capabilities(self):
        return set(self.caps)

    def execute(self, action):
        self.executed.append(action)
        if self.pending_next:
            return {"success": True, "pending": True, "action": "mine"}
        return {"success": True, "action": action.get("action_type"),
                "describe": "stub"}

    def poll_action(self):
        self.polls += 1
        return {"status": "done", "detail": {}}

    def cancel(self):
        self.cancelled += 1
        return {"ok": True}

    def perceive(self):
        return {}

    def raw_state(self):
        return {"health": 20}


base = os.path.join(tempfile.gettempdir(), "fas_test_screen_observe")
shutil.rmtree(base, ignore_errors=True)
os.makedirs(base, exist_ok=True)

kg = KnowledgeGraph()
kg.add_node(Node(id="Haru", graph_space="self"))
kg.add_node(Node(id="看屏幕", graph_space="self"))
kg.add_node(Node(id="Minecraft", weight=0.8, graph_space="semantic"))
kg.nodes["Minecraft"].activation = 1.8
eng = DiffusionEngine(kg, dict(CFG))
eng.name_to_node = dict(kg.nodes)

observer = ScreenObserver(kg, engine=eng, embedder_fn=lambda: None)
observer.capture = make_capture([
    item("Minecraft 正在运行中", 0.92, cy=100),
    item("asdkj qw zxcv 乱码噪声", 0.95, cy=200)])
mc = StubMinecraft(caps=set())          # MC 离线：屏幕动作必须照做
router = EmbodimentRouter(default=mc, routes={"screen_observe": observer})
am = ActionManager(embodiment=router, kg=kg, engine=eng, config={},
                   data_dir=base)

# ── 1. 全链：propose → router → observer → OCR → 图谱 ─────────
n_before = sum(1 for nid in kg.nodes if str(nid).startswith("eye_text_"))
out = am.propose({"action_type": "screen_observe", "motivation": "curiosity"},
                 source="autonomy")
check("1 ActionManager 接受并立即启动 screen_observe",
      out.get("started") and out.get("success"), str(out))
n_after = sum(1 for nid in kg.nodes if str(nid).startswith("eye_text_"))
check("1' 观察结果成为一条 eye_text_* 情景记忆", n_after == n_before + 1,
      f"{n_before}→{n_after}")
check("1'' 同步执行器不占用承诺期（结算完 busy=False）", not am.busy())
eye_id = next(nid for nid in kg.nodes if str(nid).startswith("eye_text_"))
texts = [t["text"] for t in kg.nodes[eye_id].extra_attrs["texts"]]
check("1''' 相关性主导：亮着的 Minecraft 排第一（junk 高分也不置顶）",
      texts and texts[0].startswith("Minecraft"), str(texts))
check("1''''' 结果回认知：describe/salient 非空",
      "注意到" in (out.get("describe") or ""), str(out))

# ── 2. 前沿不变量：激活直写点必须 mark_active ──────────────────
check("2 '看屏幕'激活被点亮且进入活跃前沿",
      kg.nodes["看屏幕"].activation >= 3.0 and "看屏幕" in eng._active_nodes,
      f"act={kg.nodes['看屏幕'].activation}")

# ── 3. 能力面：MC 离线时屏幕能力仍在（自主发现的真相源）───────
check("3 router.capabilities = MC∪屏幕；离线 MC 只剩屏幕",
      router.capabilities() == {"screen_observe"}
      and mc.capabilities() == set())
router2mc = EmbodimentRouter(default=StubMinecraft(caps=("mine",)),
                             routes={"screen_observe": observer})
check("3' 在线时能力取并集",
      router2mc.capabilities() == {"mine", "screen_observe"})

# ── 4. 路由正确性：非屏幕动作交给默认具身 ──────────────────────
r = router.execute({"action_type": "mine", "params": {"block": "dirt"}})
check("4 非 screen_observe 路由到默认具身执行",
      r.get("success") and mc.executed
      and mc.executed[-1]["action_type"] == "mine")

# ── 5. pending 生命周期：poll/cancel 跟着当前子执行器走 ────────
mc2 = StubMinecraft(caps=("mine",), pending_next=True)
router2 = EmbodimentRouter(default=mc2, routes={"screen_observe": observer})
router2.execute({"action_type": "mine"})
check("5 pending 动作记住归属子执行器", router2._busy is mc2)
router2.poll_action()
check("5' done 后释放归属", mc2.polls == 1 and router2._busy is None)
router2.cancel()
check("5'' 空闲时 cancel 透传默认具身", mc2.cancelled == 1)

# ── 6. 属性透传：ActionManager/autonomy 读到的其余接口零感知 ───
check("6 raw_state/perceive/未知属性透传默认具身",
      router.raw_state() == mc.raw_state()
      and router.perceive() == mc.perceive()
      and router.ctx == {"marker": "mc-only"})

# ── 7. 失败诚实：capture 抛异常 → success=False 带原因，不乱写记忆 ──
def bad_capture(region=None):
    raise RuntimeError("no display")
observer.capture = bad_capture
res = observer.execute({"action_type": "screen_observe"})
check("7 OCR 失败如实报告（不伪造观察成功）",
      res.get("success") is False and "ocr_failed" in res.get("reason", ""),
      str(res))

# ── 8. 空观察：看了但没值得记的 → 仍算执行成功，不建节点 ──────
kg3 = KnowledgeGraph()
kg3.add_node(Node(id="看屏幕"))
obs3 = ScreenObserver(kg3, embedder_fn=lambda: None)
# 第一次写入一条记忆，第二次整屏与之完全重复且图谱无关 → 显著性塌陷
obs3.capture = make_capture([item("aaa bbb ccc ddd eee", 0.9)])
run_first = run_observation(kg3, embedder=None, capture=obs3.capture)
node_after_first = [n for n in kg3.nodes if str(n).startswith("eye_text_")]
second = obs3.execute({"action_type": "screen_observe"})
node_after_second = [n for n in kg3.nodes if str(n).startswith("eye_text_")]
check("8 重复且无关的屏幕：第二次观察不再产生新记忆节点",
      second.get("success") and len(node_after_second) == len(node_after_first),
      str(node_after_second))
check("8' 空观察的 describe 如实（没什么值得注意），执行仍算成功",
      second.get("success") and "值得注意" in second.get("describe", ""),
      str(second))

# ── 9. 复用而非复制：executor 与对话路径同一管线同一注入函数 ──
import inspect
from eye.screen_ocr import salient_texts, inject_observation
src = inspect.getsource(run_observation)
check("9 run_observation 复用 screen_ocr 的显著性与注入（无第二套 OCR 逻辑）",
      "salient_texts" in src and "inject_observation" in src
      and "recognize_text" in src)

# ── 10. 区域参数：params.region 传给截屏层 ────────────────────
seen_region = {}
def cap_r(region=None):
    seen_region["region"] = region
    return {"items": [], "count": 0, "full_text": "", "region": region}
observer.capture = cap_r
observer.execute({"action_type": "screen_observe",
                  "params": {"region": [0, 0, 800, 600]}})
check("10 params.region 透传到 capture",
      seen_region.get("region") == [0, 0, 800, 600], str(seen_region))

print()
if FAILURES:
    print(f"✗ {len(FAILURES)} 项失败: {FAILURES}")
    sys.exit(1)
print("✓ SCREEN_OBSERVE 真执行链测试全过（ActionManager→router→眼睛→图谱）")
shutil.rmtree(base, ignore_errors=True)
