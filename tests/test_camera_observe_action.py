# test_camera_observe_action.py — CAMERA_OBSERVE 真执行链（The Institute Eyes 移植验收）
# 链路：propose → ActionManager → EmbodimentRouter → CameraObserver
#       → PoseWatcher.analyze（移植自 live_inference.py）→ eye_pose_* 入图
# 离线：假 grab/analyze（不开摄像头、不加载 ultralytics）、零 LLM。
# 运行: E:/Miniforge.envs/Fascinator/python.exe -X utf8 tests/test_camera_observe_action.py

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
from eye.observer import CameraObserver, EmbodimentRouter

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


def make_grab(people=1):
    """假抓帧：返回非 None 即代表摄像头在场（帧内容由假 analyze 消费）。"""
    def grab(index):
        grab.seen_index = index
        return f"frame_{index}"
    grab.seen_index = None
    return grab


def make_analyze(people=1, behaviors=None, debug=None):
    """假姿态分析：绕开 ultralytics，直接喂检测结果。"""
    def analyze(frame):
        return {"people": people, "behaviors": list(behaviors or []),
                "debug": list(debug or []),
                "keypoints": [], "boxes": []}
    return analyze


class StubMinecraft:
    def __init__(self, caps=("mine",)):
        self.caps = set(caps)
        self.executed = []

    def available(self):
        return bool(self.caps)

    def capabilities(self):
        return set(self.caps)

    def execute(self, action):
        self.executed.append(action)
        return {"success": True, "action": action.get("action_type"),
                "describe": "stub"}

    def poll_action(self):
        return {"status": "done", "detail": {}}

    def cancel(self):
        return {"ok": True}

    def perceive(self):
        return {}

    def raw_state(self):
        return {"health": 20}


base = os.path.join(tempfile.gettempdir(), "fas_test_camera_observe")
shutil.rmtree(base, ignore_errors=True)
os.makedirs(base, exist_ok=True)

kg = KnowledgeGraph()
kg.add_node(Node(id="Haru", graph_space="self"))
kg.add_node(Node(id="看摄像头", graph_space="self"))
eng = DiffusionEngine(kg, dict(CFG))
eng.name_to_node = dict(kg.nodes)

observer = CameraObserver(kg, engine=eng)
observer.grab = make_grab()
observer._watcher = True          # 绕过 PoseWatcher 懒加载（离线）
observer._deps_ok = True          # 假装 ultralytics/cv2 在场
observer._get_watcher = lambda: type("W", (), {
    "inspect_frame": staticmethod(make_analyze(
        people=2, behaviors=["快速挥拳 (人物 1)"],
        debug=["腕部速度 88.5"]))})()
mc = StubMinecraft(caps=set())    # MC 离线：摄像头动作必须照做
router = EmbodimentRouter(default=mc, routes={"camera_observe": observer})
am = ActionManager(embodiment=router, kg=kg, engine=eng, config={},
                   data_dir=base)

# ── 1. 全链：propose → router → observer → 检测 → 图谱 ─────────
n_before = sum(1 for nid in kg.nodes if str(nid).startswith("eye_pose_"))
out = am.propose({"action_type": "camera_observe", "motivation": "curiosity"},
                 source="autonomy")
check("1 ActionManager 接受并立即启动 camera_observe",
      out.get("started") and out.get("success"), str(out))
n_after = sum(1 for nid in kg.nodes if str(nid).startswith("eye_pose_"))
check("1' 观察结果成为一条 eye_pose_* 情景记忆", n_after == n_before + 1,
      f"{n_before}→{n_after}")
check("1'' 同步执行器不占用承诺期", not am.busy())
eye_id = next(nid for nid in kg.nodes if str(nid).startswith("eye_pose_"))
ea = kg.nodes[eye_id].extra_attrs
check("1''' 记忆内容：人数与行为文本入图（图像本身不入图）",
      ea.get("people") == 2
      and ea.get("behaviors") == ["快速挥拳 (人物 1)"]
      and ea.get("type") == "camera_perception", str(ea))
check("1'''' describe 如实：注意到挥拳 + 人数",
      "快速挥拳" in (out.get("describe") or "") and "2 个人" in out.get("describe", ""),
      str(out))

# ── 2. 前沿不变量：激活直写点必须 mark_active ──────────────────
check("2 '看摄像头'激活被点亮且进入活跃前沿",
      kg.nodes["看摄像头"].activation >= 3.0 and "看摄像头" in eng._active_nodes,
      f"act={kg.nodes['看摄像头'].activation}")

# ── 3. 能力面：能力取并集；离线 MC 时只剩摄像头 ────────────────
check("3 router.capabilities = MC∪摄像头",
      router.capabilities() == {"camera_observe"}
      and mc.capabilities() == set())
router2 = EmbodimentRouter(default=StubMinecraft(caps=("mine",)),
                           routes={"camera_observe": observer})
check("3' 在线时能力取并集",
      router2.capabilities() == {"mine", "camera_observe"})

# ── 4. 路由正确性：非摄像头动作交给默认具身 ────────────────────
r = router.execute({"action_type": "mine", "params": {"block": "dirt"}})
check("4 非 camera_observe 路由到默认具身执行",
      r.get("success") and mc.executed
      and mc.executed[-1]["action_type"] == "mine")

# ── 5. 依赖缺失：能力如实缺席（自主不会选中死能力）─────────────
obs_nodeps = CameraObserver(kg, engine=eng)
obs_nodeps._deps_ok = False
check("5 依赖缺失 → available False / capabilities 空",
      obs_nodeps.available() is False and obs_nodeps.capabilities() == set())
res = obs_nodeps.execute({"action_type": "camera_observe"})
check("5' 依赖缺失执行 → success=False 带 pose_deps_missing（不伪造）",
      res.get("success") is False and res.get("reason") == "pose_deps_missing",
      str(res))

# ── 6. 硬件不可用：抓帧 None → 诚实失败 ────────────────────────
observer.grab = lambda index: None
res = observer.execute({"action_type": "camera_observe"})
check("6 摄像头打不开 → success=False 带 camera_unavailable",
      res.get("success") is False and res.get("reason") == "camera_unavailable",
      str(res))

# ── 7. 空画面：没看到人没行为 → 执行成功但不建记忆节点 ─────────
kg3 = KnowledgeGraph()
kg3.add_node(Node(id="看摄像头"))
obs3 = CameraObserver(kg3, engine=None)
obs3.grab = make_grab()
obs3._deps_ok = True
obs3._get_watcher = lambda: type("W", (), {
    "inspect_frame": staticmethod(make_analyze(people=0, behaviors=[]))})()
res = obs3.execute({"action_type": "camera_observe"})
check("7 空画面 describe 如实（没有看到人），执行仍算成功",
      res.get("success") and "没有看到人" in res.get("describe", ""),
      str(res))
check("7' 空画面不建 eye_pose_* 节点",
      not any(str(nid).startswith("eye_pose_") for nid in kg3.nodes))

# ── 8. 有画面无行为：describe 说有人但没可疑动作，仍入图 ────────
kg4 = KnowledgeGraph()
kg4.add_node(Node(id="看摄像头"))
obs4 = CameraObserver(kg4, engine=None)
obs4.grab = make_grab()
obs4._deps_ok = True
obs4._get_watcher = lambda: type("W", (), {
    "inspect_frame": staticmethod(make_analyze(people=1, behaviors=[]))})()
res = obs4.execute({"action_type": "camera_observe"})
check("8 有人无行为 → describe 说有人、没有可疑动作",
      res.get("success") and "1 个人" in res.get("describe", "")
      and "没有可疑动作" in res.get("describe", ""), str(res))
check("8' 有人的观察仍成为情景记忆",
      any(str(nid).startswith("eye_pose_") for nid in kg4.nodes))

# ── 9. params.camera_index 透传到抓帧层 ────────────────────────
grab_seen = make_grab()
observer.grab = grab_seen
observer.execute({"action_type": "camera_observe",
                  "params": {"camera_index": 2}})
check("9 params.camera_index 透传到 grab",
      grab_seen.seen_index == 2, str(grab_seen.seen_index))

# ── 10. 行为判定移植保真：PoseWatcher.analyze 原文逻辑 ──────────
from eye.pose import PoseWatcher
import numpy as np
w = PoseWatcher()
w.frame_width, w.frame_height, w.fps = 1280, 720, 30
# 单人 17 关键点（标准站姿骨架，低置信度 → 应被核心置信度门槛拦下）
kp = np.zeros((1, 17, 2)); conf = np.full((1, 17), 0.9)
conf[0, [5, 6, 7, 8, 9, 10]] = 0.3   # 核心点低置信
beh, dbg = w.analyze(kp, conf, np.array([[0, 0, 100, 200]], dtype=float))
check("10 核心关键点低置信 → 不产出行为（原文门槛保留）",
      beh == [] and any("置信度过低" in d for d in dbg), str((beh, dbg)))
check("10' 几何函数原文保留",
      round(PoseWatcher.calculate_angle([0, 0], [1, 0], [1, 1])) == 90
      and round(PoseWatcher.calculate_distance([0, 0], [3, 4])) == 5)
# 时间窗：行为须连续 3 帧在场才确认（原文语义）
w2 = PoseWatcher()
w2.frame_width, w2.frame_height, w2.fps = 1280, 720, 30
conf_ok = np.full((1, 17), 0.9)
kp_stand = np.zeros((1, 17, 2))
kp_stand[0, 11] = [100, 200]; kp_stand[0, 12] = [160, 200]  # 髋
kp_stand[0, 13] = [100, 300]; kp_stand[0, 14] = [160, 300]  # 膝（不下蹲）
kp_stand[0, 0] = [130, 50]; kp_stand[0, 5] = [100, 100]
kp_stand[0, 6] = [160, 100]; kp_stand[0, 7] = [95, 150]
kp_stand[0, 8] = [165, 150]; kp_stand[0, 9] = [95, 190]
kp_stand[0, 10] = [165, 190]; kp_stand[0, 15] = [100, 400]
kp_stand[0, 16] = [160, 400]
w2.analyze(kp_stand.copy(), conf_ok.copy(),
           np.array([[0, 0, 200, 400]], dtype=float))
beh2, dbg2 = w2.analyze(kp_stand.copy(), conf_ok.copy(),
                        np.array([[0, 0, 200, 400]], dtype=float))
check("10'' 行为历史不足 3 帧 → 不确认（疑似通道如实说明）",
      beh2 == [] and any("疑似" in d or "不足 3 帧" in d for d in dbg2),
      str(dbg2))

print()
if FAILURES:
    print(f"✗ {len(FAILURES)} 项失败: {FAILURES}")
    sys.exit(1)
print("✓ CAMERA_OBSERVE 真执行链测试全过（ActionManager→router→眼睛→图谱）")
shutil.rmtree(base, ignore_errors=True)
