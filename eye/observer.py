# eye/observer.py — SCREEN_OBSERVE 的真实执行路径（ActionManager 可调用）
# ============================================================================
# 2026-09-22（用户指令）："看屏幕"原来只有对话管线的 inline 执行块
# （executor="inline:screen_observe"，capability_graph 明确排除 inline 概念
# ——它每拍都报 no_executor 缺口）。本模块给它接上真正的行动执行链：
#
#   Autonomy（图谱点亮 CuriosityDrive-[驱动]->看屏幕）
#       ↓ discover_candidates（executor 名 = action_type）
#   ActionManager.propose → _start → embodiment.execute
#       ↓ EmbodimentRouter 按 action_type 路由
#   ScreenObserver → run_observation（复用 eye/screen_ocr 同一套函数：
#       recognize_text → 图谱驱动 salient_texts → inject_observation）
#       ↓
#   eye_text_* 情景记忆 + "看屏幕"前沿点亮 → 观察结果回认知
#
# 边界：本模块不做"什么时候该看"的决策——那是既有自主机制（激活/门槛/
# 行动间隔）的职责；也不新建 OCR 逻辑，只是把已有管线挂进行动系统。
# EmbodimentRouter 是薄适配器：默认子执行器 = Minecraft 具身（全部现有
# 属性透传），新增路由表把屏幕动作交给眼睛。ActionManager 架构零改动。
# ============================================================================

import logging

logger = logging.getLogger(__name__)


def run_observation(kg, engine=None, embedder=None, region=None, capture=None):
    """一次完整的屏幕观察（识别→显著性→入图）。对话/自主两条路共用。

    capture=None 用真实截屏+RapidOCR；测试注入假 capture。
    """
    from eye.screen_ocr import recognize_text, salient_texts, inject_observation
    result = (capture or recognize_text)(region=region)
    result["salient"] = salient_texts(result, kg=kg, embedder=embedder)
    result["graph_node"] = inject_observation(kg, result, engine=engine)
    top = [s.get("text") for s in result["salient"][:3]]
    if result["salient"]:
        result["describe"] = (f"看了一眼屏幕，注意到：{('、'.join(top))}"
                              f"（共识别 {result.get('count', 0)} 条文字）")
    else:
        result["describe"] = "看了一眼屏幕，没什么值得注意的"
    # 观测：感知只在有显著内容时逐条，否则聚合（防 per-tick 刷屏，§6）
    try:
        import fas_log
        if result["salient"]:
            fas_log.get_logger(fas_log.PERCEPTION).info(
                "screen_observed", "屏幕感知入图",
                texts=result.get("count", 0), salient_n=len(result["salient"]),
                node_id=result.get("graph_node"),
                top=[str(t)[:40] for t in top])
        else:
            fas_log.aggregator("screen_none", fas_log.PERCEPTION,
                               "screen_observed_none", flush_every_s=300.0,
                               msg="多次看屏无显著内容（聚合）").add()
    except Exception:
        pass
    return result


class ScreenObserver:
    """"看屏幕"的子执行器（executor 名 = action_type = screen_observe）。

    与 MinecraftEmbodiment 同接口（execute/capabilities/poll/cancel/
    raw_state），但屏幕永远在场：不依赖任何桥连接态。同步完成——
    观察是毫秒~秒级动作，不需要 pending/轮询生命周期。
    """

    name = "screen"

    def __init__(self, kg, engine=None, embedder_fn=None):
        self.kg = kg
        self.engine = engine
        # embedder 延迟取（app 里 emb_mgr 晚于本对象构造）
        self._embedder_fn = embedder_fn or (lambda: None)
        self.capture = None          # 测试注入口（None=真实截屏）
        self.last_result = None

    def available(self) -> bool:
        return True

    def capabilities(self):
        return {"screen_observe"}

    def execute(self, action: dict) -> dict:
        params = dict(action.get("params") or {})
        region = params.get("region")
        try:
            emb = None
            try:
                emb = self._embedder_fn()
            except Exception:
                pass
            result = run_observation(self.kg, engine=self.engine,
                                     embedder=emb, region=region,
                                     capture=self.capture)
        except Exception as e:
            logger.warning(f"[ScreenObserver] 观察失败: {e}")
            return {"success": False, "action": "screen_observe",
                    "reason": f"ocr_failed:{type(e).__name__}"}
        self.last_result = result
        return {"success": True, "action": "screen_observe",
                "describe": result.get("describe", ""),
                "result": {"count": result.get("count", 0),
                           "graph_node": result.get("graph_node"),
                           "salient": [s.get("text")
                                       for s in result.get("salient", [])]}}

    def poll_action(self) -> dict:
        return {"status": "done", "detail": {}}

    def cancel(self) -> dict:
        return {"ok": True}

    def raw_state(self) -> dict:
        return {}


# ── 摄像头观察（2026-10-01，移植 The Institute Eyes）──────────


def inject_camera_observation(kg, cam_result: dict, engine=None):
    """一次摄像头观察 = 一条视觉情景记忆（eye_pose_* episodic 节点）。

    只记有内容可记的观察（看到了人 / 有行为判定）；空画面不建节点
    （与 screen 的"没什么值得注意的"同一原则）。文本结果入图，
    图像本身不落盘（eye 隐私边界）。engine 给定时点亮"看摄像头"。
    """
    import time as _t
    behaviors = [str(b) for b in (cam_result.get("behaviors") or [])]
    people = int(cam_result.get("people") or 0)
    if not behaviors and people == 0:
        return None
    node_id = f"eye_pose_{int(_t.time())}"
    if kg.get_node(node_id) is not None:   # 同秒二次观察
        node_id = f"eye_pose_{int(_t.time())}_{people}"
    from graph_model import Node
    kg.add_node(Node(
        id=node_id,
        weight=0.4,
        label="declarative-episodic",
        graph_space="episodic",
        extra_attrs={
            "type": "camera_perception",
            "people": people,
            "behaviors": behaviors[:10],
            # 行为判定依据（阈值细节，供审计；限量）
            "debug": [str(d)[:80] for d in (cam_result.get("debug") or [])][:10],
        }))
    if engine is not None:
        try:
            node = kg.get_node("看摄像头")
            if node is not None:
                node.activation = min(5.0, float(node.activation or 0.0) + 3.0)
                node.touch()
                engine.mark_active([node.id])
        except Exception as e:
            logger.debug(f"[Eye] 看摄像头激活跳过: {e}")
    logger.info(f"[Eye] 摄像头记录入图: {node_id} "
                f"({people} 人, {len(behaviors)} 条行为)")
    return node_id


class CameraObserver:
    """"看摄像头"的子执行器（executor 名 = action_type = camera_observe）。

    与 ScreenObserver 同接口（execute/capabilities/poll/cancel/
    raw_state）。一次观察 = 开摄像头抓一帧 → YOLOv8-Pose 姿态推理
    → 行为判定（eye/pose.PoseWatcher，移植自 The Institute Eyes）→
    文本结果入图。同步完成；图像仅在本机内存中处理，不落盘。

    依赖缺失（无 cv2/ultralytics）时 capabilities/available 如实为空，
    自主发现不会选中一个死能力；硬件不可用则在 execute 里诚实失败。
    """

    name = "camera"

    def __init__(self, kg, engine=None, model_path=None, camera_index=0):
        self.kg = kg
        self.engine = engine
        self.model_path = model_path
        self.camera_index = camera_index
        self._watcher = None
        self._deps_ok = None          # None=未检测（惰性一次）
        self.grab = None              # 测试注入口（None=真实摄像头取帧）
        self.last_result = None

    # ── 依赖面 ────────────────────────────────────────────────

    def _deps_available(self) -> bool:
        if self._deps_ok is None:
            try:
                import cv2  # noqa: F401
                import ultralytics  # noqa: F401
                self._deps_ok = True
            except Exception as e:
                logger.info(f"[Eye] 摄像头依赖缺失（能力不可用）: {e}")
                self._deps_ok = False
        return self._deps_ok

    def _get_watcher(self):
        if self._watcher is None:
            from eye.pose import PoseWatcher
            self._watcher = PoseWatcher(model_path=self.model_path)
        return self._watcher

    def _grab_real(self, index: int):
        """打开摄像头抓一帧 BGR。前几帧丢弃（很多摄像头首帧欠曝/陈旧）。"""
        import cv2
        cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            return None
        try:
            frame = None
            for _ in range(6):
                ok, frame = cap.read()
                if not ok:
                    return None
                if _ >= 2:      # 前 3 帧预热，之后取到的即用
                    break
            return frame
        finally:
            cap.release()

    # ── 行动接口（与 ScreenObserver 同形）────────────────────

    def available(self) -> bool:
        return self._deps_available()

    def capabilities(self):
        return {"camera_observe"} if self._deps_available() else set()

    def execute(self, action: dict) -> dict:
        params = dict(action.get("params") or {})
        index = params.get("camera_index")
        try:
            index = int(index) if index is not None else int(self.camera_index)
        except (TypeError, ValueError):
            index = int(self.camera_index)
        if not self._deps_available():
            return {"success": False, "action": "camera_observe",
                    "reason": "pose_deps_missing"}
        try:
            frame = (self.grab or self._grab_real)(index)
            if frame is None:
                return {"success": False, "action": "camera_observe",
                        "reason": "camera_unavailable"}
            result = self._get_watcher().inspect_frame(frame)
        except Exception as e:
            logger.warning(f"[CameraObserver] 观察失败: {e}")
            return {"success": False, "action": "camera_observe",
                    "reason": f"camera_failed:{type(e).__name__}"}
        result["graph_node"] = inject_camera_observation(
            self.kg, result, engine=self.engine)
        behaviors = result.get("behaviors") or []
        people = result.get("people") or 0
        if behaviors:
            result["describe"] = (
                f"看了一眼摄像头，注意到：{'、'.join(behaviors)}"
                f"（画面里 {people} 个人）")
        elif people > 0:
            result["describe"] = (
                f"看了一眼摄像头，画面里有 {people} 个人，没有可疑动作")
        else:
            result["describe"] = "看了一眼摄像头，没有看到人"
        # 观测：与 screen 同规——有内容逐条，无内容聚合（防 per-tick 刷屏）
        try:
            import fas_log
            if behaviors or people:
                fas_log.get_logger(fas_log.PERCEPTION).info(
                    "camera_observed", "摄像头感知入图",
                    people=people, behaviors=len(behaviors),
                    node_id=result.get("graph_node"))
            else:
                fas_log.aggregator("camera_none", fas_log.PERCEPTION,
                                   "camera_observed_none", flush_every_s=300.0,
                                   msg="多次看摄像头无人（聚合）").add()
        except Exception:
            pass
        self.last_result = result
        return {"success": True, "action": "camera_observe",
                "describe": result["describe"],
                "observation": {"people": people,
                                "behaviors": behaviors,
                                "graph_node": result.get("graph_node"),
                                "debug": result.get("debug", [])}}

    def poll_action(self) -> dict:
        return {"status": "done", "detail": {}}

    def cancel(self) -> dict:
        return {"ok": True}

    def raw_state(self) -> dict:
        return {}


class EmbodimentRouter:
    """薄路由器：默认子执行器透传一切；routes 表按 action_type 分发。

    ActionManager/Autonomy 看到的仍是一个 embodiment 接口；谁在执行
    当前动作由 _busy 记住（同步子执行器不会占 _busy，pending 才路由
    poll/cancel 过去）。未列出的属性一律透传给默认（ctx/detect_events/
    skill_catalog/cap_index…），保证对既有代码零感知。
    """

    def __init__(self, default, routes: dict):
        self.default = default
        self.routes = dict(routes or {})
        self._busy = None
        self.name = getattr(default, "name", "router")

    def _route_for(self, action_type: str):
        return self.routes.get(str(action_type or "").strip())

    # ── 行动接口 ──────────────────────────────────────────

    def execute(self, action: dict) -> dict:
        atype = str(action.get("action_type")
                    or action.get("type") or "").strip()
        child = self._route_for(atype) or self.default
        result = child.execute(action) or {}
        if result.get("pending"):
            self._busy = child
        return result

    def poll_action(self) -> dict:
        target = self._busy or self.default
        out = target.poll_action() or {}
        if str(out.get("status") or "") in ("done", "failed", "partial"):
            self._busy = None
        return out

    def cancel(self) -> dict:
        target = self._busy or self.default
        self._busy = None
        return target.cancel() or {}

    # ── 能力面（自主候选发现的真相源）─────────────────────

    def capabilities(self):
        caps = set()
        for child in [self.default] + list(self.routes.values()):
            fn = getattr(child, "capabilities", None)
            if callable(fn):
                try:
                    caps |= set(fn() or set())
                except Exception:
                    pass
        return caps

    def available(self) -> bool:
        try:
            return bool(self.default.available())
        except Exception:
            return False

    def raw_state(self) -> dict:
        return self.default.raw_state() or {}

    def perceive(self) -> dict:
        return self.default.perceive() or {}

    def skill_catalog(self) -> list:
        fn = getattr(self.default, "skill_catalog", None)
        return list(fn() or []) if callable(fn) else []

    def __getattr__(self, item):
        # 未显式列出的属性透传默认具身（_default/_routes 尚未建好时防递归）
        if item in ("default", "routes", "_busy"):
            raise AttributeError(item)
        return getattr(self.default, item)
