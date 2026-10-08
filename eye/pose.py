# eye/pose.py — 摄像头姿态/行为感知器（移植自 The Institute Eyes V0.39）
# ============================================================================
# 来源（2026-10-01，用户指令）："E:/Project/The Institute Eyes" 的核心检测
# 逻辑原样移植进 FAS。原实现长在 Tkinter 类 PoseDetectionApp 里
# （live_inference.py），无法直接 import（构造即建 GUI，detect_crime 内部
# 直接调 root.after / status_bar）——按"幅度大则复制"的裁决改为复制适配。
#
# 移植边界：
#   保留（原样）：calculate_angle / calculate_distance / detect_crime 的
#     全部行为判定（快速挥拳、偷窃、逃逸跑动、隐秘蹲伏、摔倒、破坏、
#     斗殴、推搡、尾随、抢夺）、动态阈值（分辨率/FPS 缩放）、3 帧行为
#     时间窗（行为须连续在场地确认）、挥拳计数确认。
#   剥离：Tkinter GUI、filedialog、关键帧画廊与 cv2.imwrite 落盘——
#     FAS 的 eye 隐私边界（见 eye/screen_ocr.py）：图像仅在本机内存中
#     处理，不落盘、不上传。检测结果只以文本入图谱。
#   适配：状态收敛进 PoseWatcher（prev_keypoints / behavior_history /
#     punch_counts），模型懒加载（ultralytics 首次加载较重）。
#
# 分工不变：本模块只负责"看见什么行为"；"什么时候看"归自主/对话决策层，
# "什么值得记"由 CameraObserver 的注入规则决定。
# ============================================================================

import logging
from collections import deque
from math import sqrt, degrees, acos

import numpy as np

logger = logging.getLogger(__name__)


class PoseWatcher:
    """YOLOv8-Pose 单帧→行为检测（无 GUI、无落盘）。

    跨帧状态（挥拳速度/摔倒位移需要连续帧参照）活在本对象里：
    一次"看摄像头"是一帧，但前后几次观察共享同一套 prev_* 历史。
    """

    def __init__(self, model_path=None, conf=0.7, keypoint_conf=0.5):
        self.model_path = model_path      # None = 交由 ultralytics 自动解析
        self.conf = conf
        self.keypoint_conf = keypoint_conf
        self._model = None
        self._model_lock = None           # 惰性建（避免无谓的 threading 依赖）
        # 动作检测的帧历史（原文参数保留）
        self.prev_keypoints = deque(maxlen=5)
        self.prev_boxes = deque(maxlen=5)
        self.behavior_history = deque(maxlen=3)
        self.punch_counts = {}
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30

    # ── 模型（懒加载）─────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            import threading
            if self._model_lock is None:
                self._model_lock = threading.Lock()
            with self._model_lock:
                if self._model is None:
                    from ultralytics import YOLO
                    kw = {"model": self.model_path} if self.model_path else {}
                    self._model = YOLO(**kw)
                    logger.info("[Eye] YOLO pose 模型加载完成")
        return self._model

    # ── 几何（原文保留）───────────────────────────────────────

    @staticmethod
    def calculate_angle(p1, p2, p3):
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 * mag2 == 0:
            return 0
        cos_angle = dot / (mag1 * mag2)
        cos_angle = min(max(cos_angle, -1), 1)
        return degrees(acos(cos_angle))

    @staticmethod
    def calculate_distance(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # ── 单帧推理 ─────────────────────────────────────────────

    def inspect_frame(self, frame) -> dict:
        """一帧 BGR 图 → 姿态推理 + 行为判定（并更新跨帧历史）。

        返回 {people, behaviors, debug, keypoints, boxes}。
        """
        self.frame_width = int(frame.shape[1]) if frame is not None else 0
        self.frame_height = int(frame.shape[0]) if frame is not None else 0
        results = self._get_model()(frame, conf=self.conf)
        r = results[0]
        keypoints = r.keypoints.xy.cpu().numpy()
        keypoint_conf = (r.keypoints.conf.cpu().numpy()
                         if r.keypoints.has_visible
                         else np.zeros_like(keypoints))
        boxes = r.boxes.xyxy.cpu().numpy()
        behaviors, debug = self.analyze(keypoints, keypoint_conf, boxes)
        if len(keypoints) > 0:
            self.prev_keypoints.append(keypoints)
            self.prev_boxes.append(boxes)
        return {
            "people": int(len(keypoints)),
            "behaviors": behaviors,
            "debug": debug,
            "keypoints": keypoints,
            "boxes": boxes,
        }

    # ── 行为判定（detect_crime 原文移植，剥离落盘/GUI）────────

    def analyze(self, keypoints, keypoint_conf, boxes):
        behaviors = []
        debug_info = []
        if len(keypoints) == 0 or len(keypoint_conf) == 0 or len(boxes) == 0:
            debug_info.append("未检测到关键点或边界框")
            return behaviors, debug_info

        # 动态阈值
        scale_factor = self.frame_width / 1280 if self.frame_width > 0 else 1
        fps_factor = min(1.0, 30 / self.fps if self.fps > 0 else 1)
        shoulder_dist_threshold = 180 * scale_factor
        collision_dist_threshold = 70 * scale_factor
        wrist_speed_threshold = 60 * scale_factor * fps_factor
        hip_drop_threshold = 100 * (self.frame_height / 720 if self.frame_height > 0 else 1)

        # 处理单人动作
        for i, (person, conf, box) in enumerate(zip(keypoints, keypoint_conf, boxes)):
            if len(person) < 17 or len(conf) < 17:
                debug_info.append(f"人物 {i + 1}: 关键点不完整")
                continue
            core_keypoints = conf[[5, 6, 7, 8, 9, 10]]
            if len(core_keypoints) == 0 or np.any(core_keypoints < self.keypoint_conf):
                debug_info.append(
                    f"人物 {i + 1}: 核心关键点置信度过低 ({np.min(core_keypoints):.2f} < {self.keypoint_conf:.2f})")
                continue

            person = person[:, :2].tolist()
            nose = person[0]
            left_shoulder, right_shoulder = person[5], person[6]
            left_elbow, right_elbow = person[7], person[8]
            left_wrist, right_wrist = person[9], person[10]
            left_hip, right_hip = person[11], person[12]
            left_knee, right_knee = person[13], person[14]
            left_ankle, right_ankle = person[15], person[16]

            # 快速挥拳
            if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                prev_person = self.prev_keypoints[-1][i]
                if len(prev_person) >= 17:
                    prev_person = prev_person[:, :2].tolist()
                    prev_left_wrist, prev_right_wrist = prev_person[9], prev_person[10]
                    wrist_speed_left = self.calculate_distance(left_wrist, prev_left_wrist)
                    wrist_speed_right = self.calculate_distance(right_wrist, prev_right_wrist)
                    left_elbow_angle = self.calculate_angle(left_wrist, left_elbow, left_shoulder)
                    right_elbow_angle = self.calculate_angle(right_wrist, right_elbow, right_shoulder)
                    if (wrist_speed_left > wrist_speed_threshold or wrist_speed_right > wrist_speed_threshold) and \
                            (left_elbow_angle < 90 or right_elbow_angle < 90):
                        punch_key = f"punch_{i}"
                        self.punch_counts[punch_key] = self.punch_counts.get(punch_key, 0) + 1
                        if self.punch_counts[punch_key] >= 2:
                            behaviors.append(f"快速挥拳 (人物 {i + 1})")
                    elif (wrist_speed_left <= wrist_speed_threshold and wrist_speed_right <= wrist_speed_threshold):
                        debug_info.append(
                            f"人物 {i + 1}: 腕部速度不足 ({max(wrist_speed_left, wrist_speed_right):.2f} < {wrist_speed_threshold:.2f})")

            # 其他动作
            body_angle = self.calculate_angle(nose, [(left_shoulder[0] + right_shoulder[0]) / 2,
                                                     (left_shoulder[1] + right_shoulder[1]) / 2],
                                              [left_hip[0], left_hip[1]])
            if nose[1] > min(left_shoulder[1], right_shoulder[1]) and \
                    (left_wrist[1] > left_hip[1] or right_wrist[1] > right_hip[1]) and body_angle > 30:
                behaviors.append(f"偷窃动作 (人物 {i + 1})")

            if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                prev_person = self.prev_keypoints[-1][i]
                if len(prev_person) >= 17:
                    prev_person = prev_person[:, :2].tolist()
                    prev_ankle = prev_person[15]
                    stride = self.calculate_distance(left_ankle, right_ankle)
                    speed = self.calculate_distance(left_ankle, prev_ankle)
                    if stride > 220 * scale_factor and speed > 90 * scale_factor * fps_factor:
                        behaviors.append(f"逃逸跑动 (人物 {i + 1})")

            if (left_hip[1] > left_knee[1] - 35 * (self.frame_height / 720)) and \
                    (right_hip[1] > right_knee[1] - 35 * (self.frame_height / 720)):
                behaviors.append(f"隐秘蹲伏 (人物 {i + 1})")

            if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                prev_person = self.prev_keypoints[-1][i]
                if len(prev_person) >= 17:
                    prev_person = prev_person[:, :2].tolist()
                    prev_hip = [(prev_person[11][0] + prev_person[12][0]) / 2,
                                (prev_person[11][1] + prev_person[12][1]) / 2]
                    curr_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
                    hip_drop = curr_hip[1] - prev_hip[1]
                    if hip_drop > hip_drop_threshold:
                        behaviors.append(f"摔倒动作 (人物 {i + 1})")

            if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                prev_person = self.prev_keypoints[-1][i]
                if len(prev_person) >= 17:
                    prev_person = prev_person[:, :2].tolist()
                    prev_left_wrist, prev_right_wrist = prev_person[9], prev_person[10]
                    wrist_speed_left = self.calculate_distance(left_wrist, prev_left_wrist)
                    wrist_speed_right = self.calculate_distance(right_wrist, prev_right_wrist)
                    if (left_wrist[1] < left_shoulder[1] or right_wrist[1] < right_shoulder[1]) and \
                            (wrist_speed_left > wrist_speed_threshold or wrist_speed_right > wrist_speed_threshold) and \
                            (left_wrist[1] > prev_left_wrist[1] or right_wrist[1] > prev_right_wrist[1]):
                        behaviors.append(f"破坏动作 (人物 {i + 1})")

        # 多人交互动作
        if len(keypoints) >= 2:
            for i in range(len(keypoints)):
                for j in range(i + 1, len(keypoints)):
                    person1, person2 = keypoints[i][:, :2].tolist(), keypoints[j][:, :2].tolist()
                    conf1, conf2 = keypoint_conf[i], keypoint_conf[j]
                    core_keypoints1 = conf1[[5, 6, 7, 8, 9, 10]]
                    core_keypoints2 = conf2[[5, 6, 7, 8, 9, 10]]
                    if len(core_keypoints1) == 0 or len(core_keypoints2) == 0 or \
                            np.any(core_keypoints1 < self.keypoint_conf) or np.any(
                        core_keypoints2 < self.keypoint_conf):
                        debug_info.append(
                            f"人物 {i + 1} ↔ {j + 1}: 核心关键点置信度过低 ({min(np.min(core_keypoints1), np.min(core_keypoints2)):.2f} < {self.keypoint_conf:.2f})")
                        continue

                    p1_shoulder = [(person1[5][0] + person1[6][0]) / 2, (person1[5][1] + person1[6][1]) / 2]
                    p2_shoulder = [(person2[5][0] + person2[6][0]) / 2, (person2[5][1] + person2[6][1]) / 2]
                    p1_hip = [(person1[11][0] + person1[12][0]) / 2, (person1[11][1] + person1[12][1]) / 2]
                    p2_hip = [(person2[11][0] + person2[12][0]) / 2, (person2[11][1] + person2[12][1]) / 2]
                    p1_wrist, p2_wrist = person1[9], person2[9]
                    p1_elbow, p2_elbow = person1[7], person2[7]

                    # 斗殴动作
                    shoulder_dist = self.calculate_distance(p1_shoulder, p2_shoulder)
                    if shoulder_dist < shoulder_dist_threshold:
                        fight_components = []
                        torso1 = [p1_shoulder, p1_hip]
                        torso2 = [p2_shoulder, p2_hip]
                        torso_dist = min(
                            self.calculate_distance(torso1[0], torso2[0]),
                            self.calculate_distance(torso1[0], torso2[1]),
                            self.calculate_distance(torso1[1], torso2[0]),
                            self.calculate_distance(torso1[1], torso2[1])
                        )
                        if torso_dist < collision_dist_threshold:
                            fight_components.append("身体冲撞")

                        arm1 = [p1_wrist, p1_elbow]
                        arm2 = [p2_wrist, p2_elbow]
                        arm_dist = min(
                            self.calculate_distance(arm1[0], arm2[0]),
                            self.calculate_distance(arm1[0], arm2[1]),
                            self.calculate_distance(arm1[1], arm2[0]),
                            self.calculate_distance(arm1[1], arm2[1])
                        )
                        if arm_dist < collision_dist_threshold:
                            fight_components.append("手臂交叉")

                        punch_detected = False
                        if self.prev_keypoints and len(self.prev_keypoints[-1]) > i and len(
                                self.prev_keypoints[-1]) > j:
                            prev_p1_wrist = self.prev_keypoints[-1][i][9][:2].tolist()
                            prev_p2_wrist = self.prev_keypoints[-1][j][9][:2].tolist()
                            wrist_speed1 = self.calculate_distance(p1_wrist, prev_p1_wrist)
                            wrist_speed2 = self.calculate_distance(p2_wrist, prev_p2_wrist)
                            wrist_angle1 = self.calculate_angle(p1_wrist, person1[7], person1[5])
                            wrist_angle2 = self.calculate_angle(p2_wrist, person2[7], person2[5])
                            if (wrist_speed1 > wrist_speed_threshold or wrist_speed2 > wrist_speed_threshold) and \
                                    (wrist_angle1 < 90 or wrist_angle2 < 90):
                                punch_key1 = f"punch_{i}"
                                punch_key2 = f"punch_{j}"
                                self.punch_counts[punch_key1] = self.punch_counts.get(punch_key1, 0) + 1
                                self.punch_counts[punch_key2] = self.punch_counts.get(punch_key2, 0) + 1
                                if self.punch_counts[punch_key1] >= 2 or self.punch_counts[punch_key2] >= 2:
                                    fight_components.append("快速挥拳")
                                    punch_detected = True
                            if not punch_detected:
                                debug_info.append(
                                    f"人物 {i + 1} ↔ {j + 1}: 腕部速度不足 ({max(wrist_speed1, wrist_speed2):.2f} < {wrist_speed_threshold:.2f})")

                        if len(fight_components) >= 2:
                            behaviors.append(f"斗殴动作 (人物 {i + 1} ↔ {j + 1}: {', '.join(fight_components)})")

                    # 推搡动作
                    if shoulder_dist < shoulder_dist_threshold:
                        if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                            prev_p1_wrist = self.prev_keypoints[-1][i][9][:2].tolist()
                            wrist_speed = self.calculate_distance(p1_wrist, prev_p1_wrist)
                            wrist_angle = self.calculate_angle(p1_wrist, person1[7], person1[5])
                            if wrist_speed > 100 * scale_factor * fps_factor and wrist_angle > 120:
                                behaviors.append(f"推搡动作 (人物 {i + 1} -> {j + 1})")

                    # 尾随动作
                    nose_to_shoulder_dist = self.calculate_distance(person1[0], p2_shoulder)
                    if nose_to_shoulder_dist < 150 * scale_factor and person1[0][1] < p2_shoulder[1]:
                        if self.prev_keypoints and len(self.prev_keypoints[-1]) > i and len(
                                self.prev_keypoints[-1]) > j:
                            prev_p1_nose = self.prev_keypoints[-1][i][0][:2].tolist()
                            prev_p2_shoulder = [
                                (self.prev_keypoints[-1][j][5][0] + self.prev_keypoints[-1][j][6][0]) / 2,
                                (self.prev_keypoints[-1][j][5][1] + self.prev_keypoints[-1][j][6][1]) / 2]
                            speed_diff = self.calculate_distance(prev_p1_nose, person1[0]) - self.calculate_distance(
                                p2_shoulder, prev_p2_shoulder)
                            if abs(speed_diff) < 20 * scale_factor * fps_factor:
                                behaviors.append(f"尾随动作 (人物 {i + 1} -> {j + 1})")

                    # 抢夺动作
                    wrist_to_shoulder_dist = self.calculate_distance(p1_wrist, p2_shoulder)
                    if wrist_to_shoulder_dist < 50 * scale_factor:
                        if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                            prev_p1_wrist = self.prev_keypoints[-1][i][9][:2].tolist()
                            wrist_speed = self.calculate_distance(p1_wrist, prev_p1_wrist)
                            if wrist_speed > 100 * scale_factor * fps_factor:
                                behaviors.append(f"抢夺动作 (人物 {i + 1} -> {j + 1})")

        # 时间窗口（原文：行为须连续在场地确认，斗殴优先）
        self.behavior_history.append(set(behaviors))
        if len(self.behavior_history) >= 3:
            common_behaviors = set.intersection(*list(self.behavior_history))
            fight_behaviors = [b for b in common_behaviors if "斗殴动作" in b]
            if fight_behaviors:
                return fight_behaviors, debug_info
            if not common_behaviors and not behaviors:
                debug_info.append("未检测到持续动作")
            return list(common_behaviors), debug_info
        debug_info.append("行为历史不足 3 帧")
        if behaviors:
            # FAS 适配（原文无）：单帧式观察下前 2 次永远无法确认会不诚实——
            # 未过时间窗的行为以"疑似"身份上报，由注入层决定记什么。
            debug_info.append(f"疑似 (待连续确认): {'; '.join(behaviors)}")
        return [], debug_info
