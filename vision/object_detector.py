# vision/object_detector.py — 基于 SAM2 的对象发现
# ============================================================================
# 输入图片/视频帧，输出多个 Object Mask。
# SAM2 只负责发现独立区域，禁止生成物体名称。
# ============================================================================

import logging
import os
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# SAM2 推理是否已接入。
# 当前 _run_sam2_inference 是占位实现（直接回退 OpenCV），因此检测实际
# 总是由 OpenCV 完成——在推理路径补全之前，后端不谎报成 SAM2。
# 补全 _load_sam2 / _run_sam2_inference 后把这里置 True 即可启用。
_SAM2_INFERENCE_READY = False

# ── 通用临时 ID 生成 ──────────────────────────────────────

_auto_counter = 0


def _next_temp_id() -> str:
    global _auto_counter
    _auto_counter += 1
    return f"UnknownObject{_auto_counter:04d}"


class ObjectDetector:
    """对象检测器

    支持多种后端：
    - sam2: Meta SAM2 模型（推荐，需安装）
    - opencv: OpenCV 轮廓检测（轻量级回退）
    """

    def __init__(self, backend: str = "auto", model_path: str = None):
        """
        Args:
            backend: "sam2" | "opencv" | "auto"
            model_path: SAM2 模型路径
        """
        self.backend = backend
        self.model_path = model_path or "E:/Models/SAM2 Tiny"
        self._model = None

        if backend == "auto":
            self._resolve_backend()

    def _resolve_backend(self):
        """自动选择可用后端——逐项检查依赖，如实报告缺什么。

        SAM2 要真正跑起来需要四样齐备：
          1) CUDA 版 PyTorch（CPU 版能 import，但分割推理慢到不可用）
          2) sam2 包
          3) 本地权重（默认 E:/Models/SAM2 Tiny）
          4) 推理路径已实现（见模块顶部 _SAM2_INFERENCE_READY）

        旧实现只看 torch.cuda，就写"PyTorch CPU 版，SAM2 不可用"并建议去装
        CUDA 版 torch——而本机 sam2 包与权重都不存在、推理路径也还是占位，
        照那条建议装完 1.9GB 轮子，后端照样是 OpenCV。日志应该说真话：
        列出真正缺的东西，不给出无效建议。
        """
        missing = []
        torch_mod = None
        try:
            import torch as torch_mod
            if not torch_mod.cuda.is_available():
                missing.append(f"CUDA 版 PyTorch（当前 {torch_mod.__version__}）")
        except ImportError:
            missing.append("PyTorch")
        try:
            import sam2  # noqa: F401
        except ImportError:
            missing.append("sam2 包")
        if not (self.model_path and os.path.isdir(self.model_path)):
            missing.append(f"权重（{self.model_path}）")
        if not _SAM2_INFERENCE_READY:
            missing.append("推理路径未实现")

        if missing:
            self.backend = "opencv"
            logger.info("[ObjectDetector] 后端 = OpenCV 轮廓检测；SAM2 未启用，缺: "
                        + "、".join(missing))
        else:
            self.backend = "sam2"
            gpu = torch_mod.cuda.get_device_name(0) if torch_mod else "?"
            logger.info(f"[ObjectDetector] 后端 = SAM2 (GPU: {gpu})")

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        检测图像中的独立对象

        Args:
            image: RGB 图像 (H, W, 3) numpy array, uint8

        Returns:
            [
                {
                    "object_id": "UnknownObject0001",
                    "mask": np.ndarray (H, W) bool,
                    "bbox": [x, y, w, h],
                    "area": int,
                    "center": [cx, cy],
                    "confidence": float,
                },
                ...
            ]
        """
        if self.backend == "sam2":
            return self._detect_sam2(image)
        elif self.backend == "opencv":
            return self._detect_opencv(image)
        else:
            logger.warning(f"[ObjectDetector] 未知后端: {self.backend}")
            return self._detect_opencv(image)

    # ── SAM2 检测 ──────────────────────────────────────────

    def _detect_sam2(self, image: np.ndarray) -> list[dict]:
        """使用 SAM2 进行对象分割"""
        try:
            import torch

            # 尝试加载 SAM2
            if self._model is None:
                self._load_sam2()

            if self._model is not None:
                return self._run_sam2_inference(image)
            else:
                logger.warning("[ObjectDetector] SAM2 加载失败，回退 OpenCV")
                return self._detect_opencv(image)

        except Exception as e:
            logger.warning(f"[ObjectDetector] SAM2 异常: {e}，回退 OpenCV")
            return self._detect_opencv(image)

    def _load_sam2(self):
        """加载 SAM2 模型（未实现：推理路径仍是占位，见 _SAM2_INFERENCE_READY）"""
        try:
            import torch
            # 尝试从本地或 hub 加载
            if self.model_path and os.path.isdir(self.model_path):
                logger.info(f"[ObjectDetector] 从本地加载 SAM2: {self.model_path}")
                # SAM2 加载方式取决于具体安装
                # from sam2.build_sam import build_sam2
                # self._model = build_sam2(self.model_path)
                self._model = "sam2_placeholder"  # 占位
            else:
                logger.info("[ObjectDetector] SAM2 模型路径不可用，将使用 OpenCV")
                self._model = None
        except ImportError:
            logger.warning("[ObjectDetector] SAM2 库不可用")
            self._model = None

    def _run_sam2_inference(self, image: np.ndarray) -> list[dict]:
        """运行 SAM2 推理"""
        # 占位实现：SAM2 的完整推理需要根据具体安装版本调用
        # 在 SAM2 不可用时回退到 OpenCV 方案
        logger.info("[ObjectDetector] SAM2 推理占位 - 回退 OpenCV")
        return self._detect_opencv(image)

    # ── OpenCV 回退检测 ────────────────────────────────────

    def _detect_opencv(self, image: np.ndarray) -> list[dict]:
        """使用 OpenCV 进行轮廓检测（轻量级回退）

        步骤：
        1. 转灰度
        2. 高斯模糊
        3. Canny 边缘检测
        4. 轮廓查找
        5. 过滤太小的区域
        """
        try:
            import cv2
        except ImportError:
            logger.error("[ObjectDetector] OpenCV 不可用")
            return []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 自适应阈值 + Canny
        edges = cv2.Canny(blurred, 50, 150)

        # 形态学闭运算连接边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        min_area = (h * w) * 0.001  # 最小面积为图像的 0.1%

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # 边界框
            x, y, bw, bh = cv2.boundingRect(cnt)

            # 中心点
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + bw / 2, y + bh / 2

            # 生成 mask
            mask = np.zeros((h, w), dtype=bool)
            cv2.drawContours(mask.astype(np.uint8), [cnt], -1, 1, -1)
            mask = mask > 0

            obj_id = _next_temp_id()

            objects.append({
                "object_id": obj_id,
                "mask": mask,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": int(area),
                "center": [float(cx), float(cy)],
                "confidence": min(1.0, area / (h * w * 0.01)),
            })

        logger.info(f"[ObjectDetector] 发现 {len(objects)} 个对象 (OpenCV)")
        return objects

    def reset_counter(self):
        global _auto_counter
        _auto_counter = 0
