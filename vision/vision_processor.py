# vision/vision_processor.py — 视觉感知主处理器
# ============================================================================
# 完整视觉管线：
#   Image → Object Discovery → Embedding → FAISS Match → Memory → Graph
#
# 不负责：
#   - LLM 看图（禁止）
#   - 自动行动
#   - 复杂规划
# ============================================================================

import os
import time
import logging
import threading
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)

from .object_detector import ObjectDetector
from .object_embedding import ObjectEmbedder
from .faiss_index import VisionFAISSIndex
from .object_memory import ObjectMemory
from .vision_graph_bridge import VisionGraphBridge

# 低层属性提取
from .low_level_attrs import extract_low_level_attributes, SIMILARITY_THRESHOLD


class VisionProcessor:
    """视觉感知主处理器

    管线步骤：
    1. Object Discovery  - 发现图像中的独立区域
    2. Object Embedding  - 为每个区域生成向量
    3. FAISS Search     - 匹配已有对象
    4. Object Memory    - 记录出现历史
    5. Graph Bridge     - 注入知识图谱
    """

    def __init__(
        self,
        kg=None,
        engine=None,
        detector_backend: str = "auto",
        embedder_model: str = None,
        cache_dir: str = "data/vision",
    ):
        self._enabled = False
        self._lock = threading.RLock()

        # 子模块
        self.detector = ObjectDetector(backend=detector_backend)
        self.embedder = ObjectEmbedder(model_name=embedder_model)
        self.faiss = VisionFAISSIndex(dim=self.embedder.dim, cache_dir=os.path.join(cache_dir, "faiss"))
        self.memory = ObjectMemory(cache_dir=os.path.join(cache_dir, "memory"))

        # 图谱桥接（延迟绑定）
        self.kg = kg
        self.engine = engine
        self.bridge: Optional[VisionGraphBridge] = None

        # 统计
        self._process_count = 0
        self._last_result = None

        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"[VisionProcessor] 初始化完成 "
                    f"(detector={self.detector.backend}, embedding_dim={self.embedder.dim})")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True
        logger.info("[Vision] 视觉感知已启用")

    def disable(self):
        self._enabled = False
        logger.info("[Vision] 视觉感知已禁用")

    def set_graph(self, kg, engine=None):
        """绑定知识图谱和扩散引擎"""
        self.kg = kg
        self.engine = engine
        self.bridge = VisionGraphBridge(kg, engine)
        logger.info("[Vision] 图谱桥接已绑定")

    def _ensure_bridge(self):
        if self.bridge is None and self.kg is not None:
            self.bridge = VisionGraphBridge(self.kg, self.engine)

    # ── 主处理接口 ─────────────────────────────────────────

    def process_image(
        self,
        image: np.ndarray,
        source: str = None,
        activate_graph: bool = True,
    ) -> dict:
        """
        处理单张图片，完整视觉管线

        Args:
            image: RGB 图像 (H, W, 3) numpy array, uint8
            source: 来源标识（如文件路径）
            activate_graph: 是否将结果注入图谱

        Returns:
            {
                "objects": [{object_id, bbox, embedding_similarity, matched_object, status, attributes}],
                "summary": {total, matched, new},
                "process_time_s": float,
            }
        """
        t0 = time.time()

        if not self._enabled:
            return {"error": "视觉感知未启用", "enabled": False}

        try:
            # Step 1: Object Discovery
            t1 = time.time()
            detections = self.detector.detect(image)
            detect_time = time.time() - t1
            logger.info(f"[Vision] 发现 {len(detections)} 个对象 (耗时: {detect_time:.2f}s)")

            # Step 2-4: 对每个检测到的对象进行 Embedding + Match + Memory
            objects = []
            matched_count = 0
            new_count = 0

            for det in detections:
                obj_id = det["object_id"]
                mask = det.get("mask")
                bbox = det.get("bbox")

                # Step 2: Embedding
                embedding = self.embedder.encode(image, mask=mask)

                # 保存 embedding 到 FAISS
                self.faiss.add(obj_id, embedding)

                # Step 3: FAISS 搜索
                search_results = self.faiss.search(embedding, k=5)

                # 过滤掉自身
                others = [(oid, sim) for oid, sim in search_results if oid != obj_id]

                matched_object = None
                embedding_similarity = 0.0
                matched_name = None

                if others and others[0][1] >= SIMILARITY_THRESHOLD:
                    best_match_id, best_sim = others[0]
                    matched_object = best_match_id
                    embedding_similarity = float(best_sim)

                    # 使用已有对象的 ID
                    obj_id = best_match_id
                    matched_count += 1

                    # 检查已有对象是否有名称
                    mem_obj = self.memory.get_object(best_match_id)
                    if mem_obj and mem_obj.get("name"):
                        matched_name = mem_obj["name"]
                else:
                    new_count += 1

                # 提取低层属性
                attrs = extract_low_level_attributes(image, mask) if mask is not None else {}

                # Step 4: 记录到 Object Memory
                position = det.get("center")
                self.memory.record_appearance(
                    object_id=obj_id,
                    embedding=embedding,
                    position=tuple(position) if position else None,
                    source_image=source,
                    attributes=attrs,
                )

                objects.append({
                    "object_id": obj_id,
                    "bbox": bbox,
                    "center": det.get("center"),
                    "area": det.get("area"),
                    "confidence": det.get("confidence", 0.5),
                    "embedding_similarity": embedding_similarity,
                    "matched_object": matched_object,
                    "matched_name": matched_name,
                    "status": "matched" if matched_object else "new",
                    "attributes": attrs,
                })

            # Step 5: 图谱注入
            if activate_graph:
                self._ensure_bridge()
                if self.bridge:
                    self.bridge.process_vision_result(objects, source_image=source)

            # Step 6: 生成可视化
            vis_base64 = None
            try:
                vis_base64 = _draw_visualization(image, detections, objects)
            except Exception as _ve:
                logger.warning(f"[Vision] 可视化生成失败: {_ve}")

            result = {
                "objects": objects,
                "summary": {
                    "total": len(objects),
                    "matched": matched_count,
                    "new": new_count,
                },
                "timing": {
                    "detect_s": round(detect_time, 3),
                    "total_s": round(time.time() - t0, 3),
                },
                "visualization": vis_base64,  # base64 PNG 或 None
            }

            with self._lock:
                self._last_result = result
                self._process_count += 1

            logger.info(f"[Vision] 处理完成: {result['summary']} "
                        f"(耗时: {result['timing']['total_s']}s)")
            return result

        except Exception as e:
            logger.exception(f"[Vision] 处理异常: {e}")
            return {"error": str(e)}

    def process_file(
        self,
        image_path: str,
        activate_graph: bool = True,
    ) -> dict:
        """处理图片文件"""
        if not os.path.exists(image_path):
            return {"error": f"图片文件不存在: {image_path}"}

        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"无法读取图片: {image_path}"}
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image
            img = Image.open(image_path)
            image = np.array(img.convert("RGB"))

        return self.process_image(image, source=image_path, activate_graph=activate_graph)

    def process_video_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        source: str = None,
        activate_graph: bool = True,
    ) -> dict:
        """处理视频帧（与 process_image 相同，标记帧索引）"""
        source_label = f"{source}#frame{frame_idx}" if source else f"frame_{frame_idx}"
        result = self.process_image(frame, source=source_label, activate_graph=activate_graph)
        result["frame_index"] = frame_idx
        return result

    # ── 查询接口 ───────────────────────────────────────────

    def get_objects(self, status: str = None) -> list[dict]:
        """获取视觉对象列表"""
        objects = self.memory.list_objects(status)
        # 去除内部数据用于 API 返回
        result = []
        for obj in objects:
            result.append({
                "object_id": obj["object_id"],
                "status": obj["status"],
                "name": obj.get("name"),
                "appear_count": obj["appear_count"],
                "first_seen": obj["first_seen"],
                "last_seen": obj["last_seen"],
                "attributes": obj.get("attributes", {}),
                "graph_node_id": obj.get("graph_node_id"),
            })
        return result

    def confirm_object(self, object_id: str, name: str) -> dict:
        """用户确认视觉对象"""
        # 更新记忆
        self.memory.confirm_object(object_id, name)

        # 更新图谱
        self._ensure_bridge()
        if self.bridge:
            self.bridge.confirm_object(object_id, name)

        # 更新 FAISS（名称可以作为额外信息）
        return {"success": True, "object_id": object_id, "name": name}

    def get_stats(self) -> dict:
        mem_stats = self.memory.get_stats()
        return {
            "enabled": self._enabled,
            "process_count": self._process_count,
            "faiss_size": self.faiss.size,
            "detector_backend": self.detector.backend,
            "embedding_dim": self.embedder.dim,
            **mem_stats,
        }

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "process_count": self._process_count,
            "stats": self.get_stats(),
        }

    def save(self):
        """持久化所有状态"""
        self.faiss.save()
        self.memory.save()
        logger.info("[Vision] 状态已保存")


# ── 可视化工具 ────────────────────────────────────────────

def _draw_visualization(image: np.ndarray, detections: list[dict], objects: list[dict],
                        max_width: int = 800) -> str:
    """
    在原始图像上绘制分割结果，返回 base64 PNG

    Args:
        image: RGB 原图 (H, W, 3)
        detections: 原始检测结果 [{object_id, mask, bbox, ...}]
        objects: 匹配后的对象 [{object_id, status, matched_name, attributes, ...}]
        max_width: 最大宽度（等比缩放）

    Returns:
        base64 编码的 PNG 图片 data URI
    """
    import cv2
    import base64
    from io import BytesIO

    h, w = image.shape[:2]

    # 等比缩放
    if w > max_width:
        scale = max_width / w
        new_w, new_h = max_width, int(h * scale)
        vis = cv2.resize(image.copy(), (new_w, new_h))
        h, w = new_h, new_w
    else:
        vis = image.copy()

    # BGR for OpenCV drawing
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    # 颜色表（按对象索引循环）
    colors = [
        (0, 180, 60),    # 绿
        (220, 120, 0),   # 蓝
        (0, 140, 255),   # 橙
        (180, 0, 180),   # 紫
        (0, 200, 200),   # 黄
        (255, 80, 80),   # 青
        (60, 60, 220),   # 红
        (120, 200, 0),   # 青绿
    ]

    # 建立 object_id → obj 映射
    obj_map = {o["object_id"]: o for o in objects}

    for i, det in enumerate(detections):
        oid = det["object_id"]
        color = colors[i % len(colors)]

        # 在 objects 中找到匹配后的 obj_id
        matched_obj = None
        for o in objects:
            if o.get("matched_object") == oid or o["object_id"] == oid:
                matched_obj = o
                break

        if matched_obj is None:
            matched_obj = obj_map.get(oid) if oid in obj_map else det

        status = matched_obj.get("status", "new")
        name = matched_obj.get("matched_name") or matched_obj.get("attributes", {}).get("color", "")
        if not name:
            name = "new" if status == "new" else ""

        # 缩放 bbox
        if w != image.shape[1]:
            bbox = det.get("bbox")
            if bbox:
                x, y, bw, bh = bbox
                bbox = [int(x * scale), int(y * scale), int(bw * scale), int(bh * scale)]
            center = det.get("center")
            if center:
                cx, cy = center
                center = [int(cx * scale), int(cy * scale)]
        else:
            bbox = det.get("bbox")
            center = det.get("center")

        # 绘制 mask 半透明叠加
        mask = det.get("mask")
        if mask is not None and mask.any():
            if w != image.shape[1]:
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
                mask_bool = mask_resized > 0
            else:
                mask_bool = mask
            overlay = vis_bgr.copy()
            overlay[mask_bool] = color
            vis_bgr = cv2.addWeighted(vis_bgr, 0.6, overlay, 0.4, 0)

        # 绘制 bbox
        if bbox:
            x, y, bw, bh = [int(v) for v in bbox]
            thickness = 2
            cv2.rectangle(vis_bgr, (x, y), (x + bw, y + bh), color, thickness)

            # 标签
            label = f"{oid}"
            if status == "matched":
                label = f"[M] {oid}"
                if name:
                    label += f" ~{name}"
            else:
                label = f"[N] {oid}"
                if name:
                    label += f" ({name})"

            # 标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis_bgr, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(vis_bgr, label, (x + 2, y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 图例
    legend_y = 16
    cv2.putText(vis_bgr, f"[N]=New  [M]=Matched  Total:{len(detections)}",
               (6, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 编码 base64
    _, buf = cv2.imencode(".png", vis_bgr)
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ── 全局单例 ──────────────────────────────────────────────

_vision_processor: Optional[VisionProcessor] = None


def get_vision_processor() -> VisionProcessor:
    global _vision_processor
    if _vision_processor is None:
        _vision_processor = VisionProcessor()
    return _vision_processor
