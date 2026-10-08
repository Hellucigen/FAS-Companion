# vision/object_memory.py — 持久对象记忆系统
# ============================================================================
# 对象不是一次性识别结果，而是长期存在的实体。
# 跟踪每个对象的出现历史、embedding 变化、属性学习。
#
# 对象状态：
#   - temporary: 初次发现，等待确认
#   - persistent: 经多次观察/用户确认后升级
# ============================================================================

import os
import json
import logging
import threading
import time
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class ObjectMemory:
    """持久对象记忆

    存储结构:
    {
        object_id: {
            "object_id": str,
            "status": "temporary" | "persistent",
            "embedding_history": [list[float]],  # 最近 N 个 embedding
            "first_seen": "ISO timestamp",
            "last_seen": "ISO timestamp",
            "appear_count": int,
            "positions": [{"x": float, "y": float, "source": str}],  # 最近出现位置
            "graph_node_id": str | None,  # 关联的图谱节点 ID
            "name": str | None,  # 用户命名的名称
            "attributes": {
                "color": str | None,
                "size_category": str | None,  # "small" | "medium" | "large"
                "motion": str | None,  # "static" | "moving"
            },
            "source_images": [str],  # 来源图片
        }
    }
    """

    def __init__(self, cache_dir: str = "data/vision_memory"):
        self.cache_dir = cache_dir
        self.objects: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._temp_counter = 0

        # 配置
        self.max_embedding_history = 50
        self.max_positions = 100
        self.max_source_images = 20
        self.upgrade_threshold = 5  # 出现多少次后自动升级为 persistent

        os.makedirs(self.cache_dir, exist_ok=True)
        self._load()

    # ── 对象管理 ───────────────────────────────────────────

    def get_or_create(self, object_id: str) -> dict:
        """获取已有对象或创建新对象记录"""
        with self._lock:
            if object_id in self.objects:
                return self.objects[object_id]

            obj = {
                "object_id": object_id,
                "status": "temporary",
                "embedding_history": [],
                "first_seen": _now_iso(),
                "last_seen": _now_iso(),
                "appear_count": 0,
                "positions": [],
                "graph_node_id": None,
                "name": None,
                "attributes": {
                    "color": None,
                    "size_category": None,
                    "motion": "static",
                },
                "source_images": [],
            }
            self.objects[object_id] = obj
            self._dirty = True
            return obj

    def record_appearance(
        self,
        object_id: str,
        embedding: np.ndarray,
        position: tuple[float, float] = None,
        source_image: str = None,
        attributes: dict = None,
    ):
        """记录一次对象出现"""
        with self._lock:
            obj = self.get_or_create(object_id)

            obj["appear_count"] += 1
            obj["last_seen"] = _now_iso()

            # Embedding 历史
            emb_list = embedding.flatten().tolist()
            obj["embedding_history"].append(emb_list)
            if len(obj["embedding_history"]) > self.max_embedding_history:
                obj["embedding_history"] = obj["embedding_history"][-self.max_embedding_history:]

            # 位置历史
            if position:
                obj["positions"].append({
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "source": source_image or "",
                })
                if len(obj["positions"]) > self.max_positions:
                    obj["positions"] = obj["positions"][-self.max_positions:]

            # 来源图片
            if source_image and source_image not in obj["source_images"]:
                obj["source_images"].append(source_image)
                if len(obj["source_images"]) > self.max_source_images:
                    obj["source_images"] = obj["source_images"][-self.max_source_images:]

            # 更新属性
            if attributes:
                for k, v in attributes.items():
                    if k in obj["attributes"] and v is not None:
                        obj["attributes"][k] = v

            # 自动升级检查
            if obj["status"] == "temporary" and obj["appear_count"] >= self.upgrade_threshold:
                obj["status"] = "persistent"
                logger.info(f"[ObjectMemory] {object_id} 自动升级为 persistent "
                           f"(出现 {obj['appear_count']} 次)")

            self._dirty = True

    def get_object(self, object_id: str) -> Optional[dict]:
        return self.objects.get(object_id)

    def get_average_embedding(self, object_id: str) -> Optional[np.ndarray]:
        """获取对象的平均 embedding（用于匹配）"""
        obj = self.objects.get(object_id)
        if not obj or not obj["embedding_history"]:
            return None
        # 使用最近 10 个 embedding 的平均
        recent = obj["embedding_history"][-10:]
        return np.mean(np.array(recent, dtype=np.float32), axis=0)

    def confirm_object(self, object_id: str, name: str, graph_node_id: str = None):
        """用户确认对象名称"""
        with self._lock:
            obj = self.get_or_create(object_id)
            obj["name"] = name
            obj["status"] = "persistent"
            if graph_node_id:
                obj["graph_node_id"] = graph_node_id
            self._dirty = True
            logger.info(f"[ObjectMemory] {object_id} 确认为: {name}")

    def link_graph_node(self, object_id: str, graph_node_id: str):
        """关联图谱节点"""
        with self._lock:
            obj = self.get_or_create(object_id)
            obj["graph_node_id"] = graph_node_id
            self._dirty = True

    def set_name(self, object_id: str, name: str):
        """设置对象名称"""
        with self._lock:
            obj = self.get_or_create(object_id)
            obj["name"] = name
            self._dirty = True

    def upgrade_to_persistent(self, object_id: str):
        """手动升级为 persistent"""
        with self._lock:
            obj = self.get_or_create(object_id)
            obj["status"] = "persistent"
            self._dirty = True

    # ── 查询 ───────────────────────────────────────────────

    def list_objects(self, status: str = None) -> list[dict]:
        """列出所有对象（可过滤状态）"""
        with self._lock:
            if status:
                return [o for o in self.objects.values() if o["status"] == status]
            return list(self.objects.values())

    def get_temporary_objects(self) -> list[dict]:
        return self.list_objects("temporary")

    def get_persistent_objects(self) -> list[dict]:
        return self.list_objects("persistent")

    def get_named_objects(self) -> list[dict]:
        with self._lock:
            return [o for o in self.objects.values() if o.get("name")]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total": len(self.objects),
                "temporary": len(self.get_temporary_objects()),
                "persistent": len(self.get_persistent_objects()),
                "named": len(self.get_named_objects()),
            }

    def _next_temp_id(self) -> str:
        self._temp_counter += 1
        return f"UnknownObject{self._temp_counter:04d}"

    # ── 持久化 ─────────────────────────────────────────────

    def save(self):
        path = os.path.join(self.cache_dir, "object_memory.json")
        with self._lock:
            try:
                # 转换为可序列化格式（numpy arrays → lists）
                serializable = {}
                for oid, obj in self.objects.items():
                    sobj = dict(obj)
                    # embedding_history 已经是 list
                    serializable[oid] = sobj

                with open(path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, ensure_ascii=False, indent=2)
                self._dirty = False
                logger.info(f"[ObjectMemory] 保存: {len(self.objects)} 对象")
            except Exception as e:
                logger.warning(f"[ObjectMemory] 保存失败: {e}")

    def _load(self):
        path = os.path.join(self.cache_dir, "object_memory.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.objects = data
            # 恢复 temp_counter
            for oid in data:
                if oid.startswith("UnknownObject"):
                    try:
                        num = int(oid.replace("UnknownObject", ""))
                        self._temp_counter = max(self._temp_counter, num)
                    except ValueError:
                        pass
            logger.info(f"[ObjectMemory] 加载: {len(self.objects)} 对象")
        except Exception as e:
            logger.warning(f"[ObjectMemory] 加载失败: {e}")

    def auto_save_if_dirty(self):
        if self._dirty:
            self.save()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
