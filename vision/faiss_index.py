# vision/faiss_index.py — 视觉对象 FAISS 向量索引
# ============================================================================
# 管理所有历史 Object Embedding 的向量数据库。
# 新对象 → Embedding → FAISS 搜索 → 相似度计算 → 匹配已有对象或创建新对象。
# ============================================================================

import os
import json
import logging
import threading
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# faiss-cpu 的 Windows wheel 不含 AVX2 版 SWIG 模块，import 时的"尝试 AVX2 →
# 回退"两步 INFO 会伪装成故障（embedding_manager.py 有同一处理）。回退后的
# 构建功能完整，本索引只存少量对象向量，AVX2 与否无感。
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


class VisionFAISSIndex:
    """视觉对象向量数据库

    使用 FAISS IndexFlatIP（内积搜索，配合 L2 归一化 = 余弦相似度）。
    每个 entry 存储 object_id → embedding 映射。
    """

    def __init__(self, dim: int = 768, cache_dir: str = "data/vision_faiss"):
        self.dim = dim
        self.cache_dir = cache_dir
        self.index = None
        self.id_to_idx: dict[str, int] = {}   # object_id → FAISS 索引位置
        self.idx_to_id: list[str] = []         # 逆映射
        self._lock = threading.RLock()
        self._dirty = False

        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_index()
        self._load_cache()

    def _init_index(self):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dim)
            logger.info(f"[VisionFAISS] IndexFlatIP 初始化 (dim={self.dim})")
        except ImportError:
            logger.error("[VisionFAISS] faiss-cpu 未安装！")
            self.index = None

    @property
    def size(self) -> int:
        return len(self.idx_to_id)

    def add(self, object_id: str, embedding: np.ndarray) -> bool:
        """添加或更新对象的 embedding"""
        if self.index is None:
            return False

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # L2 归一化（用于余弦相似度）
        norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
        embedding = embedding / norm

        with self._lock:
            if object_id in self.id_to_idx:
                # 更新已有：FAISS 不支持直接更新，用重建策略
                # 标记旧位置为无效（用零向量覆盖），然后添加新的
                old_idx = self.id_to_idx[object_id]
                self.index.add(np.zeros((1, self.dim), dtype=np.float32))  # dummy
                # 重新添加
                self.index.add(embedding)
                new_idx = self.index.ntotal - 1
                self.id_to_idx[object_id] = new_idx
                # 更新逆映射
                while len(self.idx_to_id) <= new_idx:
                    self.idx_to_id.append(None)
                self.idx_to_id[new_idx] = object_id
            else:
                self.index.add(embedding)
                idx = self.index.ntotal - 1
                self.id_to_idx[object_id] = idx
                while len(self.idx_to_id) <= idx:
                    self.idx_to_id.append(None)
                self.idx_to_id[idx] = object_id

            self._dirty = True

        logger.debug(f"[VisionFAISS] add/update: {object_id} (total={self.size})")
        return True

    def search(self, embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """搜索最相似的 k 个对象

        Returns:
            [(object_id, similarity_score), ...] 按相似度降序
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # L2 归一化
        norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
        embedding = embedding / norm

        with self._lock:
            distances, indices = self.index.search(embedding, min(k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.idx_to_id):
                obj_id = self.idx_to_id[idx]
                if obj_id:
                    # IndexFlatIP 返回内积，归一化后 = 余弦相似度
                    results.append((obj_id, float(dist)))

        return results

    def remove(self, object_id: str):
        """移除对象（标记为 None）"""
        with self._lock:
            if object_id in self.id_to_idx:
                idx = self.id_to_idx.pop(object_id)
                if idx < len(self.idx_to_id):
                    self.idx_to_id[idx] = None
                self._dirty = True

    def get_embedding(self, object_id: str) -> Optional[np.ndarray]:
        """获取已存储的 embedding（用于重建）"""
        # FAISS 不直接支持按 ID 获取向量，需从缓存读取
        cache_path = os.path.join(self.cache_dir, f"{object_id}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None

    def save(self):
        """持久化索引和映射"""
        if self.index is None:
            return

        import faiss

        idx_path = os.path.join(self.cache_dir, "faiss.index")
        map_path = os.path.join(self.cache_dir, "id_mapping.json")

        with self._lock:
            try:
                faiss.write_index(self.index, idx_path)
                mapping = {
                    "id_to_idx": self.id_to_idx,
                    "idx_to_id": self.idx_to_id,
                    "dim": self.dim,
                }
                with open(map_path, "w", encoding="utf-8") as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
                self._dirty = False
                logger.info(f"[VisionFAISS] 保存: {self.index.ntotal} 向量")
            except Exception as e:
                logger.warning(f"[VisionFAISS] 保存失败: {e}")

    def _load_cache(self):
        idx_path = os.path.join(self.cache_dir, "faiss.index")
        map_path = os.path.join(self.cache_dir, "id_mapping.json")

        if not os.path.exists(idx_path) or not os.path.exists(map_path):
            return

        import faiss

        try:
            self.index = faiss.read_index(idx_path)
            with open(map_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            self.id_to_idx = mapping.get("id_to_idx", {})
            self.idx_to_id = mapping.get("idx_to_id", [])
            self.dim = mapping.get("dim", self.dim)
            logger.info(f"[VisionFAISS] 加载缓存: {self.index.ntotal} 向量")
        except Exception as e:
            logger.warning(f"[VisionFAISS] 加载缓存失败: {e}")
            self._init_index()

    def rebuild_from_memory(self, memory: dict[str, np.ndarray]):
        """从 object_memory 重建索引（清理无效条目后）"""
        import faiss

        if not memory:
            return

        embeddings = []
        ids = []
        for obj_id, emb in memory.items():
            emb = np.asarray(emb, dtype=np.float32).flatten()
            embeddings.append(emb)
            ids.append(obj_id)

        if not embeddings:
            return

        embeddings = np.stack(embeddings, axis=0)
        # L2 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

        with self._lock:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            self.id_to_idx = {oid: i for i, oid in enumerate(ids)}
            self.idx_to_id = list(ids)
            self.dim = embeddings.shape[1]
            self._dirty = True

        logger.info(f"[VisionFAISS] 重建索引: {len(ids)} 对象")
