# vision/vision_graph_bridge.py — 视觉对象到知识图谱的桥梁
# ============================================================================
# 将视觉对象注入 Fascinator 知识图谱：
# - 新增节点类型: object-instance (label="declarative-semantic")
# - 新增关系: instance_of, has_attribute, located_at, appears_with
# - 视觉产生的信息与文本输入具有同等地位
# ============================================================================

import logging
from typing import Optional

from graph_model import KnowledgeGraph, Node, Edge, now_str

logger = logging.getLogger(__name__)


class VisionGraphBridge:
    """视觉-图谱桥接器

    职责：
    1. 为视觉对象创建/更新图谱节点
    2. 建立对象间关系
    3. 将对象激活注入扩散引擎
    """

    def __init__(self, kg: KnowledgeGraph, engine=None):
        self.kg = kg
        self.engine = engine  # DiffusionEngine 引用，用于注入激活

    # ── 节点管理 ───────────────────────────────────────────

    def ensure_object_node(
        self,
        object_id: str,
        name: str = None,
        attributes: dict = None,
        status: str = "temporary",
    ) -> Node:
        """确保视觉对象在图谱中有对应节点

        Args:
            object_id: 视觉对象 ID (如 UnknownObject0001)
            name: 用户命名的名称（可选）
            attributes: 低级属性 dict
            status: "temporary" | "persistent"

        Returns:
            Node 对象
        """
        attributes = attributes or {}

        node = self.kg.get_node(object_id)
        if node is None:
            # 创建新节点
            node = Node(
                id=object_id,
                weight=0.3 if status == "temporary" else 0.5,
                label="declarative-semantic",
                confidence=0.3 if status == "temporary" else 0.6,
                extra_attrs={
                    "object_type": "visual-instance",
                    "status": status,
                    "name": name,
                    "color": attributes.get("color"),
                    "size_category": attributes.get("size_category"),
                    "motion": attributes.get("motion"),
                },
            )
            self.kg.add_node(node)
            logger.info(f"[VisionBridge] 创建对象节点: {object_id} (status={status})")

            # 更新引擎名称索引
            if self.engine:
                with self.engine._lock:
                    self.engine.name_to_node[object_id] = node
        else:
            # 更新已有节点
            if name:
                node.extra_attrs["name"] = name
            if status == "persistent":
                node.extra_attrs["status"] = "persistent"
                node.weight = max(node.weight, 0.5)
                node.confidence = max(node.confidence, 0.6)
            for k, v in attributes.items():
                if v is not None:
                    node.extra_attrs[k] = v
            node.touch()

        return node

    def ensure_concept_node(self, concept_name: str) -> Node:
        """确保概念节点存在（如"狗"、"树"等用户确认的类别）"""
        node = self.kg.get_node(concept_name)
        if node is None:
            node = Node(
                id=concept_name,
                weight=0.6,
                label="declarative-semantic",
                confidence=0.7,
                extra_attrs={"category": "concept"},
            )
            self.kg.add_node(node)
            logger.info(f"[VisionBridge] 创建概念节点: {concept_name}")

            if self.engine:
                with self.engine._lock:
                    self.engine.name_to_node[concept_name] = node
        return node

    # ── 关系管理 ───────────────────────────────────────────

    def link_instance_to_concept(self, object_id: str, concept_name: str):
        """建立 instance_of 关系: Object → 概念"""
        self.ensure_object_node(object_id)
        concept_node = self.ensure_concept_node(concept_name)

        # 检查边是否已存在
        if self.kg.get_edge(object_id, concept_name, "instance_of"):
            return

        edge = Edge(
            src=object_id,
            dst=concept_name,
            relation="instance_of",
            weight=0.8,
        )
        self.kg.add_edge(edge)
        logger.info(f"[VisionBridge] {object_id} instance_of {concept_name}")

    def link_attribute(self, object_id: str, attr_name: str, attr_value: str):
        """建立 has_attribute 关系"""
        self.ensure_object_node(object_id)

        # 确保属性值节点存在
        attr_node_id = attr_value
        if self.kg.get_node(attr_node_id) is None:
            node = Node(
                id=attr_node_id,
                weight=0.4,
                label="declarative-semantic",
                extra_attrs={"category": "attribute", "attr_name": attr_name},
            )
            self.kg.add_node(node)

        if not self.kg.get_edge(object_id, attr_node_id, "has_attribute"):
            edge = Edge(
                src=object_id,
                dst=attr_node_id,
                relation="has_attribute",
                weight=0.6,
            )
            self.kg.add_edge(edge)

    def link_appears_with(self, obj_a: str, obj_b: str):
        """建立 appears_with 关系（两个对象同时出现）"""
        if self.kg.get_edge(obj_a, obj_b, "appears_with"):
            return

        edge = Edge(
            src=obj_a,
            dst=obj_b,
            relation="appears_with",
            weight=0.4,
        )
        self.kg.add_edge(edge)

    # ── 激活注入 ───────────────────────────────────────────

    def activate_object(self, object_id: str, activation: float = 0.3):
        """激活视觉对象节点，使其进入扩散引擎"""
        node = self.kg.get_node(object_id)
        if node is None:
            return

        node.activation = max(node.activation, activation)
        node.touch()
        if self.engine is not None:
            self.engine.mark_active([node.id])  # 直写激活 → 同步活跃前沿

        logger.debug(f"[VisionBridge] 激活对象: {object_id} (act={activation})")

    def activate_concept(self, concept_name: str, activation: float = 0.3):
        """激活概念节点"""
        node = self.kg.get_node(concept_name)
        if node is None:
            return

        node.activation = max(node.activation, activation)
        node.touch()
        if self.engine is not None:
            self.engine.mark_active([node.id])

    # ── 用户确认 ───────────────────────────────────────────

    def confirm_object(self, object_id: str, name: str, attributes: dict = None):
        """用户确认可视化对象是什么

        1. 创建概念节点（如"狗"）
        2. 建立 instance_of 关系
        3. 更新对象节点属性
        4. 升级为 persistent
        """
        self.ensure_object_node(object_id, name=name, status="persistent")
        self.link_instance_to_concept(object_id, name)

        if attributes:
            for k, v in attributes.items():
                if isinstance(v, str):
                    self.link_attribute(object_id, k, v)

        # 激活
        self.activate_object(object_id, 0.5)
        self.activate_concept(name, 0.5)

        logger.info(f"[VisionBridge] 用户确认: {object_id} = {name}")

    # ── 批量处理 ───────────────────────────────────────────

    def process_vision_result(
        self,
        objects: list[dict],
        source_image: str = None,
    ) -> dict:
        """处理视觉结果，将对象注入图谱

        Args:
            objects: [{object_id, matched_object, attributes, ...}]
            source_image: 来源图片路径

        Returns:
            {created: [str], activated: [str], linked: [(str, str)]}
        """
        created = []
        activated = []
        linked = []

        node_ids = [obj.get("object_id") for obj in objects if obj.get("object_id")]

        for obj in objects:
            oid = obj.get("object_id")
            if not oid:
                continue

            attrs = obj.get("attributes", {})
            status = obj.get("status", "temporary")

            # 确保节点存在
            node = self.ensure_object_node(oid, attributes=attrs, status=status)
            if node:
                created.append(oid)

            # 注入激活
            self.activate_object(oid, 0.3)
            activated.append(oid)

            # 如果有匹配对象，建立关系
            matched = obj.get("matched_object")
            if matched and matched != oid:
                self.link_appears_with(oid, matched)
                linked.append((oid, matched))

            # 属性连接
            for attr_key in ["color", "size_category"]:
                val = attrs.get(attr_key)
                if val:
                    self.link_attribute(oid, attr_key, str(val))

        # 同帧对象建立 appears_with 关系
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = node_ids[i], node_ids[j]
                if a and b:
                    self.link_appears_with(a, b)

        return {
            "created": created,
            "activated": activated,
            "linked": linked,
        }
