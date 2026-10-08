# vision/__init__.py — Vision 视觉感知模块
# ============================================================================
# 不依赖 LLM 的视觉对象发现与持续追踪系统。
#
# 管线：
#   Image/Video Frame
#     → Object Discovery (SAM2)
#     → Object Embedding (SigLIP/DINOv2)
#     → Object Matching (FAISS)
#     → Persistent Object Memory
#     → Knowledge Graph
#
# 设计原则：
#   - 不修改核心认知架构
#   - 不让 LLM 负责视觉识别
#   - 不直接生成图片描述
#   - 视觉输出最终进入知识图谱
# ============================================================================

from .vision_processor import VisionProcessor, get_vision_processor

__all__ = ["VisionProcessor", "get_vision_processor"]
