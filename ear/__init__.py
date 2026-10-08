# ear/__init__.py — Ear 听觉感知模块
# ============================================================================
# 从 "The Ear of Fascinator" 结课设计改造而来。
# 作为 Fascinator 认知系统的可选感知通道，处理音频输入并注入知识图谱。
#
# 启用方式：
#   1. 前端设置面板中开启 "Ear 听觉感知"
#   2. POST /api/ear/process 上传音频文件
#   3. 结果自动注入图谱
# ============================================================================

from .ear_processor import EarProcessor, get_ear_processor

__all__ = ["EarProcessor", "get_ear_processor"]
