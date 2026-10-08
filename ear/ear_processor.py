# ear/ear_processor.py — Ear 听觉感知主处理器
# ============================================================================
# 职责：
#   1. 接收音频输入（文件路径或 numpy 数组）
#   2. 路由分类（语音/音乐/环境声/混合）
#   3. 调度各子流水线处理
#   4. 将结果注入 Fascinator 知识图谱
#   5. 支持启用/禁用切换
# ============================================================================

import os
import json
import logging
import threading
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# 懒加载子模块
_audio_router = None
_feature_extractor = None
_sound_pipeline = None
_speech_pipeline = None


def _lazy_import(module_name: str):
    """懒加载 ear 子模块，避免启动时导入重型依赖"""
    import importlib
    if module_name == "audio_router":
        from . import audio_router as m
        return m
    elif module_name == "feature_extractor":
        from . import feature_extractor as m
        return m
    elif module_name == "sound_pipeline":
        from . import pipeline_sound as m
        return m
    elif module_name == "speech_pipeline":
        from . import pipeline_speech as m
        return m
    raise ValueError(f"Unknown module: {module_name}")


class EarProcessor:
    """Ear 听觉感知处理器

    设计原则：
    - 不修改核心认知架构（KnowledgeGraph / DiffusionEngine / NLP Pipeline）
    - 处理结果为结构化事件列表，由外部调用者决定如何注入图谱
    - 默认不启用，需显式调用 enable()
    """

    def __init__(self):
        self._enabled = False
        self._lock = threading.RLock()
        self._last_result = None
        self._process_count = 0

        # 模型路径配置
        self._yamnet_model_dir = os.environ.get(
            "YAMNET_MODEL_DIR", "E:/Models/yamnet_model"
        )
        self._asr_model_dir = os.environ.get(
            "ASR_MODEL_DIR", "E:/Models/Paraformer"
        )
        self._emo_model_dir = os.environ.get(
            "EMO_MODEL_DIR", "E:/Models/emotion2vec_plus_large"
        )
        self._speaker_model_dir = os.environ.get(
            "SPEAKER_MODEL_DIR", None
        )

    # ── 启用/禁用 ──────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        with self._lock:
            self._enabled = True
            logger.info("[Ear] 听觉感知已启用")

    def disable(self):
        with self._lock:
            self._enabled = False
            logger.info("[Ear] 听觉感知已禁用")

    # ── 状态 ───────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "process_count": self._process_count,
            "has_last_result": self._last_result is not None,
            "models": {
                "yamnet": os.path.isdir(self._yamnet_model_dir),
                "asr": os.path.isdir(self._asr_model_dir) if self._asr_model_dir else False,
                "emotion": os.path.isdir(self._emo_model_dir) if self._emo_model_dir else False,
            }
        }

    # ── 主处理流程 ─────────────────────────────────────────

    def process_file(
        self,
        audio_path: str,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """处理音频文件，返回结构化感知结果

        Args:
            audio_path: 音频文件路径 (wav/mp3/etc.)
            progress_callback: 可选进度回调 (phase, progress)

        Returns:
            {
                "audio_type": "speech" | "music" | "sound" | "mixed",
                "timeline": [...],   # 时间线事件列表
                "summary": {...},    # 摘要信息
            }
        """
        import time
        t0 = time.time()

        if not self._enabled:
            logger.warning("[Ear] 听觉感知未启用，跳过处理")
            return {"error": "Ear 听觉感知未启用", "enabled": False}

        if not os.path.exists(audio_path):
            return {"error": f"音频文件不存在: {audio_path}"}

        logger.info(f"[Ear] 开始处理: {audio_path}")

        try:
            # Step 1: 加载音频
            if progress_callback:
                progress_callback("load", 0.1)

            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            logger.info(f"[Ear] 音频加载完成: sr={sr}, dur={len(y)/sr:.2f}s")

            # Step 2: 特征提取
            if progress_callback:
                progress_callback("features", 0.2)

            feat_mod = _lazy_import("feature_extractor")
            features = feat_mod.extract_features(y, sr)

            # Step 3: 音频类型分类
            if progress_callback:
                progress_callback("classify", 0.3)

            router_mod = _lazy_import("audio_router")
            type_result = router_mod.classify_audio_type(features)
            audio_type = type_result["audio_type"]
            logger.info(f"[Ear] 音频类型: {audio_type} (confidence={type_result['confidence']})")

            # Step 4: 按类型调度流水线
            speech_events = []
            sound_events = []

            if progress_callback:
                progress_callback("pipeline", 0.4)

            # 语音处理
            if audio_type in ("speech", "mixed"):
                try:
                    speech_mod = _lazy_import("speech_pipeline")
                    speech_events = speech_mod.run_speech_pipeline(
                        y, sr,
                        enable_diarization=(audio_type == "mixed"),
                        diarization_model_dir=self._speaker_model_dir,
                    )
                    logger.info(f"[Ear] 语音事件: {len(speech_events)}")
                except Exception as e:
                    logger.warning(f"[Ear] 语音处理失败: {e}")

            # 环境声处理
            if audio_type in ("sound", "mixed"):
                try:
                    sound_mod = _lazy_import("sound_pipeline")
                    sound_events = sound_mod.run_sound_pipeline(y, sr)
                    logger.info(f"[Ear] 环境声事件: {len(sound_events)}")
                except Exception as e:
                    logger.warning(f"[Ear] 环境声处理失败: {e}")

            # Step 5: 构建时间线
            timeline = _build_timeline(speech_events, sound_events)

            # Step 6: 生成摘要
            summary = _generate_summary(audio_type, timeline, features)

            if progress_callback:
                progress_callback("done", 1.0)

            result = {
                "audio_type": audio_type,
                "type_confidence": type_result.get("confidence", 0),
                "type_scores": type_result.get("scores", {}),
                "duration_s": len(y) / sr,
                "timeline": timeline,
                "summary": summary,
                "features": {
                    "energy": features.get("energy", 0),
                    "pitch": features.get("pitch", 0),
                    "zcr": features.get("zcr", 0),
                },
                "process_time_s": round(time.time() - t0, 2),
            }

            with self._lock:
                self._last_result = result
                self._process_count += 1

            logger.info(f"[Ear] 处理完成 (耗时: {result['process_time_s']}s)")
            return result

        except Exception as e:
            logger.exception(f"[Ear] 处理异常: {e}")
            return {"error": str(e), "enabled": self._enabled}

    def process_array(
        self,
        y: np.ndarray,
        sr: int,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """直接处理 numpy 音频数组"""
        import time
        t0 = time.time()

        if not self._enabled:
            return {"error": "Ear 听觉感知未启用", "enabled": False}

        try:
            feat_mod = _lazy_import("feature_extractor")
            features = feat_mod.extract_features(y, sr)

            router_mod = _lazy_import("audio_router")
            type_result = router_mod.classify_audio_type(features)
            audio_type = type_result["audio_type"]

            speech_events = []
            sound_events = []

            if audio_type in ("speech", "mixed"):
                try:
                    speech_mod = _lazy_import("speech_pipeline")
                    speech_events = speech_mod.run_speech_pipeline(y, sr)
                except Exception as e:
                    logger.warning(f"[Ear] 语音处理失败: {e}")

            if audio_type in ("sound", "mixed"):
                try:
                    sound_mod = _lazy_import("sound_pipeline")
                    sound_events = sound_mod.run_sound_pipeline(y, sr)
                except Exception as e:
                    logger.warning(f"[Ear] 环境声处理失败: {e}")

            timeline = _build_timeline(speech_events, sound_events)
            summary = _generate_summary(audio_type, timeline, features)

            result = {
                "audio_type": audio_type,
                "type_confidence": type_result.get("confidence", 0),
                "type_scores": type_result.get("scores", {}),
                "duration_s": len(y) / sr,
                "timeline": timeline,
                "summary": summary,
                "features": {
                    "energy": features.get("energy", 0),
                    "pitch": features.get("pitch", 0),
                    "zcr": features.get("zcr", 0),
                },
                "process_time_s": round(time.time() - t0, 2),
            }

            with self._lock:
                self._last_result = result
                self._process_count += 1

            return result

        except Exception as e:
            logger.exception(f"[Ear] 处理异常: {e}")
            return {"error": str(e), "enabled": self._enabled}

    def get_last_result(self) -> Optional[dict]:
        return self._last_result


# ── 辅助函数 ──────────────────────────────────────────────

def _build_timeline(speech_events, sound_events):
    """合并多个事件列表为统一时间线"""
    events = []
    if speech_events:
        events.extend(speech_events)
    if sound_events:
        events.extend(sound_events)
    events.sort(key=lambda x: x.get("start", 0))
    return events


def _generate_summary(audio_type: str, timeline: list, features: dict) -> dict:
    """从时间线生成人类可读摘要"""
    summary = {
        "audio_type": audio_type,
        "event_count": len(timeline),
    }

    # 收集语音文本
    speech_texts = []
    for ev in timeline:
        if ev.get("type") == "speech":
            text = ev.get("content", {}).get("text", "")
            if text:
                speech_texts.append(text)

    if speech_texts:
        summary["transcript"] = " ".join(speech_texts)
        summary["speech_segments"] = len(speech_texts)

    # 收集环境声类别
    sound_classes = {}
    for ev in timeline:
        if ev.get("type") == "sound":
            cname = ev.get("content", {}).get("yamnet_class_name", "")
            if cname:
                sound_classes[cname] = sound_classes.get(cname, 0) + 1

    if sound_classes:
        top_sounds = sorted(sound_classes.items(), key=lambda x: x[1], reverse=True)[:5]
        summary["top_sounds"] = [{"class": name, "count": cnt} for name, cnt in top_sounds]

    # 情绪分布
    emotions = {}
    for ev in timeline:
        emo = ev.get("emotion", "")
        if emo:
            emotions[emo] = emotions.get(emo, 0) + 1
    if emotions:
        summary["emotions"] = emotions

    return summary


# ── 全局单例 ──────────────────────────────────────────────

_ear_processor: Optional[EarProcessor] = None


def get_ear_processor() -> EarProcessor:
    global _ear_processor
    if _ear_processor is None:
        _ear_processor = EarProcessor()
    return _ear_processor
