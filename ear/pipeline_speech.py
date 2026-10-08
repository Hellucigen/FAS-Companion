# ear/pipeline_speech.py — 语音识别流水线
# 改编自 "The Ear of Fascinator" pipeline_speech.py
# 使用 FunASR Paraformer (ASR) + emotion2vec+ (情绪识别)

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

_ASR_MODEL_DIR = os.environ.get("ASR_MODEL_DIR", "E:/Models/Paraformer")
_EMO_MODEL_DIR = os.environ.get("EMO_MODEL_DIR", "E:/Models/emotion2vec_plus_large")

_asr_model = None
_emo_model = None


def _get_device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    device = _get_device()
    try:
        from funasr import AutoModel
        _asr_model = AutoModel(
            model=_ASR_MODEL_DIR,
            device=device,
            disable_update=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        logger.info(f"[Ear] ASR model loaded (device={device})")
    except Exception as e:
        logger.warning(f"[Ear] ASR model load failed: {e}")
        _asr_model = None
    return _asr_model


def _get_emo_model():
    global _emo_model
    if _emo_model is not None:
        return _emo_model

    device = _get_device()
    try:
        from funasr import AutoModel
        _emo_model = AutoModel(
            model=_EMO_MODEL_DIR,
            device=device,
            disable_update=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        logger.info(f"[Ear] Emotion model loaded (device={device})")
    except Exception as e:
        logger.warning(f"[Ear] Emotion model load failed: {e}")
        _emo_model = None
    return _emo_model


def _parse_funasr_text(res):
    if not res:
        return ""
    item = res[0] if isinstance(res, list) else res
    if isinstance(item, dict):
        return str(item.get("text", "")).strip()
    return str(item).strip()


def _majority_vote(labels: list[str]) -> str:
    if not labels:
        return ""
    counts = {}
    for x in labels:
        x = str(x).strip()
        if not x:
            continue
        counts[x] = counts.get(x, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


def run_speech_pipeline(
    y: np.ndarray,
    sr: int,
    enable_diarization: bool = False,
    diarization_model_dir: str | None = None,
    max_segment_s: float = 15.0,
    overlap_s: float = 1.0,
    emotion_subsegment_s: float = 8.0,
) -> list[dict]:
    """
    对音频运行语音识别 + 情绪识别

    Args:
        y: 音频信号
        sr: 采样率
        enable_diarization: 是否启用说话人日志
        diarization_model_dir: 说话人模型目录
        max_segment_s: 最大分段长度
        overlap_s: 上下文重叠
        emotion_subsegment_s: 情绪识别子段长度

    Returns:
        [{start, end, type: "speech", speaker_id, content: {text}, emotion}]
    """
    y = np.asarray(y, dtype=np.float32)

    asr_model = _get_asr_model()
    if asr_model is None:
        logger.warning("[Ear] ASR 模型不可用，跳过语音处理")
        return []

    emo_model = _get_emo_model()

    # 简单 VAD 分段
    segments = _simple_vad_segments(y, sr)
    if not segments:
        logger.info("[Ear] 未检测到语音段")
        return []

    # 分割长段
    segments = _split_long_segments(segments, max_segment_s)
    logger.info(f"[Ear] 语音段数: {len(segments)}")

    events = []
    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        ctx = max(0.0, float(overlap_s) / 2.0)
        ss = max(0.0, s - ctx)
        ee = min(float(len(y) / sr), e + ctx)
        a = int(max(0, ss * sr))
        b = int(min(len(y), ee * sr))
        chunk = y[a:b]

        # ASR
        try:
            asr_res = asr_model.generate(input=chunk)
            text = _parse_funasr_text(asr_res)
        except Exception as exc:
            logger.warning(f"[Ear] ASR 失败: {exc}")
            text = ""

        # 情绪识别
        emotion = ""
        if emo_model is not None and text:
            try:
                emo_labels = []
                if emotion_subsegment_s and (e - s) > emotion_subsegment_s:
                    t = s
                    while t < e - 1e-6:
                        sub_s = float(t)
                        sub_e = float(min(e, t + emotion_subsegment_s))
                        aa = int(max(0, sub_s * sr))
                        bb = int(min(len(y), sub_e * sr))
                        sub = y[aa:bb]
                        emo_res = emo_model.generate(input=sub)
                        emo_labels.append(_parse_funasr_text(emo_res))
                        t += emotion_subsegment_s
                else:
                    emo_res = emo_model.generate(input=chunk)
                    emo_labels.append(_parse_funasr_text(emo_res))
                emotion = _majority_vote(emo_labels)
            except Exception as exc:
                logger.debug(f"[Ear] 情绪识别失败: {exc}")

        events.append({
            "start": s,
            "end": e,
            "type": "speech",
            "speaker_id": seg.get("speaker_id", "spk0"),
            "content": {"text": text},
            "emotion": emotion,
        })

    return events


def _simple_vad_segments(
    y: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    energy_thresh: float = 0.004,
    min_speech_ms: int = 300,
    min_silence_ms: int = 200,
) -> list[dict]:
    """基于能量的简单 VAD"""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len
    n_frames = max(1, (len(y) - frame_len + 1) // hop_len)

    voiced = []
    for i in range(n_frames):
        a = i * hop_len
        frame = y[a: a + frame_len]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        if rms > energy_thresh:
            start_t = a / sr
            end_t = (a + frame_len) / sr
            voiced.append((start_t, end_t))

    if not voiced:
        return []

    # 合并相邻段
    merged = []
    cur_s, cur_e = voiced[0]
    for s, e in voiced[1:]:
        if s - cur_e <= (min_silence_ms / 1000.0):
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    min_speech_s = min_speech_ms / 1000.0
    merged = [(s, e) for s, e in merged if (e - s) >= min_speech_s]

    return [{"start": float(s), "end": float(e), "speaker_id": "spk0"} for s, e in merged]


def _split_long_segments(segments: list[dict], max_len_s: float) -> list[dict]:
    """将过长段分割"""
    out = []
    for seg in segments:
        s0 = float(seg["start"])
        e0 = float(seg["end"])
        spk = seg.get("speaker_id", "spk0")
        if max_len_s <= 0 or (e0 - s0) <= max_len_s:
            out.append({"start": s0, "end": e0, "speaker_id": spk})
            continue
        t = s0
        while t < e0 - 1e-6:
            out.append({"start": float(t), "end": float(min(e0, t + max_len_s)), "speaker_id": spk})
            t += max_len_s
    return out
