# ear/audio_router.py — 音频类型路由器
# 改编自 "The Ear of Fascinator" router.py
# 将音频分类为 speech / music / sound / mixed

import numpy as np

MIXED_GAP_THRESHOLD = 0.20
DUAL_HIGH_THRESHOLD = 0.50
MAX_RAW_SCORE = 3.0


def classify_audio_type(features: dict) -> dict:
    """
    音频类型路由器（鲁棒版）

    不再返回单一绝对类型，而是计算 Speech / Music / Sound 三类归一化置信度得分，
    并根据「最高分-次高分差距」与「多模态共激活」阈值自动判定 mixed。

    返回:
        {
            "audio_type": "speech" | "music" | "sound" | "mixed",
            "scores": {"speech": 0.xx, "music": 0.xx, "sound": 0.xx},
            "confidence": 0.xx,
            "mixed_reason": "gap" | "dual_high" | None
        }
    """
    energy = features.get("energy", 0)
    zcr = features.get("zcr", 0)
    pitch = features.get("pitch", 0)
    mfcc = np.array(features.get("mfcc", []))

    speech_score = 0
    music_score = 0
    sound_score = 0

    if pitch > 80:
        speech_score += 1
    if energy > 0.015:
        speech_score += 1
    if zcr < 0.1:
        speech_score += 1

    if zcr > 0.1:
        music_score += 1
    if energy > 0.02:
        music_score += 1
    if len(mfcc) > 0 and np.std(mfcc) > 10:
        music_score += 1

    if energy < 0.015:
        sound_score += 2
    if pitch < 60:
        sound_score += 1

    norm_scores = {
        "speech": round(speech_score / MAX_RAW_SCORE, 4),
        "music": round(music_score / MAX_RAW_SCORE, 4),
        "sound": round(sound_score / MAX_RAW_SCORE, 4),
    }

    sorted_items = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_items[0]
    _, second_best_score = sorted_items[1]

    audio_type = best_type
    mixed_reason = None

    gap = best_score - second_best_score

    if gap < MIXED_GAP_THRESHOLD:
        audio_type = "mixed"
        mixed_reason = "gap"
    elif norm_scores["speech"] >= DUAL_HIGH_THRESHOLD and norm_scores["music"] >= DUAL_HIGH_THRESHOLD:
        audio_type = "mixed"
        mixed_reason = "dual_high"

    if audio_type == "mixed":
        confidence = round(max(best_score, 1.0 - gap), 4)
    else:
        confidence = round(best_score, 4)

    return {
        "audio_type": audio_type,
        "scores": norm_scores,
        "confidence": confidence,
        "mixed_reason": mixed_reason,
    }
