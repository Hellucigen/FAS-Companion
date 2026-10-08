# ear/pipeline_sound.py — 环境声分类流水线
# 改编自 "The Ear of Fascinator" pipeline_sound.py
# 使用 YAMNet (AudioSet-521) 进行环境声分类

import os
import csv
import numpy as np
import logging

logger = logging.getLogger(__name__)

_YAMNET_MODEL_DIR = os.environ.get("YAMNET_MODEL_DIR", "E:/Models/yamnet_model")
_YAMNET_CLASS_MAP = os.path.join(_YAMNET_MODEL_DIR, "assets", "yamnet_class_map.csv")

_yamnet_model = None
_class_names = None


def _load_class_names():
    global _class_names
    if _class_names is not None:
        return _class_names
    names = []
    if os.path.exists(_YAMNET_CLASS_MAP):
        with open(_YAMNET_CLASS_MAP, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row.get("display_name", ""))
    else:
        logger.warning(f"[Ear] YAMNet class map not found: {_YAMNET_CLASS_MAP}")
    _class_names = names
    return _class_names


def _load_yamnet_model():
    global _yamnet_model
    if _yamnet_model is not None:
        return _yamnet_model

    import tensorflow as tf

    if os.path.isdir(_YAMNET_MODEL_DIR):
        logger.info(f"[Ear] YAMNet loading from: {_YAMNET_MODEL_DIR}")
        _yamnet_model = tf.saved_model.load(_YAMNET_MODEL_DIR)
    else:
        logger.info("[Ear] YAMNet loading from TensorFlow Hub...")
        import tensorflow_hub as hub
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    logger.info("[Ear] YAMNet loaded")
    return _yamnet_model


def run_sound_pipeline(y: np.ndarray, sr: int) -> list[dict]:
    """
    对音频运行 YAMNet 环境声分类

    Args:
        y: 音频信号 (float32)
        sr: 采样率

    Returns:
        [{start, end, type: "sound", content: {yamnet_class_id, yamnet_class_name, top5}}]
    """
    try:
        model = _load_yamnet_model()
    except Exception as e:
        logger.warning(f"[Ear] YAMNet 加载失败: {e}")
        return []

    class_names = _load_class_names()

    def _name(i: int):
        return class_names[i] if 0 <= i < len(class_names) else ""

    y = np.asarray(y, dtype=np.float32)

    if sr != 16000:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception:
            pass

    import tensorflow as tf

    block_s = 10.0
    block_len = int(sr * block_s)
    n_blocks = int(np.ceil(len(y) / float(block_len))) if block_len > 0 else 1

    events = []
    for bi in range(n_blocks):
        a = bi * block_len
        b = min(len(y), (bi + 1) * block_len)
        chunk = y[a:b]
        if chunk.size <= 0:
            continue

        input_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)
        scores, _, _ = model(input_tensor)
        scores = scores.numpy()
        if scores.ndim == 3:
            scores = scores[0]
        scores = np.mean(scores, axis=0)
        top_class = int(np.argmax(scores))
        top5 = np.argsort(scores)[::-1][:5].tolist()

        events.append({
            "start": float(a / sr),
            "end": float(b / sr),
            "type": "sound",
            "content": {
                "yamnet_class_id": top_class,
                "yamnet_class_name": _name(top_class),
                "top5": [
                    {
                        "yamnet_class_id": int(i),
                        "yamnet_class_name": _name(int(i)),
                        "score": float(scores[int(i)])
                    }
                    for i in top5
                ],
            },
        })

    return events
