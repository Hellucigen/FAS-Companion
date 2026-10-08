# ear/feature_extractor.py — 声学特征提取
# 改编自 "The Ear of Fascinator" feature_extractor.py
# 提取 RMS能量、过零率、基频、MFCC、频谱特征

import numpy as np
import logging

logger = logging.getLogger(__name__)


def _estimate_pitch_autocorr(y: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 300.0) -> float:
    """自相关法基频估计"""
    if y.size < sr * 0.2:
        return 0.0

    y = y - float(np.mean(y))
    y = y.astype(np.float32, copy=False)

    n = int(2 ** np.ceil(np.log2(y.size)))
    yf = np.fft.rfft(y, n=n)
    ac = np.fft.irfft(np.abs(yf) ** 2)

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    max_lag = min(max_lag, ac.size - 1)

    if max_lag <= min_lag + 1:
        return 0.0

    seg = ac[min_lag:max_lag]
    if not np.any(np.isfinite(seg)):
        return 0.0

    lag = int(np.argmax(seg)) + min_lag
    if lag <= 0:
        return 0.0
    return float(sr / lag)


def _frame_stats(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 1024):
    """频谱质心 & 频谱带宽"""
    if y.size < frame_length:
        return 0.0, 0.0
    win = np.hanning(frame_length).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sr).astype(np.float32)
    total_centroid = 0.0
    total_bw = 0.0
    n = 0
    for a in range(0, y.size - frame_length + 1, hop_length):
        frame = y[a: a + frame_length] * win
        spec = np.abs(np.fft.rfft(frame)) ** 2
        s = float(np.sum(spec))
        if s <= 0:
            continue
        centroid = float(np.sum(freqs * spec) / s)
        bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / s))
        total_centroid += centroid
        total_bw += bw
        n += 1
    if n <= 0:
        return 0.0, 0.0
    return total_centroid / n, total_bw / n


def extract_features(
    y: np.ndarray,
    sr: int,
    pitch_window_s: float = 2.0,
    mfcc_chunk_s: float = 3.0,
) -> dict:
    """
    提取声学特征用于路由分类 + LLM 上下文

    Returns:
        {
            "energy": float,
            "zcr": float,
            "pitch": float,
            "mfcc": list[float] (13维),
            "spectral_centroid": float,
            "spectral_bandwidth": float,
        }
    """
    y = np.asarray(y, dtype=np.float32)

    # RMS Energy
    energy = float(np.sqrt(np.mean(y * y) + 1e-12))

    # Zero Crossing Rate
    zcr = float(np.mean((y[:-1] * y[1:]) < 0))

    # Pitch (自相关法)
    win_len = int(sr * float(pitch_window_s))
    n_win = int(np.ceil(len(y) / float(win_len))) if win_len > 0 else 1

    pitch_vals = []
    for wi in range(n_win):
        a = wi * win_len
        b = min(len(y), (wi + 1) * win_len)
        p = _estimate_pitch_autocorr(y[a:b], sr)
        if p > 0:
            pitch_vals.append(p)
    pitch = float(np.mean(pitch_vals)) if pitch_vals else 0.0

    # MFCC (13维)
    mfcc_means = []
    chunk_len = int(sr * float(mfcc_chunk_s))
    n_chunks = int(np.ceil(len(y) / float(chunk_len))) if chunk_len > 0 else 1

    try:
        import torch
        import torchaudio

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mfcc_tf = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40},
        ).to(device)

        for ci in range(n_chunks):
            a = ci * chunk_len
            b = min(len(y), (ci + 1) * chunk_len)
            chunk = y[a:b]
            if chunk.size <= 0:
                continue
            waveform = torch.from_numpy(chunk).unsqueeze(0).to(device)
            mfcc_t = mfcc_tf(waveform).squeeze(0)
            mfcc_means.append(mfcc_t.mean(dim=-1).detach().cpu().numpy().astype(np.float32))
    except Exception as e:
        logger.debug(f"[Ear] MFCC 提取跳过: {e}")

    mfcc_mean = (
        np.mean(np.stack(mfcc_means, axis=0), axis=0).astype(np.float32).tolist()
        if mfcc_means
        else [0.0] * 13
    )

    # Spectral features
    spectral_centroid, spectral_bandwidth = _frame_stats(y, sr)

    return {
        "energy": float(energy),
        "zcr": float(zcr),
        "pitch": float(pitch),
        "mfcc": mfcc_mean,
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
    }
