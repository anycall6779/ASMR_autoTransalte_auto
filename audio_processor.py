# -*- coding: utf-8 -*-
"""
ASMR 오디오 전처리 모듈 (Termux / CPU-only)
- 스테레오 → 모노 변환
- 노이즈 감소 (noisereduce)
- 정규화
"""

import numpy as np


def load_audio(file_path: str):
    """
    오디오 파일 로드 (WAV / MP3 / FLAC / M4A 등)
    반환: (audio: np.ndarray float32, sample_rate: int)
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(str(file_path), dtype="float32", always_2d=False)
    except Exception:
        import librosa
        audio, sr = librosa.load(str(file_path), sr=None, mono=False)
        audio = np.asarray(audio, dtype=np.float32)

    # 스테레오 / 멀티채널 → 모노 (채널 평균)
    if audio.ndim > 1:
        if audio.shape[0] < audio.shape[1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)

    return audio.astype(np.float32), sr


def apply_denoise(audio: np.ndarray, sr: int, strength: float = 0.55) -> np.ndarray:
    """
    ASMR 전용 노이즈 감소.
    strength: 0.0(감소 없음) ~ 1.0(최대), 기본 0.55
    너무 높으면 속삭임 음성 손상.
    noisereduce / scipy 미설치 시 원본 오디오 반환 (기능 스킵).
    """
    try:
        import noisereduce as nr
        import scipy  # scipy 없으면 noisereduce 런타임 오류 → 조기 차단
    except (ImportError, ModuleNotFoundError):
        return audio  # 노이즈감소 스킵, 앱 정상 동작 유지

    prop_decrease = max(0.0, min(1.0, strength))
    denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=prop_decrease,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        time_mask_smooth_ms=50,
        freq_mask_smooth_hz=500,
    )
    return denoised.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """피크 정규화. ASMR 볼륨 낮은 경우 인식률 향상."""
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val * target_peak
    return audio
