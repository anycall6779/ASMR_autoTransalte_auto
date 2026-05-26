# -*- coding: utf-8 -*-
"""
ASMR 음성 인식 모듈 (Termux / CPU-only)
- faster-whisper (우선) / openai-whisper (폴백)
- GPU 없이 항상 CPU int8 실행
- 기본 모델: small (모바일 메모리 배려)
"""

import os
import tempfile
import numpy as np
from pathlib import Path


def seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


ASMR_PROMPTS = {
    "ja": "ASMRの囁き声。ゆっくりとした丁寧な話し方。雑音が混じることがあります。",
    "ko": "ASMR 속삭임 음성. 천천히 부드럽게 말하는 방식. 배경 잡음이 있을 수 있습니다.",
    "en": "ASMR whispering voice. Slow and gentle speech. Background noise may be present.",
    "zh": "ASMR耳语声音。缓慢温柔的说话方式。可能有背景噪音。",
}


def _ensure_numba_mock():
    """
    openai-whisper의 timing.py가 numba를 무조건 import함.
    numba 미설치 시 mock으로 대체하여 import 성공.
    word_timestamps=False 사용 시 numba JIT 실제 호출 없음.
    """
    import sys
    try:
        import numba  # noqa
    except ImportError:
        class _JitDecorator:
            def __call__(self, *a, **kw):
                def dec(fn): return fn
                return dec
            def __getattr__(self, name):
                return self
        _jit = _JitDecorator()
        mock = type('numba', (), {
            'jit': _jit,
            'float32': float,
            'int32': int,
            'int64': int,
        })()
        sys.modules.setdefault('numba', mock)
        sys.modules.setdefault('numba.core', mock)
        sys.modules.setdefault('numba.core.types', mock)


def detect_whisper_backend() -> str:
    _ensure_numba_mock()
    try:
        import faster_whisper  # noqa
        return "faster_whisper"
    except ImportError:
        pass
    try:
        import whisper  # noqa
        return "openai_whisper"
    except ImportError:
        pass
    return "none"


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",        # Termux 기본: small (메모리 절약)
    language: str = "ja",
    use_denoise: bool = True,
    denoise_strength: float = 0.55,
    vad_threshold: float = 0.30,
    no_speech_threshold: float = 0.35,
    beam_size: int = 1,
    cpu_threads: int = 0,
    log_fn=None,
    progress_fn=None,
) -> str:
    from audio_processor import load_audio, apply_denoise, normalize_audio
    import soundfile as sf

    def log(msg):
        if log_fn:
            log_fn(msg)

    def prog(pct: int):
        if progress_fn:
            progress_fn(pct)

    audio_path = Path(audio_path)
    output_srt = audio_path.with_suffix(".srt")

    log(f"[{audio_path.name}] 오디오 로드 중...")
    prog(5)
    audio, sr = load_audio(str(audio_path))

    if use_denoise:
        log(f"[{audio_path.name}] 노이즈 감소 중... (강도={denoise_strength:.2f})")
        prog(15)
        audio = apply_denoise(audio, sr, strength=denoise_strength)

    audio = normalize_audio(audio)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, audio, sr)

        backend = detect_whisper_backend()
        if backend == "none":
            raise ImportError(
                "faster-whisper 또는 openai-whisper가 설치되지 않았습니다.\n"
                "setup.sh 를 실행하세요."
            )

        log(f"[{audio_path.name}] Whisper 로드 중 ({model_size}, {backend}, CPU)...")
        prog(25)

        if backend == "faster_whisper":
            srt_lines = _transcribe_faster_whisper(
                tmp_path, model_size, language,
                vad_threshold, no_speech_threshold,
                beam_size, cpu_threads, log, prog,
            )
        else:
            srt_lines = _transcribe_openai_whisper(
                tmp_path, model_size, language,
                no_speech_threshold, log, prog,
            )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    prog(95)
    log(f"[{audio_path.name}] SRT 저장 중...")
    with open(str(output_srt), "w", encoding="utf-8-sig") as f:
        f.write("\n".join(srt_lines))

    prog(100)
    log(f"[{audio_path.name}] 완료 → {output_srt.name}")
    return str(output_srt)


def _transcribe_faster_whisper(
    audio_path, model_size, language,
    vad_threshold, no_speech_threshold,
    beam_size, cpu_threads,
    log, prog,
):
    from faster_whisper import WhisperModel

    # Termux: 항상 CPU + int8
    device = "cpu"
    compute_type = "int8"
    _threads = cpu_threads if cpu_threads > 0 else (os.cpu_count() or 4)
    _workers = min(2, _threads // 4) or 1

    log(f"  디바이스: CPU / compute_type: {compute_type} / threads: {_threads}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=_threads,
        num_workers=_workers,
    )
    prog(40)

    _best_of = beam_size if beam_size > 1 else 1

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        best_of=_best_of,
        temperature=0.0,
        condition_on_previous_text=True,
        initial_prompt=ASMR_PROMPTS.get(language, ""),
        vad_filter=True,
        vad_parameters={
            "threshold": vad_threshold,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 300,
        },
        no_speech_threshold=no_speech_threshold,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        word_timestamps=False,
    )

    log(f"  감지 언어: {info.language} (확률 {info.language_probability:.2f})")
    prog(60)

    srt_lines = []
    idx = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        srt_lines.append(str(idx))
        srt_lines.append(f"{seconds_to_srt_time(seg.start)} --> {seconds_to_srt_time(seg.end)}")
        srt_lines.append(text)
        srt_lines.append("")
        idx += 1

    prog(90)
    return srt_lines


def _transcribe_openai_whisper(
    audio_path, model_size, language,
    no_speech_threshold, log, prog,
):
    _ensure_numba_mock()  # numba mock 주입 후 import
    import whisper

    model = whisper.load_model(model_size)
    prog(40)
    log("  openai-whisper 백엔드 사용 중")

    result = model.transcribe(
        audio_path,
        language=language,
        fp16=False,
        initial_prompt=ASMR_PROMPTS.get(language, ""),
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=True,
        temperature=0.0,
    )

    prog(85)
    srt_lines = []
    idx = 1
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if not text:
            continue
        srt_lines.append(str(idx))
        srt_lines.append(
            f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}"
        )
        srt_lines.append(text)
        srt_lines.append("")
        idx += 1

    prog(90)
    return srt_lines
