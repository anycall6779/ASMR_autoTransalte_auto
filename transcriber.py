# -*- coding: utf-8 -*-
# HuggingFace hf-xet 다운로더 비활성화 (Android/Termux rustls 오류 방지)
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
"""
ASMR 음성 인식 모듈 (Termux / CPU-only)
백엔드 우선순위:
  1. faster-whisper  (ctranslate2 설치된 경우)
  2. whisper.cpp     (~/whisper.cpp/main 컴파일된 경우) ← 권장
  3. transformers    (torch + HuggingFace, tokenizers 설치된 경우)
  4. none → 오류

whisper.cpp 설치:
  pkg install clang make git
  cd ~ && git clone https://github.com/ggerganov/whisper.cpp --depth=1
  cd whisper.cpp && make -j$(nproc) main
  bash models/download-ggml-model.sh small
"""

import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# whisper.cpp 바이너리 후보 경로
_WCPP_CANDIDATES = [
    os.path.expanduser("~/whisper.cpp/main"),
    os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"),
    "/data/data/com.termux/files/usr/bin/whisper-cli",
    "/data/data/com.termux/files/usr/bin/whisper",
]

# whisper.cpp 모델 디렉터리
_WCPP_MODEL_DIR = os.path.expanduser("~/whisper.cpp/models")


def seconds_to_srt_time(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
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


def _find_wcpp_binary() -> str | None:
    """whisper.cpp 바이너리 경로 탐색."""
    for p in _WCPP_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _find_wcpp_model(model_size: str) -> str | None:
    """ggml 모델 파일 탐색."""
    names = [
        f"ggml-{model_size}.bin",
        f"ggml-{model_size}-q5_1.bin",
        f"ggml-{model_size}-q8_0.bin",
    ]
    for name in names:
        p = os.path.join(_WCPP_MODEL_DIR, name)
        if os.path.isfile(p):
            return p
    return None


def detect_whisper_backend() -> str:
    """설치된 STT 백엔드 자동 감지."""
    # 1) faster-whisper
    try:
        import faster_whisper  # noqa
        return "faster_whisper"
    except ImportError:
        pass
    # 2) whisper.cpp 바이너리
    if _find_wcpp_binary():
        return "whisper_cpp"
    # 3) HuggingFace transformers
    try:
        import transformers  # noqa
        import torch          # noqa
        return "transformers"
    except ImportError:
        pass
    return "none"


def transcribe_audio(
    audio_path: str,
    model_size: str = "tiny",        # Termux 기본: tiny (모바일 CPU 최적화, small보다 4배 빠름)
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
    def log(msg):
        if log_fn:
            log_fn(msg)

    def prog(pct: int):
        if progress_fn:
            progress_fn(pct)

    audio_path = Path(audio_path)
    output_srt = audio_path.with_suffix(".srt")

    # ffmpeg로 16kHz mono WAV 변환 (모든 포맷 지원: m4a/mp3/flac 등)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    log(f"[{audio_path.name}] 오디오 변환 중...")
    prog(5)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        backend = detect_whisper_backend()
        log(f"[{audio_path.name}] 백엔드: {backend} / 모델: {model_size}")
        prog(25)

        if backend == "none":
            raise ImportError(
                "STT 백엔드가 없습니다.\n"
                "▶ 권장: pkg install clang make git\n"
                "  cd ~ && git clone https://github.com/ggerganov/whisper.cpp --depth=1\n"
                "  cd whisper.cpp && make -j$(nproc) main\n"
                "  bash models/download-ggml-model.sh small"
            )

        if backend == "faster_whisper":
            srt_lines = _transcribe_faster_whisper(
                tmp_path, model_size, language,
                vad_threshold, no_speech_threshold,
                beam_size, cpu_threads, log, prog,
            )
        elif backend == "whisper_cpp":
            srt_lines = _transcribe_whisper_cpp(
                tmp_path, model_size, language,
                cpu_threads, log, prog,
            )
        else:
            srt_lines = _transcribe_transformers(
                tmp_path, model_size, language,
                log, prog,
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


# ─────────────────────────────────────────────────────────────
# 백엔드 1: faster-whisper
# ─────────────────────────────────────────────────────────────
def _transcribe_faster_whisper(
    audio_path, model_size, language,
    vad_threshold, no_speech_threshold,
    beam_size, cpu_threads, log, prog,
):
    from faster_whisper import WhisperModel

    _threads = cpu_threads if cpu_threads > 0 else (os.cpu_count() or 4)
    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         cpu_threads=_threads, num_workers=1)
    prog(40)

    segments, info = model.transcribe(
        audio_path, language=language,
        beam_size=beam_size, temperature=0.0,
        condition_on_previous_text=True,
        initial_prompt=ASMR_PROMPTS.get(language, ""),
        vad_filter=True,
        vad_parameters={"threshold": vad_threshold, "min_speech_duration_ms": 100,
                        "min_silence_duration_ms": 500, "speech_pad_ms": 300},
        no_speech_threshold=no_speech_threshold,
        word_timestamps=False,
    )
    log(f"  감지 언어: {info.language} ({info.language_probability:.2f})")
    prog(60)

    srt_lines, idx = [], 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        srt_lines += [str(idx),
                      f"{seconds_to_srt_time(seg.start)} --> {seconds_to_srt_time(seg.end)}",
                      text, ""]
        idx += 1
    prog(90)
    return srt_lines


# ─────────────────────────────────────────────────────────────
# 백엔드 2: whisper.cpp subprocess ← Termux 권장
# pkg install clang make && cd ~/whisper.cpp && make main
# ─────────────────────────────────────────────────────────────
def _transcribe_whisper_cpp(
    audio_path, model_size, language,
    cpu_threads, log, prog,
):
    binary = _find_wcpp_binary()
    model  = _find_wcpp_model(model_size)

    if model is None:
        log(f"  모델 없음. 다운로드 중: {model_size} ...")
        dl_script = os.path.expanduser("~/whisper.cpp/models/download-ggml-model.sh")
        wcpp_dir  = os.path.expanduser("~/whisper.cpp")
        subprocess.run(["bash", dl_script, model_size],
                       cwd=wcpp_dir, check=True)
        model = _find_wcpp_model(model_size)
        if model is None:
            raise FileNotFoundError(f"모델 다운로드 실패: {model_size}")

    _threads = str(cpu_threads if cpu_threads > 0 else (os.cpu_count() or 4))
    lang_arg = language if language else "auto"

    # 임시 SRT 출력 경로
    srt_out = audio_path + "_wcpp"

    cmd = [
        binary,
        "-m", model,
        "-f", audio_path,
        "-l", lang_arg,
        "-t", _threads,
        "-osrt",          # SRT 출력
        "-of", srt_out,   # 출력 파일 prefix
        "--no-prints",    # 진행 출력 억제 (없으면 생략)
    ]
    log(f"  whisper.cpp 실행 중 (threads={_threads})...")
    prog(40)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    prog(85)

    srt_file = srt_out + ".srt"
    if os.path.isfile(srt_file):
        with open(srt_file, encoding="utf-8") as f:
            content = f.read().strip()
        os.unlink(srt_file)
        return content.split("\n") if content else []

    # --no-prints 미지원 시 stderr 없이 재시도
    if proc.returncode != 0:
        cmd_retry = [x for x in cmd if x != "--no-prints"]
        proc = subprocess.run(cmd_retry, capture_output=True, text=True)
        srt_file = srt_out + ".srt"
        if os.path.isfile(srt_file):
            with open(srt_file, encoding="utf-8") as f:
                content = f.read().strip()
            os.unlink(srt_file)
            return content.split("\n") if content else []
        raise RuntimeError(f"whisper.cpp 오류:\n{proc.stderr[-500:]}")

    prog(90)
    return []


# ─────────────────────────────────────────────────────────────
# 백엔드 3: HuggingFace transformers (torch 필요)
# pipeline 대신 model.generate() 직접 호용 — hang 해결
# ─────────────────────────────────────────────────────────────
def _transcribe_transformers(
    audio_path, model_size, language,
    log, prog,
):
    import torch
    import soundfile as sf
    import numpy as np
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    model_id = f"openai/whisper-{model_size}"
    log(f"  모델: {model_id} (첫 실행 시 자동 다운로드)")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()
    prog(35)

    # 16kHz WAV 읽기 (ffmpeg가 이미 변환해 놓은 tmp 파일)
    audio_array, sr = sf.read(audio_path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1).astype(np.float32)
    else:
        audio_array = audio_array.astype(np.float32)

    CHUNK = 16000 * 28           # 28씽 청크 (Whisper 30시 제한 이하)
    total_chunks = max(1, int(np.ceil(len(audio_array) / CHUNK)))
    segments = []

    log(f"  음성 인식 중... ({total_chunks}개 청크)")
    for ci in range(total_chunks):
        chunk = audio_array[ci * CHUNK : (ci + 1) * CHUNK]
        if len(chunk) < 400:
            break

        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs.input_features,
                language=language,
                task="transcribe",
                return_timestamps=True,
            )

        decoded = processor.decode(
            predicted_ids[0],
            output_offsets=True,
            skip_special_tokens=True,
        )
        offset_sec = ci * CHUNK / 16000
        for seg in decoded.get("offsets", []):
            ts   = seg.get("timestamp", (None, None))
            text = seg.get("text", "").strip()
            if not text:
                continue
            t0 = (ts[0] if ts[0] is not None else 0.0) + offset_sec
            t1 = (ts[1] if ts[1] is not None else t0 + 3.0) + offset_sec
            segments.append({"start": t0, "end": t1, "text": text})

        prog(35 + int((ci + 1) / total_chunks * 50))
        log(f"    청크 {ci + 1}/{total_chunks} 완료")

    # fallback: offsets 없으면 전체 텍스트 하나의 세그먼트로
    if not segments:
        with torch.no_grad():
            ids = model.generate(
                processor(audio_array[:CHUNK], sampling_rate=16000,
                          return_tensors="pt").input_features,
                language=language, task="transcribe",
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            segments = [{"start": 0.0, "end": 5.0, "text": text}]

    srt_lines, idx = [], 1
    for seg in segments:
        srt_lines += [
            str(idx),
            f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}",
            seg["text"], "",
        ]
        idx += 1
    prog(90)
    return srt_lines
