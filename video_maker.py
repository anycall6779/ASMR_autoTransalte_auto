# -*- coding: utf-8 -*-
"""
ASMR 영상 합성 모듈 (Termux / CPU-only)
- GPU 없이 libx264 -crf 0 (CPU 무손실)
- 오디오: FLAC 무손실
- 자막: SRT 하드 번인
"""

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

VIDEO_OUT_DIR = Path(__file__).parent / "outputs"


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3000 <= cp <= 0x9FFF      # CJK + 일본어 가나
        or 0xAC00 <= cp <= 0xD7A3  # 한국어 (원본 누락 버그 수정)
        or 0xF900 <= cp <= 0xFAFF  # CJK 호환
        or 0x20000 <= cp <= 0x2A6DF
    )


def _wrap_line(text: str, max_chars: int = 40) -> str:
    if not text.strip():
        return text
    cjk_ratio = sum(1 for c in text if _is_cjk(c)) / max(len(text), 1)
    if cjk_ratio > 0.3:
        limit = min(max_chars, 22)
        lines, buf, count = [], [], 0
        for ch in text:
            buf.append(ch)
            count += 1
            if count >= limit:
                lines.append("".join(buf))
                buf, count = [], 0
        if buf:
            lines.append("".join(buf))
        return "\n".join(lines)
    else:
        return textwrap.fill(text, width=max_chars)


def _parse_tc(s: str) -> float:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", s.strip())
    if not m:
        return 0.0
    h, mn, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
    ms_str = m.group(4)
    ms = int(ms_str) * (10 ** (3 - len(ms_str))) if len(ms_str) < 3 else int(ms_str)
    return h * 3600 + mn * 60 + sec + ms / 1000


def _fmt_tc(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    mn = int((t % 3600) // 60)
    sec = int(t % 60)
    ms = min(999, int(round((t - int(t)) * 1000)))
    return f"{h:02d}:{mn:02d}:{sec:02d},{ms:03d}"


def _split_sentences_srt(src: str, dst: str, max_chars: int = 40):
    with open(src, encoding="utf-8-sig", errors="replace") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    out_blocks = []
    counter = 1

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            out_blocks.append(block)
            continue

        tc_parts = lines[1].split("-->")
        if len(tc_parts) != 2:
            out_blocks.append(block)
            continue

        t_start = _parse_tc(tc_parts[0])
        t_end = _parse_tc(tc_parts[1])
        duration = t_end - t_start
        if duration <= 0:
            out_blocks.append(block)
            continue

        full_text = " ".join(lines[2:]).strip()
        sents = re.split(r"(?<=[。！？!?])", full_text)
        sents = [s.strip() for s in sents if s.strip()]

        if len(sents) <= 1:
            wrapped = _wrap_line(full_text, max_chars)
            out_blocks.append(f"{counter}\n{_fmt_tc(t_start)} --> {_fmt_tc(t_end)}\n{wrapped}")
            counter += 1
            continue

        total_chars = sum(len(s) for s in sents) or 1
        cur = t_start
        for i, sent in enumerate(sents):
            if i == len(sents) - 1:
                seg_end = t_end
            else:
                seg_end = cur + duration * len(sent) / total_chars
                seg_end = max(seg_end, cur + 0.3)
                seg_end = min(seg_end, t_end - 0.1 * (len(sents) - i - 1))

            wrapped = _wrap_line(sent, max_chars)
            out_blocks.append(f"{counter}\n{_fmt_tc(cur)} --> {_fmt_tc(seg_end)}\n{wrapped}")
            counter += 1
            cur = seg_end

    with open(dst, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n".join(out_blocks))


def _get_ffmpeg() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise FileNotFoundError(
        "FFmpeg를 찾을 수 없습니다.\n"
        "Termux: pkg install ffmpeg"
    )


def _get_duration(ffmpeg: str, path: str) -> float:
    r = subprocess.run(
        [ffmpeg, "-i", path, "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    output = (r.stderr or "") + (r.stdout or "")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", output)
    if m:
        h, mn, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600 + mn * 60 + s + cs / 100
    return 0.0


def _run_ffmpeg(cmd: list, total_sec: float, log, prog):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout:
        m = re.search(r"time=(\d+):(\d+):(\d+)\.(\d+)", line)
        if m and total_sec > 0:
            cur = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                   + int(m.group(3)) + int(m.group(4)) / 100)
            pct = min(int(cur / total_sec * 80) + 15, 95)
            prog(pct)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 실패 (exit code {proc.returncode})")


def create_video(
    audio_path: str,
    srt_path: str,
    image_path: str,
    output_path: str = None,
    font_size: int = 22,
    font_color: str = "white",
    use_gpu: bool = False,    # Termux: 항상 False
    log_fn=None,
    progress_fn=None,
) -> str:
    def log(msg):
        if log_fn:
            log_fn(msg)

    def prog(pct):
        if progress_fn:
            progress_fn(pct)

    audio_path = Path(audio_path)
    srt_path = Path(srt_path)
    image_path = Path(image_path)

    for p, label in [(audio_path, "오디오"), (srt_path, "SRT"), (image_path, "이미지")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} 파일을 찾을 수 없습니다: {p}")

    if output_path is None:
        VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = VIDEO_OUT_DIR / (audio_path.stem + ".mkv")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _get_ffmpeg()
    prog(5)

    # Termux: 항상 CPU libx264
    vcodec = "libx264"
    encode_opts = ["-crf", "0", "-preset", "ultrafast"]
    pix_fmt = "yuv420p"
    log("  영상 코덱: libx264 -crf 0 (CPU 무손실)")
    log("  오디오 코덱: FLAC (무손실)")
    prog(10)

    duration = _get_duration(ffmpeg, str(audio_path))
    log(f"  오디오 길이: {int(duration // 60)}분 {int(duration % 60)}초")
    prog(15)

    tmp_dir = tempfile.mkdtemp()
    tmp_srt = os.path.join(tmp_dir, "subtitle.srt")
    try:
        _split_sentences_srt(str(srt_path), tmp_srt)

        _srt_fwd = tmp_srt.replace("\\", "/")
        _srt_fwd = re.sub(r"^([A-Za-z]):/", r"\1\\:/", _srt_fwd)

        color_map = {"white": "&H00FFFFFF", "yellow": "&H0000FFFF", "cyan": "&H00FFFF00"}
        primary = color_map.get(font_color, "&H00FFFFFF")

        style = (
            f"FontSize={font_size},FontName=Arial,Bold=1,"
            f"PrimaryColour={primary},OutlineColour=&H00000000,"
            f"BackColour=&H80000000,Outline=2,Shadow=1,"
            f"Alignment=2,MarginV=30,MarginL=40,MarginR=40,"
            f"WrapStyle=0"
        )
        sub_filter = f"subtitles='{_srt_fwd}':force_style='{style}'"
        vf = f"scale=trunc(iw/2)*2:trunc(ih/2)*2,{sub_filter}"

        log(f"[{audio_path.name}] MKV 합성 중...")

        cmd = [
            ffmpeg, "-y",
            "-loop", "1", "-framerate", "1", "-i", str(image_path),
            "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
            "-vf", vf,
            "-c:v", vcodec,
            *encode_opts,
            "-pix_fmt", pix_fmt,
            "-c:a", "flac",
            "-t", str(duration),
            str(output_path),
        ]

        _run_ffmpeg(cmd, duration, log, prog)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    prog(100)
    log(f"[{audio_path.name}] 완료 → {output_path.name}  ({size_mb:.1f} MB)")
    return str(output_path)
