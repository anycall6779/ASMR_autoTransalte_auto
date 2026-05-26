# -*- coding: utf-8 -*-
"""
ASMR Studio — Flask 웹 서버 (Termux용)
실행: python app.py
접속: http://localhost:5000
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# sys.path에 자신 폴더 추가 (같은 폴더 모듈 import)
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

# ── 작업 관리 ──────────────────────────────────────────────
_tasks: dict = {}
_tasks_lock = threading.Lock()

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
SRT_EXTS   = {".srt"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mkv", ".mp4"}


def _new_task() -> str:
    tid = uuid.uuid4().hex
    with _tasks_lock:
        _tasks[tid] = {
            "q":     queue.Queue(),
            "done":  False,
            "result": None,
            "error": None,
        }
    return tid


def _task_log(tid: str, msg: str):
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid]["q"].put({"type": "log", "data": msg})


def _task_prog(tid: str, pct: int):
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid]["q"].put({"type": "progress", "data": pct})


def _task_done(tid: str, result=None, error=None):
    with _tasks_lock:
        if tid in _tasks:
            t = _tasks[tid]
            t["done"]   = True
            t["result"] = result
            t["error"]  = error
            t["q"].put({"type": "done", "data": result, "error": error})


# ── 라우트 ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# SSE 실시간 스트림
@app.route("/stream/<tid>")
def stream(tid):
    def generate():
        with _tasks_lock:
            exists = tid in _tasks
        if not exists:
            yield 'data: {"type":"error","data":"task not found"}\n\n'
            return

        while True:
            try:
                msg = _tasks[tid]["q"].get(timeout=25)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg["type"] == "done":
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
                with _tasks_lock:
                    if _tasks.get(tid, {}).get("done"):
                        break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 파일 목록
@app.route("/api/files")
def api_files():
    def _info(p: Path):
        return {"name": p.name, "size": p.stat().st_size, "ext": p.suffix.lower()}

    uploads = sorted(
        [_info(f) for f in UPLOAD_DIR.iterdir() if f.is_file()],
        key=lambda x: x["name"],
    )
    outputs = sorted(
        [_info(f) for f in OUTPUT_DIR.iterdir() if f.is_file()],
        key=lambda x: x["name"],
    )
    return jsonify({"uploads": uploads, "outputs": outputs})


# 파일 업로드
@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "파일 없음"}), 400
    saved = []
    for f in files:
        dest = UPLOAD_DIR / f.filename
        f.save(str(dest))
        saved.append(f.filename)
    return jsonify({"saved": saved})


# 파일 삭제
@app.route("/api/delete", methods=["POST"])
def api_delete():
    data = request.json or {}
    name = data.get("name", "")
    folder = data.get("folder", "uploads")
    base = UPLOAD_DIR if folder == "uploads" else OUTPUT_DIR
    target = base / name
    if target.exists() and target.is_file():
        target.unlink()
        return jsonify({"ok": True})
    return jsonify({"error": "파일 없음"}), 404


# ── STT 자막 생성 ───────────────────────────────────────────
@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    data = request.json or {}
    audio_files  = data.get("files", [])
    model_size   = data.get("model_size", "small")
    language     = data.get("language", "ja")
    use_denoise  = data.get("use_denoise", True)
    denoise_str  = float(data.get("denoise_strength", 0.55))
    vad_thr      = float(data.get("vad_threshold", 0.30))
    no_speech    = float(data.get("no_speech_threshold", 0.35))

    if not audio_files:
        return jsonify({"error": "오디오 파일을 선택하세요"}), 400

    tid = _new_task()

    def _run():
        try:
            from transcriber import transcribe_audio
            total = len(audio_files)
            for i, fname in enumerate(audio_files):
                fpath = UPLOAD_DIR / fname
                if not fpath.exists():
                    _task_log(tid, f"[건너뜀] {fname}: 파일 없음")
                    continue

                _task_log(tid, f"\n── [{i + 1}/{total}] {fname}")

                def _log(msg, _i=i): _task_log(tid, msg)
                def _prog(pct, _i=i): _task_prog(tid, int(_i * 100 / total + pct / total))

                out = transcribe_audio(
                    str(fpath),
                    model_size=model_size,
                    language=language,
                    use_denoise=use_denoise,
                    denoise_strength=denoise_str,
                    vad_threshold=vad_thr,
                    no_speech_threshold=no_speech,
                    log_fn=_log,
                    progress_fn=_prog,
                )
                # SRT → outputs/ 이동
                srt_src = Path(out)
                srt_dst = OUTPUT_DIR / srt_src.name
                srt_src.rename(srt_dst)
                _task_log(tid, f"  ✓ {srt_dst.name}")

            _task_done(tid, "STT 완료")
        except Exception as e:
            _task_log(tid, f"[오류] {e}")
            _task_done(tid, error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"task_id": tid})


# ── 번역 ────────────────────────────────────────────────────
@app.route("/api/translate", methods=["POST"])
def api_translate():
    data      = request.json or {}
    srt_files = data.get("files", [])
    src_lang  = data.get("src_lang", "ja")
    dst_lang  = data.get("dst_lang", "ko")

    if not srt_files:
        return jsonify({"error": "SRT 파일을 선택하세요"}), 400

    tid = _new_task()

    def _run():
        try:
            from translator import translate_srt
            total = len(srt_files)
            for i, fname in enumerate(srt_files):
                # outputs/ 우선, 없으면 uploads/ 탐색
                fpath = OUTPUT_DIR / fname
                if not fpath.exists():
                    fpath = UPLOAD_DIR / fname
                if not fpath.exists():
                    _task_log(tid, f"[건너뜀] {fname}: 파일 없음")
                    continue

                _task_log(tid, f"\n── [{i + 1}/{total}] {fname}")
                out_name = f"{fpath.stem}_{dst_lang}.srt"
                out_path = OUTPUT_DIR / out_name

                def _log(msg): _task_log(tid, msg)
                def _prog(pct): _task_prog(tid, int(i * 100 / total + pct / total))

                translate_srt(
                    str(fpath), src_lang, dst_lang,
                    str(out_path),
                    log_fn=_log, progress_fn=_prog,
                )
                _task_log(tid, f"  ✓ {out_name}")
                _task_prog(tid, int((i + 1) * 100 / total))

            _task_done(tid, "번역 완료")
        except Exception as e:
            _task_log(tid, f"[오류] {e}")
            _task_done(tid, error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"task_id": tid})


# ── 영상 생성 ────────────────────────────────────────────────
@app.route("/api/create_video", methods=["POST"])
def api_create_video():
    data       = request.json or {}
    pairs      = data.get("pairs", [])        # [{audio, srt}]
    image_file = data.get("image", "")
    font_size  = int(data.get("font_size", 22))
    font_color = data.get("font_color", "white")

    if not pairs:
        return jsonify({"error": "오디오+SRT 쌍을 선택하세요"}), 400
    if not image_file:
        return jsonify({"error": "배경 이미지를 선택하세요"}), 400

    image_path = UPLOAD_DIR / image_file
    if not image_path.exists():
        return jsonify({"error": f"이미지 파일 없음: {image_file}"}), 400

    tid = _new_task()

    def _run():
        try:
            from video_maker import create_video
            total = len(pairs)
            for i, pair in enumerate(pairs):
                audio_p = UPLOAD_DIR / pair["audio"]
                # SRT: outputs/ 우선
                srt_p = OUTPUT_DIR / pair["srt"]
                if not srt_p.exists():
                    srt_p = UPLOAD_DIR / pair["srt"]
                if not audio_p.exists() or not srt_p.exists():
                    _task_log(tid, f"[건너뜀] 파일 없음: {pair['audio']}")
                    continue

                _task_log(tid, f"\n── [{i + 1}/{total}] {pair['audio']}")
                out_path = OUTPUT_DIR / (audio_p.stem + ".mkv")

                def _log(msg): _task_log(tid, msg)
                def _prog(pct): _task_prog(tid, int(i * 100 / total + pct / total))

                create_video(
                    str(audio_p), str(srt_p), str(image_path),
                    str(out_path),
                    font_size=font_size,
                    font_color=font_color,
                    use_gpu=False,
                    log_fn=_log,
                    progress_fn=_prog,
                )
                _task_log(tid, f"  ✓ {out_path.name}")
                _task_prog(tid, int((i + 1) * 100 / total))

            _task_done(tid, "영상 생성 완료")
        except Exception as e:
            _task_log(tid, f"[오류] {e}")
            _task_done(tid, error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"task_id": tid})


# ── 파일 다운로드 ────────────────────────────────────────────
@app.route("/download/<path:filename>")
def download(filename):
    p = OUTPUT_DIR / filename
    if not p.exists():
        p = UPLOAD_DIR / filename
    if not p.exists():
        return "파일 없음", 404
    return send_file(str(p), as_attachment=True)


# ── 진입점 ──────────────────────────────────────────────────
if __name__ == "__main__":
    def _open_browser():
        time.sleep(1.5)
        url = "http://localhost:5000"
        # Termux
        try:
            subprocess.Popen(["termux-open-url", url])
            return
        except FileNotFoundError:
            pass
        # Linux
        try:
            subprocess.Popen(["xdg-open", url])
            return
        except FileNotFoundError:
            pass
        print(f"브라우저에서 수동으로 접속하세요: {url}")

    threading.Thread(target=_open_browser, daemon=True).start()
    print("=" * 40)
    print("  ASMR Studio — 웹 서버 시작")
    print("  접속: http://localhost:5000")
    print("  종료: Ctrl+C")
    print("=" * 40)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
