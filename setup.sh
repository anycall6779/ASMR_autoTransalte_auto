#!/data/data/com.termux/files/usr/bin/bash
# ASMR Studio — Termux 설치 스크립트

set -e
echo "========================================"
echo "  ASMR Studio — Termux 환경 설치"
echo "========================================"
echo ""

echo "[1/4] 시스템 패키지 설치..."
pkg update -y
pkg install -y python ffmpeg libsndfile

echo ""
echo "[2/4] pip 업그레이드..."
pip install --upgrade pip

echo ""
echo "[3/4] Python 패키지 설치..."
pip install flask
pip install deep-translator
pip install soundfile
pip install noisereduce
pip install numpy

echo ""
echo "[4/4] Whisper STT 설치 (시간이 걸립니다)..."
pip install faster-whisper || {
    echo "faster-whisper 실패 → openai-whisper 시도..."
    pip install openai-whisper
}

echo ""
echo "========================================"
echo "  설치 완료!"
echo "  run.sh 로 서버를 실행하세요."
echo "========================================"
