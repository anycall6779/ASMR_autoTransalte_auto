#!/data/data/com.termux/files/usr/bin/bash
# ASMR Studio — Termux 설치 스크립트

set -e
echo "========================================"
echo "  ASMR Studio — Termux 환경 설치"
echo "========================================"
echo ""

# 이 스크립트가 있는 실제 경로 저장
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] 시스템 패키지 설치..."
pkg update -y
pkg install -y python ffmpeg libsndfile git

echo ""
echo "[2/4] Python 패키지 설치..."
# Termux에서는 pip upgrade 금지 — 직접 패키지만 설치
pip install flask
pip install deep-translator
pip install soundfile
pip install noisereduce
pip install numpy

echo ""
echo "[3/4] Whisper STT 설치 (시간이 걸립니다)..."
pip install faster-whisper || {
    echo "faster-whisper 실패 → openai-whisper 시도..."
    pip install openai-whisper
}

echo ""
echo "[4/4] ASMRT 단축 명령 등록..."

BASHRC="$HOME/.bashrc"
ALIAS_LINE="alias ASMRT='cd \"$SCRIPT_DIR\" && bash run.sh'"

# 이미 등록된 경우 중복 방지
if grep -q "alias ASMRT=" "$BASHRC" 2>/dev/null; then
    # 기존 라인 교체
    sed -i "/alias ASMRT=/c\\$ALIAS_LINE" "$BASHRC"
    echo "  ASMRT 단축 명령 업데이트됨"
else
    echo "" >> "$BASHRC"
    echo "# ASMR Studio 단축 명령" >> "$BASHRC"
    echo "$ALIAS_LINE" >> "$BASHRC"
    echo "  ASMRT 단축 명령 등록됨"
fi

echo ""
echo "========================================"
echo "  설치 완료!"
echo ""
echo "  지금 바로 실행:  bash run.sh"
echo "  다음부터 실행:   ASMRT  (새 터미널 세션부터 적용)"
echo ""
echo "  단축 명령이 바로 적용되려면:"
echo "    source ~/.bashrc"
echo "========================================"
