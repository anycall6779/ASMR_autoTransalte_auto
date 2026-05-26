#!/data/data/com.termux/files/usr/bin/bash
# ASMR Studio — Termux 설치

echo "========================================"
echo "  ASMR Studio — Termux 환경 설치"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1단계: 시스템 패키지 ─────────────────────────────────────
echo "[1/3] 시스템 패키지 설치..."
pkg update -y
pkg install -y python ffmpeg libsndfile git rust python-torch
echo "  ✓ 완료"

# ── 2단계: pip 패키지 ─────────────────────────────────────────
echo ""
echo "[2/3] pip 패키지 설치..."
pip install flask           && echo "  ✓ flask"
pip install transformers    && echo "  ✓ transformers (STT + 번역 백엔드)"
pip install sentencepiece   && echo "  ✓ sentencepiece (로컬 AI 번역)" || echo "  - sentencepiece 스킵 (Google Translate로 대체)"
pip install deep-translator && echo "  ✓ deep-translator (번역 폴백)"
pip install soundfile --no-deps 2>/dev/null && echo "  ✓ soundfile" || echo "  - soundfile 스킵"

# ── 3단계: ASMRT 단축 명령 ────────────────────────────────────
echo ""
echo "[3/3] ASMRT 단축 명령 등록..."
BASHRC="$HOME/.bashrc"
ALIAS_LINE="alias ASMRT='cd \"$SCRIPT_DIR\" && bash run.sh'"
if grep -q "alias ASMRT=" "$BASHRC" 2>/dev/null; then
    sed -i "/alias ASMRT=/c\\$ALIAS_LINE" "$BASHRC"
else
    { echo ""; echo "# ASMR Studio"; echo "$ALIAS_LINE"; } >> "$BASHRC"
fi
echo "  ✓ ASMRT 등록됨"

echo ""
echo "========================================"
echo "  설치 완료!"
echo ""
echo "  지금 실행:   bash run.sh"
echo "  다음부터:    source ~/.bashrc && ASMRT"
echo ""
echo "  STT 백엔드: HuggingFace transformers"
echo "  첫 실행 시 Whisper 모델 자동 다운로드"
echo "========================================"
