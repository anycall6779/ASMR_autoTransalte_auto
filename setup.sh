#!/data/data/com.termux/files/usr/bin/bash
# ASMR Studio — Termux 설치 (순수 Python만, 빌드 없음)

# set -e 제거 — 개별 실패 무시하고 계속 진행
echo "========================================"
echo "  ASMR Studio — Termux 환경 설치"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1단계: pkg 기본 도구 ────────────────────────────────────────
echo "[1/4] 시스템 패키지 설치..."
pkg update -y
pkg install -y python ffmpeg libsndfile git
echo "  ✓ 완료"

# ── 2단계: 순수 Python pip 패키지만 ────────────────────────────
echo ""
echo "[2/4] pip 패키지 설치 (순수 Python — 빌드 없음)..."

# 핵심 패키지 (반드시 설치)
pip install flask       && echo "  ✓ flask"
pip install deep-translator && echo "  ✓ deep-translator"

# soundfile: 바이너리 있으면 설치 (없어도 WAV는 numpy로 처리)
pip install --only-binary=:all: --no-deps soundfile 2>/dev/null \
    && echo "  ✓ soundfile" \
    || echo "  - soundfile 스킵 (WAV 기본 지원)"

echo ""
echo "  [STT] faster-whisper는 ARM 바이너리 없음 — 별도 설치"
echo "  [노이즈] noisereduce/scipy는 ARM 바이너리 없음 — 기능 비활성화"
echo "  ※ 위 두 기능 없어도 앱 실행 및 번역 기능은 정상 동작"

# ── 3단계: ASMRT 단축 명령 ────────────────────────────────────
echo ""
echo "[3/4] ASMRT 단축 명령 등록..."
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
echo "  지금 실행:       bash run.sh"
echo "  단축 명령 적용:  source ~/.bashrc"
echo "  다음부터:        ASMRT"
echo ""
echo "  ※ STT 기능 원할 시 별도 설치:"
echo "    pip install faster-whisper"
echo "========================================"
