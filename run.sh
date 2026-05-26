#!/data/data/com.termux/files/usr/bin/bash
# ASMR Studio — 서버 실행 스크립트

cd "$(dirname "$0")"

echo "========================================"
echo "  ASMR Studio 서버 시작 중..."
echo "  접속 주소: http://localhost:5000"
echo "  종료: Ctrl+C"
echo "========================================"
echo ""

# Flask 설치 여부 사전 확인
python -c "import flask" 2>/dev/null || {
    echo "[오류] flask가 설치되지 않았습니다."
    echo "  먼저 setup.sh 를 실행하세요:"
    echo "    bash setup.sh"
    exit 1
}

export HF_HUB_DISABLE_XET=1
python app.py
