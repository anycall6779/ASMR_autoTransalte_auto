#!/data/data/com.termux/files/usr/bin/bash
# ASMR Studio — 서버 실행 스크립트

cd "$(dirname "$0")"

echo "ASMR Studio 서버 시작 중..."
echo "접속 주소: http://localhost:5000"
echo "종료: Ctrl+C"
echo ""

python app.py
