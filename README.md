# 🎙 ASMR Studio — Termux Mobile Edition

> ASMR 오디오 자동 자막 추출 → 번역 → 영상 합성 파이프라인을 Android(Termux) 위에서 **localhost 웹앱**으로 구동하는 포팅 버전입니다.

**📦 이 저장소:** [anycall6779/ASMR_autoTransalte_auto](https://github.com/anycall6779/ASMR_autoTransalte_auto)

---

## 📌 참고(기반) 프로젝트

이 프로젝트는 아래 두 저장소를 기반으로 제작되었습니다.

| 원본 레포 | 역할 |
|-----------|------|
| [anycall6779/ASMR_translation_auto](https://github.com/anycall6779/ASMR_translation_auto) | ASMR 속삭임 최적화 STT(Whisper) + 오디오 전처리 + Tkinter GUI |
| [anycall6779/ASMR_autoTransalte_auto](https://github.com/anycall6779/ASMR_autoTransalte_auto) | Termux 모바일 포팅 — Flask 웹앱 + deep_translator 번역 + CPU-only 영상합성 |

### 데스크톱 원본 대비 주요 변경점

| 항목 | 원본 (Windows) | 이 버전 (Termux) |
|------|---------------|-----------------|
| UI | Tkinter GUI | Flask localhost 웹앱 |
| 번역 엔진 | Playwright + Gemini 웹 자동화 | `deep_translator` (Google Translate 무료) |
| GPU 가속 | CUDA / h264_nvenc | CPU-only (libx264 강제) |
| Whisper 기본 모델 | large-v3-turbo | small (모바일 메모리 배려) |
| 실시간 로그 | ScrolledText 위젯 | SSE (Server-Sent Events) |
| 브라우저 열기 | 해당 없음 | `termux-open-url http://localhost:5000` 자동 실행 |
| 한글 자막 줄바꿈 | 버그 (CJK 범위 누락) | 수정됨 (`0xAC00~0xD7A3` 추가) |

---

## ✨ 주요 기능

- **🎤 STT 자막 추출** — faster-whisper(우선) / openai-whisper(폴백), ASMR 속삭임 전용 VAD 튜닝
- **🔇 노이즈 전처리** — noisereduce CPU 모드, 강도 조절 가능
- **🌐 다국어 번역** — Google Translate 무료 (ja/ko/en/zh/zh-tw), 40개 배치 처리
- **🎬 영상 합성** — FFmpeg CPU 무손실 MKV + FLAC 오디오 + 자막 하드 번인
- **📱 모바일 웹 UI** — 4탭 SPA (파일/STT/번역/영상), 드래그앤드롭 업로드, SSE 실시간 진행바

---

## 📁 파일 구성

```
termux_git/
├── app.py               # Flask 서버 (API 8개 + SSE 스트림)
├── audio_processor.py   # 오디오 로드·노이즈감소·정규화
├── transcriber.py       # Whisper STT — CPU int8 고정
├── translator.py        # deep_translator 기반 SRT 번역
├── video_maker.py       # FFmpeg CPU-only 영상 합성
├── templates/
│   └── index.html       # 모바일 SPA (Catppuccin Dark)
├── uploads/             # 업로드 파일 저장 (자동 생성)
├── outputs/             # SRT / MKV 출력 (자동 생성)
├── requirements.txt
├── setup.sh             # Termux 패키지 + pip 설치
└── run.sh               # 서버 실행 + 브라우저 자동 열기
```

---

## 📲 설치 방법 (Termux)

### 1단계 — Termux 기본 설정

```bash
# F-Droid 버전 Termux 권장 (Google Play 버전은 패키지 업데이트 제한 있음)
pkg update -y && pkg upgrade -y
```

### 2단계 — 파일 복사

```bash
# Termux 내부에서 git clone
git clone https://github.com/anycall6779/ASMR_autoTransalte_auto.git
cd ASMR_autoTransalte_auto
```

### 3단계 — 의존성 설치

```bash
bash setup.sh
```

> **⚠ Termux 주의:** `pip install --upgrade pip` 는 Termux에서 금지됩니다.  
> `setup.sh`는 pip 자체를 업그레이드하지 않고 패키지만 직접 설치합니다.

설치 항목:
- `pkg`: python, ffmpeg, libsndfile, git, **python-numpy**, **python-scipy** (사전 컴파일 바이너리)
- `pip`: flask, deep-translator, soundfile, noisereduce, faster-whisper (`--prefer-binary` 옵션으로 소스 컴파일 없이 설치)
- `~/.bashrc`: `ASMRT` 단축 명령 자동 등록

> **속도 개선 포인트**: `scipy`/`numpy`를 `pkg`로 미리 설치하면 pip이 소스를 컴파일하지 않아 설치 시간이 크게 단축됩니다.

### 4단계 — 실행

```bash
bash run.sh
```

> 서버 시작 후 1.5초 뒤 `termux-open-url`로 기본 브라우저가 자동 열립니다.  
> 수동 접속: `http://localhost:5000`

---

## ⚡ ASMRT 단축 명령

`setup.sh` 실행 시 `~/.bashrc`에 단축 명령이 자동 등록됩니다.

```bash
# 설치 직후 1회만 실행 (새 세션부터는 자동 적용)
source ~/.bashrc

# 이후 어디서든 한 단어로 실행
ASMRT
```

`ASMRT` = 프로젝트 폴더로 이동 + `bash run.sh` 자동 실행  
Termux를 새로 열 때마다 `ASMRT`만 입력하면 바로 서버가 시작됩니다.

---

## 🖥️ 사용 방법

### 📁 파일 탭
1. **파일 선택** 또는 드래그앤드롭으로 오디오/이미지 파일 업로드
2. **⬆ 업로드** 버튼으로 서버에 전송

### 🎤 STT 탭
1. 모델 크기 선택 (권장: `small` — 모바일 / `large-v3-turbo` — 고정확도)
2. 언어 선택 (일본어 ASMR → `日本語`)
3. 노이즈 강도 조절 (기본 0.55, 너무 높으면 속삭임 손상)
4. 오디오 파일 체크 → **▶ 자막 생성 시작**
5. 완료 후 `outputs/` 폴더에 `.srt` 저장됨

### 🌐 번역 탭
1. 원본/번역 언어 선택
2. SRT 파일 체크 → **▶ 번역 시작**
3. 완료 후 `파일명_ko.srt` 등으로 저장됨

### 🎬 영상 탭
1. 배경 이미지 선택
2. 자막 크기/색상 설정
3. 오디오+SRT 쌍 확인 (자동 연결됨) → **▶ 영상 생성 시작**
4. 완료 후 `outputs/파일명.mkv`로 저장, ⬇ 버튼으로 다운로드

---

## ⚙️ 권장 설정 (ASMR 일본어)

| 항목 | 권장값 | 비고 |
|------|--------|------|
| 모델 | `small` | 모바일 / `large-v3-turbo`는 고사양 기기용 |
| 언어 | `ja` | 일본어 ASMR |
| 노이즈 강도 | `0.45~0.60` | 너무 높으면 속삭임 손상 |
| VAD 임계값 | `0.25~0.35` | 낮을수록 작은 소리 감지 |

---

## 🔧 의존성

```
flask>=3.0.0
deep-translator>=1.11.0
soundfile>=0.12.1
noisereduce>=3.0.2
numpy>=1.24.0
faster-whisper>=1.0.0   # 또는 openai-whisper
```

시스템:
- **Python** 3.9 이상
- **FFmpeg** — `pkg install ffmpeg`

---

## 🌐 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 웹 UI |
| `GET` | `/api/files` | 업로드/출력 파일 목록 |
| `POST` | `/api/upload` | 파일 업로드 |
| `POST` | `/api/delete` | 파일 삭제 |
| `POST` | `/api/transcribe` | STT 작업 시작 (task_id 반환) |
| `POST` | `/api/translate` | 번역 작업 시작 (task_id 반환) |
| `POST` | `/api/create_video` | 영상 합성 작업 시작 (task_id 반환) |
| `GET` | `/stream/<task_id>` | SSE 실시간 로그/진행률 |
| `GET` | `/download/<filename>` | 결과 파일 다운로드 |

---

## ⚠️ 주의사항

- **pip 업그레이드 금지**: Termux에서 `pip install --upgrade pip`는 오류 발생. `setup.sh`는 이를 건너뜁니다
- **flask 미설치 시**: `run.sh` 실행 전 반드시 `bash setup.sh` 먼저 실행
- **메모리**: `large-v3` 모델은 RAM 8GB 이상 필요. Termux 기본 기기는 `small` 권장
- **속도**: CPU-only이므로 오디오 1분당 STT 약 1~5분 소요 (기기 성능에 따라 다름)
- **영상 크기**: 무손실 MKV는 파일이 크므로 스토리지 여유 확인 필요
- **번역**: `deep_translator`는 인터넷 연결 필요 (Google Translate 무료 사용)
- **포트 충돌**: 5000 포트 사용 중이면 `app.py`에서 `port=5000` 수정

---

## 📝 라이선스

MIT License

---

## 🙏 크레딧

- [anycall6779/ASMR_translation_auto](https://github.com/anycall6779/ASMR_translation_auto) — STT 파이프라인 원본 (Windows Tkinter 버전)
- [anycall6779/ASMR_autoTransalte_auto](https://github.com/anycall6779/ASMR_autoTransalte_auto) — 이 저장소 (Termux Flask 웹앱 버전)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 기반 고속 Whisper
- [deep-translator](https://github.com/nidhaloff/deep-translator) — 무료 번역 라이브러리
- [FFmpeg](https://ffmpeg.org) — 영상 합성 엔진
