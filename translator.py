# -*- coding: utf-8 -*-
"""
ASMR 번역 모듈 (Termux용 — API 불필요, 완전 무료)

번역 백엔드 (자동 선택):
  1순위: Helsinki-NLP/opus-mt 로컬 AI 모델 (오프라인, sentencepiece 필요)
  2순위: Google Translate 무료 (온라인, API 키 불필요, deep-translator)

설치:
  pip install sentencepiece   # 로컬 AI 번역용 (권장)
  pip install deep-translator # Google Translate 폴백용
"""

import os
import re
import time
from pathlib import Path
from typing import List, Optional, Callable, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BATCH_SIZE = 16  # MarianMT 최적 배치 크기

OPUS_MT_MODELS = {
    ("ja", "ko"): "Helsinki-NLP/opus-mt-ja-ko",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("en", "ko"): "Helsinki-NLP/opus-mt-en-ko",
    ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
    ("ko", "ja"): "Helsinki-NLP/opus-mt-ko-jap",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("zh", "ko"): "Helsinki-NLP/opus-mt-zh-ko",
}

GOOGLE_LANG = {
    "ja": "ja", "ko": "ko", "en": "en",
    "zh": "zh-CN", "zh-tw": "zh-TW",
}

_model_cache: dict = {}


# ── SRT 파싱 / 빌드 ──────────────────────────────────────
def _parse_srt(content: str) -> List[Tuple[str, str, str]]:
    blocks = re.split(r"\n\s*\n", content.strip())
    result = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        result.append((
            lines[0].strip(),
            lines[1].strip(),
            "\n".join(lines[2:]).strip(),
        ))
    return result


def _build_srt(blocks: List[Tuple[str, str, str]]) -> str:
    return "\n\n".join(f"{idx}\n{tc}\n{text}" for idx, tc, text in blocks)


# ── 로컬 opus-mt 번역 ────────────────────────────────────
def _load_opus_mt(src: str, dst: str, log_fn=None):
    key = (src, dst)
    if key in _model_cache:
        return _model_cache[key]

    model_id = OPUS_MT_MODELS.get(key)
    if not model_id:
        raise ValueError(f"지원 안 하는 언어 쌍: {src}→{dst}")

    if log_fn:
        log_fn(f"  번역 모델 로딩: {model_id}")
        log_fn("  (첫 실행 시 HuggingFace 자동 다운로드 ~300MB)")

    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    _model_cache[key] = (tokenizer, model)
    return tokenizer, model


def _translate_local(texts: List[str], src: str, dst: str, log_fn=None) -> List[str]:
    import torch
    tokenizer, model = _load_opus_mt(src, dst, log_fn)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


# ── Google Translate 무료 fallback ───────────────────────
def _translate_google(texts: List[str], src: str, dst: str, log_fn=None) -> List[str]:
    from deep_translator import GoogleTranslator
    gl_src = GOOGLE_LANG.get(src, src)
    gl_dst = GOOGLE_LANG.get(dst, dst)
    translator = GoogleTranslator(source=gl_src, target=gl_dst)
    results = []
    for text in texts:
        try:
            translated = translator.translate(text.strip()) or text
            results.append(translated)
        except Exception as e:
            if log_fn:
                log_fn(f"  ⚠ 번역 오류: {e}")
            results.append(text)
        time.sleep(0.12)
    return results


# ── 백엔드 선택 ──────────────────────────────────────────
def _detect_backend(src: str, dst: str) -> str:
    if (src, dst) not in OPUS_MT_MODELS:
        return "google"
    try:
        import sentencepiece  # noqa
        return "local"
    except ImportError:
        return "google"


# ── 메인 함수 ────────────────────────────────────────────
def translate_srt(
    srt_path: str,
    src_lang: str = "ja",
    dst_lang: str = "ko",
    output_path: Optional[str] = None,
    log_fn: Optional[Callable] = None,
    progress_fn: Optional[Callable] = None,
    stop_event=None,
    **kwargs,  # api_key 등 미사용 인자 무시
) -> str:
    def log(msg):
        if log_fn:
            log_fn(msg)

    def prog(pct: float):
        if progress_fn:
            progress_fn(pct)

    backend = _detect_backend(src_lang, dst_lang)

    src = Path(srt_path)
    with open(src, encoding="utf-8-sig", errors="replace") as f:
        content = f.read()

    blocks = _parse_srt(content)
    if not blocks:
        raise ValueError(f"SRT 파싱 실패: {src.name}")

    total = len(blocks)

    if output_path is None:
        output_path = str(src.parent / f"{src.stem}_{dst_lang}.srt")

    if backend == "local":
        log(f"[{src.name}] {total}개 자막 — 로컬 AI 번역 ({src_lang}→{dst_lang})")
    else:
        if (src_lang, dst_lang) in OPUS_MT_MODELS:
            log("  ⚠ sentencepiece 미설치 → Google Translate 사용")
            log("  (pip install sentencepiece 후 고품질 로컬 모델 사용 가능)")
        log(f"[{src.name}] {total}개 자막 — Google Translate ({src_lang}→{dst_lang})")

    translated: List[Tuple[str, str, str]] = []

    for start in range(0, total, BATCH_SIZE):
        if stop_event and stop_event.is_set():
            log("  중단됨.")
            break

        end = min(start + BATCH_SIZE, total)
        batch = blocks[start:end]
        texts = [b[2] for b in batch]

        log(f"  {start + 1}~{end}/{total}")
        prog(start / total * 95)

        if backend == "local":
            try:
                t_texts = _translate_local(texts, src_lang, dst_lang, log)
            except Exception as e:
                log(f"  로컬 번역 오류 → Google Translate 전환: {e}")
                backend = "google"
                t_texts = _translate_google(texts, src_lang, dst_lang, log)
        else:
            t_texts = _translate_google(texts, src_lang, dst_lang, log)

        for (idx, tc, _orig), t in zip(batch, t_texts):
            translated.append((idx, tc, t))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(_build_srt(translated))

    log(f"  ✓ 저장: {out.name}")
    prog(100)
    return str(out)
