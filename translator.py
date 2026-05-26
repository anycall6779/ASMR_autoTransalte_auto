# -*- coding: utf-8 -*-
"""
ASMR 번역 모듈 (Termux용 — deep_translator 기반)
Playwright / Gemini Web 대신 Google Translate 무료 API 사용.
"""

import re
from pathlib import Path
from typing import List, Optional, Callable, Tuple

BATCH_SIZE = 40  # deep_translator 한 번 요청 크기

# deep_translator 언어코드 매핑
LANG_MAP = {
    "ja":    "ja",
    "ko":    "ko",
    "en":    "en",
    "zh":    "zh-CN",
    "zh-tw": "zh-TW",
}


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


# ── 배치 번역 ─────────────────────────────────────────────
def _translate_batch(
    texts: List[str],
    src: str,
    dst: str,
    log_fn=None,
) -> List[str]:
    """deep_translator로 텍스트 리스트 번역 (구분자 합치기 → 1번 API 요청)"""
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        if log_fn:
            log_fn("  ⚠ deep-translator 미설치: pip install deep-translator")
        return list(texts)

    src_code = LANG_MAP.get(src, src)
    dst_code = LANG_MAP.get(dst, dst)
    translator = GoogleTranslator(source=src_code, target=dst_code)

    # 텍스트 내 줄바꿈 → 공백 치환 후 구분자로 합쳐서 1번 요청
    SEP = " ▶ "
    combined = SEP.join(t.replace("\n", " ") for t in texts)
    try:
        translated_combined = translator.translate(combined)
        if translated_combined:
            parts = translated_combined.split(SEP)
            if len(parts) == len(texts):
                return [p.strip() for p in parts]
    except Exception as e:
        if log_fn:
            log_fn(f"  ⚠ 배치 번역 실패, 개별 시도: {e}")

    # 폴백: 개별 번역
    results = []
    for text in texts:
        try:
            translated = translator.translate(text)
            results.append(translated if translated else text)
        except Exception as e:
            if log_fn:
                log_fn(f"  ⚠ 번역 실패 (원본 유지): {e}")
            results.append(text)
    return results


# ── 메인 함수 ─────────────────────────────────────────────
def translate_srt(
    srt_path: str,
    src_lang: str = "ja",
    dst_lang: str = "ko",
    output_path: Optional[str] = None,
    log_fn: Optional[Callable] = None,
    progress_fn: Optional[Callable] = None,
    stop_event=None,
) -> str:
    def log(msg):
        if log_fn:
            log_fn(msg)

    def prog(pct: float):
        if progress_fn:
            progress_fn(pct)

    src = Path(srt_path)
    with open(src, encoding="utf-8-sig", errors="replace") as f:
        content = f.read()

    blocks = _parse_srt(content)
    if not blocks:
        raise ValueError(f"SRT 파싱 실패: {src.name}")

    total = len(blocks)
    log(f"[{src.name}] {total}개 자막 블록 번역 시작 (Google Translate)")

    if output_path is None:
        output_path = str(src.parent / f"{src.stem}_{dst_lang}.srt")

    translated: List[Tuple[str, str, str]] = []

    for start in range(0, total, BATCH_SIZE):
        if stop_event and stop_event.is_set():
            log("  중단됨.")
            break

        end = min(start + BATCH_SIZE, total)
        batch = blocks[start:end]
        texts = [b[2] for b in batch]

        log(f"  번역 {start + 1}~{end}/{total}")
        prog(start / total * 95)

        t_texts = _translate_batch(texts, src_lang, dst_lang, log_fn)

        for (idx, tc, _orig), t in zip(batch, t_texts):
            translated.append((idx, tc, t))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(_build_srt(translated))

    log(f"  ✓ 저장: {out.name}")
    prog(100)
    return str(out)
