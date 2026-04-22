from __future__ import annotations


_ZH_ALIASES = {"zh", "cn", "chinese", "中文", "sims"}
_EN_ALIASES = {"en", "english", "英文", "cmumosi", "mosi", "mosei", "cmumosei"}


def _count_cjk(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _count_latin(text: str) -> int:
    return sum(1 for char in text if ("a" <= char.lower() <= "z"))


def normalize_language_request(language: str | None) -> str | None:
    value = (language or "").strip().lower()
    if value in _ZH_ALIASES:
        return "zh"
    if value in _EN_ALIASES:
        return "en"
    return None


def detect_text_language(text: str | None) -> str:
    if not text:
        return "unknown"
    cjk_count = _count_cjk(text)
    latin_count = _count_latin(text)
    if cjk_count == 0 and latin_count == 0:
        return "unknown"
    if cjk_count and cjk_count >= max(2, int(latin_count * 0.35) + 1):
        return "zh"
    if latin_count > 0:
        return "en"
    return "unknown"


def dataset_for_language(language: str) -> str:
    return "SIMS" if normalize_language_request(language) == "zh" else "CMUMOSI"


def dataset_for_text(text: str | None) -> str:
    return dataset_for_language(detect_text_language(text))
