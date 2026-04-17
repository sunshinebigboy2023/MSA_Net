from __future__ import annotations


def _count_cjk(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _count_latin(text: str) -> int:
    return sum(1 for char in text if ("a" <= char.lower() <= "z"))


def detect_text_language(text: str | None) -> str:
    if not text:
        return "unknown"
    if _count_cjk(text) > 0:
        return "zh"
    if _count_latin(text) > 0:
        return "en"
    return "unknown"


def dataset_for_language(language: str) -> str:
    return "SIMS" if language == "zh" else "CMUMOSI"


def dataset_for_text(text: str | None) -> str:
    return dataset_for_language(detect_text_language(text))
