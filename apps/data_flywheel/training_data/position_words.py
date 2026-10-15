from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional


POSITION_ALIASES = {
    "left": (
        "leftmost",
        "left side",
        "on the left",
        "to the left",
        "left",
        "最左边",
        "最左侧",
        "最左",
        "左边",
        "左侧",
        "左边的",
        "左边数",
        "左",
    ),
    "right": (
        "rightmost",
        "right side",
        "on the right",
        "to the right",
        "right",
        "最右边",
        "最右侧",
        "最右",
        "右边",
        "右侧",
        "右边的",
        "右边数",
        "右",
    ),
    "middle": (
        "middle",
        "center",
        "centre",
        "mid",
        "middle of the image",
        "in the middle",
        "in the center",
        "中间",
        "中央",
        "中部",
        "居中",
    ),
    "top": (
        "top",
        "upper",
        "on top",
        "at the top",
        "above",
        "上面",
        "上方",
        "顶部",
        "最上",
        "最上面",
    ),
    "bottom": (
        "bottom",
        "lower",
        "at the bottom",
        "below",
        "under",
        "beneath",
        "下面",
        "下方",
        "底部",
        "最下",
        "最下面",
    ),
}


ORDINAL_WORDS_EN = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
}


_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


_EN_ORDINAL_PATTERN = re.compile(
    r"\b(?P<ordinal>"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|"
    r"[0-9]+(?:st|nd|rd|th)?"
    r")\b",
    re.IGNORECASE,
)
_ZH_ORDINAL_PATTERN = re.compile(r"第\s*(?P<ordinal>[一二三四五六七八九十零〇两0-9]+)\s*(?:个|名|位|只|次)?")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _alias_matches(text: str, alias: str) -> bool:
    alias_norm = normalize_text(alias)
    if not alias_norm:
        return False
    if re.fullmatch(r"[a-z0-9 \-']+", alias_norm):
        pattern = rf"(?<!\w){re.escape(alias_norm)}(?!\w)"
        return re.search(pattern, text) is not None
    return alias_norm in text


@lru_cache(maxsize=None)
def _ordered_position_aliases() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for canonical, aliases in POSITION_ALIASES.items():
        for alias in aliases:
            pairs.append((canonical, alias))
    pairs.sort(key=lambda item: len(item[1]), reverse=True)
    return tuple(pairs)


def extract_position_word(text: str) -> Optional[str]:
    normalized = normalize_text(text)
    for canonical, alias in _ordered_position_aliases():
        if _alias_matches(normalized, alias):
            return canonical
    return None


def _parse_chinese_number(value: str) -> Optional[int]:
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value in _CHINESE_DIGITS and value not in {"十"}:
        return _CHINESE_DIGITS[value]
    if value == "十":
        return 10

    if "十" in value:
        left, right = value.split("十", 1)
        tens = _CHINESE_DIGITS.get(left, 1) if left else 1
        ones = _CHINESE_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones

    return _CHINESE_DIGITS.get(value)


def extract_ordinal(text: str) -> Optional[int]:
    normalized = normalize_text(text)

    match = _ZH_ORDINAL_PATTERN.search(normalized)
    if match:
        parsed = _parse_chinese_number(match.group("ordinal"))
        if parsed is not None:
            return parsed

    match = _EN_ORDINAL_PATTERN.search(normalized)
    if match:
        token = match.group("ordinal").lower()
        if token in ORDINAL_WORDS_EN:
            return ORDINAL_WORDS_EN[token]
        token = re.sub(r"(st|nd|rd|th)$", "", token)
        if token.isdigit():
            return int(token)

    return None


def position_axis(position_word: Optional[str]) -> str:
    if position_word in {"top", "bottom"}:
        return "y"
    return "x"

