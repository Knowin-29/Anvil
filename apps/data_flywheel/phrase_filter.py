"""Step3 增强: Phrase Filtering — 对齐论文去除含形容词的名词短语。"""

from __future__ import annotations

from typing import List, Set, Tuple

# 过于泛化、无法被检测模型定位的短语黑名单
_STOPLIST: Set[str] = {
    # 场景/图片级别
    "a scene", "scene", "an image", "image", "the image", "a photo", "photo",
    "a picture", "picture", "the room", "a room", "the picture", "the photo",
    # 泛化集合
    "objects", "various objects", "items", "things", "stuff", "elements",
    "features", "details", "components", "pieces", "parts",
    # 空间/位置
    "area", "room", "space", "place", "view", "location", "position",
    "background", "foreground", "environment", "setting", "surroundings",
    "surface", "side", "part", "section", "corner", "edge",
    "top", "bottom", "front", "back", "left", "right",
    "center", "middle", "the surface", "the side", "the part",
    # 抽象概念
    "color", "shape", "size", "texture", "material", "pattern",
    "style", "design", "arrangement", "layout", "composition",
    # 动作/状态
    "use", "way", "manner", "purpose", "function",
    # 数量
    "number", "amount", "variety", "collection", "group", "set",
    "pair", "bunch", "pile", "stack", "row", "line",
    # 时间/光照
    "light", "shadow", "reflection", "lighting", "atmosphere",
}


def filter_phrases(
    phrases_with_spans: List[Tuple[str, int, int]],
    nlp,
    remove_adjectives: bool = True,
) -> List[Tuple[str, int, int]]:
    """过滤名词短语：
    - 去除黑名单中的泛化短语（scene / objects / image 等）
    - 去除含形容词修饰的短语（论文 Phrase Filtering 策略）
    - 去除过短（<2字符）或过长（>50字符）的短语
    - 去除纯代词短语
    """
    filtered: List[Tuple[str, int, int]] = []
    for phrase, start, end in phrases_with_spans:
        text = phrase.strip()
        if len(text) < 2 or len(text) > 50:
            continue

        # 黑名单过滤
        if text.lower() in _STOPLIST:
            continue

        # 用 spaCy 分析短语内部结构
        doc = nlp(text)
        has_adj = any(tok.pos_ == "ADJ" for tok in doc)
        is_pronoun = all(tok.pos_ == "PRON" for tok in doc)

        if is_pronoun:
            continue
        if remove_adjectives and has_adj:
            # 尝试提取去掉形容词后的核心名词
            nouns = [tok.text for tok in doc if tok.pos_ in ("NOUN", "PROPN")]
            if nouns:
                core = " ".join(nouns)
                if core.lower() not in _STOPLIST:
                    filtered.append((core, start, end))
            continue

        filtered.append((text, start, end))

    # 去重
    seen = set()
    unique: List[Tuple[str, int, int]] = []
    for phrase, s, e in filtered:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            unique.append((phrase, s, e))
    return unique
