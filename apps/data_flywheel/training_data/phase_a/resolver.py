from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence

from apps.data_flywheel.geometry import bbox_center
from apps.data_flywheel.types import SceneNode

from apps.data_flywheel.training_data.position_words import (
    extract_ordinal,
    extract_position_word,
    normalize_text,
    position_axis,
)


_LOGGER = logging.getLogger("training_data.resolver")


@dataclass
class _ResolveResult:
    status: str
    node_id: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    matched_rule: str = ""


_COLOR_ALIASES = {
    "red": ("red", "红", "红色", "scarlet", "crimson"),
    "orange": ("orange", "橙", "橙色"),
    "yellow": ("yellow", "黄", "黄色"),
    "green": ("green", "绿", "绿色"),
    "cyan": ("cyan", "青", "青色", "teal"),
    "blue": ("blue", "蓝", "蓝色", "navy"),
    "purple": ("purple", "violet", "紫", "紫色"),
    "pink": ("pink", "粉", "粉色"),
    "brown": ("brown", "棕", "棕色", "咖啡色"),
    "black": ("black", "黑", "黑色"),
    "white": ("white", "白", "白色"),
    "gray": ("gray", "grey", "灰", "灰色"),
}

_COLOR_ALIAS_TO_CANONICAL = {
    alias: canonical for canonical, aliases in _COLOR_ALIASES.items() for alias in aliases
}


def _normalize_node_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return normalize_text(value)


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9 \-']+", term):
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        return re.search(pattern, text) is not None
    return term in text


def _find_exact_matches(nodes: Sequence[SceneNode], phrase_norm: str, attr: str) -> List[SceneNode]:
    matches: List[SceneNode] = []
    for node in nodes:
        values = getattr(node, attr, None)
        if attr == "hierarchical_name":
            candidate_values = [values] if isinstance(values, str) and values else []
        else:
            candidate_values = [value for value in (values or []) if isinstance(value, str)]
        for candidate in candidate_values:
            if _normalize_node_text(candidate) == phrase_norm:
                matches.append(node)
                break
    return matches


def _canonical_color(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = _normalize_node_text(value)
    return _COLOR_ALIAS_TO_CANONICAL.get(normalized, normalized)


@lru_cache(maxsize=None)
def _ordered_color_aliases() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for canonical, aliases in _COLOR_ALIASES.items():
        for alias in aliases:
            pairs.append((canonical, alias))
    pairs.sort(key=lambda item: len(item[1]), reverse=True)
    return tuple(pairs)


def _extract_color_word(phrase: str) -> Optional[str]:
    normalized = _normalize_node_text(phrase)
    for canonical, alias in _ordered_color_aliases():
        if _contains_term(normalized, _normalize_node_text(alias)):
            return canonical
    return None


def _extract_slots(phrase: str, language: str) -> dict:
    normalized = _normalize_node_text(phrase)
    return {
        "core_noun": None,
        "color_word": _extract_color_word(normalized),
        "position_word": extract_position_word(normalized),
        "ordinal": extract_ordinal(normalized),
        "language": language,
        "normalized_phrase": normalized,
    }


def _find_core_noun_from_nodes(phrase: str, nodes: Sequence[SceneNode]) -> Optional[str]:
    normalized = _normalize_node_text(phrase)
    class_names = []
    seen = set()
    for node in nodes:
        class_name = _normalize_node_text(node.class_name)
        if class_name and class_name not in seen:
            seen.add(class_name)
            class_names.append(class_name)

    class_names.sort(key=lambda item: (-len(item), item))
    for class_name in class_names:
        if _contains_term(normalized, class_name):
            return class_name
    return None


def _load_nlp(language: str):
    try:
        import spacy
    except Exception:
        return None

    model_name = "en_core_web_sm" if not language.startswith("zh") else "zh_core_web_sm"
    try:
        return spacy.load(model_name)
    except Exception:
        try:
            return spacy.blank("en" if not language.startswith("zh") else "zh")
        except Exception:
            return None


def _singularize(token: str) -> str:
    token = token.strip().lower()
    if not token:
        return token
    irregular = {
        "men": "man",
        "women": "woman",
        "children": "child",
        "people": "person",
        "teeth": "tooth",
        "feet": "foot",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") or token.endswith("xes") or token.endswith("zes") or token.endswith("ches") or token.endswith("shes"):
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def _guess_core_noun_with_spacy(phrase: str, language: str) -> Optional[str]:
    nlp = _load_nlp(language)
    if nlp is None:
        return None

    try:
        doc = nlp(phrase)
    except Exception:
        return None

    noun_chunks = getattr(doc, "noun_chunks", None)
    if noun_chunks is not None:
        try:
            chunks = list(noun_chunks)
        except Exception:
            chunks = []
        for chunk in reversed(chunks):
            root = getattr(chunk, "root", None)
            if root is None:
                continue
            lemma = _singularize(getattr(root, "lemma_", "") or getattr(root, "text", ""))
            if lemma:
                return lemma

    for token in reversed(list(doc)):
        text = getattr(token, "lemma_", "") or getattr(token, "text", "")
        if not text:
            continue
        if getattr(token, "is_punct", False) or getattr(token, "is_space", False):
            continue
        lemma = _singularize(text)
        if lemma:
            return lemma
    return None


def _find_core_noun(phrase: str, nodes: Sequence[SceneNode], language: str) -> Optional[str]:
    class_name = _find_core_noun_from_nodes(phrase, nodes)
    if class_name:
        return class_name

    guessed = _guess_core_noun_with_spacy(phrase, language)
    if guessed:
        return guessed
    return None


def _sort_nodes(nodes: Sequence[SceneNode], axis: str) -> List[SceneNode]:
    if axis == "y":
        return sorted(
            nodes,
            key=lambda node: (bbox_center(node.bbox)[1], bbox_center(node.bbox)[0], node.id),
        )
    return sorted(
        nodes,
        key=lambda node: (bbox_center(node.bbox)[0], bbox_center(node.bbox)[1], node.id),
    )


def _match_by_position(
    candidates: List[SceneNode],
    position_word: Optional[str],
    ordinal: Optional[int],
) -> Optional[str]:
    if not candidates:
        return None

    axis = position_axis(position_word)
    ordered = _sort_nodes(candidates, axis)
    if position_word in {"right", "bottom"}:
        ordered = list(reversed(ordered))

    if position_word == "middle":
        return ordered[len(ordered) // 2].id

    if ordinal is not None:
        if 1 <= ordinal <= len(ordered):
            return ordered[ordinal - 1].id
        return None

    if position_word in {"left", "right", "top", "bottom"}:
        return ordered[0].id

    return None


def _make_result(node_ids: Iterable[str], matched_rule: str, status: str) -> _ResolveResult:
    ordered = list(node_ids)
    if status == "resolved" and len(ordered) == 1:
        return _ResolveResult(status=status, node_id=ordered[0], matched_rule=matched_rule)
    return _ResolveResult(status=status, candidates=ordered, matched_rule=matched_rule)


def resolve_reference(
    phrase: str,
    nodes: List[SceneNode],
    language: str = "en",
) -> _ResolveResult:
    normalized = _normalize_node_text(phrase)
    if not normalized:
        return _ResolveResult(status="not_found", matched_rule="empty")

    hierarchical_matches = _find_exact_matches(nodes, normalized, "hierarchical_name")
    if hierarchical_matches:
        if len(hierarchical_matches) == 1:
            return _ResolveResult(
                status="resolved",
                node_id=hierarchical_matches[0].id,
                matched_rule="hierarchical_name_exact",
            )
        return _ResolveResult(
            status="ambiguous",
            candidates=[node.id for node in hierarchical_matches],
            matched_rule="hierarchical_name_exact",
        )

    referring_matches = _find_exact_matches(nodes, normalized, "referring_expressions")
    if referring_matches:
        if len(referring_matches) == 1:
            return _ResolveResult(
                status="resolved",
                node_id=referring_matches[0].id,
                matched_rule="referring_expressions_exact",
            )
        return _ResolveResult(
            status="ambiguous",
            candidates=[node.id for node in referring_matches],
            matched_rule="referring_expressions_exact",
        )

    slots = _extract_slots(phrase, language)
    core_noun = _find_core_noun(phrase, nodes, language)
    if core_noun is None:
        return _ResolveResult(status="not_found", matched_rule="no_core_noun")

    slots["core_noun"] = core_noun

    core_norm = _normalize_node_text(core_noun)
    candidates = [node for node in nodes if _normalize_node_text(node.class_name) == core_norm]
    if not candidates:
        return _ResolveResult(status="not_found", matched_rule="class_not_found")

    color_word = slots.get("color_word")
    if color_word:
        candidates = [node for node in candidates if _canonical_color(node.attributes.color_primary) == color_word]
        if not candidates:
            return _ResolveResult(status="not_found", matched_rule="class_color_not_found")

    if len(candidates) == 1:
        matched_rule = "class"
        if color_word:
            matched_rule = "class_color"
        if slots.get("position_word") or slots.get("ordinal") is not None:
            matched_rule = f"{matched_rule}_position" if slots.get("position_word") else f"{matched_rule}_ordinal"
        return _ResolveResult(
            status="resolved",
            node_id=candidates[0].id,
            matched_rule=matched_rule,
        )

    position_word = slots.get("position_word")
    ordinal = slots.get("ordinal")
    if position_word or ordinal is not None:
        node_id = _match_by_position(candidates, position_word, ordinal)
        if node_id is not None:
            matched_rule = "class_position"
            if color_word:
                matched_rule = "class_color_position"
            elif ordinal is not None and not position_word:
                matched_rule = "class_ordinal"
            return _ResolveResult(
                status="resolved",
                node_id=node_id,
                matched_rule=matched_rule,
            )

    return _ResolveResult(
        status="ambiguous",
        candidates=[node.id for node in _sort_nodes(candidates, "x")],
        matched_rule="class_ambiguous",
    )

