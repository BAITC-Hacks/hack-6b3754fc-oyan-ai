"""Baseline lexical excerpt selection, NOT neural semantic matching.

Only explicit event/category terms guide quote selection. This module neither
scores nor excludes contractors. Negations stay in the verbatim excerpt.
"""

import re

MATCHING_METHOD = "lexical_excerpt_v1"
_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
_BOUNDARIES = re.compile(r"[.!?]+(?=\s|$)")
_ABBREVIATIONS = {"г", "гг", "т", "д", "е", "п", "им", "ул", "пр", "др", "тыс", "млн", "млрд", "ч", "мин", "руб", "стр", "тел"}


def _tokens(text):
    return {word.casefold().replace("ё", "е") for word in _WORDS.findall(text)}


def _sentences(text):
    start = 0
    for boundary in _BOUNDARIES.finditer(text):
        words = _WORDS.findall(text[start:boundary.start()])
        if boundary.group() == "." and words:
            last = words[-1]
            if last.casefold() in _ABBREVIATIONS or (len(last) == 1 and last.isupper()):
                continue
        yield start, boundary.start()
        start = boundary.end()
    yield start, len(text)


def select_description_excerpt(description, query, max_chars=240):
    """Return source offsets and literal text, or None for an empty description.

    Exact-token overlap selects a complete short sentence; ties use source
    position. A zero overlap selects the first short sentence as context,
    without asserting semantic relevance. Long sentences are skipped, never
    clipped: a late negation or qualification must not be removed.
"""
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    terms = _tokens(" ".join(str(query.get(k) or "") for k in ("event_type", "category")))
    excerpts = []
    for begin, stop in _sentences(description):
        raw = description[begin:stop]
        start = begin + len(raw) - len(raw.lstrip())
        end = stop - len(raw) + len(raw.rstrip())
        if end <= start or end - start > max_chars:
            continue
        text = description[start:end]
        if not _WORDS.search(text):
            continue
        matched = sorted(terms & _tokens(text))
        excerpts.append({"value": text, "start": start, "end": end,
                         "matched_terms": matched})
    return min(excerpts, key=lambda item: (-len(item["matched_terms"]), item["start"])) if excerpts else None
