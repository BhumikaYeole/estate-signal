"""Multiword collocations and regex-inspired noun phrase chunks."""

from __future__ import annotations

import re
from collections import Counter

from .morphology import MorphToken

KNOWN_PHRASES = (
    "open concept kitchen", "open-concept kitchen", "walk in closet", "walk-in closet",
    "floor to ceiling windows", "floor-to-ceiling windows", "hardwood floors", "finished basement",
    "rooftop terrace", "fenced yard", "private garden", "two car garage", "2 car garage",
)


def extract_collocations(text: str, tokens: list[MorphToken], limit: int = 12) -> list[str]:
    lowered = text.lower()
    found = [phrase for phrase in KNOWN_PHRASES if phrase in lowered]
    lemmas = [token.lemma for token in tokens if token.pos in {"ADJ", "NOUN"}]
    counts: Counter[str] = Counter()
    for size in (2, 3):
        counts.update(" ".join(lemmas[index:index + size]) for index in range(len(lemmas) - size + 1))
    generic = [phrase for phrase, count in counts.most_common() if count > 1]
    return list(dict.fromkeys(found + generic))[:limit]


def chunk_noun_phrases(tokens: list[MorphToken]) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token.pos in {"ADJ", "NOUN", "NUM"}:
            current.append(token.text)
        else:
            if len(current) >= 2:
                chunks.append(" ".join(current))
            current = []
    if len(current) >= 2:
        chunks.append(" ".join(current))
    return list(dict.fromkeys(chunks))
