"""Cleaning and tokenization with optional spaCy/NLTK acceleration."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

try:
    import spacy
except ImportError:  # pragma: no cover - exercised in minimal installs
    spacy = None

try:
    from nltk.corpus import stopwords
except ImportError:  # pragma: no cover - exercised in minimal installs
    stopwords = None


FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "near", "of", "on", "or", "that", "the", "this",
    "to", "was", "with", "w", "bed", "beds", "bath", "baths",
}
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,2}[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}(?!\w)"
)
CURRENCY_PATTERN = re.compile(r"[$€£¥₹]")
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:['’-][a-zA-Z]+)?|\d+(?:\.\d+)?")


def _emoji_or_symbol(character: str) -> bool:
    category = unicodedata.category(character)
    return category in {"So", "Sk"} or "EMOJI" in unicodedata.name(character, "")


def load_stop_words() -> set[str]:
    if stopwords is not None:
        try:
            return set(stopwords.words("english")) | FALLBACK_STOP_WORDS
        except LookupError:
            pass
    return FALLBACK_STOP_WORDS


@dataclass(frozen=True)
class PreprocessedText:
    cleaned_text: str
    tokens: list[str]
    normalized_tokens: list[str]
    normalized_text: str


class TextPreprocessor:
    def __init__(self) -> None:
        self.stop_words = load_stop_words()
        self._nlp = None
        if spacy is not None:
            try:
                self._nlp = spacy.load("en_core_web_sm", disable=["ner"])
            except (OSError, IOError):
                self._nlp = None

    @property
    def spacy_available(self) -> bool:
        return self._nlp is not None

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Listing text must be a string.")
        text_without_phones = PHONE_PATTERN.sub(" ", text)
        text_without_currency = CURRENCY_PATTERN.sub(" ", text_without_phones)
        text_without_emoji = "".join(
            " " if _emoji_or_symbol(character) else character
            for character in text_without_currency
        )
        return re.sub(r"\s+", " ", text_without_emoji).strip()

    def lemmatize(self, token: str) -> str:
        if self._nlp is not None:
            parsed = self._nlp(token)
            if parsed and parsed[0].lemma_:
                return parsed[0].lemma_.lower()
        lower = token.lower()
        irregular = {"balconies": "balcony", "men": "man", "women": "woman", "feet": "foot"}
        if lower in irregular:
            return irregular[lower]
        if len(lower) > 4 and lower.endswith("ies"):
            return lower[:-3] + "y"
        if len(lower) > 4 and lower.endswith("ves"):
            return lower[:-3] + "f"
        if len(lower) > 4 and lower.endswith("ing"):
            stem = lower[:-3]
            return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
        if len(lower) > 3 and lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
            return lower[:-1]
        return lower

    def process(self, text: str) -> PreprocessedText:
        cleaned = self.clean(text)
        tokens = TOKEN_PATTERN.findall(cleaned.lower())
        filtered = [token for token in tokens if token not in self.stop_words]
        normalized = [self.lemmatize(token) for token in filtered]
        return PreprocessedText(
            cleaned_text=cleaned,
            tokens=tokens,
            normalized_tokens=normalized,
            normalized_text=" ".join(normalized),
        )
