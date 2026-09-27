"""Named entity extraction with spaCy and conservative regex fallbacks."""

from __future__ import annotations

import re

from .preprocessing import TextPreprocessor

MONEY_PATTERN = re.compile(
    r"(?i)(?:\$\s?\d[\d,]*(?:\.\d{1,2})?\s?(?:k|m|million|thousand)?|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:k|m|million|thousand)\b|\$\s?\d[\d,]*(?:/\s?(?:mo|month))?)"
)
LOCATION_PATTERN = re.compile(
    r"\b(?:San Francisco|New York|Los Angeles|Portland|Austin|Brooklyn|Seattle|Chicago|Boston|Denver|Miami)(?:,\s*[A-Z]{2}|,\s*(?:Texas|Oregon|California|New York))?\b",
    re.IGNORECASE,
)


def extract_entities(raw_text: str, preprocessor: TextPreprocessor) -> dict[str, list[str]]:
    locations = [match.group(0).strip(" ,") for match in LOCATION_PATTERN.finditer(raw_text)]
    money = [match.group(0).strip() for match in MONEY_PATTERN.finditer(raw_text)]
    nlp = preprocessor._nlp
    if nlp is not None:
        document = nlp(raw_text)
        for entity in document.ents:
            if entity.label_ in {"GPE", "LOC"}:
                locations.append(entity.text)
            elif entity.label_ == "MONEY":
                money.append(entity.text)
    return {
        "GPE": list(dict.fromkeys(locations)),
        "MONEY": list(dict.fromkeys(money)),
    }
