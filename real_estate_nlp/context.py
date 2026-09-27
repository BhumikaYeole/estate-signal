"""Small explainable context rules for ambiguous real-estate vocabulary."""

from __future__ import annotations

AMBIGUOUS_SENSES = {
    "lot": {
        "land parcel": {"acre", "acres", "land", "sqft", "square", "zoning", "buildable"},
        "a great deal / sale": {"deal", "discount", "price", "offer", "bargain"},
    },
    "room": {
        "interior space": {"bedroom", "bathroom", "living", "kitchen", "suite", "floor"},
        "availability or negotiating space": {"price", "negotiate", "offer", "flexible"},
    },
    "yard": {
        "outdoor property area": {"fenced", "garden", "backyard", "patio", "acre", "lawn"},
        "measurement unit": {"feet", "meters", "measurement", "distance"},
    },
    "open": {
        "open-plan design": {"concept", "kitchen", "layout", "living", "floorplan"},
        "available for viewing": {"house", "viewing", "showing", "sunday", "visit"},
    },
}


def disambiguate(text: str) -> list[dict[str, str]]:
    words = set(text.lower().replace("-", " ").split())
    interpretations: list[dict[str, str]] = []
    for term, senses in AMBIGUOUS_SENSES.items():
        if term not in words:
            continue
        ranked = sorted(
            ((len(words & cues), sense) for sense, cues in senses.items()),
            reverse=True,
        )
        if ranked and ranked[0][0] > 0:
            interpretations.append({"term": term, "meaning": ranked[0][1], "confidence": "rule-based"})
        else:
            interpretations.append({"term": term, "meaning": "ambiguous in this context", "confidence": "low"})
    return interpretations
