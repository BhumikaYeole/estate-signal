"""Orchestration for end-to-end listing analysis."""

from __future__ import annotations

from typing import Any

from .context import disambiguate
from .entities import extract_entities
from .morphology import MorphologyAnalyzer
from .phrases import chunk_noun_phrases, extract_collocations
from .preprocessing import TextPreprocessor


class ListingAnalyzer:
    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()
        self.morphology = MorphologyAnalyzer(self.preprocessor)

    def analyze(self, raw_text: str, listing_id: str = "custom") -> dict[str, Any]:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Enter a non-empty listing description to analyze.")
        print(f"[pipeline] Analyzing listing {listing_id}")
        print("[pipeline] Step 1/5: cleaning, tokenization, and lemmatization")
        processed = self.preprocessor.process(raw_text)
        print("[pipeline] Step 2/5: POS tagging and morphology")
        morphology = self.morphology.analyze(processed.cleaned_text)
        adjectives, nouns = self.morphology.descriptors(morphology)
        print("[pipeline] Step 3/5: collocations and noun phrase chunks")
        collocations = extract_collocations(processed.cleaned_text, morphology)
        noun_phrases = chunk_noun_phrases(morphology)
        print("[pipeline] Step 4/5: named entity recognition")
        entities = extract_entities(raw_text, self.preprocessor)
        print("[pipeline] Step 5/5: contextual disambiguation")
        context = disambiguate(raw_text)
        return {
            "listing_id": listing_id,
            "raw_text": raw_text,
            "cleaned_text": processed.cleaned_text,
            "tokens": processed.tokens,
            "normalized_tokens": processed.normalized_tokens,
            "normalized_text": processed.normalized_text,
            "morphology": [token.as_dict() for token in morphology],
            "adjectives": adjectives,
            "nouns": nouns,
            "collocations": collocations,
            "noun_phrases": noun_phrases,
            "entities": entities,
            "context": context,
            "search_text": " ".join(processed.normalized_tokens),
            "nlp_backend": "spaCy en_core_web_sm" if self.preprocessor.spacy_available else "rule-based fallback",
        }
