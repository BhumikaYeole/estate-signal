"""TF-IDF retrieval with cosine similarity for listing search."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ListingMatcher:
    def __init__(self, listings: list[dict[str, Any]]) -> None:
        self.listings = listings
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
        documents = [listing["search_text"] or listing["raw_text"] for listing in listings]
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if not self.vectorizer.vocabulary_:
            return []
        query_vector = self.vectorizer.transform([normalized_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [
            {**self.listings[index], "similarity": float(score)}
            for index, score in ranked[:limit]
            if score > 0
        ]
