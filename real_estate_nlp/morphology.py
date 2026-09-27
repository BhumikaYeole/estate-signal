"""POS descriptors and lightweight morphology extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .preprocessing import TOKEN_PATTERN, TextPreprocessor

ADJECTIVES = {
    "bright", "charming", "cozy", "finished", "fenced", "gorgeous", "large", "luxury",
    "modern", "private", "renovated", "spacious", "sunny", "updated", "beautiful",
}
KNOWN_FEATURES = {
    "balcony", "basement", "bathroom", "bedroom", "closet", "concierge", "floor", "garden",
    "garage", "kitchen", "loft", "terrace", "window", "yard", "hardwood", "concept",
}


@dataclass(frozen=True)
class MorphToken:
    text: str
    lemma: str
    pos: str
    tense: str | None
    number: str | None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


class MorphologyAnalyzer:
    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self.preprocessor = preprocessor
        self._nlp = preprocessor._nlp

    def analyze(self, text: str) -> list[MorphToken]:
        if self._nlp is not None:
            document = self._nlp(text)
            return [
                MorphToken(
                    text=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_ or "X",
                    tense=token.morph.get("Tense")[0] if token.morph.get("Tense") else None,
                    number=token.morph.get("Number")[0] if token.morph.get("Number") else None,
                )
                for token in document
                if not token.is_space and not token.is_punct
            ]
        output: list[MorphToken] = []
        for token in TOKEN_PATTERN.findall(text.lower()):
            lemma = self.preprocessor.lemmatize(token)
            if token in ADJECTIVES or token.endswith(("ous", "ful", "ive", "able", "y")):
                pos, number = "ADJ", None
            elif token.isdigit():
                pos, number = "NUM", None
            elif token.endswith("ing"):
                pos, number = "VERB", None
            elif lemma in KNOWN_FEATURES or token.endswith("s"):
                pos, number = "NOUN", "Plur" if token != lemma else "Sing"
            else:
                pos, number = "NOUN", "Sing"
            tense = "Pres" if pos == "VERB" and token.endswith("ing") else None
            output.append(MorphToken(token, lemma, pos, tense, number))
        return output

    @staticmethod
    def descriptors(tokens: list[MorphToken]) -> tuple[list[str], list[str]]:
        adjectives = list(dict.fromkeys(item.lemma for item in tokens if item.pos == "ADJ"))
        nouns = list(dict.fromkeys(item.lemma for item in tokens if item.pos == "NOUN"))
        return adjectives, nouns
