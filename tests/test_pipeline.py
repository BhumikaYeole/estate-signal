from real_estate_nlp.data import SAMPLE_LISTINGS
from real_estate_nlp.matching import ListingMatcher
from real_estate_nlp.pipeline import ListingAnalyzer


def test_preprocessing_removes_phone_currency_and_emoji_but_keeps_money_extraction():
    analyzer = ListingAnalyzer()
    result = analyzer.analyze("Bright balconies for $625,000! Call (415) 555-0138 🏡 in Austin, Texas")

    assert "415" not in result["cleaned_text"]
    assert "$" not in result["cleaned_text"]
    assert "🏡" not in result["cleaned_text"]
    assert "balcony" in result["normalized_tokens"]
    assert "$625,000" in result["entities"]["MONEY"]
    assert "Austin, Texas" in result["entities"]["GPE"]


def test_pipeline_extracts_phrases_and_context():
    result = ListingAnalyzer().analyze("Open concept kitchen with walk-in closet and fenced yard")

    assert "open concept kitchen" in result["collocations"]
    open_interpretation = next(item for item in result["context"] if item["term"] == "open")
    assert open_interpretation["meaning"] == "open-plan design"
    assert result["noun_phrases"]


def test_search_ranks_relevant_listing_first():
    analyzer = ListingAnalyzer()
    indexed = [analyzer.analyze(item["raw_text"], item["id"]) for item in SAMPLE_LISTINGS]
    results = ListingMatcher(indexed).search("Brooklyn loft rooftop terrace")

    assert results
    assert results[0]["listing_id"] == "RE-103"
    assert 0 < results[0]["similarity"] <= 1
