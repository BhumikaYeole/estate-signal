# Real Estate Listing Analysis System

A modular NLP mini-project that cleans noisy property descriptions, extracts linguistic and real-estate attributes, resolves a few context-dependent terms, and ranks listings against a free-text query.

## Features

- Regex cleaning for phone numbers, currency symbols, and emoji/symbol characters; tokenization, stop-word removal, and lemmatization with built-in fallbacks.
- Optional spaCy English-model POS tagging and morphology, including tense and grammatical number.
- Adjective/noun descriptors, bigram/trigram collocations, noun phrase chunks, and GPE/MONEY extraction.
- Explainable context rules for ambiguous terms such as `lot`, `room`, `yard`, and `open`.
- TF-IDF word n-grams and cosine similarity over four included sample listings.
- Streamlit UI for analyzing pasted listings and searching the mock dataset.

## Install and run

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

The app works without the spaCy model, using a deterministic lightweight fallback. NLTK stopwords are used if their corpus is installed; otherwise, a built-in stop-word set is used. To add NLTK stopwords and WordNet data explicitly:

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Run tests

```powershell
python -m pytest -q
```

The `real_estate_nlp` package separates preprocessing, morphology, phrase extraction, entities, contextual rules, matching, and pipeline orchestration. Pipeline steps also print progress messages when run from a terminal.
