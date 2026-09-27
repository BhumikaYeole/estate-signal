"""Interactive Streamlit interface for real-estate listing analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from real_estate_nlp.data import SAMPLE_LISTINGS
from real_estate_nlp.matching import ListingMatcher
from real_estate_nlp.pipeline import ListingAnalyzer

st.set_page_config(page_title="Estate Signal | Listing Intelligence", page_icon="⌂", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#182923; --muted:#526158; --paper:#f4f4ec; --panel:#fffef9; --green:#176b4c; --lime:#d9ec82; --line:#dce2d7; --coral:#e87e5d; }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); color-scheme:light; --text-color:var(--ink); --secondary-text-color:var(--muted); --background-color:var(--paper); --secondary-background-color:var(--panel); --primary-color:var(--green); }
    header[data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1280px; padding-top:2.2rem; padding-bottom:4rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:0 !important; }
    .stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] *, .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color:var(--ink) !important; }
    .eyebrow { font:500 11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:0; color:var(--green); }
    .mast { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding:0 0 18px; margin-bottom:32px; }
    .brand { font-weight:800; font-size:16px; letter-spacing:0; }
    .brand-mark { display:inline-grid; place-items:center; width:30px; height:30px; background:var(--green); color:white; margin-right:9px; border-radius:4px; }
    .hero { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; margin-bottom:28px; }
    .hero h1 { font-size:38px; line-height:1.05; margin:8px 0; font-weight:800; }
    .hero p { max-width:510px; color:var(--muted); margin:0; }
    .stamp { border:1px solid var(--line); padding:10px 14px; font:11px 'DM Mono',monospace; white-space:nowrap; color:var(--green); }
    [data-testid="stTabs"] [role="tablist"] { border-bottom:1px solid var(--line); gap:22px; }
    [data-testid="stTabs"] button[role="tab"] { color:var(--muted); font-weight:700; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:var(--green); }
    [data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input { background:var(--panel); color:var(--ink) !important; border-color:var(--line); border-radius:5px; caret-color:var(--green); }
    [data-testid="stTextArea"] textarea::placeholder,[data-testid="stTextInput"] input::placeholder { color:var(--muted) !important; opacity:1; }
    [data-baseweb="select"] > div { background:var(--panel); color:var(--ink) !important; }
    [data-baseweb="select"] *, [role="option"] { color:var(--ink) !important; }
    [role="option"] { background:var(--panel); }
    [data-testid="stButton"] button { background:var(--green); color:white; border:0; border-radius:4px; font-weight:700; min-height:42px; }
    [data-testid="stButton"] button:hover { background:#10573d; color:white; }
    .metric { background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--green); padding:13px 15px; min-height:88px; }
    .metric .label { color:var(--muted); font:10px 'DM Mono',monospace; text-transform:uppercase; }
    .metric .value { font-weight:800; font-size:23px; margin-top:5px; }
    .result { background:var(--panel); border:1px solid var(--line); border-radius:5px; padding:18px 20px; margin:10px 0; }
    .result-id { color:var(--green); font:11px 'DM Mono',monospace; }
    .score { background:var(--lime); color:var(--ink); padding:5px 8px; font:11px 'DM Mono',monospace; float:right; }
    .muted { color:var(--muted); font-size:13px; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); --gdg-bg-cell:var(--panel); --gdg-text-dark:var(--ink); --gdg-text-medium:var(--muted); --gdg-text-light:var(--muted); --gdg-header-bg:#e8ece4; --gdg-header-fg:var(--ink); }
    @media(max-width:700px) { .hero { display:block; } .hero h1 { font-size:31px; } .stamp { display:inline-block; margin-top:15px; } }
    </style>
    <div class="mast"><div class="brand"><span class="brand-mark">⌂</span>ESTATE SIGNAL</div><div class="eyebrow">LISTING INTELLIGENCE / FIELD TOOL 01</div></div>
    <div class="hero"><div><div class="eyebrow">LANGUAGE, MADE LEGIBLE</div><h1>Property, parsed.</h1><p>Turn noisy listing copy into structured signals, then find the details that matter across the sample inventory.</p></div><div class="stamp">NLP ENGINE&nbsp;&nbsp; / &nbsp;&nbsp;READY</div></div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_analyzer() -> ListingAnalyzer:
    return ListingAnalyzer()


@st.cache_data
def get_analyzed_samples() -> list[dict]:
    analyzer = get_analyzer()
    return [analyzer.analyze(item["raw_text"], item["id"]) for item in SAMPLE_LISTINGS]


analyzer = get_analyzer()
sample_results = get_analyzed_samples()


def show_analysis(result: dict) -> None:
    metrics = st.columns(4)
    metric_values = [
        ("TOKENS", len(result["tokens"])),
        ("FEATURE NOUNS", len(result["nouns"])),
        ("PHRASES", len(result["collocations"])),
        ("ENTITIES", len(result["entities"]["GPE"]) + len(result["entities"]["MONEY"])),
    ]
    for column, (label, value) in zip(metrics, metric_values):
        column.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

    st.caption(f"Analysis backend: {result['nlp_backend']}")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Extraction matrix")
        st.dataframe(
            pd.DataFrame([
                {"Signal": "Quality descriptors", "Extracted values": ", ".join(result["adjectives"]) or "None"},
                {"Signal": "Property features", "Extracted values": ", ".join(result["nouns"]) or "None"},
                {"Signal": "Collocations", "Extracted values": "; ".join(result["collocations"]) or "None"},
                {"Signal": "Noun phrase chunks", "Extracted values": "; ".join(result["noun_phrases"]) or "None"},
                {"Signal": "Locations (GPE)", "Extracted values": ", ".join(result["entities"]["GPE"]) or "None"},
                {"Signal": "Monetary values", "Extracted values": ", ".join(result["entities"]["MONEY"]) or "None"},
                {"Signal": "Context", "Extracted values": "; ".join(f"{item['term']}: {item['meaning']}" for item in result["context"]) or "No ambiguous terms found"},
            ]),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.subheader("Normalized text")
        st.code(result["normalized_text"] or "No tokens after cleanup", language=None)
        st.subheader("Token morphology")
        st.dataframe(pd.DataFrame(result["morphology"]), hide_index=True, use_container_width=True, height=270)


analysis_tab, search_tab = st.tabs(["Analyze listing", "Search inventory"])
with analysis_tab:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<div class="eyebrow">01 / INPUT</div>', unsafe_allow_html=True)
        sample_choice = st.selectbox("Start with a sample or paste your own", ["Paste a listing"] + [item["id"] for item in SAMPLE_LISTINGS])
        selected = next((item for item in SAMPLE_LISTINGS if item["id"] == sample_choice), None)
        initial_text = selected["raw_text"] if selected else ""
        raw_listing = st.text_area("Raw listing text", value=initial_text, height=200, placeholder="Paste a property listing, including its original pricing and contact details...")
        analyze_clicked = st.button("Analyze listing", type="primary", use_container_width=True)
    with right:
        st.markdown('<div class="eyebrow">02 / INCLUDED INVENTORY</div>', unsafe_allow_html=True)
        st.write("Four noisy listings are ready to explore.")
        for sample in SAMPLE_LISTINGS:
            st.markdown(f"**{sample['id']}**  <span class='muted'>{sample['raw_text']}</span>", unsafe_allow_html=True)
    if analyze_clicked:
        try:
            with st.spinner("Running the extraction pipeline..."):
                result = analyzer.analyze(raw_listing, selected["id"] if selected else "custom")
            show_analysis(result)
        except (TypeError, ValueError) as error:
            st.error(str(error))
    elif selected:
        show_analysis(next(result for result in sample_results if result["listing_id"] == selected["id"]))

with search_tab:
    st.markdown('<div class="eyebrow">RETRIEVAL / TF-IDF + COSINE SIMILARITY</div>', unsafe_allow_html=True)
    query = st.text_input("Search properties", placeholder="e.g. bright loft with rooftop terrace in Brooklyn")
    if query.strip():
        matcher = ListingMatcher(sample_results)
        matches = matcher.search(query)
        if not matches:
            st.info("No meaningful match found in the sample inventory. Try a feature, room type, or location.")
        for match in matches:
            score_percent = round(match["similarity"] * 100)
            st.markdown(
                f"<div class='result'><span class='score'>{score_percent}% MATCH</span><div class='result-id'>{match['listing_id']}</div><p>{match['raw_text']}</p><div class='muted'>Features: {', '.join(match['nouns'][:8])} &nbsp; / &nbsp; Locations: {', '.join(match['entities']['GPE']) or 'Not detected'}</div></div>",
                unsafe_allow_html=True,
            )
