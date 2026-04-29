import streamlit as st
import pandas as pd
import ast
import requests
import csv
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LensLore · Film Discovery",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS  — Cinematic Editorial  (Playfair Display + DM Sans)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --gold:        #C9A84C;
    --gold-light:  #E8C97A;
    --gold-dim:    #7A6228;
    --bg:          #080808;
    --surface:     #101010;
    --surface2:    #181818;
    --border:      #1e1e1e;
    --text:        #F0E8D8;
    --text-muted:  #666;
    --text-dim:    #333;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 5rem !important; max-width: 1180px; margin: auto; }

::-webkit-scrollbar              { width: 5px; }
::-webkit-scrollbar-track        { background: var(--bg); }
::-webkit-scrollbar-thumb        { background: var(--gold-dim); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover  { background: var(--gold); }

/* ── Keyframes ── */
@keyframes apertureOpen {
    0%   { transform: scale(0.4) rotate(-60deg); opacity:0; }
    65%  { transform: scale(1.08) rotate(5deg); }
    100% { transform: scale(1) rotate(0deg); opacity:1; }
}
@keyframes titleReveal {
    0%   { opacity:0; letter-spacing: 0.5em; }
    100% { opacity:1; letter-spacing:-1px; }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(22px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes goldPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(201,168,76,0); }
    50%      { box-shadow: 0 0 18px 3px rgba(201,168,76,0.22); }
}
@keyframes filmScroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
@keyframes shimmerGold {
    0%   { background-position: -300% center; }
    100% { background-position:  300% center; }
}
@keyframes cardIn {
    from { opacity:0; transform: translateY(18px) scale(0.96); }
    to   { opacity:1; transform: translateY(0)    scale(1); }
}
@keyframes grain {
    0%,100% { transform: translate(0,0); }
    25%     { transform: translate(-1%,-2%); }
    50%     { transform: translate(2%,1%); }
    75%     { transform: translate(-1%,2%); }
}

/* Film grain */
body::before {
    content:'';
    position:fixed; inset:0; z-index:0; pointer-events:none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-size:180px 180px;
    opacity:0.025;
    animation: grain 0.35s steps(1) infinite;
}

/* ── Film strip ticker ── */
.strip-wrap {
    overflow:hidden;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    user-select:none;
}
.strip-inner {
    display:flex;
    width:max-content;
    animation: filmScroll 28s linear infinite;
    align-items:center;
    padding: 7px 0;
}
.s-hole {
    width:13px; height:9px;
    background:var(--bg);
    border-radius:2px;
    border:1px solid var(--border);
    margin:0 10px;
    flex-shrink:0;
}
.s-label {
    color:var(--gold-dim);
    font-size:0.6em;
    letter-spacing:0.28em;
    text-transform:uppercase;
    padding:0 6px;
    white-space:nowrap;
}

/* ── Hero ── */
.hero { text-align:center; padding:3.8rem 1rem 2rem; }
.hero-icon {
    font-size:3em; display:block; margin-bottom:0.3em;
    animation: apertureOpen 1s cubic-bezier(.34,1.56,.64,1) both;
}
.hero-logo {
    font-family:'Playfair Display', serif;
    font-size: clamp(3em, 8vw, 6em);
    font-weight:900;
    line-height:1;
    animation: titleReveal 1s cubic-bezier(.77,0,.18,1) 0.2s both;
}
.logo-l { color: var(--gold); }
.logo-r { color: var(--text); }
.hero-sub {
    font-size:0.78em;
    color:var(--text-muted);
    letter-spacing:0.38em;
    text-transform:uppercase;
    margin-top:0.9em;
    animation: fadeUp 0.8s ease 0.7s both;
}
.hero-line {
    width:70px; height:1.5px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.4rem auto 0;
    animation: fadeUp 0.8s ease 0.9s both;
}

/* ── Gold shimmer divider ── */
.divider {
    height:1px;
    background: linear-gradient(90deg, transparent 0%, var(--gold-dim) 25%, var(--gold) 50%, var(--gold-dim) 75%, transparent 100%);
    background-size:300% auto;
    animation: shimmerGold 3.5s linear infinite;
    margin: 2.2rem 0;
    border:none;
}

/* ── Section labels ── */
.s-eyebrow {
    font-size:0.68em;
    font-weight:600;
    letter-spacing:0.32em;
    text-transform:uppercase;
    color: var(--gold);
    margin-bottom:0.3em;
}
.s-heading {
    font-family:'Playfair Display', serif;
    font-size:1.65em;
    font-weight:700;
    color:var(--text);
    margin:0 0 1.1em;
    line-height:1.15;
}

/* ── Selectbox ── */
label[data-testid="stWidgetLabel"] {
    color:var(--text-muted) !important;
    font-size:0.74em !important;
    font-weight:500 !important;
    letter-spacing:0.15em !important;
    text-transform:uppercase !important;
}
div[data-baseweb="select"] > div {
    background-color:var(--surface2) !important;
    border:1px solid var(--border) !important;
    border-radius:8px !important;
    color:var(--text) !important;
    transition: border-color 0.25s !important;
}
div[data-baseweb="select"] > div:hover { border-color:var(--gold-dim) !important; }

/* ── Button ── */
.stButton > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1.5px solid var(--gold) !important;
    padding: 0.78em 2.6em !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82em !important;
    font-weight: 600 !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    transition: all 0.3s ease !important;
    animation: goldPulse 2.8s ease-in-out infinite !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: #080808 !important;
    box-shadow: 0 0 28px rgba(201,168,76,0.38) !important;
    transform: scale(1.04) !important;
}

/* ── Selected movie card ── */
.sel-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 12px;
    padding: 22px 26px;
    animation: fadeUp 0.65s ease both;
    height:100%;
}
.sel-title {
    font-family:'Playfair Display', serif;
    font-size:1.75em;
    font-weight:700;
    color:var(--text);
    margin:6px 0 4px;
    line-height:1.2;
}
.sel-year { color:var(--gold); font-size:0.88em; font-weight:500; margin-bottom:4px; }
.rating-pill {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(201,168,76,0.1);
    border:1px solid var(--gold-dim);
    color:var(--gold-light);
    font-size:0.8em; font-weight:600;
    border-radius:40px;
    padding:3px 13px;
}
.sel-overview {
    color:var(--text-muted);
    font-size:0.88em;
    line-height:1.68;
    margin: 12px 0 10px;
    border-top:1px solid var(--border);
    padding-top:12px;
}
.film-link {
    display:inline-flex; align-items:center; gap:5px;
    color:var(--gold);
    font-size:0.8em; font-weight:600;
    letter-spacing:0.06em;
    text-decoration:none;
    border-bottom:1px solid var(--gold-dim);
    padding-bottom:1px;
    transition:color 0.2s, border-color 0.2s;
    margin-top:6px;
}
.film-link:hover { color:var(--gold-light); border-color:var(--gold-light); }

/* ── Rec cards ── */
.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow:hidden;
    transition: transform 0.32s ease, box-shadow 0.32s ease, border-color 0.32s ease;
    animation: cardIn 0.5s ease both;
}
.rec-card:hover {
    transform: translateY(-8px) scale(1.015);
    box-shadow: 0 18px 44px rgba(0,0,0,0.65), 0 0 0 1px var(--gold-dim);
    border-color: var(--gold-dim);
}
.rc-body { padding:12px 14px 14px; }
.rc-title {
    font-family:'Playfair Display', serif;
    font-size:0.95em;
    font-weight:700;
    color:var(--text);
    margin:4px 0 3px;
    line-height:1.3;
}
.rc-meta { font-size:0.72em; color:var(--text-muted); margin-bottom:7px; }
.rc-link {
    display:inline-flex; align-items:center; gap:4px;
    font-size:0.7em;
    color:var(--gold);
    font-weight:600;
    text-decoration:none;
    border-bottom:1px dotted var(--gold-dim);
    letter-spacing:0.05em;
}
.rc-link:hover { color:var(--gold-light); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="metric-container"] label { color:var(--text-muted) !important; font-size:0.75em !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:var(--gold-light) !important; font-family:'Playfair Display', serif !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background:transparent !important;
    color:var(--text-muted) !important;
    border:1px solid var(--border) !important;
    border-radius:4px !important;
    font-size:0.76em !important;
    letter-spacing:0.1em !important;
    transition:all 0.25s !important;
}
.stDownloadButton > button:hover {
    border-color:var(--gold-dim) !important;
    color:var(--gold) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── Poster image ── */
.stImage img { border-radius: 8px 8px 0 0 !important; margin:0 !important; }

/* ── Footer ── */
.ll-footer {
    text-align:center;
    padding:3rem 0 1.5rem;
    color:var(--text-dim);
    font-size:0.7em;
    letter-spacing:0.22em;
    text-transform:uppercase;
    line-height:2;
}
.ll-footer strong { color:var(--gold-dim); font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FILM STRIP TICKER
# ══════════════════════════════════════════════════════════════
genres_ticker = [
    "Action", "Drama", "Thriller", "Romance", "Comedy",
    "Sci-Fi", "Horror", "Mystery", "Biography", "Fantasy",
    "Adventure", "Crime", "Animation", "History", "Musical"
]
holes = ''.join(
    f'<div class="s-hole"></div><span class="s-label">{g}</span>'
    for g in genres_ticker * 5
)
st.markdown(f"""
<div class="strip-wrap">
  <div class="strip-inner">{holes}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <span class="hero-icon">🎞️</span>
  <div class="hero-logo">
    <span class="logo-l">Lens</span><span class="logo-r">Lore</span>
  </div>
  <p class="hero-sub">Intelligent Cinema Discovery &nbsp;·&nbsp; Bollywood &amp; Hollywood</p>
  <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    boll = pd.read_csv('dataset/cleaned/bollywood_cleaned.csv')
    holl = pd.read_csv('dataset/cleaned/hollywood_cleaned.csv')
    return boll, holl

movie_boll, movie_holl = load_data()


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def parse_list(value):
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value) if isinstance(value, str) and value.startswith('[') else value.split(',')
    except Exception:
        return []


TMDB_HEADERS = {
    "accept": "application/json",
    "Authorization": (
        "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmMzhjMjk5NDU1ZDBiNTJjY"
        "2EyZTFlNzBjNjliNjJiMCIsIm5iZiI6MTc1Mzk1MTA4Ni44MTksInN1YiI6Ij"
        "Y4OGIyYjZlYTBhYTJlM2RkZDc2N2Q5MyIsInNjb3BlcyI6WyJhcGlfcmVhZCJd"
        "LCJ2ZXJzaW9uIjoxfQ.o2aZuhksRxjxvrD1UajVcNtdFDBQA5bPyj9CagvmHOA"
    )
}

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    if not movie_id:
        return None, None, None, None, None
    try:
        url  = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
        data = requests.get(url, headers=TMDB_HEADERS, timeout=6).json()
        poster_path = data.get('poster_path')
        poster_url  = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        homepage    = data.get('homepage') or None
        rating      = round(data.get('vote_average', 0), 1)
        year        = (data.get('release_date') or "")[:4]
        overview    = data.get('overview', "")
        return poster_url, homepage, rating, year, overview
    except Exception:
        return None, None, None, None, None


# ══════════════════════════════════════════════════════════════
#  OUTCOME LOGGER
# ══════════════════════════════════════════════════════════════
LOG_PATH = "logs/recommendation_log.csv"
LOG_COLS = ["timestamp","industry","genre_filter","actor_filter",
            "selected_movie","rec_1","rec_2","rec_3","rec_4","rec_5"]

def log_recommendation(industry, genre, actor, selected_movie, recs):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)
    row = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "industry":       industry,
        "genre_filter":   genre,
        "actor_filter":   actor,
        "selected_movie": selected_movie,
    }
    for i, rec in enumerate(recs[:5], 1):
        row[f"rec_{i}"] = rec
    for j in range(len(recs)+1, 6):
        row[f"rec_{j}"] = ""
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════
#  FILTERS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="s-eyebrow">Step 1 — Set the Stage</p>'
            '<p class="s-heading">Choose Your Cinema</p>', unsafe_allow_html=True)

industry    = st.selectbox('Industry', ('Bollywood', 'Hollywood'))
selected_df = (movie_boll if industry == 'Bollywood' else movie_holl).copy()

selected_df['genres_parsed'] = selected_df['genres'].apply(parse_list)
all_genres = sorted(set(
    g.strip().lower()
    for genres in selected_df['genres_parsed']
    for g in genres if isinstance(g, str)
))

col1, col2 = st.columns(2, gap="large")
with col2:
    genre = st.selectbox('Genre', ['All'] + all_genres, key='genre_select')
if genre != 'All':
    selected_df = selected_df[
        selected_df['genres_parsed'].apply(lambda x: genre.lower() in [g.lower() for g in x])
    ]

selected_df['cast_parsed'] = selected_df['cast'].apply(parse_list)
all_actors = sorted(set(a.strip() for cast in selected_df['cast_parsed'] for a in cast if isinstance(a, str)))
with col1:
    actor = st.selectbox('Actor', ['All'] + all_actors, key='actor_select')
if actor != 'All':
    selected_df = selected_df[
        selected_df['cast_parsed'].apply(lambda x: actor.lower() in [a.lower() for a in x])
    ]

selected_df = selected_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  GENRE LANDSCAPE CHART
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="s-eyebrow">Catalogue Insight</p>'
            '<p class="s-heading">Genre Landscape</p>', unsafe_allow_html=True)

genre_counts = {}
for genres in selected_df['genres_parsed']:
    for g in genres:
        if isinstance(g, str):
            k = g.strip().lower()
            genre_counts[k] = genre_counts.get(k, 0) + 1

if genre_counts:
    gdf = (
        pd.DataFrame(list(genre_counts.items()), columns=['Genre','Count'])
        .sort_values('Count', ascending=False).head(12)
    )
    fig_g = go.Figure(go.Bar(
        x=gdf['Genre'], y=gdf['Count'],
        marker=dict(
            color=gdf['Count'],
            colorscale=[[0,'#120d03'],[0.35,'#7A6228'],[0.7,'#C9A84C'],[1,'#E8C97A']],
            line=dict(width=0),
        ),
        text=gdf['Count'], textposition='outside',
        textfont=dict(color='#555', size=10, family='DM Sans'),
    ))
    fig_g.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#555', family='DM Sans'),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#555', size=11)),
        yaxis=dict(showgrid=True, gridcolor='#151515', zeroline=False, tickfont=dict(color='#444')),
        margin=dict(t=10, b=10, l=0, r=0), height=270,
    )
    st.plotly_chart(fig_g, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3, gap="medium")
    with mc1:
        st.metric("🎬  Movies in Selection", f"{len(selected_df):,}")
    with mc2:
        st.metric("🏆  Dominant Genre", gdf.iloc[0]['Genre'].title() if not gdf.empty else "—")
    with mc3:
        st.metric("🎭  Unique Genres", len(genre_counts))


# ══════════════════════════════════════════════════════════════
#  MOVIE SELECTOR
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="s-eyebrow">Step 2 — Pick Your Film</p>'
            '<p class="s-heading">Search the Archive</p>', unsafe_allow_html=True)

movie_list     = selected_df['title'].tolist()
selected_movie = st.selectbox("Title", sorted(movie_list), key="movie_select")


# ══════════════════════════════════════════════════════════════
#  SELECTED MOVIE PREVIEW
# ══════════════════════════════════════════════════════════════
sel_row = selected_df[selected_df['title'] == selected_movie]
if not sel_row.empty:
    sel_id = sel_row.iloc[0].get('movie_id') if 'movie_id' in sel_row.columns else None
    s_poster, s_homepage, s_rating, s_year, s_overview = fetch_movie_details(sel_id)

    pc1, pc2 = st.columns([1, 2], gap="large")
    with pc1:
        if s_poster:
            st.image(s_poster, use_container_width=True)
    with pc2:
        link_html = (
            f'<a href="{s_homepage}" target="_blank" class="film-link">🌐 Official Website ↗</a>'
            if s_homepage else ''
        )
        st.markdown(f"""
        <div class="sel-card">
            <div class="rating-pill">★ &nbsp;{s_rating} / 10</div>
            <div class="sel-title">{selected_movie}</div>
            <div class="sel-year">{s_year}</div>
            <div class="sel-overview">{s_overview or "No synopsis available."}</div>
            {link_html}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════
def recommend(movie_name_input):
    movies, posters, homepages, ratings, years, scores = [], [], [], [], [], []
    selected_df['tags'] = selected_df['tags'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else str(x)
    )
    if selected_df.shape[0] < 6:
        st.warning("Not enough films for similarity matching. Showing all results instead.")
        for _, row in selected_df.iterrows():
            mid = row.get('movie_id') if 'movie_id' in row else None
            p, h, r, y, _ = fetch_movie_details(mid)
            movies.append(row['title']); posters.append(p)
            homepages.append(h); ratings.append(r); years.append(y); scores.append(1.0)
        return movies, posters, homepages, ratings, years, scores

    tfidf   = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2)
    vectors = tfidf.fit_transform(selected_df['tags']).toarray()
    model   = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(vectors)

    idx                = selected_df[selected_df['title'] == movie_name_input].index[0]
    distances, indices = model.kneighbors([vectors[idx]], n_neighbors=6)

    for dist, i in zip(distances[0][1:], indices[0][1:]):
        movie = selected_df.iloc[i]
        mid   = movie.get('movie_id') if 'movie_id' in movie else None
        p, h, r, y, _ = fetch_movie_details(mid)
        movies.append(movie['title']); posters.append(p)
        homepages.append(h); ratings.append(r); years.append(y)
        scores.append(round(1 - dist, 4))

    return movies, posters, homepages, ratings, years, scores


# ══════════════════════════════════════════════════════════════
#  DISCOVER BUTTON
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
bc = st.columns([2,1,2])
with bc[1]:
    if st.button("✦  Discover Films"):
        st.session_state.recommend_triggered = True


# ══════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════
if st.session_state.get("recommend_triggered"):
    with st.spinner("Scanning the archive…"):
        recs, posters, homepages, ratings, years, scores = recommend(selected_movie)

    log_recommendation(industry, genre, actor, selected_movie, recs)

    if recs:
        # ── Similarity chart ──────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="s-eyebrow">Match Analysis</p>'
                    '<p class="s-heading">Similarity Scores</p>', unsafe_allow_html=True)

        sdf = pd.DataFrame({'Film': recs, 'Score': scores}).sort_values('Score', ascending=True)
        fig_s = go.Figure(go.Bar(
            x=sdf['Score'], y=sdf['Film'], orientation='h',
            marker=dict(
                color=sdf['Score'],
                colorscale=[[0,'#120d03'],[0.4,'#7A6228'],[0.8,'#C9A84C'],[1,'#E8C97A']],
                line=dict(width=0),
            ),
            text=[f"{s:.1%}" for s in sdf['Score']],
            textposition='outside',
            textfont=dict(color='#888', size=11, family='DM Sans'),
        ))
        fig_s.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#555', family='DM Sans'),
            xaxis=dict(range=[0,1.18], showgrid=False, zeroline=False,
                       tickformat='.0%', tickfont=dict(color='#444')),
            yaxis=dict(showgrid=False, tickfont=dict(color='#888', size=12)),
            margin=dict(t=10, b=10, l=10, r=60), height=240,
        )
        st.plotly_chart(fig_s, use_container_width=True)

        # ── Poster grid ───────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="s-eyebrow">Step 3 — Your Picks</p>'
                    '<p class="s-heading">Curated Selections</p>', unsafe_allow_html=True)

        def rec_card(col, title, poster, homepage, rating, year, delay_ms=0):
            with col:
                link_html = (
                    f'<a href="{homepage}" target="_blank" class="rc-link">🌐 Website ↗</a>'
                    if homepage else ''
                )
                st.markdown(
                    f'<div class="rec-card" style="animation-delay:{delay_ms}ms">',
                    unsafe_allow_html=True
                )
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"""
                <div class="rc-body">
                    <div class="rc-title">{title}</div>
                    <div class="rc-meta">★ {rating} &nbsp;·&nbsp; {year}</div>
                    {link_html}
                </div></div>
                """, unsafe_allow_html=True)

        r1 = st.columns(2, gap="medium")
        for i in range(min(2, len(recs))):
            rec_card(r1[i], recs[i], posters[i], homepages[i], ratings[i], years[i], i*110)

        if len(recs) > 2:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            r2 = st.columns(2, gap="medium")
            for j in range(2, min(4, len(recs))):
                rec_card(r2[j-2], recs[j], posters[j], homepages[j], ratings[j], years[j], j*110)

        if len(recs) > 4:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            r3 = st.columns([1,2,1], gap="medium")
            rec_card(r3[1], recs[4], posters[4], homepages[4], ratings[4], years[4], 480)

    else:
        st.error("No recommendations found.")

    # ── Log export ────────────────────────────────────────
    if os.path.isfile(LOG_PATH):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        with open(LOG_PATH, "rb") as f:
            st.download_button(
                label="⬇  Export Recommendation Log  (.csv)",
                data=f,
                file_name="lenslore_recommendation_log.csv",
                mime="text/csv",
                help="Download logs for model retraining & analysis"
            )


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="ll-footer">
    <strong>LensLore</strong> &nbsp;·&nbsp; Intelligent Film Discovery
    &nbsp;&nbsp;|&nbsp;&nbsp; TF-IDF &amp; Cosine Similarity
    &nbsp;&nbsp;|&nbsp;&nbsp; Powered by <strong>TMDB API</strong>
    <br>Built with Streamlit &nbsp;·&nbsp; &copy; 2025
</div>
""", unsafe_allow_html=True)
