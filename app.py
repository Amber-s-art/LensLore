import streamlit as st
import pandas as pd
import ast
import requests
import csv
import os
import html
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from PIL import Image



# ══════════════════════════════════════════════════════════════
#  Load the image from your assets folder
# ══════════════════════════════════════════════════════════════
img_icon = Image.open("assets/logo.png")


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LensLore · Film Discovery",
    page_icon=img_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS  — Enhanced Cinematic Editorial 
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
    --text-muted:  #888;
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
    from { opacity:0; transform:translateY(30px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes buttonGlow {
    0% { box-shadow: 0 0 10px rgba(201,168,76,0.2); border-color: var(--gold-dim); }
    50% { box-shadow: 0 0 25px rgba(201,168,76,0.6); border-color: var(--gold-light); }
    100% { box-shadow: 0 0 10px rgba(201,168,76,0.2); border-color: var(--gold-dim); }
}
@keyframes filmScroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
@keyframes shimmerGold {
    0%   { background-position: -300% center; }
    100% { background-position:  300% center; }
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
    opacity:0.035;
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
.hero { text-align:center; padding:3.8rem 1rem 2rem; position:relative; z-index:1; }
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

/* ── Button (Enhanced) ── */
.stButton > button {
    background: linear-gradient(45deg, rgba(8,8,8,1) 0%, rgba(201,168,76,0.1) 50%, rgba(8,8,8,1) 100%) !important;
    background-size: 200% auto !important;
    color: var(--gold-light) !important;
    border: 1.5px solid var(--gold) !important;
    padding: 0.9em 2.8em !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88em !important;
    font-weight: 600 !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    animation: buttonGlow 3s infinite, shimmerGold 3s linear infinite !important;
    margin: 0 auto;
    display: block;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: #080808 !important;
    box-shadow: 0 0 35px rgba(201,168,76,0.5) !important;
    transform: scale(1.05) translateY(-3px) !important;
    animation: none !important;
}

/* ── Custom Movie Cards (With Hover Reveal) ── */
.rec-card-link {
    text-decoration: none;
    display: block;
    border-radius: 12px;
    outline: none;
}
.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
    animation: fadeUp 0.8s ease backwards;
    position: relative;
    cursor: pointer;
    height: 100%;
}
.rec-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.8), 0 0 0 1px var(--gold);
}
.card-image-wrapper {
    position: relative;
    width: 100%;
    aspect-ratio: 2/3;
    overflow: hidden;
    background: #111;
}
.card-image-wrapper img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: filter 0.4s ease, transform 0.6s ease;
}
.card-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(8, 8, 8, 0.85);
    color: #fff;
    opacity: 0;
    transition: opacity 0.4s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    text-align: center;
    font-size: 0.9em;
    line-height: 1.6;
    border-bottom: 1px solid var(--gold-dim);
}
.rec-card:hover .card-image-wrapper img {
    filter: blur(6px) brightness(0.5);
    transform: scale(1.08);
}
.rec-card:hover .card-overlay {
    opacity: 1;
}
.rc-body { 
    padding: 16px; 
    border-top: 1px solid var(--border);
}
.rc-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1em;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 6px;
    line-height: 1.2;
}
.rc-meta { 
    font-size: 0.8em; 
    color: var(--gold); 
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* ── Selected movie card layout ── */
.sel-card-wrapper {
    display: flex;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 12px;
    overflow: hidden;
    animation: fadeUp 0.65s ease both;
}
.sel-poster-wrap {
    flex: 0 0 35%;
    position: relative;
    overflow: hidden;
}
.sel-poster-wrap img {
    width: 100%; height: 100%; object-fit: cover;
    transition: filter 0.4s ease, transform 0.6s ease;
}
.sel-overlay {
    position: absolute; inset:0;
    background: rgba(8, 8, 8, 0.85);
    opacity: 0; transition: opacity 0.4s;
    display: flex; align-items: center; justify-content: center;
    padding: 20px; text-align: center; font-size: 0.9em;
}
.sel-card-wrapper:hover .sel-poster-wrap img {
    filter: blur(6px) brightness(0.5); transform: scale(1.05);
}
.sel-card-wrapper:hover .sel-overlay { opacity: 1; }
.sel-info {
    flex: 1; padding: 25px;
    display: flex; flex-direction: column; justify-content: center;
}
.rating-pill {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(201,168,76,0.1);
    border:1px solid var(--gold-dim);
    color:var(--gold-light);
    font-size:0.8em; font-weight:600;
    border-radius:40px;
    padding:3px 13px;
    width: fit-content;
    margin-bottom: 12px;
}
.sel-title { font-family:'Playfair Display', serif; font-size:2em; font-weight:700; margin:0 0 5px; line-height:1.1; }
.sel-year { color:var(--text-muted); font-size:1em; letter-spacing: 0.1em; margin-bottom:15px; }
.sel-click-hint { color: var(--gold-dim); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.1em; margin-top: auto; padding-top: 15px;}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

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
st.markdown(f"""
<div class="hero">
  <img src="data:image/png;base64,{img_icon}" class="hero-icon" style="height: 60px; width: auto; margin-bottom: 10px;">
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
    # Make sure your dataset paths are correct relative to where you run this
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
        # Appended videos to fetch trailers directly
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US&append_to_response=videos"
        data = requests.get(url, headers=TMDB_HEADERS, timeout=6).json()
        
        poster_path = data.get('poster_path')
        poster_url  = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        
        # Determine the best link (Trailer -> Homepage -> TMDB page)
        videos = data.get('videos', {}).get('results', [])
        trailer_url = None
        for v in videos:
            if v.get('type') == 'Trailer' and v.get('site') == 'YouTube':
                trailer_url = f"https://www.youtube.com/watch?v={v.get('key')}"
                break
                
        homepage = data.get('homepage')
        target_link = trailer_url or homepage or f"https://www.themoviedb.org/movie/{movie_id}"
        
        rating   = round(data.get('vote_average', 0), 1)
        year     = (data.get('release_date') or "")[:4]
        overview = data.get('overview', "No synopsis available.")
        
        return poster_url, target_link, rating, year, overview
    except Exception:
        return None, None, None, None, None


# ══════════════════════════════════════════════════════════════
#  OUTCOME LOGGER (SILENT BACKEND LOGGING)
# ══════════════════════════════════════════════════════════════
LOG_PATH = "logs/recom.csv"
LOG_COLS = ["timestamp", "industry", "genre_filter", "actor_filter",
            "selected_movie", "rec_1", "rec_2", "rec_3", "rec_4", "rec_5"]

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
    s_poster, s_link, s_rating, s_year, s_overview = fetch_movie_details(sel_id)

    safe_title = html.escape(selected_movie)
    safe_overview = html.escape(s_overview or "No synopsis available.")
    display_poster = s_poster if s_poster else "https://via.placeholder.com/500x750?text=No+Poster"

    # Display as a cohesive custom HTML card
    st.markdown(f"""
    <a href="{s_link}" target="_blank" style="text-decoration: none; display: block; margin-top: 15px;">
        <div class="sel-card-wrapper">
            <div class="sel-poster-wrap">
                <img src="{display_poster}" alt="{safe_title}">
                <div class="sel-overlay">
                    <p>Click to watch trailer or visit official site.</p>
                </div>
            </div>
            <div class="sel-info">
                <div class="rating-pill">★ &nbsp;{s_rating} / 10</div>
                <div class="sel-title">{safe_title}</div>
                <div class="sel-year">{s_year}</div>
                <div style="color:var(--text-muted); font-size:0.95em; line-height:1.6; padding-top: 15px; border-top: 1px solid var(--border);">
                    {safe_overview}
                </div>
                <div class="sel-click-hint">▶ Click card to view media</div>
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════
def recommend(movie_name_input):
    movies, posters, links, ratings, years, overviews = [], [], [], [], [], []
    selected_df['tags'] = selected_df['tags'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else str(x)
    )
    if selected_df.shape[0] < 6:
        st.warning("Not enough films for similarity matching. Showing all results instead.")
        for _, row in selected_df.iterrows():
            mid = row.get('movie_id') if 'movie_id' in row else None
            p, l, r, y, o = fetch_movie_details(mid)
            movies.append(row['title']); posters.append(p)
            links.append(l); ratings.append(r); years.append(y); overviews.append(o)
        return movies, posters, links, ratings, years, overviews

    tfidf   = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2)
    vectors = tfidf.fit_transform(selected_df['tags']).toarray()
    model   = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(vectors)

    idx                = selected_df[selected_df['title'] == movie_name_input].index[0]
    distances, indices = model.kneighbors([vectors[idx]], n_neighbors=6)

    for dist, i in zip(distances[0][1:], indices[0][1:]):
        movie = selected_df.iloc[i]
        mid   = movie.get('movie_id') if 'movie_id' in movie else None
        p, l, r, y, o = fetch_movie_details(mid)
        movies.append(movie['title']); posters.append(p)
        links.append(l); ratings.append(r); years.append(y); overviews.append(o)

    return movies, posters, links, ratings, years, overviews


# ══════════════════════════════════════════════════════════════
#  DISCOVER BUTTON
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider" style="margin-top: 3.5rem;"></div>', unsafe_allow_html=True)
bc = st.columns([2,1,2])
with bc[1]:
    if st.button("✦  Discover Films"):
        st.session_state.recommend_triggered = True


# ══════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════
if st.session_state.get("recommend_triggered"):
    with st.spinner("Scanning the archive…"):
        recs, posters, links, ratings, years, overviews = recommend(selected_movie)

    # Silent Logging Process
    log_recommendation(industry, genre, actor, selected_movie, recs)

    if recs:
        # ── Poster grid (Interactive Cards) ────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="s-eyebrow">Step 3 — Your Picks</p>'
                    '<p class="s-heading">Curated Selections</p>', unsafe_allow_html=True)

        def rec_card(col, title, poster, link, rating, year, overview, delay_ms=0):
            safe_title = html.escape(title)
            safe_desc = html.escape(overview or "")
            if len(safe_desc) > 160:
                safe_desc = safe_desc[:160] + "..."
                
            display_img = poster if poster else "https://via.placeholder.com/500x750?text=No+Poster"
            # Fallback link if none
            href = link if link else "#"
            
            with col:
                st.markdown(f"""
                <a href="{href}" target="_blank" class="rec-card-link">
                    <div class="rec-card" style="animation-delay:{delay_ms}ms">
                        <div class="card-image-wrapper">
                            <img src="{display_img}" alt="{safe_title}">
                            <div class="card-overlay">
                                <p>{safe_desc}</p>
                            </div>
                        </div>
                        <div class="rc-body">
                            <div class="rc-title">{safe_title}</div>
                            <div class="rc-meta">★ {rating} &nbsp;&nbsp;|&nbsp;&nbsp; {year}</div>
                        </div>
                    </div>
                </a>
                """, unsafe_allow_html=True)

        r1 = st.columns(2, gap="large")
        for i in range(min(2, len(recs))):
            rec_card(r1[i], recs[i], posters[i], links[i], ratings[i], years[i], overviews[i], i*150)

        if len(recs) > 2:
            st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
            r2 = st.columns(2, gap="large")
            for j in range(2, min(4, len(recs))):
                rec_card(r2[j-2], recs[j], posters[j], links[j], ratings[j], years[j], overviews[j], j*150)

        if len(recs) > 4:
            st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
            r3 = st.columns([1,2,1], gap="large")
            rec_card(r3[1], recs[4], posters[4], links[4], ratings[4], years[4], overviews[4], 600)

    else:
        st.error("No recommendations found.")


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider" style="margin-top: 4rem;"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="ll-footer">
    <strong>LensLore</strong> &nbsp;·&nbsp; Intelligent Film Discovery
    &nbsp;&nbsp;|&nbsp;&nbsp; TF-IDF &amp; Cosine Similarity
    &nbsp;&nbsp;|&nbsp;&nbsp; Powered by <strong>TMDB API</strong>
    <br>Built with Streamlit &nbsp;·&nbsp; &copy; 2025
</div>
""", unsafe_allow_html=True)
