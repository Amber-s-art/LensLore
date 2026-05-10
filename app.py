"""
╔══════════════════════════════════════════════════════════════════╗
║   LensLore v3  —  Intelligent Cinema Discovery                   ║
║   Bollywood & Hollywood · TF-IDF + KNN · TMDB + OMDb APIs       ║
╚══════════════════════════════════════════════════════════════════╝

CHANGES vs v2
─────────────
API LAYER
  • TMDB_API_KEY  embedded  (f38c299455d0b52cca2e1e70c69b62b0)
  • TMDB_TOKEN    embedded  (eyJhbGciOi…)
  • OMDB_API_KEY  embedded  (fc26a904)
  • fetch_tmdb()   — id-based fetch (poster, trailer, genres, overview, TMDB rating)
  • fetch_tmdb_by_title() — string-search fallback when no numeric id
  • fetch_omdb()   — title+year → IMDb rating (fallbacks to TMDB vote_average)
  • fetch_all_info() — runs TMDB + OMDb concurrently via ThreadPoolExecutor,
                       returns unified dict with both ratings

ML ENGINE
  • min_df=1 → Hollywood sparse tags never crash
  • int(float(val)) fix for float64 movie_id columns
  • build_tags() smart builder works on both datasets

CACHING
  • @st.cache_data(ttl=86400) on all API calls  (24 h TTL)
  • @st.cache_data on load_raw_data
  • @st.cache_data on build_tfidf_model so TF-IDF only runs once per df change

UI / UX
  • IMDb rating pill (amber) + TMDB rating pill (gold) shown side-by-side
  • Full synopsis overlay on poster hover with smooth cubic-bezier transition
  • Entire card is a clickable trailer link
  • New: `@keyframes posterReveal` — iris-wipe overlay on hover
  • New: `@keyframes ratingPop` — rating badges spring in
  • New: `@keyframes cardLift` — cards float on entrance
  • Poster blur+scale on hover before overlay appears
  • IMDb badge uses official amber (#F5C518) colour
  • match-score bar now animates with spring easing
  • Skeleton grid uses 2-row layout matching final card layout
  • JS: card hover now tracks exact mouse XY for per-pixel tilt
  • JS: poster overlay text fades in 100 ms after overlay itself
"""

import streamlit as st
import pandas as pd
import ast
import requests
import csv
import os
import html
import base64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# ══════════════════════════════════════════════════════════════════
#  API CREDENTIALS
# ══════════════════════════════════════════════════════════════════
TMDB_API_KEY = "f38c299455d0b52cca2e1e70c69b62b0"
TMDB_TOKEN   = (
    "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmMzhjMjk5NDU1ZDBiNTJjY"
    "2EyZTFlNzBjNjliNjJiMCIsIm5iZiI6MTc1Mzk1MTA4Ni44MTksInN1"
    "YiI6IjY4OGIyYjZlYTBhYTJlM2RkZDc2N2Q5MyIsInNjb3BlcyI6WyJh"
    "cGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.o2aZuhksRxjxvrD1UajVcNtdF"
    "DBQa5bPyj9CagvmHOA"
)
OMDB_API_KEY = "fc26a904"

TMDB_HEADERS = {
    "accept":        "application/json",
    "Authorization": f"Bearer {TMDB_TOKEN}",
}
OMDB_BASE = "https://www.omdbapi.com/"
TMDB_BASE = "https://api.themoviedb.org/3"

# ══════════════════════════════════════════════════════════════════
#  ASSET HELPERS
# ══════════════════════════════════════════════════════════════════
def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_icon      = Image.open("assets/logo.png")
HERO_ICON_B64 = _b64("assets/logo.png")

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LensLore · Film Discovery",
    page_icon=img_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Cinematic Noir · v3
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&family=Cormorant+Garamond:ital,wght@0,300;0,600;1,300&display=swap');

/* ─── DESIGN TOKENS ───────────────────────────────────────────── */
:root {
    --gold:        #C9A84C;
    --gold-lt:     #F0D080;
    --gold-dim:    #7A6228;
    --gold-pale:   rgba(201,168,76,.07);
    --imdb:        #F5C518;
    --imdb-dim:    rgba(245,197,24,.15);
    --bg:          #060608;
    --surf:        #0E0E12;
    --surf2:       #16161C;
    --surf3:       #1E1E26;
    --bdr:         #1A1A22;
    --bdr-g:       rgba(201,168,76,.2);
    --text:        #EDE5D4;
    --text-2:      #9A927F;
    --muted:       #555;
    --dim:         #2A2A35;
    --ease-spring: cubic-bezier(.34,1.56,.64,1);
    --ease-out:    cubic-bezier(.25,.8,.25,1);
}

html,body,[class*="css"] { background:#060608!important; color:var(--text); font-family:'DM Sans',sans-serif; }
#MainMenu,footer,header  { visibility:hidden; }
.block-container         { padding:0 2rem 8rem!important; max-width:1260px; margin:auto; position:relative; z-index:1; }

::-webkit-scrollbar             { width:5px; }
::-webkit-scrollbar-track       { background:var(--bg); }
::-webkit-scrollbar-thumb       { background:linear-gradient(180deg,var(--gold-dim),var(--gold)); border-radius:5px; }
::-webkit-scrollbar-thumb:hover { background:var(--gold); }

/* ═══════════════════════════════════
   ALL KEYFRAMES
═══════════════════════════════════ */
@keyframes apertureOpen  {
  0%  { transform:scale(.3)rotate(-90deg); opacity:0; filter:blur(10px); }
  60% { transform:scale(1.1)rotate(8deg); opacity:1; }
  80% { transform:scale(.97)rotate(-2deg); }
  100%{ transform:scale(1)rotate(0); opacity:1; filter:blur(0); }
}
@keyframes titleReveal {
  0%  { opacity:0; letter-spacing:.6em; transform:translateY(20px); }
  100%{ opacity:1; letter-spacing:-2px; transform:translateY(0); }
}
@keyframes fadeUp        { from{opacity:0;transform:translateY(36px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn        { from{opacity:0} to{opacity:1} }
@keyframes buttonGlow    { 0%,100%{box-shadow:0 0 8px rgba(201,168,76,.15)} 50%{box-shadow:0 0 30px rgba(201,168,76,.55),0 0 70px rgba(201,168,76,.12)} }
@keyframes filmScroll    { from{transform:translateX(0)} to{transform:translateX(-50%)} }
@keyframes filmScrollRev { from{transform:translateX(-50%)} to{transform:translateX(0)} }
@keyframes shimmerGold   { 0%{background-position:-400% center} 100%{background-position:400% center} }
@keyframes grain         { 0%,100%{transform:translate(0,0)} 20%{transform:translate(-1.5%,-1%)} 40%{transform:translate(1%,-2%)} 60%{transform:translate(-.5%,1.5%)} 80%{transform:translate(2%,.5%)} }
@keyframes ambFloat1     { 0%,100%{transform:translate(0,0)scale(1);opacity:.04} 33%{transform:translate(30px,-20px)scale(1.1);opacity:.07} 66%{transform:translate(-20px,15px)scale(.95);opacity:.03} }
@keyframes ambFloat2     { 0%,100%{transform:translate(0,0)scale(1);opacity:.03} 33%{transform:translate(-25px,30px)scale(1.08);opacity:.06} 66%{transform:translate(20px,-10px)scale(.92);opacity:.04} }
@keyframes pulseRing     { 0%{transform:scale(.95);box-shadow:0 0 0 0 rgba(201,168,76,.4)} 70%{transform:scale(1);box-shadow:0 0 0 14px rgba(201,168,76,0)} 100%{transform:scale(.95);box-shadow:0 0 0 0 rgba(201,168,76,0)} }
@keyframes rippleOut     { 0%{transform:scale(0);opacity:.6} 100%{transform:scale(4);opacity:0} }
@keyframes revealSlide   { from{opacity:0;transform:translateY(50px)scale(.97)} to{opacity:1;transform:translateY(0)scale(1)} }
@keyframes logoGlitch    { 0%,94%,100%{text-shadow:none;transform:none} 95%{text-shadow:3px 0 #C9A84C,-3px 0 #8B2020;transform:translateX(2px)} 96%{text-shadow:-3px 0 #C9A84C,3px 0 #8B2020;transform:translateX(-2px)} 97%{text-shadow:2px 0 #C9A84C;transform:translateX(1px)} }
@keyframes starTwinkle   { 0%,100%{opacity:.2;transform:scale(1)} 50%{opacity:1;transform:scale(1.4)} }
@keyframes floatParticle { 0%{transform:translateY(100vh)translateX(0)scale(0);opacity:0} 10%{opacity:.9} 90%{opacity:.5} 100%{transform:translateY(-130px)translateX(var(--drift))scale(1);opacity:0} }
@keyframes skelPulse     { 0%,100%{background:var(--surf2)} 50%{background:var(--surf3)} }
@keyframes marqueeScroll { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
@keyframes matchFill     { from{width:0} to{width:var(--match-pct)} }
@keyframes badgePop      { from{transform:scale(.75);opacity:0} to{transform:scale(1);opacity:1} }
@keyframes genrePop      { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes progressFill  { from{width:0} to{width:100%} }
@keyframes successPulse  { 0%{box-shadow:0 0 0 0 rgba(201,168,76,.6)} 70%{box-shadow:0 0 0 26px rgba(201,168,76,0)} 100%{box-shadow:0 0 0 0 rgba(201,168,76,0)} }

/* NEW v3 keyframes */
@keyframes cardLift {
  0%  { opacity:0; transform:translateY(70px)rotateX(14deg)scale(.93); filter:blur(5px); }
  60% { opacity:1; transform:translateY(-6px)rotateX(-2deg)scale(1.01); filter:blur(0); }
  100%{ opacity:1; transform:translateY(0)rotateX(0)scale(1); }
}
@keyframes ratingPop {
  0%  { transform:scale(0)rotate(-12deg); opacity:0; }
  70% { transform:scale(1.12)rotate(2deg); }
  100%{ transform:scale(1)rotate(0); opacity:1; }
}
@keyframes posterReveal {
  0%  { clip-path:circle(0% at 50% 100%); opacity:0; }
  100%{ clip-path:circle(150% at 50% 100%); opacity:1; }
}
@keyframes synopsisIn {
  0%  { opacity:0; transform:translateY(12px); }
  100%{ opacity:1; transform:translateY(0); }
}
@keyframes scanLine {
  0%  { top:-4%; }
  100%{ top:104%; }
}
@keyframes imdbShimmer {
  0%  { background-position:-200% center; }
  100%{ background-position:200% center; }
}

/* ─── CURSOR GLOW ─────────────────────────────────────────────── */
#cursor-glow {
  width:340px; height:340px; border-radius:50%;
  position:fixed; pointer-events:none; z-index:9999;
  background:radial-gradient(circle,rgba(201,168,76,.055) 0%,transparent 70%);
  transform:translate(-50%,-50%); mix-blend-mode:screen;
}

/* ─── PARTICLES ───────────────────────────────────────────────── */
.ptc-wrap { position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden; }
.ptc      { position:absolute;bottom:0;border-radius:50%;background:var(--gold);opacity:0; }

/* ─── AMBIENT BLOBS ───────────────────────────────────────────── */
.amb-bg { position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden; }
.ab     { position:absolute;border-radius:50%;filter:blur(80px); }
.ab-1   { width:650px;height:650px;top:-10%;left:-8%;background:radial-gradient(circle,rgba(201,168,76,.06) 0%,transparent 70%);animation:ambFloat1 18s ease-in-out infinite; }
.ab-2   { width:520px;height:520px;bottom:8%;right:-6%;background:radial-gradient(circle,rgba(139,32,32,.05) 0%,transparent 70%);animation:ambFloat2 22s ease-in-out infinite; }
.ab-3   { width:420px;height:420px;top:40%;left:48%;background:radial-gradient(circle,rgba(26,64,64,.05) 0%,transparent 70%);animation:ambFloat1 26s ease-in-out infinite reverse; }

/* Film grain */
body::before { content:'';position:fixed;inset:0;z-index:0;pointer-events:none;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");background-size:200px 200px;opacity:.04;animation:grain .3s steps(1) infinite; }
/* Vignette */
body::after  { content:'';position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(ellipse at center,transparent 40%,rgba(0,0,0,.74) 100%); }

/* ─── FILM STRIPS ─────────────────────────────────────────────── */
.strips    { background:var(--surf);border-bottom:1px solid var(--bdr); }
.strip-row { overflow:hidden;user-select:none;border-bottom:1px solid var(--bdr); }
.strip-row:last-child { border-bottom:none; }
.strip-in  { display:flex;width:max-content;align-items:center;padding:6px 0; }
.strip-in.fwd { animation:filmScroll 34s linear infinite; }
.strip-in.rev { animation:filmScrollRev 28s linear infinite; }
.s-hole    { width:14px;height:10px;background:var(--bg);border-radius:2px;border:1px solid var(--bdr);margin:0 10px;flex-shrink:0; }
.s-lbl     { color:var(--gold-dim);font-size:.58em;letter-spacing:.32em;text-transform:uppercase;padding:0 8px;white-space:nowrap; }
.s-num     { color:var(--dim);font-size:.48em;font-family:monospace;padding:0 4px; }

/* ─── HERO ────────────────────────────────────────────────────── */
.hero           { text-align:center;padding:4rem 1rem 2.5rem;position:relative;z-index:1; }
.hero-icon-wrap { position:relative;display:inline-block;margin-bottom:.5em;animation:apertureOpen 1.2s var(--ease-spring) both; }
.hero-icon-wrap::before { content:'';position:absolute;inset:-15px;border-radius:50%;background:radial-gradient(circle,rgba(201,168,76,.14) 0%,transparent 70%);animation:pulseRing 3s ease-out infinite; }
.hero-icon  { height:68px;width:auto;display:block;filter:drop-shadow(0 0 24px rgba(201,168,76,.5)); }
.hero-logo  { font-family:'Playfair Display',serif;font-size:clamp(3em,9vw,6.5em);font-weight:900;line-height:1;margin:.1em 0 0;animation:titleReveal 1.1s cubic-bezier(.77,0,.18,1) .3s both; }
.logo-l     { color:var(--gold);animation:logoGlitch 9s ease-in-out infinite;display:inline-block;text-shadow:0 0 40px rgba(201,168,76,.3); }
.logo-r     { color:var(--text);display:inline-block; }
.hero-tagline { font-family:'Cormorant Garamond',serif;font-style:italic;font-size:1.15em;color:var(--muted);margin-top:.6em;animation:fadeUp .8s ease 1s both;letter-spacing:.05em; }
.hero-sub   { font-size:.72em;color:var(--muted);letter-spacing:.4em;text-transform:uppercase;margin-top:.7em;animation:fadeUp .8s ease 1.1s both;white-space:nowrap;display:inline-block; }
.hero-line  { width:80px;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);margin:1.6rem auto 0;animation:fadeUp .8s ease 1.2s both;position:relative; }
.hero-line::before,.hero-line::after { content:'◆';position:absolute;top:50%;transform:translateY(-50%);font-size:6px;color:var(--gold); }
.hero-line::before { left:-10px; }
.hero-line::after  { right:-10px; }
.hero-stars { position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:0; }
.star       { position:absolute;border-radius:50%;background:var(--gold); }

/* ─── INDUSTRY BADGE ──────────────────────────────────────────── */
.ind-badge { display:inline-flex;align-items:center;gap:8px;margin-top:1.2em;padding:6px 20px;border-radius:30px;font-size:.68em;letter-spacing:.3em;text-transform:uppercase;font-weight:600;animation:badgePop .45s var(--ease-spring) both;border:1px solid var(--gold-dim);background:rgba(201,168,76,.05);color:var(--gold);transition:background .3s,box-shadow .3s;cursor:default; }
.ind-badge:hover { background:rgba(201,168,76,.09);box-shadow:0 0 18px rgba(201,168,76,.15); }
.ind-dot   { width:7px;height:7px;border-radius:50%;background:var(--gold);animation:pulseRing 2.2s ease-out infinite; }

/* ─── DIVIDER ─────────────────────────────────────────────────── */
.divider { height:1px;background:linear-gradient(90deg,transparent 0%,var(--gold-dim) 20%,var(--gold) 50%,var(--gold-dim) 80%,transparent 100%);background-size:400% auto;animation:shimmerGold 4s linear infinite;margin:2.5rem 0;border:none;position:relative; }
.divider::before { content:'✦';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:8px;color:var(--gold);background:var(--bg);padding:0 8px; }

/* ─── SECTION LABELS ──────────────────────────────────────────── */
.s-eye  { font-size:.65em;font-weight:600;letter-spacing:.38em;text-transform:uppercase;color:var(--gold);margin-bottom:.2em;display:flex;align-items:center;gap:10px; }
.s-eye::before { content:'';display:inline-block;width:24px;height:1px;background:var(--gold); }
.s-head { font-family:'Playfair Display',serif;font-size:1.8em;font-weight:700;color:var(--text);margin:0 0 1.2em;line-height:1.1; }

/* ─── SELECTBOX ───────────────────────────────────────────────── */
label[data-testid="stWidgetLabel"]             { color:var(--muted)!important;font-size:.72em!important;font-weight:500!important;letter-spacing:.2em!important;text-transform:uppercase!important; }
div[data-baseweb="select"] > div              { background:var(--surf2)!important;border:1px solid var(--bdr)!important;border-radius:6px!important;color:var(--text)!important;transition:border-color .3s,box-shadow .3s!important; }
div[data-baseweb="select"] > div:hover        { border-color:var(--gold-dim)!important;box-shadow:0 0 14px rgba(201,168,76,.1)!important; }
div[data-baseweb="select"] > div:focus-within { border-color:var(--gold)!important;box-shadow:0 0 0 2px rgba(201,168,76,.15)!important; }
[data-baseweb="popover"]                      { background:var(--surf2)!important;border:1px solid var(--bdr)!important; }
[role="option"]:hover                         { background:rgba(201,168,76,.08)!important; }
[role="option"][aria-selected="true"]         { background:rgba(201,168,76,.12)!important;color:var(--gold-lt)!important; }

/* ─── DISCOVER BUTTON ─────────────────────────────────────────── */
.stButton>button {
  background:linear-gradient(135deg,rgba(6,6,8,1) 0%,rgba(201,168,76,.12) 50%,rgba(6,6,8,1) 100%)!important;
  background-size:300% auto!important; color:var(--gold-lt)!important;
  border:1px solid var(--gold)!important; padding:1em 3.5em!important;
  font-family:'DM Sans',sans-serif!important; font-size:.82em!important;
  font-weight:600!important; letter-spacing:.35em!important; text-transform:uppercase!important;
  border-radius:3px!important; transition:all .5s cubic-bezier(.175,.885,.32,1.275)!important;
  animation:buttonGlow 4s ease-in-out infinite!important;
  position:relative!important; overflow:hidden!important;
  display:block!important; margin:0 auto!important; cursor:pointer!important;
}
.stButton>button::before {
  content:''!important; position:absolute!important; inset:0!important;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.06),transparent)!important;
  transform:translateX(-100%)!important; transition:transform .65s ease!important;
}
.stButton>button:hover::before { transform:translateX(100%)!important; }
.stButton>button:hover {
  background:linear-gradient(135deg,var(--gold) 0%,var(--gold-lt) 50%,var(--gold) 100%)!important;
  color:#060608!important; box-shadow:0 0 44px rgba(201,168,76,.55),0 8px 28px rgba(0,0,0,.6)!important;
  transform:translateY(-4px)scale(1.04)!important; border-color:var(--gold-lt)!important;
  letter-spacing:.42em!important; animation:none!important;
}
.stButton>button:active { transform:translateY(-1px)scale(1.01)!important; }
.ripple-fx {
  position:absolute; border-radius:50%; background:rgba(201,168,76,.4);
  width:100px; height:100px; margin:-50px 0 0 -50px;
  animation:rippleOut .85s linear; pointer-events:none;
}

/* small log button */
.log-btn .stButton>button {
  padding:.45em 1.2em!important; font-size:.6em!important; letter-spacing:.18em!important;
  animation:none!important; background:transparent!important;
  border-color:var(--gold-dim)!important; color:var(--gold-dim)!important;
}
.log-btn .stButton>button:hover {
  background:rgba(201,168,76,.07)!important; border-color:var(--gold)!important;
  color:var(--gold)!important; transform:translateY(-2px)scale(1)!important; box-shadow:none!important;
}

/* ═══════════════════════════════════
   RATING PILLS  (v3 — IMDb + TMDB)
═══════════════════════════════════ */
.ratings-row   { display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap; }

/* IMDb pill — official amber */
.imdb-pill {
  display:inline-flex; align-items:center; gap:6px;
  background:var(--imdb-dim); border:1px solid rgba(245,197,24,.4);
  color:var(--imdb); font-size:.76em; font-weight:700; border-radius:40px;
  padding:4px 13px; letter-spacing:.04em;
  animation:ratingPop .5s var(--ease-spring) .1s both;
  background-image:linear-gradient(90deg,transparent,rgba(245,197,24,.08),transparent);
  background-size:200% auto;
  transition:box-shadow .3s;
}
.imdb-pill:hover { box-shadow:0 0 14px rgba(245,197,24,.3); animation:imdbShimmer 1.5s linear infinite; }
.imdb-icon { font-size:1em; }

/* TMDB pill — gold */
.tmdb-pill {
  display:inline-flex; align-items:center; gap:6px;
  background:rgba(201,168,76,.08); border:1px solid rgba(201,168,76,.3);
  color:var(--gold-lt); font-size:.76em; font-weight:600; border-radius:40px;
  padding:4px 13px; letter-spacing:.04em;
  animation:ratingPop .5s var(--ease-spring) .2s both;
  transition:box-shadow .3s;
}
.tmdb-pill:hover { box-shadow:0 0 14px rgba(201,168,76,.25); }

/* ═══════════════════════════════════
   SELECTED MOVIE CARD
═══════════════════════════════════ */
.sel-wrap       { display:flex;background:var(--surf);border:1px solid var(--bdr);border-left:3px solid var(--gold);border-radius:12px;overflow:hidden;animation:revealSlide .7s ease both;transition:box-shadow .4s,transform .4s;position:relative; }
.sel-wrap::before { content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(201,168,76,.03) 0%,transparent 60%);pointer-events:none; }
.sel-wrap:hover { box-shadow:0 22px 55px rgba(0,0,0,.85),-5px 0 24px rgba(201,168,76,.16); }
.sel-poster     { flex:0 0 30%;position:relative;overflow:hidden;min-height:320px; }
.sel-poster img { width:100%;height:100%;object-fit:cover;transition:filter .55s ease,transform .7s ease; display:block; }

/* ─ POSTER HOVER OVERLAY (v3 — synopsis reveal) */
.sel-ov {
  position:absolute; inset:0;
  background:linear-gradient(to top,
    rgba(6,6,8,.97) 0%,
    rgba(6,6,8,.88) 50%,
    rgba(6,6,8,.55) 100%);
  opacity:0; transition:opacity .42s cubic-bezier(.25,.8,.25,1);
  display:flex; flex-direction:column; align-items:center;
  justify-content:flex-end; padding:28px 20px;
  border-right:1px solid var(--bdr-g);
}
.sel-ov-play  { font-size:2.4em;color:var(--gold);margin-bottom:14px;display:block;transform:scale(.8);transition:transform .4s var(--ease-spring); }
.sel-ov-syn   { font-family:'Cormorant Garamond',serif;font-size:.95em;color:#C8C0AE;line-height:1.65;text-align:center;opacity:0;transform:translateY(10px);transition:opacity .35s ease .12s,transform .35s ease .12s; }
.sel-wrap:hover .sel-poster img { filter:blur(10px)brightness(.28)saturate(.35); transform:scale(1.07); }
.sel-wrap:hover .sel-ov         { opacity:1; }
.sel-wrap:hover .sel-ov-play    { transform:scale(1); }
.sel-wrap:hover .sel-ov-syn     { opacity:1; transform:translateY(0); }

.sel-info      { flex:1;padding:28px 32px;display:flex;flex-direction:column;justify-content:center;gap:5px; }
.sel-title     { font-family:'Playfair Display',serif;font-size:2em;font-weight:700;margin:0 0 3px;line-height:1.1;color:var(--text); }
.sel-year      { color:var(--muted);font-size:.9em;letter-spacing:.14em;margin-bottom:8px; }
.sel-genres    { display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px; }
.genre-pill    { display:inline-block;padding:3px 11px;border-radius:20px;font-size:.63em;font-weight:600;letter-spacing:.14em;text-transform:uppercase;border:1px solid var(--bdr-g);color:var(--gold-dim);background:rgba(201,168,76,.05);animation:genrePop .4s ease both; }
.sel-syn       { color:var(--text-2);line-height:1.72;padding-top:14px;border-top:1px solid var(--bdr);font-family:'Cormorant Garamond',serif;font-size:1.06em; }
.sel-hint      { color:var(--gold-dim);font-size:.68em;text-transform:uppercase;letter-spacing:.16em;margin-top:auto;padding-top:14px;display:flex;align-items:center;gap:7px; }
.sel-hint::before { content:'';display:inline-block;width:16px;height:1px;background:var(--gold-dim); }

/* ─── MARQUEE ─────────────────────────────────────────────────── */
.marquee-wrap        { width:100%;overflow:hidden;border-top:1px solid var(--bdr);border-bottom:1px solid var(--bdr);padding:14px 0;margin:2rem 0;background:linear-gradient(90deg,var(--bg),var(--surf),var(--bg));position:relative; }
.marquee-wrap::before,.marquee-wrap::after { content:'';position:absolute;top:0;bottom:0;width:80px;z-index:2; }
.marquee-wrap::before { left:0;background:linear-gradient(90deg,var(--bg),transparent); }
.marquee-wrap::after  { right:0;background:linear-gradient(270deg,var(--bg),transparent); }
.marquee-inner { display:inline-block;white-space:nowrap;animation:marqueeScroll 22s linear infinite;font-family:'Playfair Display',serif;font-size:1.9em;font-weight:700;color:var(--gold-dim);letter-spacing:.04em; }
.marquee-inner span { color:var(--gold);margin:0 .4em; }

/* ─── STATS BAR ───────────────────────────────────────────────── */
.stats-bar { display:flex;border:1px solid var(--bdr);border-radius:8px;overflow:hidden;margin:2rem 0;animation:fadeUp .8s ease .5s both; }
.stat-item { flex:1;text-align:center;padding:18px 12px;border-right:1px solid var(--bdr);background:var(--surf);transition:background .3s,transform .3s;cursor:default; }
.stat-item:last-child { border-right:none; }
.stat-item:hover { background:var(--surf2);transform:translateY(-2px); }
.stat-num  { font-family:'Playfair Display',serif;font-size:1.8em;font-weight:700;color:var(--gold);line-height:1; }
.stat-lbl  { font-size:.6em;color:var(--muted);letter-spacing:.25em;text-transform:uppercase;margin-top:4px; }

/* ─── SKELETON LOADERS ────────────────────────────────────────── */
.skel-grid { display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:1rem 0; }
.skel-card { border-radius:10px;overflow:hidden;background:var(--surf);border:1px solid var(--bdr); }
.skel-post { width:100%;height:340px;background:var(--surf2);animation:skelPulse 1.6s ease-in-out infinite; }
.skel-body { padding:14px; }
.skel-line { height:11px;border-radius:4px;background:var(--surf2);animation:skelPulse 1.6s ease-in-out infinite;margin-bottom:8px; }
.skel-line.s2 { width:55%;animation-delay:.18s; }
.skel-line.s3 { width:35%;animation-delay:.34s; }
.skel-msg  { text-align:center;color:var(--gold-dim);font-size:.74em;letter-spacing:.28em;text-transform:uppercase;padding:1rem;animation:fadeUp .4s ease both; }

/* ═══════════════════════════════════
   REC CARDS  (v3 — full rework)
═══════════════════════════════════ */
.rec-link { text-decoration:none;display:block;border-radius:12px;outline:none; }

.rec-card {
  background:var(--surf); border:1px solid var(--bdr); border-radius:12px;
  overflow:hidden; position:relative; cursor:pointer; height:100%;
  transition:transform .48s var(--ease-out), box-shadow .48s var(--ease-out);
  transform-style:preserve-3d;
}

/* Staggered spring entrance */
.rc-e1 { animation:cardLift .75s var(--ease-spring) .06s both; }
.rc-e2 { animation:cardLift .75s var(--ease-spring) .19s both; }
.rc-e3 { animation:cardLift .75s var(--ease-spring) .32s both; }
.rc-e4 { animation:cardLift .75s var(--ease-spring) .45s both; }
.rc-e5 { animation:cardLift .75s var(--ease-spring) .58s both; }

/* Holographic border + glow */
.rec-card::after {
  content:''; position:absolute; inset:0; border-radius:12px;
  border:1px solid transparent;
  transition:border-color .4s, box-shadow .4s;
  pointer-events:none;
}
.rec-card:hover::after {
  border-color:var(--gold);
  box-shadow:0 0 26px rgba(201,168,76,.22) inset;
}
.rec-card:hover {
  transform:translateY(-16px)scale(1.028);
  box-shadow:
    0 40px 80px rgba(0,0,0,.95),
    0 0 0 1px var(--gold-dim),
    0 0 60px rgba(201,168,76,.1);
}

/* ─ CARD IMAGE SECTION */
.card-img     { position:relative;width:100%;aspect-ratio:2/3;overflow:hidden;background:#0A0A0F; }
.card-img img { width:100%;height:100%;object-fit:cover;display:block;transition:filter .52s ease,transform .72s ease; }

/* scan-line sweep */
.card-scan { position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(201,168,76,.4),transparent);opacity:0;pointer-events:none;z-index:5;transition:opacity .3s; }
.rec-card:hover .card-scan { opacity:1;animation:scanLine 2.2s linear infinite; }

/* ─ POSTER HOVER OVERLAY  (v3 — full synopsis + play icon) */
.card-ov {
  position:absolute; inset:0;
  background:linear-gradient(to top,
    rgba(6,6,8,.98) 0%,
    rgba(6,6,8,.92) 45%,
    rgba(6,6,8,.65) 75%,
    rgba(6,6,8,.25) 100%);
  opacity:0; transition:opacity .42s cubic-bezier(.25,.8,.25,1);
  display:flex; flex-direction:column; align-items:center;
  justify-content:flex-end; padding:20px 18px;
  border-bottom:2px solid var(--gold);
}
.ov-play  { font-size:2em;color:var(--gold);margin-bottom:10px;display:block;transform:scale(.75);transition:transform .4s var(--ease-spring); }
.ov-syn   { font-family:'Cormorant Garamond',serif;font-size:.88em;color:#C5BCA9;line-height:1.65;text-align:center;opacity:0;transform:translateY(12px);transition:opacity .35s ease .1s,transform .35s ease .1s; }
.rec-card:hover .card-img img { filter:blur(9px)brightness(.32)saturate(.45); transform:scale(1.12); }
.rec-card:hover .card-ov      { opacity:1; }
.rec-card:hover .ov-play      { transform:scale(1); }
.rec-card:hover .ov-syn       { opacity:1; transform:translateY(0); }

/* card number + industry badges */
.card-num { position:absolute;top:11px;left:11px;background:rgba(6,6,8,.88);border:1px solid var(--gold-dim);border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:.63em;font-weight:600;color:var(--gold);z-index:6;font-family:'Cormorant Garamond',serif; }
.card-ind { position:absolute;top:11px;right:11px;background:rgba(6,6,8,.88);border:1px solid var(--gold-dim);border-radius:3px;padding:3px 8px;font-size:.52em;font-weight:700;color:var(--gold);letter-spacing:.22em;text-transform:uppercase;z-index:6; }

/* ─ CARD BODY */
.rc-body  { padding:14px 13px;border-top:1px solid var(--bdr);background:linear-gradient(180deg,var(--surf) 0%,var(--surf2) 100%); }
.rc-title { font-family:'Playfair Display',serif;font-size:1em;font-weight:700;color:var(--text);margin:0 0 5px;line-height:1.25; }
.rc-meta  { font-size:.72em;color:var(--gold);font-weight:500;display:flex;align-items:center;gap:6px;margin-bottom:7px;flex-wrap:wrap; }
.rc-dot   { width:3px;height:3px;border-radius:50%;background:var(--gold-dim);display:inline-block; }
.rc-grow  { display:flex;flex-wrap:wrap;gap:4px;margin-bottom:9px; }
.rc-gpill { display:inline-block;padding:2px 7px;border-radius:20px;font-size:.55em;font-weight:600;letter-spacing:.12em;text-transform:uppercase;border:1px solid rgba(201,168,76,.18);color:var(--gold-dim);background:rgba(201,168,76,.04); }

/* ─ IMDb MINI-BADGE inside card */
.rc-imdb {
  display:inline-flex; align-items:center; gap:4px;
  background:rgba(245,197,24,.1); border:1px solid rgba(245,197,24,.3);
  color:var(--imdb); font-size:.64em; font-weight:700; border-radius:4px;
  padding:2px 7px; letter-spacing:.05em;
}

/* ─ MATCH SCORE BAR */
.match-wrap  { margin-top:4px; }
.match-row   { display:flex;justify-content:space-between;align-items:center;font-size:.57em;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:5px; }
.match-score { color:var(--gold);font-weight:700;font-size:1.1em; }
.match-track { height:3px;background:var(--surf3);border-radius:3px;overflow:hidden; }
.match-fill  { height:100%;background:linear-gradient(90deg,var(--gold-dim),var(--gold),var(--gold-lt));border-radius:3px;width:var(--match-pct);animation:matchFill 1s cubic-bezier(.22,1,.36,1) .4s both; }

/* ─── RESULTS HEADER ──────────────────────────────────────────── */
.res-hdr  { animation:successPulse .8s ease both; }
.prog-wrap { width:100%;height:2px;background:var(--surf3);border-radius:2px;overflow:hidden;margin:1rem 0; }
.prog-bar  { height:100%;background:linear-gradient(90deg,var(--gold-dim),var(--gold),var(--gold-lt));animation:progressFill 1.6s ease forwards; }

/* ─── EMPTY STATE ─────────────────────────────────────────────── */
.empty-state { text-align:center;padding:4rem 2rem;background:var(--surf);border:1px dashed var(--gold-dim);border-radius:12px;margin-top:2rem;animation:fadeUp .6s ease both; }
.empty-icon  { font-size:3.5em;color:var(--gold-dim);font-family:'Playfair Display',serif;margin-bottom:.3em; }
.empty-title { font-family:'Playfair Display',serif;font-size:1.8em;color:var(--text);margin-bottom:.6em; }
.empty-body  { color:var(--muted);font-size:.95em;max-width:440px;margin:0 auto;line-height:1.75; }

/* ─── SCROLL REVEAL ───────────────────────────────────────────── */
.rvl    { opacity:0;transform:translateY(40px);transition:opacity .7s,transform .7s; }
.rvl.on { opacity:1;transform:translateY(0); }

/* ─── TOAST ───────────────────────────────────────────────────── */
.ll-toast    { position:fixed;bottom:28px;right:28px;background:var(--surf2);border:1px solid var(--gold-dim);border-left:3px solid var(--gold);border-radius:6px;padding:13px 20px;color:var(--text);font-size:.8em;letter-spacing:.05em;z-index:9998;animation:fadeUp .4s ease both;box-shadow:0 10px 32px rgba(0,0,0,.8);display:flex;align-items:center;gap:10px; }
.ll-toast-ic { color:var(--gold); }

/* ─── SPINNER ─────────────────────────────────────────────────── */
.stSpinner>div { border-top-color:var(--gold)!important;border-right-color:rgba(201,168,76,.3)!important;border-bottom-color:rgba(201,168,76,.1)!important;border-left-color:rgba(201,168,76,.2)!important; }

/* ─── SUCCESS MESSAGE ─────────────────────────────────────────── */
div[data-testid="stAlert"] { background:rgba(201,168,76,.07)!important;border:1px solid rgba(201,168,76,.25)!important;border-radius:6px!important; }
div[data-testid="stAlert"] p { color:var(--gold-lt)!important; }

/* ─── FOOTER ──────────────────────────────────────────────────── */
.ll-foot { text-align:center;padding:3rem 0 2rem;color:var(--dim);font-size:.67em;letter-spacing:.28em;text-transform:uppercase;line-height:2.4; }
.ll-foot::before { content:'— — —';display:block;color:var(--gold-dim);font-size:.9em;letter-spacing:.5em;margin-bottom:.5em; }
.ll-foot strong { color:var(--gold-dim);font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  BACKGROUND LAYERS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="amb-bg"><div class="ab ab-1"></div><div class="ab ab-2"></div><div class="ab ab-3"></div></div>
<div class="ptc-wrap" id="ptcWrap"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  FILM STRIP TICKER
# ══════════════════════════════════════════════════════════════════
_TOP = ["Action","Drama","Thriller","Romance","Comedy","Sci-Fi",
        "Horror","Mystery","Biography","Fantasy","Adventure","Crime",
        "Animation","History","Musical"]
_BOT = ["Noir","Western","War","Sport","Family","Documentary",
        "Superhero","Anthology","Neo-noir","Surrealist","Gangster",
        "Heist","Psychological","Existential","Satire"]

def _strip(genres, n=6, f0=1):
    out, fr = [], f0
    for g in genres * n:
        out.append(f'<span class="s-num">{fr:04d}</span>'
                   f'<div class="s-hole"></div>'
                   f'<span class="s-lbl">{g}</span>'
                   f'<div class="s-hole"></div>')
        fr += 1
    return "".join(out)

st.markdown(f"""
<div class="strips">
  <div class="strip-row"><div class="strip-in fwd">{_strip(_TOP,f0=1)}</div></div>
  <div class="strip-row"><div class="strip-in rev">{_strip(_BOT,f0=100)}</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_raw_data():
    boll = pd.read_csv("dataset/cleaned/bollywood_cleaned.csv")
    holl = pd.read_csv("dataset/cleaned/hollywood_cleaned.csv")
    return boll, holl

movie_boll, movie_holl = load_raw_data()


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def parse_list(value) -> list:
    if value is None: return []
    try:
        if pd.isna(value): return []
    except (TypeError, ValueError): pass
    s = str(value).strip()
    if not s or s.lower() in ("nan","none","nat"): return []
    if s.startswith("["):
        try:
            p = ast.literal_eval(s)
            return p if isinstance(p, list) else [p]
        except Exception: pass
    return [x.strip() for x in s.split(",") if x.strip()]


# ══════════════════════════════════════════════════════════════════
#  SAFE movie_id  (THE CRASH FIX)
#  Pandas reads numeric NaN-containing cols as float64 → "12345.0"
#  int("12345.0") crashes.  int(float("12345.0")) = 12345  ✓
# ══════════════════════════════════════════════════════════════════
def safe_movie_id(row, columns):
    for col in ["movie_id","id","tmdb_id","movieid","film_id"]:
        if col not in columns: continue
        val = row.get(col)
        if val is None: continue
        try:
            if pd.isna(val): continue
        except (TypeError, ValueError): pass
        s = str(val).strip()
        if s.lower() in ("","nan","none","nat","null"): continue
        try:
            return int(float(s))          # ← THE FIX
        except (ValueError, OverflowError): continue
    return None


# ══════════════════════════════════════════════════════════════════
#  TAG BUILDER (works for both Bollywood & Hollywood)
# ══════════════════════════════════════════════════════════════════
def build_tags(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    col_l = {c.lower(): c for c in df.columns}

    def find(name): return col_l.get(name.lower())

    def to_text(series: pd.Series) -> pd.Series:
        def _row(v):
            if v is None: return ""
            try:
                if pd.isna(v): return ""
            except (TypeError, ValueError): pass
            s = str(v).strip()
            if s.lower() in ("nan","none","nat",""): return ""
            if s.startswith("["):
                try:
                    lst = ast.literal_eval(s)
                    if isinstance(lst, list):
                        return " ".join(str(x).strip() for x in lst if x)
                except Exception: pass
            return s.replace(","," ")
        return series.apply(_row)

    # 1. Try existing tags column
    tc = find("tags")
    if tc:
        conv = to_text(df[tc])
        if (conv.str.strip() != "").sum() > len(df) * 0.25:
            df["tags"] = conv.fillna("")
            return df

    # 2. Build from components
    parts = []
    for name in ["genres","cast","keywords","overview","director","tagline","production_companies"]:
        rc = find(name)
        if rc: parts.append(to_text(df[rc]))

    if parts:
        combined = parts[0]
        for p in parts[1:]: combined = combined + " " + p
        df["tags"] = combined.str.strip().fillna("")
    else:
        txt_cols = [c for c in df.columns if df[c].dtype == object]
        df["tags"] = (df[txt_cols].fillna("").agg(" ".join, axis=1)
                      if txt_cols else df["title"].fillna(""))

    df["tags"] = df["tags"].fillna("").astype(str).str.lower()
    return df


# ══════════════════════════════════════════════════════════════════
#  API LAYER — TMDB
# ══════════════════════════════════════════════════════════════════
_TMDB_EMPTY = dict(poster=None, trailer=None, tmdb_rating=0.0,
                   year="", overview="No synopsis available.", genres=[])

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_tmdb(movie_id: int) -> dict:
    """
    Fetch TMDB data by numeric movie_id.
    Returns: poster URL, YouTube trailer URL, TMDB vote_average,
             release year, overview text, list of genre names.
    """
    if not movie_id:
        return _TMDB_EMPTY.copy()
    try:
        url  = (f"{TMDB_BASE}/movie/{movie_id}"
                f"?api_key={TMDB_API_KEY}&language=en-US&append_to_response=videos")
        data = requests.get(url, headers=TMDB_HEADERS, timeout=7).json()

        if data.get("success") is False:        # invalid id
            return _TMDB_EMPTY.copy()

        poster_path = data.get("poster_path")
        trailer = next(
            (f"https://www.youtube.com/watch?v={v['key']}"
             for v in data.get("videos", {}).get("results", [])
             if v.get("type") == "Trailer" and v.get("site") == "YouTube"),
            data.get("homepage") or f"https://www.themoviedb.org/movie/{movie_id}",
        )
        return dict(
            poster       = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
            trailer      = trailer,
            tmdb_rating  = round(data.get("vote_average", 0) or 0, 1),
            year         = (data.get("release_date") or "")[:4],
            overview     = data.get("overview") or "No synopsis available.",
            genres       = [g["name"] for g in data.get("genres", [])],
        )
    except Exception:
        return _TMDB_EMPTY.copy()


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_tmdb_by_title(title: str, year: str = "") -> dict:
    """
    FALLBACK: search TMDB by title string when no numeric id is available.
    Uses /search/movie, picks the top result, then calls fetch_tmdb().
    """
    try:
        params = {"api_key": TMDB_API_KEY, "query": title, "language": "en-US"}
        if year:
            params["year"] = year
        data = requests.get(f"{TMDB_BASE}/search/movie",
                            headers=TMDB_HEADERS, params=params, timeout=7).json()
        results = data.get("results", [])
        if not results:
            return _TMDB_EMPTY.copy()
        mid = results[0].get("id")
        return fetch_tmdb(mid) if mid else _TMDB_EMPTY.copy()
    except Exception:
        return _TMDB_EMPTY.copy()


# ══════════════════════════════════════════════════════════════════
#  API LAYER — OMDb  (IMDb rating)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_omdb(title: str, year: str = "") -> str:
    """
    Fetch official IMDb rating from OMDb API using title + year.
    Returns a string like "8.3" or "N/A".
    Falls back gracefully if the API is unavailable or returns no rating.
    """
    try:
        params = {"apikey": OMDB_API_KEY, "t": title, "type": "movie"}
        if year:
            params["y"] = year
        data = requests.get(OMDB_BASE, params=params, timeout=6).json()
        if data.get("Response") == "True":
            rating = data.get("imdbRating", "N/A")
            return rating if rating and rating != "N/A" else "N/A"
        return "N/A"
    except Exception:
        return "N/A"


# ══════════════════════════════════════════════════════════════════
#  UNIFIED INFO FETCHER  (TMDB + OMDb concurrently)
#  This is the main entry point used by both selected-card and recs.
#
#  Strategy:
#   1. If movie_id exists → fetch_tmdb(id)  [fast, definitive]
#      Else              → fetch_tmdb_by_title(title, year)  [fallback]
#   2. Simultaneously   → fetch_omdb(title, year)
#   3. Combine: use OMDb imdb_rating; fallback to tmdb_rating if "N/A"
# ══════════════════════════════════════════════════════════════════
def fetch_all_info(title: str, movie_id=None, year_hint: str = "") -> dict:
    """
    Runs TMDB and OMDb fetches in parallel threads.
    Returns a merged dict with both ratings.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        # Submit both API calls simultaneously
        if movie_id:
            tmdb_future = pool.submit(fetch_tmdb, movie_id)
        else:
            tmdb_future = pool.submit(fetch_tmdb_by_title, title, year_hint)

        omdb_future = pool.submit(fetch_omdb, title, year_hint)

        tmdb_data   = tmdb_future.result()
        imdb_rating = omdb_future.result()

    # If year_hint wasn't available, retry OMDb with the year we got from TMDB
    if imdb_rating == "N/A" and tmdb_data.get("year"):
        imdb_rating = fetch_omdb(title, tmdb_data["year"])

    # Fallback: if OMDb completely fails, show TMDB vote_average
    display_imdb = (
        imdb_rating if imdb_rating != "N/A"
        else (str(tmdb_data["tmdb_rating"]) if tmdb_data["tmdb_rating"] else "N/A")
    )

    return {**tmdb_data, "imdb_rating": display_imdb}


# ══════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
#  Returns list of dicts: title, poster, trailer, tmdb_rating,
#  imdb_rating, year, overview, genres, match_score (0–100).
#  All 5 movies fetched concurrently via ThreadPoolExecutor.
# ══════════════════════════════════════════════════════════════════
def recommend(movie_name: str, df: pd.DataFrame) -> list:
    df    = build_tags(df).reset_index(drop=True)
    n     = len(df)
    if n < 2: return []

    n_rec = min(5, n - 1)

    # min_df=1: Hollywood tags are sparse; min_df=2 would discard most terms
    try:
        tfidf   = TfidfVectorizer(max_features=5000, stop_words="english", min_df=1)
        vectors = tfidf.fit_transform(df["tags"]).toarray()
    except Exception as e:
        st.error(f"TF-IDF error: {e}")
        return []

    if vectors.sum() == 0:
        # All-zero matrix: fall back to top-n titles
        subset = df[df["title"] != movie_name].head(5)
        results = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(
                    fetch_all_info,
                    row["title"],
                    safe_movie_id(row, df.columns),
                    ""
                ): row["title"]
                for _, row in subset.iterrows()
            }
            for fut in as_completed(futures):
                t    = futures[fut]
                info = fut.result()
                results.append({"title": t, "match_score": 50.0, **info})
        return results

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(vectors)

    mask = df["title"] == movie_name
    if not mask.any(): return []

    idx        = int(mask.idxmax())
    k          = min(n_rec + 1, n)
    dists, ids = model.kneighbors([vectors[idx]], n_neighbors=k)

    # Prepare fetch jobs — all 5 submitted at once
    jobs = {}
    for dist, i in zip(dists[0][1:], ids[0][1:]):
        row   = df.iloc[i]
        score = round((1.0 - dist) * 100, 1)
        jobs[(row["title"], score)] = (
            row["title"],
            safe_movie_id(row, df.columns),
        )

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(fetch_all_info, title, mid, ""): (title, score)
            for (title, score), (title, mid) in jobs.items()
        }
        for fut in as_completed(futures):
            title, score = futures[fut]
            info         = fut.result()
            results.append({"title": title, "match_score": score, **info})

    # Re-sort by original KNN order (as_completed returns in completion order)
    order = [t for (t, _) in jobs.keys()]
    results.sort(key=lambda r: order.index(r["title"]) if r["title"] in order else 99)
    return results


# ══════════════════════════════════════════════════════════════════
#  LOGGER → logs/prediction.csv
# ══════════════════════════════════════════════════════════════════
LOG_PATH = "logs/prediction.csv"
LOG_COLS = [
    "timestamp","industry","genre_filter","actor_filter","input_movie",
    "rec_1","rec_2","rec_3","rec_4","rec_5",
    "score_1","score_2","score_3","score_4","score_5",
    "imdb_1","imdb_2","imdb_3","imdb_4","imdb_5",
    "user_clicked",
]

def log_prediction(industry, genre, actor, movie, recs):
    os.makedirs("logs", exist_ok=True)
    exists = os.path.isfile(LOG_PATH)
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "industry": industry, "genre_filter": genre,
           "actor_filter": actor, "input_movie": movie, "user_clicked": ""}
    for i, r in enumerate(recs[:5], 1):
        row[f"rec_{i}"]   = r.get("title","")
        row[f"score_{i}"] = f"{r.get('match_score',0):.1f}%"
        row[f"imdb_{i}"]  = r.get("imdb_rating","N/A")
    for j in range(len(recs)+1, 6):
        row[f"rec_{j}"] = row[f"score_{j}"] = row[f"imdb_{j}"] = ""
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if not exists: w.writeheader()
        w.writerow(row)

def log_click(clicked_title: str):
    if not os.path.isfile(LOG_PATH): return
    with open(LOG_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows: rows[-1]["user_clicked"] = clicked_title
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        w.writeheader(); w.writerows(rows)


# ══════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-stars" id="heroStars"></div>
  <div class="hero-icon-wrap">
    <img src="data:image/png;base64,{HERO_ICON_B64}" class="hero-icon" alt="LensLore">
  </div>
  <div class="hero-logo"><span class="logo-l">Lens</span><span class="logo-r">Lore</span></div>
  <p class="hero-tagline">Where every frame tells a story</p>
  <p class="hero-sub">Intelligent Cinema Discovery &nbsp;·&nbsp; Bollywood &amp; Hollywood</p>
  <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  STATS BAR
# ══════════════════════════════════════════════════════════════════
_bc, _hc = len(movie_boll), len(movie_holl)
st.markdown(f"""
<div class="stats-bar rvl">
  <div class="stat-item"><div class="stat-num" data-target="{_bc}">{_bc:,}</div><div class="stat-lbl">Bollywood Films</div></div>
  <div class="stat-item"><div class="stat-num" data-target="{_hc}">{_hc:,}</div><div class="stat-lbl">Hollywood Films</div></div>
  <div class="stat-item"><div class="stat-num" data-target="{_bc+_hc}">{_bc+_hc:,}</div><div class="stat-lbl">Total Archive</div></div>
  <div class="stat-item"><div class="stat-num">TMDB+OMDb</div><div class="stat-lbl">Dual-API Engine</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  STEP 01 — FILTERS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="s-eye">Step 01</p><p class="s-head">Set the Stage</p>',
            unsafe_allow_html=True)

industry    = st.selectbox("Industry", ("Bollywood", "Hollywood"))
selected_df = (movie_boll if industry == "Bollywood" else movie_holl).copy()

st.markdown(f"""
<div style="margin:-0.3rem 0 1.4rem">
  <span class="ind-badge"><span class="ind-dot"></span>
  {industry} &nbsp;·&nbsp; {len(selected_df):,} films loaded</span>
</div>
""", unsafe_allow_html=True)

selected_df["_genres"] = selected_df["genres"].apply(parse_list)
all_genres = sorted({g.strip().lower()
                     for gs in selected_df["_genres"]
                     for g in gs if isinstance(g, str) and g.strip()})

c1, c2 = st.columns(2, gap="large")
with c2:
    genre = st.selectbox("Genre", ["All"] + all_genres, key="genre_sel")
if genre != "All":
    selected_df = selected_df[selected_df["_genres"].apply(
        lambda x: genre.lower() in [g.lower() for g in x])]

selected_df["_cast"] = selected_df["cast"].apply(parse_list)
all_actors = sorted({a.strip()
                     for cs in selected_df["_cast"]
                     for a in cs if isinstance(a, str) and a.strip()})
with c1:
    actor = st.selectbox("Actor", ["All"] + all_actors, key="actor_sel")
if actor != "All":
    selected_df = selected_df[selected_df["_cast"].apply(
        lambda x: actor.lower() in [a.lower() for a in x])]

selected_df = selected_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  STEP 02 — MOVIE SELECTOR
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="s-eye">Step 02</p><p class="s-head">Search the Archive</p>',
            unsafe_allow_html=True)

selected_movie = st.selectbox("Title", sorted(selected_df["title"].tolist()), key="movie_sel")


# ══════════════════════════════════════════════════════════════════
#  SELECTED MOVIE PREVIEW  (TMDB + OMDb dual-API)
# ══════════════════════════════════════════════════════════════════
sel_row = selected_df[selected_df["title"] == selected_movie]
if not sel_row.empty:
    sel_id   = safe_movie_id(sel_row.iloc[0], selected_df.columns)  # crash-safe
    sel_info = fetch_all_info(selected_movie, sel_id)

    t    = html.escape(selected_movie)
    ov   = html.escape(sel_info["overview"])
    img  = sel_info["poster"]  or "https://via.placeholder.com/500x750/0E0E12/C9A84C?text=No+Poster"
    lnk  = sel_info["trailer"] or "#"
    trat = sel_info["tmdb_rating"]
    irat = sel_info["imdb_rating"]
    yr   = sel_info["year"]
    gns  = "".join(f'<span class="genre-pill">{html.escape(g)}</span>'
                   for g in sel_info["genres"][:5])

    # Show IMDb pill + TMDB pill side by side
    imdb_pill = (f'<div class="imdb-pill"><span class="imdb-icon">⭐</span>'
                 f'IMDb &nbsp;{irat}</div>')
    tmdb_pill = (f'<div class="tmdb-pill">🎬 TMDB &nbsp;{trat}/10</div>')

    st.markdown(f"""
    <a href="{lnk}" target="_blank" style="text-decoration:none;display:block;margin-top:18px">
      <div class="sel-wrap">
        <div class="sel-poster">
          <img src="{img}" alt="{t}" loading="lazy">
          <div class="sel-ov">
            <span class="sel-ov-play">▶</span>
            <p class="sel-ov-syn">{ov[:280]}{"…" if len(ov)>280 else ""}</p>
          </div>
        </div>
        <div class="sel-info">
          <div class="ratings-row">{imdb_pill}{tmdb_pill}</div>
          <div class="sel-title">{t}</div>
          <div class="sel-year">{yr}</div>
          <div class="sel-genres">{gns}</div>
          <div class="sel-syn">{ov}</div>
          <div class="sel-hint">Hover poster for synopsis &nbsp;·&nbsp; Click for trailer</div>
        </div>
      </div>
    </a>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  MARQUEE
# ══════════════════════════════════════════════════════════════════
_mq = f"  {selected_movie}  <span>✦</span>  " * 10
st.markdown(f"""
<div class="marquee-wrap">
  <div class="marquee-inner">{_mq}{_mq}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  DISCOVER BUTTON
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="divider" style="margin-top:2rem"></div>', unsafe_allow_html=True)
_bc = st.columns([2, 1, 2])
with _bc[1]:
    if st.button("✦  Discover Films"):
        st.session_state.update({
            "recs":        None,
            "in_movie":    selected_movie,
            "in_industry": industry,
            "in_genre":    genre,
            "in_actor":    actor,
            "fired":       True,
        })


# ══════════════════════════════════════════════════════════════════
#  RESULTS — skeleton → parallel fetch → cards
# ══════════════════════════════════════════════════════════════════
if st.session_state.get("fired"):

    # ── Skeleton placeholders ────────────────────────────────────
    skel = st.empty()
    skel.markdown("""
    <div class="divider"></div>
    <div class="skel-grid">
      <div class="skel-card"><div class="skel-post"></div>
        <div class="skel-body"><div class="skel-line"></div>
        <div class="skel-line s2"></div><div class="skel-line s3"></div></div></div>
      <div class="skel-card"><div class="skel-post" style="animation-delay:.14s"></div>
        <div class="skel-body"><div class="skel-line" style="animation-delay:.14s"></div>
        <div class="skel-line s2" style="animation-delay:.28s"></div>
        <div class="skel-line s3" style="animation-delay:.42s"></div></div></div>
      <div class="skel-card"><div class="skel-post" style="animation-delay:.28s"></div>
        <div class="skel-body"><div class="skel-line" style="animation-delay:.28s"></div>
        <div class="skel-line s2" style="animation-delay:.42s"></div>
        <div class="skel-line s3" style="animation-delay:.56s"></div></div></div>
      <div class="skel-card"><div class="skel-post" style="animation-delay:.42s"></div>
        <div class="skel-body"><div class="skel-line" style="animation-delay:.42s"></div>
        <div class="skel-line s2" style="animation-delay:.56s"></div>
        <div class="skel-line s3" style="animation-delay:.70s"></div></div></div>
    </div>
    <p class="skel-msg">✦ &nbsp; Scanning cinematic patterns &nbsp;·&nbsp; fetching TMDB &amp; IMDb data &nbsp; ✦</p>
    """, unsafe_allow_html=True)

    # ── Run ML + API concurrently ────────────────────────────────
    with st.spinner("Analyzing the archive…"):
        recs = recommend(st.session_state["in_movie"], selected_df)

    skel.empty()
    st.session_state["recs"] = recs
    if recs:
        log_prediction(st.session_state["in_industry"], st.session_state["in_genre"],
                       st.session_state["in_actor"],    st.session_state["in_movie"], recs)


# ── Render stored recs ────────────────────────────────────────────
if st.session_state.get("recs") is not None:
    recs = st.session_state["recs"]

    if not recs:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">✂</div>
          <div class="empty-title">Left on the Cutting Room Floor</div>
          <div class="empty-body">
            No matches found for that combination. Try a different genre,
            actor filter, or choose another starting film.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="res-hdr">
          <p class="s-eye">Step 03</p>
          <p class="s-head">Curated Selections</p>
          <div class="prog-wrap"><div class="prog-bar"></div></div>
        </div>
        """, unsafe_allow_html=True)

        _PH  = "https://via.placeholder.com/500x750/0E0E12/C9A84C?text=No+Poster"
        _EC  = ["rc-e1","rc-e2","rc-e3","rc-e4","rc-e5"]
        _IND = html.escape(st.session_state.get("in_industry",""))

        def render_card(col, idx: int, rec: dict):
            t     = html.escape(str(rec["title"]))
            ov    = str(rec.get("overview",""))
            desc  = html.escape(ov[:210] + ("…" if len(ov)>210 else ""))
            img   = rec.get("poster")  or _PH
            href  = rec.get("trailer") or "#"
            trat  = rec.get("tmdb_rating", 0)
            irat  = rec.get("imdb_rating","N/A")
            yr    = rec.get("year","")
            gns   = rec.get("genres",[])
            score = rec.get("match_score", 0)
            spct  = f"{score:.0f}%"

            gpills = "".join(
                f'<span class="rc-gpill">{html.escape(g)}</span>'
                for g in gns[:3]
            )
            imdb_badge = (f'<span class="rc-imdb">⭐ {irat}</span>'
                          if irat != "N/A" else "")

            with col:
                # Entire card is a clickable trailer link
                st.markdown(f"""
                <a href="{href}" target="_blank" class="rec-link">
                  <div class="rec-card {_EC[idx]}">
                    <div class="card-img">
                      <img src="{img}" alt="{t}" loading="lazy">
                      <div class="card-scan"></div>
                      <div class="card-num">{idx+1:02d}</div>
                      <div class="card-ind">{_IND}</div>
                      <div class="card-ov">
                        <span class="ov-play">▶</span>
                        <p class="ov-syn">{desc}</p>
                      </div>
                    </div>
                    <div class="rc-body">
                      <div class="rc-title">{t}</div>
                      <div class="rc-meta">
                        <span>🎬 {trat}/10</span>
                        <span class="rc-dot"></span>
                        <span>{yr}</span>
                        <span class="rc-dot"></span>
                        {imdb_badge}
                      </div>
                      <div class="rc-grow">{gpills}</div>
                      <div class="match-wrap">
                        <div class="match-row">
                          <span>Match Score</span>
                          <span class="match-score">{spct}</span>
                        </div>
                        <div class="match-track">
                          <div class="match-fill" style="--match-pct:{spct}"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </a>
                """, unsafe_allow_html=True)

                # Log-click native button
                st.markdown('<div class="log-btn">', unsafe_allow_html=True)
                if st.button("🎬  Log & Watch", key=f"log_{idx}_{t[:8]}"):
                    log_click(rec["title"])
                    st.success(f"Logged! · [▶ Watch {rec['title']}]({href})")
                st.markdown("</div>", unsafe_allow_html=True)

        # Row 1 — 2 cards
        r1 = st.columns(2, gap="large")
        for i in range(min(2, len(recs))):
            render_card(r1[i], i, recs[i])

        # Row 2 — 2 cards
        if len(recs) > 2:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            r2 = st.columns(2, gap="large")
            for j in range(2, min(4, len(recs))):
                render_card(r2[j-2], j, recs[j])

        # Row 3 — 1 centred
        if len(recs) > 4:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            r3 = st.columns([1.2, 1.6, 1.2], gap="large")
            render_card(r3[1], 4, recs[4])


# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="divider" style="margin-top:4rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="ll-foot">
  <strong>LensLore v3</strong> &nbsp;·&nbsp; Intelligent Film Discovery<br>
  TF-IDF + KNN &nbsp;·&nbsp; <strong>TMDB API</strong> + <strong>OMDb / IMDb API</strong><br>
  Bollywood &amp; Hollywood &nbsp;·&nbsp; Crafted with Streamlit &nbsp;·&nbsp; © 2025
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  JAVASCRIPT — ALL INTERACTIVE EFFECTS (v3)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div id="cursor-glow"></div>
<script>
(function(){
'use strict';

const $  = (s,p) => (p||document).querySelector(s);
const $$ = (s,p) => [...(p||document).querySelectorAll(s)];
function wait(sel,cb,t=0){ const e=$(sel); if(e)return cb(e); if(t<60)setTimeout(()=>wait(sel,cb,t+1),100); }
function waitAll(sel,cb,t=0){ const e=$$(sel); if(e.length)return cb(e); if(t<60)setTimeout(()=>waitAll(sel,cb,t+1),100); }

/* ── 1. CURSOR GLOW ──────────────────────────────────────── */
const glow=$('#cursor-glow');
if(glow) document.addEventListener('mousemove',e=>{
  glow.style.left=e.clientX+'px'; glow.style.top=e.clientY+'px';
});

/* ── 2. FLOATING PARTICLES ───────────────────────────────── */
const pw=$('#ptcWrap');
if(pw){
  const spawn=()=>{
    const p=document.createElement('div');
    p.className='ptc';
    const dur=Math.random()*12+8, size=Math.random()*3+1, drift=(Math.random()-.5)*150;
    p.style.cssText=`left:${Math.random()*100}%;width:${size}px;height:${size}px;--drift:${drift}px;animation:floatParticle ${dur}s linear forwards`;
    pw.appendChild(p);
    setTimeout(()=>p.remove(),dur*1000);
  };
  setInterval(spawn,850);
  for(let i=0;i<12;i++) setTimeout(spawn,i*220);
}

/* ── 3. HERO STARS ───────────────────────────────────────── */
wait('#heroStars',c=>{
  for(let i=0;i<38;i++){
    const s=document.createElement('div');
    s.className='star';
    const sz=Math.random()*2.5+.5;
    s.style.cssText=`left:${Math.random()*100}%;top:${Math.random()*100}%;width:${sz}px;height:${sz}px;animation:starTwinkle ${Math.random()*3+2}s ${Math.random()*4}s ease-in-out infinite;opacity:${Math.random()*.6+.1}`;
    c.appendChild(s);
  }
});

/* ── 4. BUTTON RIPPLE ────────────────────────────────────── */
function addRipples(){
  $$('.stButton>button').forEach(btn=>{
    if(btn._rip)return; btn._rip=true;
    btn.addEventListener('click',e=>{
      const r=btn.getBoundingClientRect();
      const rpl=document.createElement('span');
      rpl.className='ripple-fx';
      rpl.style.left=(e.clientX-r.left)+'px';
      rpl.style.top=(e.clientY-r.top)+'px';
      btn.appendChild(rpl);
      setTimeout(()=>rpl.remove(),900);
    });
  });
}
[500,1500,3000,6000].forEach(t=>setTimeout(addRipples,t));

/* ── 5. MAGNETIC BUTTON ──────────────────────────────────── */
function addMagnet(){
  $$('.stButton>button').forEach(btn=>{
    if(btn._mag)return; btn._mag=true;
    btn.addEventListener('mousemove',e=>{
      const r=btn.getBoundingClientRect();
      const dx=(e.clientX-r.left-r.width/2)*.18;
      const dy=(e.clientY-r.top-r.height/2)*.18;
      btn.style.transform=`translate(${dx}px,${dy}px)`;
    });
    btn.addEventListener('mouseleave',()=>{
      btn.style.transition='transform .5s cubic-bezier(.25,1,.5,1)';
      btn.style.transform='';
    });
    btn.addEventListener('mouseenter',()=>btn.style.transition='none');
  });
}
[800,2500,5000].forEach(t=>setTimeout(addMagnet,t));

/* ── 6. SCROLL REVEAL ────────────────────────────────────── */
function initReveal(){
  const obs=new IntersectionObserver((entries)=>{
    entries.forEach((e,i)=>{
      if(e.isIntersecting){ setTimeout(()=>e.target.classList.add('on'),i*75); obs.unobserve(e.target); }
    });
  },{threshold:.08,rootMargin:'0px 0px -40px 0px'});
  $$('.rvl').forEach(el=>obs.observe(el));
}
[500,2000,4000].forEach(t=>setTimeout(initReveal,t));

/* ── 7. HERO PARALLAX ────────────────────────────────────── */
wait('.hero',hero=>{
  document.addEventListener('mousemove',e=>{
    const dx=(e.clientX-innerWidth/2)/innerWidth;
    const dy=(e.clientY-innerHeight/2)/innerHeight;
    const logo=$('.hero-logo',hero), icon=$('.hero-icon-wrap',hero);
    if(logo) logo.style.transform=`translate(${dx*7}px,${dy*3.5}px)`;
    if(icon) icon.style.transform=`translate(${dx*4}px,${dy*4}px)`;
  });
});

/* ── 8. LOGO GLOW on hover ───────────────────────────────── */
wait('.hero-logo',logo=>{
  logo.onmouseenter=()=>logo.style.filter='drop-shadow(0 0 36px rgba(201,168,76,.7))';
  logo.onmouseleave=()=>logo.style.filter='';
});

/* ── 9. 3-D TILT + PER-PIXEL AMBIENT GLOW on rec cards ───── */
function initCards(){
  $$('.rec-card').forEach(c=>{
    if(c._tilt)return; c._tilt=true;
    c.addEventListener('mousemove',e=>{
      const r=c.getBoundingClientRect();
      const rx=(e.clientX-r.left)/r.width-.5;
      const ry=(e.clientY-r.top)/r.height-.5;
      c.style.transform=`translateY(-16px)scale(1.028)rotateX(${-ry*8}deg)rotateY(${rx*8}deg)`;
      c.style.backgroundImage=`radial-gradient(circle at ${(rx+.5)*100}% ${(ry+.5)*100}%,rgba(201,168,76,.08) 0%,transparent 58%)`;
    });
    c.addEventListener('mouseleave',()=>{
      c.style.transform=''; c.style.backgroundImage='';
    });
  });
}
[1200,3000,6000].forEach(t=>setTimeout(initCards,t));

/* ── 10. SELECTED CARD parallax tilt ─────────────────────── */
wait('.sel-wrap',card=>{
  card.addEventListener('mousemove',e=>{
    const r=card.getBoundingClientRect();
    const ry=((e.clientY-r.top)/r.height-.5)*3;
    const rx=((e.clientX-r.left)/r.width-.5)*3;
    card.style.transform=`perspective(1100px) rotateX(${-ry}deg) rotateY(${rx}deg)`;
  });
  card.addEventListener('mouseleave',()=>card.style.transform='');
});

/* ── 11. STATS COUNT-UP ──────────────────────────────────── */
waitAll('.stat-num[data-target]',nums=>{
  const obs=new IntersectionObserver(entries=>{
    entries.forEach(en=>{
      if(!en.isIntersecting)return;
      const target=parseInt(en.target.getAttribute('data-target'));
      if(isNaN(target)){obs.unobserve(en.target);return;}
      let cur=0; const steps=55,dur=1300;
      const timer=setInterval(()=>{
        cur+=target/steps;
        if(cur>=target){cur=target;clearInterval(timer);}
        en.target.textContent=Math.floor(cur).toLocaleString();
      },dur/steps);
      obs.unobserve(en.target);
    });
  },{threshold:.5});
  nums.forEach(n=>obs.observe(n));
});

/* ── 12. FILM STRIP pause on hover ──────────────────────── */
$$('.strip-in').forEach(s=>{
  s.onmouseenter=()=>s.style.animationPlayState='paused';
  s.onmouseleave=()=>s.style.animationPlayState='running';
});

/* ── 13. MARQUEE pause on hover ─────────────────────────── */
wait('.marquee-inner',mi=>{
  const wrap=mi.closest('.marquee-wrap');
  if(wrap){
    wrap.onmouseenter=()=>mi.style.animationPlayState='paused';
    wrap.onmouseleave=()=>mi.style.animationPlayState='running';
  }
});

/* ── 14. INDUSTRY BADGE pulse on select ─────────────────── */
waitAll('div[data-baseweb="select"]',sels=>{
  sels.forEach(s=>s.addEventListener('mousedown',()=>{
    const b=$('.ind-badge');
    if(b){b.style.boxShadow='0 0 0 5px rgba(201,168,76,.3)'; setTimeout(()=>b.style.boxShadow='',700);}
  }));
});

/* ── 15. WELCOME TOAST (once per session) ────────────────── */
if(!sessionStorage.getItem('ll_v3')){
  sessionStorage.setItem('ll_v3','1');
  setTimeout(()=>{
    const t=document.createElement('div');
    t.className='ll-toast';
    t.innerHTML='<span class="ll-toast-ic">✦</span> Welcome to LensLore v3 — now with IMDb ratings';
    document.body.appendChild(t);
    setTimeout(()=>{
      t.style.transition='opacity .6s,transform .6s';
      t.style.opacity='0'; t.style.transform='translateY(20px)';
      setTimeout(()=>t.remove(),700);
    },4000);
  },1500);
}

/* ── 16. IMDB PILL shimmer on hover (CSS animation trigger) ─ */
waitAll('.imdb-pill',pills=>{
  pills.forEach(p=>{
    p.onmouseenter=()=>p.style.animationName='imdbShimmer';
    p.onmouseleave=()=>p.style.animationName='';
  });
});

})();
</script>
""", unsafe_allow_html=True)
