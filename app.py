import streamlit as st
import pandas as pd
import ast
import requests
import csv
import os
import html
import base64
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# ══════════════════════════════════════════════════════════════
#  ASSET LOADING
# ══════════════════════════════════════════════════════════════
img_icon = Image.open("assets/logo.png")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

hero_icon_base64 = get_base64_image("assets/logo.png")


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
#  GLOBAL CSS  — Cinematic Noir Luxury
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&family=Cormorant+Garamond:ital,wght@0,300;0,600;1,300&display=swap');

:root {
    --gold:         #C9A84C;
    --gold-light:   #F0D080;
    --gold-dim:     #7A6228;
    --gold-pale:    rgba(201,168,76,0.08);
    --bg:           #060608;
    --surface:      #0E0E12;
    --surface2:     #16161C;
    --surface3:     #1E1E26;
    --border:       #1A1A22;
    --border-gold:  rgba(201,168,76,0.2);
    --text:         #EDE5D4;
    --text-muted:   #777;
    --text-dim:     #2A2A35;
    --red-accent:   #8B2020;
    --teal-accent:  #1A4040;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 6rem !important; max-width: 1200px; margin: auto; position: relative; z-index: 1; }

::-webkit-scrollbar              { width: 4px; }
::-webkit-scrollbar-track        { background: var(--bg); }
::-webkit-scrollbar-thumb        { background: linear-gradient(180deg, var(--gold-dim), var(--gold)); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover  { background: var(--gold); }

/* ══════════════════════════
   KEYFRAMES
══════════════════════════ */
@keyframes apertureOpen {
    0%   { transform: scale(0.3) rotate(-90deg); opacity: 0; filter: blur(10px); }
    60%  { transform: scale(1.1) rotate(8deg); opacity: 1; }
    80%  { transform: scale(0.97) rotate(-2deg); }
    100% { transform: scale(1) rotate(0deg); opacity: 1; filter: blur(0); }
}
@keyframes titleReveal {
    0%   { opacity: 0; letter-spacing: 0.6em; transform: translateY(20px); }
    100% { opacity: 1; letter-spacing: -2px; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(40px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes buttonGlow {
    0%   { box-shadow: 0 0 8px rgba(201,168,76,0.15), inset 0 0 0px transparent; }
    50%  { box-shadow: 0 0 30px rgba(201,168,76,0.5), 0 0 60px rgba(201,168,76,0.15), inset 0 0 20px rgba(201,168,76,0.05); }
    100% { box-shadow: 0 0 8px rgba(201,168,76,0.15), inset 0 0 0px transparent; }
}
@keyframes filmScroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
@keyframes filmScrollReverse {
    0%   { transform: translateX(-50%); }
    100% { transform: translateX(0); }
}
@keyframes shimmerGold {
    0%   { background-position: -400% center; }
    100% { background-position:  400% center; }
}
@keyframes grain {
    0%,100% { transform: translate(0, 0) rotate(0deg); }
    20%     { transform: translate(-1.5%, -1%); }
    40%     { transform: translate(1%, -2%); }
    60%     { transform: translate(-0.5%, 1.5%); }
    80%     { transform: translate(2%, 0.5%); }
}
@keyframes ambientFloat1 {
    0%, 100% { transform: translate(0,0) scale(1); opacity: 0.04; }
    33%      { transform: translate(30px,-20px) scale(1.1); opacity: 0.07; }
    66%      { transform: translate(-20px,15px) scale(0.95); opacity: 0.03; }
}
@keyframes ambientFloat2 {
    0%, 100% { transform: translate(0,0) scale(1); opacity: 0.03; }
    33%      { transform: translate(-25px,30px) scale(1.08); opacity: 0.06; }
    66%      { transform: translate(20px,-10px) scale(0.92); opacity: 0.04; }
}
@keyframes pulseRing {
    0%   { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(201,168,76,0.4); }
    70%  { transform: scale(1); box-shadow: 0 0 0 12px rgba(201,168,76,0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(201,168,76,0); }
}
@keyframes scanLine {
    0%   { top: -10%; }
    100% { top: 110%; }
}
@keyframes rippleOut {
    0%   { transform: scale(0); opacity: 0.6; }
    100% { transform: scale(4); opacity: 0; }
}
@keyframes typewriter {
    from { width: 0; }
    to   { width: 100%; }
}
@keyframes blinkCaret {
    from, to { border-color: transparent; }
    50%      { border-color: var(--gold); }
}
@keyframes revealSlide {
    from { opacity: 0; transform: translateY(50px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes logoGlitch {
    0%, 95%, 100% { text-shadow: none; transform: none; }
    96%  { text-shadow: 2px 0 #C9A84C, -2px 0 #8B2020; transform: translateX(2px); }
    97%  { text-shadow: -2px 0 #C9A84C, 2px 0 #8B2020; transform: translateX(-2px); }
    98%  { text-shadow: 1px 0 #C9A84C; transform: translateX(1px); }
}
@keyframes starTwinkle {
    0%, 100% { opacity: 0.2; transform: scale(1); }
    50%      { opacity: 1; transform: scale(1.4); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes floatParticle {
    0%   { transform: translateY(100vh) translateX(0px) scale(0); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 0.6; }
    100% { transform: translateY(-100px) translateX(var(--drift)) scale(1); opacity: 0; }
}
@keyframes cardEntrance {
    0%   { opacity: 0; transform: translateY(60px) rotateX(10deg) scale(0.95); filter: blur(4px); }
    100% { opacity: 1; transform: translateY(0) rotateX(0deg) scale(1); filter: blur(0); }
}
@keyframes holographicShimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes borderDash {
    to { stroke-dashoffset: -100; }
}
@keyframes industryPill {
    from { transform: scale(0.9); opacity: 0; }
    to   { transform: scale(1); opacity: 1; }
}
@keyframes progressFill {
    from { width: 0%; }
    to   { width: 100%; }
}
@keyframes countUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes successPulse {
    0%   { box-shadow: 0 0 0 0 rgba(201,168,76,0.6); }
    70%  { box-shadow: 0 0 0 20px rgba(201,168,76,0); }
    100% { box-shadow: 0 0 0 0 rgba(201,168,76,0); }
}

/* ══════════════════════════
   CURSOR GLOW (via JS)
══════════════════════════ */
#cursor-glow {
    width: 300px; height: 300px;
    border-radius: 50%;
    position: fixed;
    pointer-events: none;
    z-index: 9999;
    background: radial-gradient(circle, rgba(201,168,76,0.06) 0%, transparent 70%);
    transform: translate(-50%, -50%);
    transition: opacity 0.3s ease;
    mix-blend-mode: screen;
}

/* ══════════════════════════
   PARTICLES
══════════════════════════ */
.particles-container {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.particle {
    position: absolute;
    bottom: 0;
    width: 3px; height: 3px;
    border-radius: 50%;
    background: var(--gold);
    opacity: 0;
}

/* ══════════════════════════
   AMBIENT BACKGROUND BLOBS
══════════════════════════ */
.ambient-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}
.amb-blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
}
.amb-blob-1 {
    width: 600px; height: 600px;
    top: -10%; left: -10%;
    background: radial-gradient(circle, rgba(201,168,76,0.06) 0%, transparent 70%);
    animation: ambientFloat1 18s ease-in-out infinite;
}
.amb-blob-2 {
    width: 500px; height: 500px;
    bottom: 10%; right: -5%;
    background: radial-gradient(circle, rgba(139,32,32,0.05) 0%, transparent 70%);
    animation: ambientFloat2 22s ease-in-out infinite;
}
.amb-blob-3 {
    width: 400px; height: 400px;
    top: 40%; left: 50%;
    background: radial-gradient(circle, rgba(26,64,64,0.06) 0%, transparent 70%);
    animation: ambientFloat1 26s ease-in-out infinite reverse;
}

/* Film grain overlay */
body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-size: 200px 200px;
    opacity: 0.04;
    animation: grain 0.3s steps(1) infinite;
}

/* Vignette */
body::after {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.7) 100%);
}

/* ══════════════════════════
   FILM STRIP TICKER (DUAL)
══════════════════════════ */
.strips-container {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
}
.strip-wrap {
    overflow: hidden;
    user-select: none;
    border-bottom: 1px solid var(--border);
}
.strip-wrap:last-child { border-bottom: none; }
.strip-inner {
    display: flex;
    width: max-content;
    align-items: center;
    padding: 6px 0;
}
.strip-inner.fwd  { animation: filmScroll 32s linear infinite; }
.strip-inner.rev  { animation: filmScrollReverse 28s linear infinite; }
.s-hole {
    width: 14px; height: 10px;
    background: var(--bg);
    border-radius: 2px;
    border: 1px solid var(--border);
    margin: 0 10px;
    flex-shrink: 0;
}
.s-label {
    color: var(--gold-dim);
    font-size: 0.58em;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    padding: 0 8px;
    white-space: nowrap;
    font-family: 'DM Sans', monospace;
}
.s-frame-num {
    color: var(--text-dim);
    font-size: 0.48em;
    font-family: monospace;
    letter-spacing: 0.05em;
    padding: 0 4px;
}

/* ══════════════════════════
   HERO
══════════════════════════ */
.hero {
    text-align: center;
    padding: 4rem 1rem 2.5rem;
    position: relative;
    z-index: 1;
}
.hero-icon-wrap {
    position: relative;
    display: inline-block;
    margin-bottom: 0.5em;
    animation: apertureOpen 1.2s cubic-bezier(.34,1.56,.64,1) both;
}
.hero-icon-wrap::before {
    content: '';
    position: absolute;
    inset: -15px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
    animation: pulseRing 3s ease-out infinite;
}
.hero-icon-img {
    height: 68px;
    width: auto;
    display: block;
    filter: drop-shadow(0 0 20px rgba(201,168,76,0.4));
}
.hero-logo {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3em, 9vw, 6.5em);
    font-weight: 900;
    line-height: 1;
    margin: 0.1em 0 0;
    animation: titleReveal 1.1s cubic-bezier(.77,0,.18,1) 0.3s both;
}
.logo-l {
    color: var(--gold);
    animation: logoGlitch 8s ease-in-out infinite;
    display: inline-block;
    text-shadow: 0 0 40px rgba(201,168,76,0.3);
}
.logo-r {
    color: var(--text);
    display: inline-block;
}
.hero-tagline {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.15em;
    color: var(--text-muted);
    margin-top: 0.6em;
    animation: fadeUp 0.8s ease 1s both;
    letter-spacing: 0.05em;
}
.hero-sub {
    font-size: 0.72em;
    color: var(--text-muted);
    letter-spacing: 0.4em;
    text-transform: uppercase;
    margin-top: 0.7em;
    animation: fadeUp 0.8s ease 1.1s both;
    overflow: hidden;
    white-space: nowrap;
    display: inline-block;
}
.hero-line {
    width: 80px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.6rem auto 0;
    animation: fadeUp 0.8s ease 1.2s both;
    position: relative;
}
.hero-line::before, .hero-line::after {
    content: '◆';
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6px;
    color: var(--gold);
}
.hero-line::before { left: -10px; }
.hero-line::after  { right: -10px; }

/* Stars in hero */
.hero-stars {
    position: absolute;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
    z-index: 0;
}
.star {
    position: absolute;
    width: 2px; height: 2px;
    background: var(--gold);
    border-radius: 50%;
}

/* Industry badge */
.industry-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 1.2em;
    padding: 6px 18px;
    border-radius: 30px;
    font-size: 0.7em;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    font-weight: 600;
    animation: industryPill 0.5s ease both;
    border: 1px solid var(--gold-dim);
    background: rgba(201,168,76,0.06);
    color: var(--gold);
}
.industry-badge .badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--gold);
    animation: pulseRing 2s ease-out infinite;
}

/* ══════════════════════════
   DIVIDER
══════════════════════════ */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--gold-dim) 20%, var(--gold) 50%, var(--gold-dim) 80%, transparent 100%);
    background-size: 400% auto;
    animation: shimmerGold 4s linear infinite;
    margin: 2.5rem 0;
    border: none;
    position: relative;
}
.divider::before {
    content: '✦';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 8px;
    color: var(--gold);
    background: var(--bg);
    padding: 0 8px;
}

/* ══════════════════════════
   SECTION LABELS
══════════════════════════ */
.s-eyebrow {
    font-size: 0.65em;
    font-weight: 600;
    letter-spacing: 0.38em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.2em;
    display: flex;
    align-items: center;
    gap: 10px;
}
.s-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px;
    height: 1px;
    background: var(--gold);
}
.s-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.8em;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 1.2em;
    line-height: 1.1;
}

/* ══════════════════════════
   SELECTBOX
══════════════════════════ */
label[data-testid="stWidgetLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72em !important;
    font-weight: 500 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--gold-dim) !important;
    box-shadow: 0 0 12px rgba(201,168,76,0.1) !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}
[data-baseweb="popover"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
}
[role="option"]:hover {
    background: rgba(201,168,76,0.08) !important;
}
[role="option"][aria-selected="true"] {
    background: rgba(201,168,76,0.12) !important;
    color: var(--gold-light) !important;
}

/* ══════════════════════════
   BUTTON (Enhanced with ripple container)
══════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg,
        rgba(6,6,8,1) 0%,
        rgba(201,168,76,0.12) 50%,
        rgba(6,6,8,1) 100%) !important;
    background-size: 300% auto !important;
    color: var(--gold-light) !important;
    border: 1px solid var(--gold) !important;
    padding: 1em 3.5em !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82em !important;
    font-weight: 600 !important;
    letter-spacing: 0.35em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    animation: buttonGlow 4s ease-in-out infinite !important;
    position: relative !important;
    overflow: hidden !important;
    display: block !important;
    margin: 0 auto !important;
    cursor: pointer !important;
}
.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent) !important;
    transform: translateX(-100%) !important;
    transition: transform 0.6s ease !important;
}
.stButton > button:hover::before {
    transform: translateX(100%) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold-light) 50%, var(--gold) 100%) !important;
    color: #060608 !important;
    box-shadow: 0 0 40px rgba(201,168,76,0.5), 0 8px 25px rgba(0,0,0,0.6) !important;
    transform: translateY(-4px) scale(1.04) !important;
    border-color: var(--gold-light) !important;
    letter-spacing: 0.42em !important;
    animation: none !important;
}
.stButton > button:active {
    transform: translateY(-1px) scale(1.01) !important;
}

/* Ripple */
.ripple-effect {
    position: absolute;
    border-radius: 50%;
    background: rgba(201,168,76,0.4);
    width: 100px; height: 100px;
    margin-top: -50px; margin-left: -50px;
    animation: rippleOut 0.8s linear;
    pointer-events: none;
}

/* ══════════════════════════
   REC CARDS (Enhanced)
══════════════════════════ */
.rec-card-link {
    text-decoration: none;
    display: block;
    border-radius: 10px;
    outline: none;
}
.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    cursor: pointer;
    height: 100%;
    perspective: 1000px;
}
.rec-card::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 10px;
    border: 1px solid transparent;
    transition: border-color 0.4s, box-shadow 0.4s;
    pointer-events: none;
}
.rec-card:hover::after {
    border-color: var(--gold);
    box-shadow: 0 0 20px rgba(201,168,76,0.2) inset;
}
.rec-card:hover {
    transform: translateY(-14px) scale(1.02) rotateX(2deg);
    box-shadow: 0 35px 70px rgba(0,0,0,0.95), 0 0 0 1px var(--gold-dim), 0 0 50px rgba(201,168,76,0.12);
}

/* Holographic edge on hover */
.rec-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 10px;
    background: linear-gradient(135deg,
        rgba(201,168,76,0) 0%,
        rgba(201,168,76,0.04) 40%,
        rgba(201,168,76,0) 60%,
        rgba(201,168,76,0.06) 100%);
    background-size: 200% 200%;
    opacity: 0;
    transition: opacity 0.4s;
    pointer-events: none;
    z-index: 1;
    animation: holographicShimmer 3s ease infinite;
}
.rec-card:hover::before { opacity: 1; }

.card-image-wrapper {
    position: relative;
    width: 100%;
    aspect-ratio: 2/3;
    overflow: hidden;
    background: #0A0A0F;
}
.card-image-wrapper img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: filter 0.5s ease, transform 0.7s ease;
}
/* Scanline effect on hover */
.card-scanline {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(201,168,76,0.3), transparent);
    opacity: 0;
    pointer-events: none;
    z-index: 5;
    transition: opacity 0.3s;
}
.rec-card:hover .card-scanline {
    opacity: 1;
    animation: scanLine 2s linear infinite;
}
.card-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, rgba(6,6,8,0.98) 0%, rgba(6,6,8,0.85) 60%, rgba(6,6,8,0.5) 100%);
    color: var(--text);
    opacity: 0;
    transition: opacity 0.45s ease;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
    text-align: center;
    font-size: 0.88em;
    line-height: 1.65;
    border-bottom: 2px solid var(--gold);
}
.card-overlay-icon {
    font-size: 1.8em;
    margin-bottom: 10px;
    display: block;
    color: var(--gold);
}
.rec-card:hover .card-image-wrapper img {
    filter: blur(8px) brightness(0.35) saturate(0.5);
    transform: scale(1.1);
}
.rec-card:hover .card-overlay { opacity: 1; }
.rc-body {
    padding: 18px 16px;
    border-top: 1px solid var(--border);
    background: linear-gradient(180deg, var(--surface) 0%, var(--surface2) 100%);
}
.rc-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05em;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 7px;
    line-height: 1.25;
}
.rc-meta {
    font-size: 0.78em;
    color: var(--gold);
    font-weight: 500;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.rc-meta-dot {
    width: 3px; height: 3px;
    border-radius: 50%;
    background: var(--gold-dim);
    display: inline-block;
}

/* Card number badge */
.card-num {
    position: absolute;
    top: 12px; left: 12px;
    background: rgba(6,6,8,0.85);
    border: 1px solid var(--gold-dim);
    border-radius: 50%;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65em;
    font-weight: 600;
    color: var(--gold);
    letter-spacing: 0.05em;
    z-index: 2;
    font-family: 'Cormorant Garamond', serif;
}

/* Industry tag on card */
.card-industry-tag {
    position: absolute;
    top: 12px; right: 12px;
    background: rgba(6,6,8,0.85);
    border: 1px solid var(--gold-dim);
    border-radius: 3px;
    padding: 3px 8px;
    font-size: 0.55em;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    z-index: 2;
}

/* Card entrance animation with stagger */
.rec-card.entrance-1 { animation: cardEntrance 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.05s both; }
.rec-card.entrance-2 { animation: cardEntrance 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.15s both; }
.rec-card.entrance-3 { animation: cardEntrance 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.25s both; }
.rec-card.entrance-4 { animation: cardEntrance 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.35s both; }
.rec-card.entrance-5 { animation: cardEntrance 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.45s both; }

/* ══════════════════════════
   SELECTED MOVIE CARD
══════════════════════════ */
.sel-card-wrapper {
    display: flex;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 10px;
    overflow: hidden;
    animation: revealSlide 0.7s ease both;
    transition: box-shadow 0.4s;
    position: relative;
}
.sel-card-wrapper::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(201,168,76,0.03) 0%, transparent 60%);
    pointer-events: none;
}
.sel-card-wrapper:hover {
    box-shadow: 0 20px 50px rgba(0,0,0,0.8), -4px 0 20px rgba(201,168,76,0.15);
}
.sel-poster-wrap {
    flex: 0 0 32%;
    position: relative;
    overflow: hidden;
    min-height: 280px;
}
.sel-poster-wrap img {
    width: 100%; height: 100%; object-fit: cover;
    transition: filter 0.5s ease, transform 0.7s ease;
}
.sel-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(to top, rgba(6,6,8,0.95), rgba(6,6,8,0.75) 50%, rgba(6,6,8,0.4));
    opacity: 0; transition: opacity 0.45s;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 24px; text-align: center; font-size: 0.9em;
    color: var(--text);
    border-right: 1px solid var(--gold-dim);
}
.sel-overlay-icon { font-size: 2em; color: var(--gold); margin-bottom: 12px; display: block; }
.sel-card-wrapper:hover .sel-poster-wrap img {
    filter: blur(8px) brightness(0.3) saturate(0.4);
    transform: scale(1.06);
}
.sel-card-wrapper:hover .sel-overlay { opacity: 1; }
.sel-info {
    flex: 1;
    padding: 28px 30px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
}
.rating-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(201,168,76,0.08);
    border: 1px solid rgba(201,168,76,0.3);
    color: var(--gold-light);
    font-size: 0.78em;
    font-weight: 600;
    border-radius: 40px;
    padding: 4px 14px;
    width: fit-content;
    margin-bottom: 12px;
    letter-spacing: 0.05em;
    animation: pulseRing 4s ease-out infinite;
}
.sel-title {
    font-family: 'Playfair Display', serif;
    font-size: 2em;
    font-weight: 700;
    margin: 0 0 5px;
    line-height: 1.1;
    color: var(--text);
}
.sel-year {
    color: var(--text-muted);
    font-size: 0.92em;
    letter-spacing: 0.12em;
    margin-bottom: 16px;
    font-family: 'DM Sans', sans-serif;
}
.sel-synopsis {
    color: #9A927F;
    line-height: 1.7;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05em;
}
.sel-click-hint {
    color: var(--gold-dim);
    font-size: 0.72em;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: auto;
    padding-top: 16px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sel-click-hint::before {
    content: '';
    display: inline-block;
    width: 16px;
    height: 1px;
    background: var(--gold-dim);
}

/* ══════════════════════════
   SPINNER
══════════════════════════ */
.stSpinner > div {
    border-top-color: var(--gold) !important;
    border-right-color: rgba(201,168,76,0.3) !important;
    border-bottom-color: rgba(201,168,76,0.1) !important;
    border-left-color: rgba(201,168,76,0.2) !important;
}

/* ══════════════════════════
   FOOTER
══════════════════════════ */
.ll-footer {
    text-align: center;
    padding: 3rem 0 2rem;
    color: #2A2A35;
    font-size: 0.67em;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    line-height: 2.4;
    position: relative;
}
.ll-footer::before {
    content: '— — —';
    display: block;
    color: var(--gold-dim);
    font-size: 0.9em;
    letter-spacing: 0.5em;
    margin-bottom: 0.5em;
}
.ll-footer strong { color: var(--gold-dim); font-weight: 600; }
.ll-footer a { color: var(--gold-dim); text-decoration: none; transition: color 0.2s; }
.ll-footer a:hover { color: var(--gold); }

/* ══════════════════════════
   SCROLL REVEAL
══════════════════════════ */
.reveal-on-scroll {
    opacity: 0;
    transform: translateY(40px);
    transition: opacity 0.7s ease, transform 0.7s ease;
}
.reveal-on-scroll.revealed {
    opacity: 1;
    transform: translateY(0);
}

/* ══════════════════════════
   STATS BAR
══════════════════════════ */
.stats-bar {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin: 2rem 0;
    animation: fadeUp 0.8s ease 0.5s both;
}
.stat-item {
    flex: 1;
    text-align: center;
    padding: 18px 12px;
    border-right: 1px solid var(--border);
    background: var(--surface);
    transition: background 0.3s, transform 0.3s;
    cursor: default;
}
.stat-item:last-child { border-right: none; }
.stat-item:hover { background: var(--surface2); transform: translateY(-2px); }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.8em;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
}
.stat-label {
    font-size: 0.6em;
    color: var(--text-muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ══════════════════════════
   TOAST NOTIFICATION
══════════════════════════ */
.ll-toast {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background: var(--surface2);
    border: 1px solid var(--gold-dim);
    border-left: 3px solid var(--gold);
    border-radius: 6px;
    padding: 14px 20px;
    color: var(--text);
    font-size: 0.82em;
    letter-spacing: 0.05em;
    z-index: 9998;
    animation: fadeUp 0.4s ease both;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    display: flex;
    align-items: center;
    gap: 10px;
}
.ll-toast-icon { color: var(--gold); font-size: 1.1em; }

/* ══════════════════════════
   SUCCESS FLASH (on discover)
══════════════════════════ */
.results-header {
    animation: successPulse 0.8s ease both;
}

/* ══════════════════════════
   LOADING PROGRESS BAR
══════════════════════════ */
.ll-progress-wrap {
    width: 100%;
    height: 2px;
    background: var(--surface3);
    border-radius: 2px;
    overflow: hidden;
    margin: 1rem 0;
}
.ll-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--gold-dim), var(--gold), var(--gold-light));
    animation: progressFill 1.5s ease forwards;
}

/* ══════════════════════════
   SECTION TRANSITION
══════════════════════════ */
.section-reveal {
    animation: revealSlide 0.6s ease both;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  AMBIENT BACKGROUND BLOBS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="ambient-bg">
    <div class="amb-blob amb-blob-1"></div>
    <div class="amb-blob amb-blob-2"></div>
    <div class="amb-blob amb-blob-3"></div>
</div>
<div class="particles-container" id="particlesContainer"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FILM STRIP TICKER (DUAL ROW)
# ══════════════════════════════════════════════════════════════
genres_top = ["Action", "Drama", "Thriller", "Romance", "Comedy",
              "Sci-Fi", "Horror", "Mystery", "Biography", "Fantasy",
              "Adventure", "Crime", "Animation", "History", "Musical"]
genres_bot = ["Noir", "Western", "War", "Sport", "Family",
              "Documentary", "Superhero", "Anthology", "Neo-noir", "Surrealist",
              "Gangster", "Heist", "Psychological", "Existential", "Satire"]

def make_strip(genres, n=6, frame_start=1):
    items = []
    frame = frame_start
    for g in genres * n:
        items.append(f'<span class="s-frame-num">{frame:04d}</span>'
                     f'<div class="s-hole"></div>'
                     f'<span class="s-label">{g}</span>'
                     f'<div class="s-hole"></div>')
        frame += 1
    return ''.join(items)

st.markdown(f"""
<div class="strips-container">
  <div class="strip-wrap">
    <div class="strip-inner fwd">{make_strip(genres_top, frame_start=1)}</div>
  </div>
  <div class="strip-wrap">
    <div class="strip-inner rev">{make_strip(genres_bot, frame_start=100)}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DATA LOADING
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
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US&append_to_response=videos"
        data = requests.get(url, headers=TMDB_HEADERS, timeout=6).json()
        poster_path = data.get('poster_path')
        poster_url  = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
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
#  SMART TAG BUILDER  ← FIX: build tags from all available cols
#  This ensures Hollywood movies work even when 'tags' col is
#  missing, empty, or structured differently from Bollywood.
# ══════════════════════════════════════════════════════════════
def build_tags_column(df):
    """
    Robust tag builder that:
    1. Uses existing 'tags' column if present and non-empty
    2. Falls back to assembling tags from genres, cast, keywords,
       overview, director, and other available text columns
    3. Returns df with a guaranteed non-null 'tags' column
    """
    df = df.copy()

    # Candidate source columns (case-insensitive lookup)
    col_map = {c.lower(): c for c in df.columns}

    def get_col(name):
        return col_map.get(name.lower())

    def col_to_text(series):
        """Convert a column (list-like or string) to space-separated text."""
        def row_text(v):
            if pd.isna(v) or str(v).strip() in ('', 'nan', 'None'):
                return ''
            if isinstance(v, list):
                return ' '.join(str(x) for x in v if x)
            try:
                parsed = ast.literal_eval(str(v))
                if isinstance(parsed, list):
                    return ' '.join(str(x) for x in parsed if x)
            except Exception:
                pass
            return str(v).replace(',', ' ')
        return series.apply(row_text)

    # Check if 'tags' column already exists and is usable
    tags_col = get_col('tags')
    if tags_col:
        existing = df[tags_col].apply(
            lambda v: col_to_text(pd.Series([v]))[0] if not pd.isna(v) else ''
        )
        non_empty = (existing.str.strip() != '').sum()
        if non_empty > len(df) * 0.3:   # at least 30% rows have real tags
            df['tags'] = existing
            return df

    # Build from components
    parts = []
    for col_name in ['genres', 'cast', 'keywords', 'overview', 'director',
                     'tagline', 'production_companies', 'spoken_languages']:
        real_col = get_col(col_name)
        if real_col:
            parts.append(col_to_text(df[real_col]))

    if parts:
        combined = parts[0].copy()
        for p in parts[1:]:
            combined = combined + ' ' + p
        df['tags'] = combined.str.strip()
    else:
        # Last resort: concat all text-like columns
        text_cols = [c for c in df.columns
                     if df[c].dtype == object and c.lower() not in ('title', 'movie_id', 'id')]
        if text_cols:
            df['tags'] = df[text_cols].fillna('').agg(' '.join, axis=1)
        else:
            df['tags'] = df['title'].fillna('')

    # Ensure no NaN
    df['tags'] = df['tags'].fillna('').astype(str)
    return df


def get_movie_id(row, df_columns):
    """
    Robustly fetch movie_id from a row, trying several common column names.
    Handles Bollywood (movie_id) and Hollywood (id, tmdb_id, etc.)
    """
    for candidate in ['movie_id', 'id', 'tmdb_id', 'movieid', 'film_id']:
        if candidate in df_columns:
            val = row.get(candidate)
            if val and not pd.isna(val):
                return int(val)
    return None


# ══════════════════════════════════════════════════════════════
#  OUTCOME LOGGER  ← FIX: saves to logs/prediction.csv
# ══════════════════════════════════════════════════════════════
LOG_PATH = "logs/prediction.csv"
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
    for j in range(len(recs) + 1, 6):
        row[f"rec_{j}"] = ""
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
#  FIX 1 — min_df=1 so Hollywood's unique tags are NOT dropped
#  FIX 2 — df passed as parameter (no global mutation)
#  FIX 3 — n_neighbors capped to dataset size (no crash on small sets)
# ══════════════════════════════════════════════════════════════
def recommend(movie_name_input, df):
    movies, posters, links, ratings, years, overviews = [], [], [], [], [], []

    # Ensure tags are built and are plain strings
    df = build_tags_column(df)
    df = df.reset_index(drop=True)

    if df.shape[0] < 2:
        st.warning("Not enough films for similarity matching.")
        return movies, posters, links, ratings, years, overviews

    # If we have fewer than 6 rows, just return them all
    n_rec = min(5, df.shape[0] - 1)
    if n_rec == 0:
        return movies, posters, links, ratings, years, overviews

    try:
        # FIX: min_df=1 ensures no terms are dropped on smaller/unique datasets
        tfidf   = TfidfVectorizer(max_features=5000, stop_words='english', min_df=1)
        vectors = tfidf.fit_transform(df['tags']).toarray()
    except Exception as e:
        st.error(f"Vectorizer error: {e}")
        return movies, posters, links, ratings, years, overviews

    # Safety: if all vectors are zero (empty tags) fall back to title matching
    if vectors.sum() == 0:
        st.warning("Tags appear empty — falling back to title-based listing.")
        subset = df[df['title'] != movie_name_input].head(5)
        for _, row in subset.iterrows():
            mid = get_movie_id(row, df.columns)
            p, l, r, y, o = fetch_movie_details(mid)
            movies.append(row['title']); posters.append(p)
            links.append(l); ratings.append(r); years.append(y); overviews.append(o)
        return movies, posters, links, ratings, years, overviews

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(vectors)

    matches = df[df['title'] == movie_name_input]
    if matches.empty:
        st.error(f'Movie "{movie_name_input}" not found in current dataset.')
        return movies, posters, links, ratings, years, overviews

    idx = matches.index[0]
    n_neighbors = min(n_rec + 1, len(df))   # +1 because first result = itself
    distances, indices = model.kneighbors([vectors[idx]], n_neighbors=n_neighbors)

    for i in indices[0][1:]:    # skip index 0 (the movie itself)
        movie = df.iloc[i]
        mid   = get_movie_id(movie, df.columns)
        p, l, r, y, o = fetch_movie_details(mid)
        movies.append(movie['title']); posters.append(p)
        links.append(l); ratings.append(r); years.append(y); overviews.append(o)

    return movies, posters, links, ratings, years, overviews


# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
industry_display = st.session_state.get("industry_choice", "Bollywood & Hollywood")
st.markdown(f"""
<div class="hero">
  <div class="hero-stars" id="heroStars"></div>
  <div class="hero-icon-wrap">
    <img src="data:image/png;base64,{hero_icon_base64}" class="hero-icon-img" alt="LensLore logo">
  </div>
  <div class="hero-logo">
    <span class="logo-l">Lens</span><span class="logo-r">Lore</span>
  </div>
  <p class="hero-tagline">Where every frame tells a story</p>
  <p class="hero-sub" id="heroSub">Intelligent Cinema Discovery &nbsp;·&nbsp; Bollywood &amp; Hollywood</p>
  <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  STATS BAR
# ══════════════════════════════════════════════════════════════
boll_count = len(movie_boll)
holl_count = len(movie_holl)
st.markdown(f"""
<div class="stats-bar reveal-on-scroll">
  <div class="stat-item">
    <div class="stat-num">{boll_count:,}</div>
    <div class="stat-label">Bollywood Films</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">{holl_count:,}</div>
    <div class="stat-label">Hollywood Films</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">{boll_count + holl_count:,}</div>
    <div class="stat-label">Total Archive</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">TF-IDF</div>
    <div class="stat-label">Similarity Engine</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FILTERS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="s-eyebrow">Step 01</p>'
            '<p class="s-heading">Set the Stage</p>', unsafe_allow_html=True)

industry    = st.selectbox('Industry', ('Bollywood', 'Hollywood'))
st.session_state["industry_choice"] = industry
selected_df = (movie_boll if industry == 'Bollywood' else movie_holl).copy()

# Show which industry is active as a badge
st.markdown(f"""
<div style="margin: -0.5rem 0 1rem;">
  <span class="industry-badge">
    <span class="badge-dot"></span>
    {industry} · {len(selected_df):,} films loaded
  </span>
</div>
""", unsafe_allow_html=True)

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
st.markdown('<p class="s-eyebrow">Step 02</p>'
            '<p class="s-heading">Search the Archive</p>', unsafe_allow_html=True)

movie_list     = selected_df['title'].tolist()
selected_movie = st.selectbox("Title", sorted(movie_list), key="movie_select")


# ══════════════════════════════════════════════════════════════
#  SELECTED MOVIE PREVIEW
# ══════════════════════════════════════════════════════════════
sel_row = selected_df[selected_df['title'] == selected_movie]
if not sel_row.empty:
    sel_id = get_movie_id(sel_row.iloc[0], selected_df.columns)
    s_poster, s_link, s_rating, s_year, s_overview = fetch_movie_details(sel_id)

    safe_title    = html.escape(selected_movie)
    safe_overview = html.escape(s_overview or "No synopsis available.")
    display_poster = s_poster if s_poster else "https://via.placeholder.com/500x750/0E0E12/C9A84C?text=No+Poster"

    st.markdown(f"""
    <a href="{s_link}" target="_blank" style="text-decoration: none; display: block; margin-top: 18px;">
        <div class="sel-card-wrapper">
            <div class="sel-poster-wrap">
                <img src="{display_poster}" alt="{safe_title}">
                <div class="sel-overlay">
                    <span class="sel-overlay-icon">▶</span>
                    <p>Watch Trailer or Visit Official Site</p>
                </div>
            </div>
            <div class="sel-info">
                <div class="rating-pill">★ &nbsp;{s_rating} / 10</div>
                <div class="sel-title">{safe_title}</div>
                <div class="sel-year">{s_year}</div>
                <div class="sel-synopsis">{safe_overview}</div>
                <div class="sel-click-hint">Click card to view media</div>
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DISCOVER BUTTON
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider" style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
bc = st.columns([2.5, 1, 2.5])
with bc[1]:
    if st.button("✦  Discover Films"):
        st.session_state.recommend_triggered = True
        st.session_state.last_industry = industry
        st.session_state.last_movie    = selected_movie


# ══════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════
if st.session_state.get("recommend_triggered"):
    with st.spinner("Scanning the archive…"):
        # FIX: pass selected_df as parameter — no global mutation
        recs, posters, links, ratings, years, overviews = recommend(selected_movie, selected_df)

    if recs:
        log_recommendation(industry, genre, actor, selected_movie, recs)

    if recs:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="results-header">
          <p class="s-eyebrow">Step 03</p>
          <p class="s-heading">Curated Selections</p>
          <div class="ll-progress-wrap"><div class="ll-progress-bar"></div></div>
        </div>
        """, unsafe_allow_html=True)

        def rec_card(col, title, poster, link, rating, year, overview, entry_class, num=1, ind=""):
            safe_title = html.escape(str(title))
            safe_desc  = html.escape(str(overview or ""))
            if len(safe_desc) > 180:
                safe_desc = safe_desc[:180] + "…"
            display_img = poster if poster else "https://via.placeholder.com/500x750/0E0E12/C9A84C?text=No+Poster"
            href = link if link else "#"
            ind_tag = f'<div class="card-industry-tag">{html.escape(ind)}</div>' if ind else ''

            with col:
                st.markdown(f"""
                <a href="{href}" target="_blank" class="rec-card-link">
                    <div class="rec-card {entry_class}">
                        <div class="card-image-wrapper">
                            <img src="{display_img}" alt="{safe_title}" loading="lazy">
                            <div class="card-scanline"></div>
                            <div class="card-num">{num:02d}</div>
                            {ind_tag}
                            <div class="card-overlay">
                                <span class="card-overlay-icon">▶</span>
                                <p>{safe_desc}</p>
                            </div>
                        </div>
                        <div class="rc-body">
                            <div class="rc-title">{safe_title}</div>
                            <div class="rc-meta">
                                <span>★ {rating}</span>
                                <span class="rc-meta-dot"></span>
                                <span>{year}</span>
                            </div>
                        </div>
                    </div>
                </a>
                """, unsafe_allow_html=True)

        entrance_classes = ["entrance-1", "entrance-2", "entrance-3", "entrance-4", "entrance-5"]

        # Row 1 — 2 cards
        r1 = st.columns(2, gap="large")
        for i in range(min(2, len(recs))):
            rec_card(r1[i], recs[i], posters[i], links[i], ratings[i], years[i],
                     overviews[i], entrance_classes[i], i + 1, industry)

        # Row 2 — 2 cards
        if len(recs) > 2:
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            r2 = st.columns(2, gap="large")
            for j in range(2, min(4, len(recs))):
                rec_card(r2[j - 2], recs[j], posters[j], links[j], ratings[j], years[j],
                         overviews[j], entrance_classes[j], j + 1, industry)

        # Row 3 — 1 centred card
        if len(recs) > 4:
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            r3 = st.columns([1.2, 1.6, 1.2], gap="large")
            rec_card(r3[1], recs[4], posters[4], links[4], ratings[4], years[4],
                     overviews[4], entrance_classes[4], 5, industry)

    else:
        st.error("No recommendations found. Try a different movie, genre, or actor filter.")


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider" style="margin-top: 4rem;"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="ll-footer">
    <strong>LensLore</strong> &nbsp;·&nbsp; Intelligent Film Discovery
    <br>
    TF-IDF &amp; Cosine Similarity &nbsp;·&nbsp; Powered by <strong>TMDB API</strong>
    <br>
    Crafted with Streamlit &nbsp;·&nbsp; &copy; 2025
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  JAVASCRIPT — Interactive Effects
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div id="cursor-glow"></div>

<script>
(function() {
    /* ── Helpers ── */
    function waitFor(sel, cb, tries=0) {
        const el = document.querySelector(sel);
        if (el) return cb(el);
        if (tries < 40) setTimeout(() => waitFor(sel, cb, tries+1), 120);
    }
    function waitForAll(sel, cb, tries=0) {
        const els = document.querySelectorAll(sel);
        if (els.length) return cb(els);
        if (tries < 40) setTimeout(() => waitForAll(sel, cb, tries+1), 120);
    }

    /* ══════════════════════
       1. CURSOR GLOW
    ══════════════════════ */
    const glow = document.getElementById('cursor-glow');
    if (glow) {
        let mx = -300, my = -300;
        document.addEventListener('mousemove', (e) => {
            mx = e.clientX; my = e.clientY;
            glow.style.left = mx + 'px';
            glow.style.top  = my + 'px';
        });
    }

    /* ══════════════════════
       2. FLOATING PARTICLES
    ══════════════════════ */
    const pContainer = document.getElementById('particlesContainer');
    if (pContainer) {
        function spawnParticle() {
            const p = document.createElement('div');
            p.className = 'particle';
            const x   = Math.random() * 100;
            const dur  = Math.random() * 12 + 8;
            const size = Math.random() * 3 + 1;
            const drift= (Math.random() - 0.5) * 120;
            p.style.cssText = `
                left: ${x}%;
                width: ${size}px; height: ${size}px;
                --drift: ${drift}px;
                animation: floatParticle ${dur}s linear forwards;
                opacity: 0;
            `;
            pContainer.appendChild(p);
            setTimeout(() => p.remove(), dur * 1000);
        }
        setInterval(spawnParticle, 900);
        for (let i = 0; i < 6; i++) setTimeout(spawnParticle, i * 400);
    }

    /* ══════════════════════
       3. HERO STARS
    ══════════════════════ */
    waitFor('#heroStars', function(container) {
        const count = 32;
        for (let i = 0; i < count; i++) {
            const star = document.createElement('div');
            star.className = 'star';
            const x = Math.random() * 100;
            const y = Math.random() * 100;
            const size = Math.random() * 2.5 + 0.5;
            const delay = Math.random() * 4;
            const duration = Math.random() * 3 + 2;
            star.style.cssText = `
                left: ${x}%; top: ${y}%;
                width: ${size}px; height: ${size}px;
                animation: starTwinkle ${duration}s ${delay}s ease-in-out infinite;
                opacity: ${Math.random() * 0.6 + 0.1};
            `;
            container.appendChild(star);
        }
    });

    /* ══════════════════════
       4. BUTTON RIPPLE
    ══════════════════════ */
    function initButtonRipple() {
        document.querySelectorAll('.stButton > button').forEach(btn => {
            if (btn._rippleAdded) return;
            btn._rippleAdded = true;
            btn.addEventListener('click', function(e) {
                const rect = btn.getBoundingClientRect();
                const ripple = document.createElement('span');
                ripple.className = 'ripple-effect';
                ripple.style.left = (e.clientX - rect.left) + 'px';
                ripple.style.top  = (e.clientY - rect.top)  + 'px';
                btn.appendChild(ripple);
                setTimeout(() => ripple.remove(), 900);
            });
        });
    }
    setTimeout(initButtonRipple, 600);
    setTimeout(initButtonRipple, 2000);

    /* ══════════════════════
       5. SCROLL REVEAL
    ══════════════════════ */
    function initScrollReveal() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, idx) => {
                if (entry.isIntersecting) {
                    setTimeout(() => entry.target.classList.add('revealed'), idx * 80);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
        document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));
    }
    setTimeout(initScrollReveal, 500);
    setTimeout(initScrollReveal, 2000);
    setTimeout(initScrollReveal, 4000);

    /* ══════════════════════
       6. PARALLAX HERO
    ══════════════════════ */
    waitFor('.hero', function(hero) {
        document.addEventListener('mousemove', (e) => {
            const cx = window.innerWidth / 2;
            const cy = window.innerHeight / 2;
            const dx = (e.clientX - cx) / cx;
            const dy = (e.clientY - cy) / cy;
            const logo = hero.querySelector('.hero-logo');
            const icon = hero.querySelector('.hero-icon-wrap');
            if (logo) logo.style.transform = `translate(${dx * 6}px, ${dy * 3}px)`;
            if (icon) icon.style.transform = `translate(${dx * 4}px, ${dy * 4}px)`;
        });
    });

    /* ══════════════════════
       7. LOGO SHIMMER on hover
    ══════════════════════ */
    waitFor('.hero-logo', function(logo) {
        logo.addEventListener('mouseenter', () => {
            logo.style.transition = 'filter 0.3s';
            logo.style.filter = 'drop-shadow(0 0 30px rgba(201,168,76,0.6))';
        });
        logo.addEventListener('mouseleave', () => {
            logo.style.filter = '';
        });
    });

    /* ══════════════════════
       8. CARD AMBIENT GLOW
    ══════════════════════ */
    function initCardGlow() {
        document.querySelectorAll('.rec-card').forEach(card => {
            if (card._glowAdded) return;
            card._glowAdded = true;
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * 100;
                const y = ((e.clientY - rect.top)  / rect.height) * 100;
                card.style.setProperty('--mouse-x', x + '%');
                card.style.setProperty('--mouse-y', y + '%');
                card.style.backgroundImage = `radial-gradient(circle at ${x}% ${y}%, rgba(201,168,76,0.06) 0%, transparent 60%)`;
            });
            card.addEventListener('mouseleave', () => {
                card.style.backgroundImage = '';
            });
            /* Tilt on hover */
            card.addEventListener('mousemove', (e) => {
                const rect  = card.getBoundingClientRect();
                const relX  = (e.clientX - rect.left) / rect.width  - 0.5;
                const relY  = (e.clientY - rect.top)  / rect.height - 0.5;
                card.style.transform = `translateY(-14px) scale(1.02) rotateX(${-relY * 6}deg) rotateY(${relX * 6}deg)`;
            });
            card.addEventListener('mouseleave', () => {
                card.style.transform = '';
            });
        });
    }
    setTimeout(initCardGlow, 1500);
    setTimeout(initCardGlow, 4000);

    /* ══════════════════════
       9. STATS COUNT-UP
    ══════════════════════ */
    function animateCountUp(el) {
        const raw = el.textContent.replace(/,/g, '');
        if (isNaN(raw)) return;
        const target = parseInt(raw);
        const duration = 1200;
        const steps = 50;
        const inc = target / steps;
        let current = 0;
        const timer = setInterval(() => {
            current += inc;
            if (current >= target) { current = target; clearInterval(timer); }
            el.textContent = Math.floor(current).toLocaleString();
        }, duration / steps);
    }
    waitForAll('.stat-num', function(nums) {
        const obs = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCountUp(entry.target);
                    obs.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });
        nums.forEach(n => obs.observe(n));
    });

    /* ══════════════════════
       10. FILM STRIP PAUSE
    ══════════════════════ */
    document.querySelectorAll('.strip-inner').forEach(strip => {
        strip.addEventListener('mouseenter', () => strip.style.animationPlayState = 'paused');
        strip.addEventListener('mouseleave', () => strip.style.animationPlayState = 'running');
    });

    /* ══════════════════════
       11. TOAST on first load
    ══════════════════════ */
    if (!sessionStorage.getItem('ll_greeted')) {
        sessionStorage.setItem('ll_greeted', '1');
        setTimeout(() => {
            const toast = document.createElement('div');
            toast.className = 'll-toast';
            toast.innerHTML = '<span class="ll-toast-icon">✦</span> Welcome to LensLore — discover your next film';
            document.body.appendChild(toast);
            setTimeout(() => {
                toast.style.transition = 'opacity 0.6s, transform 0.6s';
                toast.style.opacity = '0';
                toast.style.transform = 'translateY(20px)';
                setTimeout(() => toast.remove(), 700);
            }, 3500);
        }, 1200);
    }

    /* ══════════════════════
       12. CARD ENTRANCE — re-trigger on results load
    ══════════════════════ */
    function triggerEntrances() {
        document.querySelectorAll('.rec-card').forEach(card => {
            card.style.opacity = '';
        });
    }
    setTimeout(triggerEntrances, 2000);
    setTimeout(triggerEntrances, 5000);

    /* ══════════════════════
       13. SELECTION-CHANGE FLASH on industry badge
    ══════════════════════ */
    waitForAll('div[data-baseweb="select"]', function(selects) {
        selects.forEach(sel => {
            sel.addEventListener('mousedown', () => {
                const badge = document.querySelector('.industry-badge');
                if (badge) {
                    badge.style.transition = 'box-shadow 0.3s';
                    badge.style.boxShadow = '0 0 0 4px rgba(201,168,76,0.25)';
                    setTimeout(() => badge.style.boxShadow = '', 600);
                }
            });
        });
    });

})();
</script>
""", unsafe_allow_html=True)
