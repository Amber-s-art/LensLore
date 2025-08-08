import streamlit as st
from streamlit_option_menu import option_menu  # optional, for future menu
import base64
import pandas as pd
import ast


# Set page config
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# Custom CSS for styling (Netflix vibes)
st.markdown("""
    <style>
    /* Background and font */
    html, body, [class*="css"] {
        background-color: #141414;
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Title */
    .title {
        font-size: 4em;
        font-weight: bold;
        color: #e50914;
        text-align: center;
        margin-bottom: 0.2em;
        text-shadow: 2px 2px 8px rgba(229, 9, 20, 0.5);
        letter-spacing: 2px;
    }

    .subtitle {
        font-size: 1.3em;
        text-align: center;
        margin-bottom: 2em;
        color: #b3b3b3;
    }

    /* Buttons */
    .stButton > button {
        background-color: #e50914;
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        font-size: 1em;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #b00610;
    }

    /* Selectbox */
    label[data-testid="stWidgetLabel"] {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.9em;
    }

    div[data-baseweb="select"] > div {
        background-color: #1f1f1f;
        border-radius: 6px;
        padding: 4px 10px;
        border: 1px solid #333;
        font-size: 0.85em;
    }

    /* Selectbox options hover (internal dropdowns) */
    div[data-baseweb="popover"] div[role="option"] {
        font-size: 0.85em;
        padding: 8px 12px;
    }

    /* Image captions */
    .element-container h1, .element-container h2, .element-container h3, .element-container h4 {
        color: #e5e5e5;
        text-align: center;
    }

    /* Markdown movie titles under posters */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-size: 1em;
        text-align: center;
        color: #f5f5f5;
        margin-top: 10px;
    }

    /* Reduce top spacing for movie cards */
    .stImage {
        margin-top: -10px;
    }
    </style>
""", unsafe_allow_html=True)


# --- App Header ---
st.markdown('<div class="title">CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your personalized movie matcher – powered by vibes, genre & cast</div>', unsafe_allow_html=True)
st.markdown("---")

# Load data
movie_boll = pd.read_csv('dataset/cleaned/bollywood_cleaned.csv')
movie_holl = pd.read_csv('dataset/cleaned/hollywood_cleaned.csv')

# Select Industry
industry = st.selectbox('Select a movie industry', ('Bollywood', 'Hollywood'))
selected_df = movie_boll if industry == 'Bollywood' else movie_holl

# Helper to safely parse list-like strings
def parse_list(value):
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value) if isinstance(value, str) and value.startswith('[') else value.split(',')
    except:
        return []

# Filter by Genre
selected_df['genres_parsed'] = selected_df['genres'].apply(parse_list)
all_genres = sorted(set(g.strip().lower() for genres in selected_df['genres_parsed'] for g in genres if isinstance(g, str)))

col1, col2 = st.columns([1, 1])
with col2:
    genre = st.selectbox('Select a genre', ['All'] + all_genres, key='genre_select')

if genre != 'All':
    selected_df = selected_df[selected_df['genres_parsed'].apply(lambda x: genre.lower() in [g.lower() for g in x])]

# Filter by Actor

selected_df['cast_parsed'] = selected_df['cast'].apply(parse_list)
all_actors = sorted(set(a.strip() for cast in selected_df['cast_parsed'] for a in cast if isinstance(a, str)))

with col1:
    actor = st.selectbox('Select an actor', ['All'] + all_actors, key='actor_select')

if actor != 'All':
    selected_df = selected_df[selected_df['cast_parsed'].apply(lambda x: actor.lower() in [a.lower() for a in x])]

selected_df = selected_df.reset_index(drop=True)  

# --- Movie Selectbox ---
movie_list = selected_df['title'].tolist()
selected_movie = st.selectbox("Search your favorite movie", sorted(movie_list), key="movie_select")

import requests

@st.cache_data(show_spinner=False)
def fetch_movie_poster(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmMzhjMjk5NDU1ZDBiNTJjY2EyZTFlNzBjNjliNjJiMCIsIm5iZiI6MTc1Mzk1MTA4Ni44MTksInN1YiI6IjY4OGIyYjZlYTBhYTJlM2RkZDc2N2Q5MyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.o2aZuhksRxjxvrD1UajVcNtdFDBQA5bPyj9CagvmHOA"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    poster_path = data.get('poster_path')
    if poster_path:
        full_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        return full_poster_url
    else:
        return None


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def recommend(movie_name_input):

    recommend_movies = []
    recommend_movies_poster = []

    # Clean tags column: convert list of words to string
    selected_df['tags'] = selected_df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

    # Warn if not enough data
    if selected_df.shape[0] < 6:
        st.warning("Not enough similar movies to generate recommendations. Showing all matching movies instead.")
        
        for idx, row in selected_df.iterrows():
            recommend_movies.append(row['title'])
            poster_url = fetch_movie_poster(row['movie_id'])
            recommend_movies_poster.append(poster_url)

        return recommend_movies, recommend_movies_poster
    
    # Recalculate vectors and fit model again (or cache these if performance matters)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2)
    vectors = tfidf.fit_transform(selected_df['tags']).toarray()

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(vectors)


    # Find index of selected movie
    movie_index = selected_df[selected_df['title'] == movie_name_input].index[0]

    # Get top 5 similar movie indices (excluding itself)
    distances, indices = model.kneighbors([vectors[movie_index]], n_neighbors=6)


    for i in indices[0][1:]:  # Skip input movie itself
        movie = selected_df.iloc[i]
        recommend_movies.append(movie['title'])

        # Make sure 'movie_id' exists and fetch function works
        movie_id = movie['movie_id'] if 'movie_id' in movie else None
        poster_url = fetch_movie_poster(movie_id) if movie_id else "https://via.placeholder.com/300x450?text=No+Image"
        recommend_movies_poster.append(poster_url)

    return recommend_movies, recommend_movies_poster


# Recommend button
center = st.columns([2, 1, 2])
with center[1]:
    if st.button("Recommend"):
        st.session_state.recommend_triggered = True

if st.session_state.get("recommend_triggered"):
    recommendations, poster_urls = recommend(selected_movie)

    if recommendations:
        st.subheader("Top Recommendations:")

        # First row – 2 movies
        row1 = st.columns(2)
        for i in range(min(2, len(recommendations))):
            with row1[i]:
                if poster_urls[i]:
                    st.image(poster_urls[i], use_container_width=True)
                st.markdown(f"**{recommendations[i]}**")

        # Second row – next 2 movies
        if len(recommendations) > 2:
            row2 = st.columns(2)
            for j in range(2, min(4, len(recommendations))):
                with row2[j - 2]:
                    if poster_urls[j]:
                        st.image(poster_urls[j], use_container_width=True)
                    st.markdown(f"**{recommendations[j]}**")

        # Third row – last 1 movie centered
        if len(recommendations) > 4:
            # Create 3 columns: empty - center - empty
            row3 = st.columns([1, 2, 1])
            with row3[1]:  # center column
                if poster_urls[4]:
                    st.image(poster_urls[4], use_container_width=True)
                st.markdown(f"**{recommendations[4]}**")

    else:
        st.error("No recommendations found.")



