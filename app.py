import streamlit as st
import pickle
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# --- Spotify API Setup ---
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# --- Load data ---
with open("df_cleaned.pkl", "rb") as f:
    df_cleaned = pickle.load(f)

with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

# Correct loading of NumPy array
feature_matrix = np.load("feature_matrix.npy")


# --- Helper Functions ---
def get_spotify_song_link(song_name, artist_name=""):
    query = f"track:{song_name}"
    if artist_name:
        query += f" artist:{artist_name}"
    try:
        result = sp.search(q=query, type='track', limit=1)
        tracks = result.get('tracks', {}).get('items', [])
        return tracks[0]['external_urls']['spotify'] if tracks else None
    except Exception as e:
        print(f"Spotify API error: {e}")
        return None

def get_spotify_image(song_name, artist_name=""):
    query = f"track:{song_name}"
    if artist_name:
        query += f" artist:{artist_name}"
    try:
        result = sp.search(q=query, type='track', limit=1)
        tracks = result.get('tracks', {}).get('items', [])
        return tracks[0]['album']['images'][0]['url'] if tracks else None
    except Exception as e:
        print(f"Spotify API error: {e}")
        return None

def recommend_song_knn(song_name, artist_name, num_recommendations_per_category=2, similarity_threshold=0.1):
    input_track_artist = f"{song_name} by {artist_name}"
    matching_rows = df_cleaned[df_cleaned["track_artist"].str.lower().str.strip() == input_track_artist.lower().strip()]
    if matching_rows.empty:
        return f"Song '{song_name}' by '{artist_name}' not found in dataset!"

    matched_row = matching_rows.iloc[0]
    song_idx = df_cleaned.index.get_loc(matched_row.name)
    input_genre = matched_row['genre']
    input_artist = matched_row['artist']

    distances, indices = knn.kneighbors([feature_matrix[song_idx]], n_neighbors=50)
    candidates = df_cleaned.iloc[indices[0][1:]].copy()
    candidates['adjusted_distance'] = distances[0][1:len(candidates) + 1]

    same_artist_candidates = candidates[candidates['artist'] == input_artist].copy()
    if not same_artist_candidates.empty:
        similar_artist_recs = same_artist_candidates[same_artist_candidates['adjusted_distance'] <= similarity_threshold].head(num_recommendations_per_category)
        if len(similar_artist_recs) < num_recommendations_per_category:
            best_artist_recs = same_artist_candidates.sort_values('adjusted_distance').head(num_recommendations_per_category)
            same_artist_recs = best_artist_recs[['song', 'track_artist']].values.tolist()
        else:
            same_artist_recs = similar_artist_recs[['song', 'track_artist']].values.tolist()
    else:
        same_artist_recs = f"⚠️ We couldn't find more songs by **{input_artist}** to give recommendations."


    same_genre_candidates = candidates[(candidates['genre'] == input_genre) & (candidates['artist'] != input_artist)].copy()
    same_genre_recs = same_genre_candidates.sort_values('adjusted_distance').head(
        num_recommendations_per_category)[['song', 'track_artist']].values.tolist()

    used_songs = set()

    if isinstance(same_artist_recs, list):
        used_songs.update([song for song, _ in same_artist_recs])
    if isinstance(same_genre_recs, list):
        used_songs.update([song for song, _ in same_genre_recs])

    feature_based_candidates = candidates[~candidates['song'].isin(used_songs)].copy()
    feature_based_recs = feature_based_candidates.sort_values('adjusted_distance').head(
        num_recommendations_per_category)[['song', 'track_artist']].values.tolist()

    recommendations = {
        "Input Song": [(song_name, input_track_artist)],
        "Same Artist": same_artist_recs,
        "Same Genre": same_genre_recs,
        "Other": feature_based_recs
    }

    return recommendations

# --- Streamlit App ---
st.set_page_config(page_title="🎵 Smart Song Recommender")
st.title("🎶 Music Recommendation Engine")

track_list = sorted(df_cleaned['track_artist'].unique())
selected_track = st.selectbox("Choose a song:", track_list)

if st.button("🔍 Recommend"):
    song, artist = selected_track.split(" by ")
    recs = recommend_song_knn(song, artist)

    if isinstance(recs, str):
        st.error(recs)
    else:
        st.subheader("Input Song")
        img = get_spotify_image(song, artist)
        link = get_spotify_song_link(song, artist)
        st.markdown(f"<h4>{song} by {artist}</h4>", unsafe_allow_html=True)
        if img:
            st.image(img, width=150)
        if link:
            st.markdown(f"[▶️ Listen on Spotify]({link})", unsafe_allow_html=True)

        for category in ["Same Artist", "Same Genre", "Other"]:
            st.subheader(f"🔹 {category} Recommendations")
            if isinstance(recs[category], str):
                st.write(recs[category])
            else:
                for song, track in recs[category]:
                    a = track.split(" by ")[-1]
                    img = get_spotify_image(song, a)
                    link = get_spotify_song_link(song, a)
                    cols = st.columns([1, 0.2, 5])
                    with cols[0]:
                        if img:
                            st.image(img, width=130)
                    with cols[2]:
                        st.markdown(f"<p style='font-size:25px'><b>{song} by {a}</b></p>", unsafe_allow_html=True)
                        if link:
                            st.markdown(f"[▶️ Listen]({link})", unsafe_allow_html=True)
