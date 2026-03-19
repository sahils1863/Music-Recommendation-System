# Music-Recommendation-System
Built a music recommendation system using KNN and Deezer API that suggests similar songs based on audio features and metadata, with an interactive UI developed in Streamlit.

## Project Structure

├── app.py                 # Main Streamlit application
├── df_cleaned.pkl         # Preprocessed dataset
├── feature_matrix.npy     # Feature vectors for songs
├── knn_model.pkl          # Trained KNN model
├── README.md              # Project documentation

## How It Works
1. User selects a song
2. System finds the song in dataset
3. KNN model computes nearest neighbors
4. Recommendations are categorized into:
    - Same Artist
    - Same Genre
    - Feature-Based
5. Spotify API fetches images and links

## Challenges Faced
- Handling missing or incorrect song matches
- Optimizing recommendation speed
- Integrating Spotify API efficiently
- Avoiding duplicate recommendations

## Solutions
- Used preprocessed dataset (df_cleaned.pkl)
- Stored trained model (knn_model.pkl) for faster execution
- Implemented filtering logic for unique recommendations
- Added error handling for API failures
