# TuneMatch — SVD Song Discovery (Desktop GUI)

TuneMatch is a Python desktop application that recommends tracks with similar numerical audio features. It uses Singular Value Decomposition (SVD) to create reduced song representations and cosine similarity to rank candidate tracks.

## Features
- Tkinter desktop GUI: no browser, localhost URL, or Streamlit server.
- Search tracks by title, artist, or genre.
- Select a track from search results.
- Adjust the number of retained SVD components (`k`).
- Choose how many recommendations to display (5–15).
- View track, artist, genre, and cosine similarity scores.
- Uses nine available numerical audio features: danceability, energy, acousticness, valence, tempo, speechiness, instrumentalness, liveness, and loudness.
- Includes the detailed CSV dataset and a small illustrative fallback dataset.

## Requirements
- Python 3.10+ recommended
- `numpy` and `pandas`
- Tkinter (usually included with standard Python on Windows)

## Run on Windows
1. Extract this ZIP into a folder.
2. Open Command Prompt or PowerShell in the extracted `svd_song_recommender` folder.
3. Install dependencies:

   ```bash
   py -m pip install -r requirements.txt
   ```

4. Launch the desktop app:

   ```bash
   py app.py
   ```

If `py` is not recognized, try `python` instead. A desktop window should open; the app does not start a web server.

The app first looks for `spotify-tracks-dataset-detailed.csv` in the same folder. If it is missing, it falls back to `songs.csv`.

## Mathematics
1. Load and clean the song dataset.
2. Construct a matrix `X` (rows are tracks, columns are numerical audio features).
3. Standardize the features to reduce scale differences.
4. Compute `X = U Σ Vᵀ` using NumPy's SVD implementation.
5. Retain the top `k` components and represent tracks as `U_k Σ_k`.
6. Rank other tracks by cosine similarity in this reduced space.
7. Exclude the selected row and same-title/same-artist duplicate entries from recommendations.

This is content-based song similarity using audio-feature metadata; it does not learn personalized user preferences from ratings.

## Files
- `app.py` — Tkinter desktop GUI
- `engine.py` — dataset cleaning, standardization, SVD, cosine similarity and recommendation functions
- `spotify-tracks-dataset-detailed.csv` — main dataset
- `songs.csv` — small illustrative fallback dataset
- `requirements.txt` — Python dependencies

## Dataset note
Before public redistribution, verify the original dataset's source and license and include the required attribution. The small `songs.csv` sample contains illustrative feature values, not official measured values.
