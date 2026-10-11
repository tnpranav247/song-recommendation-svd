"""Core content-based song recommendation logic using SVD and cosine similarity."""
from pathlib import Path

import numpy as np
import pandas as pd

TITLE_COLUMN = "track_name"
ARTIST_COLUMN = "artists"
GENRE_COLUMN = "track_genre"
FEATURE_CANDIDATES = [
    "danceability", "energy", "acousticness", "valence", "tempo",
    "speechiness", "instrumentalness", "liveness", "loudness",
]


def load_dataset(path):
    """Load a Spotify-style CSV and clean the columns used by the app."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path, low_memory=False)
    required = [TITLE_COLUMN, ARTIST_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    features = [column for column in FEATURE_CANDIDATES if column in df.columns]
    if not features:
        raise ValueError("No supported numerical audio-feature columns were found.")

    df = df.dropna(subset=required).copy()
    # The source can list the same Spotify track under multiple genres. Keep
    # one row per track ID so a song does not recommend itself as a duplicate.
    if "track_id" in df.columns:
        df["track_id"] = df["track_id"].astype(str)
        df = df.drop_duplicates(subset=["track_id"], keep="first")
    df[TITLE_COLUMN] = df[TITLE_COLUMN].astype(str).str.strip()
    df[ARTIST_COLUMN] = df[ARTIST_COLUMN].astype(str).str.strip()
    df = df[(df[TITLE_COLUMN] != "") & (df[ARTIST_COLUMN] != "")].copy()

    for column in features:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        median = df[column].median()
        df[column] = df[column].fillna(0.0 if pd.isna(median) else median)

    # Constant columns carry no information and can be removed safely.
    features = [column for column in features if df[column].nunique(dropna=False) > 1]
    if not features:
        raise ValueError("The selected numerical features have no variation.")

    df = df.reset_index(drop=True)
    df.insert(0, "_row_id", np.arange(len(df), dtype=int))
    return df, features


def standardize_features(df, features):
    """Z-score each feature so units such as BPM do not dominate."""
    matrix = df[features].to_numpy(dtype=np.float64, copy=True)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    return (matrix - means) / stds


def calculate_svd(df, features):
    """Return the compact SVD factors for the standardized feature matrix."""
    standardized = standardize_features(df, features)
    # There are only a few feature columns, so compact SVD is efficient here.
    U, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    return U, singular_values


def song_label(row, include_genre=True):
    """Build a readable label while retaining the row index for duplicate titles."""
    label = f"{row[TITLE_COLUMN]} — {row[ARTIST_COLUMN]}"
    if include_genre and GENRE_COLUMN in row.index and pd.notna(row[GENRE_COLUMN]):
        label += f" · {row[GENRE_COLUMN]}"
    return label


def build_reduced_matrix(U, singular_values, components):
    """Represent each song with X_k = U_k Sigma_k."""
    max_components = min(U.shape[1], len(singular_values))
    if not 1 <= int(components) <= max_components:
        raise ValueError(f"Components must be between 1 and {max_components}.")
    k = int(components)
    return U[:, :k] * singular_values[:k]


def recommend(df, U, singular_values, song_index, components=4, top_n=5):
    """Rank all other tracks by cosine similarity in the reduced SVD space."""
    if not 0 <= int(song_index) < len(df):
        raise IndexError("The selected song index is outside the dataset.")

    reduced = build_reduced_matrix(U, singular_values, components)
    norms = np.linalg.norm(reduced, axis=1)
    norms[norms == 0] = 1.0
    normalized = reduced / norms[:, None]
    scores = normalized @ normalized[int(song_index)]
    selected_index = int(song_index)
    scores[selected_index] = -np.inf

    # Different rows can represent the same song (for example, duplicate
    # title/artist entries or alternate track IDs). Exclude those too, using
    # trimmed, case-insensitive title and artist comparisons.
    selected_title = str(df.iloc[selected_index][TITLE_COLUMN]).strip().casefold()
    selected_artist = str(df.iloc[selected_index][ARTIST_COLUMN]).strip().casefold()
    same_song = (
        df[TITLE_COLUMN].astype(str).str.strip().str.casefold().eq(selected_title)
        & df[ARTIST_COLUMN].astype(str).str.strip().str.casefold().eq(selected_artist)
    ).to_numpy()
    scores[same_song] = -np.inf

    available = int(np.isfinite(scores).sum())
    if available == 0:
        return pd.DataFrame(columns=["Track", "Artist", "Genre", "Similarity", "_row_id"])

    count = min(max(1, int(top_n)), available)
    candidates = np.argpartition(scores, len(scores) - count)[-count:]
    indices = candidates[np.argsort(scores[candidates])[::-1]]

    records = []
    for idx in indices:
        row = df.iloc[int(idx)]
        records.append({
            "Track": row[TITLE_COLUMN],
            "Artist": row[ARTIST_COLUMN],
            "Genre": row[GENRE_COLUMN] if GENRE_COLUMN in df.columns else "—",
            "Similarity": float(scores[idx]),
            "_row_id": int(idx),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    default_path = Path(__file__).with_name("spotify-tracks-dataset-detailed.csv")
    if not default_path.exists():
        default_path = Path(__file__).with_name("songs.csv")
    data, feature_columns = load_dataset(default_path)
    left, singular_values = calculate_svd(data, feature_columns)
    result = recommend(data, left, singular_values, 0, components=min(4, len(singular_values)), top_n=5)
    print(f"Loaded {len(data):,} tracks using {len(feature_columns)} features.")
    print(f"Example recommendations for {song_label(data.iloc[0])}:")
    print(result[["Track", "Artist", "Similarity"]].to_string(index=False))
