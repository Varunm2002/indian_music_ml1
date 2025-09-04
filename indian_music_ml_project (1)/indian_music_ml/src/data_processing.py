"""
Data loading and preprocessing utilities.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd

# Default feature set for recommendation
FEATURES = [
    "danceability","energy","valence","acousticness",
    "instrumentalness","liveness","speechiness","tempo","loudness","popularity"
]

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure required columns exist
    missing = [c for c in ["id","name","artist"] + FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    # Drop rows with any missing feature values
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df

def build_feature_matrix(df: pd.DataFrame, features: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    feats = features or FEATURES
    X = df[feats].astype(float).to_numpy()
    # Standardize (z-score)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xz = (X - mu) / sigma
    return Xz, feats
