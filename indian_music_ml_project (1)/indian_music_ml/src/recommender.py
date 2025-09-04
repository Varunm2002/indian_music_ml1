"""
Simple content-based recommender using cosine similarity (NumPy only).
"""
from typing import List
import numpy as np
import pandas as pd
from .data_processing import build_feature_matrix, FEATURES

class ContentRecommender:
    def __init__(self, df: pd.DataFrame, features: List[str] = None):
        self.df = df.reset_index(drop=True)
        self.features = features or FEATURES
        self.X, _ = build_feature_matrix(self.df, self.features)
        # Precompute L2 norms for cosine similarity
        self.norms = np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-8

    def _cosine_sim(self, i: int) -> np.ndarray:
        # cosine similarity of row i vs all rows
        v = self.X[i:i+1]
        vnorm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        sims = (self.X @ v.T) / (self.norms * vnorm)
        return sims.ravel()

    def recommend_by_id(self, track_id: str, top_k: int = 5) -> pd.DataFrame:
        idxs = self.df.index[self.df["id"] == track_id].tolist()
        if not idxs:
            raise ValueError(f"Track id '{track_id}' not found in dataset.")
        i = idxs[0]
        sims = self._cosine_sim(i)
        # Exclude the same track, get top_k
        order = np.argsort(-sims)
        order = [j for j in order if j != i][:top_k]
        out = self.df.loc[order, ["id","name","artist"]].copy()
        out["similarity"] = sims[order]
        return out.reset_index(drop=True)

    def recommend_random(self, n: int = 3, top_k: int = 5) -> List[pd.DataFrame]:
        rng = np.random.default_rng(42)
        choices = rng.choice(len(self.df), size=min(n, len(self.df)), replace=False)
        outs = []
        for i in choices:
            track_id = self.df.loc[i, "id"]
            outs.append(self.recommend_by_id(track_id, top_k=top_k))
        return outs
