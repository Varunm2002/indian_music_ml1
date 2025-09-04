"""
Spotify API helper (optional). Only used if you set SPOTIPY_* env vars.
"""
import os
from typing import List, Dict
import pandas as pd

def _has_spotify_creds() -> bool:
    return all([
        os.getenv("SPOTIPY_CLIENT_ID"),
        os.getenv("SPOTIPY_CLIENT_SECRET"),
        os.getenv("SPOTIPY_REDIRECT_URI")
    ])

def get_spotify_client():
    if not _has_spotify_creds():
        raise RuntimeError("Spotify credentials are not set. Export SPOTIPY_* env vars.")
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    auth_manager = SpotifyClientCredentials()
    return spotipy.Spotify(auth_manager=auth_manager)

def fetch_playlist_tracks_df(playlist_id: str, market: str = "IN") -> pd.DataFrame:
    """
    Returns basic track metadata from a public playlist.
    """
    sp = get_spotify_client()
    items = []
    results = sp.playlist_items(playlist_id, additional_types=("track",))
    while results:
        for it in results["items"]:
            t = it.get("track") or {}
            if not t:
                continue
            items.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "artist": ", ".join([a["name"] for a in t.get("artists", [])]),
                "album": (t.get("album") or {}).get("name"),
                "release_date": (t.get("album") or {}).get("release_date"),
                "popularity": t.get("popularity")
            })
        if results.get("next"):
            results = sp.next(results)
        else:
            break
    df_meta = pd.DataFrame(items).dropna(subset=["id"]).drop_duplicates("id")
    return df_meta

def fetch_audio_features(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Spotify audio features into df_meta by 'id'.
    """
    sp = get_spotify_client()
    track_ids = df_meta["id"].dropna().tolist()
    feats_rows = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        feats = sp.audio_features(batch)
        for f in feats:
            if not f:
                continue
            feats_rows.append(f)
    df_feats = pd.DataFrame(feats_rows).dropna(subset=["id"])
    # Keep a consistent subset of features
    keep = [
        "id","danceability","energy","valence","acousticness",
        "instrumentalness","liveness","speechiness","tempo","loudness"
    ]
    df_feats = df_feats[keep]
    return df_meta.merge(df_feats, on="id", how="left")
