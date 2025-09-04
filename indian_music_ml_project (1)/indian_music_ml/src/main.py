"""
CLI to run recommendations or (optionally) fetch Spotify data.
"""
import argparse
import pandas as pd
from .data_processing import load_dataset
from .recommender import ContentRecommender

def run_recommender(args):
    df = load_dataset(args.dataset)
    rec = ContentRecommender(df)

    if args.random and args.random > 0:
        outs = rec.recommend_random(n=args.random, top_k=args.top_k)
        for k, out in enumerate(outs, start=1):
            print(f"\n=== Random seed {k} ===")
            seed_id = out.iloc[0]['id']
            seed_row = df[df['id'] == seed_id].iloc[0]
            print(f"Seed: {seed_row['id']} | {seed_row['name']} — {seed_row['artist']}")
            print(out.to_string(index=False, formatters={'similarity': '{:.3f}'.format}))
        return

    track_id = args.track_id or df.iloc[0]['id']
    seed_row = df[df['id'] == track_id].iloc[0]
    print(f"Query: {seed_row['id']} | {seed_row['name']} — {seed_row['artist']}")
    out = rec.recommend_by_id(track_id, top_k=args.top_k)
    print(out.to_string(index=False, formatters={'similarity': '{:.3f}'.format}))

def main():
    parser = argparse.ArgumentParser(description="Indian Music ML - Content Recommender")
    subparsers = parser.add_subparsers(dest="command")

    # default: recommend
    parser_reco = subparsers.add_parser("recommend", help="Run content-based recommendations")
    parser_reco.add_argument("--dataset", default="data/indian_songs.csv", help="Path to CSV dataset")
    parser_reco.add_argument("--track-id", default=None, help="Seed track id present in the dataset")
    parser_reco.add_argument("--top-k", type=int, default=5, help="Number of similar tracks to return")
    parser_reco.add_argument("--random", type=int, default=0, help="Recommend for N random seed tracks")
    parser_reco.set_defaults(func=run_recommender)

    # fetch command (optional; requires Spotify credentials)
    parser_fetch = subparsers.add_parser("fetch", help="Fetch playlist features from Spotify")
    parser_fetch.add_argument("--playlist-id", required=True, help="Spotify playlist id")
    parser_fetch.add_argument("--market", default="IN", help="Market code (e.g., IN)")

    args = parser.parse_args()

    if args.command == "fetch":
        try:
            from .spotify_client import fetch_playlist_tracks_df, fetch_audio_features
        except Exception as e:
            raise RuntimeError("Spotipy/Spotify not available. Install spotipy and set env vars.") from e

        df_meta = fetch_playlist_tracks_df(args.playlist_id, market=args.market)
        df_full = fetch_audio_features(df_meta)
        df_full.to_csv("data/indian_songs.csv", index=False)
        print("Saved: data/indian_songs.csv")
    else:
        # default to recommend
        if not args.command:
            # mimic "recommend"
            args.command = "recommend"
            args.dataset = "data/indian_songs.csv"
            args.top_k = getattr(args, "top_k", 5)
            args.random = getattr(args, "random", 0)
            args.track_id = getattr(args, "track_id", None)
        args.func(args)

if __name__ == "__main__":
    main()
