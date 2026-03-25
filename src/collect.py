import os
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "loudness", "speechiness", "acousticness",
    "instrumentalness", "duration_ms",
]


def get_spotify_client():
    """Retorna cliente Spotipy autenticado via Client Credentials."""
    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        )
    )


def _extract_year(release_date: str) -> int:
    """Extrai o ano de uma string de data no formato YYYY ou YYYY-MM-DD."""
    return int(release_date[:4])


def _fetch_with_retry(func, *args, max_retries=3, backoff=2):
    """Executa chamada à API com retry automático em caso de rate limit."""
    for attempt in range(max_retries):
        try:
            return func(*args)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait = backoff ** (attempt + 1)
                print(f"Rate limit. Aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise
    return None


def get_playlist_tracks(sp, playlist_id: str) -> list[dict]:
    """Coleta tracks de uma playlist com metadados e features de áudio."""
    tracks = []
    results = _fetch_with_retry(sp.playlist_tracks, playlist_id)
    if not results:
        return tracks

    items = results["items"]
    while results["next"]:
        results = _fetch_with_retry(sp.next, results)
        if results:
            items.extend(results["items"])

    track_ids, track_meta = [], []
    for item in items:
        t = item.get("track")
        if not t or not t.get("id"):
            continue
        track_ids.append(t["id"])
        track_meta.append({
            "id": t["id"],
            "name": t["name"],
            "artist": t["artists"][0]["name"] if t["artists"] else "Unknown",
            "popularity": t["popularity"],
            "year": _extract_year(t["album"]["release_date"]),
        })

    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        features = _fetch_with_retry(sp.audio_features, batch)
        if not features:
            continue
        for j, feat in enumerate(features):
            if feat is None:
                continue
            idx = i + j
            for key in AUDIO_FEATURES:
                track_meta[idx][key] = feat[key]

    return [t for t in track_meta if all(k in t for k in AUDIO_FEATURES)]


def collect_multiple_playlists(
    playlist_ids: list[str],
    output_path: str = "data/raw/tracks.csv",
):
    """Coleta tracks de várias playlists, remove duplicatas e salva CSV."""
    sp = get_spotify_client()
    all_tracks = []

    for pid in playlist_ids:
        print(f"Coletando playlist {pid}...")
        tracks = get_playlist_tracks(sp, pid)
        all_tracks.extend(tracks)
        print(f"  → {len(tracks)} tracks")
        time.sleep(1)

    df = pd.DataFrame(all_tracks)
    df = df.drop_duplicates(subset=["id"])
    df.to_csv(output_path, index=False)
    print(f"\nTotal: {len(df)} tracks únicas salvas em {output_path}")
    return df