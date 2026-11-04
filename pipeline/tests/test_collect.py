from unittest.mock import MagicMock

from spotipy.exceptions import SpotifyException

from resonance_pipeline import collect


def _track(i):
    return {
        "track": {
            "id": f"t{i}",
            "name": f"n{i}",
            "artists": [{"name": "a"}],
            "popularity": 50,
            "album": {"release_date": "2020-01-01"},
        }
    }


def test_get_playlist_tracks_merges_audio_features():
    sp = MagicMock()
    sp.playlist_tracks.return_value = {"items": [_track(0), _track(1)], "next": None}
    sp.audio_features.return_value = [{k: 0.5 for k in collect.AUDIO_FEATURES}, None]  # 2ª sem features
    out = collect.get_playlist_tracks(sp, "pl")
    assert [t["id"] for t in out] == ["t0"] and out[0]["year"] == 2020


def test_collect_multiple_playlists_dedups(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "get_spotify_client", lambda: MagicMock())
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)
    row = {"id": "t0", "name": "n", **{k: 0.5 for k in collect.AUDIO_FEATURES}}
    monkeypatch.setattr(collect, "get_playlist_tracks", lambda sp, pid: [row])
    df = collect.collect_multiple_playlists(["a", "b"], output_path=str(tmp_path / "o.csv"))
    assert len(df) == 1 and (tmp_path / "o.csv").exists()


def test_retry_on_rate_limit_then_skip_on_404(monkeypatch):
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise SpotifyException(429, -1, "slow down")
        return "ok"

    def gone():
        raise SpotifyException(404, -1, "nope")

    assert collect._fetch_with_retry(flaky) == "ok"
    assert collect._fetch_with_retry(gone) is None
