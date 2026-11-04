import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from resonance_features import ARTIST, MODEL_COLUMNS, engineer, expected_popularity, factor_groups, fit_stats
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from app.core.config import Settings
from app.main import create_app

FEATURES = [
    "danceability", "energy", "valence", "tempo", "loudness",
    "speechiness", "acousticness", "instrumentalness", "duration_ms",
]  # fmt: skip
GENRE = "track_genre"

SAMPLE = {
    "danceability": 0.8, "energy": 0.7, "valence": 0.5, "tempo": 120.0, "loudness": -5.0,
    "speechiness": 0.05, "acousticness": 0.1, "instrumentalness": 0.0, "duration_ms": 200000,
    "track_genre": "pop",
}  # fmt: skip
WITH_ARTIST = {**SAMPLE, "artist_n_tracks": 12, "artist_mean_popularity": 75.0, "artist_hit_share": 0.6}


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "track_id": [f"id{i}" for i in range(n)],
            "name": [f"Song {i}" for i in range(n)],
            "artist": [f"Artist {i % 20}" for i in range(n)],
            "danceability": rng.random(n), "energy": rng.random(n), "valence": rng.random(n),
            "tempo": rng.uniform(60, 200, n), "loudness": rng.uniform(-30, 0, n),
            "speechiness": rng.random(n) * 0.5, "acousticness": rng.random(n),
            "instrumentalness": rng.random(n) * 0.3, "duration_ms": rng.integers(120_000, 400_000, n),
            "track_genre": rng.choice(["pop", "rock", "jazz"], n),
        }
    )  # fmt: skip
    artist_effect = rng.normal(0, 1, 20)  # a fama do artista pesa na popularidade (como nos dados reais)
    score = 3 * df["danceability"] + 2.5 * artist_effect[np.arange(n) % 20] + rng.normal(0, 0.5, n)
    df["popularity"] = (100 * (score - score.min()) / (score.max() - score.min())).astype(int)
    df["popularity_percentile_within_genre"] = df.groupby("track_genre")["popularity"].rank(pct=True) * 100
    df["is_hit"] = (df["popularity"] >= 60).astype(int)
    for c, v in (("liveness", 0.1), ("key", 5), ("mode", 1), ("time_signature", 4), ("explicit", False)):
        df[c] = v
    g = df.groupby("artist")["popularity"]
    k = (g.transform("count") - 1).replace(0, np.nan)
    df["artist_n_tracks"] = k
    df["artist_mean_popularity"] = (g.transform("sum") - df["popularity"]) / k
    df["artist_hit_share"] = (df.groupby("artist")["is_hit"].transform("sum") - df["is_hit"]) / k
    return df


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory, dataset) -> str:
    """Bundle mínimo com o mesmo formato que pipeline/train.py produz (usa o mesmo pacote de features)."""
    stats = fit_stats(dataset)
    pre = ColumnTransformer(
        [
            ("num", "passthrough", MODEL_COLUMNS),
            ("genre", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [GENRE]),
        ],
        sparse_threshold=0,
    )
    rng = np.random.default_rng(1)
    train_e = engineer(dataset, stats)
    masked = rng.random(len(train_e)) < 0.5  # metade sem histórico, como no treino real
    train_e.loc[masked, [*ARTIST, "artist_pct_in_genre"]] = np.nan
    X = pre.fit_transform(train_e)
    model = XGBRegressor(n_estimators=30, max_depth=3, random_state=0).fit(
        X, dataset["popularity_percentile_within_genre"]
    )
    calibrators, thresholds = {}, {}
    for mode in ("artist_aware", "audio_only"):
        e = engineer(dataset, stats, use_artist_context=mode == "artist_aware")
        score = np.clip(model.predict(pre.transform(e)), 0, 100)
        ep = expected_popularity(score, dataset[GENRE].to_numpy(), stats)
        calibrators[mode] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(ep, dataset["is_hit"])
        thresholds[mode] = 0.3
    names = list(pre.get_feature_names_out())
    bundle = {
        "version": "test-v1", "preprocessor": pre, "model": model, "stats": stats,
        "calibrators": calibrators, "thresholds": thresholds,
        "features": FEATURES, "genre_col": GENRE, "feature_names": names, "factor_groups": factor_groups(names),
        "known_genres": ["jazz", "pop", "rock"],
        "metrics": {"spearman": 0.5, "ndcg_at_k": 0.9, "brier": 0.05, "ece": 0.01, "reliability": []},
        "baseline": {
            "ridge": {"spearman": 0.4, "ndcg_at_k": 0.8},
            "artist_mean": {"spearman": 0.45, "ndcg_at_k": 0.85},
        },
        "hit_means": dataset[dataset["is_hit"] == 1][FEATURES].mean().to_dict(),
        "dataset_md5": "abc",
    }  # fmt: skip
    d = tmp_path_factory.mktemp("model")
    joblib.dump(bundle, d / "bundle.joblib")
    return str(d)


@pytest.fixture
def settings(bundle_dir, dataset, tmp_path) -> Settings:
    csv = tmp_path / "tracks.csv"
    dataset.to_csv(csv, index=False)
    return Settings(
        model_registry_uri=bundle_dir,
        data_path=str(csv),
        prediction_log_dir=str(tmp_path / "logs"),
        max_batch_size=5,
        rate_limit_per_minute=0,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def degraded_client(settings):
    settings.model_registry_uri = "/nonexistent"
    settings.data_path = "/nonexistent.csv"
    with TestClient(create_app(settings)) as c:
        yield c
