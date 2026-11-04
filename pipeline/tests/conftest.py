import numpy as np
import pandas as pd
import pytest

SMALL_PARAMS = {
    "data": {"hit_threshold": 60, "max_duration_ms": 900000, "min_tempo": 30},
    "split": {"test_size": 0.2, "calib_size": 0.25, "seed": 42},
    "features": {"min_genre_count": 20, "artist_mask_rate": 0.5},
    "xgb": {"learning_rate": 0.2},
    "tune": {"grid": {"max_depth": [2, 3], "n_estimators": [20]}, "folds": 2},
    "decision": {"fn_cost": 3, "fp_cost": 1},
    "evaluate": {
        "cv_folds": 3,
        "niche_genres": ["niche", "ghost"],
        "ndcg_k": 20,
        "batch_size": 100,
        "n_batches": 10,
        "gate_tolerance": 0.005,
        "brier_tolerance": 0.002,
    },
}


@pytest.fixture
def params():
    return SMALL_PARAMS


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """~600 linhas sintéticas onde danceability/loudness puxam a popularidade."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "track_id": [f"id{i}" for i in range(n)],
            "name": [f"song {i}" for i in range(n)],
            "artist": [f"artist {i % 40}" for i in range(n)],
            "danceability": rng.random(n),
            "energy": rng.random(n),
            "valence": rng.random(n),
            "speechiness": rng.random(n) * 0.5,
            "acousticness": rng.random(n),
            "instrumentalness": rng.random(n) * 0.3,
            "tempo": rng.uniform(60, 200, n),
            "loudness": rng.uniform(-30, 0, n),
            "duration_ms": rng.integers(120_000, 400_000, n),
            "liveness": rng.random(n),
            "key": rng.integers(0, 12, n),
            "mode": rng.integers(0, 2, n),
            "time_signature": rng.choice([3, 4, 5], n),
            "explicit": rng.random(n) < 0.2,
            "track_genre": rng.choice(["pop", "rock", "jazz"], n),
        }
    )
    df.loc[:9, "track_genre"] = "niche"
    score = 3 * df["danceability"] + 0.05 * df["loudness"] + rng.normal(0, 0.4, n)
    df["popularity"] = (100 * (score - score.min()) / (score.max() - score.min())).astype(int)
    return df


@pytest.fixture
def clean_df(raw_df, params) -> pd.DataFrame:
    """Dataset pronto p/ treino: limpo + alvo (percentil no gênero) + is_hit."""
    from resonance_pipeline.preprocess import clean
    from resonance_pipeline.target import add_targets

    d = params["data"]
    return add_targets(clean(raw_df, d["max_duration_ms"], d["min_tempo"]), d["hit_threshold"])
