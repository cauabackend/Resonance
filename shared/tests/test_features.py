import numpy as np
import pandas as pd
import pytest

from resonance_features import (
    ARTIST,
    AUDIO,
    GENRE,
    MODEL_COLUMNS,
    aggregate_shap,
    engineer,
    expected_popularity,
    factor_groups,
    fit_stats,
    pct_in_genre,
)


@pytest.fixture
def train_df():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({f: rng.random(n) for f in AUDIO})
    df[GENRE] = np.where(np.arange(n) % 2 == 0, "pop", "metal")
    df["popularity"] = np.where(df[GENRE] == "pop", rng.integers(30, 100, n), rng.integers(0, 40, n))
    return df


def test_engineer_has_fixed_columns_even_with_audio_only_input(train_df):
    stats = fit_stats(train_df)
    out = engineer(train_df[[*AUDIO, GENRE]], stats)  # sem extras e sem artista
    assert list(out.columns) == [*MODEL_COLUMNS, GENRE]
    assert out[ARTIST].isna().all().all() and out["liveness"].isna().all()
    assert out["danceability_z"].notna().all()


def test_z_scores_are_relative_to_the_genre(train_df):
    stats = fit_stats(train_df)
    out = engineer(train_df, stats)
    # dentro de cada gênero a média do z é ~0: a mesma feature "alta" em dois gêneros vira o mesmo sinal
    assert abs(out.groupby(GENRE)["energy_z"].mean()).max() < 1e-6


def test_unknown_genre_degrades_to_nan_not_error(train_df):
    stats = fit_stats(train_df)
    row = train_df.head(1).assign(**{GENRE: "inedito"})
    out = engineer(row, stats)
    assert np.isnan(out["energy_z"].iloc[0])
    assert expected_popularity(np.array([50.0]), np.array(["inedito"]), stats)[0] == pytest.approx(
        np.percentile(train_df["popularity"], 50)
    )  # cai nos quantis globais


def test_artist_context_can_be_switched_off(train_df):
    stats = fit_stats(train_df)
    df = train_df.assign(artist_n_tracks=5, artist_mean_popularity=70.0, artist_hit_share=0.5)
    on, off = engineer(df, stats), engineer(df, stats, use_artist_context=False)
    assert on["artist_mean_popularity"].eq(70).all() and on["artist_pct_in_genre"].notna().all()
    assert off[ARTIST].isna().all().all() and off["artist_pct_in_genre"].isna().all()


def test_expected_popularity_restores_genre_information(train_df):
    stats = fit_stats(train_df)
    pop, metal = (expected_popularity(np.array([80.0]), np.array([g]), stats)[0] for g in ("pop", "metal"))
    assert pop > 60 > metal  # o mesmo percentil 80 vale popularidade muito diferente em cada gênero


def test_pct_in_genre_is_the_inverse_of_expected_popularity(train_df):
    stats = fit_stats(train_df)
    scores = np.array([10.0, 50.0, 90.0])
    genres = np.array(["pop"] * 3)
    back = pct_in_genre(expected_popularity(scores, genres, stats), genres, stats)
    assert back == pytest.approx(scores, abs=3)


def test_explicit_bool_and_none_are_handled():
    stats = fit_stats(pd.DataFrame({**{f: [0.1, 0.9] for f in AUDIO}, GENRE: ["a", "a"], "popularity": [10, 80]}))
    df = pd.DataFrame({**{f: [0.5, 0.5, 0.5] for f in AUDIO}, GENRE: ["a"] * 3, "explicit": [True, False, None]})
    out = engineer(df, stats)
    assert out["explicit"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["explicit"].iloc[2])


def test_factor_groups_merge_raw_and_z_and_onehots():
    names = ["num__danceability", "num__danceability_z", "num__artist_n_tracks", "num__artist_pct_in_genre",
             "genre__track_genre_pop", "genre__track_genre_rock"]  # fmt: skip
    groups = factor_groups(names)
    assert groups == {"danceability": [0, 1], "artist_context": [2, 3], GENRE: [4, 5]}
    sv = np.array([[1.0, 2.0, 0.5, 0.5, -1.0, 3.0]])
    keys, mat = aggregate_shap(sv, groups)
    assert keys == ["danceability", "artist_context", GENRE] and mat.tolist() == [[3.0, 1.0, 2.0]]


def test_tables_live_in_stats_not_in_a_global_cache(train_df):
    a, b = fit_stats(train_df), fit_stats(train_df.assign(energy=train_df["energy"] + 0.4))
    za = engineer(train_df, a)["energy_z"]
    zb = engineer(train_df, b)["energy_z"]
    assert not np.allclose(za, zb)  # cada stats usa as próprias tabelas, mesmo se o id() do outro for reaproveitado
    assert "_tables" in a and a["_tables"] is not b["_tables"]
