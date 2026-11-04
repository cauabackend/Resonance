import numpy as np
import pandas as pd
import pytest

from resonance_pipeline import train
from resonance_pipeline.schema import PCT


@pytest.fixture
def bundle(clean_df, params):
    tr, _ = train.split(clean_df, params)
    return train.fit_model(tr, params, {"max_depth": 3, "n_estimators": 30})


def _patch(monkeypatch, tmp_path, clean_df, params):
    csv = tmp_path / "tracks.csv"
    clean_df.to_csv(csv, index=False)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "load_params", lambda: {**params, "data": {**params["data"], "processed": str(csv)}})
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")


def test_split_never_shares_an_artist(clean_df, params):
    tr, te = train.split(clean_df, params)
    assert not set(tr["artist"]) & set(te["artist"])


def test_model_ranks_better_than_chance(clean_df, params, bundle):
    _, te = train.split(clean_df, params)
    score = train.predict_score(bundle, te)
    assert ((score >= 0) & (score <= 100)).all()
    assert train.ranking_metrics(te[PCT], score, 20, 50, 5, 0)["spearman"] > 0.3


def test_unseen_genre_does_not_break_inference(clean_df, bundle):
    row = clean_df.head(1).assign(track_genre="genero-inedito")
    assert train.predict_score(bundle, row).shape == (1,)
    assert 0 <= train.predict_proba(bundle, row)[0] <= 1


def test_rare_genre_grouped(bundle):
    assert "niche" not in bundle["known_genres"] and "pop" in bundle["known_genres"]


def test_probability_is_monotonic_in_expected_popularity(bundle):
    pops = np.linspace(0, 100, 50)
    for mode in train.MODES:
        assert (np.diff(bundle["calibrators"][mode].predict(pops)) >= -1e-6).all()  # float32


def test_artist_history_improves_ranking_and_audio_only_still_works(clean_df, params, bundle):
    _, te = train.split(clean_df, params)
    aware = train.predict_score(bundle, te, use_artist_context=True)
    audio = train.predict_score(bundle, te, use_artist_context=False)
    r_aware = train.ranking_metrics(te[PCT], aware, 20, 50, 5, 0)["spearman"]
    r_audio = train.ranking_metrics(te[PCT], audio, 20, 50, 5, 0)["spearman"]
    assert r_audio > 0.3  # a modalidade só-áudio funciona (foi treinada com histórico mascarado)
    assert not np.allclose(aware, audio) and r_aware >= r_audio - 0.05


def test_mask_artist_blanks_history_in_a_fraction_of_rows(clean_df, params):
    from resonance_features import ARTIST, engineer, fit_stats

    e = train.mask_artist(engineer(clean_df, fit_stats(clean_df)), 0.5, 0)
    blank = e[ARTIST[1]].isna().mean()
    original = engineer(clean_df, fit_stats(clean_df))[ARTIST[1]].isna().mean()
    assert 0.35 < blank - original + 0.0 < 0.65 or blank > 0.4  # ~metade das linhas perde o histórico
    assert e["artist_pct_in_genre"].isna().mean() >= blank - 1e-9  # o derivado some junto, sem vazar o histórico


def test_ranking_metrics_perfect_vs_inverted():
    y = np.arange(100, dtype=float)
    perfect = train.ranking_metrics(y, y, 10, 50, 5, 0)
    inverted = train.ranking_metrics(y, -y, 10, 50, 5, 0)
    assert perfect["spearman"] == pytest.approx(1) and perfect["ndcg_at_k"] == pytest.approx(1)
    assert inverted["spearman"] == pytest.approx(-1) and inverted["ndcg_at_k"] < 0.5


def test_best_threshold_respects_cost():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.3, 0.35, 0.1, 0.1, 0.1])
    assert train.best_threshold(y, p, fn_cost=10, fp_cost=1) <= 0.3  # FN caro: aceita o FP de 0.35
    assert train.best_threshold(y, p, fn_cost=1, fp_cost=10) > 0.35


def test_calibration_perfect_and_bad():
    y = np.array([0, 1] * 50)
    assert train.calibration(y, np.full(100, 0.5))["ece"] < 1e-9
    assert train.calibration(y, np.where(y == 1, 0.05, 0.95))["ece"] > 0.5


def test_baselines_and_tuning(clean_df, params):
    tr, te = train.split(clean_df, params)
    b = train.baselines(tr, te, params)
    assert b["ridge"]["spearman"] > 0.3 and "artist_mean" in b
    assert b["genre_mean"]["spearman"] < b["ridge"]["spearman"]  # gênero sozinho não ordena dentro do gênero
    hp, grid = train.tune(tr, params)
    assert hp in [g["params"] for g in grid] and len(grid) == 2


def test_lineage_has_keys():
    assert set(train.lineage()) == {"dataset_md5", "git_commit"}


def test_train_and_save_end_to_end(clean_df, params, tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, clean_df, params)
    b = train.train_and_save()
    assert (tmp_path / "model" / "candidate" / "bundle.joblib").exists()
    m = b["metrics"]
    assert m["spearman"] > 0.3 and 0 <= m["ndcg_at_k"] <= 1 and m["brier"] <= m["brier_base_rate"] + 0.05
    assert set(b["baseline"]) == {"genre_mean", "ridge", "artist_mean"}
    assert m["audio_only_spearman"] > 0.3 and "audio_only_hit_roc_auc" in m
    assert pd.Series(b["hit_means"]).notna().all()


def test_has_history_only_where_the_artist_has_other_tracks(clean_df):
    h = train.has_history(clean_df)
    assert h.sum() < len(clean_df) or h.all()
    assert (clean_df.loc[h, "artist_n_tracks"] > 0).all()
    assert clean_df.loc[~h, "artist_mean_popularity"].isna().all()


def test_aware_calibrator_and_metrics_use_only_rows_with_history(clean_df, params):
    one_track = clean_df.head(3).assign(
        artist=["solo1", "solo2", "solo3"],
        artist_n_tracks=0.0,
        artist_mean_popularity=float("nan"),
        artist_hit_share=float("nan"),
    )
    df = pd.concat([clean_df.iloc[3:], one_track])
    tr, te = train.split(df, params)
    bundle = train.fit_model(tr, params, {"max_depth": 2, "n_estimators": 20})
    m = train.holdout_metrics(bundle, te, params)
    assert m["spearman"] == m["spearman"]  # existe (não NaN) mesmo com artistas sem histórico no holdout
