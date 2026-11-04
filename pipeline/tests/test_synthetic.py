import pandas as pd
import pytest

from resonance_pipeline import leaderboard, synthetic
from resonance_pipeline.preprocess import clean
from resonance_pipeline.target import add_targets
from resonance_pipeline.validate import validate_raw


def test_synthetic_dataset_is_valid_and_reproducible():
    a, b = synthetic.make_dataset(800, seed=1), synthetic.make_dataset(800, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(validate_raw(a)) == 800


def test_synthetic_runs_through_the_pipeline_stages():
    df = add_targets(clean(synthetic.make_dataset(1500), 900_000, 30), 60)
    assert df["popularity_percentile_within_genre"].between(0, 100).all() and df["is_hit"].isin([0, 1]).all()


def test_synthetic_cli_writes_csv(tmp_path, monkeypatch):
    out = tmp_path / "raw" / "d.csv"
    monkeypatch.setattr("sys.argv", ["synthetic", "--n", "200", "--out", str(out)])
    synthetic.main()
    assert len(pd.read_csv(out)) == 200


@pytest.fixture
def lb_params(params):
    return {**params, "evaluate": {**params["evaluate"], "ndcg_k": 20, "batch_size": 100, "n_batches": 5}}


def test_leaderboard_ranks_every_approach(lb_params):
    df = add_targets(clean(synthetic.make_dataset(1500), 900_000, 30), 60)
    board = leaderboard.run(df, lb_params, {"max_depth": 3, "n_estimators": 40})
    assert set(board) == {"ridge", "random_forest", "hist_gradient_boosting", "xgb_regressor", "xgb_ranker_lambdamart"}
    # no sintético o histórico do artista domina; só áudio fica fraco, mas acima do acaso
    for m in board.values():
        assert m["artist_aware"]["spearman"] > 0.5 and 0 < m["artist_aware"]["ndcg_at_k"] <= 1
        assert m["audio_only"]["spearman"] > 0.0 and 0 < m["audio_only"]["ndcg_at_k"] <= 1
