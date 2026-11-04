import joblib
import numpy as np
import pandas as pd

from resonance_pipeline import drift
from resonance_pipeline.common import FEATURES, GENRE


def test_synthetic_shift_triggers_alert(clean_df, tmp_path):
    cur = drift.synthetic_production(clean_df, n=300, shift=True)
    res = drift.drift_report(clean_df, cur, tmp_path / "d.html", share_threshold=0.2)
    assert res["alert"] and (tmp_path / "d.html").exists()


def test_no_shift_no_alert(clean_df, tmp_path):
    cur = drift.synthetic_production(clean_df, n=300, shift=False)
    assert not drift.drift_report(clean_df, cur, tmp_path / "d.html", share_threshold=0.3)["alert"]


def test_load_prediction_logs(tmp_path):
    d = tmp_path / "dt=2026-10-29"
    d.mkdir()
    (d / "a.jsonl").write_text('{"features": {"danceability": 0.5}}\n', encoding="utf-8")
    df = drift.load_prediction_logs(tmp_path, 7)
    assert list(df.columns) == [*FEATURES, GENRE, "percentile_score", "mode"] and len(df) == 1


def test_main_cli(tmp_path, monkeypatch, clean_df):
    csv = tmp_path / "t.csv"
    clean_df.to_csv(csv, index=False)
    monkeypatch.setattr(drift, "load_params", lambda: {"data": {"processed": str(csv)}})
    no_bundle = str(tmp_path / "sem-modelo.joblib")  # nunca ler o bundle real de model/production
    assert drift.main(["--logs", str(tmp_path / "none"), "--bundle", no_bundle]) == 0
    assert drift.main(["--synthetic", "--out", str(tmp_path / "o.html"), "--bundle", no_bundle]) == 1


def test_prediction_drift_detects_shifted_scores(tmp_path):
    ref = pd.Series(np.random.default_rng(0).uniform(30, 70, 500))
    same = drift.score_drift_report(ref, ref.sample(300, random_state=1), tmp_path / "a.html")
    shifted = drift.score_drift_report(ref, ref.sample(300, random_state=1) + 25, tmp_path / "b.html")
    assert not same["alert"] and shifted["alert"] and shifted["mean_current"] > same["mean_current"] + 20


def test_main_reports_prediction_drift_with_bundle(tmp_path, monkeypatch, clean_df, params):
    from resonance_pipeline import train

    csv = tmp_path / "t.csv"
    clean_df.to_csv(csv, index=False)
    bundle = train.fit_model(clean_df, params, {"max_depth": 2, "n_estimators": 10})
    joblib.dump(bundle, tmp_path / "bundle.joblib")
    monkeypatch.setattr(drift, "load_params", lambda: {"data": {"processed": str(csv)}})
    rc = drift.main(["--synthetic", "--bundle", str(tmp_path / "bundle.joblib"), "--out", str(tmp_path / "f.html"),
                     "--out-scores", str(tmp_path / "s.html")])  # fmt: skip
    assert rc == 1 and (tmp_path / "s.html").exists()
