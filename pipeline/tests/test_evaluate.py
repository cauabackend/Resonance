import json

import joblib
import pytest

from resonance_pipeline import evaluate, train

GATE_P = {"evaluate": {"gate_tolerance": 0.005, "brier_tolerance": 0.002}}


def test_nested_cv_reports_both_modes_and_niche_segments(clean_df, params):
    cv = evaluate.nested_cv(clean_df, params)
    assert len(cv["chosen_hyperparams"]) == params["evaluate"]["cv_folds"]
    for mode in ("artist_aware", "audio_only"):
        assert cv[mode]["spearman_mean"] > 0.3 and 0 < cv[mode]["ndcg_at_k_mean"] <= 1
        niche = cv[mode]["niche_segments"]["niche"]
        assert niche["n"] > 0 and "spearman" in niche  # percentil existe mesmo sem nenhum hit
        assert cv[mode]["niche_segments"]["ghost"] == {"n": 0, "hits": 0}
    stab = cv["shap_stability"]
    assert stab["mean_pairwise_spearman"] > 0.3 and set(stab["mean_abs_shap"]) >= {"danceability", "track_genre"}
    assert "artist_context" in stab["mean_abs_shap"]  # as 3 colunas de histórico + o percentil viram UM fator


def _m(ndcg, spearman, brier=0.07):
    return {
        "ndcg_at_k": ndcg,
        "spearman": spearman,
        "brier": brier,
        "audio_only_ndcg_at_k": ndcg,
        "audio_only_spearman": spearman,
        "audio_only_brier": brier,
    }


def _baselines(ndcg=0.80, spearman=0.50):
    b = {"ndcg_at_k": ndcg, "spearman": spearman}
    return {"artist_mean": b, "ridge": b}


def test_gate():
    base = _baselines()
    assert evaluate.gate(_m(0.85, 0.55), base, None, GATE_P)[0]
    assert not evaluate.gate(_m(0.79, 0.55), base, None, GATE_P)[0]  # ndcg abaixo do baseline
    assert not evaluate.gate(_m(0.85, 0.40), base, None, GATE_P)[0]  # spearman abaixo do baseline
    prod = _m(0.86, 0.56, 0.07)
    assert not evaluate.gate(_m(0.85, 0.56), base, prod, GATE_P)[0]  # ndcg piorou além da tolerância
    assert not evaluate.gate(_m(0.86, 0.56, 0.08), base, prod, GATE_P)[0]  # brier piorou
    assert evaluate.gate(_m(0.857, 0.557, 0.0705), base, prod, GATE_P)[0]  # dentro das tolerâncias


def test_gate_fails_when_only_one_mode_regresses():
    base, prod = _baselines(), _m(0.86, 0.56)
    new = {**prod, "audio_only_ndcg_at_k": 0.70}  # a modalidade só-áudio piorou; a outra não
    ok, reason = evaluate.gate(new, base, prod, GATE_P)
    assert not ok and "audio_only_ndcg_at_k" in reason


def _setup(tmp_path, monkeypatch, clean_df, params, degrade_candidate=False):
    csv = tmp_path / "tracks.csv"
    clean_df.to_csv(csv, index=False)
    p = {**params, "data": {**params["data"], "processed": str(csv)}}
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "load_params", lambda: p)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    train.train_and_save()
    cand, prod = tmp_path / "model" / "candidate", tmp_path / "model" / "production"
    monkeypatch.setattr(evaluate, "CANDIDATE", cand)
    monkeypatch.setattr(evaluate, "PRODUCTION", prod)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "load_params", lambda: p)
    b = joblib.load(cand / "bundle.joblib")
    for base in ("ridge", "artist_mean"):  # dado sintético é fácil: os baselines empatariam o XGBoost
        b["baseline"][base].update(ndcg_at_k=0.0, spearman=0.0)
    joblib.dump(b, cand / "bundle.joblib")
    if degrade_candidate:  # produção = modelo genuíno; o candidato vira um modelo pior
        prod.mkdir(parents=True)
        joblib.dump(b, prod / "bundle.joblib")
        b["metrics"].update(ndcg_at_k=0.5, spearman=0.0)
        joblib.dump(b, cand / "bundle.joblib")
    return prod


def test_main_promotes(tmp_path, monkeypatch, clean_df, params):
    prod = _setup(tmp_path, monkeypatch, clean_df, params)
    evaluate.main()
    assert (prod / "bundle.joblib").exists()
    report = json.loads((tmp_path / "reports" / "evaluation.json").read_text())
    assert report["promoted"] and "nested_cv" in report
    flat = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert "holdout_ndcg_at_k" in flat and "holdout_audio_only_ndcg_at_k" in flat


def test_main_blocks_regression_and_keeps_production(tmp_path, monkeypatch, clean_df, params):
    prod = _setup(tmp_path, monkeypatch, clean_df, params, degrade_candidate=True)
    before = joblib.load(prod / "bundle.joblib")["version"]
    with pytest.raises(SystemExit):
        evaluate.main()
    assert joblib.load(prod / "bundle.joblib")["version"] == before


def test_importance_stability_detects_instability():
    stable = [{"a": 3.0, "b": 2.0, "c": 1.0}] * 3
    flipped = [{"a": 3.0, "b": 2.0, "c": 1.0}, {"a": 1.0, "b": 2.0, "c": 3.0}]
    assert evaluate.importance_stability(stable)["mean_pairwise_spearman"] == 1.0
    assert evaluate.importance_stability(flipped)["min_pairwise_spearman"] == -1.0
