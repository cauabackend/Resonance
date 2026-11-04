from resonance_pipeline import analyze
from resonance_pipeline.common import FEATURES


def test_feature_tests_detect_signal(clean_df):
    rows = {r["feature"]: r for r in analyze.feature_tests(clean_df, 0.05)}
    assert rows["danceability"]["significant_bonferroni"]
    assert rows["danceability"]["p_bonferroni"] >= rows["danceability"]["p"]
    assert not rows["valence"]["significant_bonferroni"]  # ruído puro


def test_multicollinearity(clean_df):
    out = analyze.multicollinearity(clean_df)
    assert set(out["vif"]) == set(FEATURES) and all(v >= 1 for v in out["vif"].values())


def test_sensitivity_prevalence_drops_with_threshold(clean_df):
    rows = analyze.threshold_sensitivity(clean_df, [40, 60, 80])
    rates = [r["hit_rate"] for r in rows]
    assert rates == sorted(rates, reverse=True)


def test_main_writes_report(tmp_path, monkeypatch, clean_df):
    csv = tmp_path / "t.csv"
    clean_df.to_csv(csv, index=False)
    monkeypatch.setattr(analyze, "ROOT", tmp_path)
    monkeypatch.setattr(
        analyze,
        "load_params",
        lambda: {
            "data": {"processed": str(csv)},
            "analysis": {"alpha": 0.05, "sensitivity_thresholds": [50, 60], "genre_cutoffs": [10, 1000]},
        },
    )
    analyze.main()
    assert (tmp_path / "reports" / "analysis.json").exists()


def test_genre_cutoff_sensitivity(clean_df):
    rows = analyze.genre_cutoff_sensitivity(clean_df, [1, 100, 10_000])
    assert rows[0]["genres_grouped"] == 0 and rows[0]["share_tracks_grouped"] == 0
    assert rows[1]["genres_grouped"] >= 1  # o gênero "niche" (10 faixas) cai fora com corte 100
    assert rows[2]["share_tracks_grouped"] == 1 and rows[2]["genres_kept"] == 0
