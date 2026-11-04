import pandas as pd

from resonance_pipeline.preprocess import clean


def test_dedup_by_id_and_filters(raw_df):
    dup = raw_df.iloc[[0]].copy()  # mesmo track_id
    remix = raw_df.iloc[[1]].copy()  # mesmo name+artist, id e duração distintos
    remix[["track_id", "duration_ms"]] = ["remix", 200_001]
    bad = raw_df.iloc[[2, 3]].copy()
    bad["track_id"] = ["b1", "b2"]
    bad.iloc[0, bad.columns.get_loc("tempo")] = 0  # falha de BPM
    bad.iloc[1, bad.columns.get_loc("duration_ms")] = 2_000_000  # > 15 min
    out = clean(pd.concat([raw_df, dup, remix, bad]), 900_000, 30)
    assert out["track_id"].is_unique
    assert "remix" in set(out["track_id"])  # remix não colide como duplicata
    assert not {"b1", "b2"} & set(out["track_id"])


def test_dedup_without_id_uses_name_artist_duration(raw_df):
    df = raw_df.drop(columns=["track_id"])
    out = clean(pd.concat([df, df.iloc[[0]]]), 900_000, 30)
    assert len(out) == len(df)


def test_null_name_dropped(raw_df):
    raw_df.loc[0, "name"] = None
    assert len(clean(raw_df, 900_000, 30)) == len(raw_df) - 1


def test_collaborations_are_grouped_under_the_first_artist(raw_df):
    raw_df.loc[0, "artist"] = "Ingrid Michaelson;ZAYN"
    raw_df.loc[1, "artist"] = "Ingrid Michaelson"
    out = clean(raw_df, 900_000, 30)
    # a colaboração e a faixa solo caem no mesmo grupo: nunca ficam em lados opostos do split
    assert out.loc[out["track_id"].isin(["id0", "id1"]), "artist"].tolist() == ["Ingrid Michaelson"] * 2
    assert not out["artist"].str.contains(";").any()
