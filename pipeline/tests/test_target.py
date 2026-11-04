import pandas as pd
import pandera.pandas as pa
import pytest

from resonance_pipeline.schema import PCT
from resonance_pipeline.target import add_targets


def _frame(rows):
    base = {
        "name": "n", "artist": "a", "danceability": 0.5, "energy": 0.5, "valence": 0.5, "speechiness": 0.1,
        "acousticness": 0.1, "instrumentalness": 0.0, "loudness": -5.0, "tempo": 100.0, "duration_ms": 200000,
        "liveness": 0.1, "key": 5, "mode": 1, "time_signature": 4, "explicit": False,
    }  # fmt: skip
    return pd.DataFrame([{**base, "track_genre": g, "popularity": pop} for g, pop in rows])


def test_percentile_is_computed_per_genre():
    df = add_targets(_frame([("pop", 10), ("pop", 50), ("pop", 90), ("pop", 95), ("metal", 1), ("metal", 3)]), 60)
    pop, metal = df[df.track_genre == "pop"], df[df.track_genre == "metal"]
    assert list(pop[PCT]) == [25, 50, 75, 100]
    assert list(metal[PCT]) == [50, 100]  # 3 de popularidade é topo no metal, sem ser hit global


def test_ties_share_the_average_rank():
    df = add_targets(_frame([("pop", 10), ("pop", 10), ("pop", 20)]), 60)
    assert df[PCT].iloc[0] == df[PCT].iloc[1] == pytest.approx(100 * 1.5 / 3)


def test_is_hit_is_only_the_historical_label():
    df = add_targets(_frame([("pop", 59), ("pop", 60), ("metal", 5)]), 60)
    assert list(df["is_hit"]) == [0, 1, 0]


def test_schema_rejects_out_of_range():
    with pytest.raises(pa.errors.SchemaErrors):
        add_targets(_frame([("pop", 101)]), 60)
